from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import gsd.hoomd
import numpy as np

from hexatic.constants import cylinder

from .analyze_case import _analyze_shell_fields, _mark_dislocations
from .backend import ArrayBackend, select_backend
from .spatial import PeriodicXTree
from .storage import (
    FrameShardWriter,
    prepare_analysis_dir,
    save_safetensors_atomic,
    write_json_atomic,
)

# Edit this tuple to choose one or more analyses. Do not add CLI flags for them.
SELECTED_ANALYSES = (
    "hexatic",
)

# Other editable run settings.
BACKEND = "auto"
REQUIRE_GPU = False
PARTICLE_BLOCK_SIZE = 2048
TARGET_SHARD_MIB = 256
OVERWRITE = False

AVAILABLE_ANALYSES = frozenset(
    {
        "coordinates",
        "active_direction",
        "hexatic",
        "defects",
        "shell_chirality",
    }
)


@dataclass(frozen=True)
class InferredGeometry:
    radius: float
    circumference: float
    lx: float


def _validate_selected_analyses() -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(SELECTED_ANALYSES))
    if not selected:
        raise ValueError("SELECTED_ANALYSES must contain at least one analysis")
    unknown = sorted(set(selected) - AVAILABLE_ANALYSES)
    if unknown:
        known = ", ".join(sorted(AVAILABLE_ANALYSES))
        raise ValueError(
            f"Unknown selective analyses: {', '.join(unknown)}. "
            f"Available analyses: {known}"
        )
    return selected


def _infer_geometry(frame) -> InferredGeometry:
    box = np.asarray(frame.configuration.box, dtype=np.float64)
    if box.shape[0] < 3 or box[0] <= 0.0:
        raise ValueError("Input GSD has an invalid simulation box")
    positions = np.asarray(frame.particles.position, dtype=np.float64)
    if not len(positions):
        raise ValueError("Input GSD contains no particles")
    radial = np.linalg.norm(positions[:, 1:3], axis=1)
    radius = float(np.quantile(radial, 0.99))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Could not infer a positive cylinder radius from the GSD")
    return InferredGeometry(
        radius=radius,
        circumference=2.0 * math.pi * radius,
        lx=float(box[0]),
    )


def _analyze_frame(
    frame,
    geometry: InferredGeometry,
    backend: ArrayBackend,
    selected: tuple[str, ...],
    frame_index: int,
) -> dict[str, np.ndarray]:
    positions = np.asarray(frame.particles.position, dtype=np.float32)
    result = {
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "step": np.asarray(int(frame.configuration.step), dtype=np.int64),
    }

    need_coords = bool(
        {"coordinates", "active_direction"} & set(selected)
    )
    coords = backend.coordinates(positions).astype(np.float32) if need_coords else None
    if "coordinates" in selected:
        assert coords is not None
        result["coords"] = coords

    if "active_direction" in selected:
        orientation = np.asarray(frame.particles.orientation, dtype=np.float32)
        directions = backend.directions(orientation).astype(np.float32)
        assert coords is not None
        result["active_direction"] = directions
        result["direction_cylindrical"] = backend.cylindrical(
            directions,
            coords[:, 1],
        ).astype(np.float32)

    need_shell = bool(
        {"hexatic", "defects", "shell_chirality"} & set(selected)
    )
    if need_shell:
        (
            psi_real,
            psi_imag,
            neighbor_counts,
            shell_mask,
            shell_chirality,
            shell_bond_count,
        ) = _analyze_shell_fields(
            positions,
            geometry,
            backend,
            PARTICLE_BLOCK_SIZE,
        )
        if "hexatic" in selected:
            result["psi_real"] = psi_real
            result["psi_imag"] = psi_imag
            result["neighbor_counts"] = neighbor_counts
            result["hexatic_shell_mask"] = shell_mask.astype(np.bool_)
        if "defects" in selected:
            charges = np.zeros(len(positions), dtype=np.int8)
            charges[shell_mask] = (
                cylinder.ANALYSIS.neighbors - neighbor_counts[shell_mask]
            ).astype(np.int8)
            result["disclination_charges"] = charges
            result["dislocation_flags"] = _mark_dislocations(
                PeriodicXTree.build(positions, geometry.lx),
                positions,
                charges,
                cylinder.ANALYSIS.dislocation_pair_distance,
            )
        if "shell_chirality" in selected:
            result["shell_bond_translation_chirality_mean"] = np.asarray(
                shell_chirality,
                dtype=np.float32,
            )
            result["shell_bond_count"] = np.asarray(
                shell_bond_count,
                dtype=np.int64,
            )
    return result


def run(input_gsd: Path, output_dir: Path) -> None:
    selected = _validate_selected_analyses()
    if PARTICLE_BLOCK_SIZE < 1:
        raise ValueError("PARTICLE_BLOCK_SIZE must be positive")
    if TARGET_SHARD_MIB < 1:
        raise ValueError("TARGET_SHARD_MIB must be positive")
    if not input_gsd.is_file():
        raise FileNotFoundError(f"Missing input GSD: {input_gsd}")

    should_run, _ = prepare_analysis_dir(
        output_dir,
        overwrite=OVERWRITE,
        resume=False,
    )
    if not should_run:
        return

    backend = select_backend(BACKEND, require_gpu=REQUIRE_GPU)
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        if not len(source):
            raise ValueError(f"Trajectory contains no frames: {input_gsd}")
        geometry = _infer_geometry(source[0])
        particle_count = int(source[0].particles.N)
        for frame_index in range(1, len(source)):
            if int(source[frame_index].particles.N) != particle_count:
                raise ValueError(
                    "Selective analysis requires a constant particle count"
                )

        manifest: dict[str, object] = {
            "schema": "hexatic.big_lx.selective_analysis.v1",
            "trajectory_gsd": str(input_gsd),
            "selected_analyses": list(selected),
            "backend": backend.name,
            "device": backend.device_description,
            "particle_count": particle_count,
            "particle_block_size": PARTICLE_BLOCK_SIZE,
            "target_shard_mib": TARGET_SHARD_MIB,
            "geometry_source": "GSD box and frame-0 radial 99th percentile",
            "radius": geometry.radius,
            "circumference": geometry.circumference,
            "lx": geometry.lx,
            "dtype": "float32",
            "complete": False,
            "shards": [],
        }
        write_json_atomic(output_dir / "manifest.json", manifest)
        save_safetensors_atomic(
            output_dir / "static.safetensors",
            {
                "box": np.asarray(
                    source[0].configuration.box,
                    dtype=np.float32,
                ),
                "radius": np.asarray(geometry.radius, dtype=np.float32),
                "circumference": np.asarray(
                    geometry.circumference,
                    dtype=np.float32,
                ),
                "lx": np.asarray(geometry.lx, dtype=np.float32),
            },
            backend_name=backend.name,
            metadata={"schema": "hexatic.big_lx.selective_static.v1"},
        )
        writer = FrameShardWriter(
            output_dir,
            manifest,
            backend_name=backend.name,
            target_bytes=TARGET_SHARD_MIB * 1024 * 1024,
        )
        for frame_index, frame in enumerate(source):
            print(
                "[big_lx.selective_analysis] "
                f"frame={frame_index + 1}/{len(source)} "
                f"analyses={','.join(selected)} backend={backend.name}",
                flush=True,
            )
            writer.add(
                _analyze_frame(
                    frame,
                    geometry,
                    backend,
                    selected,
                    frame_index,
                )
            )
        writer.flush()
        manifest["frame_count"] = len(source)
    manifest["complete"] = True
    write_json_atomic(output_dir / "manifest.json", manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the analyses declared in SELECTED_ANALYSES on one GSD."
        )
    )
    parser.add_argument("--input-gsd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(args.input_gsd, args.output_dir)


if __name__ == "__main__":
    main()
