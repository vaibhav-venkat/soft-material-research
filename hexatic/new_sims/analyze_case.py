from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gsd.hoomd
import numpy as np
from scipy.spatial import KDTree

from hexatic.big_lx.backend import ArrayBackend, select_backend
from hexatic.constants import cylinder
from hexatic.confinement_comparison.storage import (
    prepare_analysis_dir,
    save_safetensors_atomic,
    write_json_atomic,
)

from .cases import (
    FRAMES_PER_SHARD,
    CaseKind,
    CasePaths,
    DEFAULT_OUTPUT_ROOT,
    NewSimCase,
    get_case,
)
from .storage import FrameShardWriter


class PeriodicTree:
    def __init__(self, positions: np.ndarray, box_lengths: tuple[float, float, float]):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.box_lengths = np.asarray(box_lengths, dtype=np.float64)
        transformed = self.positions.copy()
        for axis, length in enumerate(self.box_lengths):
            if length > 0:
                transformed[:, axis] = np.mod(
                    transformed[:, axis] + 0.5 * length, length
                )
        self.tree = KDTree(transformed, boxsize=self.box_lengths)

    def nearest_bonds(self, n_neighbors: int = 6) -> np.ndarray:
        transformed = self.positions.copy()
        for axis, length in enumerate(self.box_lengths):
            if length > 0:
                transformed[:, axis] = np.mod(
                    transformed[:, axis] + 0.5 * length, length
                )
        distances, hits = self.tree.query(
            transformed, k=n_neighbors + 1, workers=-1
        )
        if not np.allclose(distances[:, 0], 0.0, atol=1e-9):
            raise ValueError("nearest-neighbor query did not return self first")
        source = np.asarray(hits[:, 1:], dtype=np.int64)
        bonds = self.positions[source] - self.positions[:, None, :]
        for axis, length in enumerate(self.box_lengths):
            if length > 0:
                bonds[..., axis] -= length * np.rint(bonds[..., axis] / length)
        return bonds.astype(np.float32)


def _planar_hexatic(
    backend: ArrayBackend,
    bonds: np.ndarray,
    first_axis: int,
    second_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    xp = backend.xp
    values = xp.asarray(bonds, dtype=xp.float32)
    angles = xp.arctan2(values[:, :, second_axis], values[:, :, first_axis])
    real = xp.mean(xp.cos(6.0 * angles), axis=1)
    imaginary = xp.mean(xp.sin(6.0 * angles), axis=1)
    return (
        backend.to_numpy(real).astype(np.float32),
        backend.to_numpy(imaginary).astype(np.float32),
    )


def _hexatic_fields(
    positions: np.ndarray,
    case: NewSimCase,
    backend: ArrayBackend,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = len(positions)
    real = np.zeros(n_particles, dtype=np.float32)
    imaginary = np.zeros(n_particles, dtype=np.float32)
    valid = np.zeros(n_particles, dtype=np.bool_)
    if case.kind == CaseKind.PERIODIC_3D_BULK:
        return real, imaginary, valid

    if case.is_2d:
        groups = (np.arange(n_particles, dtype=np.int64),)
        lengths = (case.lx, case.circumference, 0.0)
        plane_axes = (0, 1)
        cylinder_surface = False
    elif case.kind == CaseKind.X_WALLED_PRISM:
        shell = 0.5 * case.lx - np.abs(positions[:, 0])
        surface = shell <= cylinder.ANALYSIS.shell_delta
        groups = (
            np.flatnonzero(surface & (positions[:, 0] >= 0.0)),
            np.flatnonzero(surface & (positions[:, 0] < 0.0)),
        )
        lengths = (0.0, case.prism_side, case.prism_side)
        plane_axes = (1, 2)
        cylinder_surface = False
    else:
        radial = np.linalg.norm(positions[:, 1:3], axis=1)
        surface = radial >= case.radius - cylinder.ANALYSIS.shell_delta
        groups = (np.flatnonzero(surface),)
        lengths = (case.lx, 0.0, 0.0)
        plane_axes = (0, 1)
        cylinder_surface = True

    for indices in groups:
        if len(indices) <= cylinder.ANALYSIS.neighbors:
            continue
        selected = positions[indices]
        bonds = PeriodicTree(selected, lengths).nearest_bonds(
            cylinder.ANALYSIS.neighbors
        )
        if cylinder_surface:
            group_real, group_imaginary = backend.hexatic(bonds, selected)
        else:
            group_real, group_imaginary = _planar_hexatic(
                backend, bonds, plane_axes[0], plane_axes[1]
            )
        real[indices] = np.asarray(group_real, dtype=np.float32)
        imaginary[indices] = np.asarray(group_imaginary, dtype=np.float32)
        valid[indices] = True
    return real, imaginary, valid


def _cylinder_coords(positions: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    return backend.coordinates(positions).astype(np.float32)


def analyze_frame(
    frame: gsd.hoomd.Frame,
    case: NewSimCase,
    backend: ArrayBackend,
    *,
    frame_index: int,
    active_path: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if int(frame.particles.N) != case.n_particles:
        raise ValueError(
            f"case {case.case_id} expects {case.n_particles} particles, "
            f"got {frame.particles.N}"
        )
    positions = np.asarray(frame.particles.position, dtype=np.float32)
    orientation = np.asarray(frame.particles.orientation, dtype=np.float32)
    directions = backend.directions(orientation).astype(np.float32)
    if case.is_cylinder:
        coords = _cylinder_coords(positions, backend)
    elif case.is_2d:
        coords = positions[:, :2]
    else:
        coords = positions
    step = frame.configuration.step
    if step is None:
        raise ValueError("trajectory frame has no simulation step")
    result: dict[str, np.ndarray] = {
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "step": np.asarray(int(step), dtype=np.int64),
        "coords": np.asarray(coords, dtype=np.float32),
    }
    if not case.is_2d:
        result["polarization"] = np.asarray(directions, dtype=np.float32)
    if case.has_hexatic:
        real, imaginary, valid = _hexatic_fields(positions, case, backend)
        result.update(
            psi_real=real,
            psi_imag=imaginary,
            hexatic_valid=valid,
        )
    if active_path is not None:
        result["active_coords_unwrapped"] = np.asarray(
            active_path[frame_index], dtype=np.float32
        )
    return result


def _particle_images(frame: gsd.hoomd.Frame, n_particles: int) -> np.ndarray:
    image = frame.particles.image
    if image is None:
        return np.zeros((n_particles, 3), dtype=np.int32)
    return np.asarray(image, dtype=np.int32)


def _active_path(
    source: gsd.hoomd.HOOMDTrajectory,
    active_ids: np.ndarray,
    case: NewSimCase,
) -> np.ndarray:
    wrapped = np.empty((len(source), len(active_ids), 3), dtype=np.float64)
    for frame_index in range(len(source)):
        frame = source[frame_index]
        positions = np.asarray(frame.particles.position, dtype=np.float64)[active_ids]
        images = _particle_images(frame, case.n_particles)[active_ids]
        wrapped[frame_index, :, 0] = positions[:, 0] + images[:, 0] * case.lx
        wrapped[frame_index, :, 1] = np.arctan2(
            positions[:, 1], positions[:, 2]
        )
        wrapped[frame_index, :, 2] = np.linalg.norm(positions[:, 1:3], axis=1)
    wrapped[:, :, 1] = np.unwrap(wrapped[:, :, 1], axis=0)
    return wrapped.astype(np.float32)


def _resume_frame(manifest: dict[str, Any]) -> int:
    shards = manifest.get("shards", [])
    return int(shards[-1]["frame_stop"]) if shards else 0


def analyze_case(
    case: NewSimCase,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    backend_name: str = "auto",
    require_gpu: bool = False,
    frames_per_shard: int = FRAMES_PER_SHARD,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    paths = CasePaths(case, output_root)
    if not paths.simulation_complete_json.exists():
        raise FileNotFoundError(f"missing completion marker: {paths.simulation_complete_json}")
    simulation_metadata = json.loads(paths.metadata_json.read_text())
    if simulation_metadata.get("status") != "complete":
        raise ValueError("simulation metadata is not complete")
    should_run, existing = prepare_analysis_dir(paths.analysis_dir, overwrite, resume)
    if not should_run:
        return
    backend = select_backend(backend_name, require_gpu=require_gpu)
    if existing is None:
        manifest: dict[str, Any] = {
            "schema": "hexatic.new_sims.analysis.v1",
            "case": simulation_metadata,
            "trajectory_gsd": str(paths.trajectory_gsd),
            "backend": backend.name,
            "device": backend.device_description,
            "frames_per_shard": frames_per_shard,
            "coordinate_order": (
                ["x", "theta", "r"]
                if case.is_cylinder
                else ["x", "y"]
                if case.is_2d
                else ["x", "y", "z"]
            ),
            "polarization_order": (
                None if case.is_2d else ["x", "y", "z"]
            ),
            "hexatic_definition": (
                None
                if not case.has_hexatic
                else "six-neighbor tangent-plane psi6"
                if not case.is_2d
                else "six-neighbor periodic planar psi6"
            ),
            "complete": False,
            "shards": [],
        }
        start_frame = 0
        write_json_atomic(paths.analysis_dir / "manifest.json", manifest)
    else:
        manifest = existing
        start_frame = _resume_frame(manifest)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["device"] = backend.device_description
        write_json_atomic(paths.analysis_dir / "manifest.json", manifest)

    active_ids = np.asarray(
        simulation_metadata.get("active_particle_ids", []), dtype=np.int64
    )
    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as source:
        if not len(source):
            raise ValueError(f"empty trajectory: {paths.trajectory_gsd}")
        if len(source) != simulation_metadata.get("frame_count"):
            raise ValueError("trajectory frame count does not match simulation metadata")
        tracer_path = (
            _active_path(source, active_ids, case) if case.is_tracer else None
        )
        static_tensors: dict[str, np.ndarray] = {
            "box": np.asarray(source[0].configuration.box, dtype=np.float32),
            "dimensions": np.asarray(case.dimensions, dtype=np.int32),
        }
        if case.is_tracer:
            active_mask = np.zeros(case.n_particles, dtype=np.bool_)
            active_mask[active_ids] = True
            static_tensors.update(
                active_mask=active_mask,
                active_particle_ids=active_ids,
            )
        static_path = paths.analysis_dir / "static.safetensors"
        if existing is None:
            save_safetensors_atomic(
                static_path,
                static_tensors,
                backend_name=backend.name,
                metadata={"schema": "hexatic.new_sims.static.v1"},
            )
        elif not static_path.is_file():
            raise FileNotFoundError(f"missing static tensors: {static_path}")

        writer = FrameShardWriter(
            paths.analysis_dir,
            manifest,
            backend_name=backend.name,
            frames_per_shard=frames_per_shard,
        )
        for frame_index in range(start_frame, len(source)):
            print(
                f"[new_sims.analysis] case={case.case_id} "
                f"frame={frame_index + 1}/{len(source)} backend={backend.name}",
                flush=True,
            )
            writer.add(
                analyze_frame(
                    source[frame_index],
                    case,
                    backend,
                    frame_index=frame_index,
                    active_path=tracer_path,
                )
            )
        writer.flush()
        manifest["frame_count"] = len(source)
    manifest["complete"] = True
    write_json_atomic(paths.analysis_dir / "manifest.json", manifest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one new simulation.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", choices=("auto", "jax", "numpy"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--frames-per-shard", type=int, default=FRAMES_PER_SHARD)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analyze_case(
        get_case(args.case),
        output_root=args.output_root,
        backend_name=args.backend,
        require_gpu=args.require_gpu,
        frames_per_shard=args.frames_per_shard,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
