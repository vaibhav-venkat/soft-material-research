from __future__ import annotations

import argparse
import math
from pathlib import Path

import gsd.hoomd
import numpy as np

from hexatic.constants import cylinder

from .backend import ArrayBackend, select_backend
from .cases import BigLxCase, CasePaths, DEFAULT_OUTPUT_ROOT, get_case
from .spatial import PeriodicXTree, exclude_self
from .storage import (
    FrameShardWriter,
    prepare_analysis_dir,
    save_safetensors_atomic,
    write_json_atomic,
)

LOCAL_POCKET_RADIUS = 2.0 * cylinder.ANALYSIS.particle_diameter


def _logged_particle_array(frame, quantity: str, n_particles: int) -> np.ndarray:
    log = getattr(frame, "log", None)
    if not log:
        raise ValueError(f"GSD frame has no logger data; expected {quantity}")
    candidates = []
    for key, value in log.items():
        array = np.asarray(value)
        if quantity.lower() in str(key).lower() and array.shape[:1] == (n_particles,):
            candidates.append(np.asarray(array, dtype=np.float32))
    if not candidates:
        available = ", ".join(str(key) for key in log)
        raise ValueError(f"Missing logged {quantity}; available keys: {available}")
    total = np.zeros_like(candidates[0], dtype=np.float32)
    for candidate in candidates:
        total += candidate
    return total


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _pad_neighbor_width(
    source_ids: np.ndarray,
    bonds: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = _next_power_of_two(source_ids.shape[1])
    if width == source_ids.shape[1]:
        return source_ids, bonds, valid
    padding = width - source_ids.shape[1]
    return (
        np.pad(source_ids, ((0, 0), (0, padding))),
        np.pad(bonds, ((0, 0), (0, padding), (0, 0))),
        np.pad(valid, ((0, 0), (0, padding))),
    )


def _mark_dislocations(
    tree: PeriodicXTree,
    positions: np.ndarray,
    charges: np.ndarray,
    pair_distance: float,
) -> np.ndarray:
    result = np.zeros(len(positions), dtype=np.bool_)
    plus = np.flatnonzero(charges == 1)
    if not len(plus) or not np.any(charges == -1):
        return result
    hit_lists = tree.tree.query_ball_point(
        positions[plus],
        pair_distance,
        workers=-1,
        return_sorted=True,
    )
    for plus_index, hits in zip(plus, hit_lists):
        sources = tree.source_indices[np.asarray(hits, dtype=np.int64)]
        opposite = sources[charges[sources] == -1]
        if len(opposite):
            result[plus_index] = True
            result[opposite] = True
    return result


def _analyze_shell_fields(
    positions: np.ndarray,
    case: BigLxCase,
    backend: ArrayBackend,
    particle_block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    radial = np.linalg.norm(positions[:, 1:3], axis=1)
    shell_mask = radial > case.radius - cylinder.ANALYSIS.shell_delta
    shell_indices = np.flatnonzero(shell_mask)
    psi_real = np.zeros(len(positions), dtype=np.float32)
    psi_imag = np.zeros(len(positions), dtype=np.float32)
    counts = np.zeros(len(positions), dtype=np.int32)
    if len(shell_indices) <= cylinder.ANALYSIS.neighbors:
        return psi_real, psi_imag, counts, shell_mask, math.nan, 0

    shell_positions = positions[shell_indices]
    tree = PeriodicXTree.build(shell_positions, case.lx)
    _, nearest_bonds = tree.nearest_bonds(
        shell_positions,
        cylinder.ANALYSIS.neighbors,
    )
    real, imaginary = backend.hexatic(nearest_bonds, shell_positions)
    psi_real[shell_indices] = real.astype(np.float32)
    psi_imag[shell_indices] = imaginary.astype(np.float32)

    chirality_total = 0.0
    chirality_count = 0
    radius = cylinder.ANALYSIS.neighbor_count_radius
    for start in range(0, len(shell_positions), particle_block_size):
        stop = min(start + particle_block_size, len(shell_positions))
        source_ids, bonds, valid = tree.radius_block(shell_positions[start:stop], radius)
        query_ids = np.arange(start, stop, dtype=np.int64)
        neighbor_valid = exclude_self(source_ids, valid, query_ids)
        distances = np.linalg.norm(bonds, axis=2)
        within = neighbor_valid & (distances <= radius)
        counts[shell_indices[start:stop]] = np.count_nonzero(within, axis=1).astype(
            np.int32
        )
        ratios = np.divide(
            np.abs(bonds[:, :, 0]),
            distances,
            out=np.zeros_like(distances, dtype=np.float32),
            where=within,
        )
        chirality_total += float(np.sum(ratios))
        chirality_count += int(np.count_nonzero(within))
    shell_mean = (
        chirality_total / chirality_count if chirality_count else math.nan
    )
    return psi_real, psi_imag, counts, shell_mask, shell_mean, chirality_count


def analyze_frame(
    frame,
    case: BigLxCase,
    backend: ArrayBackend,
    *,
    frame_index: int,
    pocket_radius: float,
    gaussian_cutoff_multiplier: float,
    particle_block_size: int,
) -> dict[str, np.ndarray]:
    n_particles = int(frame.particles.N)
    if n_particles != case.n_particles:
        raise ValueError(
            f"case {case.case_id} expects {case.n_particles} particles, got {n_particles}"
        )
    positions = np.asarray(frame.particles.position, dtype=np.float32)
    orientation = np.asarray(frame.particles.orientation, dtype=np.float32)
    forces = _logged_particle_array(frame, "forces", n_particles)[:, :3]
    force_velocity = forces / float(cylinder.SIMULATION.gamma)

    directions = backend.directions(orientation).astype(np.float32)
    coords = backend.coordinates(positions).astype(np.float32)
    velocities = (
        float(cylinder.SIMULATION.u0) * directions + force_velocity
    ).astype(np.float32)
    direction_cylindrical = backend.cylindrical(directions, coords[:, 1]).astype(
        np.float32
    )

    rho = np.zeros(n_particles, dtype=np.float32)
    polar = np.zeros((n_particles, 3), dtype=np.float32)
    force_density = np.zeros((n_particles, 3), dtype=np.float32)
    flux = np.zeros((n_particles, 3), dtype=np.float32)
    translation_chirality = np.zeros(n_particles, dtype=np.float32)

    tree = PeriodicXTree.build(positions, case.lx)
    cutoff = gaussian_cutoff_multiplier * pocket_radius
    chirality_radius = cylinder.ANALYSIS.neighbor_count_radius
    for start in range(0, n_particles, particle_block_size):
        stop = min(start + particle_block_size, n_particles)
        source_ids, bonds, valid = tree.radius_block(positions[start:stop], cutoff)
        source_ids, bonds, valid = _pad_neighbor_width(source_ids, bonds, valid)
        distances_sq = np.sum(bonds * bonds, axis=2, dtype=np.float32)
        block_rho, block_polar, block_force, block_flux = backend.weighted_fields(
            distances_sq,
            valid,
            directions[source_ids],
            force_velocity[source_ids],
            velocities[source_ids],
            pocket_radius,
        )
        rho[start:stop] = block_rho
        polar[start:stop] = block_polar
        force_density[start:stop] = block_force
        flux[start:stop] = block_flux

        query_ids = np.arange(start, stop, dtype=np.int64)
        neighbor_valid = exclude_self(source_ids, valid, query_ids)
        distances = np.sqrt(distances_sq)
        chirality_valid = neighbor_valid & (distances <= chirality_radius)
        translation_chirality[start:stop] = np.sum(
            np.divide(
                bonds[:, :, 0],
                distances,
                out=np.zeros_like(distances, dtype=np.float32),
                where=chirality_valid,
            ),
            axis=1,
        )

    polar_cylindrical = backend.cylindrical(polar, coords[:, 1]).astype(np.float32)
    force_density_cylindrical = backend.cylindrical(
        force_density, coords[:, 1]
    ).astype(np.float32)
    flux_cylindrical = backend.cylindrical(flux, coords[:, 1]).astype(np.float32)

    (
        psi_real,
        psi_imag,
        neighbor_counts,
        hexatic_shell_mask,
        shell_chirality_mean,
        shell_bond_count,
    ) = _analyze_shell_fields(positions, case, backend, particle_block_size)
    charges = np.zeros(n_particles, dtype=np.int8)
    charges[hexatic_shell_mask] = (
        cylinder.ANALYSIS.neighbors - neighbor_counts[hexatic_shell_mask]
    ).astype(np.int8)
    dislocations = _mark_dislocations(
        PeriodicXTree.build(positions, case.lx),
        positions,
        charges,
        cylinder.ANALYSIS.dislocation_pair_distance,
    )
    active_shell_mask = (
        (coords[:, 2] > case.radius - cylinder.ANALYSIS.wall_cutoff)
        & (coords[:, 2] < case.radius)
    )

    return {
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "step": np.asarray(int(frame.configuration.step), dtype=np.int64),
        "coords": coords,
        "active_shell_mask": active_shell_mask.astype(np.bool_),
        "hexatic_shell_mask": hexatic_shell_mask.astype(np.bool_),
        "rho": rho,
        "active_direction": directions,
        "direction_cylindrical": direction_cylindrical,
        "polar_density": polar,
        "polar_cylindrical": polar_cylindrical,
        "flux_cylindrical": flux_cylindrical,
        "force_density": force_density,
        "force_density_cylindrical": force_density_cylindrical,
        "psi_real": psi_real,
        "psi_imag": psi_imag,
        "neighbor_counts": neighbor_counts,
        "disclination_charges": charges,
        "dislocation_flags": dislocations,
        "translation_chirality": translation_chirality,
        "shell_bond_translation_chirality_mean": np.asarray(
            shell_chirality_mean, dtype=np.float32
        ),
        "shell_bond_count": np.asarray(shell_bond_count, dtype=np.int64),
    }


def analyze_case(
    case: BigLxCase,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    backend_name: str = "auto",
    require_gpu: bool = False,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    gaussian_cutoff_multiplier: float = 5.0,
    particle_block_size: int = 2048,
    target_shard_mib: int = 256,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    if gaussian_cutoff_multiplier <= 0.0:
        raise ValueError("gaussian_cutoff_multiplier must be positive")
    if particle_block_size < 1:
        raise ValueError("particle_block_size must be positive")
    if target_shard_mib < 1:
        raise ValueError("target_shard_mib must be positive")

    paths = CasePaths(case, output_root)
    if not paths.simulation_complete_json.exists():
        raise FileNotFoundError(
            f"Simulation completion marker is missing: {paths.simulation_complete_json}"
        )
    if not paths.trajectory_gsd.exists():
        raise FileNotFoundError(f"Missing trajectory: {paths.trajectory_gsd}")
    should_run, existing_manifest = prepare_analysis_dir(
        paths.analysis_dir,
        overwrite,
        resume,
    )
    if not should_run:
        return

    backend = select_backend(backend_name, require_gpu=require_gpu)
    if existing_manifest is None:
        manifest: dict[str, object] = {
            "schema": "hexatic.big_lx.analysis.v1",
            "case": case.as_metadata(),
            "trajectory_gsd": str(paths.trajectory_gsd),
            "backend": backend.name,
            "device": backend.device_description,
            "pocket_radius": pocket_radius,
            "gaussian_cutoff_multiplier": gaussian_cutoff_multiplier,
            "gaussian_cutoff": gaussian_cutoff_multiplier * pocket_radius,
            "particle_block_size": particle_block_size,
            "target_shard_mib": target_shard_mib,
            "dtype": "float32",
            "complete": False,
            "shards": [],
        }
        start_frame = 0
        write_json_atomic(paths.analysis_dir / "manifest.json", manifest)
    else:
        manifest = existing_manifest
        start_frame = _validate_resume_manifest(
            manifest,
            paths,
            case,
            backend,
            pocket_radius=pocket_radius,
            gaussian_cutoff_multiplier=gaussian_cutoff_multiplier,
            particle_block_size=particle_block_size,
            target_shard_mib=target_shard_mib,
        )
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["resumed_from_frame"] = start_frame
        manifest["device"] = backend.device_description
        write_json_atomic(paths.analysis_dir / "manifest.json", manifest)
        print(
            f"[big_lx.analysis] resuming case={case.case_id} frame={start_frame}",
            flush=True,
        )

    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as source:
        if not len(source):
            raise ValueError(f"Trajectory contains no frames: {paths.trajectory_gsd}")
        if start_frame > len(source):
            raise ValueError(
                f"Resume frame {start_frame} exceeds trajectory length {len(source)}"
            )
        if start_frame and manifest["shards"][-1]["steps"][-1] != int(
            source[start_frame - 1].configuration.step
        ):
            raise ValueError("Last saved shard step does not match the trajectory")
        static_path = paths.analysis_dir / "static.safetensors"
        if existing_manifest is None:
            first_box = np.asarray(source[0].configuration.box, dtype=np.float32)
            save_safetensors_atomic(
                static_path,
                {
                    "box": first_box,
                    "radius": np.asarray(case.radius, dtype=np.float32),
                    "circumference": np.asarray(case.circumference, dtype=np.float32),
                    "lx": np.asarray(case.lx, dtype=np.float32),
                    "pocket_radius": np.asarray(pocket_radius, dtype=np.float32),
                    "gaussian_cutoff": np.asarray(
                        gaussian_cutoff_multiplier * pocket_radius, dtype=np.float32
                    ),
                },
                backend_name=backend.name,
                metadata={"schema": "hexatic.big_lx.static.v1"},
            )
        elif not static_path.exists():
            raise FileNotFoundError(f"Missing static resume data: {static_path}")
        writer = FrameShardWriter(
            paths.analysis_dir,
            manifest,
            backend_name=backend.name,
            target_bytes=target_shard_mib * 1024 * 1024,
        )
        for frame_index in range(start_frame, len(source)):
            frame = source[frame_index]
            print(
                f"[big_lx.analysis] case={case.case_id} "
                f"frame={frame_index + 1}/{len(source)} backend={backend.name}",
                flush=True,
            )
            writer.add(
                analyze_frame(
                    frame,
                    case,
                    backend,
                    frame_index=frame_index,
                    pocket_radius=pocket_radius,
                    gaussian_cutoff_multiplier=gaussian_cutoff_multiplier,
                    particle_block_size=particle_block_size,
                )
            )
        writer.flush()
        manifest["frame_count"] = len(source)
    manifest["complete"] = True
    write_json_atomic(paths.analysis_dir / "manifest.json", manifest)


def _validate_resume_manifest(
    manifest: dict[str, object],
    paths: CasePaths,
    case: BigLxCase,
    backend: ArrayBackend,
    *,
    pocket_radius: float,
    gaussian_cutoff_multiplier: float,
    particle_block_size: int,
    target_shard_mib: int,
) -> int:
    if manifest.get("schema") != "hexatic.big_lx.analysis.v1":
        raise ValueError("Cannot resume an analysis with an incompatible schema")
    case_payload = manifest.get("case")
    if not isinstance(case_payload, dict) or case_payload.get("case_id") != case.case_id:
        raise ValueError("Resume manifest case does not match the requested case")
    expected = {
        "backend": backend.name,
        "pocket_radius": pocket_radius,
        "gaussian_cutoff_multiplier": gaussian_cutoff_multiplier,
        "particle_block_size": particle_block_size,
        "target_shard_mib": target_shard_mib,
        "dtype": "float32",
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"Cannot resume with changed {name}: "
                f"manifest={manifest.get(name)!r}, requested={value!r}"
            )
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("Resume manifest has no shard list")
    expected_start = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("Resume manifest contains an invalid shard entry")
        start = shard.get("frame_start")
        stop = shard.get("frame_stop")
        filename = shard.get("file")
        if start != expected_start or not isinstance(stop, int) or stop <= expected_start:
            raise ValueError("Resume shards are not contiguous from frame zero")
        if not isinstance(filename, str):
            raise ValueError("Resume shard filename is invalid")
        shard_path = paths.analysis_dir / filename
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty resume shard: {shard_path}")
        steps = shard.get("steps")
        if not isinstance(steps, list) or len(steps) != stop - expected_start:
            raise ValueError(f"Resume shard has invalid step metadata: {shard_path}")
        expected_start = stop
    return expected_start


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one big-Lx trajectory.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", choices=("auto", "jax", "numpy"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--pocket-radius", type=float, default=LOCAL_POCKET_RADIUS)
    parser.add_argument("--gaussian-cutoff-multiplier", type=float, default=5.0)
    parser.add_argument("--particle-block-size", type=int, default=2048)
    parser.add_argument("--target-shard-mib", type=int, default=256)
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--overwrite", action="store_true")
    write_mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    analyze_case(
        get_case(args.case),
        output_root=args.output_root,
        backend_name=args.backend,
        require_gpu=args.require_gpu,
        pocket_radius=args.pocket_radius,
        gaussian_cutoff_multiplier=args.gaussian_cutoff_multiplier,
        particle_block_size=args.particle_block_size,
        target_shard_mib=args.target_shard_mib,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
