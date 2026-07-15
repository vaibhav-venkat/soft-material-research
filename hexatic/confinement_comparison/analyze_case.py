from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import gsd.hoomd
import numpy as np

from hexatic.big_lx.backend import ArrayBackend, select_backend
from hexatic.big_lx.spatial import PeriodicXTree, exclude_self
from hexatic.constants import cylinder

from .cases import ComparisonCase, CasePaths, DEFAULT_OUTPUT_ROOT, GeometryKind, get_case
from .geometry import stored_to_logical
from .storage import (
    FrameShardWriter,
    prepare_analysis_dir,
    save_safetensors_atomic,
    write_json_atomic,
)

LOCAL_POCKET_RADIUS = 2.0 * cylinder.ANALYSIS.particle_diameter
FACE_NAMES = ("+y", "-y", "+z", "-z")


def _logged_force(frame, name: str, n_particles: int) -> np.ndarray:
    log = getattr(frame, "log", None)
    if not log:
        raise ValueError(f"GSD frame has no logger data; expected forces/{name}")
    candidates = []
    for key, value in log.items():
        key_text = "/".join(key) if isinstance(key, tuple) else str(key)
        array = np.asarray(value)
        if "forces" in key_text and name in key_text and array.shape[:1] == (n_particles,):
            candidates.append(np.asarray(array[:, :3], dtype=np.float32))
    if len(candidates) != 1:
        available = ", ".join(str(key) for key in log)
        raise ValueError(
            f"Expected one logged forces/{name} array, found {len(candidates)}; "
            f"available: {available}"
        )
    return candidates[0]


def _project_tangent(vectors: np.ndarray, positions: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    radii = np.linalg.norm(positions[:, 1:3], axis=1)
    normals[:, 1:3] = positions[:, 1:3] / radii[:, None]
    return vectors - np.sum(vectors * normals, axis=1)[:, None] * normals


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _pad_neighbor_width(
    source_ids: np.ndarray,
    bonds: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = _next_power_of_two(source_ids.shape[1])
    padding = width - source_ids.shape[1]
    if not padding:
        return source_ids, bonds, valid
    return (
        np.pad(source_ids, ((0, 0), (0, padding))),
        np.pad(bonds, ((0, 0), (0, padding), (0, 0))),
        np.pad(valid, ((0, 0), (0, padding))),
    )


def _mark_dislocations(
    positions: np.ndarray,
    charges: np.ndarray,
    lx: float,
) -> np.ndarray:
    result = np.zeros(len(positions), dtype=np.bool_)
    plus = np.flatnonzero(charges == 1)
    minus_mask = charges == -1
    if not len(plus) or not np.any(minus_mask):
        return result
    tree = PeriodicXTree.build(positions, lx)
    hits = tree.tree.query_ball_point(
        positions[plus],
        cylinder.ANALYSIS.dislocation_pair_distance,
        workers=-1,
        return_sorted=True,
    )
    for plus_index, row in zip(plus, hits):
        sources = tree.source_indices[np.asarray(row, dtype=np.int64)]
        opposite = sources[minus_mask[sources]]
        if len(opposite):
            result[plus_index] = True
            result[opposite] = True
    return result


def _cylinder_surface_fields(
    positions: np.ndarray,
    case: ComparisonCase,
    backend: ArrayBackend,
    particle_block_size: int,
) -> dict[str, np.ndarray]:
    n_particles = len(positions)
    tree = PeriodicXTree.build(positions, case.lx)
    _, bonds = tree.nearest_bonds(positions, cylinder.ANALYSIS.neighbors)
    psi_real, psi_imag = backend.hexatic(bonds, positions)
    counts = np.zeros(n_particles, dtype=np.int32)
    translation_chirality = np.zeros(n_particles, dtype=np.float32)
    radius = cylinder.ANALYSIS.neighbor_count_radius
    for start in range(0, n_particles, particle_block_size):
        stop = min(start + particle_block_size, n_particles)
        source_ids, block_bonds, valid = tree.radius_block(positions[start:stop], radius)
        query_ids = np.arange(start, stop, dtype=np.int64)
        neighbor_valid = exclude_self(source_ids, valid, query_ids)
        distances = np.linalg.norm(block_bonds, axis=2)
        within = neighbor_valid & (distances <= radius)
        counts[start:stop] = np.count_nonzero(within, axis=1).astype(np.int32)
        translation_chirality[start:stop] = np.sum(
            np.divide(
                block_bonds[:, :, 0],
                distances,
                out=np.zeros_like(distances, dtype=np.float32),
                where=within,
            ),
            axis=1,
        )
    charges = (cylinder.ANALYSIS.neighbors - counts).astype(np.int8)
    return {
        "surface_mask": np.ones(n_particles, dtype=np.bool_),
        "psi_real": np.asarray(psi_real, dtype=np.float32),
        "psi_imag": np.asarray(psi_imag, dtype=np.float32),
        "neighbor_counts": counts,
        "disclination_charges": charges,
        "dislocation_flags": _mark_dislocations(positions, charges, case.lx),
        "translation_chirality": translation_chirality,
    }


def _prism_face_geometry(
    positions: np.ndarray,
    case: ComparisonCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_y = 0.5 * case.prism_side - np.abs(positions[:, 1])
    distance_z = 0.5 * case.prism_side - np.abs(positions[:, 2])
    use_y = distance_y <= distance_z
    face_id = np.empty(len(positions), dtype=np.int8)
    face_id[use_y & (positions[:, 1] >= 0.0)] = 0
    face_id[use_y & (positions[:, 1] < 0.0)] = 1
    face_id[~use_y & (positions[:, 2] >= 0.0)] = 2
    face_id[~use_y & (positions[:, 2] < 0.0)] = 3
    normals = np.zeros_like(positions, dtype=np.float32)
    normals[face_id == 0, 1] = 1.0
    normals[face_id == 1, 1] = -1.0
    normals[face_id == 2, 2] = 1.0
    normals[face_id == 3, 2] = -1.0
    tangents = np.zeros_like(normals)
    tangents[:, 1] = normals[:, 2]
    tangents[:, 2] = -normals[:, 1]
    nearest = np.minimum(distance_y, distance_z).astype(np.float32)
    corner = (
        (distance_y <= cylinder.ANALYSIS.neighbor_count_radius)
        & (distance_z <= cylinder.ANALYSIS.neighbor_count_radius)
    )
    return face_id, normals, tangents, nearest, corner


def _face_components(
    vectors: np.ndarray,
    normals: np.ndarray,
    tangents: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        (
            vectors[:, 0],
            np.sum(vectors * tangents, axis=1),
            np.sum(vectors * normals, axis=1),
        )
    ).astype(np.float32)


def _prism_surface_fields(
    positions: np.ndarray,
    case: ComparisonCase,
    particle_block_size: int,
) -> dict[str, np.ndarray]:
    del particle_block_size
    face_id, normals, tangents, wall_distance, corner = _prism_face_geometry(
        positions, case
    )
    surface = wall_distance <= cylinder.ANALYSIS.shell_delta
    valid_hexatic = surface & ~corner
    n_particles = len(positions)
    psi_real = np.zeros(n_particles, dtype=np.float32)
    psi_imag = np.zeros(n_particles, dtype=np.float32)
    counts = np.zeros(n_particles, dtype=np.int32)
    valid_result = np.zeros(n_particles, dtype=np.bool_)
    for face in range(4):
        face_indices = np.flatnonzero(valid_hexatic & (face_id == face))
        if len(face_indices) <= cylinder.ANALYSIS.neighbors:
            continue
        face_positions = positions[face_indices]
        tree = PeriodicXTree.build(face_positions, case.lx)
        source_ids, bonds = tree.nearest_bonds(
            face_positions, cylinder.ANALYSIS.neighbors
        )
        del source_ids
        tangent = tangents[face_indices]
        bond_tangent = np.sum(bonds * tangent[:, None, :], axis=2)
        angles = np.arctan2(bond_tangent, bonds[:, :, 0])
        psi_real[face_indices] = np.mean(np.cos(6.0 * angles), axis=1)
        psi_imag[face_indices] = np.mean(np.sin(6.0 * angles), axis=1)
        hits = tree.tree.query_ball_point(
            face_positions,
            cylinder.ANALYSIS.neighbor_count_radius,
            workers=-1,
            return_length=True,
        )
        counts[face_indices] = np.asarray(hits, dtype=np.int32) - 1
        valid_result[face_indices] = True
    charges = np.zeros(n_particles, dtype=np.int8)
    charges[valid_result] = (
        cylinder.ANALYSIS.neighbors - counts[valid_result]
    ).astype(np.int8)
    return {
        "surface_mask": surface.astype(np.bool_),
        "face_id": face_id,
        "face_normal": normals,
        "face_tangent": tangents,
        "wall_distance": wall_distance,
        "corner_mask": corner.astype(np.bool_),
        "face_hexatic_valid": valid_result,
        "psi_real": psi_real,
        "psi_imag": psi_imag,
        "neighbor_counts": counts,
        "disclination_charges": charges,
        "dislocation_flags": _mark_dislocations(positions, charges, case.lx),
    }


def analyze_frame(
    frame,
    case: ComparisonCase,
    backend: ArrayBackend,
    *,
    frame_index: int,
    pocket_radius: float,
    gaussian_cutoff_multiplier: float,
    particle_block_size: int,
) -> dict[str, np.ndarray]:
    n_particles = int(frame.particles.N)
    if n_particles != case.n_particles:
        raise ValueError(f"expected {case.n_particles} particles, got {n_particles}")
    stored_positions = np.asarray(frame.particles.position, dtype=np.float32)
    stored_orientation = np.asarray(frame.particles.orientation, dtype=np.float32)
    positions = stored_to_logical(stored_positions, case).astype(np.float32)
    orientation_stored = backend.directions(stored_orientation).astype(np.float32)
    orientation_direction = stored_to_logical(orientation_stored, case).astype(np.float32)
    pair_force = stored_to_logical(
        _logged_force(frame, "pair", n_particles), case
    ).astype(np.float32)
    wall_force = stored_to_logical(
        _logged_force(frame, "wall", n_particles), case
    ).astype(np.float32)
    active_force = stored_to_logical(
        _logged_force(frame, "active", n_particles), case
    ).astype(np.float32)
    mechanical_force = pair_force + wall_force
    propulsion_direction = active_force / float(
        cylinder.SIMULATION.gamma * cylinder.SIMULATION.u0
    )
    if case.is_constrained:
        mechanical_force = _project_tangent(mechanical_force, positions)
        if case.kind == GeometryKind.CYLINDER_RATTLE:
            propulsion_direction = _project_tangent(orientation_direction, positions)
    force_velocity = mechanical_force / float(cylinder.SIMULATION.gamma)
    velocity = (
        float(cylinder.SIMULATION.u0) * propulsion_direction + force_velocity
    ).astype(np.float32)

    rho = np.zeros(n_particles, dtype=np.float32)
    propulsion_p = np.zeros((n_particles, 3), dtype=np.float32)
    orientation_p = np.zeros((n_particles, 3), dtype=np.float32)
    force_density = np.zeros((n_particles, 3), dtype=np.float32)
    flux = np.zeros((n_particles, 3), dtype=np.float32)
    tree = PeriodicXTree.build(positions, case.lx)
    cutoff = gaussian_cutoff_multiplier * pocket_radius
    for start in range(0, n_particles, particle_block_size):
        stop = min(start + particle_block_size, n_particles)
        source_ids, bonds, valid = tree.radius_block(positions[start:stop], cutoff)
        source_ids, bonds, valid = _pad_neighbor_width(source_ids, bonds, valid)
        distances_sq = np.sum(bonds * bonds, axis=2, dtype=np.float32)
        combined_directions = np.concatenate(
            (
                propulsion_direction[source_ids],
                orientation_direction[source_ids],
            ),
            axis=2,
        )
        block_rho, combined_p, block_force, block_flux = backend.weighted_fields(
            distances_sq,
            valid,
            combined_directions,
            force_velocity[source_ids],
            velocity[source_ids],
            pocket_radius,
        )
        rho[start:stop] = block_rho
        propulsion_p[start:stop] = combined_p[:, :3]
        orientation_p[start:stop] = combined_p[:, 3:]
        force_density[start:stop] = block_force
        flux[start:stop] = block_flux
    polarization = np.divide(
        propulsion_p,
        rho[:, None],
        out=np.zeros_like(propulsion_p),
        where=rho[:, None] > np.finfo(np.float32).eps,
    )
    result: dict[str, np.ndarray] = {
        "frame_index": np.asarray(frame_index, dtype=np.int64),
        "step": np.asarray(int(frame.configuration.step), dtype=np.int64),
        "position_cartesian": positions,
        "rho": rho,
        "orientation_direction_cartesian": orientation_direction,
        "propulsion_direction_cartesian": propulsion_direction,
        "P_cartesian": propulsion_p,
        "orientation_P_cartesian": orientation_p,
        "polarization_cartesian": polarization,
        "force_density_cartesian": force_density,
        "flux_cartesian": flux,
        "velocity_cartesian": velocity,
    }
    if case.is_constrained:
        coords = backend.coordinates(positions).astype(np.float32)
        theta = coords[:, 1]
        result.update(
            {
                "coords": coords,
                "propulsion_direction_cylindrical": backend.cylindrical(
                    propulsion_direction, theta
                ).astype(np.float32),
                "P_cylindrical": backend.cylindrical(propulsion_p, theta).astype(
                    np.float32
                ),
                "polarization_cylindrical": backend.cylindrical(
                    polarization, theta
                ).astype(np.float32),
                "force_density_cylindrical": backend.cylindrical(
                    force_density, theta
                ).astype(np.float32),
                "flux_cylindrical": backend.cylindrical(flux, theta).astype(
                    np.float32
                ),
            }
        )
        result.update(
            _cylinder_surface_fields(positions, case, backend, particle_block_size)
        )
    else:
        surface = _prism_surface_fields(positions, case, particle_block_size)
        result.update(surface)
        result.update(
            {
                "P_face_local": _face_components(
                    propulsion_p, surface["face_normal"], surface["face_tangent"]
                ),
                "polarization_face_local": _face_components(
                    polarization, surface["face_normal"], surface["face_tangent"]
                ),
                "flux_face_local": _face_components(
                    flux, surface["face_normal"], surface["face_tangent"]
                ),
            }
        )
    return result


def _resume_frame(manifest: dict[str, object], paths: CasePaths, case: ComparisonCase) -> int:
    if manifest.get("schema") != "hexatic.confinement_comparison.analysis.v1":
        raise ValueError("incompatible analysis manifest schema")
    payload = manifest.get("case")
    if not isinstance(payload, dict) or payload.get("case_id") != case.case_id:
        raise ValueError("analysis manifest case mismatch")
    expected = 0
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise ValueError("analysis manifest has no shard list")
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("analysis manifest contains an invalid shard")
        shard_payload = cast(dict[str, object], shard)
        if shard_payload.get("frame_start") != expected:
            raise ValueError("analysis shards are not contiguous")
        stop = shard_payload.get("frame_stop")
        filename = shard_payload.get("file")
        steps = shard_payload.get("steps")
        if (
            not isinstance(stop, int)
            or not isinstance(filename, str)
            or not isinstance(steps, list)
            or len(steps) != stop - expected
        ):
            raise ValueError("analysis shard metadata is invalid")
        expected = stop
        path = paths.analysis_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing analysis shard: {path}")
    return expected


def analyze_case(
    case: ComparisonCase,
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
    paths = CasePaths(case, output_root)
    if not paths.simulation_complete_json.exists():
        raise FileNotFoundError(f"missing completion marker: {paths.simulation_complete_json}")
    if not paths.trajectory_gsd.exists():
        raise FileNotFoundError(f"missing trajectory: {paths.trajectory_gsd}")
    if not paths.metadata_json.exists():
        raise FileNotFoundError(f"missing simulation metadata: {paths.metadata_json}")
    simulation_metadata = json.loads(paths.metadata_json.read_text())
    completion_metadata = json.loads(paths.simulation_complete_json.read_text())
    if (
        simulation_metadata.get("case_id") != case.case_id
        or simulation_metadata.get("status") != "complete"
        or completion_metadata.get("case_id") != case.case_id
        or completion_metadata.get("status") != "complete"
    ):
        raise ValueError("simulation metadata is incomplete or belongs to another case")
    should_run, existing = prepare_analysis_dir(paths.analysis_dir, overwrite, resume)
    if not should_run:
        return
    backend = select_backend(backend_name, require_gpu=require_gpu)
    if existing is None:
        manifest: dict[str, object] = {
            "schema": "hexatic.confinement_comparison.analysis.v1",
            "case": simulation_metadata,
            "trajectory_gsd": str(paths.trajectory_gsd),
            "backend": backend.name,
            "device": backend.device_description,
            "pocket_radius": pocket_radius,
            "gaussian_cutoff_multiplier": gaussian_cutoff_multiplier,
            "particle_block_size": particle_block_size,
            "target_shard_mib": target_shard_mib,
            "cartesian_component_order": ["x", "y", "z"],
            "cylindrical_component_order": ["x", "radial", "azimuthal"],
            "face_component_order": ["axial", "in_face_tangent", "outward_normal"],
            "face_names": list(FACE_NAMES),
            "complete": False,
            "shards": [],
        }
        start_frame = 0
        write_json_atomic(paths.analysis_dir / "manifest.json", manifest)
    else:
        manifest = existing
        requested = {
            "backend": backend.name,
            "pocket_radius": pocket_radius,
            "gaussian_cutoff_multiplier": gaussian_cutoff_multiplier,
            "particle_block_size": particle_block_size,
            "target_shard_mib": target_shard_mib,
        }
        for name, value in requested.items():
            if manifest.get(name) != value:
                raise ValueError(
                    f"cannot resume with changed {name}: "
                    f"stored={manifest.get(name)!r}, requested={value!r}"
                )
        start_frame = _resume_frame(manifest, paths, case)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        write_json_atomic(paths.analysis_dir / "manifest.json", manifest)

    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as source:
        if not len(source):
            raise ValueError(f"empty trajectory: {paths.trajectory_gsd}")
        final_step = int(source[-1].configuration.step)
        if (
            simulation_metadata.get("frame_count") != len(source)
            or completion_metadata.get("frame_count") != len(source)
            or simulation_metadata.get("final_step") != final_step
            or completion_metadata.get("final_step") != final_step
        ):
            raise ValueError("trajectory does not match its simulation metadata")
        if start_frame > len(source):
            raise ValueError("resume frame exceeds trajectory length")
        if start_frame:
            shards = manifest.get("shards")
            if not isinstance(shards, list) or not shards:
                raise ValueError("resume manifest has no completed shards")
            last_shard = shards[-1]
            if not isinstance(last_shard, dict):
                raise ValueError("resume manifest has an invalid final shard")
            saved_steps = last_shard.get("steps")
            if not isinstance(saved_steps, list) or not saved_steps:
                raise ValueError("resume manifest final shard has no steps")
            trajectory_step = int(source[start_frame - 1].configuration.step)
            if saved_steps[-1] != trajectory_step:
                raise ValueError(
                    "last saved analysis step does not match the current trajectory"
                )
        static_path = paths.analysis_dir / "static.safetensors"
        if existing is None:
            save_safetensors_atomic(
                static_path,
                {
                    "stored_box": np.asarray(source[0].configuration.box, dtype=np.float32),
                    "logical_lx": np.asarray(case.lx, dtype=np.float32),
                    "radius": np.asarray(case.radius, dtype=np.float32),
                    "circumference": np.asarray(case.circumference, dtype=np.float32),
                    "prism_side": np.asarray(case.prism_side, dtype=np.float32),
                    "logical_to_stored_axes": np.asarray(
                        case.logical_to_stored_axes, dtype=np.int32
                    ),
                },
                backend_name=backend.name,
                metadata={"schema": "hexatic.confinement_comparison.static.v1"},
            )
        elif not static_path.is_file() or static_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing static resume data: {static_path}")
        writer = FrameShardWriter(
            paths.analysis_dir,
            manifest,
            backend_name=backend.name,
            target_bytes=target_shard_mib * 1024 * 1024,
        )
        for frame_index in range(start_frame, len(source)):
            print(
                f"[confinement.analysis] case={case.case_id} "
                f"frame={frame_index + 1}/{len(source)} backend={backend.name}",
                flush=True,
            )
            writer.add(
                analyze_frame(
                    source[frame_index],
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one confinement trajectory.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--backend", choices=("auto", "jax", "numpy"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--pocket-radius", type=float, default=LOCAL_POCKET_RADIUS)
    parser.add_argument("--gaussian-cutoff-multiplier", type=float, default=5.0)
    parser.add_argument("--particle-block-size", type=int, default=2048)
    parser.add_argument("--target-shard-mib", type=int, default=256)
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
        pocket_radius=args.pocket_radius,
        gaussian_cutoff_multiplier=args.gaussian_cutoff_multiplier,
        particle_block_size=args.particle_block_size,
        target_shard_mib=args.target_shard_mib,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
