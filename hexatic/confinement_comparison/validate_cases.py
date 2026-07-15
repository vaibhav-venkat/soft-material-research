from __future__ import annotations

import argparse
import json
import math

import numpy as np

from hexatic.big_lx.backend import select_backend
from hexatic.big_lx.lattice import generate_unwrapped_lattice
from hexatic.big_lx.spatial import PeriodicXTree
from hexatic.constants import cylinder

from .cases import GeometryKind, select_cases
from .geometry import (
    generate_cylinder_film,
    generate_prism_lattice,
    stored_to_logical,
)


def validate_case(case) -> dict[str, object]:
    backend = select_backend("numpy")
    if case.kind == GeometryKind.PRISM_VOLUME:
        positions, quaternions, expected_directions = generate_prism_lattice(case)
        volume = case.prism_side**2 * case.lx
        if not math.isclose(volume, case.base.volume, rel_tol=1e-12):
            raise AssertionError("prism and cylinder volumes differ")
        if not np.allclose(np.mean(positions, axis=0), 0.0, atol=1e-12):
            raise AssertionError("prism lattice center of mass is not zero")
        actual_directions = backend.directions(quaternions)
        alignment = np.sum(actual_directions * expected_directions, axis=1)
        tree = PeriodicXTree.build(positions, case.lx)
        distances, _ = tree.tree.query(positions, k=2, workers=-1)
        minimum_distance = float(np.min(distances[:, 1]))
        if minimum_distance <= cylinder.ANALYSIS.wall_cutoff:
            raise AssertionError("prism lattice contains an initial interaction overlap")
        if float(np.min(alignment)) < 0.99999:
            raise AssertionError("prism orientations are not nearest-wall normal")
        direction_mean = np.mean(expected_directions, axis=0)
        if float(np.linalg.norm(direction_mean)) > 1e-12:
            raise AssertionError("prism wall-normal orientations are not balanced")
        face_counts = {
            "+y": int(np.count_nonzero(expected_directions[:, 1] > 0.0)),
            "-y": int(np.count_nonzero(expected_directions[:, 1] < 0.0)),
            "+z": int(np.count_nonzero(expected_directions[:, 2] > 0.0)),
            "-z": int(np.count_nonzero(expected_directions[:, 2] < 0.0)),
        }
        return {
            "case_id": case.case_id,
            "n_particles": len(positions),
            "volume": volume,
            "volume_relative_error": abs(volume / case.base.volume - 1.0),
            "minimum_pair_distance": minimum_distance,
            "orientation_alignment_min": float(np.min(alignment)),
            "direction_mean": direction_mean.tolist(),
            "face_counts": face_counts,
            "center_of_mass": np.mean(positions, axis=0).tolist(),
        }

    stored_positions, stored_quaternions, stored_directions = generate_cylinder_film(case)
    positions = stored_to_logical(stored_positions, case)
    directions = stored_to_logical(stored_directions, case)
    actual_directions = stored_to_logical(
        backend.directions(stored_quaternions), case
    )
    base_positions, _ = generate_unwrapped_lattice(case.base)
    if not np.allclose(positions, base_positions, atol=1e-12):
        raise AssertionError("stored-axis rotation changed the logical twisted lattice")
    radius_residual = np.max(
        np.abs(np.linalg.norm(positions[:, 1:3], axis=1) - case.radius)
    )
    tree = PeriodicXTree.build(positions, case.lx)
    _, bonds = tree.nearest_bonds(positions, cylinder.ANALYSIS.neighbors)
    psi_real, psi_imag = backend.hexatic(bonds, positions)
    psi_abs = np.hypot(psi_real, psi_imag)
    direction_error = float(np.max(np.abs(actual_directions - directions)))
    normals = np.zeros_like(positions)
    normals[:, 1:3] = positions[:, 1:3] / np.linalg.norm(
        positions[:, 1:3], axis=1, keepdims=True
    )
    normal_component = np.sum(actual_directions * normals, axis=1)
    if case.kind == GeometryKind.CYLINDER_RATTLE:
        if float(np.min(normal_component)) < 0.99999:
            raise AssertionError("RATTLE case does not start outward-normal")
    elif float(np.max(np.abs(normal_component))) > 1e-6:
        raise AssertionError("ActiveOnManifold case does not start tangent")
    if float(np.min(psi_abs)) < 0.99:
        raise AssertionError("cylinder film lost perfect initial hexatic order")
    if direction_error > 1e-5:
        raise AssertionError("orientation quaternion does not encode expected direction")
    return {
        "case_id": case.case_id,
        "n_particles": len(positions),
        "radius_residual_max": float(radius_residual),
        "psi_abs_min": float(np.min(psi_abs)),
        "direction_error_max": direction_error,
        "normal_component_min": float(np.min(normal_component)),
        "normal_component_max": float(np.max(normal_component)),
        "direction_mean": np.mean(actual_directions, axis=0).tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate confinement geometries.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = [validate_case(case) for case in select_cases(args.all, args.case)]
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
