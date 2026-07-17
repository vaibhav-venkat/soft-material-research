from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import KDTree

from hexatic.big_lx.lattice import generate_unwrapped_lattice
from hexatic.constants import cylinder

from .cases import add_case_selection_arguments, select_cases
from .geometry import (
    dense_2d_vacancy_sites,
    generate_dense_2d,
    generate_passive_cylinder,
)
from .simulate_case import make_simulation, write_initial_state


def _validate_dynamics(case) -> tuple[str, bool]:
    with tempfile.TemporaryDirectory(prefix="passive_dense_validate_") as directory:
        root = Path(directory)
        initial = root / "initial.gsd"
        trajectory = root / "trajectory.gsd"
        write_initial_state(case, initial)
        objects = make_simulation(
            case,
            initial_gsd=initial,
            trajectory_gsd=trajectory,
            write_period=case.trajectory_write_period,
            device_name="cpu",
            gpu_id=None,
        )
        method_name = type(objects.method).__name__
        has_active = objects.active is not None
        if case.is_passive_cylinder:
            if method_name != "Brownian" or has_active:
                raise AssertionError("passive cylinder dynamics are not purely Brownian")
        elif method_name != "OverdampedViscous" or not has_active:
            raise AssertionError("dense 2D dynamics are not active overdamped viscous")
        return method_name, has_active


def _validate_cylinder(case) -> dict[str, object]:
    positions, _, _ = generate_passive_cylinder(case)
    source_positions, _ = generate_unwrapped_lattice(case.base)
    if not np.array_equal(positions, source_positions):
        raise AssertionError("passive cylinder does not exactly reuse the twisted lattice")
    radius = np.linalg.norm(positions[:, 1:3], axis=1)
    if not np.allclose(radius, case.base.radius, atol=1e-12, rtol=0.0):
        raise AssertionError("passive cylinder initial sites are not on the source radius")
    method_name, has_active = _validate_dynamics(case)
    return {
        "case_id": case.case_id,
        "n_particles": len(positions),
        "source_coordinates_exact": True,
        "radius_residual_max": float(np.max(np.abs(radius - case.base.radius))),
        "initial_wall_distance": float(case.base.wall_radius - np.max(radius)),
        "method": method_name,
        "has_active_force": has_active,
    }


def _validate_dense_2d(case) -> dict[str, object]:
    positions, _, directions = generate_dense_2d(case)
    vacancy_sites = dense_2d_vacancy_sites(case)
    if not np.all(positions[:, 2] == 0.0):
        raise AssertionError("dense 2D lattice has nonzero z coordinates")
    reconstructed = np.concatenate((positions, vacancy_sites), axis=0)
    if not np.allclose(np.mean(reconstructed, axis=0), 0.0, atol=1e-12):
        raise AssertionError("dense 2D parent lattice is not centered")

    transformed = positions.copy()
    transformed[:, 0] = np.mod(transformed[:, 0] + 0.5 * case.lx, case.lx)
    tree = KDTree(transformed, boxsize=np.asarray((case.lx, 0.0, 0.0)))
    distances, hits = tree.query(transformed, k=7, workers=-1)
    nearest = distances[:, 1]
    if not np.allclose(nearest, case.lattice_spacing, atol=1e-12, rtol=1e-12):
        raise AssertionError("dense 2D nearest-neighbor spacing is not exactly D")

    sixfold = np.all(
        np.isclose(
            distances[:, 1:],
            case.lattice_spacing,
            atol=1e-12,
            rtol=1e-12,
        ),
        axis=1,
    )
    if not np.any(sixfold):
        raise AssertionError("dense 2D lattice has no six-neighbor interior sites")
    source = positions[np.asarray(hits[sixfold, 1:], dtype=np.int64)]
    query = positions[sixfold, np.newaxis, :]
    bonds = source - query
    bonds[..., 0] -= case.lx * np.rint(bonds[..., 0] / case.lx)
    angles = np.arctan2(bonds[..., 1], bonds[..., 0])
    psi6 = np.mean(np.exp(6.0j * angles), axis=1)
    if not np.allclose(np.abs(psi6), 1.0, atol=1e-12, rtol=0.0):
        raise AssertionError("dense 2D interior lost perfect hexatic order")

    first_x = float(np.min(positions[:, 0]))
    last_x = float(np.max(positions[:, 0]))
    first = positions[np.isclose(positions[:, 0], first_x, atol=1e-14)]
    last = positions[np.isclose(positions[:, 0], last_x, atol=1e-14)]
    seam_dx = last[:, np.newaxis, 0] - first[np.newaxis, :, 0]
    seam_dx -= case.lx * np.rint(seam_dx / case.lx)
    seam_dy = last[:, np.newaxis, 1] - first[np.newaxis, :, 1]
    seam_distances = np.hypot(seam_dx, seam_dy)
    seam_bonds = seam_distances[
        np.isclose(
            seam_distances,
            case.lattice_spacing,
            atol=1e-12,
            rtol=1e-12,
        )
    ]
    expected_seam_bonds = 2 * case.ny - 1
    if len(seam_bonds) != expected_seam_bonds:
        raise AssertionError(
            f"expected {expected_seam_bonds} exact seam bonds, found {len(seam_bonds)}"
        )

    wall_distance = float(np.min(0.5 * case.ly - np.abs(positions[:, 1])))
    if wall_distance <= 0.0:
        raise AssertionError("a dense 2D particle starts on or beyond a wall")
    if not math.isclose(
        wall_distance,
        0.5 * cylinder.ANALYSIS.particle_diameter,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise AssertionError("dense 2D wall clearance is not D/2")
    expected_directions = np.where(positions[:, 1] >= 0.0, 1.0, -1.0)
    if not np.array_equal(directions[:, 1], expected_directions):
        raise AssertionError("dense 2D directions do not point to the nearest y wall")
    method_name, has_active = _validate_dynamics(case)
    if len(vacancy_sites) != case.vacancy_count:
        raise AssertionError("dense 2D vacancy count is incorrect")
    if case.kind.value.endswith("center_vacancy"):
        minimum_radius = float(np.min(np.linalg.norm(positions[:, :2], axis=1)))
        vacancy_radius = float(np.linalg.norm(vacancy_sites[0, :2]))
        if vacancy_radius > minimum_radius + 1e-12:
            raise AssertionError("center vacancy did not remove a closest-to-origin site")
    elif case.kind.value.endswith("wall_vacancy"):
        if not math.isclose(
            float(vacancy_sites[0, 1]),
            0.5 * case.ly - 0.5 * case.lattice_spacing,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise AssertionError("wall vacancy is not in the outer +y layer")
    elif case.kind.value.endswith("opposite_wall_vacancies"):
        if not np.allclose(
            vacancy_sites[0],
            -vacancy_sites[1],
            atol=1e-12,
            rtol=0.0,
        ):
            raise AssertionError("opposite-wall vacancies are not inversion symmetric")
    return {
        "case_id": case.case_id,
        "n_particles": len(positions),
        "nx": case.nx,
        "ny": case.ny,
        "stored_box": case.stored_box,
        "nearest_neighbor_min": float(np.min(nearest)),
        "nearest_neighbor_max": float(np.max(nearest)),
        "periodic_seam_bonds": len(seam_bonds),
        "interior_psi6_abs_min": float(np.min(np.abs(psi6))),
        "initial_wall_distance": wall_distance,
        "method": method_name,
        "has_active_force": has_active,
        "vacancy_sites": vacancy_sites.tolist(),
        "run_steps": case.run_steps,
        "expected_frames": case.run_steps // case.trajectory_write_period,
    }


def validate_case(case) -> dict[str, object]:
    if case.is_passive_cylinder:
        return _validate_cylinder(case)
    return _validate_dense_2d(case)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate passive/dense initial states.")
    add_case_selection_arguments(parser)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = [validate_case(case) for case in select_cases(args.all, args.case)]
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
