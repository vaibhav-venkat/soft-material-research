from __future__ import annotations

import argparse
import json

import numpy as np

from hexatic.analysis.cylinder import (
    compute_hexatic_order_frame_on_cylinder,
    compute_neighbor_counts_frame_on_cylinder,
)
from hexatic.constants import cylinder

from .cases import UnwrappedCase, all_cases, get_case
from .simulate_case import generate_unwrapped_lattice, outward_normal_quaternions


def validate_case(case: UnwrappedCase) -> dict[str, object]:
    positions, theta = generate_unwrapped_lattice(case)
    quaternions = outward_normal_quaternions(theta)
    psi, _ = compute_hexatic_order_frame_on_cylinder(
        positions,
        cylinder_radius=case.radius,
        shell_delta=cylinder.ANALYSIS.shell_delta,
        box_length_x=case.twisted_lx,
    )
    counts = compute_neighbor_counts_frame_on_cylinder(
        positions,
        cylinder_radius=case.radius,
        shell_delta=cylinder.ANALYSIS.shell_delta,
        neighbor_radius=cylinder.ANALYSIS.neighbor_count_radius,
        box_length_x=case.twisted_lx,
    )

    surface_mask = np.abs(psi) > 0.0
    surface_counts = counts[surface_mask]
    unique_counts, count_totals = np.unique(surface_counts, return_counts=True)
    active_direction = _active_direction_from_quaternion(quaternions)
    normals = np.zeros_like(positions)
    normals[:, 1:3] = positions[:, 1:3] / np.linalg.norm(
        positions[:, 1:3],
        axis=1,
    )[:, np.newaxis]

    return {
        "case_id": case.case_id,
        "n_theta": case.n_theta,
        "n_x": case.n_x,
        "n_particles": case.n_particles,
        "radius": case.radius,
        "circumference": case.circumference,
        "a": case.a,
        "h": case.h,
        "lx_target": case.lx_target,
        "lx": case.twisted_lx,
        "neighbor_counts": dict(
            zip(unique_counts.astype(int).tolist(), count_totals.astype(int).tolist())
        ),
        "psi_abs_min": float(np.abs(psi[surface_mask]).min()),
        "psi_abs_mean": float(np.abs(psi[surface_mask]).mean()),
        "psi_abs_max": float(np.abs(psi[surface_mask]).max()),
        "normal_alignment_min": float(np.einsum("ij,ij->i", active_direction, normals).min()),
    }


def _active_direction_from_quaternion(orientation: np.ndarray) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=np.float64)
    norms = np.linalg.norm(orientation, axis=1)
    assert np.all(norms > 0.0)
    quat = orientation / norms[:, np.newaxis]
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        )
    )


def _selected_cases(args: argparse.Namespace) -> tuple[UnwrappedCase, ...]:
    if args.all:
        return all_cases()
    if args.case:
        return tuple(get_case(case_id) for case_id in args.case)
    raise SystemExit("Select --all or one or more --case values.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = [validate_case(case) for case in _selected_cases(args)]
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        print(
            f"{result['case_id']}: "
            f"Ntheta={result['n_theta']} Nx={result['n_x']} N={result['n_particles']} "
            f"counts={result['neighbor_counts']} "
            f"|psi6|=({result['psi_abs_min']:.12g}, "
            f"{result['psi_abs_mean']:.12g}, {result['psi_abs_max']:.12g}) "
            f"normal_dot_min={result['normal_alignment_min']:.12g}"
        )


if __name__ == "__main__":
    main()
