from __future__ import annotations

import argparse
import json

import numpy as np

from hexatic.constants import cylinder

from .backend import select_backend
from .cases import BigLxCase, select_cases
from .lattice import generate_unwrapped_lattice, outward_normal_quaternions
from .spatial import PeriodicXTree


def validate_case(case: BigLxCase) -> dict[str, object]:
    primitive_p, primitive_q = case.primitive_axial_lattice_vector
    p, q = case.axial_lattice_vector
    if (p, q) != (
        case.lx_multiplier * primitive_p,
        case.lx_multiplier * primitive_q,
    ):
        raise AssertionError("axial lattice vector did not scale exactly")
    if not np.isclose(case.lx, case.lx_multiplier * case.base_lx):
        raise AssertionError("Lx did not scale exactly")
    if case.n_particles != case.lx_multiplier * case.base_n_particles:
        raise AssertionError("particle count did not scale exactly")

    positions, theta = generate_unwrapped_lattice(case)
    if positions.shape != (case.n_particles, 3):
        raise AssertionError("generated lattice shape is incorrect")
    tree = PeriodicXTree.build(positions, case.lx)
    _, bonds = tree.nearest_bonds(positions, cylinder.ANALYSIS.neighbors)
    backend = select_backend("numpy")
    psi_real, psi_imag = backend.hexatic(bonds, positions)
    psi_abs = np.sqrt(psi_real * psi_real + psi_imag * psi_imag)

    counts = tree.tree.query_ball_point(
        positions,
        cylinder.ANALYSIS.neighbor_count_radius,
        workers=-1,
        return_length=True,
    ).astype(np.int64) - 1
    quaternions = outward_normal_quaternions(theta)
    directions = backend.directions(quaternions)
    normals = np.zeros_like(positions)
    normals[:, 1:3] = positions[:, 1:3] / np.linalg.norm(
        positions[:, 1:3], axis=1, keepdims=True
    )
    alignment = np.einsum("ij,ij->i", directions, normals)

    if not np.all(counts == cylinder.ANALYSIS.neighbors):
        unique, totals = np.unique(counts, return_counts=True)
        raise AssertionError(
            f"initial neighbor counts are not all six: {dict(zip(unique.tolist(), totals.tolist()))}"
        )
    if float(psi_abs.min()) < 0.99:
        raise AssertionError(f"initial |psi6| is too small: {psi_abs.min()}")
    if float(alignment.min()) < 0.99999:
        raise AssertionError(f"orientation is not outward-normal: {alignment.min()}")

    return {
        "case_id": case.case_id,
        "lx_multiplier": case.lx_multiplier,
        "lx": case.lx,
        "n_particles": case.n_particles,
        "volume_density": case.volume_density,
        "surface_density": case.surface_density,
        "neighbor_count_min": int(counts.min()),
        "neighbor_count_max": int(counts.max()),
        "psi_abs_min": float(psi_abs.min()),
        "psi_abs_mean": float(psi_abs.mean()),
        "normal_alignment_min": float(alignment.min()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exact big-Lx geometries.")
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
        print(
            f"{result['case_id']}: Lx={result['lx']:.12g} "
            f"N={result['n_particles']} rho={result['volume_density']:.12g} "
            f"neighbors={result['neighbor_count_min']} "
            f"psi6_min={result['psi_abs_min']:.9g}"
        )


if __name__ == "__main__":
    main()
