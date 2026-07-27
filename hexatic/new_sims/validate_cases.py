from __future__ import annotations

from typing import cast

import numpy as np

from hexatic.constants import cylinder

from .cases import CaseKind, all_cases
from .geometry import generate_initial_arrays, quaternion_directions


def _minimum_distance(
    positions: np.ndarray,
    box: tuple[float, float, float],
    periodic: tuple[bool, bool, bool],
) -> float:
    from scipy.spatial import KDTree

    lengths = np.asarray(
        [length if use else 0.0 for length, use in zip(box, periodic)],
        dtype=np.float64,
    )
    transformed = np.asarray(positions, dtype=np.float64).copy()
    for axis, length in enumerate(lengths):
        if length > 0:
            transformed[:, axis] = np.mod(
                transformed[:, axis] + 0.5 * length, length
            )
    distances, _ = KDTree(transformed, boxsize=lengths).query(
        transformed, k=2, workers=-1
    )
    return float(np.min(distances[:, 1]))


def validate() -> None:
    for case in all_cases():
        positions, orientations, active_ids, metadata = generate_initial_arrays(case)
        if positions.shape != (case.n_particles, 3):
            raise AssertionError(f"{case.case_id}: bad position shape")
        if orientations.shape != (case.n_particles, 4):
            raise AssertionError(f"{case.case_id}: bad orientation shape")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(orientations)):
            raise AssertionError(f"{case.case_id}: non-finite initial data")
        if not np.allclose(np.linalg.norm(orientations, axis=1), 1.0, atol=1e-6):
            raise AssertionError(f"{case.case_id}: non-unit quaternions")
        if len(active_ids) != case.active_count or len(np.unique(active_ids)) != len(
            active_ids
        ):
            raise AssertionError(f"{case.case_id}: invalid active IDs")
        if not case.is_tracer:
            mean = quaternion_directions(orientations).mean(axis=0)
            tolerance = 0.06 if case.is_2d else 0.04
            if np.max(np.abs(mean)) > tolerance:
                raise AssertionError(
                    f"{case.case_id}: biased initial polarization {mean}"
                )

        periodic = (
            "x" in case.periodic_axes,
            "y" in case.periodic_axes,
            "z" in case.periodic_axes,
        )
        minimum = _minimum_distance(positions, case.stored_box, periodic)
        if case.has_pair_interaction and minimum <= cylinder.ANALYSIS.wall_cutoff:
            raise AssertionError(
                f"{case.case_id}: initial separation {minimum} is inside LJ cutoff"
            )
        if minimum <= 0.0:
            raise AssertionError(f"{case.case_id}: duplicate initial positions")
        if case.is_cylinder:
            radial = np.linalg.norm(positions[:, 1:3], axis=1)
            if np.max(radial) >= case.wall_radius - cylinder.ANALYSIS.wall_cutoff:
                raise AssertionError(f"{case.case_id}: particle begins inside wall force")
            if case.is_tracer and not np.allclose(radial, case.radius, atol=1e-8):
                raise AssertionError(f"{case.case_id}: twisted film radius changed")
        if case.has_x_walls:
            clearance = 0.5 * case.lx - np.max(np.abs(positions[:, 0]))
            if clearance <= cylinder.ANALYSIS.wall_cutoff:
                raise AssertionError(f"{case.case_id}: insufficient x-wall clearance")
        if case.kind == CaseKind.INVERSION_ACTIVE_PAIR_CYLINDER:
            inversion_error = cast(float, metadata["inversion_target_error"])
            if inversion_error > 1.1 * case.base.lattice_spacing:
                raise AssertionError(f"{case.case_id}: inversion partner is too far away")
        print(
            f"[new_sims.validate] case={case.case_id} N={case.n_particles} "
            f"min_distance={minimum:.8g} active={len(active_ids)}",
            flush=True,
        )

    plane = next(
        case for case in all_cases() if case.kind == CaseKind.PERIODIC_2D_PLANE
    )
    positions, _, _, _ = generate_initial_arrays(plane)
    # The exact flattened supercell should retain its triangular six-fold order.
    from scipy.spatial import KDTree

    shifted = positions[:, :2] + np.asarray((0.5 * plane.lx, 0.5 * plane.circumference))
    distances, hits = KDTree(
        shifted, boxsize=(plane.lx, plane.circumference)
    ).query(shifted, k=7, workers=-1)
    del distances
    bonds = positions[hits[:, 1:], :2] - positions[:, None, :2]
    bonds[:, :, 0] -= plane.lx * np.rint(bonds[:, :, 0] / plane.lx)
    bonds[:, :, 1] -= plane.circumference * np.rint(
        bonds[:, :, 1] / plane.circumference
    )
    angles = np.arctan2(bonds[:, :, 1], bonds[:, :, 0])
    psi = np.mean(np.exp(6.0j * angles), axis=1)
    if float(np.min(np.abs(psi))) < 1.0 - 1e-6:
        raise AssertionError("periodic plane lost exact initial hexatic order")


def main() -> None:
    validate()


if __name__ == "__main__":
    main()
