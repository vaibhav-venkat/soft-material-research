from __future__ import annotations

import numpy as np

from hexatic.big_lx.lattice import (
    generate_unwrapped_lattice,
    outward_normal_quaternions,
)

from .cases import CaseKind, PassiveDenseCase


def generate_passive_cylinder(
    case: PassiveDenseCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not case.is_passive_cylinder:
        raise ValueError("passive-cylinder geometry requested for a 2D case")
    positions, theta = generate_unwrapped_lattice(case.base)
    orientations = outward_normal_quaternions(theta)
    directions = np.column_stack(
        (np.zeros(case.n_particles), np.sin(theta), np.cos(theta))
    )
    return positions, orientations, directions


def _full_dense_2d_positions(case: PassiveDenseCase) -> np.ndarray:
    if not case.is_dense_2d:
        raise ValueError("dense 2D geometry requested for a cylinder case")
    a = case.lattice_spacing
    h = case.lattice_height
    full_count = case.nx * case.ny
    points = np.empty((full_count, 3), dtype=np.float64)
    particle = 0
    for ix in range(case.nx):
        x = (ix + 0.5) * h - 0.5 * case.lx
        y_offset = 0.5 * (ix % 2)
        for iy in range(case.ny):
            y = (iy - 0.5 * (case.ny - 1) + y_offset - 0.25) * a
            points[particle] = (x, y, 0.0)
            particle += 1
    if particle != full_count:
        raise AssertionError(f"generated {particle} particles, expected {full_count}")
    return points


def dense_2d_vacancy_indices(
    case: PassiveDenseCase,
    points: np.ndarray | None = None,
) -> np.ndarray:
    values = _full_dense_2d_positions(case) if points is None else np.asarray(points)
    if case.kind == CaseKind.DENSE_2D:
        return np.empty(0, dtype=np.int64)
    if case.kind == CaseKind.DENSE_2D_CENTER_VACANCY:
        order = np.lexsort(
            (
                np.arange(len(values)),
                np.abs(values[:, 0]),
                np.abs(values[:, 1]),
                np.sum(values[:, :2] ** 2, axis=1),
            )
        )
        return np.asarray((order[0],), dtype=np.int64)
    if case.kind == CaseKind.DENSE_2D_WALL_VACANCY:
        order = np.lexsort(
            (
                np.arange(len(values)),
                np.abs(values[:, 0]),
                -values[:, 1],
            )
        )
        return np.asarray((order[0],), dtype=np.int64)
    if case.kind == CaseKind.DENSE_2D_OPPOSITE_WALL_VACANCIES:
        lookup = {
            tuple(np.round(point[:2], 12)): int(index)
            for index, point in enumerate(values)
        }
        pairs = [
            (int(index), lookup[tuple(np.round(-values[index, :2], 12))])
            for index in range(len(values))
            if values[index, 1] < 0.0
            and tuple(np.round(-values[index, :2], 12)) in lookup
        ]
        if not pairs:
            raise AssertionError("could not find inversion-paired vacancy sites")
        lower, upper = max(
            pairs,
            key=lambda pair: (
                min(
                    abs(float(values[pair[0], 1])),
                    abs(float(values[pair[1], 1])),
                ),
                -max(
                    abs(float(values[pair[0], 0])),
                    abs(float(values[pair[1], 0])),
                ),
            ),
        )
        return np.asarray((lower, upper), dtype=np.int64)
    raise ValueError(f"unsupported dense 2D vacancy case: {case.kind}")


def dense_2d_vacancy_sites(case: PassiveDenseCase) -> np.ndarray:
    points = _full_dense_2d_positions(case)
    return points[dense_2d_vacancy_indices(case, points)]


def generate_dense_2d(
    case: PassiveDenseCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = _full_dense_2d_positions(case)
    vacancy_indices = dense_2d_vacancy_indices(case, points)
    if len(vacancy_indices):
        keep = np.ones(len(points), dtype=np.bool_)
        keep[vacancy_indices] = False
        points = points[keep]
    if len(points) != case.n_particles:
        raise AssertionError(
            f"generated {len(points)} particles, expected {case.n_particles}"
        )

    directions = np.zeros_like(points)
    directions[:, 1] = np.where(points[:, 1] >= 0.0, 1.0, -1.0)
    half_turn = np.sqrt(0.5)
    orientations = np.zeros((len(points), 4), dtype=np.float64)
    orientations[:, 0] = half_turn
    orientations[:, 3] = directions[:, 1] * half_turn
    return points, orientations, directions


def generate_initial_arrays(
    case: PassiveDenseCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case.is_passive_cylinder:
        return generate_passive_cylinder(case)
    return generate_dense_2d(case)
