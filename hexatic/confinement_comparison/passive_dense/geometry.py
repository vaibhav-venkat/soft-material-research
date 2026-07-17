from __future__ import annotations

import numpy as np

from hexatic.big_lx.lattice import (
    generate_unwrapped_lattice,
    outward_normal_quaternions,
)

from .cases import PassiveDenseCase


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


def generate_dense_2d(
    case: PassiveDenseCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not case.is_dense_2d:
        raise ValueError("dense 2D geometry requested for a cylinder case")
    a = case.lattice_spacing
    h = case.lattice_height
    points = np.empty((case.n_particles, 3), dtype=np.float64)
    particle = 0
    for ix in range(case.nx):
        x = (ix + 0.5) * h - 0.5 * case.lx
        y_offset = 0.5 * (ix % 2)
        for iy in range(case.ny):
            y = (iy - 0.5 * (case.ny - 1) + y_offset - 0.25) * a
            points[particle] = (x, y, 0.0)
            particle += 1
    if particle != case.n_particles:
        raise AssertionError(f"generated {particle} particles, expected {case.n_particles}")

    directions = np.zeros_like(points)
    directions[:, 1] = np.where(points[:, 1] >= 0.0, 1.0, -1.0)
    half_turn = np.sqrt(0.5)
    orientations = np.zeros((case.n_particles, 4), dtype=np.float64)
    orientations[:, 0] = half_turn
    orientations[:, 3] = directions[:, 1] * half_turn
    return points, orientations, directions


def generate_initial_arrays(
    case: PassiveDenseCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case.is_passive_cylinder:
        return generate_passive_cylinder(case)
    return generate_dense_2d(case)
