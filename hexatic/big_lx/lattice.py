from __future__ import annotations

import math

import numpy as np

from .cases import BigLxCase


def generate_unwrapped_lattice(case: BigLxCase) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((case.lattice_site_count, 3), dtype=np.float64)
    theta = np.empty(case.lattice_site_count, dtype=np.float64)

    circumference_vector = np.asarray(case.circumference_lattice_vector, dtype=int)
    axial_vector = np.asarray(case.axial_lattice_vector, dtype=int)
    supercell = np.column_stack((circumference_vector, axial_vector))
    inverse_supercell = np.linalg.inv(supercell)

    corners = np.asarray(
        ((0, 0), circumference_vector, axial_vector, circumference_vector + axial_vector)
    )
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)

    particle_idx = 0
    tolerance = 1e-12
    for j in range(int(lower[0]), int(upper[0]) + 1):
        for i in range(int(lower[1]), int(upper[1]) + 1):
            circumference_fraction, axial_fraction = inverse_supercell @ (j, i)
            if not (
                -tolerance <= circumference_fraction < 1.0 - tolerance
                and -tolerance <= axial_fraction < 1.0 - tolerance
            ):
                continue
            angle = circumference_fraction * case.circumference / case.radius
            positions[particle_idx] = (
                (axial_fraction - 0.5) * case.lx,
                case.radius * math.sin(angle),
                case.radius * math.cos(angle),
            )
            theta[particle_idx] = angle
            particle_idx += 1

    if particle_idx != case.lattice_site_count:
        raise AssertionError(
            f"enumerated {particle_idx} lattice sites, "
            f"expected {case.lattice_site_count}"
        )
    if case.n_particles < case.lattice_site_count:
        rng = np.random.default_rng(case.seed)
        selected = np.sort(
            rng.choice(
                case.lattice_site_count,
                size=case.n_particles,
                replace=False,
            )
        )
        positions = positions[selected]
        theta = theta[selected]
    return positions, theta


def outward_normal_quaternions(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    half_turn = math.sqrt(0.5)
    quaternions = np.column_stack(
        (
            np.full(theta.shape, half_turn, dtype=np.float64),
            np.zeros(theta.shape, dtype=np.float64),
            -np.cos(theta) * half_turn,
            np.sin(theta) * half_turn,
        )
    )
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]
    return quaternions
