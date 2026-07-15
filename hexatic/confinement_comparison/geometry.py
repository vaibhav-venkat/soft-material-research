from __future__ import annotations

import math

import numpy as np

from hexatic.big_lx.lattice import (
    generate_unwrapped_lattice,
    outward_normal_quaternions,
)

from .cases import ComparisonCase, GeometryKind


def logical_to_stored(vectors: np.ndarray, case: ComparisonCase) -> np.ndarray:
    return np.asarray(vectors)[..., case.logical_to_stored_axes]


def stored_to_logical(vectors: np.ndarray, case: ComparisonCase) -> np.ndarray:
    values = np.asarray(vectors)
    if case.kind == GeometryKind.PRISM_VOLUME:
        return values.copy()
    return values[..., (2, 0, 1)]


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def rotate_logical_quaternions_to_stored(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    rotation = np.broadcast_to(
        np.asarray((0.5, -0.5, -0.5, -0.5), dtype=np.float64),
        values.shape,
    )
    result = quaternion_multiply(rotation, values)
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def quaternions_from_x_directions(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    x_component = directions[:, 0]
    result = np.zeros((len(directions), 4), dtype=np.float64)
    regular = x_component > -1.0 + 1e-12
    result[regular, 0] = np.sqrt(0.5 * (1.0 + x_component[regular]))
    denominator = 2.0 * result[regular, 0]
    result[regular, 2] = -directions[regular, 2] / denominator
    result[regular, 3] = directions[regular, 1] / denominator
    # A 180-degree rotation around +y maps +x to -x.
    result[~regular, 2] = 1.0
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def _inversion_partner(index: tuple[int, int, int]) -> tuple[int, int, int]:
    ix, iy, iz = index
    return 70 - ix, 9 - iy, 9 - iz


def _vacancy_indices(points: np.ndarray) -> set[tuple[int, int, int]]:
    indices = [(ix, iy, iz) for ix in range(71) for iy in range(10) for iz in range(10)]
    representatives = [index for index in indices if index < _inversion_partner(index)]
    normalized = points.reshape(71, 10, 10, 3)
    endpoints = {
        index: np.stack(
            (normalized[index], normalized[_inversion_partner(index)]), axis=0
        )
        for index in representatives
    }
    selected: list[tuple[int, int, int]] = []
    target = np.asarray((0.173, -0.419, 0.731))
    first = min(
        representatives,
        key=lambda index: float(np.linalg.norm(endpoints[index][0] / np.ptp(points, axis=0) - target)),
    )
    selected.append(first)
    while len(selected) < 9:
        selected_points = np.concatenate([endpoints[index] for index in selected])

        def separation(index: tuple[int, int, int]) -> tuple[float, tuple[int, int, int]]:
            delta = endpoints[index][:, None, :] - selected_points[None, :, :]
            distance = float(np.min(np.linalg.norm(delta, axis=2)))
            ix, iy, iz = index
            return distance, (-ix, -iy, -iz)

        remaining = (index for index in representatives if index not in selected)
        selected.append(max(remaining, key=separation))
    removed = set(selected)
    removed.update(_inversion_partner(index) for index in selected)
    return removed


def generate_prism_lattice(case: ComparisonCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case.kind != GeometryKind.PRISM_VOLUME:
        raise ValueError("prism lattice requested for a cylindrical case")
    x = (np.arange(71, dtype=np.float64) + 0.5) * case.lx / 71.0 - 0.5 * case.lx
    y = (
        (np.arange(10, dtype=np.float64) + 0.5) * case.prism_side / 10.0
        - 0.5 * case.prism_side
    )
    z = y.copy()
    points = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)
    removed = _vacancy_indices(points)
    keep = np.asarray(
        [
            (ix, iy, iz) not in removed
            for ix in range(71)
            for iy in range(10)
            for iz in range(10)
        ],
        dtype=np.bool_,
    )
    positions = points[keep]
    if len(positions) != case.n_particles:
        raise AssertionError(f"expected {case.n_particles} prism particles, got {len(positions)}")

    directions = np.zeros_like(positions)
    abs_y = np.abs(positions[:, 1])
    abs_z = np.abs(positions[:, 2])
    use_y = abs_y > abs_z
    ties = np.isclose(abs_y, abs_z, atol=1e-12)
    tie_indices = set(np.flatnonzero(ties).tolist())
    lookup = {
        tuple(np.round(position, 12)): index
        for index, position in enumerate(positions)
    }
    y_total = int(np.count_nonzero(use_y & ~ties))
    z_total = int(np.count_nonzero(~use_y & ~ties))
    while tie_indices:
        index = min(tie_indices)
        partner_key = tuple(np.round(-positions[index], 12))
        partner = lookup.get(partner_key)
        if partner is None or partner not in tie_indices:
            raise AssertionError("prism lattice lost inversion-paired tie sites")
        choose_y = y_total <= z_total
        use_y[index] = choose_y
        use_y[partner] = choose_y
        if choose_y:
            y_total += 2
        else:
            z_total += 2
        tie_indices.remove(index)
        tie_indices.remove(partner)
    directions[use_y, 1] = np.where(positions[use_y, 1] >= 0.0, 1.0, -1.0)
    directions[~use_y, 2] = np.where(positions[~use_y, 2] >= 0.0, 1.0, -1.0)
    quaternions = quaternions_from_x_directions(directions)
    return positions, quaternions, directions


def generate_cylinder_film(
    case: ComparisonCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not case.is_constrained:
        raise ValueError("cylinder film requested for a prism case")
    logical_positions, theta = generate_unwrapped_lattice(case.base)
    logical_quaternions = outward_normal_quaternions(theta)
    if case.kind == GeometryKind.CYLINDER_RATTLE_TANGENT:
        rng = np.random.default_rng(case.seed)
        tangent_angle = rng.uniform(0.0, 2.0 * math.pi, case.n_particles)
        e_theta = np.column_stack(
            (
                np.zeros(case.n_particles),
                np.cos(theta),
                -np.sin(theta),
            )
        )
        e_axial = np.broadcast_to((1.0, 0.0, 0.0), e_theta.shape)
        logical_directions = (
            np.cos(tangent_angle)[:, None] * e_axial
            + np.sin(tangent_angle)[:, None] * e_theta
        )
        logical_quaternions = quaternions_from_x_directions(logical_directions)
    else:
        logical_directions = np.column_stack(
            (np.zeros(case.n_particles), np.sin(theta), np.cos(theta))
        )
    stored_positions = logical_to_stored(logical_positions, case)
    stored_quaternions = rotate_logical_quaternions_to_stored(logical_quaternions)
    stored_directions = logical_to_stored(logical_directions, case)
    return stored_positions, stored_quaternions, stored_directions
