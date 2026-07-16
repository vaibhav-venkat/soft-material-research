from __future__ import annotations

import math

import numpy as np

from hexatic.big_lx.lattice import (
    generate_unwrapped_lattice,
    outward_normal_quaternions,
)
from hexatic.constants import cylinder

from .cases import ComparisonCase, GeometryKind


def logical_to_stored(vectors: np.ndarray, case: ComparisonCase) -> np.ndarray:
    return np.asarray(vectors)[..., case.logical_to_stored_axes]


def stored_to_logical(vectors: np.ndarray, case: ComparisonCase) -> np.ndarray:
    values = np.asarray(vectors)
    if not case.is_cylinder:
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


def _legacy_prism_lattice(
    case: ComparisonCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _balanced_grid_shape(spans: tuple[float, ...], n_particles: int) -> tuple[int, ...]:
    dimensions = len(spans)
    scale = (n_particles / math.prod(spans)) ** (1.0 / dimensions)
    targets = tuple(max(1, int(round(span * scale))) for span in spans)
    ranges = [range(max(1, target - 6), target + 7) for target in targets]
    candidates: list[tuple[float, tuple[int, ...]]] = []
    if dimensions == 2:
        shapes = ((nx, ny) for nx in ranges[0] for ny in ranges[1])
    else:
        shapes = (
            (nx, ny, nz)
            for nx in ranges[0]
            for ny in ranges[1]
            for nz in ranges[2]
        )
    for shape in shapes:
        size = math.prod(shape)
        if size < n_particles or (size - n_particles) % 2:
            continue
        spacings = np.asarray(spans, dtype=np.float64) / np.asarray(shape)
        anisotropy = float(np.var(np.log(spacings)))
        score = (size - n_particles) / n_particles + anisotropy
        candidates.append((score, shape))
    if not candidates:
        raise ValueError(f"could not construct a symmetric grid for N={n_particles}")
    return min(candidates)[1]


def _symmetric_keep_mask(shape: tuple[int, ...], n_particles: int) -> np.ndarray:
    size = math.prod(shape)
    remove_pairs = (size - n_particles) // 2
    indices = list(np.ndindex(shape))

    def partner(index: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(width - 1 - value for width, value in zip(shape, index))

    representatives = [index for index in indices if index < partner(index)]
    if remove_pairs > len(representatives):
        raise ValueError("requested particle count is too small for the selected grid")
    removed: set[tuple[int, ...]] = set()
    if remove_pairs:
        selected = np.linspace(
            0,
            len(representatives) - 1,
            remove_pairs,
            dtype=np.int64,
        )
        for selected_index in selected:
            index = representatives[int(selected_index)]
            removed.add(index)
            removed.add(partner(index))
    return np.asarray([index not in removed for index in indices], dtype=np.bool_)


def _cell_centered_points(
    spans: tuple[float, ...],
    n_particles: int,
) -> np.ndarray:
    shape = _balanced_grid_shape(spans, n_particles)
    axes = [
        (np.arange(width, dtype=np.float64) + 0.5) * span / width - 0.5 * span
        for span, width in zip(spans, shape)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(spans)
    )
    return points[_symmetric_keep_mask(shape, n_particles)]


def _triangular_2d_points(case: ComparisonCase) -> np.ndarray:
    cutoff = cylinder.ANALYSIS.wall_cutoff
    seam_gap = cutoff * 1.001
    candidates: list[tuple[float, int, int, float]] = []
    for nx in range(4, int(case.lx / cutoff) + 1, 2):
        for ny in range(3, 101, 2):
            size = nx * ny
            if size < case.n_particles - 1 or (size - (case.n_particles - 1)) % 2:
                continue
            dx = case.lx / nx
            dy = (case.transverse_span - seam_gap) / (ny - 1)
            diagonal = math.hypot(0.5 * dx, dy)
            if min(dx, diagonal, seam_gap) <= cutoff:
                continue
            score = (size - (case.n_particles - 1)) / case.n_particles + abs(
                dy / dx - math.sqrt(3.0) / 2.0
            )
            candidates.append((score, nx, ny, dy))
    if not candidates:
        raise ValueError("could not construct a non-overlapping 2D triangular grid")
    _, nx, ny, dy = min(candidates)
    dx = case.lx / nx
    half_x = nx // 2
    half_y = ny // 2
    points: list[tuple[float, float]] = []
    for iy in range(ny):
        j = iy - half_y
        offset = 0.5 * (j & 1)
        for ix in range(nx):
            i = ix - half_x + 0.5
            x = ((i + offset) * dx + 0.5 * case.lx) % case.lx - 0.5 * case.lx
            points.append((x, j * dy))
    values = np.asarray(points, dtype=np.float64)

    def key(point: np.ndarray) -> tuple[float, float]:
        x = (float(point[0]) + 0.5 * case.lx) % case.lx - 0.5 * case.lx
        return round(x, 10), round(float(point[1]), 10)

    lookup = {key(point): index for index, point in enumerate(values)}
    partners = np.empty(len(values), dtype=np.int64)
    for index, point in enumerate(values):
        partner_index = lookup.get(key(-point))
        if partner_index is None:
            raise AssertionError("2D lattice lost an inversion partner")
        partners[index] = partner_index
    representatives = [index for index in range(len(values)) if index < partners[index]]
    remove_pairs = (len(values) - (case.n_particles - 1)) // 2
    forced = [
        index
        for index in representatives
        if min(
            np.linalg.norm(values[index]),
            np.linalg.norm(values[partners[index]]),
        )
        <= cutoff
    ]
    if len(forced) > remove_pairs:
        raise ValueError("not enough 2D vacancies to insert the center particle")
    remaining = [index for index in representatives if index not in set(forced)]
    extra_count = remove_pairs - len(forced)
    extra = (
        [
            remaining[int(index)]
            for index in np.linspace(
                0, len(remaining) - 1, extra_count, dtype=np.int64
            )
        ]
        if extra_count
        else []
    )
    removed = set(forced + extra)
    removed.update(int(partners[index]) for index in tuple(removed))
    result = values[
        np.asarray([index not in removed for index in range(len(values))], dtype=np.bool_)
    ]
    result = np.concatenate((result, np.zeros((1, 2), dtype=np.float64)), axis=0)
    result[:, 0] -= np.mean(result[:, 0])
    if np.max(np.abs(result[:, 0])) >= 0.5 * case.lx:
        raise AssertionError("centering the 2D lattice crossed the periodic seam")
    return result


def _nearest_prism_directions(positions: np.ndarray) -> np.ndarray:
    directions = np.zeros_like(positions)
    use_y = np.abs(positions[:, 1]) >= np.abs(positions[:, 2])
    directions[use_y, 1] = np.where(positions[use_y, 1] >= 0.0, 1.0, -1.0)
    directions[~use_y, 2] = np.where(positions[~use_y, 2] >= 0.0, 1.0, -1.0)
    return directions


def _paired_wall_sign(coordinate: np.ndarray, tie_breakers: np.ndarray) -> np.ndarray:
    sign = np.where(coordinate >= 0.0, 1.0, -1.0)
    tied = np.isclose(coordinate, 0.0, atol=1e-14)
    for values in np.asarray(tie_breakers).T:
        unresolved = tied & ~np.isclose(values, 0.0, atol=1e-14)
        sign[unresolved] = np.where(values[unresolved] >= 0.0, 1.0, -1.0)
        tied[unresolved] = False
    sign[tied] = 1.0
    return sign


def generate_planar_lattice(
    case: ComparisonCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case.is_cylinder:
        raise ValueError("planar lattice requested for a cylindrical case")
    if case.kind == GeometryKind.PRISM_VOLUME:
        return _legacy_prism_lattice(case)
    if case.is_2d:
        points_2d = _triangular_2d_points(case)
        positions = np.column_stack((points_2d, np.zeros(case.n_particles)))
        directions = np.zeros_like(positions)
        directions[:, 1] = _paired_wall_sign(positions[:, 1], positions[:, [0]])
    else:
        positions = _cell_centered_points(
            (case.lx, case.initial_span_y, case.initial_span_z),
            case.n_particles,
        )
        if case.is_prism:
            directions = _nearest_prism_directions(positions)
        else:
            directions = np.zeros_like(positions)
            directions[:, 2] = _paired_wall_sign(
                positions[:, 2], positions[:, [0, 1]]
            )
    quaternions = quaternions_from_x_directions(directions)
    return positions, quaternions, directions


def generate_prism_lattice(case: ComparisonCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not case.is_prism:
        raise ValueError("prism lattice requested for a non-prism case")
    return generate_planar_lattice(case)


def generate_cylinder_film(
    case: ComparisonCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not case.is_cylinder:
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
