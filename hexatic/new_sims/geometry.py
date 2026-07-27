from __future__ import annotations

import math

import numpy as np

from hexatic.big_lx.lattice import generate_unwrapped_lattice, outward_normal_quaternions
from hexatic.constants import cylinder

from .cases import CaseKind, NewSimCase


def random_uniform_quaternions(n: int, seed: int) -> np.ndarray:
    """Return seeded Haar-uniform rotations in HOOMD's (w, x, y, z) order."""
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random((3, n))
    q = np.column_stack(
        (
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3),
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
        )
    )
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def random_planar_quaternions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    q = np.zeros((n, 4), dtype=np.float64)
    q[:, 0] = np.cos(0.5 * angle)
    q[:, 3] = np.sin(0.5 * angle)
    return q


def quaternion_directions(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    w, x, y, z = (q[:, index] for index in range(4))
    return np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        )
    )


def _grid_points(spans: tuple[float, float, float], n_particles: int) -> np.ndarray:
    volume = math.prod(spans)
    scale = (n_particles / volume) ** (1.0 / 3.0)
    target = tuple(max(1, int(round(span * scale))) for span in spans)
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for nx in range(max(1, target[0] - 5), target[0] + 7):
        for ny in range(max(1, target[1] - 5), target[1] + 7):
            for nz in range(max(1, target[2] - 5), target[2] + 7):
                size = nx * ny * nz
                if size < n_particles:
                    continue
                spacing = np.asarray(spans) / (nx, ny, nz)
                score = (size - n_particles) / n_particles + float(
                    np.var(np.log(spacing))
                )
                candidates.append((score, (nx, ny, nz)))
    if not candidates:
        raise ValueError(f"could not create a bulk grid for N={n_particles}")
    shape = min(candidates)[1]
    axes = [
        (np.arange(width, dtype=np.float64) + 0.5) * span / width - 0.5 * span
        for width, span in zip(shape, spans)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    if len(points) == n_particles:
        return points
    keep = np.linspace(0, len(points) - 1, n_particles, dtype=np.int64)
    return points[keep]


def _cylinder_bulk_points(case: NewSimCase) -> np.ndarray:
    target_spacing = (math.pi * case.radius**2 * case.lx / case.n_particles) ** (
        1.0 / 3.0
    )
    spacing = target_spacing
    for _ in range(100):
        nx = max(2, int(math.ceil(case.lx / spacing)))
        nt = max(2, int(math.ceil(2.0 * case.radius / spacing)))
        x = (np.arange(nx) + 0.5) * case.lx / nx - 0.5 * case.lx
        transverse = (
            (np.arange(nt) + 0.5) * (2.0 * case.radius) / nt - case.radius
        )
        points = np.stack(
            np.meshgrid(x, transverse, transverse, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        radial = np.linalg.norm(points[:, 1:3], axis=1)
        points = points[radial <= case.radius - 0.02 * cylinder.PARTICLE_DIAMETER]
        if len(points) >= case.n_particles:
            keep = np.linspace(0, len(points) - 1, case.n_particles, dtype=np.int64)
            return points[keep]
        spacing *= 0.995
    raise ValueError("could not place the requested cylinder bulk particles")


def _unwrapped_plane(case: NewSimCase) -> np.ndarray:
    wrapped, _ = generate_unwrapped_lattice(case.base)
    theta = np.mod(np.arctan2(wrapped[:, 1], wrapped[:, 2]), 2.0 * np.pi)
    y = theta * case.radius - 0.5 * case.circumference
    x = wrapped[:, 0].copy()
    if case.kind == CaseKind.IDEAL_2D_X_WALLS:
        clearance = (
            cylinder.ANALYSIS.wall_cutoff
            + cylinder.SIMULATION.wall_clearance_epsilon
            * cylinder.PARTICLE_DIAMETER
        )
        available_lx = case.lx - 2.0 * clearance
        if available_lx <= 0.0:
            raise ValueError("x-walled 2D box has no wall-free axial span")
        x *= available_lx / case.lx
    return np.column_stack((x, y, np.zeros(case.n_particles)))


def _active_ids(positions: np.ndarray, case: NewSimCase) -> tuple[np.ndarray, dict[str, float]]:
    theta = np.mod(np.arctan2(positions[:, 1], positions[:, 2]), 2.0 * np.pi)
    first_distance = np.hypot(positions[:, 0], case.radius * np.minimum(theta, 2 * np.pi - theta))
    first = int(np.argmin(first_distance))
    if case.kind == CaseKind.SINGLE_ACTIVE_TRACER_CYLINDER:
        return np.asarray([first], dtype=np.int64), {
            "first_target_error": float(first_distance[first])
        }
    target_x = -positions[first, 0]
    target_theta = np.mod(theta[first] + np.pi, 2.0 * np.pi)
    angular_error = np.abs(theta - target_theta)
    angular_error = np.minimum(angular_error, 2.0 * np.pi - angular_error)
    second_distance = np.hypot(positions[:, 0] - target_x, case.radius * angular_error)
    second_distance[first] = np.inf
    second = int(np.argmin(second_distance))
    return np.asarray([first, second], dtype=np.int64), {
        "first_target_error": float(first_distance[first]),
        "inversion_target_error": float(second_distance[second]),
    }


def generate_initial_arrays(
    case: NewSimCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if case.is_tracer:
        positions, theta = generate_unwrapped_lattice(case.base)
        orientations = outward_normal_quaternions(theta)
        active_ids, errors = _active_ids(positions, case)
    elif case.is_2d:
        positions = _unwrapped_plane(case)
        orientations = random_planar_quaternions(case.n_particles, case.seed)
        active_ids = np.arange(case.n_particles, dtype=np.int64)
        errors = {}
    elif case.kind == CaseKind.IDEAL_ABP_CYLINDER:
        positions = _cylinder_bulk_points(case)
        orientations = random_uniform_quaternions(case.n_particles, case.seed)
        active_ids = np.arange(case.n_particles, dtype=np.int64)
        errors = {}
    else:
        spans = case.stored_box
        if case.kind == CaseKind.X_WALLED_PRISM:
            clearance = (
                cylinder.ANALYSIS.wall_cutoff
                + cylinder.SIMULATION.wall_clearance_epsilon
                * cylinder.PARTICLE_DIAMETER
            )
            spans = (case.lx - 2.0 * clearance, spans[1], spans[2])
        positions = _grid_points(spans, case.n_particles)
        orientations = random_uniform_quaternions(case.n_particles, case.seed)
        active_ids = np.arange(case.n_particles, dtype=np.int64)
        errors = {}
    metadata: dict[str, object] = {
        "active_particle_ids": active_ids.tolist(),
        "initial_direction_mean": quaternion_directions(orientations).mean(axis=0).tolist(),
        "initial_orientation_norm_min": float(
            np.linalg.norm(orientations, axis=1).min()
        ),
        "initial_orientation_norm_max": float(
            np.linalg.norm(orientations, axis=1).max()
        ),
    }
    metadata.update(errors)
    return positions, orientations, active_ids, metadata
