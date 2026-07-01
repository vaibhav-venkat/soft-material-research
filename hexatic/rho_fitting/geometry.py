"""Cylinder geometry helpers."""

from __future__ import annotations

import numpy as np


def theta_to_y(theta: np.ndarray, radius: float) -> np.ndarray:
    return radius * theta


def minimum_image(delta: np.ndarray, period: float) -> np.ndarray:
    return delta - period * np.round(delta / period)


def grid_centers(edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def surface_lengths(x_edges: np.ndarray, theta_edges: np.ndarray, radius: float) -> tuple[float, float]:
    lx = float(x_edges[-1] - x_edges[0])
    theta_span = float(theta_edges[-1] - theta_edges[0])
    assert np.isclose(theta_span, 2.0 * np.pi, rtol=1.0e-6, atol=1.0e-8), (
        "theta_edges must span 2*pi for full-cylinder derivatives"
    )
    ly = float(2.0 * np.pi * radius)
    assert lx > 0.0 and ly > 0.0, "surface lengths must be positive"
    return lx, ly


def active_direction_from_quaternion(orientation: np.ndarray) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=float)
    assert orientation.ndim == 3 and orientation.shape[-1] == 4, (
        "orientation must have shape (frames, particles, 4)"
    )

    norms = np.linalg.norm(orientation, axis=-1)
    assert np.all(norms > 0.0), "orientation contains zero-norm quaternions"
    quat = orientation / norms[..., np.newaxis]
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        axis=-1,
    )


def cartesian_to_cylindrical_components(vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    theta = np.asarray(theta, dtype=float)
    assert vectors.shape[-1] == 3 and vectors.shape[:-1] == theta.shape, (
        "vectors must have shape theta.shape + (3,)"
    )

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    radial = vectors[..., 1] * sin_theta + vectors[..., 2] * cos_theta
    azimuthal = vectors[..., 1] * cos_theta - vectors[..., 2] * sin_theta
    return np.stack((vectors[..., 0], radial, azimuthal), axis=-1)


def tangential_particle_vectors(
    coords: np.ndarray,
    *,
    direction_cylindrical: np.ndarray | None = None,
    active_direction: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    assert coords.ndim == 3 and coords.shape[-1] >= 2, "coords must have shape (frames, particles, >=2)"

    if direction_cylindrical is not None:
        cylindrical = np.asarray(direction_cylindrical, dtype=float)
    else:
        if active_direction is None:
            if orientation is None:
                assert False, "missing particle orientation source"
            active_direction = active_direction_from_quaternion(orientation)
        cylindrical = cartesian_to_cylindrical_components(active_direction, coords[..., 1])

    assert cylindrical.shape[:2] == coords.shape[:2] and cylindrical.shape[-1] == 3, (
        "particle directions must match coords frame/particle axes"
    )
    return np.ascontiguousarray(np.stack((cylindrical[..., 0], cylindrical[..., 2]), axis=-1))
