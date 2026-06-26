from __future__ import annotations

import numpy as np


def cylindrical_to_cartesian(coords: np.ndarray) -> np.ndarray:
    """Convert cached ``(x, theta, r)`` particle coordinates to ``(x, y, z)``."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[-1] != 3:
        raise ValueError(f"coords last dimension must be 3, got {coords.shape}")

    cartesian = np.empty_like(coords, dtype=np.float64)
    cartesian[..., 0] = coords[..., 0]
    cartesian[..., 1] = coords[..., 2] * np.sin(coords[..., 1])
    cartesian[..., 2] = coords[..., 2] * np.cos(coords[..., 1])
    return cartesian


def minimum_image_x_delta(dx: np.ndarray | float, box_length_x: float) -> np.ndarray:
    dx_array = np.asarray(dx, dtype=np.float64)
    if box_length_x <= 0.0:
        return dx_array
    return dx_array - box_length_x * np.round(dx_array / box_length_x)


def cylinder_distances(
    points: np.ndarray,
    centers: np.ndarray,
    box_length_x: float,
) -> np.ndarray:
    """Euclidean distances with minimum-image wrapping along cylinder x."""
    points = np.asarray(points, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    delta = points[..., np.newaxis, :] - centers[np.newaxis, ...]
    delta[..., 0] = minimum_image_x_delta(delta[..., 0], box_length_x)
    return np.linalg.norm(delta, axis=-1)


def annulus_mask(
    particle_coords: np.ndarray,
    defect_coords: np.ndarray,
    box_length_x: float,
    inner_radius: float,
    outer_radius: float,
    disclination_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mask particles in any defect-centered annulus, optionally excluding defects."""
    if inner_radius < 0.0 or outer_radius <= inner_radius:
        raise ValueError("annulus radii must satisfy 0 <= inner_radius < outer_radius")

    distances = cylinder_distances(particle_coords, defect_coords, box_length_x)
    mask = np.any((distances > inner_radius) & (distances < outer_radius), axis=-1)
    if disclination_mask is not None:
        mask &= ~np.asarray(disclination_mask, dtype=bool)
    return mask
