"""Convex particle-hull area and circumference for periodic 2D clusters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True)
class ParticleHull:
    """Geometry of a center hull inflated by the common particle radius."""

    area: float
    circumference: float
    center_area: float
    center_perimeter: float


def _center_hull(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return SciPy center-hull vertices in counterclockwise boundary order."""
    ordered = np.unique(points, axis=0)
    if ordered.shape[0] <= 2:
        return ordered
    try:
        hull = ConvexHull(ordered)
    except QhullError:
        # Three or more collinear unique centers form a two-endpoint segment.
        return ordered[[0, -1]]
    return ordered[hull.vertices]


def particle_hull(
    points: NDArray[np.float64],
    *,
    particle_diameter: float,
) -> ParticleHull:
    """Compute the exact disk-inflated hull of native-unwrapped centers."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("points must have shape (N, 2) with N >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("cluster points must be finite")
    if particle_diameter <= 0.0:
        raise ValueError("particle diameter must be positive")

    hull = _center_hull(values)
    if hull.shape[0] == 1:
        center_area = 0.0
        center_perimeter = 0.0
    elif hull.shape[0] == 2:
        center_area = 0.0
        center_perimeter = 2.0 * float(np.linalg.norm(hull[1] - hull[0]))
    else:
        shifted = np.roll(hull, -1, axis=0)
        center_area = 0.5 * abs(
            float(np.sum(hull[:, 0] * shifted[:, 1] - hull[:, 1] * shifted[:, 0]))
        )
        center_perimeter = float(np.linalg.norm(shifted - hull, axis=1).sum())

    radius = particle_diameter / 2.0
    return ParticleHull(
        area=center_area + radius * center_perimeter + np.pi * radius**2,
        circumference=center_perimeter + 2.0 * np.pi * radius,
        center_area=center_area,
        center_perimeter=center_perimeter,
    )


def cluster_hulls(
    clusters: tuple[NDArray[np.float64], ...],
    *,
    particle_diameter: float,
) -> tuple[ParticleHull, ...]:
    """Compute inflated convex-hull geometry for every native cluster."""
    return tuple(
        particle_hull(
            cluster,
            particle_diameter=particle_diameter,
        )
        for cluster in clusters
    )
