"""Cylinder geometry helpers."""

from __future__ import annotations

import numpy as np

from . import _rho_fitting_core, _rho_fitting_core_import_error


def theta_to_y(theta: np.ndarray, radius: float) -> np.ndarray:
    """Convert cylindrical angle coordinates to unwrapped surface distance."""
    return radius * theta


def minimum_image(delta: np.ndarray, period: float) -> np.ndarray:
    """Wrap displacements into the centered minimum-image interval for a periodic axis."""
    return delta - period * np.round(delta / period)


def grid_centers(edges: np.ndarray) -> np.ndarray:
    """Return cell centers for a one-dimensional edge array."""
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def surface_lengths(x_edges: np.ndarray, theta_edges: np.ndarray, radius: float) -> tuple[float, float]:
    """Return the unwrapped cylinder side lengths used by periodic surface derivatives.

    Parameters:
        x_edges: One-dimensional axial grid edges.
        theta_edges: One-dimensional angular edges expected to span a full ``2*pi`` cylinder.
        radius: Cylinder radius used to convert angular length to surface distance.

    Returns:
        ``(lx, ly)`` where ``lx`` is the axial period and ``ly = 2*pi*radius``.

    Edge cases:
        Does not support partial angular domains; derivative code assumes full periodicity.
    """
    lx = float(x_edges[-1] - x_edges[0])
    theta_span = float(theta_edges[-1] - theta_edges[0])
    assert np.isclose(theta_span, 2.0 * np.pi, rtol=1.0e-6, atol=1.0e-8), (
        "theta_edges must span 2*pi for full-cylinder derivatives"
    )
    ly = float(2.0 * np.pi * radius)
    assert lx > 0.0 and ly > 0.0, "surface lengths must be positive"
    return lx, ly


def active_direction_from_quaternion(orientation: np.ndarray) -> np.ndarray:
    """Rotate the particle body-frame active axis into Cartesian coordinates.

    Parameters:
        orientation: Quaternion array with shape ``(frames, particles, 4)`` in HOOMD
            ``(w, x, y, z)`` order.

    Returns:
        Cartesian active directions with shape ``(frames, particles, 3)``.

    Edge cases:
        Zero-norm quaternions are rejected instead of silently producing NaNs.
    """
    return np.asarray(_core().particle_active_direction(np.ascontiguousarray(orientation, dtype=float)))


def cartesian_to_cylindrical_components(vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Project Cartesian vectors onto axial, radial, and azimuthal cylinder directions.

    Parameters:
        vectors: Cartesian vector field with shape ``theta.shape + (3,)``.
        theta: Per-vector angular coordinate in radians.

    Returns:
        Array with components ``(x, radial, azimuthal)`` and the same leading axes as
        ``theta``.
    """
    vectors = np.ascontiguousarray(vectors, dtype=float)
    theta = np.asarray(theta, dtype=float)
    assert vectors.ndim == 3 and vectors.shape[-1] == 3 and vectors.shape[:-1] == theta.shape, (
        "vectors must have shape theta.shape + (3,)"
    )
    return np.asarray(
        _core().cartesian_to_cylindrical(vectors, np.ascontiguousarray(theta, dtype=float))
    )


def tangential_particle_vectors(
    coords: np.ndarray,
    *,
    direction_cylindrical: np.ndarray | None = None,
    active_direction: np.ndarray | None = None,
    orientation: np.ndarray | None = None,
) -> np.ndarray:
    """Return particle orientation vectors in axial, azimuthal, radial order.

    Parameters:
        coords: Particle coordinates with at least ``(x, theta)`` in the last axis.
        direction_cylindrical: Optional precomputed ``(x, radial, azimuthal)`` directions.
        active_direction: Optional Cartesian active directions, used when cylindrical
            directions are not already cached.
        orientation: Optional HOOMD quaternions used as the final fallback.

    Returns:
        Contiguous array shaped ``(frames, particles, 3)`` with components ordered as
        ``(x, R theta tangent, radial)`` for moment construction.

    Examples:
        ``tangential_particle_vectors(coords, active_direction=directors)``
        ``tangential_particle_vectors(coords, orientation=quaternions)``

    Edge cases:
        Exactly one orientation source is not required, but at least one usable source must
        be present when cached cylindrical directions are absent.
    """
    coords = np.ascontiguousarray(coords, dtype=float)
    assert coords.ndim == 3 and coords.shape[-1] == 3, "coords must have shape (frames, particles, 3)"
    return np.asarray(
        _core().particle_directions(
            coords,
            None if direction_cylindrical is None else np.ascontiguousarray(direction_cylindrical, dtype=float),
            None if active_direction is None else np.ascontiguousarray(active_direction, dtype=float),
            None if orientation is None else np.ascontiguousarray(orientation, dtype=float),
        )
    )


def _core():
    """Return the compiled particle-transform core."""
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    return _rho_fitting_core
