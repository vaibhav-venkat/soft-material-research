"""Particle-derived inputs for rho fitting."""

from __future__ import annotations

import numpy as np

from .config import RhoFittingConfig
from .geometry import minimum_image, surface_lengths, tangential_particle_vectors
from .io import ActiveMatterArrays, load_gsd_orientations, validate_step_alignment


def particle_tangent_directions(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    orientation = None
    if active.direction_cylindrical is None and active.active_direction is None:
        gsd = load_gsd_orientations(config.paths.gsd_path)
        validate_step_alignment(active, gsd)
        orientation = gsd.orientation
    return tangential_particle_vectors(
        active.coords,
        direction_cylindrical=active.direction_cylindrical,
        active_direction=active.active_direction,
        orientation=orientation,
    )


def particle_surface_velocities(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    coords = np.asarray(active.coords, dtype=np.float64)
    assert coords.shape[0] >= 2, "at least two frames are required for surface velocities"
    velocities = np.zeros((*coords.shape[:2], 2), dtype=np.float64)

    lx, _ = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    times = (np.asarray(active.steps, dtype=np.float64) - float(active.steps[0])) * config.settings.timestep
    for frame in range(coords.shape[0]):
        left = max(0, frame - 1)
        right = min(coords.shape[0] - 1, frame + 1)
        dt = times[right] - times[left]
        assert dt > 0.0, "steps must increase over time"
        velocities[frame, :, 0] = minimum_image(coords[right, :, 0] - coords[left, :, 0], lx) / dt
        velocities[frame, :, 1] = (
            active.radius * minimum_image(coords[right, :, 1] - coords[left, :, 1], 2.0 * np.pi) / dt
        )
    return np.ascontiguousarray(velocities)
