"""Particle-derived inputs for rho fitting."""

from __future__ import annotations

import numpy as np

from . import _rho_fitting_core, _rho_fitting_core_import_error
from .config import RhoFittingConfig
from .geometry import surface_lengths
from .io import ActiveMatterArrays, load_gsd_orientations, validate_step_alignment


def particle_tangent_directions(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    """Return particle active directions in the surface-oriented moment basis.

    Parameters:
        active: Loaded particle arrays, optionally including cached direction fields.
        config: Run configuration used to locate the GSD trajectory if quaternions are
            needed as a fallback.

    Returns:
        Contiguous ``(frames, particles, 3)`` array ordered as axial, azimuthal, radial.

    Edge cases:
        GSD orientations are loaded only when neither cylindrical nor Cartesian cached
        directions exist, and their steps must exactly match the NPZ cache.
    """
    orientation = None
    if active.direction_cylindrical is None and active.active_direction is None:
        gsd = load_gsd_orientations(config.paths.gsd_path)
        validate_step_alignment(active, gsd)
        orientation = gsd.orientation
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    return np.ascontiguousarray(
        _rho_fitting_core.particle_directions(
            np.ascontiguousarray(active.coords, dtype=np.float64),
            None
            if active.direction_cylindrical is None
            else np.ascontiguousarray(active.direction_cylindrical, dtype=np.float64),
            None
            if active.active_direction is None
            else np.ascontiguousarray(active.active_direction, dtype=np.float64),
            None if orientation is None else np.ascontiguousarray(orientation, dtype=np.float64),
        )
    )


def particle_surface_velocities(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    """Estimate particle velocities in the cylindrical basis.

    Parameters:
        active: Particle coordinates whose last axis stores ``x`` and ``theta``.
        config: Run configuration providing the simulation timestep.

    Returns:
        ``(frames, particles, 3)`` velocities in axial, azimuthal, and radial directions.

    Edge cases:
        End frames use one-sided two-frame differences; all angular and axial
        displacements use minimum-image wrapping.
    """
    coords = np.ascontiguousarray(active.coords, dtype=np.float64)
    assert coords.shape[0] >= 2, "at least two frames are required for surface velocities"
    assert config.settings is not None, "rho fitting settings were not initialized"
    lx, _ = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    theta_period = float(active.theta_edges[-1] - active.theta_edges[0])
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    return np.ascontiguousarray(
        _rho_fitting_core.particle_surface_velocities(
            coords,
            np.ascontiguousarray(active.steps, dtype=np.int64),
            float(config.settings.timestep),
            float(lx),
            theta_period,
        )
    )
