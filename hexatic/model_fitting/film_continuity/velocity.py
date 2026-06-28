from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised on environments without numba
    njit = None


def compute_velocities(
    coords: np.ndarray,
    shell_mask: np.ndarray | None,
    lx: float,
    cylinder_radius: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-transition film-surface velocities aligned to frame t."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] < 2:
        raise ValueError("coords must have shape (frames, particles, >=2).")
    if coords.shape[0] < 2:
        raise ValueError("At least two frames are required to compute velocities.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if shell_mask is not None and np.shape(shell_mask) != coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes.")

    if _compute_velocities_numba is not None:
        return _compute_velocities_numba(coords, float(lx), float(cylinder_radius), float(dt))
    return _compute_velocities_numpy(coords, float(lx), float(cylinder_radius), float(dt))


def minimum_image_delta(delta: np.ndarray, period: float) -> np.ndarray:
    return delta - period * np.round(delta / period)


def _compute_velocities_numpy(
    coords: np.ndarray,
    lx: float,
    cylinder_radius: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    dx = minimum_image_delta(coords[1:, :, 0] - coords[:-1, :, 0], lx)
    dtheta = minimum_image_delta(coords[1:, :, 1] - coords[:-1, :, 1], 2.0 * np.pi)
    return dx / dt, cylinder_radius * dtheta / dt


if njit is not None:

    @njit(cache=False)
    def _compute_velocities_numba(
        coords: np.ndarray,
        lx: float,
        cylinder_radius: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_frames = coords.shape[0]
        n_particles = coords.shape[1]
        vx = np.empty((n_frames - 1, n_particles), dtype=np.float64)
        vy = np.empty((n_frames - 1, n_particles), dtype=np.float64)
        theta_period = 2.0 * np.pi
        for frame in range(n_frames - 1):
            for particle in range(n_particles):
                dx = coords[frame + 1, particle, 0] - coords[frame, particle, 0]
                dx -= lx * np.round(dx / lx)
                dtheta = coords[frame + 1, particle, 1] - coords[frame, particle, 1]
                dtheta -= theta_period * np.round(dtheta / theta_period)
                vx[frame, particle] = dx / dt
                vy[frame, particle] = cylinder_radius * dtheta / dt
        return vx, vy

else:
    _compute_velocities_numba = None
