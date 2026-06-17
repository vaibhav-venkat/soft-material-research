import numpy as np

from .geometric_common import _radial_groups, _ratio_by_group, _xtheta_groups
from .geometric_config import GeometricChiralityConfig, _TrajectoryData


def _trajectory_chi_frame(
    frame_idx: int,
    data: _TrajectoryData,
    config: GeometricChiralityConfig,
    particle_mask: np.ndarray | None = None,
) -> tuple[float, int, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_radial = len(data.radial_centers)
    x_bins = len(data.x_centers)
    theta_bins = len(data.theta_centers)
    nan_radial = np.full(n_radial, np.nan, dtype=np.float64)
    zero_radial_counts = np.zeros(n_radial, dtype=np.int64)
    zero_radial = np.zeros(n_radial, dtype=np.float64)
    nan_xtheta = np.full((x_bins, theta_bins), np.nan, dtype=np.float64)
    zero_xtheta_counts = np.zeros((x_bins, theta_bins), dtype=np.int64)
    zero_xtheta = np.zeros((x_bins, theta_bins), dtype=np.float64)

    lag = config.trajectory_lag_frames
    last_needed = frame_idx + 3 * lag
    if last_needed >= len(data.steps):
        return (
            np.nan,
            0,
            0.0,
            0.0,
            nan_radial,
            zero_radial_counts,
            zero_radial,
            zero_radial,
            nan_xtheta,
            zero_xtheta_counts,
            zero_xtheta,
            zero_xtheta,
        )

    frame_indices = np.asarray(
        [frame_idx, frame_idx + lag, frame_idx + 2 * lag, last_needed],
        dtype=np.int64,
    )
    sampled = data.unwrapped_positions[frame_indices]
    displacements = np.diff(sampled, axis=0)
    a = displacements[0]
    b = displacements[1]
    c = displacements[2]
    numerator = np.einsum("ij,ij->i", np.cross(a, b), c)
    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
        * np.linalg.norm(c, axis=1)
    )

    mean_radii = np.mean(data.coords[frame_indices, :, 2], axis=0)
    radial_groups = _radial_groups(
        mean_radii,
        data.radial_edges,
        config.radius_epsilon,
    )
    if particle_mask is not None:
        radial_groups = np.where(particle_mask, radial_groups, -1)
    radial_values, radial_counts, radial_num, radial_den = _ratio_by_group(
        numerator,
        denominator,
        radial_groups,
        n_radial,
        config.min_count,
        config.denominator_epsilon,
    )

    xtheta_groups = _xtheta_groups(
        data.coords[frame_idx],
        data.x_edges,
        data.theta_edges,
        data.box_lengths_x[frame_idx],
        config.radius_epsilon,
    )
    if particle_mask is not None:
        xtheta_groups = np.where(particle_mask, xtheta_groups, -1)
    xtheta_values, xtheta_counts, xtheta_num, xtheta_den = _ratio_by_group(
        numerator,
        denominator,
        xtheta_groups,
        x_bins * theta_bins,
        config.min_count,
        config.denominator_epsilon,
    )

    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > config.denominator_epsilon)
    )
    if particle_mask is not None:
        valid = valid & particle_mask
    global_count = int(np.count_nonzero(valid & (radial_groups >= 0)))
    global_numerator = float(np.sum(numerator[valid])) if np.any(valid) else 0.0
    global_denominator = float(np.sum(denominator[valid])) if np.any(valid) else 0.0
    global_value = np.nan
    if global_count >= config.min_count and global_denominator > config.denominator_epsilon:
        global_value = float(np.clip(global_numerator / global_denominator, -1.0, 1.0))

    return (
        global_value,
        global_count,
        global_numerator,
        global_denominator,
        radial_values,
        radial_counts,
        radial_num,
        radial_den,
        xtheta_values.reshape((x_bins, theta_bins)),
        xtheta_counts.reshape((x_bins, theta_bins)),
        xtheta_num.reshape((x_bins, theta_bins)),
        xtheta_den.reshape((x_bins, theta_bins)),
    )
