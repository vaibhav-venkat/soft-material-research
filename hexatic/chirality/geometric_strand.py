import numpy as np

from .geometric_common import (
    _group_counts,
    _valid_particle_mask,
    _weighted_center,
    _xtheta_groups,
)
from .geometric_config import GeometricChiralityConfig

try:
    from hexatic.active_matter_cylinder import _theta_bin_indices
except ImportError:
    from active_matter_cylinder import _theta_bin_indices


def _chi_contributions(
    points: np.ndarray,
    ordered_indices: np.ndarray,
    denominator_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 4:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return empty_float, empty_float, empty_int

    vectors = np.diff(points, axis=0)
    a = vectors[:-2]
    b = vectors[1:-1]
    c = vectors[2:]
    numerator = np.einsum("ij,ij->i", np.cross(a, b), c)
    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
        * np.linalg.norm(c, axis=1)
    )
    middle_indices = ordered_indices[2:-1]
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > denominator_epsilon)
    )
    return numerator[valid], denominator[valid], middle_indices[valid]


def _strand_chi_frame(
    positions: np.ndarray,
    coords: np.ndarray,
    masses: np.ndarray,
    radial_groups: np.ndarray,
    n_radial: int,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    box_length_x: float,
    config: GeometricChiralityConfig,
):
    radial_numerators = np.zeros(n_radial, dtype=np.float64)
    radial_denominators = np.zeros(n_radial, dtype=np.float64)
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    valid_particles = _valid_particle_mask(positions, masses) & (radial_groups >= 0)
    xtheta_groups = _xtheta_groups(
        coords,
        x_edges,
        theta_edges,
        box_length_x,
        config.radius_epsilon,
    )
    valid_radial_groups = np.where(valid_particles, radial_groups, -1)
    valid_xtheta_groups = np.where(valid_particles, xtheta_groups, -1)
    radial_counts = _group_counts(valid_radial_groups, n_radial)
    xtheta_counts = _group_counts(valid_xtheta_groups, x_bins * theta_bins)
    xtheta_numerators = np.zeros(x_bins * theta_bins, dtype=np.float64)
    xtheta_denominators = np.zeros(x_bins * theta_bins, dtype=np.float64)
    total_contributions = 0

    if n_radial == 0:
        empty_xtheta_values = np.full((x_bins, theta_bins), np.nan, dtype=np.float64)
        empty_xtheta_counts = np.zeros((x_bins, theta_bins), dtype=np.int64)
        empty_xtheta_sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
        return (
            np.nan,
            0,
            0.0,
            0.0,
            np.full(0, np.nan, dtype=np.float64),
            radial_counts,
            radial_numerators,
            radial_denominators,
            empty_xtheta_values,
            empty_xtheta_counts,
            empty_xtheta_sums,
            empty_xtheta_sums,
        )

    sector_indices = np.full(len(coords), -1, dtype=np.int64)
    theta_valid = valid_particles & np.isfinite(coords[:, 1])
    sector_indices[theta_valid] = _theta_bin_indices(
        coords[theta_valid, 1],
        config.n_strand_theta_sectors,
    )

    for radial_idx in range(n_radial):
        radial_mask = valid_particles & (radial_groups == radial_idx)
        radial_indices = np.flatnonzero(radial_mask)
        if radial_indices.size < config.min_count:
            continue

        center = _weighted_center(positions, masses, radial_indices)
        if center is None:
            continue

        for sector_idx in range(config.n_strand_theta_sectors):
            indices = radial_indices[sector_indices[radial_indices] == sector_idx]
            if indices.size < config.chi_min_ordered_points:
                continue

            ordered_indices = indices[np.argsort(positions[indices, 0])]
            points = positions[ordered_indices] - center
            numerator, denominator, middle_indices = _chi_contributions(
                points,
                ordered_indices,
                config.denominator_epsilon,
            )
            if numerator.size == 0:
                continue

            radial_numerators[radial_idx] += float(np.sum(numerator))
            radial_denominators[radial_idx] += float(np.sum(denominator))
            total_contributions += int(numerator.size)

            contribution_groups = xtheta_groups[middle_indices]
            contribution_valid = (contribution_groups >= 0) & valid_particles[middle_indices]
            np.add.at(
                xtheta_numerators,
                contribution_groups[contribution_valid],
                numerator[contribution_valid],
            )
            np.add.at(
                xtheta_denominators,
                contribution_groups[contribution_valid],
                denominator[contribution_valid],
            )

    radial_values = np.full(n_radial, np.nan, dtype=np.float64)
    usable_radial = (
        (radial_counts >= config.min_count)
        & (radial_denominators > config.denominator_epsilon)
    )
    radial_values[usable_radial] = np.clip(
        radial_numerators[usable_radial] / radial_denominators[usable_radial],
        -1.0,
        1.0,
    )

    global_numerator = float(np.sum(radial_numerators))
    global_denominator = float(np.sum(radial_denominators))
    global_value = np.nan
    if total_contributions > 0 and global_denominator > config.denominator_epsilon:
        global_value = float(np.clip(global_numerator / global_denominator, -1.0, 1.0))

    xtheta_values = np.full(x_bins * theta_bins, np.nan, dtype=np.float64)
    usable_xtheta = (
        (xtheta_counts >= config.min_count)
        & (xtheta_denominators > config.denominator_epsilon)
    )
    xtheta_values[usable_xtheta] = np.clip(
        xtheta_numerators[usable_xtheta] / xtheta_denominators[usable_xtheta],
        -1.0,
        1.0,
    )

    return (
        global_value,
        total_contributions,
        global_numerator,
        global_denominator,
        radial_values,
        radial_counts,
        radial_numerators,
        radial_denominators,
        xtheta_values.reshape((x_bins, theta_bins)),
        xtheta_counts.reshape((x_bins, theta_bins)),
        xtheta_numerators.reshape((x_bins, theta_bins)),
        xtheta_denominators.reshape((x_bins, theta_bins)),
    )
