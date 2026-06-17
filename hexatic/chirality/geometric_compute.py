from pathlib import Path

import numpy as np

from .geometric_common import _ccm_by_group, _radial_groups, _read_trajectory, _xtheta_groups
from .geometric_config import (
    GEOMETRIC_METRIC_LABELS,
    GEOMETRIC_METRIC_NAMES,
    GeometricChiralityConfig,
    GeometricChiralityFields,
)
from .geometric_strand import _strand_chi_frame
from .geometric_trajectory import _trajectory_chi_frame


def compute_geometric_chirality_fields(
    input_gsd: str | Path,
    config: GeometricChiralityConfig = GeometricChiralityConfig(),
    particle_masks: np.ndarray | None = None,
) -> GeometricChiralityFields:
    data = _read_trajectory(input_gsd, config)
    n_frames = len(data.steps)
    n_particles = data.positions.shape[1]
    n_metrics = len(GEOMETRIC_METRIC_NAMES)
    n_radial = len(data.radial_centers)
    x_bins = len(data.x_centers)
    theta_bins = len(data.theta_centers)
    if particle_masks is not None:
        particle_masks = np.asarray(particle_masks, dtype=bool)
        assert particle_masks.shape == (n_frames, n_particles)

    global_values = np.full((n_metrics, n_frames), np.nan, dtype=np.float64)
    global_counts = np.zeros((n_metrics, n_frames), dtype=np.int64)
    global_numerators = np.zeros((n_metrics, n_frames), dtype=np.float64)
    global_denominators = np.zeros((n_metrics, n_frames), dtype=np.float64)
    radial_values = np.full((n_metrics, n_frames, n_radial), np.nan, dtype=np.float64)
    radial_counts = np.zeros((n_metrics, n_frames, n_radial), dtype=np.int64)
    radial_numerators = np.zeros((n_metrics, n_frames, n_radial), dtype=np.float64)
    radial_denominators = np.zeros((n_metrics, n_frames, n_radial), dtype=np.float64)
    xtheta_values = np.full(
        (n_metrics, n_frames, x_bins, theta_bins),
        np.nan,
        dtype=np.float64,
    )
    xtheta_counts = np.zeros(
        (n_metrics, n_frames, x_bins, theta_bins),
        dtype=np.int64,
    )
    xtheta_numerators = np.zeros_like(xtheta_values)
    xtheta_denominators = np.zeros_like(xtheta_values)

    ccm_idx = GEOMETRIC_METRIC_NAMES.index("ccm")
    strand_idx = GEOMETRIC_METRIC_NAMES.index("chi_strand")
    trajectory_idx = GEOMETRIC_METRIC_NAMES.index("chi_trajectory")

    for frame_idx in range(n_frames):
        positions = data.positions[frame_idx]
        coords = data.coords[frame_idx]
        masses = data.masses[frame_idx]
        box_length_x = data.box_lengths_x[frame_idx]
        radial_groups = _radial_groups(
            coords[:, 2],
            data.radial_edges,
            config.radius_epsilon,
        )
        frame_particle_mask = (
            particle_masks[frame_idx]
            if particle_masks is not None
            else np.ones(len(positions), dtype=bool)
        )
        radial_groups = np.where(frame_particle_mask, radial_groups, -1)
        xtheta_groups = _xtheta_groups(
            coords,
            data.x_edges,
            data.theta_edges,
            box_length_x,
            config.radius_epsilon,
        )
        xtheta_groups = np.where(frame_particle_mask, xtheta_groups, -1)

        ccm_global, ccm_global_counts, ccm_global_num, ccm_global_den = _ccm_by_group(
            positions,
            masses,
            np.where(frame_particle_mask, 0, -1),
            1,
            config.min_count,
            config.denominator_epsilon,
        )
        global_values[ccm_idx, frame_idx] = ccm_global[0]
        global_counts[ccm_idx, frame_idx] = ccm_global_counts[0]
        global_numerators[ccm_idx, frame_idx] = ccm_global_num[0]
        global_denominators[ccm_idx, frame_idx] = ccm_global_den[0]

        ccm_radial, ccm_radial_counts, ccm_radial_num, ccm_radial_den = _ccm_by_group(
            positions,
            masses,
            radial_groups,
            n_radial,
            config.min_count,
            config.denominator_epsilon,
        )
        radial_values[ccm_idx, frame_idx] = ccm_radial
        radial_counts[ccm_idx, frame_idx] = ccm_radial_counts
        radial_numerators[ccm_idx, frame_idx] = ccm_radial_num
        radial_denominators[ccm_idx, frame_idx] = ccm_radial_den

        ccm_xtheta, ccm_xtheta_counts, ccm_xtheta_num, ccm_xtheta_den = _ccm_by_group(
            positions,
            masses,
            xtheta_groups,
            x_bins * theta_bins,
            config.min_count,
            config.denominator_epsilon,
        )
        xtheta_values[ccm_idx, frame_idx] = ccm_xtheta.reshape((x_bins, theta_bins))
        xtheta_counts[ccm_idx, frame_idx] = ccm_xtheta_counts.reshape((x_bins, theta_bins))
        xtheta_numerators[ccm_idx, frame_idx] = ccm_xtheta_num.reshape(
            (x_bins, theta_bins)
        )
        xtheta_denominators[ccm_idx, frame_idx] = ccm_xtheta_den.reshape(
            (x_bins, theta_bins)
        )

        (
            strand_global,
            strand_count,
            strand_num,
            strand_den,
            strand_radial,
            strand_radial_counts,
            strand_radial_num,
            strand_radial_den,
            strand_xtheta,
            strand_xtheta_counts,
            strand_xtheta_num,
            strand_xtheta_den,
        ) = _strand_chi_frame(
            positions,
            coords,
            masses,
            radial_groups,
            n_radial,
            data.x_edges,
            data.theta_edges,
            box_length_x,
            config,
        )
        global_values[strand_idx, frame_idx] = strand_global
        global_counts[strand_idx, frame_idx] = strand_count
        global_numerators[strand_idx, frame_idx] = strand_num
        global_denominators[strand_idx, frame_idx] = strand_den
        radial_values[strand_idx, frame_idx] = strand_radial
        radial_counts[strand_idx, frame_idx] = strand_radial_counts
        radial_numerators[strand_idx, frame_idx] = strand_radial_num
        radial_denominators[strand_idx, frame_idx] = strand_radial_den
        xtheta_values[strand_idx, frame_idx] = strand_xtheta
        xtheta_counts[strand_idx, frame_idx] = strand_xtheta_counts
        xtheta_numerators[strand_idx, frame_idx] = strand_xtheta_num
        xtheta_denominators[strand_idx, frame_idx] = strand_xtheta_den

        (
            trajectory_global,
            trajectory_count,
            trajectory_num,
            trajectory_den,
            trajectory_radial,
            trajectory_radial_counts,
            trajectory_radial_num,
            trajectory_radial_den,
            trajectory_xtheta,
            trajectory_xtheta_counts,
            trajectory_xtheta_num,
            trajectory_xtheta_den,
        ) = _trajectory_chi_frame(frame_idx, data, config, frame_particle_mask)
        global_values[trajectory_idx, frame_idx] = trajectory_global
        global_counts[trajectory_idx, frame_idx] = trajectory_count
        global_numerators[trajectory_idx, frame_idx] = trajectory_num
        global_denominators[trajectory_idx, frame_idx] = trajectory_den
        radial_values[trajectory_idx, frame_idx] = trajectory_radial
        radial_counts[trajectory_idx, frame_idx] = trajectory_radial_counts
        radial_numerators[trajectory_idx, frame_idx] = trajectory_radial_num
        radial_denominators[trajectory_idx, frame_idx] = trajectory_radial_den
        xtheta_values[trajectory_idx, frame_idx] = trajectory_xtheta
        xtheta_counts[trajectory_idx, frame_idx] = trajectory_xtheta_counts
        xtheta_numerators[trajectory_idx, frame_idx] = trajectory_xtheta_num
        xtheta_denominators[trajectory_idx, frame_idx] = trajectory_xtheta_den

    return GeometricChiralityFields(
        steps=data.steps,
        metric_names=GEOMETRIC_METRIC_NAMES,
        metric_labels=GEOMETRIC_METRIC_LABELS,
        x_edges=data.x_edges,
        x_centers=data.x_centers,
        theta_edges=data.theta_edges,
        theta_centers=data.theta_centers,
        radial_edges=data.radial_edges,
        radial_centers=data.radial_centers,
        global_values=global_values,
        global_counts=global_counts,
        global_numerators=global_numerators,
        global_denominators=global_denominators,
        radial_values=radial_values,
        radial_counts=radial_counts,
        radial_numerators=radial_numerators,
        radial_denominators=radial_denominators,
        xtheta_values=xtheta_values,
        xtheta_counts=xtheta_counts,
        xtheta_numerators=xtheta_numerators,
        xtheta_denominators=xtheta_denominators,
    )
