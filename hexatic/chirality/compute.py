from pathlib import Path

import gsd.hoomd
import numpy as np

try:
    from hexatic.active_matter_cylinder import (
        _active_direction_from_quaternion,
        _logged_particle_array,
        _theta_edges_and_centers,
        _x_edges_and_centers,
    )
except ImportError:
    from active_matter_cylinder import (
        _active_direction_from_quaternion,
        _logged_particle_array,
        _theta_edges_and_centers,
        _x_edges_and_centers,
    )

from .common import (
    _cylinder_coords,
    _global_mean,
    _global_ratio,
    _global_screw,
    _image_array,
    _mass_array,
    _metric_names_and_labels,
    _radial_mean,
    _radial_ratio,
    _radial_screw,
    _radius_edges_and_centers,
    _unwrapped_x,
    _weighted_mean,
    _xtheta_mean,
    _xtheta_ratio,
    _xtheta_screw,
)
from .config import CYLINDER_SIM, ChiralityConfig, ChiralityFields


def compute_chirality_fields(
    input_gsd: str | Path,
    config: ChiralityConfig = ChiralityConfig(),
    particle_masks: np.ndarray | None = None,
) -> ChiralityFields:
    metric_names, metric_labels = _metric_names_and_labels(config.lag_frames)
    n_metrics = len(metric_names)
    radial_edges, radial_centers = _radius_edges_and_centers(config.radial_bin_width)
    theta_edges, theta_centers = _theta_edges_and_centers(config.n_theta_bins)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        assert len(source) > 0
        n_frames = len(source)
        n_particles = int(source[0].particles.N)
        box_length_x = float(source[0].configuration.box[0])
        x_edges, x_centers = _x_edges_and_centers(box_length_x, config.n_x_bins)

        steps = np.empty(n_frames, dtype=np.int64)
        positions = np.empty((n_frames, n_particles, 3), dtype=np.float64)
        velocities = np.empty((n_frames, n_particles, 3), dtype=np.float64)
        masses = np.empty((n_frames, n_particles), dtype=np.float64)
        coords = np.empty((n_frames, n_particles, 3), dtype=np.float64)
        box_lengths_x = np.empty(n_frames, dtype=np.float64)
        image_values = np.zeros((n_frames, n_particles, 3), dtype=np.int64)
        has_images = True

        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert int(particles.N) == n_particles
            assert particles.position is not None
            assert particles.orientation is not None

            positions[frame_idx] = np.asarray(particles.position, dtype=np.float64)
            coords[frame_idx] = _cylinder_coords(positions[frame_idx])
            masses[frame_idx] = _mass_array(particles, n_particles)
            directions = _active_direction_from_quaternion(particles.orientation)
            forces = _logged_particle_array(frame, "forces", n_particles)
            assert forces.ndim == 2 and forces.shape[1] >= 3
            velocities[frame_idx] = (
                CYLINDER_SIM.u0 * directions + forces[:, :3] / CYLINDER_SIM.gamma
            )
            images = _image_array(particles, n_particles)
            if images is None:
                has_images = False
            else:
                image_values[frame_idx] = images
            box_lengths_x[frame_idx] = float(frame.configuration.box[0])
            steps[frame_idx] = int(frame.configuration.step)

    if particle_masks is not None:
        particle_masks = np.asarray(particle_masks, dtype=bool)
        assert particle_masks.shape == (n_frames, n_particles)

    images_for_unwrap = image_values if has_images else None
    x_unwrapped = _unwrapped_x(positions, images_for_unwrap, box_lengths_x)
    theta_unwrapped = np.unwrap(coords[:, :, 1], axis=0)

    global_values = np.full((n_metrics, n_frames), np.nan, dtype=np.float64)
    radial_values = np.full(
        (n_metrics, n_frames, len(radial_centers)),
        np.nan,
        dtype=np.float64,
    )
    radial_counts = np.zeros(
        (n_metrics, n_frames, len(radial_centers)),
        dtype=np.int64,
    )
    xtheta_values = np.full(
        (n_metrics, n_frames, len(x_centers), len(theta_centers)),
        np.nan,
        dtype=np.float64,
    )
    xtheta_counts = np.zeros(
        (n_metrics, n_frames, len(x_centers), len(theta_centers)),
        dtype=np.int64,
    )

    metric_index = {name: idx for idx, name in enumerate(metric_names)}
    for frame_idx in range(n_frames):
        frame_coords = coords[frame_idx]
        radii = frame_coords[:, 2]
        frame_positions = positions[frame_idx]
        frame_velocities = velocities[frame_idx]
        frame_masses = masses[frame_idx]
        y = frame_positions[:, 1]
        z = frame_positions[:, 2]
        vx = frame_velocities[:, 0]
        vy = frame_velocities[:, 1]
        vz = frame_velocities[:, 2]
        valid = (
            (radii > config.radius_epsilon)
            & np.isfinite(radii)
            & np.all(np.isfinite(frame_velocities), axis=1)
        )
        if particle_masks is not None:
            valid = valid & particle_masks[frame_idx]

        theta_dot = np.full(n_particles, np.nan, dtype=np.float64)
        cross = z * vy - y * vz
        theta_dot[valid] = cross[valid] / (radii[valid] ** 2)

        cm_vx = _weighted_mean(vx, frame_masses, valid)
        relative_vx = vx - cm_vx

        terms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
        helix_raw = vx * theta_dot
        helix_relative = relative_vx * theta_dot
        terms["instant_helix_raw"] = (
            helix_raw,
            np.abs(helix_raw),
            valid & np.isfinite(helix_raw),
            "ratio",
        )
        terms["instant_helix_relative"] = (
            helix_relative,
            np.abs(helix_relative),
            valid & np.isfinite(helix_relative),
            "ratio",
        )

        lx = frame_masses * (y * vz - z * vy)
        perpendicular_speed = np.sqrt(vy * vy + vz * vz)
        angular_momentum_denominator = frame_masses * radii * perpendicular_speed
        terms["angular_momentum"] = (
            lx,
            angular_momentum_denominator,
            valid & np.isfinite(lx) & np.isfinite(angular_momentum_denominator),
            "ratio",
        )

        terms["angular_velocity"] = (
            theta_dot,
            np.abs(theta_dot),
            valid & np.isfinite(theta_dot),
            "ratio",
        )

        screw_coupling = vx * theta_dot
        vx_squared = vx * vx
        screw_mask = (
            valid
            & np.isfinite(screw_coupling)
            & np.isfinite(vx_squared)
            & (vx_squared > 0.0)
        )
        terms["screw"] = (
            vx_squared,
            screw_coupling,
            screw_mask,
            "screw",
        )

        for lag in config.lag_frames:
            if frame_idx + lag >= n_frames:
                continue
            dx = x_unwrapped[frame_idx + lag] - x_unwrapped[frame_idx]
            dtheta = theta_unwrapped[frame_idx + lag] - theta_unwrapped[frame_idx]
            finite_valid = valid & np.isfinite(dx) & np.isfinite(dtheta)
            cm_dx = _weighted_mean(dx, frame_masses, finite_valid)
            dx_relative = dx - cm_dx
            finite_raw = dx * dtheta
            finite_relative = dx_relative * dtheta
            terms[f"finite_time_lag_{lag}_raw"] = (
                finite_raw,
                np.abs(finite_raw),
                finite_valid & np.isfinite(finite_raw),
                "ratio",
            )
            terms[f"finite_time_lag_{lag}_relative"] = (
                finite_relative,
                np.abs(finite_relative),
                finite_valid & np.isfinite(finite_relative),
                "ratio",
            )

        for name, (numerator, denominator, mask, kind) in terms.items():
            metric_idx = metric_index[name]
            if kind == "screw":
                global_values[metric_idx, frame_idx] = _global_screw(
                    numerator,
                    denominator,
                    mask,
                    config.screw_min_screw_rate,
                )
                radial_result, radial_count = _radial_screw(
                    radii,
                    numerator,
                    denominator,
                    mask,
                    radial_edges,
                    config.screw_min_screw_rate,
                )
                xtheta_result, xtheta_count = _xtheta_screw(
                    frame_coords,
                    numerator,
                    denominator,
                    mask,
                    x_edges,
                    theta_edges,
                    config.screw_min_screw_rate,
                )
            elif kind == "mean":
                global_values[metric_idx, frame_idx] = _global_mean(numerator, mask)
                radial_result, radial_count = _radial_mean(
                    radii,
                    numerator,
                    mask,
                    radial_edges,
                )
                xtheta_result, xtheta_count = _xtheta_mean(
                    frame_coords,
                    numerator,
                    mask,
                    x_edges,
                    theta_edges,
                )
            else:
                global_values[metric_idx, frame_idx] = _global_ratio(
                    numerator,
                    denominator,
                    mask,
                )
                radial_result, radial_count = _radial_ratio(
                    radii,
                    numerator,
                    denominator,
                    mask,
                    radial_edges,
                )
                xtheta_result, xtheta_count = _xtheta_ratio(
                    frame_coords,
                    numerator,
                    denominator,
                    mask,
                    x_edges,
                    theta_edges,
                )

            radial_values[metric_idx, frame_idx] = radial_result
            radial_counts[metric_idx, frame_idx] = radial_count
            xtheta_values[metric_idx, frame_idx] = xtheta_result
            xtheta_counts[metric_idx, frame_idx] = xtheta_count

    return ChiralityFields(
        steps=steps,
        metric_names=metric_names,
        metric_labels=metric_labels,
        lag_frames=config.lag_frames,
        x_edges=x_edges,
        x_centers=x_centers,
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        radial_edges=radial_edges,
        radial_centers=radial_centers,
        global_values=global_values,
        radial_values=radial_values,
        radial_counts=radial_counts,
        xtheta_values=xtheta_values,
        xtheta_counts=xtheta_counts,
    )
