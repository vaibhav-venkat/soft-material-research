from pathlib import Path

import gsd.hoomd
import numpy as np

try:
    from hexatic.active_matter_cylinder import (
        _minimum_image_delta,
        _theta_bin_indices,
        _theta_edges_and_centers,
        _x_bin_indices,
        _x_edges_and_centers,
    )
except ImportError:
    from active_matter_cylinder import (
        _minimum_image_delta,
        _theta_bin_indices,
        _theta_edges_and_centers,
        _x_bin_indices,
        _x_edges_and_centers,
    )

from .geometric_config import CYLINDER, GeometricChiralityConfig, _TrajectoryData


def _radius_edges_and_centers(radial_bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    assert radial_bin_width > 0.0
    n_bins = int(np.ceil(CYLINDER.cylinder_radius / radial_bin_width))
    edges = radial_bin_width * np.arange(n_bins + 1)
    edges[-1] = CYLINDER.cylinder_radius
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _cylinder_coords(positions: np.ndarray) -> np.ndarray:
    radii = np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    theta = np.mod(np.arctan2(positions[:, 1], positions[:, 2]), 2.0 * np.pi)
    return np.column_stack((positions[:, 0], theta, radii))


def _mass_array(particles, n_particles: int) -> np.ndarray:
    masses = getattr(particles, "mass", None)
    if masses is None:
        return np.ones(n_particles, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (n_particles,):
        return np.ones(n_particles, dtype=np.float64)
    return masses


def _image_array(particles, n_particles: int) -> np.ndarray | None:
    images = getattr(particles, "image", None)
    if images is None:
        return None
    images = np.asarray(images, dtype=np.int64)
    if images.shape != (n_particles, 3):
        return None
    return images


def _unwrapped_x(
    positions: np.ndarray,
    images: np.ndarray | None,
    box_lengths_x: np.ndarray,
) -> np.ndarray:
    if images is not None:
        return positions[:, :, 0] + images[:, :, 0] * box_lengths_x[:, np.newaxis]

    unwrapped = np.empty_like(positions[:, :, 0])
    unwrapped[0] = positions[0, :, 0]
    for frame_idx in range(1, len(positions)):
        delta = positions[frame_idx, :, 0] - positions[frame_idx - 1, :, 0]
        delta = _minimum_image_delta(delta, float(box_lengths_x[frame_idx]))
        unwrapped[frame_idx] = unwrapped[frame_idx - 1] + delta
    return unwrapped


def _read_trajectory(
    input_gsd: str | Path,
    config: GeometricChiralityConfig,
) -> _TrajectoryData:
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
        coords = np.empty((n_frames, n_particles, 3), dtype=np.float64)
        masses = np.empty((n_frames, n_particles), dtype=np.float64)
        box_lengths_x = np.empty(n_frames, dtype=np.float64)
        image_values = np.zeros((n_frames, n_particles, 3), dtype=np.int64)
        has_images = True

        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert int(particles.N) == n_particles
            assert particles.position is not None

            positions[frame_idx] = np.asarray(particles.position, dtype=np.float64)
            coords[frame_idx] = _cylinder_coords(positions[frame_idx])
            masses[frame_idx] = _mass_array(particles, n_particles)
            box_lengths_x[frame_idx] = float(frame.configuration.box[0])
            steps[frame_idx] = int(frame.configuration.step)

            images = _image_array(particles, n_particles)
            if images is None:
                has_images = False
            else:
                image_values[frame_idx] = images

    images_for_unwrap = image_values if has_images else None
    unwrapped_positions = positions.copy()
    unwrapped_positions[:, :, 0] = _unwrapped_x(
        positions,
        images_for_unwrap,
        box_lengths_x,
    )

    return _TrajectoryData(
        steps=steps,
        positions=positions,
        unwrapped_positions=unwrapped_positions,
        coords=coords,
        masses=masses,
        box_lengths_x=box_lengths_x,
        x_edges=x_edges,
        x_centers=x_centers,
        theta_edges=theta_edges,
        theta_centers=theta_centers,
        radial_edges=radial_edges,
        radial_centers=radial_centers,
    )


def _radial_groups(
    radii: np.ndarray,
    radial_edges: np.ndarray,
    radius_epsilon: float,
) -> np.ndarray:
    groups = np.full(radii.shape, -1, dtype=np.int64)
    valid = (
        np.isfinite(radii)
        & (radii >= 0.0)
        & (radii <= radial_edges[-1] + radius_epsilon)
    )
    indices = np.searchsorted(radial_edges, radii[valid], side="right") - 1
    groups[valid] = np.clip(indices, 0, len(radial_edges) - 2)
    return groups


def _xtheta_groups(
    coords: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    box_length_x: float,
    radius_epsilon: float,
) -> np.ndarray:
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    groups = np.full(coords.shape[0], -1, dtype=np.int64)
    valid = (
        np.all(np.isfinite(coords), axis=1)
        & (coords[:, 2] >= 0.0)
        & (coords[:, 2] <= CYLINDER.cylinder_radius + radius_epsilon)
    )
    x_indices = _x_bin_indices(coords[valid, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[valid, 1], theta_bins)
    groups[valid] = x_indices * theta_bins + theta_indices
    return groups


def _group_counts(groups: np.ndarray, n_groups: int) -> np.ndarray:
    valid = groups >= 0
    if not np.any(valid):
        return np.zeros(n_groups, dtype=np.int64)
    return np.bincount(groups[valid], minlength=n_groups).astype(np.int64)


def _weighted_center(
    positions: np.ndarray,
    masses: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray | None:
    if indices.size == 0:
        return None
    weights = masses[indices]
    total_mass = float(np.sum(weights))
    if not np.isfinite(total_mass) or np.isclose(total_mass, 0.0):
        return None
    return np.sum(positions[indices] * weights[:, np.newaxis], axis=0) / total_mass


def _valid_particle_mask(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return (
        np.all(np.isfinite(positions), axis=1)
        & np.isfinite(masses)
        & (masses > 0.0)
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=~np.isclose(denominator, 0.0),
    )


def _ccm_by_group(
    positions: np.ndarray,
    masses: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    min_count: int,
    denominator_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.full(n_groups, np.nan, dtype=np.float64)
    numerators = np.full(n_groups, np.nan, dtype=np.float64)
    denominators = np.full(n_groups, np.nan, dtype=np.float64)
    valid = (
        (groups >= 0)
        & np.all(np.isfinite(positions), axis=1)
        & np.isfinite(masses)
        & (masses > 0.0)
    )
    counts = _group_counts(groups[valid], n_groups)
    if not np.any(valid):
        return values, counts, numerators, denominators

    valid_groups = groups[valid]
    weights = masses[valid]
    valid_positions = positions[valid]
    mass_sums = np.bincount(valid_groups, weights=weights, minlength=n_groups)
    weighted_sums = np.column_stack(
        [
            np.bincount(
                valid_groups,
                weights=weights * valid_positions[:, component],
                minlength=n_groups,
            )
            for component in range(3)
        ]
    )
    position_sq = np.sum(valid_positions * valid_positions, axis=1)
    projected_sq = valid_positions[:, 0] ** 2 + valid_positions[:, 2] ** 2
    centered_sq_sums = np.bincount(
        valid_groups,
        weights=weights * position_sq,
        minlength=n_groups,
    )
    projected_sums = np.bincount(
        valid_groups,
        weights=weights * projected_sq,
        minlength=n_groups,
    )

    centers = np.divide(
        weighted_sums,
        mass_sums[:, np.newaxis],
        out=np.zeros_like(weighted_sums),
        where=mass_sums[:, np.newaxis] > 0.0,
    )
    denominator = centered_sq_sums - mass_sums * np.sum(centers * centers, axis=1)
    numerator = projected_sums - mass_sums * (
        centers[:, 0] ** 2 + centers[:, 2] ** 2
    )
    numerators[:] = numerator
    denominators[:] = denominator

    usable = (
        (counts >= min_count)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > denominator_epsilon)
    )
    ccm = 1.0 - _safe_ratio(numerator, denominator)
    values[usable] = np.clip(ccm[usable], 0.0, 1.0)
    return values, counts, numerators, denominators


def _ratio_by_group(
    numerator: np.ndarray,
    denominator: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    min_count: int,
    denominator_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.full(n_groups, np.nan, dtype=np.float64)
    numerator_sums = np.zeros(n_groups, dtype=np.float64)
    denominator_sums = np.zeros(n_groups, dtype=np.float64)
    valid = (
        (groups >= 0)
        & np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > denominator_epsilon)
    )
    counts = _group_counts(groups[valid], n_groups)
    if np.any(valid):
        valid_groups = groups[valid]
        numerator_sums = np.bincount(
            valid_groups,
            weights=numerator[valid],
            minlength=n_groups,
        )
        denominator_sums = np.bincount(
            valid_groups,
            weights=denominator[valid],
            minlength=n_groups,
        )

    usable = (counts >= min_count) & (denominator_sums > denominator_epsilon)
    values[usable] = np.clip(
        numerator_sums[usable] / denominator_sums[usable],
        -1.0,
        1.0,
    )
    return values, counts, numerator_sums, denominator_sums
