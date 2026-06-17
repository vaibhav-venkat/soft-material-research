from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

if __package__:
    from hexatic.active_matter_cylinder import (
        ACTIVE_MOVIE_FPS,
        _minimum_image_delta,
        _theta_bin_indices,
        _theta_edges_and_centers,
        _time_edges,
        _x_bin_indices,
        _x_edges_and_centers,
    )
    from hexatic.constants import cylinder
else:
    from active_matter_cylinder import (
        ACTIVE_MOVIE_FPS,
        _minimum_image_delta,
        _theta_bin_indices,
        _theta_edges_and_centers,
        _time_edges,
        _x_bin_indices,
        _x_edges_and_centers,
    )
    from constants import cylinder


CYLINDER = cylinder.ANALYSIS
CYLINDER_PATHS = cylinder.PATHS
GEOMETRIC_CHIRALITY_DATA_DIR = Path(CYLINDER_PATHS.in_gsd).parent
GEOMETRIC_CHIRALITY_IMAGE_DIR = (
    Path(CYLINDER_PATHS.com_plot).parent / "chirality" / "geometric"
)
GEOMETRIC_XTHETA_X_BINS = 32
GEOMETRIC_XTHETA_THETA_BINS = 18

GEOMETRIC_METRIC_NAMES = ("ccm", "chi_strand", "chi_trajectory")
GEOMETRIC_METRIC_LABELS = ("CCM", "strand chi", "trajectory chi")


@dataclass(frozen=True)
class GeometricChiralityConfig:
    radial_bin_width: float = CYLINDER.particle_diameter
    n_x_bins: int = GEOMETRIC_XTHETA_X_BINS
    n_theta_bins: int = GEOMETRIC_XTHETA_THETA_BINS
    min_count: int = 3
    chi_min_ordered_points: int = 5
    n_strand_theta_sectors: int = 24
    trajectory_lag_frames: int = 3
    radius_epsilon: float = 1e-12
    denominator_epsilon: float = 1e-12
    movie_fps: int = ACTIVE_MOVIE_FPS


@dataclass(frozen=True)
class GeometricChiralityFields:
    steps: np.ndarray
    metric_names: tuple[str, ...]
    metric_labels: tuple[str, ...]
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    radial_edges: np.ndarray
    radial_centers: np.ndarray
    global_values: np.ndarray
    global_counts: np.ndarray
    global_numerators: np.ndarray
    global_denominators: np.ndarray
    radial_values: np.ndarray
    radial_counts: np.ndarray
    radial_numerators: np.ndarray
    radial_denominators: np.ndarray
    xtheta_values: np.ndarray
    xtheta_counts: np.ndarray
    xtheta_numerators: np.ndarray
    xtheta_denominators: np.ndarray


@dataclass(frozen=True)
class _TrajectoryData:
    steps: np.ndarray
    positions: np.ndarray
    unwrapped_positions: np.ndarray
    coords: np.ndarray
    masses: np.ndarray
    box_lengths_x: np.ndarray
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    radial_edges: np.ndarray
    radial_centers: np.ndarray


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


def _trajectory_chi_frame(
    frame_idx: int,
    data: _TrajectoryData,
    config: GeometricChiralityConfig,
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


def compute_geometric_chirality_fields(
    input_gsd: str | Path,
    config: GeometricChiralityConfig = GeometricChiralityConfig(),
) -> GeometricChiralityFields:
    data = _read_trajectory(input_gsd, config)
    n_frames = len(data.steps)
    n_metrics = len(GEOMETRIC_METRIC_NAMES)
    n_radial = len(data.radial_centers)
    x_bins = len(data.x_centers)
    theta_bins = len(data.theta_centers)

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
        xtheta_groups = _xtheta_groups(
            coords,
            data.x_edges,
            data.theta_edges,
            box_length_x,
            config.radius_epsilon,
        )

        ccm_global, ccm_global_counts, ccm_global_num, ccm_global_den = _ccm_by_group(
            positions,
            masses,
            np.zeros(len(positions), dtype=np.int64),
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
        ) = _trajectory_chi_frame(frame_idx, data, config)
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


def save_geometric_chirality_fields(
    fields: GeometricChiralityFields,
    filename: str | Path,
    config: GeometricChiralityConfig = GeometricChiralityConfig(),
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        steps=fields.steps,
        metric_names=np.asarray(fields.metric_names),
        metric_labels=np.asarray(fields.metric_labels),
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        theta_edges=fields.theta_edges,
        theta_centers=fields.theta_centers,
        radial_edges=fields.radial_edges,
        radial_centers=fields.radial_centers,
        global_values=fields.global_values,
        global_counts=fields.global_counts,
        global_numerators=fields.global_numerators,
        global_denominators=fields.global_denominators,
        radial_values=fields.radial_values,
        radial_counts=fields.radial_counts,
        radial_numerators=fields.radial_numerators,
        radial_denominators=fields.radial_denominators,
        xtheta_values=fields.xtheta_values,
        xtheta_counts=fields.xtheta_counts,
        xtheta_numerators=fields.xtheta_numerators,
        xtheta_denominators=fields.xtheta_denominators,
        radial_bin_width=np.asarray(config.radial_bin_width, dtype=np.float64),
        n_x_bins=np.asarray(config.n_x_bins, dtype=np.int64),
        n_theta_bins=np.asarray(config.n_theta_bins, dtype=np.int64),
        min_count=np.asarray(config.min_count, dtype=np.int64),
        chi_min_ordered_points=np.asarray(
            config.chi_min_ordered_points,
            dtype=np.int64,
        ),
        n_strand_theta_sectors=np.asarray(
            config.n_strand_theta_sectors,
            dtype=np.int64,
        ),
        trajectory_lag_frames=np.asarray(
            config.trajectory_lag_frames,
            dtype=np.int64,
        ),
        radius_epsilon=np.asarray(config.radius_epsilon, dtype=np.float64),
        denominator_epsilon=np.asarray(
            config.denominator_epsilon,
            dtype=np.float64,
        ),
        movie_fps=np.asarray(config.movie_fps, dtype=np.int64),
    )


def _format_theta_axis(axis) -> None:
    axis.set_ylabel("theta")
    axis.set_ylim(0.0, 2.0 * np.pi)
    axis.set_yticks(
        [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: {
                0.0: "0",
                0.5: r"$\pi/2$",
                1.0: r"$\pi$",
                1.5: r"$3\pi/2$",
                2.0: r"$2\pi$",
            }.get(round(value / np.pi, 1), "")
        )
    )


def _metric_norm(values: np.ndarray, name: str):
    finite = values[np.isfinite(values)]
    if name == "ccm":
        if finite.size == 0:
            return Normalize(vmin=0.0, vmax=1.0)
        vmax = float(np.nanpercentile(finite, 98.0))
        if np.isclose(vmax, 0.0):
            vmax = 1.0
        return Normalize(vmin=0.0, vmax=min(max(vmax, 0.05), 1.0))
    return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)


def _metric_colormap(name: str):
    colormap = plt.get_cmap("viridis" if name == "ccm" else "coolwarm").copy()
    colormap.set_bad("0.85")
    return colormap


def plot_geometric_chirality_global(
    fields: GeometricChiralityFields,
    image_dir: str | Path = GEOMETRIC_CHIRALITY_IMAGE_DIR,
) -> None:
    output_path = Path(image_dir) / "geometric_chirality_global_ccm_chi.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_index = {name: idx for idx, name in enumerate(fields.metric_names)}

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(
        fields.steps,
        fields.global_values[metric_index["ccm"]],
        label="global CCM",
        linewidth=2.0,
    )
    axis.plot(
        fields.steps,
        fields.global_values[metric_index["chi_strand"]],
        label="global strand chi",
    )
    axis.plot(
        fields.steps,
        fields.global_values[metric_index["chi_trajectory"]],
        label="global trajectory chi",
    )
    axis.axhline(0.0, color="0.35", linewidth=0.9)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Geometric chirality")
    axis.set_title("Geometric chirality diagnostics")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_geometric_chirality_radial_heatmaps(
    fields: GeometricChiralityFields,
    image_dir: str | Path = GEOMETRIC_CHIRALITY_IMAGE_DIR,
    min_count: int = GeometricChiralityConfig.min_count,
) -> None:
    output_path = Path(image_dir) / "geometric_chirality_radial_heatmaps.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_edges = _time_edges(fields.steps)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=True)
    for metric_idx, axis in enumerate(axes):
        name = fields.metric_names[metric_idx]
        label = fields.metric_labels[metric_idx]
        values = fields.radial_values[metric_idx].copy()
        values[fields.radial_counts[metric_idx] < min_count] = np.nan
        mesh = axis.pcolormesh(
            time_edges,
            fields.radial_edges,
            np.ma.masked_invalid(values.T),
            shading="auto",
            cmap=_metric_colormap(name),
            norm=_metric_norm(values, name),
        )
        fig.colorbar(mesh, ax=axis, label=label)
        axis.set_ylabel("r")
        axis.set_title(label)
    axes[-1].set_xlabel("Simulation step")
    fig.suptitle(f"Geometric chirality by radius (N >= {min_count})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _draw_xtheta_heatmap(
    fields: GeometricChiralityFields,
    metric_idx: int,
    frame_idx: int,
    fig,
    axis,
    min_count: int,
    norm,
    colormap,
) -> None:
    values = fields.xtheta_values[metric_idx, frame_idx].copy()
    values[fields.xtheta_counts[metric_idx, frame_idx] < min_count] = np.nan
    masked_values = np.ma.masked_invalid(values)
    x_grid, theta_grid = np.meshgrid(
        fields.x_edges,
        fields.theta_edges,
        indexing="ij",
    )
    mesh = axis.pcolormesh(
        x_grid,
        theta_grid,
        masked_values,
        shading="auto",
        cmap=colormap,
        norm=norm,
    )
    fig.colorbar(mesh, ax=axis, label=fields.metric_labels[metric_idx])
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(
        f"{fields.metric_labels[metric_idx]} in x-theta, "
        f"step {fields.steps[frame_idx]}"
    )
    if masked_values.count() == 0:
        axis.text(
            0.5,
            0.5,
            "no valid bins",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="0.25",
        )


def _write_metric_movie(
    fields: GeometricChiralityFields,
    metric_idx: int,
    filename: str | Path,
    min_count: int,
    fps: int,
) -> None:
    name = fields.metric_names[metric_idx]
    values = fields.xtheta_values[metric_idx].copy()
    values[fields.xtheta_counts[metric_idx] < min_count] = np.nan
    norm = _metric_norm(values, name)
    colormap = _metric_colormap(name)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 5))
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(output_path), dpi=160):
        for frame_idx in range(len(fields.steps)):
            fig.clear()
            axis = fig.add_subplot(111)
            _draw_xtheta_heatmap(
                fields,
                metric_idx,
                frame_idx,
                fig,
                axis,
                min_count,
                norm,
                colormap,
            )
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)


def write_geometric_chirality_xtheta_movies(
    fields: GeometricChiralityFields,
    image_dir: str | Path = GEOMETRIC_CHIRALITY_IMAGE_DIR,
    min_count: int = GeometricChiralityConfig.min_count,
    fps: int = ACTIVE_MOVIE_FPS,
) -> None:
    image_path = Path(image_dir)
    movie_names = {
        "ccm": "geometric_ccm_xtheta.mp4",
        "chi_strand": "geometric_chi_strand_xtheta.mp4",
        "chi_trajectory": "geometric_chi_trajectory_xtheta.mp4",
    }
    for metric_idx, name in enumerate(fields.metric_names):
        _write_metric_movie(
            fields,
            metric_idx,
            image_path / movie_names[name],
            min_count=min_count,
            fps=fps,
        )


def write_geometric_chirality_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    data_dir: str | Path = GEOMETRIC_CHIRALITY_DATA_DIR,
    image_dir: str | Path = GEOMETRIC_CHIRALITY_IMAGE_DIR,
    config: GeometricChiralityConfig = GeometricChiralityConfig(),
    write_movies: bool = True,
) -> GeometricChiralityFields:
    fields = compute_geometric_chirality_fields(input_gsd, config=config)
    save_geometric_chirality_fields(
        fields,
        Path(data_dir) / "geometric_chirality_fields.npz",
        config=config,
    )
    plot_geometric_chirality_global(fields, image_dir=image_dir)
    plot_geometric_chirality_radial_heatmaps(
        fields,
        image_dir=image_dir,
        min_count=config.min_count,
    )
    if write_movies:
        write_geometric_chirality_xtheta_movies(
            fields,
            image_dir=image_dir,
            min_count=config.min_count,
            fps=config.movie_fps,
        )
    return fields
