from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

if __package__:
    from hexatic.active_matter_cylinder import (
        ACTIVE_MOVIE_FPS,
        _active_direction_from_quaternion,
        _logged_particle_array,
        _minimum_image_delta,
        _theta_bin_indices,
        _theta_edges_and_centers,
        _time_edges,
        _x_bin_indices,
        _x_edges_and_centers,
    )
    from hexatic.constants import cylinder
    from hexatic.chirality_geometric import (
        GeometricChiralityConfig,
        write_geometric_chirality_outputs,
    )
else:
    from active_matter_cylinder import (
        ACTIVE_MOVIE_FPS,
        _active_direction_from_quaternion,
        _logged_particle_array,
        _minimum_image_delta,
        _theta_bin_indices,
        _theta_edges_and_centers,
        _time_edges,
        _x_bin_indices,
        _x_edges_and_centers,
    )
    from constants import cylinder
    from chirality_geometric import (
        GeometricChiralityConfig,
        write_geometric_chirality_outputs,
    )


CYLINDER = cylinder.ANALYSIS
CYLINDER_PATHS = cylinder.PATHS
CYLINDER_SIM = cylinder.SIMULATION
CHIRALITY_DATA_DIR = Path(CYLINDER_PATHS.in_gsd).parent
CHIRALITY_IMAGE_DIR = Path(CYLINDER_PATHS.com_plot).parent / "chirality"


@dataclass(frozen=True)
class ChiralityConfig:
    radial_bin_width: float = CYLINDER.particle_diameter
    n_x_bins: int = 100
    n_theta_bins: int = 72
    lag_frames: tuple[int, ...] = (5,)
    min_count: int = 4
    xtheta_min_count: int = 1
    screw_min_screw_rate: float = 0
    radius_epsilon: float = 1e-12
    movie_fps: int = ACTIVE_MOVIE_FPS
    limit_disclination: bool = False


@dataclass(frozen=True)
class ChiralityFields:
    steps: np.ndarray
    metric_names: tuple[str, ...]
    metric_labels: tuple[str, ...]
    lag_frames: tuple[int, ...]
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    radial_edges: np.ndarray
    radial_centers: np.ndarray
    global_values: np.ndarray
    radial_values: np.ndarray
    radial_counts: np.ndarray
    xtheta_values: np.ndarray
    xtheta_counts: np.ndarray


@dataclass(frozen=True)
class NeighborCountMatrix:
    steps: np.ndarray
    counts: np.ndarray


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


def _weighted_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return np.nan
    denominator = float(np.sum(weights[mask]))
    if np.isclose(denominator, 0.0):
        return np.nan
    return float(np.sum(values[mask] * weights[mask]) / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return np.nan
    return float(numerator / denominator)


def _global_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return np.nan
    return _safe_ratio(float(np.sum(numerator[mask])), float(np.sum(denominator[mask])))


def _global_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return np.nan
    return float(np.mean(values[mask]))


def _radial_ratio(
    radii: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    mask: np.ndarray,
    radial_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = len(radial_edges) - 1
    values = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    if not np.any(mask):
        return values, counts

    indices = np.searchsorted(radial_edges, radii[mask], side="right") - 1
    indices = np.clip(indices, 0, n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    num_sums = np.bincount(indices, weights=numerator[mask], minlength=n_bins)
    den_sums = np.bincount(indices, weights=denominator[mask], minlength=n_bins)
    values = np.divide(
        num_sums,
        den_sums,
        out=np.full(n_bins, np.nan, dtype=np.float64),
        where=~np.isclose(den_sums, 0.0),
    )
    return values, counts


def _radial_mean(
    radii: np.ndarray,
    particle_values: np.ndarray,
    mask: np.ndarray,
    radial_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = len(radial_edges) - 1
    values = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    if not np.any(mask):
        return values, counts

    indices = np.searchsorted(radial_edges, radii[mask], side="right") - 1
    indices = np.clip(indices, 0, n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    sums = np.bincount(indices, weights=particle_values[mask], minlength=n_bins)
    values = np.divide(
        sums,
        counts,
        out=np.full(n_bins, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return values, counts


def _xtheta_ratio(
    coords: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    mask: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    values = np.full((x_bins, theta_bins), np.nan, dtype=np.float64)
    counts = np.zeros((x_bins, theta_bins), dtype=np.int64)
    if not np.any(mask):
        return values, counts

    box_length_x = float(x_edges[-1] - x_edges[0])
    x_indices = _x_bin_indices(coords[mask, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[mask, 1], theta_bins)
    np.add.at(counts, (x_indices, theta_indices), 1)
    num_sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
    den_sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
    np.add.at(num_sums, (x_indices, theta_indices), numerator[mask])
    np.add.at(den_sums, (x_indices, theta_indices), denominator[mask])
    values = np.divide(
        num_sums,
        den_sums,
        out=np.full_like(num_sums, np.nan),
        where=~np.isclose(den_sums, 0.0),
    )
    return values, counts


def _xtheta_mean(
    coords: np.ndarray,
    particle_values: np.ndarray,
    mask: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    values = np.full((x_bins, theta_bins), np.nan, dtype=np.float64)
    counts = np.zeros((x_bins, theta_bins), dtype=np.int64)
    if not np.any(mask):
        return values, counts

    box_length_x = float(x_edges[-1] - x_edges[0])
    x_indices = _x_bin_indices(coords[mask, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[mask, 1], theta_bins)
    np.add.at(counts, (x_indices, theta_indices), 1)
    sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
    np.add.at(sums, (x_indices, theta_indices), particle_values[mask])
    values = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return values, counts


def _screw_from_sums(
    vx2_sums: np.ndarray,
    coupling_sums: np.ndarray,
    min_screw_rate: float,
) -> np.ndarray:
    kappa = np.divide(
        coupling_sums,
        vx2_sums,
        out=np.full_like(coupling_sums, np.nan, dtype=np.float64),
        where=~np.isclose(vx2_sums, 0.0),
    )
    return np.divide(
        kappa,
        1,
        out=np.full_like(kappa, np.nan, dtype=np.float64),
        where=np.abs(kappa) >= min_screw_rate,
    )


def _global_screw(
    vx2: np.ndarray,
    coupling: np.ndarray,
    mask: np.ndarray,
    min_screw_rate: float,
) -> float:
    if not np.any(mask):
        return np.nan
    screw = _screw_from_sums(
        np.asarray(float(np.sum(vx2[mask]))),
        np.asarray(float(np.sum(coupling[mask]))),
        min_screw_rate,
    )
    return float(screw) if np.isfinite(screw) else np.nan


def _radial_screw(
    radii: np.ndarray,
    vx2: np.ndarray,
    coupling: np.ndarray,
    mask: np.ndarray,
    radial_edges: np.ndarray,
    min_screw_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_bins = len(radial_edges) - 1
    values = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    if not np.any(mask):
        return values, counts

    indices = np.searchsorted(radial_edges, radii[mask], side="right") - 1
    indices = np.clip(indices, 0, n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(np.int64)
    vx2_sums = np.bincount(indices, weights=vx2[mask], minlength=n_bins)
    coupling_sums = np.bincount(indices, weights=coupling[mask], minlength=n_bins)
    values = _screw_from_sums(vx2_sums, coupling_sums, min_screw_rate)
    return values, counts


def _xtheta_screw(
    coords: np.ndarray,
    vx2: np.ndarray,
    coupling: np.ndarray,
    mask: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    min_screw_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_bins = len(x_edges) - 1
    theta_bins = len(theta_edges) - 1
    values = np.full((x_bins, theta_bins), np.nan, dtype=np.float64)
    counts = np.zeros((x_bins, theta_bins), dtype=np.int64)
    if not np.any(mask):
        return values, counts

    box_length_x = float(x_edges[-1] - x_edges[0])
    x_indices = _x_bin_indices(coords[mask, 0], box_length_x, x_bins)
    theta_indices = _theta_bin_indices(coords[mask, 1], theta_bins)
    np.add.at(counts, (x_indices, theta_indices), 1)
    vx2_sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
    coupling_sums = np.zeros((x_bins, theta_bins), dtype=np.float64)
    np.add.at(vx2_sums, (x_indices, theta_indices), vx2[mask])
    np.add.at(coupling_sums, (x_indices, theta_indices), coupling[mask])
    values = _screw_from_sums(vx2_sums, coupling_sums, min_screw_rate)
    return values, counts


def _metric_names_and_labels(lag_frames: tuple[int, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names: list[str] = []
    labels: list[str] = []
    for lag in lag_frames:
        names.extend(
            [
                f"finite_time_lag_{lag}_raw",
                f"finite_time_lag_{lag}_relative",
            ]
        )
        labels.extend(
            [
                f"finite-time helix lag {lag} raw",
                f"finite-time helix lag {lag} relative",
            ]
        )
    names.extend(
        [
            "instant_helix_raw",
            "instant_helix_relative",
            "angular_momentum",
            "angular_velocity",
            "screw",
        ]
    )
    labels.extend(
        [
            "instantaneous helix raw",
            "instantaneous helix relative",
            "angular momentum chirality",
            "angular velocity chirality",
            "screw",
        ]
    )
    return tuple(names), tuple(labels)


def _raw_relative_metric_pairs(
    fields: ChiralityFields,
) -> tuple[list[tuple[int, int, str, str]], set[str]]:
    metric_index = {name: idx for idx, name in enumerate(fields.metric_names)}
    pairs: list[tuple[int, int, str, str]] = []
    paired_names: set[str] = set()
    for lag in fields.lag_frames:
        raw_name = f"finite_time_lag_{lag}_raw"
        relative_name = f"finite_time_lag_{lag}_relative"
        if raw_name in metric_index and relative_name in metric_index:
            pairs.append(
                (
                    metric_index[raw_name],
                    metric_index[relative_name],
                    f"finite_time_lag_{lag}",
                    f"finite-time helix lag {lag}",
                )
            )
            paired_names.update((raw_name, relative_name))

    if "instant_helix_raw" in metric_index and "instant_helix_relative" in metric_index:
        pairs.append(
            (
                metric_index["instant_helix_raw"],
                metric_index["instant_helix_relative"],
                "instant_helix",
                "instantaneous helix",
            )
        )
        paired_names.update(("instant_helix_raw", "instant_helix_relative"))
    return pairs, paired_names


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


def _plot_norm(values: np.ndarray, is_screw: bool):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    if is_screw:
        low, high = np.percentile(finite, [2.0, 98.0])
        limit = max(abs(float(low)), abs(float(high)))
        if np.isclose(limit, 0.0):
            limit = 1.0
        return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

    return TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)


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


def _plot_or_mark_undefined(
    axis,
    steps: np.ndarray,
    values: np.ndarray,
    label: str | None = None,
    linestyle: str = "-",
    undefined_message: str = "undefined",
) -> bool:
    finite = np.isfinite(values)
    if np.any(finite):
        axis.plot(steps, values, label=label, linestyle=linestyle)
        return True

    axis.text(
        0.5,
        0.5,
        undefined_message,
        transform=axis.transAxes,
        ha="center",
        va="center",
        color="0.35",
    )
    return False


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


def save_chirality_fields(
    fields: ChiralityFields,
    filename: str | Path,
    config: ChiralityConfig = ChiralityConfig(),
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        steps=fields.steps,
        metric_names=np.asarray(fields.metric_names),
        metric_labels=np.asarray(fields.metric_labels),
        lag_frames=np.asarray(fields.lag_frames, dtype=np.int64),
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        theta_edges=fields.theta_edges,
        theta_centers=fields.theta_centers,
        radial_edges=fields.radial_edges,
        radial_centers=fields.radial_centers,
        global_values=fields.global_values,
        radial_values=fields.radial_values,
        radial_counts=fields.radial_counts,
        xtheta_values=fields.xtheta_values,
        xtheta_counts=fields.xtheta_counts,
        min_count=np.asarray(config.min_count, dtype=np.int64),
        xtheta_min_count=np.asarray(config.xtheta_min_count, dtype=np.int64),
        screw_min_screw_rate=np.asarray(
            config.screw_min_screw_rate,
            dtype=np.float64,
        ),
    )


def plot_chirality_global(
    fields: ChiralityFields,
    image_dir: str | Path = CHIRALITY_IMAGE_DIR,
) -> None:
    metric_index = {name: idx for idx, name in enumerate(fields.metric_names)}
    output_path = Path(image_dir) / "chirality_global_timeseries.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for lag in fields.lag_frames:
        raw = metric_index[f"finite_time_lag_{lag}_raw"]
        relative = metric_index[f"finite_time_lag_{lag}_relative"]
        axes[0].plot(fields.steps, fields.global_values[raw], label=f"lag {lag} raw")
        axes[0].plot(
            fields.steps,
            fields.global_values[relative],
            linestyle="--",
            label=f"lag {lag} relative",
        )
    axes[0].set_ylabel("finite-time")
    axes[0].legend(loc="best")

    axes[1].plot(
        fields.steps,
        fields.global_values[metric_index["instant_helix_raw"]],
        label="raw",
    )
    axes[1].plot(
        fields.steps,
        fields.global_values[metric_index["instant_helix_relative"]],
        linestyle="--",
        label="relative",
    )
    axes[1].set_ylabel("instant helix")
    axes[1].legend(loc="best")

    axes[2].plot(fields.steps, fields.global_values[metric_index["angular_momentum"]])
    axes[2].set_ylabel("angular momentum")
    axes[3].plot(fields.steps, fields.global_values[metric_index["angular_velocity"]])
    axes[3].set_ylabel("angular velocity")
    _plot_or_mark_undefined(
        axes[4],
        fields.steps,
        fields.global_values[metric_index["screw"]],
        undefined_message="screw undefined\n|screw rate| below cutoff",
    )
    axes[4].set_ylabel("screw")
    axes[4].set_xlabel("Simulation step")

    for axis in axes[:4]:
        axis.axhline(0.0, color="0.35", linewidth=0.9)
        axis.set_ylim(-1.05, 1.05)
        axis.grid(True, linestyle="--", alpha=0.35)
    axes[4].axhline(0.0, color="0.35", linewidth=0.9)
    axes[4].grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("Cylinder chirality diagnostics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_chirality_radial_heatmaps(
    fields: ChiralityFields,
    image_dir: str | Path = CHIRALITY_IMAGE_DIR,
    min_count: int = ChiralityConfig.min_count,
) -> None:
    image_path = Path(image_dir)
    image_path.mkdir(parents=True, exist_ok=True)
    time_edges = _time_edges(fields.steps)
    pairs, paired_names = _raw_relative_metric_pairs(fields)
    for raw_idx, relative_idx, filename_stem, title in pairs:
        raw_values = fields.radial_values[raw_idx].copy()
        relative_values = fields.radial_values[relative_idx].copy()
        raw_values[fields.radial_counts[raw_idx] < min_count] = np.nan
        relative_values[fields.radial_counts[relative_idx] < min_count] = np.nan
        combined_values = np.concatenate((raw_values.ravel(), relative_values.ravel()))
        colormap = plt.get_cmap("coolwarm").copy()
        colormap.set_bad("0.85")
        norm = _plot_norm(combined_values, is_screw=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for axis, values, subtitle in (
            (axes[0], raw_values, "raw"),
            (axes[1], relative_values, "relative"),
        ):
            mesh = axis.pcolormesh(
                time_edges,
                fields.radial_edges,
                np.ma.masked_invalid(values.T),
                shading="auto",
                cmap=colormap,
                norm=norm,
            )
            fig.colorbar(mesh, ax=axis, label=title)
            axis.set_xlabel("Simulation step")
            axis.set_title(subtitle)
        axes[0].set_ylabel("r")
        fig.suptitle(f"{title} by radius (N >= {min_count})")
        fig.tight_layout()
        fig.savefig(image_path / f"chirality_radial_{filename_stem}.png", dpi=200)
        plt.close(fig)

    for metric_idx, (name, label) in enumerate(
        zip(fields.metric_names, fields.metric_labels)
    ):
        if name in paired_names:
            continue
        values = fields.radial_values[metric_idx].copy()
        values[fields.radial_counts[metric_idx] < min_count] = np.nan
        is_screw = name == "screw"
        colormap = plt.get_cmap("coolwarm").copy()
        colormap.set_bad("0.85")

        fig, axis = plt.subplots(figsize=(10, 5))
        mesh = axis.pcolormesh(
            time_edges,
            fields.radial_edges,
            np.ma.masked_invalid(values.T),
            shading="auto",
            cmap=colormap,
            norm=_plot_norm(values, is_screw=is_screw),
        )
        fig.colorbar(mesh, ax=axis, label=label)
        axis.set_xlabel("Simulation step")
        axis.set_ylabel("r")
        axis.set_title(f"{label} by radius (N >= {min_count})")
        fig.tight_layout()
        fig.savefig(image_path / f"chirality_radial_{name}.png", dpi=200)
        plt.close(fig)


def _draw_xtheta_heatmap(
    fields: ChiralityFields,
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
    x_grid, theta_grid = np.meshgrid(
        fields.x_edges,
        fields.theta_edges,
        indexing="ij",
    )
    mesh = axis.pcolormesh(
        x_grid,
        theta_grid,
        np.ma.masked_invalid(values),
        shading="auto",
        cmap=colormap,
        norm=norm,
    )
    fig.colorbar(mesh, ax=axis, label=fields.metric_labels[metric_idx])
    axis.set_xlabel("x")
    _format_theta_axis(axis)
    axis.set_title(
        f"{fields.metric_labels[metric_idx]} in x-theta, step {fields.steps[frame_idx]}"
    )


def _write_metric_movie(
    fields: ChiralityFields,
    metric_idx: int,
    filename: str | Path,
    min_count: int,
    fps: int,
) -> None:
    values = fields.xtheta_values[metric_idx].copy()
    values[fields.xtheta_counts[metric_idx] < min_count] = np.nan
    is_screw = fields.metric_names[metric_idx] == "screw"
    norm = _plot_norm(values, is_screw=is_screw)
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad("0.85")

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


def _draw_xtheta_pair_heatmap(
    fields: ChiralityFields,
    raw_idx: int,
    relative_idx: int,
    frame_idx: int,
    fig,
    axes,
    min_count: int,
    norm,
    colormap,
    title: str,
) -> None:
    for axis, metric_idx, subtitle in (
        (axes[0], raw_idx, "raw"),
        (axes[1], relative_idx, "relative"),
    ):
        values = fields.xtheta_values[metric_idx, frame_idx].copy()
        values[fields.xtheta_counts[metric_idx, frame_idx] < min_count] = np.nan
        x_grid, theta_grid = np.meshgrid(
            fields.x_edges,
            fields.theta_edges,
            indexing="ij",
        )
        mesh = axis.pcolormesh(
            x_grid,
            theta_grid,
            np.ma.masked_invalid(values),
            shading="auto",
            cmap=colormap,
            norm=norm,
        )
        fig.colorbar(mesh, ax=axis, label=title)
        axis.set_xlabel("x")
        _format_theta_axis(axis)
        axis.set_title(subtitle)
    fig.suptitle(f"{title} in x-theta, step {fields.steps[frame_idx]}")


def _write_metric_pair_movie(
    fields: ChiralityFields,
    raw_idx: int,
    relative_idx: int,
    filename: str | Path,
    title: str,
    min_count: int,
    fps: int,
) -> None:
    raw_values = fields.xtheta_values[raw_idx].copy()
    relative_values = fields.xtheta_values[relative_idx].copy()
    raw_values[fields.xtheta_counts[raw_idx] < min_count] = np.nan
    relative_values[fields.xtheta_counts[relative_idx] < min_count] = np.nan
    combined_values = np.concatenate((raw_values.ravel(), relative_values.ravel()))
    norm = _plot_norm(combined_values, is_screw=False)
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad("0.85")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 5))
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, str(output_path), dpi=160):
        for frame_idx in range(len(fields.steps)):
            fig.clear()
            axes = fig.subplots(1, 2, sharey=True)
            _draw_xtheta_pair_heatmap(
                fields,
                raw_idx,
                relative_idx,
                frame_idx,
                fig,
                axes,
                min_count,
                norm,
                colormap,
                title,
            )
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)


def write_chirality_xtheta_movies(
    fields: ChiralityFields,
    image_dir: str | Path = CHIRALITY_IMAGE_DIR,
    min_count: int = ChiralityConfig.xtheta_min_count,
    fps: int = ACTIVE_MOVIE_FPS,
) -> None:
    image_path = Path(image_dir)
    pairs, paired_names = _raw_relative_metric_pairs(fields)
    for raw_idx, relative_idx, filename_stem, title in pairs:
        _write_metric_pair_movie(
            fields,
            raw_idx,
            relative_idx,
            image_path / f"chirality_xtheta_{filename_stem}.mp4",
            title=title,
            min_count=min_count,
            fps=fps,
        )

    for metric_idx, name in enumerate(fields.metric_names):
        if name in paired_names:
            continue
        _write_metric_movie(
            fields,
            metric_idx,
            image_path / f"chirality_xtheta_{name}.mp4",
            min_count=min_count,
            fps=fps,
        )


def write_chirality_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    data_dir: str | Path = CHIRALITY_DATA_DIR,
    image_dir: str | Path = CHIRALITY_IMAGE_DIR,
    config: ChiralityConfig = ChiralityConfig(),
    write_movies: bool = True,
) -> ChiralityFields:
    fields = compute_chirality_fields(input_gsd, config=config)
    save_chirality_fields(
        fields,
        Path(data_dir) / "chirality_fields.npz",
        config=config,
    )
    plot_chirality_global(fields, image_dir=image_dir)
    plot_chirality_radial_heatmaps(
        fields,
        image_dir=image_dir,
        min_count=config.min_count,
    )
    if write_movies:
        write_chirality_xtheta_movies(
            fields,
            image_dir=image_dir,
            min_count=config.xtheta_min_count,
            fps=config.movie_fps,
        )
    write_geometric_chirality_outputs(
        input_gsd,
        data_dir=data_dir,
        image_dir=Path(image_dir) / "geometric",
        config=_geometric_config_from_chirality_config(config),
        write_movies=write_movies,
    )
    return fields
