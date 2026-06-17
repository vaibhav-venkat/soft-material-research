from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

try:
    from hexatic.active_matter_cylinder import (
        _minimum_image_delta,
        _theta_bin_indices,
        _x_bin_indices,
    )
except ImportError:
    from active_matter_cylinder import (
        _minimum_image_delta,
        _theta_bin_indices,
        _x_bin_indices,
    )

from .config import CYLINDER, ChiralityFields, NeighborCountMatrix


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


def _load_neighbor_count_matrix(filename: str | Path) -> NeighborCountMatrix:
    table = np.loadtxt(filename, dtype=np.int64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    assert table.ndim == 2 and table.shape[1] >= 4

    frame_indices = table[:, 0]
    step_values = table[:, 1]
    particle_indices = table[:, 2]
    neighbor_counts = table[:, 3]

    n_frames = int(frame_indices.max()) + 1
    n_particles = int(particle_indices.max()) + 1
    flat_indices = frame_indices * n_particles + particle_indices
    assert np.unique(flat_indices).size == flat_indices.size

    steps = np.full(n_frames, -1, dtype=np.int64)
    counts = np.full((n_frames, n_particles), -1, dtype=np.int64)
    counts[frame_indices, particle_indices] = neighbor_counts
    for frame_idx in range(n_frames):
        frame_steps = np.unique(step_values[frame_indices == frame_idx])
        assert frame_steps.size == 1
        steps[frame_idx] = frame_steps[0]

    assert np.all(counts >= 0)
    return NeighborCountMatrix(steps=steps, counts=counts)


def _cylinder_shell_mask(radii: np.ndarray) -> np.ndarray:
    return (
        (radii > CYLINDER.cylinder_radius - CYLINDER.wall_cutoff)
        & (radii < CYLINDER.cylinder_radius)
    )


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
