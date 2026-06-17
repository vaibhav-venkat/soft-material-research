from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from .common import _color_limits, _time_edges
from .config import (
    ACTIVE_IMAGE_DIR,
    ACTIVE_MOVIE_FPS,
    ACTIVE_RADIAL_BIN_WIDTH,
    ACTIVE_RADIAL_MIN_MEAN_COUNT,
    CYLINDER,
    ActiveMatterFields,
)
from .movie_utils import _write_movie


def _radius_edges_and_centers(
    radial_bin_width: float = ACTIVE_RADIAL_BIN_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    assert radial_bin_width > 0.0
    n_bins = int(np.ceil(CYLINDER.cylinder_radius / radial_bin_width))
    edges = radial_bin_width * np.arange(n_bins + 1)
    edges[-1] = CYLINDER.cylinder_radius
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _radius_bin_indices(radii: np.ndarray, radius_edges: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(radius_edges, radii, side="right") - 1
    return np.clip(indices, 0, len(radius_edges) - 2)


def _radial_px_series(
    fields: ActiveMatterFields,
    statistic: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert statistic in {"mean", "mean_abs", "signed_mean", "sum"}
    radius_edges, radius_centers = _radius_edges_and_centers()
    n_radius_bins = len(radius_centers)
    values = np.zeros((len(fields.steps), n_radius_bins), dtype=np.float64)

    for frame_idx in range(len(fields.steps)):
        radii = fields.coords[frame_idx, :, 2]
        valid = (radii >= radius_edges[0]) & (radii <= radius_edges[-1])
        bin_indices = _radius_bin_indices(radii[valid], radius_edges)
        px = fields.active_direction[frame_idx, valid, 0]
        sums = np.bincount(bin_indices, weights=px, minlength=n_radius_bins)
        if statistic in {"mean", "mean_abs", "signed_mean"}:
            counts = np.bincount(bin_indices, minlength=n_radius_bins)
            weights = np.abs(px) if statistic == "mean_abs" else px
            sums = np.bincount(bin_indices, weights=weights, minlength=n_radius_bins)
            mean_px = np.divide(
                sums,
                counts,
                out=np.full(n_radius_bins, np.nan, dtype=np.float64),
                where=counts >= ACTIVE_RADIAL_MIN_MEAN_COUNT,
            )
            if statistic == "mean_abs":
                values[frame_idx] = mean_px
            elif statistic == "signed_mean":
                values[frame_idx] = mean_px
            else:
                values[frame_idx] = np.abs(mean_px)
        else:
            values[frame_idx] = np.abs(sums)

    return radius_edges, radius_centers, values


def _radial_px_labels(statistic: str) -> tuple[str, str]:
    assert statistic in {"mean", "mean_abs", "signed_mean", "sum"}
    if statistic == "mean_abs":
        return "mean |P_x|", "Raw mean |P_x| by radius"
    if statistic == "signed_mean":
        return "mean P_x", "Raw signed mean P_x by radius"
    if statistic == "mean":
        return "|mean P_x|", "Raw |mean P_x| by radius"
    return "|sum P_x|", "Raw |sum P_x| by radius"

def _plot_radial_px_heatmap(
    fields: ActiveMatterFields,
    filename: str | Path,
    statistic: str,
) -> None:
    radius_edges, _, values = _radial_px_series(fields, statistic=statistic)
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _color_limits(values)
    label, title = _radial_px_labels(statistic)

    fig, axis = plt.subplots(figsize=(10, 5))
    if statistic == "signed_mean":
        limit = max(abs(vmin), abs(vmax))
        if np.isclose(limit, 0.0):
            limit = 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        colormap = plt.get_cmap("coolwarm").copy()
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap("magma").copy()
    colormap.set_bad("0.85")
    mesh = axis.pcolormesh(
        _time_edges(fields.steps),
        radius_edges,
        np.ma.masked_invalid(values.T),
        shading="auto",
        cmap=colormap,
        norm=norm,
    )
    fig.colorbar(mesh, ax=axis, label=label)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("r")
    if statistic in {"mean", "mean_abs", "signed_mean"}:
        title += f" (N >= {ACTIVE_RADIAL_MIN_MEAN_COUNT})"
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _radial_px_limits(values: np.ndarray) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None

    low, high = np.percentile(finite, [1.0, 99.0])
    if np.isclose(low, high):
        low = float(np.min(finite))
        high = float(np.max(finite))
    if np.isclose(low, high):
        high = low + 1.0
    pad = 0.08 * (high - low)
    return float(low - pad), float(high + pad)


def _draw_radial_px_profile(
    fields: ActiveMatterFields,
    fig,
    axis,
    frame_idx: int,
    radius_centers: np.ndarray,
    values: np.ndarray,
    statistic: str,
    y_limits: tuple[float, float] | None,
) -> None:
    ylabel, title = _radial_px_labels(statistic)
    axis.plot(radius_centers, values[frame_idx], marker="o", linewidth=1.5)
    axis.set_xlabel("r")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title}, step {fields.steps[frame_idx]}")
    axis.grid(True, ls="--", alpha=0.35)
    if y_limits is not None:
        axis.set_ylim(*y_limits)


def _write_radial_px_movie(
    fields: ActiveMatterFields,
    filename: str | Path,
    statistic: str,
    fps: int = ACTIVE_MOVIE_FPS,
) -> None:
    _, radius_centers, values = _radial_px_series(fields, statistic=statistic)
    y_limits = _radial_px_limits(values)
    _write_movie(
        fields,
        filename,
        lambda fig, axis, frame_idx: _draw_radial_px_profile(
            fields,
            fig,
            axis,
            frame_idx,
            radius_centers,
            values,
            statistic,
            y_limits,
        ),
        fps=fps,
    )


def plot_radial_px_fields(
    fields: ActiveMatterFields,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    image_path = Path(image_dir)
    _plot_radial_px_heatmap(
        fields,
        image_path / "active_px_radius_mean_heatmap.png",
        statistic="mean",
    )
    _plot_radial_px_heatmap(
        fields,
        image_path / "active_px_radius_mean_abs_heatmap.png",
        statistic="mean_abs",
    )
    _plot_radial_px_heatmap(
        fields,
        image_path / "active_px_radius_signed_mean_heatmap.png",
        statistic="signed_mean",
    )
    _plot_radial_px_heatmap(
        fields,
        image_path / "active_px_radius_sum_heatmap.png",
        statistic="sum",
    )
