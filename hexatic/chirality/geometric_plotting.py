from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, TwoSlopeNorm

try:
    from hexatic.active_matter_cylinder import ACTIVE_MOVIE_FPS, _time_edges
    from hexatic.constants import cylinder
except ImportError:
    from constants import cylinder
    from active_matter_cylinder import ACTIVE_MOVIE_FPS, _time_edges

from .geometric_compute import compute_geometric_chirality_fields
from .geometric_config import (
    GEOMETRIC_CHIRALITY_DATA_DIR,
    GEOMETRIC_CHIRALITY_IMAGE_DIR,
    GeometricChiralityConfig,
    GeometricChiralityFields,
)
from .common import _format_theta_axis


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


def _masked_by_count(values: np.ndarray, counts: np.ndarray, min_count: int) -> np.ndarray:
    masked = values.copy()
    masked[counts < min_count] = np.nan
    return masked


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
    title: str = "Geometric chirality diagnostics",
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
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_geometric_chirality_radial_heatmaps(
    fields: GeometricChiralityFields,
    image_dir: str | Path = GEOMETRIC_CHIRALITY_IMAGE_DIR,
    min_count: int = GeometricChiralityConfig.min_count,
    title: str = "Geometric chirality by radius",
) -> None:
    output_path = Path(image_dir) / "geometric_chirality_radial_heatmaps.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    time_edges = _time_edges(fields.steps)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=True)
    for metric_idx, axis in enumerate(axes):
        name = fields.metric_names[metric_idx]
        label = fields.metric_labels[metric_idx]
        values = _masked_by_count(
            fields.radial_values[metric_idx],
            fields.radial_counts[metric_idx],
            min_count,
        )
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
    fig.suptitle(f"{title} (N >= {min_count})")
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
    values = _masked_by_count(
        fields.xtheta_values[metric_idx, frame_idx],
        fields.xtheta_counts[metric_idx, frame_idx],
        min_count,
    )
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
    values = _masked_by_count(
        fields.xtheta_values[metric_idx],
        fields.xtheta_counts[metric_idx],
        min_count,
    )
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
    input_gsd: str | Path = cylinder.PATHS.in_gsd,
    data_dir: str | Path = GEOMETRIC_CHIRALITY_DATA_DIR,
    image_dir: str | Path = GEOMETRIC_CHIRALITY_IMAGE_DIR,
    config: GeometricChiralityConfig = GeometricChiralityConfig(),
    write_movies: bool = True,
    particle_masks: np.ndarray | None = None,
    data_filename: str = "geometric_chirality_fields.npz",
    plot_title: str = "Geometric chirality diagnostics",
    radial_title: str = "Geometric chirality by radius",
) -> GeometricChiralityFields:
    fields = compute_geometric_chirality_fields(
        input_gsd,
        config=config,
        particle_masks=particle_masks,
    )
    save_geometric_chirality_fields(
        fields,
        Path(data_dir) / data_filename,
        config=config,
    )
    plot_geometric_chirality_global(fields, image_dir=image_dir, title=plot_title)
    plot_geometric_chirality_radial_heatmaps(
        fields,
        image_dir=image_dir,
        min_count=config.min_count,
        title=radial_title,
    )
    if write_movies:
        write_geometric_chirality_xtheta_movies(
            fields,
            image_dir=image_dir,
            min_count=config.min_count,
            fps=config.movie_fps,
        )
    return fields
