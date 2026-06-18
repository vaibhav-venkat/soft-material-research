from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

try:
    from hexatic.active_matter_cylinder import ACTIVE_MOVIE_FPS, _time_edges
except ImportError:
    from active_matter_cylinder import ACTIVE_MOVIE_FPS, _time_edges

from .common import (
    _format_theta_axis,
    _plot_norm,
    _plot_or_mark_undefined,
    _raw_relative_metric_pairs,
)
from .config import CHIRALITY_IMAGE_DIR, ChiralityConfig, ChiralityFields


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


def _masked_by_count(values: np.ndarray, counts: np.ndarray, min_count: int) -> np.ndarray:
    masked = values.copy()
    masked[counts < min_count] = np.nan
    return masked


def _bad_coolwarm():
    colormap = plt.get_cmap("coolwarm").copy()
    colormap.set_bad("0.85")
    return colormap


def plot_chirality_global(
    fields: ChiralityFields,
    image_dir: str | Path = CHIRALITY_IMAGE_DIR,
    title: str = "Cylinder chirality diagnostics",
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
    fig.suptitle(title)
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
        raw_values = _masked_by_count(
            fields.radial_values[raw_idx],
            fields.radial_counts[raw_idx],
            min_count,
        )
        relative_values = _masked_by_count(
            fields.radial_values[relative_idx],
            fields.radial_counts[relative_idx],
            min_count,
        )
        combined_values = np.concatenate((raw_values.ravel(), relative_values.ravel()))
        colormap = _bad_coolwarm()
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
        values = _masked_by_count(
            fields.radial_values[metric_idx],
            fields.radial_counts[metric_idx],
            min_count,
        )
        is_screw = name == "screw"
        colormap = _bad_coolwarm()

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
    values = _masked_by_count(
        fields.xtheta_values[metric_idx, frame_idx],
        fields.xtheta_counts[metric_idx, frame_idx],
        min_count,
    )
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
    values = _masked_by_count(
        fields.xtheta_values[metric_idx],
        fields.xtheta_counts[metric_idx],
        min_count,
    )
    is_screw = fields.metric_names[metric_idx] == "screw"
    norm = _plot_norm(values, is_screw=is_screw)
    colormap = _bad_coolwarm()

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
        values = _masked_by_count(
            fields.xtheta_values[metric_idx, frame_idx],
            fields.xtheta_counts[metric_idx, frame_idx],
            min_count,
        )
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
    raw_values = _masked_by_count(
        fields.xtheta_values[raw_idx],
        fields.xtheta_counts[raw_idx],
        min_count,
    )
    relative_values = _masked_by_count(
        fields.xtheta_values[relative_idx],
        fields.xtheta_counts[relative_idx],
        min_count,
    )
    combined_values = np.concatenate((raw_values.ravel(), relative_values.ravel()))
    norm = _plot_norm(combined_values, is_screw=False)
    colormap = _bad_coolwarm()

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
