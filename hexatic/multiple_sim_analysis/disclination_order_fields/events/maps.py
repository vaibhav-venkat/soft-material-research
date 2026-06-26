from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .io import event_plot_path


def _resolve_output_path(filename: str | Path) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = event_plot_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _frame_offset(frame_axis: np.ndarray, frame: int) -> int:
    matches = np.flatnonzero(np.asarray(frame_axis, dtype=np.int64) == int(frame))
    if matches.size == 0:
        raise ValueError(f"frame {frame} is not present in frame_axis")
    return int(matches[0])


def _event_positions_for_frame(
    event_values: Mapping[str, np.ndarray] | None,
    event_kind: str,
    frame: int,
) -> np.ndarray:
    if event_values is None:
        return np.full((0, 3), np.nan, dtype=np.float64)
    frames_key = f"{event_kind}_frames"
    if frames_key not in event_values:
        return np.full((0, 3), np.nan, dtype=np.float64)
    mask = np.asarray(event_values[frames_key], dtype=np.int64) == int(frame)
    positions_key = f"{event_kind}_positions"
    if positions_key in event_values:
        return np.asarray(event_values[positions_key], dtype=np.float64)[mask]
    names = (f"{event_kind}_x", f"{event_kind}_theta", f"{event_kind}_r")
    if all(name in event_values for name in names):
        return np.column_stack([np.asarray(event_values[name], dtype=np.float64)[mask] for name in names])
    return np.full((0, 3), np.nan, dtype=np.float64)


def _scatter_defects(axis, defect_positions: np.ndarray, defect_charges: np.ndarray) -> None:
    if defect_positions.size == 0:
        return
    plus = np.asarray(defect_charges) > 0
    minus = np.asarray(defect_charges) < 0
    axis.scatter(
        defect_positions[plus, 0],
        defect_positions[plus, 1],
        marker="+",
        s=90,
        linewidths=1.6,
        color="#d55e00",
        label="+ defect",
    )
    axis.scatter(
        defect_positions[minus, 0],
        defect_positions[minus, 1],
        marker="x",
        s=70,
        linewidths=1.4,
        color="#0072b2",
        label="- defect",
    )


def plot_frame_map(
    coords: np.ndarray,
    *,
    frame_axis: np.ndarray,
    frame: int,
    output_png: str | Path,
    scalar_fields: Mapping[str, np.ndarray] | None = None,
    defect_positions: np.ndarray | None = None,
    defect_charges: np.ndarray | None = None,
    event_values: Mapping[str, np.ndarray] | None = None,
    field_names: Sequence[str] | None = None,
    title: str | None = None,
) -> Path:
    """Plot representative-frame particle, field, defect, and event maps.

    All inputs are already-loaded arrays. ``coords`` must use the cylinder map
    coordinates ``(x, theta, r)``; the plot unfolds the cylinder as ``x`` vs
    ``theta`` so no trajectory or cached-field loading happens here.
    """
    import matplotlib.pyplot as plt

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] < 2:
        raise ValueError("coords must have shape (frames, particles, 3)")
    offset = _frame_offset(frame_axis, frame)
    frame_coords = coords[offset]
    scalar_fields = {} if scalar_fields is None else scalar_fields
    if field_names is None:
        field_names = tuple(scalar_fields)
    field_names = tuple(field_names)
    panels = ("particles",) + field_names
    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.0 * n_cols, 4.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    defects = (
        np.full((0, 3), np.nan, dtype=np.float64)
        if defect_positions is None
        else np.asarray(defect_positions, dtype=np.float64)
    )
    charges = (
        np.asarray([], dtype=np.int64)
        if defect_charges is None
        else np.asarray(defect_charges, dtype=np.int64)
    )
    births = _event_positions_for_frame(event_values, "birth", frame)
    deaths = _event_positions_for_frame(event_values, "death", frame)

    for axis_idx, panel_name in enumerate(panels):
        axis = axes_flat[axis_idx]
        if panel_name == "particles":
            axis.scatter(frame_coords[:, 0], frame_coords[:, 1], s=8, color="#555555", alpha=0.45)
        else:
            values = np.asarray(scalar_fields[panel_name], dtype=np.float64)
            if values.shape[:2] != coords.shape[:2]:
                raise ValueError(f"{panel_name} must have shape (frames, particles)")
            scatter = axis.scatter(
                frame_coords[:, 0],
                frame_coords[:, 1],
                c=values[offset],
                s=10,
                cmap="viridis",
                alpha=0.88,
            )
            fig.colorbar(scatter, ax=axis, shrink=0.84, label=panel_name)
        _scatter_defects(axis, defects, charges)
        if births.size:
            axis.scatter(
                births[:, 0],
                births[:, 1],
                marker="o",
                facecolors="none",
                edgecolors="#009e73",
                linewidths=1.5,
                s=120,
                label="birth",
            )
        if deaths.size:
            axis.scatter(
                deaths[:, 0],
                deaths[:, 1],
                marker="s",
                facecolors="none",
                edgecolors="#cc79a7",
                linewidths=1.5,
                s=105,
                label="death",
            )
        axis.set_title(panel_name)
        axis.set_xlabel("x")
        axis.set_ylabel("theta")
        axis.grid(True, linestyle="--", alpha=0.18)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), fontsize=7, loc="best")

    for axis in axes_flat[n_panels:]:
        axis.set_visible(False)

    fig.suptitle(title or f"Representative event map, frame {frame}")
    output_path = _resolve_output_path(output_png)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_representative_frame_maps(
    coords: np.ndarray,
    *,
    frame_axis: np.ndarray,
    frames: Sequence[int],
    output_dir: str | Path = "maps",
    scalar_fields: Mapping[str, np.ndarray] | None = None,
    defects_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]] | None = None,
    event_values: Mapping[str, np.ndarray] | None = None,
    field_names: Sequence[str] | None = None,
) -> list[Path]:
    """Write one representative map per requested frame from supplied arrays."""
    output_dir_path = Path(output_dir)
    written: list[Path] = []
    for frame in frames:
        defect_positions = None
        defect_charges = None
        if defects_by_frame is not None and int(frame) in defects_by_frame:
            defect_positions, defect_charges = defects_by_frame[int(frame)]
        written.append(
            plot_frame_map(
                coords,
                frame_axis=frame_axis,
                frame=int(frame),
                output_png=output_dir_path / f"frame_{int(frame):05d}.png",
                scalar_fields=scalar_fields,
                defect_positions=defect_positions,
                defect_charges=defect_charges,
                event_values=event_values,
                field_names=field_names,
            )
        )
    return written


def compute_event_maps(*args, **kwargs):
    """Compatibility wrapper for representative-frame map plotting."""
    return plot_representative_frame_maps(*args, **kwargs)
