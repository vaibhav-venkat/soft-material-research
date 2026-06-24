from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .best_fit import ExponentialFit, FitCurve
from .common import PLOT_OUTPUT_DIR, ensure_output_dirs, group_names_for_cases

GROUP_STYLES = {
    "circumference": {"marker": "o", "color": "#1f77b4"},
    "scaled_radius": {"marker": "s", "color": "#d62728"},
    "other": {"marker": "^", "color": "#2ca02c"},
}


def plot_radius_values(
    radii: np.ndarray,
    values: dict[str, np.ndarray],
    output_png: str | Path,
    title: str,
    ylabel: str,
    case_labels: np.ndarray | None = None,
    group_names: np.ndarray | None = None,
    fits: dict[str, ExponentialFit | FitCurve | None] | None = None,
) -> None:
    ensure_output_dirs()
    radii = np.asarray(radii, dtype=np.float64)
    output_path = Path(output_png)
    if not output_path.is_absolute():
        output_path = PLOT_OUTPUT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if group_names is None:
        group_names = np.asarray(["other"] * radii.size)
    if case_labels is None:
        case_labels = np.asarray([""] * radii.size)

    fig, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    series_colors = ["#111111", "#0072b2", "#d55e00", "#009e73"]

    for series_idx, (series_name, series_values) in enumerate(values.items()):
        series_values = np.asarray(series_values, dtype=np.float64)
        base_color = series_colors[series_idx % len(series_colors)]
        for group_name in np.unique(group_names):
            mask = group_names == group_name
            style = GROUP_STYLES.get(str(group_name), GROUP_STYLES["other"])
            axis.scatter(
                radii[mask],
                series_values[mask],
                label=f"{series_name} ({group_name})",
                marker=style["marker"],
                facecolor=base_color,
                edgecolor=style["color"],
                linewidth=1.2,
                s=58,
                alpha=0.92,
            )
        if fits is not None and fits.get(series_name) is not None:
            fit = fits[series_name]
            assert fit is not None
            axis.plot(
                fit.radii,
                fit.values,
                color=base_color,
                linestyle="--",
                linewidth=1.6,
                label=f"{series_name} exp fit",
            )

    for radius, label in zip(radii, case_labels):
        axis.annotate(
            str(label),
            (radius, axis.get_ylim()[0]),
            xytext=(0, -20),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            rotation=35,
            alpha=0.75,
            clip_on=False,
        )

    axis.set_xlabel("Cylinder radius R")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.28)
    axis.legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_for_cases(
    cases,
    values: dict[str, np.ndarray],
    output_png: str | Path,
    title: str,
    ylabel: str,
    fits: dict[str, ExponentialFit | FitCurve | None] | None = None,
) -> None:
    radii = np.asarray([case.radius for case in cases], dtype=np.float64)
    labels = np.asarray([case.label or case.case_id for case in cases])
    plot_radius_values(
        radii,
        values,
        output_png,
        title,
        ylabel,
        case_labels=labels,
        group_names=group_names_for_cases(cases),
        fits=fits,
    )
