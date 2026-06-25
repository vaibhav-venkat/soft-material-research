from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hexatic.constants import cylinder

from .best_fit import ExponentialFit, FitCurve, symbolic_regression_report
from .common import (
    FIT_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    ensure_output_dirs,
    group_names_for_cases,
)

def plot_radius_values(
    radii: np.ndarray,
    values: dict[str, np.ndarray],
    output_png: str | Path,
    title: str,
    ylabel: str,
    case_labels: np.ndarray | None = None,
    group_names: np.ndarray | None = None,
    fits: dict[str, ExponentialFit | FitCurve | None] | None = None,
    xlabel: str = "Cylinder radius R",
    fit_label: str = "exp fit",
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
        axis.scatter(
            radii,
            series_values,
            label=series_name,
            marker="s",
            facecolor=base_color,
            edgecolor="#222222",
            linewidth=0.8,
            s=58,
            alpha=0.92,
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

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.28)
    axis.legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def companion_circumference_plot_path(output_png: str | Path) -> Path:
    output_path = Path(output_png)
    if not output_path.is_absolute():
        output_path = PLOT_OUTPUT_DIR / output_path
    return output_path.with_name(f"{output_path.stem}_circumference{output_path.suffix}")


def symbolic_report_path(output_png: str | Path) -> Path:
    output_path = Path(output_png)
    if not output_path.is_absolute():
        output_path = PLOT_OUTPUT_DIR / output_path
    return FIT_OUTPUT_DIR / f"{output_path.stem}_pysr.txt"


def plots_missing(cases, output_png: str | Path) -> bool:
    output_path = Path(output_png)
    if not output_path.is_absolute():
        output_path = PLOT_OUTPUT_DIR / output_path
    if not output_path.exists():
        return True
    if not symbolic_report_path(output_path).exists():
        return True
    groups = group_names_for_cases(cases)
    if not bool(np.any(groups == "circumference")):
        return False
    circumference_plot = companion_circumference_plot_path(output_path)
    return (
        not circumference_plot.exists()
        or not symbolic_report_path(circumference_plot).exists()
    )


def _filter_values(
    values: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(series, dtype=np.float64)[mask]
        for name, series in values.items()
    }


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
    case_ids = np.asarray([case.case_id for case in cases])
    group_names = group_names_for_cases(cases)
    regular_mask = (group_names != "circumference") | (case_ids == "circ_60_0D")
    if not np.any(regular_mask):
        regular_mask = np.ones(len(cases), dtype=bool)

    regular_radii = radii[regular_mask]
    regular_values = _filter_values(values, regular_mask)
    plot_radius_values(
        regular_radii,
        regular_values,
        output_png,
        title,
        ylabel,
        case_labels=labels[regular_mask],
        group_names=group_names[regular_mask],
        fits=None,
    )
    symbolic_regression_report(
        regular_radii,
        regular_values,
        symbolic_report_path(output_png),
        title=f"{title}: symbolic regression",
        x_label="R",
    )

    circumference_mask = group_names == "circumference"
    if not np.any(circumference_mask):
        return
    circumference_values = _filter_values(values, circumference_mask)
    circumference_x = (
        2.0 * np.pi * radii[circumference_mask] / cylinder.PARTICLE_DIAMETER
    )
    circumference_plot_path = companion_circumference_plot_path(output_png)
    plot_radius_values(
        circumference_x,
        circumference_values,
        circumference_plot_path,
        f"{title} (circumference sweep)",
        ylabel,
        case_labels=labels[circumference_mask],
        group_names=group_names[circumference_mask],
        fits=None,
        xlabel="Circumference C / D",
    )
    symbolic_regression_report(
        circumference_x,
        circumference_values,
        symbolic_report_path(circumference_plot_path),
        title=f"{title} (circumference sweep): symbolic regression",
        x_label="C_over_D",
    )
