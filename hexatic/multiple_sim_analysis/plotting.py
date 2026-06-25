from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hexatic.constants import cylinder

from .best_fit import ExponentialFit, FitCurve, fit_exponential_radius_trend
from .common import (
    PLOT_OUTPUT_DIR,
    ensure_output_dirs,
    group_names_for_cases,
)

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
                label=f"{series_name} {fit_label}",
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


def plots_missing(cases, output_png: str | Path) -> bool:
    output_path = Path(output_png)
    if not output_path.is_absolute():
        output_path = PLOT_OUTPUT_DIR / output_path
    if not output_path.exists():
        return True
    groups = group_names_for_cases(cases)
    return bool(np.any(groups == "circumference")) and not companion_circumference_plot_path(
        output_path
    ).exists()


def _filter_values(
    values: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(series, dtype=np.float64)[mask]
        for name, series in values.items()
    }


def _exponential_fits(
    radii: np.ndarray,
    values: dict[str, np.ndarray],
) -> dict[str, ExponentialFit | None]:
    return {
        name: fit_exponential_radius_trend(radii, series)
        for name, series in values.items()
    }


def _linear_fit_curve(x_values: np.ndarray, y_values: np.ndarray) -> FitCurve | None:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    fit_x = x_values[finite]
    fit_y = y_values[finite]
    if fit_x.size < 2 or np.ptp(fit_x) <= 0.0:
        return None
    slope, intercept = np.polyfit(fit_x, fit_y, deg=1)
    curve_x = np.linspace(float(np.min(fit_x)), float(np.max(fit_x)), 200)
    return FitCurve(
        radii=curve_x,
        values=slope * curve_x + intercept,
    )


def _linear_fits(
    x_values: np.ndarray,
    values: dict[str, np.ndarray],
) -> dict[str, FitCurve | None]:
    return {
        name: _linear_fit_curve(x_values, series)
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
        fits=_exponential_fits(regular_radii, regular_values),
    )

    circumference_mask = group_names == "circumference"
    if not np.any(circumference_mask):
        return
    circumference_values = _filter_values(values, circumference_mask)
    circumference_x = (
        2.0 * np.pi * radii[circumference_mask] / cylinder.PARTICLE_DIAMETER
    )
    plot_radius_values(
        circumference_x,
        circumference_values,
        companion_circumference_plot_path(output_png),
        f"{title} (circumference sweep)",
        ylabel,
        case_labels=labels[circumference_mask],
        group_names=group_names[circumference_mask],
        fits=_linear_fits(circumference_x, circumference_values),
        xlabel="Circumference C / D",
        fit_label="linear fit",
    )
