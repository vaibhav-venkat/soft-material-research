from __future__ import annotations

from pathlib import Path

import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from .best_fit import symbolic_regression_report
from .common import FIT_OUTPUT_DIR, PLOT_OUTPUT_DIR, group_names_for_cases
from .plotting import plot_radius_values

NORMALIZED_PLOT_DIR = PLOT_OUTPUT_DIR / "normalized"
NORMALIZED_FIT_DIR = FIT_OUTPUT_DIR / "normalized"


def normalized_count_plots_missing(metric_name: str) -> bool:
    return any(
        not path.exists()
        for path in (
            NORMALIZED_PLOT_DIR / f"{metric_name}_per_particle.png",
            NORMALIZED_PLOT_DIR / f"{metric_name}_per_surface_area.png",
            NORMALIZED_PLOT_DIR / f"{metric_name}_per_shell_volume.png",
            NORMALIZED_FIT_DIR / f"{metric_name}_per_particle_pysr.txt",
            NORMALIZED_FIT_DIR / f"{metric_name}_per_surface_area_pysr.txt",
            NORMALIZED_FIT_DIR / f"{metric_name}_per_shell_volume_pysr.txt",
        )
    )


def write_normalized_count_plots(
    cases: tuple[RadiusCase, ...],
    values: dict[str, np.ndarray],
    metric_name: str,
    title: str,
) -> None:
    NORMALIZED_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    NORMALIZED_FIT_DIR.mkdir(parents=True, exist_ok=True)

    radii = np.asarray([case.radius for case in cases], dtype=np.float64)
    labels = np.asarray([case.label or case.case_id for case in cases])
    case_ids = np.asarray([case.case_id for case in cases])
    groups = group_names_for_cases(cases)
    regular_mask = (groups != "circumference") | (case_ids == "circ_60_0D")
    if not np.any(regular_mask):
        regular_mask = np.ones(len(cases), dtype=bool)

    n_particles = np.asarray([case.n_particles for case in cases], dtype=np.float64)
    surface_area = np.asarray(
        [2.0 * np.pi * case.radius * case.lx for case in cases],
        dtype=np.float64,
    )
    shell_volume = np.asarray(
        [
            np.pi
            * case.lx
            * (
                case.radius**2
                - max(0.0, case.radius - cylinder.ANALYSIS.wall_cutoff) ** 2
            )
            for case in cases
        ],
        dtype=np.float64,
    )

    radii = radii[regular_mask]
    labels = labels[regular_mask]
    groups = groups[regular_mask]
    per_particle = _filter_values(_divide_values(values, n_particles), regular_mask)
    per_surface_area = _filter_values(_divide_values(values, surface_area), regular_mask)
    per_shell_volume = _filter_values(_divide_values(values, shell_volume), regular_mask)

    particle_plot = NORMALIZED_PLOT_DIR / f"{metric_name}_per_particle.png"
    plot_radius_values(
        radii,
        per_particle,
        particle_plot,
        f"{title} per particle",
        "mean count / N",
        case_labels=labels,
        group_names=groups,
        fits=None,
    )
    symbolic_regression_report(
        radii,
        per_particle,
        NORMALIZED_FIT_DIR / f"{metric_name}_per_particle_pysr.txt",
        title=f"{title} per particle: symbolic regression",
        x_label="R",
    )

    area_plot = NORMALIZED_PLOT_DIR / f"{metric_name}_per_surface_area.png"
    plot_radius_values(
        radii,
        per_surface_area,
        area_plot,
        f"{title} per cylinder surface area",
        r"mean count / $(2\pi R L_x)$",
        case_labels=labels,
        group_names=groups,
        fits=None,
    )
    symbolic_regression_report(
        radii,
        per_surface_area,
        NORMALIZED_FIT_DIR / f"{metric_name}_per_surface_area_pysr.txt",
        title=f"{title} per cylinder surface area: symbolic regression",
        x_label="R",
    )

    shell_volume_plot = NORMALIZED_PLOT_DIR / f"{metric_name}_per_shell_volume.png"
    plot_radius_values(
        radii,
        per_shell_volume,
        shell_volume_plot,
        f"{title} per shell volume",
        r"mean count / $V_{shell}$",
        case_labels=labels,
        group_names=groups,
        fits=None,
    )
    symbolic_regression_report(
        radii,
        per_shell_volume,
        NORMALIZED_FIT_DIR / f"{metric_name}_per_shell_volume_pysr.txt",
        title=f"{title} per shell volume: symbolic regression",
        x_label="R",
    )


def _divide_values(
    values: dict[str, np.ndarray],
    denominator: np.ndarray,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, series in values.items():
        series = np.asarray(series, dtype=np.float64)
        output[name] = np.divide(
            series,
            denominator,
            out=np.full_like(series, np.nan, dtype=np.float64),
            where=np.isfinite(denominator) & (denominator != 0.0),
        )
    return output


def _filter_values(
    values: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(series, dtype=np.float64)[mask]
        for name, series in values.items()
    }
