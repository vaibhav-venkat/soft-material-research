"""Report writing for rho fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .regression import StabilityResult


def write_report(path: Path, lines: list[str], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def density_report_lines(
    *,
    case_id: str,
    nd: int,
    frames: int,
    grid_shape: tuple[int, int],
    sigma: float,
    cheb_cutoff: int,
    n_rho_power: int,
    n_rho_lap_power: int,
    fit: StabilityResult,
    warnings: tuple[str, ...] = (),
) -> list[str]:
    lines = [
        f"# Rho fitting report: {case_id}",
        "",
        "## Settings",
        f"- frames: {frames}",
        f"- grid: {grid_shape[0]} x {grid_shape[1]}",
        f"- samples: {nd}",
        f"- sigma: {sigma:.8g}",
        f"- cheb_cutoff: {cheb_cutoff}",
        f"- n_rho_power: {n_rho_power}",
        f"- n_rho_lap_power: {n_rho_lap_power}",
        "",
        "## Fit",
        f"- rmse: {fit.rmse:.8g}",
        f"- r2: {fit.r2:.8g}",
        f"- tau_index: {fit.tau_index if fit.tau_index is not None else 'none'}",
        "",
    ]
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(
        [
            "## Coefficients",
            "| term | active | coefficient | importance | corr(partial_t rho) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, active, coefficient, importance, raw_correlation in zip(
        fit.labels,
        fit.active,
        fit.coefficients,
        fit.importance,
        fit.raw_correlations,
        strict=True,
    ):
        lines.append(
            f"| {label} | {int(active)} | {coefficient:.10g} | "
            f"{importance:.4f} | {_format_float(raw_correlation)} |"
        )
    if not np.any(fit.active):
        lines.extend(["", "No terms passed the configured importance threshold."])
    return lines


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"
