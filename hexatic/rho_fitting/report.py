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
        "## Coefficients",
        "| term | active | coefficient | importance |",
        "|---|---:|---:|---:|",
    ]
    for label, active, coefficient, importance in zip(
        fit.labels,
        fit.active,
        fit.coefficients,
        fit.importance,
        strict=True,
    ):
        lines.append(
            f"| {label} | {int(active)} | {coefficient:.10g} | {importance:.4f} |"
        )
    if not np.any(fit.active):
        lines.extend(["", "No terms passed the configured importance threshold."])
    return lines
