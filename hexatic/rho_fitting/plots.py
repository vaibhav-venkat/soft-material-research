"""Density fitting plots."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .regression import StabilityResult


def write_density_plots(
    output_dir: Path,
    case_id: str,
    fit: StabilityResult,
    y_true: np.ndarray,
    temporal_power: np.ndarray,
    cheb_cutoff: int,
) -> None:
    plt = _plotter(output_dir)
    _coefficient_plot(output_dir / f"{case_id}_density_coefficients.png", fit, plt)
    _predicted_true_plot(output_dir / f"{case_id}_density_predicted_vs_true.png", fit, y_true, plt)
    write_temporal_power_plot(output_dir, case_id, temporal_power, cheb_cutoff)


def write_temporal_power_plot(
    output_dir: Path,
    case_id: str,
    temporal_power: np.ndarray,
    cheb_cutoff: int,
) -> Path:
    plt = _plotter(output_dir)
    path = output_dir / f"{case_id}_temporal_power_chebyshev.png"
    _temporal_power_plot(path, temporal_power, cheb_cutoff, plt)
    return path


def _plotter(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_dir / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    return plt


def _coefficient_plot(path: Path, fit: StabilityResult, plt) -> None:
    x = np.arange(len(fit.labels))
    fig, axis = plt.subplots(figsize=(max(6.0, 0.7 * len(fit.labels)), 4.0))
    axis.bar(x, fit.coefficients, color=np.where(fit.active, "#2b6cb0", "#a0aec0"))
    axis.set_xticks(x)
    axis.set_xticklabels(fit.labels, rotation=35, ha="right")
    axis.set_ylabel("coefficient")
    twin = axis.twinx()
    twin.plot(x, fit.importance, color="#c53030", marker="o", linewidth=1.5)
    twin.set_ylim(0.0, 1.05)
    twin.set_ylabel("importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _predicted_true_plot(path: Path, fit: StabilityResult, y_true: np.ndarray, plt) -> None:
    max_points = min(20_000, y_true.size)
    if y_true.size > max_points:
        rows = np.linspace(0, y_true.size - 1, max_points, dtype=int)
        y_plot = y_true[rows]
        pred_plot = fit.y_pred[rows]
    else:
        y_plot = y_true
        pred_plot = fit.y_pred
    fig, axis = plt.subplots(figsize=(4.5, 4.5))
    axis.scatter(y_plot, pred_plot, s=3, alpha=0.35, linewidths=0)
    lo = float(min(np.min(y_plot), np.min(pred_plot)))
    hi = float(max(np.max(y_plot), np.max(pred_plot)))
    axis.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    axis.set_xlabel("true partial_t rho")
    axis.set_ylabel("predicted partial_t rho")
    axis.set_title(f"R2 = {fit.r2:.3g}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _temporal_power_plot(path: Path, temporal_power: np.ndarray, cheb_cutoff: int, plt) -> None:
    modes = np.arange(temporal_power.size)
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.semilogy(modes, temporal_power, marker="o", markersize=3, linewidth=1.2)
    axis.axvline(cheb_cutoff - 1, color="#c53030", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Chebyshev mode n")
    axis.set_ylabel("temporal spectral power")
    axis.set_title("rho temporal spectrum")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
