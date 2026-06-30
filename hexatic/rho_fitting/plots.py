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
    rho: np.ndarray,
    partial_t_rho: np.ndarray,
    temporal_power: np.ndarray,
    cheb_cutoff: int,
    lx: float,
    ly: float,
) -> None:
    plt = _plotter(output_dir)
    _coefficient_plot(output_dir / f"{case_id}_density_coefficients.png", fit, plt)
    _importance_plot(output_dir / f"{case_id}_density_importance_scores.png", fit, plt)
    _predicted_true_plot(output_dir / f"{case_id}_density_true_vs_predicted.png", fit, y_true, plt)
    _predicted_true_plot(output_dir / f"{case_id}_density_predicted_vs_true.png", fit, y_true, plt)
    write_temporal_power_plot(output_dir, case_id, temporal_power, cheb_cutoff)
    write_temporal_power_plot(output_dir, f"{case_id}_temporal_map", temporal_power, cheb_cutoff)
    predicted = density_prediction_fields(rho, fit.names, fit.coefficients, lx, ly)
    residual = partial_t_rho - predicted
    _kymograph(output_dir / f"{case_id}_rho_kymograph.png", rho, "rho", plt)
    _kymograph(output_dir / f"{case_id}_partial_t_rho_kymograph.png", partial_t_rho, "partial_t rho", plt)
    _kymograph(output_dir / f"{case_id}_predicted_partial_t_rho_kymograph.png", predicted, "predicted partial_t rho", plt)
    _kymograph(output_dir / f"{case_id}_density_residual_kymograph.png", residual, "density residual", plt)
    _heatmap(output_dir / f"{case_id}_density_residual_heatmap.png", np.mean(residual, axis=0), "mean residual", plt)
    _heatmap(
        output_dir / f"{case_id}_density_abs_residual_heatmap.png",
        np.mean(np.abs(residual), axis=0),
        "mean abs residual",
        plt,
    )
    _heatmap(output_dir / f"{case_id}_rho_snapshot.png", rho[rho.shape[0] // 2], "rho snapshot", plt)
    _heatmap(
        output_dir / f"{case_id}_partial_t_rho_snapshot.png",
        partial_t_rho[partial_t_rho.shape[0] // 2],
        "partial_t rho snapshot",
        plt,
    )
    _heatmap(
        output_dir / f"{case_id}_predicted_partial_t_rho_snapshot.png",
        predicted[predicted.shape[0] // 2],
        "predicted partial_t rho snapshot",
        plt,
    )


def write_temporal_power_plot(
    output_dir: Path,
    case_id: str,
    temporal_power: np.ndarray,
    cheb_cutoff: int,
) -> Path:
    plt = _plotter(output_dir)
    suffix = "" if case_id.endswith("_temporal_map") else "_temporal_power"
    path = output_dir / f"{case_id}{suffix}_chebyshev.png"
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


def _importance_plot(path: Path, fit: StabilityResult, plt) -> None:
    tau = np.asarray(fit.tau_values, dtype=np.float64)
    tau_max = tau[0] if tau.size and tau[0] != 0.0 else 1.0
    x = -np.log10(np.maximum(tau / tau_max, np.finfo(float).tiny))
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    for idx, label in enumerate(fit.labels):
        axis.plot(x, fit.importance_path[:, idx], marker="o", markersize=2.5, linewidth=1.2, label=label)
    axis.axhline(0.6, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("-log10(tau / tau_max)")
    axis.set_ylabel("importance score")
    axis.set_ylim(-0.03, 1.03)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
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


def density_prediction_fields(
    rho: np.ndarray,
    term_names: tuple[str, ...],
    coefficients: np.ndarray,
    lx: float,
    ly: float,
) -> np.ndarray:
    predicted = np.zeros_like(rho, dtype=np.float64)
    lap_cache: dict[int, np.ndarray] = {}
    for name, coefficient in zip(term_names, coefficients, strict=True):
        if coefficient == 0.0:
            continue
        if name == "rho":
            field = rho
        elif name.startswith("rho"):
            field = rho ** int(name.removeprefix("rho"))
        else:
            order = 1 if name == "lap_rho" else int(name.removeprefix("lap").removesuffix("_rho"))
            field = lap_cache.setdefault(order, _repeated_laplacian(rho, lx, ly, order))
        predicted += coefficient * field
    return predicted


def _repeated_laplacian(field: np.ndarray, lx: float, ly: float, order: int) -> np.ndarray:
    _, nx, ny = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    k2 = kx[:, np.newaxis] ** 2 + ky[np.newaxis, :] ** 2
    spectrum = np.fft.fft2(field, axes=(1, 2))
    return np.fft.ifft2(((-k2) ** order)[np.newaxis, :, :] * spectrum, axes=(1, 2)).real


def _kymograph(path: Path, field: np.ndarray, title: str, plt) -> None:
    values = np.mean(field, axis=2)
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    image = axis.imshow(values, aspect="auto", origin="lower", interpolation="nearest")
    axis.set_xlabel("x bin")
    axis.set_ylabel("frame")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _heatmap(path: Path, field: np.ndarray, title: str, plt) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    image = axis.imshow(field.T, aspect="auto", origin="lower", interpolation="nearest")
    axis.set_xlabel("x bin")
    axis.set_ylabel("theta bin")
    axis.set_title(title)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
