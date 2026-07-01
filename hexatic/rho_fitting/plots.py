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
    write_temporal_power_plots(output_dir, case_id, temporal_power, cheb_cutoff)

    predicted = density_prediction_fields(rho, fit.names, fit.coefficients, lx, ly)
    residual = partial_t_rho - predicted
    _kymograph(output_dir / f"{case_id}_rho_kymograph.png", rho, "rho", plt)
    _kymograph(output_dir / f"{case_id}_partial_t_rho_kymograph.png", partial_t_rho, "partial_t rho", plt)
    _kymograph(
        output_dir / f"{case_id}_predicted_partial_t_rho_kymograph.png",
        predicted,
        "predicted partial_t rho",
        plt,
    )
    _kymograph(output_dir / f"{case_id}_density_residual_kymograph.png", residual, "density residual", plt)
    _heatmap(output_dir / f"{case_id}_density_residual_heatmap.png", np.mean(residual, axis=0), "mean residual", plt)
    _heatmap(
        output_dir / f"{case_id}_density_abs_residual_heatmap.png",
        np.mean(np.abs(residual), axis=0),
        "mean abs residual",
        plt,
    )
    mid = rho.shape[0] // 2
    _heatmap(output_dir / f"{case_id}_rho_snapshot.png", rho[mid], "rho snapshot", plt)
    _heatmap(output_dir / f"{case_id}_partial_t_rho_snapshot.png", partial_t_rho[mid], "partial_t rho snapshot", plt)
    _heatmap(
        output_dir / f"{case_id}_predicted_partial_t_rho_snapshot.png",
        predicted[mid],
        "predicted partial_t rho snapshot",
        plt,
    )


def write_temporal_power_plots(
    output_dir: Path,
    case_id: str,
    temporal_power: np.ndarray,
    cheb_cutoff: int,
) -> tuple[Path, Path, Path]:
    plt = _plotter(output_dir)
    base = output_dir / f"{case_id}_temporal_power_chebyshev"
    paths = (
        base.with_suffix(".png"),
        output_dir / f"{base.name}_semilog.png",
        output_dir / f"{base.name}_loglog.png",
    )
    _temporal_power_plot(paths[0], temporal_power, cheb_cutoff, plt, scale="linear")
    _temporal_power_plot(paths[1], temporal_power, cheb_cutoff, plt, scale="semilog")
    _temporal_power_plot(paths[2], temporal_power, cheb_cutoff, plt, scale="loglog")
    return paths


def density_prediction_fields(
    rho: np.ndarray,
    term_names: tuple[str, ...],
    coefficients: np.ndarray,
    lx: float,
    ly: float,
) -> np.ndarray:
    predicted = np.zeros_like(rho, dtype=np.float64)
    for name, coefficient in zip(term_names, coefficients, strict=True):
        if coefficient != 0.0:
            predicted += coefficient * _density_term_field(rho, name, lx, ly)
    return predicted


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
    axis.set_xticklabels(fit.labels, rotation=25, ha="right")
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
    rows = np.linspace(0, y_true.size - 1, min(20_000, y_true.size), dtype=int)
    y_plot = y_true[rows]
    pred_plot = fit.y_pred[rows]
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


def _temporal_power_plot(path: Path, temporal_power: np.ndarray, cheb_cutoff: int, plt, *, scale: str) -> None:
    modes = np.arange(temporal_power.size)
    positive_power = np.maximum(temporal_power, np.finfo(float).tiny)
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    if scale == "loglog":
        axis.loglog(modes[1:], positive_power[1:], marker="o", markersize=3, linewidth=1.2)
    elif scale == "semilog":
        axis.semilogy(modes, positive_power, marker="o", markersize=3, linewidth=1.2)
    else:
        axis.plot(modes, temporal_power, marker="o", markersize=3, linewidth=1.2)
    axis.axvline(cheb_cutoff - 1, color="#c53030", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Chebyshev mode n")
    axis.set_ylabel("temporal spectral power")
    axis.set_title(f"rho temporal spectrum ({scale})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _density_term_field(rho: np.ndarray, name: str, lx: float, ly: float) -> np.ndarray:
    if name == "neg_div_grad_rho":
        return -_laplacian(rho, lx, ly)
    if name == "neg_div_grad_lap_rho":
        return -_laplacian(_laplacian(rho, lx, ly), lx, ly)
    if name == "neg_div_lap_rho_grad_rho":
        return -_divergence(_laplacian(rho, lx, ly)[..., np.newaxis] * _gradient(rho, lx, ly), lx, ly)
    if name == "neg_div_grad_rho_cubed":
        grad_rho = _gradient(rho, lx, ly)
        return -_divergence(np.sum(grad_rho * grad_rho, axis=3, keepdims=True) * grad_rho, lx, ly)
    assert False, f"unknown density term for plotting: {name}"


def _gradient(field: np.ndarray, lx: float, ly: float) -> np.ndarray:
    _, nx, ny = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    spectrum = np.fft.fft2(field, axes=(1, 2))
    dx = np.fft.ifft2(1j * kx[np.newaxis, :, np.newaxis] * spectrum, axes=(1, 2)).real
    dy = np.fft.ifft2(1j * ky[np.newaxis, np.newaxis, :] * spectrum, axes=(1, 2)).real
    return np.stack((dx, dy), axis=3)


def _divergence(field: np.ndarray, lx: float, ly: float) -> np.ndarray:
    _, nx, ny, _ = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    spectrum_x = np.fft.fft2(field[..., 0], axes=(1, 2))
    spectrum_y = np.fft.fft2(field[..., 1], axes=(1, 2))
    div_spectrum = (
        1j * kx[np.newaxis, :, np.newaxis] * spectrum_x
        + 1j * ky[np.newaxis, np.newaxis, :] * spectrum_y
    )
    return np.fft.ifft2(div_spectrum, axes=(1, 2)).real


def _laplacian(field: np.ndarray, lx: float, ly: float) -> np.ndarray:
    _, nx, ny = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    k2 = kx[:, np.newaxis] ** 2 + ky[np.newaxis, :] ** 2
    spectrum = np.fft.fft2(field, axes=(1, 2))
    return np.fft.ifft2((-k2)[np.newaxis, :, :] * spectrum, axes=(1, 2)).real


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
