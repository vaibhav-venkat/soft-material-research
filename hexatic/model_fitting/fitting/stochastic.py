"""Adaptive Fourier stochastic residual model diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

from . import operators as ops
from .fit import FittingResult


STOCHASTIC_ETA_POWER_FRACTION = 0.85
STOCHASTIC_ROLLOUT_SEED = 0


@dataclass(frozen=True)
class StochasticMechanismSummary:
    """Adaptive Fourier AR(1) mechanism fitted to the residual flux."""

    r2: float
    mae: float
    normalized_mae: float
    correlation: float
    empirical_rms_eta: float
    empirical_std_eta: float
    empirical_retained_power_fraction: float
    empirical_dominant_mode_x: int
    empirical_dominant_mode_theta: int
    empirical_lag1_autocorrelation: float
    empirical_correlation_time: float
    model_rms_eta: float
    model_std_eta: float
    model_retained_power_fraction: float
    model_dominant_mode_x: int
    model_dominant_mode_theta: int
    model_lag1_autocorrelation: float
    model_correlation_time: float
    retained_modes: int
    mean_abs_alpha: float
    mean_mode_correlation_time: float
    mean_sigma: float
    note: str


def text_report_lines(summary: StochasticMechanismSummary) -> list[str]:
    lines = [
        "    Adaptive Fourier stochastic mechanism:",
        "      xi_k(t+dt) = alpha_k xi_k(t) + sigma_k zeta_k",
        (
            "      selected modes: empirical eta power ranking, "
            f"keep {100.0 * STOCHASTIC_ETA_POWER_FRACTION:.0f}%"
        ),
        f"      retained Fourier current modes: {summary.retained_modes}",
        f"      mean abs(alpha_k): {summary.mean_abs_alpha:.8g}",
        f"      mean sigma_k:   {summary.mean_sigma:.8g}",
        f"      mean modal correlation time: {summary.mean_mode_correlation_time:.8g}",
        "      Final density model with seeded AR(1) xi:",
        f"        R2:  {summary.r2:.8g}",
        f"        MAE: {summary.mae:.8g}",
        f"        normalized MAE: {summary.normalized_mae:.8g}",
        f"        correlation:    {summary.correlation:.8g}",
        "      eta = -div xi statistics, empirical vs AR(1) mechanism:",
        f"        rms: {summary.empirical_rms_eta:.8g} vs {summary.model_rms_eta:.8g}",
        f"        std: {summary.empirical_std_eta:.8g} vs {summary.model_std_eta:.8g}",
        (
            "        retained eta power fraction: "
            f"{summary.empirical_retained_power_fraction:.8g} vs "
            f"{summary.model_retained_power_fraction:.8g}"
        ),
        (
            "        dominant mode (x,theta): "
            f"({summary.empirical_dominant_mode_x}, {summary.empirical_dominant_mode_theta}) "
            f"vs ({summary.model_dominant_mode_x}, {summary.model_dominant_mode_theta})"
        ),
        (
            "        lag-1 autocorrelation: "
            f"{summary.empirical_lag1_autocorrelation:.8g} vs "
            f"{summary.model_lag1_autocorrelation:.8g}"
        ),
        (
            "        correlation time: "
            f"{summary.empirical_correlation_time:.8g} vs "
            f"{summary.model_correlation_time:.8g}"
        ),
    ]
    if summary.note:
        lines.append(f"      note: {summary.note}")
    return lines


def markdown_report_lines(summary: StochasticMechanismSummary) -> list[str]:
    """Return Markdown report lines for the stochastic mechanism."""
    lines = [
        "Adaptive Fourier stochastic mechanism:",
        "",
        "| quantity | value |",
        "|---|---:|",
        (
            "| selected modes | empirical `eta` power ranking, "
            f"keep {100.0 * STOCHASTIC_ETA_POWER_FRACTION:.0f}% |"
        ),
        f"| retained Fourier current modes | `{summary.retained_modes}` |",
        f"| mean abs(`alpha_k`) | `{summary.mean_abs_alpha:.8g}` |",
        f"| mean `sigma_k` | `{summary.mean_sigma:.8g}` |",
        f"| mean modal correlation time | `{summary.mean_mode_correlation_time:.8g}` |",
        "",
        "Final density model with seeded AR(1) `xi`:",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| R2 vs `partial_t rho` | `{summary.r2:.8g}` |",
        f"| MAE vs `partial_t rho` | `{summary.mae:.8g}` |",
        f"| normalized MAE | `{summary.normalized_mae:.8g}` |",
        f"| correlation | `{summary.correlation:.8g}` |",
        "",
        "`eta = -div xi` statistics, empirical vs AR(1) mechanism:",
        "",
        "| statistic | empirical | AR(1) mechanism |",
        "|---|---:|---:|",
        f"| rms | `{summary.empirical_rms_eta:.8g}` | `{summary.model_rms_eta:.8g}` |",
        f"| std | `{summary.empirical_std_eta:.8g}` | `{summary.model_std_eta:.8g}` |",
        (
            f"| retained eta power fraction | `{summary.empirical_retained_power_fraction:.8g}` | "
            f"`{summary.model_retained_power_fraction:.8g}` |"
        ),
        (
            f"| dominant mode `(x,theta)` | "
            f"`({summary.empirical_dominant_mode_x}, {summary.empirical_dominant_mode_theta})` | "
            f"`({summary.model_dominant_mode_x}, {summary.model_dominant_mode_theta})` |"
        ),
        (
            f"| lag-1 autocorrelation | `{summary.empirical_lag1_autocorrelation:.8g}` | "
            f"`{summary.model_lag1_autocorrelation:.8g}` |"
        ),
        (
            f"| correlation time | `{summary.empirical_correlation_time:.8g}` | "
            f"`{summary.model_correlation_time:.8g}` |"
        ),
        "",
    ]
    if summary.note:
        lines.extend((f"Stochastic note: {summary.note}", ""))
    return lines


def compute_stochastic_mechanism(
    result: FittingResult,
    base_current: np.ndarray,
) -> StochasticMechanismSummary | None:
    """Fit the 85%-eta-power Fourier AR(1) residual model."""
    try:
        j_res = result.fields.material_current - base_current
        j_sys = np.mean(j_res, axis=0, keepdims=True)
        xi = j_res - j_sys
        eta_empirical = -_divergence(result, xi)
        mode_mask = _adaptive_eta_power_mode_mask(
            eta_empirical,
            result.mask,
            keep_fraction=STOCHASTIC_ETA_POWER_FRACTION,
        )
        xi_model, retained_modes, mean_abs_alpha, mean_sigma = _ar1_rollout(
            xi,
            mode_mask=mode_mask,
            seed=STOCHASTIC_ROLLOUT_SEED,
        )
    except (AttributeError, ValueError):
        return None

    if xi_model is None:
        empty_stats = _eta_statistics(
            eta_empirical,
            result.mask,
            result.dt,
            mode_mask=mode_mask,
        )
        return StochasticMechanismSummary(
            r2=float("nan"),
            mae=float("nan"),
            normalized_mae=float("nan"),
            correlation=float("nan"),
            empirical_rms_eta=empty_stats["rms"],
            empirical_std_eta=empty_stats["std"],
            empirical_retained_power_fraction=empty_stats["retained_power_fraction"],
            empirical_dominant_mode_x=int(empty_stats["dominant_mode_x"]),
            empirical_dominant_mode_theta=int(empty_stats["dominant_mode_theta"]),
            empirical_lag1_autocorrelation=empty_stats["lag1_autocorrelation"],
            empirical_correlation_time=empty_stats["correlation_time"],
            model_rms_eta=float("nan"),
            model_std_eta=float("nan"),
            model_retained_power_fraction=float("nan"),
            model_dominant_mode_x=0,
            model_dominant_mode_theta=0,
            model_lag1_autocorrelation=float("nan"),
            model_correlation_time=float("nan"),
            retained_modes=0,
            mean_abs_alpha=float("nan"),
            mean_mode_correlation_time=float("nan"),
            mean_sigma=float("nan"),
            note="not enough time samples or retained Fourier modes for AR(1)",
        )

    eta_model = -_divergence(result, xi_model)
    stochastic_prediction = (
        -_divergence(result, base_current + j_sys)
        + result.fields.S_cross
        + eta_model
    )
    metrics = _metrics_for_partial_t_prediction(result, stochastic_prediction)
    empirical_stats = _eta_statistics(
        eta_empirical,
        result.mask,
        result.dt,
        mode_mask=mode_mask,
    )
    model_stats = _eta_statistics(
        eta_model,
        result.mask,
        result.dt,
        mode_mask=mode_mask,
    )
    mean_mode_tau = (
        -result.dt / np.log(mean_abs_alpha)
        if np.isfinite(mean_abs_alpha) and 0.0 < mean_abs_alpha < 1.0
        else float("nan")
    )
    return StochasticMechanismSummary(
        r2=metrics["r2"],
        mae=metrics["mae"],
        normalized_mae=metrics["normalized_mae"],
        correlation=metrics["correlation"],
        empirical_rms_eta=empirical_stats["rms"],
        empirical_std_eta=empirical_stats["std"],
        empirical_retained_power_fraction=empirical_stats["retained_power_fraction"],
        empirical_dominant_mode_x=int(empirical_stats["dominant_mode_x"]),
        empirical_dominant_mode_theta=int(empirical_stats["dominant_mode_theta"]),
        empirical_lag1_autocorrelation=empirical_stats["lag1_autocorrelation"],
        empirical_correlation_time=empirical_stats["correlation_time"],
        model_rms_eta=model_stats["rms"],
        model_std_eta=model_stats["std"],
        model_retained_power_fraction=model_stats["retained_power_fraction"],
        model_dominant_mode_x=int(model_stats["dominant_mode_x"]),
        model_dominant_mode_theta=int(model_stats["dominant_mode_theta"]),
        model_lag1_autocorrelation=model_stats["lag1_autocorrelation"],
        model_correlation_time=model_stats["correlation_time"],
        retained_modes=retained_modes,
        mean_abs_alpha=mean_abs_alpha,
        mean_mode_correlation_time=mean_mode_tau,
        mean_sigma=mean_sigma,
        note=(
            "seeded adaptive Fourier AR(1) rollout of xi using empirical eta modes "
            f"that retain {100.0 * STOCHASTIC_ETA_POWER_FRACTION:.0f}% of eta power; "
            "pointwise metrics are reproducible but stochastic and should be read "
            "with the distributional statistics"
        ),
    )


def _ar1_rollout(
    xi: np.ndarray,
    *,
    mode_mask: np.ndarray,
    seed: int,
) -> tuple[np.ndarray | None, int, float, float]:
    """Return a seeded AR(1) stochastic rollout for selected xi Fourier modes."""
    xi = np.asarray(xi, dtype=float)
    if xi.ndim != 4 or xi.shape[-1] != 2 or xi.shape[0] < 2:
        return None, 0, float("nan"), float("nan")
    _, nx, ntheta, components = xi.shape
    mode_mask = np.asarray(mode_mask, dtype=bool)
    if mode_mask.shape != (nx, ntheta):
        raise ValueError(
            f"mode_mask must have shape {(nx, ntheta)}, got {mode_mask.shape}."
        )
    retained_modes = int(np.count_nonzero(mode_mask) * components)
    if retained_modes == 0:
        return None, 0, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    xi_model_hat = np.zeros((xi.shape[0], nx, ntheta, components), dtype=complex)
    alpha_values: list[complex] = []
    sigma_values: list[float] = []
    for component in range(components):
        coeff = fft.fft2(xi[..., component], axes=(1, 2))
        for ix, itheta in zip(*np.nonzero(mode_mask), strict=True):
            series = coeff[:, ix, itheta]
            previous = series[:-1]
            following = series[1:]
            denom = np.vdot(previous, previous).real
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            alpha = np.vdot(previous, following) / denom
            innovation = following - alpha * previous
            sigma = float(np.sqrt(np.mean(np.abs(innovation) ** 2)))
            xi_model_hat[0, ix, itheta, component] = series[0]
            for t in range(1, xi.shape[0]):
                noise = sigma * (
                    rng.normal() + 1j * rng.normal()
                ) / np.sqrt(2.0)
                xi_model_hat[t, ix, itheta, component] = (
                    alpha * xi_model_hat[t - 1, ix, itheta, component] + noise
                )
            alpha_values.append(alpha)
            sigma_values.append(sigma)

    if not alpha_values:
        return None, 0, float("nan"), float("nan")

    xi_model = np.empty_like(xi)
    for component in range(components):
        xi_model[..., component] = fft.ifft2(
            xi_model_hat[..., component],
            axes=(1, 2),
        ).real
    mean_abs_alpha = float(np.mean(np.abs(alpha_values)))
    mean_sigma = float(np.mean(sigma_values)) if sigma_values else float("nan")
    return xi_model, retained_modes, mean_abs_alpha, mean_sigma


def _eta_statistics(
    eta: np.ndarray,
    mask: np.ndarray,
    dt: float,
    *,
    mode_mask: np.ndarray,
) -> dict[str, float]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(eta)
    values = np.asarray(eta, dtype=float)[valid]
    if values.size == 0:
        return _empty_eta_statistics()

    spectrum = _spatial_power_spectrum(eta, valid)
    autocorr = _temporal_autocorrelation(eta, valid)
    return {
        "rms": float(np.sqrt(np.mean(values * values))),
        "std": float(np.std(values)),
        "retained_power_fraction": _retained_power_fraction(spectrum, mode_mask),
        "dominant_mode_x": spectrum["dominant_mode_x"],
        "dominant_mode_theta": spectrum["dominant_mode_theta"],
        "lag1_autocorrelation": autocorr[1] if autocorr.size > 1 else float("nan"),
        "correlation_time": _correlation_time_from_autocorrelation(autocorr, dt),
    }


def _empty_eta_statistics() -> dict[str, float]:
    return {
        "rms": float("nan"),
        "std": float("nan"),
        "retained_power_fraction": float("nan"),
        "dominant_mode_x": 0,
        "dominant_mode_theta": 0,
        "lag1_autocorrelation": float("nan"),
        "correlation_time": float("nan"),
    }


def _spatial_power_spectrum(eta: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray | int]:
    eta = np.asarray(eta, dtype=float)
    filled = np.zeros_like(eta)
    for t in range(eta.shape[0]):
        time_valid = valid[t]
        if np.any(time_valid):
            time_values = eta[t, time_valid]
            filled[t] = np.where(time_valid, eta[t], float(np.mean(time_values)))
    coeff = fft.fft2(filled, axes=(1, 2))
    power = np.mean(np.abs(coeff) ** 2, axis=0) / float(eta.shape[1] * eta.shape[2]) ** 2
    power_no_zero = power.copy()
    power_no_zero[0, 0] = 0.0
    flat_idx = int(np.argmax(power_no_zero))
    ix, itheta = np.unravel_index(flat_idx, power_no_zero.shape)
    x_modes = (fft.fftfreq(eta.shape[1]) * eta.shape[1]).astype(int)
    theta_modes = (fft.fftfreq(eta.shape[2]) * eta.shape[2]).astype(int)
    return {
        "power": power,
        "dominant_mode_x": int(x_modes[ix]),
        "dominant_mode_theta": int(theta_modes[itheta]),
    }


def _retained_power_fraction(
    spectrum: dict[str, np.ndarray | int],
    mode_mask: np.ndarray,
) -> float:
    power = np.asarray(spectrum["power"], dtype=float)
    mode_mask = np.asarray(mode_mask, dtype=bool)
    if mode_mask.shape != power.shape:
        raise ValueError(f"mode_mask must match power shape {power.shape}.")
    nonzero = np.ones_like(power, dtype=bool)
    nonzero[0, 0] = False
    total = float(np.sum(power[nonzero]))
    if total <= 0.0 or not np.isfinite(total):
        return float("nan")
    return float(np.sum(power[mode_mask]) / total)


def _adaptive_eta_power_mode_mask(
    eta: np.ndarray,
    mask: np.ndarray,
    *,
    keep_fraction: float,
) -> np.ndarray:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1].")
    valid = np.asarray(mask, dtype=bool) & np.isfinite(eta)
    spectrum = _spatial_power_spectrum(eta, valid)
    power = np.asarray(spectrum["power"], dtype=float).copy()
    power[0, 0] = 0.0
    flat_power = power.ravel()
    total = float(np.sum(flat_power))
    selected = np.zeros_like(flat_power, dtype=bool)
    if total <= 0.0 or not np.isfinite(total):
        return selected.reshape(power.shape)
    running = 0.0
    for idx in np.argsort(flat_power)[::-1]:
        if flat_power[idx] <= 0.0:
            break
        selected[idx] = True
        running += float(flat_power[idx])
        if running / total >= keep_fraction:
            break
    return selected.reshape(power.shape)


def _temporal_autocorrelation(eta: np.ndarray, valid: np.ndarray) -> np.ndarray:
    all_valid = np.all(valid, axis=0)
    if eta.shape[0] < 2 or not np.any(all_valid):
        return np.asarray([1.0])
    series = np.asarray(eta[:, all_valid], dtype=float)
    series = series - np.mean(series, axis=0, keepdims=True)
    denom = float(np.mean(series * series))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.asarray([1.0])
    autocorr = np.empty(series.shape[0], dtype=float)
    autocorr[0] = 1.0
    for lag in range(1, series.shape[0]):
        autocorr[lag] = float(np.mean(series[:-lag] * series[lag:]) / denom)
    return autocorr


def _correlation_time_from_autocorrelation(autocorr: np.ndarray, dt: float) -> float:
    if autocorr.size < 2:
        return float("nan")
    positive = autocorr[1:]
    nonpositive = np.nonzero(positive <= 0.0)[0]
    stop = int(nonpositive[0]) if nonpositive.size else positive.size
    if stop <= 0:
        return 0.0
    return float(dt * np.sum(positive[:stop]))


def _metrics_for_partial_t_prediction(result: FittingResult, prediction: np.ndarray) -> dict[str, float]:
    valid = result.mask & np.isfinite(result.fields.partial_t_rho) & np.isfinite(prediction)
    y = result.fields.partial_t_rho[valid]
    p = prediction[valid]
    if y.size == 0:
        return {
            "r2": float("nan"),
            "mae": float("nan"),
            "normalized_mae": float("nan"),
            "correlation": float("nan"),
        }
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - float(np.sum((y - p) ** 2)) / ss_tot
    mae = float(np.mean(np.abs(y - p)))
    scale = float(np.mean(np.abs(y)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    corr = float("nan")
    if y.size >= 2 and np.std(y) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(y, p)[0, 1])
    return {
        "r2": r2,
        "mae": mae,
        "normalized_mae": mae / scale,
        "correlation": corr,
    }


def _divergence(result: FittingResult, current: np.ndarray) -> np.ndarray:
    fields = result.fields
    ly = fields.cylinder_radius * (fields.theta_edges[-1] - fields.theta_edges[0])
    kx, ky = ops.build_k_vectors(
        fields.x_centers.size,
        fields.theta_centers.size,
        fields.lx,
        ly,
    )
    return ops.fft_divergence(current, kx, ky)
