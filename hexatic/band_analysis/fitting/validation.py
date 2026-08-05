"""Evaluate residual, predictive, trajectory, and holdout behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from .model import (
    Scaling,
    build_state_space_sequences,
    build_transition_blocks,
    helmert_basis,
    oscillatory_interval_moments,
    persistent_innovations,
    transition,
    transition_log_density,
)
from ..detection.segments import StableSegment


# Defined once so their trace caches are shared across blocks, validation sets,
# and lags rather than rebuilt on every loop iteration.
_transition_rows = jax.vmap(transition, in_axes=(0, 0, None))
_transition_draws = jax.vmap(_transition_rows, in_axes=(None, None, 0))
_density_rows = jax.vmap(transition_log_density, in_axes=(0, 0, 0, None))
_density_draws = jax.vmap(_density_rows, in_axes=(None, None, None, 0))


@dataclass(frozen=True)
class ValidationResult:
    arrays: dict[str, np.ndarray]
    metrics: dict[str, float | int]


def _draw_indices(sample_count: int, draws: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.choice(sample_count, size=draws, replace=sample_count < draws)


def _posterior_matrix(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("posterior samples must have shape (..., 7)")
    return values


def _concatenate(
    values: list[np.ndarray], *, dtype: np.dtype[Any] = np.dtype(np.float64)
) -> np.ndarray:
    return np.concatenate(values) if values else np.empty(0, dtype=dtype)


def _one_step(
    segments: list[StableSegment],
    lag: int,
    scaling: Scaling,
    posterior: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    normalized = [scaling.normalize_segment(segment) for segment in segments]
    blocks = build_transition_blocks(normalized, lag)
    selected = posterior[_draw_indices(len(posterior), draws, seed)]
    selected_device = jnp.asarray(selected)
    median_parameters = np.median(selected, axis=0)
    median_device = jnp.asarray(median_parameters)
    generator = np.random.default_rng(seed + 1)
    observed_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    low_parts: list[np.ndarray] = []
    high_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    residual_total_parts: list[np.ndarray] = []
    residual_deviation_parts: list[np.ndarray] = []
    residual_band_count_parts: list[np.ndarray] = []
    residual_dt_parts: list[np.ndarray] = []
    observed_delta_parts: list[np.ndarray] = []
    predicted_delta_parts: list[np.ndarray] = []
    observed_total_delta_parts: list[np.ndarray] = []
    predicted_total_delta_parts: list[np.ndarray] = []
    observed_deviation_parts: list[np.ndarray] = []
    predicted_deviation_parts: list[np.ndarray] = []
    observed_exchange_parts: list[np.ndarray] = []
    predicted_exchange_parts: list[np.ndarray] = []
    residual_covariances: list[np.ndarray] = []
    residual_covariance_band_counts: list[int] = []
    residual_covariance_offsets = [0]
    log_scores: list[np.ndarray] = []
    covered = 0
    values = 0
    for block in blocks:
        current = np.asarray(block.current)
        following = np.asarray(block.following)
        means, covariances = _transition_draws(
            block.current, block.dt, selected_device
        )
        means_np = np.asarray(means)
        cholesky = np.linalg.cholesky(np.asarray(covariances))
        noise = generator.standard_normal(means_np.shape)
        predictions = means_np + np.einsum("...ij,...j->...i", cholesky, noise)
        draw_logp = np.asarray(
            _density_draws(
                block.following,
                block.current,
                block.dt,
                selected_device,
            )
        )
        predictive_mean = predictions.mean(axis=0)
        transition_draws = generator.integers(0, len(selected), size=len(current))
        predictive_sample = predictions[
            transition_draws, np.arange(len(current))
        ]
        low, high = np.quantile(predictions, [0.025, 0.975], axis=0)
        median_means, median_covariances = _transition_rows(
            block.current, block.dt, median_device
        )
        block_residuals = np.linalg.solve(
            np.linalg.cholesky(np.asarray(median_covariances)),
            (following - np.asarray(median_means))[..., None],
        )[..., 0]
        residual_parts.append(block_residuals.ravel())
        residual_total_parts.append(
            np.repeat(current.sum(axis=1), current.shape[1])
        )
        residual_deviation_parts.append(
            (current - current.mean(axis=1, keepdims=True)).ravel()
        )
        residual_band_count_parts.append(
            np.full(block_residuals.size, current.shape[1], dtype=np.int64)
        )
        residual_dt_parts.append(np.repeat(np.asarray(block.dt), current.shape[1]))
        residual_covariance = (
            np.cov(block_residuals, rowvar=False)
            if len(block_residuals) > 1
            else np.zeros((following.shape[1], following.shape[1]))
        )
        residual_covariances.append(np.atleast_2d(residual_covariance).ravel())
        residual_covariance_band_counts.append(following.shape[1])
        residual_covariance_offsets.append(
            residual_covariance_offsets[-1] + following.shape[1] ** 2
        )
        observed_parts.append(following.ravel())
        mean_parts.append(predictive_mean.ravel())
        observed_delta = following - current
        predicted_delta = predictive_sample - current
        observed_delta_parts.append(observed_delta.ravel())
        predicted_delta_parts.append(predicted_delta.ravel())
        observed_total_delta_parts.append(observed_delta.sum(axis=1))
        predicted_total_delta_parts.append(predicted_delta.sum(axis=1))
        observed_deviation_parts.append(
            (following - following.mean(axis=1, keepdims=True)).ravel()
        )
        predicted_deviation_parts.append(
            (
                predictive_sample
                - predictive_sample.mean(axis=1, keepdims=True)
            ).ravel()
        )
        weights = current / current.sum(axis=1, keepdims=True)
        observed_exchange_parts.append(
            (observed_delta - weights * observed_delta.sum(axis=1, keepdims=True)).ravel()
        )
        predicted_exchange_parts.append(
            (predicted_delta - weights * predicted_delta.sum(axis=1, keepdims=True)).ravel()
        )
        low_parts.append(low.ravel())
        high_parts.append(high.ravel())
        log_scores.append(
            logsumexp(draw_logp, axis=0)
            - np.log(draws)
            - following.shape[1] * np.log(scaling.area)
        )
        covered += int(np.count_nonzero((following >= low) & (following <= high)))
        values += following.size
    observed = _concatenate(observed_parts)
    predicted = _concatenate(mean_parts)
    whitened = _concatenate(residual_parts)
    temporal_acf = _temporal_residual_acf(normalized, lag, median_parameters)
    arrays = {
        "one_step_observed": observed * scaling.area,
        "one_step_mean": predicted * scaling.area,
        "one_step_low": _concatenate(low_parts) * scaling.area,
        "one_step_high": _concatenate(high_parts) * scaling.area,
        "whitened_residual": whitened,
        "residual_total_area": _concatenate(residual_total_parts) * scaling.area,
        "residual_area_deviation": _concatenate(residual_deviation_parts)
        * scaling.area,
        "residual_band_count": _concatenate(
            residual_band_count_parts, dtype=np.dtype(np.int64)
        ),
        "residual_dt": _concatenate(residual_dt_parts) * scaling.time,
        "whitened_residual_acf": temporal_acf,
        "whitened_residual_covariance": _concatenate(residual_covariances),
        "residual_covariance_offset": np.asarray(
            residual_covariance_offsets, dtype=np.int64
        ),
        "residual_covariance_band_count": np.asarray(
            residual_covariance_band_counts, dtype=np.int64
        ),
        "transition_dt": _concatenate(
            [np.asarray(block.dt) for block in blocks]
        )
        * scaling.time,
        "observed_delta_area": _concatenate(observed_delta_parts) * scaling.area,
        "predicted_delta_area": _concatenate(predicted_delta_parts) * scaling.area,
        "observed_delta_total": _concatenate(observed_total_delta_parts) * scaling.area,
        "predicted_delta_total": _concatenate(predicted_total_delta_parts) * scaling.area,
        "observed_area_deviation": _concatenate(observed_deviation_parts) * scaling.area,
        "predicted_area_deviation": _concatenate(predicted_deviation_parts) * scaling.area,
        "observed_conservative_increment": _concatenate(observed_exchange_parts)
        * scaling.area,
        "predicted_conservative_increment": _concatenate(predicted_exchange_parts)
        * scaling.area,
    }
    metrics = {
        "predictive_log_score": float(np.mean(np.concatenate(log_scores)))
        if log_scores
        else float("nan"),
        "coverage_95": float(covered / values) if values else float("nan"),
        "residual_mean": float(np.mean(whitened)) if whitened.size else float("nan"),
        "residual_variance": float(np.var(whitened)) if whitened.size else float("nan"),
        "residual_lag1_autocorrelation": (
            float(temporal_acf[1]) if len(temporal_acf) > 1 else float("nan")
        ),
        "one_step_rmse": float(np.sqrt(np.mean((observed - predicted) ** 2)) * scaling.area)
        if observed.size
        else float("nan"),
    }
    return arrays, metrics


def _pooled_autocorrelation(series: list[np.ndarray]) -> np.ndarray:
    """Pool temporal products without joining trajectories or vector components."""
    usable = [np.asarray(values, dtype=np.float64) for values in series if len(values)]
    if not usable:
        return np.empty(0, dtype=np.float64)
    mean = float(np.mean(np.concatenate(usable)))
    centered = [values - mean for values in usable]
    denominator = sum(float(np.dot(values, values)) for values in centered)
    maximum = max(len(values) for values in centered)
    if denominator == 0.0:
        return np.zeros(maximum, dtype=np.float64)
    return np.asarray(
        [
            sum(
                float(np.dot(values[:-lag], values[lag:]))
                for values in centered
                if lag and len(values) > lag
            )
            / denominator
            if lag
            else 1.0
            for lag in range(maximum)
        ],
        dtype=np.float64,
    )


def _temporal_residual_acf(
    segments: list[StableSegment], lag: int, parameters: np.ndarray
) -> np.ndarray:
    """Pool total-mode and Kalman innovations without joining trajectories."""
    series: list[np.ndarray] = []
    _, _, kappa_total, _, diffusion_total, area_star, _ = parameters
    for segment in segments:
        if len(segment.tau) <= lag:
            continue
        for phase in range(lag):
            areas = segment.areas[phase::lag]
            tau = segment.tau[phase::lag]
            if len(tau) < 2:
                continue
            current_total = areas[:-1].sum(axis=1)
            following_total = areas[1:].sum(axis=1)
            intervals = np.diff(tau)
            total_mean = current_total - kappa_total * (
                current_total - area_star
            ) * intervals
            total_residual = (following_total - total_mean) / np.sqrt(
                2.0 * diffusion_total * intervals
            )
            series.append(total_residual)
        for sequence in build_state_space_sequences([segment], lag):
            innovations = np.asarray(
                persistent_innovations(sequence, jnp.asarray(parameters))
            )
            series.extend(
                innovations[:, component]
                for component in range(innovations.shape[1])
            )
    return _pooled_autocorrelation(series)


def _pooled_increment_statistic(
    series: list[np.ndarray], reduce: Callable[[np.ndarray], float]
) -> np.ndarray:
    """Reduce the pooled lag-`k` increments of every series, for each lag."""
    usable = [np.asarray(values, dtype=np.float64) for values in series if len(values)]
    if not usable:
        return np.empty(0, dtype=np.float64)
    maximum = max(len(values) for values in usable)
    statistics = [0.0]
    for lag in range(1, maximum):
        increments = [item[lag:] - item[:-lag] for item in usable if len(item) > lag]
        statistics.append(
            reduce(np.concatenate(increments)) if increments else float("nan")
        )
    return np.asarray(statistics, dtype=np.float64)


def _pooled_msd(series: list[np.ndarray]) -> np.ndarray:
    return _pooled_increment_statistic(
        series, lambda increments: float(np.mean(increments**2))
    )


def _grouped_msd_quantiles(
    series: list[np.ndarray], group_sizes: list[int]
) -> np.ndarray:
    """MSD per path, reduced to a 2.5/97.5 envelope across paths at every lag."""
    curves = []
    start = 0
    for size in group_sizes:
        curve = _pooled_msd(series[start : start + size])
        if len(curve):
            curves.append(curve)
        start += size
    if not curves:
        return np.empty((2, 0), dtype=np.float64)
    padded = np.full((len(curves), max(map(len, curves))), np.nan, dtype=np.float64)
    for index, curve in enumerate(curves):
        padded[index, : len(curve)] = curve
    return np.nanquantile(padded, (0.025, 0.975), axis=0)


def _replicate_curves(replicates: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Pool each replicate dataset exactly as the observed one is pooled.

    Every replicate holds one simulated path per segment, so the drop-out
    normalization in ``_pooled_autocorrelation`` biases it the same way it biases
    the observed estimate and the comparison stays like-for-like.
    """
    return [_pooled_autocorrelation(series) for series in replicates if series]


def _replicate_acf_quantiles(curves: list[np.ndarray]) -> np.ndarray:
    """Reduce pooled replicate ACFs to a 2.5/97.5 envelope at every lag."""
    if not curves:
        return np.empty((2, 0), dtype=np.float64)
    padded = np.full((len(curves), max(map(len, curves))), np.nan, dtype=np.float64)
    for index, curve in enumerate(curves):
        padded[index, : len(curve)] = curve
    return np.nanquantile(padded, (0.025, 0.975), axis=0)


def _acf_discrepancy(
    observed: np.ndarray,
    curves: list[np.ndarray],
    simulated: np.ndarray,
    counts: np.ndarray,
    lag_time: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Score the observed ACF against the same-geometry null past the model's decay.

    Two statistics, because a signed integral cancels a damped oscillation against
    its own rebound and is blind to exactly the alternative of interest:

    ``negative_area`` -- the area below zero only. The fitted model's ACF stays
    positive through the window, so negativity is pure discrepancy and the rebound
    contributes nothing. Capped at the last lag with five contributing segments.

    ``shape_chi2`` -- squared deviation from the null mean in units of the null
    standard deviation, summed over the window. Slow decay, lobe, and rebound all
    add rather than cancel, and thin long lags are automatically down-weighted by
    their inflated null variance.

    The window opens where the fitted model's own pooled ACF falls below 0.05, so
    both edges come from the null and the geometry rather than from the observed
    curve. The replicates supply both the null moments and the null distribution;
    with a few hundred of them the reuse is negligible.
    """
    empty = (
        {
            "transfer_rate_negative_area": float("nan"),
            "transfer_rate_negative_area_p": float("nan"),
            "transfer_rate_shape_chi2": float("nan"),
            "transfer_rate_shape_chi2_p": float("nan"),
        },
        np.empty((2, 0), dtype=np.float64),
    )
    decayed = np.flatnonzero(np.abs(simulated) < 0.05)
    shape_lags = np.flatnonzero(counts >= 3)
    negative_lags = np.flatnonzero(counts >= 5)
    if not decayed.size or not shape_lags.size or not curves:
        return empty
    start = int(decayed[0])
    shape_stop = int(shape_lags[-1]) + 1
    negative_stop = int(negative_lags[-1]) + 1 if negative_lags.size else shape_stop
    if shape_stop - start < 2 or start >= len(lag_time):
        return empty

    padded = np.full((len(curves), max(map(len, curves))), np.nan, dtype=np.float64)
    for index, curve in enumerate(curves):
        padded[index, : len(curve)] = curve
    null_mean = np.nanmean(padded, axis=0)
    null_deviation = np.sqrt(np.nanvar(padded, axis=0))

    def negative_area(curve: np.ndarray) -> float:
        end = min(negative_stop, len(curve), len(lag_time))
        if end - start < 2:
            return float("nan")
        return float(
            np.trapezoid(np.minimum(curve[start:end], 0.0), lag_time[start:end])
        )

    def shape_chi2(curve: np.ndarray) -> float:
        end = min(shape_stop, len(curve), len(null_mean))
        if end - start < 2:
            return float("nan")
        deviation = curve[start:end] - null_mean[start:end]
        scale = np.maximum(null_deviation[start:end], 1e-9)
        return float(np.sum((deviation / scale) ** 2))

    observed_negative, observed_shape = negative_area(observed), shape_chi2(observed)
    null = np.asarray(
        [[negative_area(curve), shape_chi2(curve)] for curve in curves],
        dtype=np.float64,
    ).T
    null = null[:, np.all(np.isfinite(null), axis=0)]
    if not null.size or not np.isfinite(observed_negative + observed_shape):
        return empty
    return (
        {
            # more negative than the null is the evidence, so both tails are one-sided
            "transfer_rate_negative_area": observed_negative,
            "transfer_rate_negative_area_p": float(
                np.mean(null[0] <= observed_negative)
            ),
            "transfer_rate_shape_chi2": observed_shape,
            "transfer_rate_shape_chi2_p": float(np.mean(null[1] >= observed_shape)),
            "transfer_rate_window_start": float(lag_time[start]),
            "transfer_rate_negative_window_stop": float(
                lag_time[min(negative_stop, len(lag_time)) - 1]
            ),
            "transfer_rate_shape_window_stop": float(
                lag_time[min(shape_stop, len(lag_time)) - 1]
            ),
            "transfer_rate_discrepancy_replicates": float(null.shape[1]),
        },
        null,
    )


def _contributing_counts(lengths: list[int]) -> np.ndarray:
    """Number of segments long enough to contribute at each lag."""
    if not lengths:
        return np.empty(0, dtype=np.int64)
    return np.asarray(
        [sum(1 for value in lengths if value > lag) for lag in range(max(lengths))],
        dtype=np.int64,
    )


def _grouped_autocorrelation(
    series: list[np.ndarray], group_sizes: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrelate each group of series alone, flattened with its offsets."""
    curves = []
    start = 0
    for size in group_sizes:
        curves.append(_pooled_autocorrelation(series[start : start + size]))
        start += size
    offsets = np.cumsum([0, *(len(curve) for curve in curves)], dtype=np.int64)
    return _concatenate(curves), offsets


def _pooled_lag_times(series: list[np.ndarray]) -> np.ndarray:
    return _pooled_increment_statistic(
        series, lambda increments: float(np.median(increments))
    )


def _increment_statistics(
    increment: np.ndarray, left_area: np.ndarray, n_bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened band, conservative-mode, and band-total covariances."""
    if len(increment) > 1:
        band = np.cov(increment, rowvar=False)
        weights = left_area / left_area.sum(axis=1, keepdims=True)
        conservative_increment = (
            increment - weights * increment.sum(axis=1, keepdims=True)
        )
        conservative = np.cov(
            conservative_increment, rowvar=False
        )
        total = np.asarray(
            [
                np.cov(increment[:, component], increment.sum(axis=1))[0, 1]
                for component in range(n_bands)
            ]
        )
    else:
        band = np.zeros((n_bands, n_bands))
        conservative = np.zeros((n_bands, n_bands))
        total = np.zeros(n_bands)
    return (
        np.atleast_2d(band).ravel(),
        np.atleast_2d(conservative).ravel(),
        total,
    )


def _relaxation_time(acf: np.ndarray, lag_times: np.ndarray) -> float:
    crossing = np.flatnonzero(np.asarray(acf) <= np.exp(-1.0))
    return float(lag_times[crossing[0]]) if len(crossing) else float("nan")


def _transfer_rate_series(
    areas: np.ndarray, tau: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return area-fraction-weighted redistribution rates."""
    if len(tau) < 2 or tau[-1] <= tau[0]:
        return [], []
    increments = np.diff(areas, axis=0)
    total_increment = increments.sum(axis=1)
    weights = areas[:-1] / areas[:-1].sum(axis=1, keepdims=True)
    transfer = (increments - weights * total_increment[:, None]) / np.diff(tau)[:, None]
    transfer_tau = tau[:-1]
    return (
        [transfer[:, component] for component in range(areas.shape[1])],
        [transfer_tau for _ in range(areas.shape[1])],
    )


def _paths(
    segments: list[StableSegment],
    lag: int,
    scaling: Scaling,
    posterior: np.ndarray,
    paths_per_segment: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    generator = np.random.default_rng(seed)
    selected = posterior[
        _draw_indices(len(posterior), paths_per_segment * max(len(segments), 1), seed)
    ]
    path_offsets = [0]
    path_time_offsets = [0]
    negative = 0
    generated = 0
    terminated = 0
    observed_covariances: list[np.ndarray] = []
    simulated_covariances: list[np.ndarray] = []
    covariance_offsets = [0]
    path_band_counts: list[int] = []
    path_segment_indices: list[int] = []
    observed_area_values: list[np.ndarray] = []
    simulated_area_values: list[np.ndarray] = []
    observed_conservative: list[np.ndarray] = []
    simulated_conservative: list[np.ndarray] = []
    observed_increment_values: list[np.ndarray] = []
    simulated_increment_values: list[np.ndarray] = []
    observed_path_offsets = [0]
    observed_path_time_offsets = [0]
    observed_path_band_counts: list[int] = []
    observed_path_tau: list[np.ndarray] = []
    path_tau: list[np.ndarray] = []
    observed_total_series: list[np.ndarray] = []
    simulated_total_series: list[np.ndarray] = []
    observed_conservative_series: list[np.ndarray] = []
    simulated_conservative_series: list[np.ndarray] = []
    observed_area_series: list[np.ndarray] = []
    simulated_area_series: list[np.ndarray] = []
    observed_tau_series: list[np.ndarray] = []
    simulated_tau_series: list[np.ndarray] = []
    observed_transfer_rate_series: list[np.ndarray] = []
    observed_transfer_rate_counts: list[int] = []
    observed_transfer_rate_lengths: list[int] = []
    # Replicate r is one simulated path from every segment: a whole synthetic
    # dataset with the observed geometry, pooled the way the observed data is.
    replicate_transfer_series: list[list[np.ndarray]] = [
        [] for _ in range(paths_per_segment)
    ]
    simulated_transfer_rate_series: list[np.ndarray] = []
    observed_transfer_rate_tau: list[np.ndarray] = []
    simulated_transfer_rate_tau: list[np.ndarray] = []
    observed_conservative_covariances: list[np.ndarray] = []
    simulated_conservative_covariances: list[np.ndarray] = []
    observed_total_increment_covariances: list[np.ndarray] = []
    simulated_total_increment_covariances: list[np.ndarray] = []
    covariance_vector_offsets = [0]
    for segment_index, segment in enumerate(segments):
        normalized = scaling.normalize_segment(segment)
        sample_indices = np.arange(0, len(normalized.tau), lag, dtype=np.int64)
        sampled_tau = normalized.tau[sample_indices]
        sampled_areas = normalized.areas[sample_indices]
        observed_total = sampled_areas.sum(axis=1)
        observed_area_values.append(sampled_areas.ravel())
        observed_path_offsets.append(
            observed_path_offsets[-1] + sampled_areas.size
        )
        observed_path_band_counts.append(segment.n_bands)
        observed_path_tau.append(sampled_tau * scaling.time)
        observed_path_time_offsets.append(
            observed_path_time_offsets[-1] + len(sampled_tau)
        )
        observed_total_series.append(observed_total)
        observed_tau_series.append(sampled_tau)
        observed_area_series.extend(
            sampled_areas[:, component] for component in range(segment.n_bands)
        )
        observed_deviation = sampled_areas - sampled_areas.mean(axis=1, keepdims=True)
        transfer_rate, transfer_tau = _transfer_rate_series(
            sampled_areas, sampled_tau
        )
        observed_transfer_rate_series.extend(transfer_rate)
        observed_transfer_rate_tau.extend(transfer_tau)
        observed_transfer_rate_counts.append(len(transfer_rate))
        observed_transfer_rate_lengths.append(
            len(transfer_rate[0]) if transfer_rate else 0
        )
        observed_conservative.append(observed_deviation.ravel())
        observed_conservative_series.extend(
            observed_deviation[:, component] for component in range(segment.n_bands)
        )
        observed_increment = np.diff(sampled_areas, axis=0)
        observed_increment_values.append(observed_increment.ravel())
        basis = helmert_basis(segment.n_bands)
        band, conservative, total = _increment_statistics(
            observed_increment, sampled_areas[:-1], segment.n_bands
        )
        observed_covariances.append(band)
        observed_conservative_covariances.append(conservative)
        observed_total_increment_covariances.append(total)
        simulated_segment_increments: list[np.ndarray] = []
        simulated_segment_left_areas: list[np.ndarray] = []
        parameter_start = segment_index * paths_per_segment
        segment_parameters = selected[
            parameter_start : parameter_start + paths_per_segment
        ]
        paths = np.full(
            (paths_per_segment, *sampled_areas.shape), np.nan, dtype=np.float64
        )
        paths[:, 0] = sampled_areas[0]
        component_count = segment.n_bands - 1
        rates = generator.standard_normal(
            (paths_per_segment, component_count)
        ) * np.sqrt(
            segment_parameters[:, 3] / segment_parameters[:, 0]
        )[:, None]
        oscillatory_rates = generator.standard_normal(
            (paths_per_segment, component_count)
        ) * np.sqrt(
            segment_parameters[:, 3] / segment_parameters[:, 0]
        )[:, None]
        slopes = generator.standard_normal(
            (paths_per_segment, component_count)
        ) * segment_parameters[:, 6, None]
        stops = np.full(paths_per_segment, len(sampled_areas), dtype=np.int64)
        active = np.ones(paths_per_segment, dtype=bool)
        for frame in range(len(sampled_areas) - 1):
            current = paths[:, frame]
            valid_current = (
                active
                & np.all(np.isfinite(current), axis=1)
                & (np.abs(current.sum(axis=1)) >= 1e-14)
            )
            newly_terminated = active & ~valid_current
            stops[newly_terminated] = frame + 1
            terminated += int(np.count_nonzero(newly_terminated))
            active = valid_current
            indices = np.flatnonzero(active)
            if not len(indices):
                break
            parameters = segment_parameters[indices]
            step = sampled_tau[frame + 1] - sampled_tau[frame]
            totals = current[indices].sum(axis=1)
            total_decay = np.exp(-parameters[:, 2] * step)
            following_totals = (
                parameters[:, 5]
                + (totals - parameters[:, 5]) * total_decay
                + np.sqrt(
                    parameters[:, 4] / parameters[:, 2] * (1.0 - total_decay**2)
                )
                * generator.standard_normal(len(indices))
            )
            weights = current[indices] / totals[:, None]
            (
                transition_matrix,
                integral_coefficients,
                state_process_variance,
                integral_process_covariance,
                integral_process_variance,
            ) = (
                np.asarray(value)
                for value in oscillatory_interval_moments(
                    step,
                    parameters[:, 0],
                    parameters[:, 1],
                    parameters[:, 3],
                )
            )
            current_rates = rates[indices].copy()
            current_oscillatory_rates = oscillatory_rates[indices].copy()
            joint_covariance = np.zeros((len(indices), 3, 3), dtype=np.float64)
            joint_covariance[:, 0, 0] = state_process_variance
            joint_covariance[:, 1, 1] = state_process_variance
            joint_covariance[:, :2, 2] = integral_process_covariance
            joint_covariance[:, 2, :2] = integral_process_covariance
            joint_covariance[:, 2, 2] = integral_process_variance
            cholesky = np.linalg.cholesky(joint_covariance)
            noise = np.einsum(
                "pij,pmj->pmi",
                cholesky,
                generator.standard_normal((len(indices), component_count, 3)),
            )
            integral = (
                integral_coefficients[:, 0, None] * current_rates
                + integral_coefficients[:, 1, None]
                * current_oscillatory_rates
                + noise[:, :, 2]
                + slopes[indices] * step
            )
            redistribution = integral @ basis.T
            following = (
                current[indices]
                + weights * (following_totals - totals)[:, None]
                + redistribution
            )
            rates[indices] = (
                transition_matrix[:, 0, 0, None] * current_rates
                + transition_matrix[:, 0, 1, None]
                * current_oscillatory_rates
                + noise[:, :, 0]
            )
            oscillatory_rates[indices] = (
                transition_matrix[:, 1, 0, None] * current_rates
                + transition_matrix[:, 1, 1, None]
                * current_oscillatory_rates
                + noise[:, :, 1]
            )
            finite = np.all(np.isfinite(following), axis=1)
            failed_indices = indices[~finite]
            active[failed_indices] = False
            stops[failed_indices] = frame + 1
            terminated += len(failed_indices)
            successful = indices[finite]
            paths[successful, frame + 1] = following[finite]
            negative += int(np.count_nonzero(following[finite] < 0.0))
            generated += following[finite].size
        for path_index, path in enumerate(paths):
            valid = path[: stops[path_index]]
            path_tau.append(sampled_tau[: len(valid)] * scaling.time)
            path_time_offsets.append(path_time_offsets[-1] + len(valid))
            path_offsets.append(path_offsets[-1] + valid.size)
            path_band_counts.append(segment.n_bands)
            path_segment_indices.append(segment_index)
            simulated_area_values.append(valid.ravel())
            deviation = valid - valid.mean(axis=1, keepdims=True)
            simulated_conservative.append(deviation.ravel())
            simulated_total_series.append(valid.sum(axis=1))
            simulated_tau_series.append(sampled_tau[: len(valid)])
            simulated_area_series.extend(
                valid[:, component] for component in range(segment.n_bands)
            )
            simulated_conservative_series.extend(
                deviation[:, component] for component in range(segment.n_bands)
            )
            valid_tau = sampled_tau[: len(valid)]
            transfer_rate, transfer_tau = _transfer_rate_series(valid, valid_tau)
            simulated_transfer_rate_series.extend(transfer_rate)
            simulated_transfer_rate_tau.extend(transfer_tau)
            replicate_transfer_series[path_index].extend(transfer_rate)
            if len(valid) > 1:
                increment = np.diff(valid, axis=0)
                simulated_segment_increments.append(increment)
                simulated_segment_left_areas.append(valid[:-1])
                simulated_increment_values.append(increment.ravel())
        combined_increment = (
            np.concatenate(simulated_segment_increments)
            if simulated_segment_increments
            else np.empty((0, segment.n_bands))
        )
        combined_left_area = (
            np.concatenate(simulated_segment_left_areas)
            if simulated_segment_left_areas
            else np.empty((0, segment.n_bands))
        )
        band, conservative, total = _increment_statistics(
            combined_increment, combined_left_area, segment.n_bands
        )
        simulated_covariances.append(band)
        simulated_conservative_covariances.append(conservative)
        simulated_total_increment_covariances.append(total)
        covariance_offsets.append(covariance_offsets[-1] + segment.n_bands**2)
        covariance_vector_offsets.append(
            covariance_vector_offsets[-1] + segment.n_bands
        )
    observed = _concatenate(observed_total_series)
    simulated = _concatenate(simulated_total_series)
    observed_area = _concatenate(observed_area_values)
    simulated_area = _concatenate(simulated_area_values)
    observed_difference = _concatenate(observed_conservative)
    simulated_difference = _concatenate(simulated_conservative)
    # `path_area`/`observed_path_area` hold the same values as their flattened
    # `simulated_area`/`observed_area` counterparts, so scale each array once.
    observed_path_area = observed_area * scaling.area
    simulated_path_area = simulated_area * scaling.area
    observed_total_acf = _pooled_autocorrelation(observed_total_series)
    simulated_total_acf = _pooled_autocorrelation(simulated_total_series)
    segment_total_acf, segment_total_acf_offsets = _grouped_autocorrelation(
        observed_total_series, [1] * len(observed_total_series)
    )
    segment_transfer_acf, segment_transfer_acf_offsets = _grouped_autocorrelation(
        observed_transfer_rate_series, observed_transfer_rate_counts
    )
    replicate_curves = _replicate_curves(replicate_transfer_series)
    transfer_rate_segment_counts = _contributing_counts(observed_transfer_rate_lengths)
    observed_conservative_acf = _pooled_autocorrelation(
        observed_conservative_series
    )
    simulated_conservative_acf = _pooled_autocorrelation(
        simulated_conservative_series
    )
    observed_transfer_rate_acf = _pooled_autocorrelation(
        observed_transfer_rate_series
    )
    simulated_transfer_rate_acf = _pooled_autocorrelation(
        simulated_transfer_rate_series
    )
    observed_lag_times = _pooled_lag_times(observed_tau_series) * scaling.time
    simulated_lag_times = _pooled_lag_times(simulated_tau_series) * scaling.time
    observed_transfer_rate_lag_times = (
        _pooled_lag_times(observed_transfer_rate_tau) * scaling.time
    )
    simulated_transfer_rate_lag_times = (
        _pooled_lag_times(simulated_transfer_rate_tau) * scaling.time
    )
    transfer_rate_discrepancy, transfer_rate_discrepancy_null = _acf_discrepancy(
        observed_transfer_rate_acf,
        replicate_curves,
        simulated_transfer_rate_acf,
        transfer_rate_segment_counts,
        observed_transfer_rate_lag_times,
    )
    arrays = {
        "path_area": simulated_path_area,
        "path_tau": _concatenate(path_tau),
        "path_offset": np.asarray(path_offsets, dtype=np.int64),
        "path_time_offset": np.asarray(path_time_offsets, dtype=np.int64),
        "path_band_count": np.asarray(path_band_counts, dtype=np.int64),
        "path_segment_index": np.asarray(path_segment_indices, dtype=np.int64),
        "observed_path_area": observed_path_area,
        "observed_path_offset": np.asarray(observed_path_offsets, dtype=np.int64),
        "observed_path_tau": _concatenate(observed_path_tau),
        "observed_path_time_offset": np.asarray(
            observed_path_time_offsets, dtype=np.int64
        ),
        "observed_path_band_count": np.asarray(
            observed_path_band_counts, dtype=np.int64
        ),
        "observed_total": observed * scaling.area,
        "simulated_total": simulated * scaling.area,
        "observed_total_acf": observed_total_acf,
        "simulated_total_acf": simulated_total_acf,
        "observed_conservative_acf": observed_conservative_acf,
        "simulated_conservative_acf": simulated_conservative_acf,
        "observed_transfer_rate_acf": observed_transfer_rate_acf,
        "simulated_transfer_rate_acf": simulated_transfer_rate_acf,
        "observed_transfer_rate_lag_time": observed_transfer_rate_lag_times,
        "simulated_transfer_rate_lag_time": simulated_transfer_rate_lag_times,
        "observed_area_msd": _pooled_msd(observed_area_series) * scaling.area**2,
        "simulated_area_msd": _pooled_msd(simulated_area_series) * scaling.area**2,
        "observed_total_msd": _pooled_msd(observed_total_series) * scaling.area**2,
        "simulated_total_msd": _pooled_msd(simulated_total_series) * scaling.area**2,
        "observed_conservative_msd": _pooled_msd(observed_conservative_series)
        * scaling.area**2,
        "simulated_conservative_msd": _pooled_msd(simulated_conservative_series)
        * scaling.area**2,
        "simulated_area_msd_envelope": _grouped_msd_quantiles(
            simulated_area_series, path_band_counts
        )
        * scaling.area**2,
        "simulated_total_msd_envelope": _grouped_msd_quantiles(
            simulated_total_series, [1] * len(simulated_total_series)
        )
        * scaling.area**2,
        "simulated_conservative_msd_envelope": _grouped_msd_quantiles(
            simulated_conservative_series, path_band_counts
        )
        * scaling.area**2,
        "observed_segment_total_acf": segment_total_acf,
        "observed_segment_total_acf_offset": segment_total_acf_offsets,
        "observed_segment_transfer_rate_acf": segment_transfer_acf,
        "observed_segment_transfer_rate_acf_offset": segment_transfer_acf_offsets,
        "simulated_transfer_rate_acf_envelope": _replicate_acf_quantiles(
            replicate_curves
        ),
        "observed_transfer_rate_segment_count": transfer_rate_segment_counts,
        "transfer_rate_discrepancy_null": transfer_rate_discrepancy_null,
        "observed_lag_time": observed_lag_times,
        "simulated_lag_time": simulated_lag_times,
        "observed_increment_covariance": _concatenate(observed_covariances)
        * scaling.area**2,
        "simulated_increment_covariance": _concatenate(simulated_covariances)
        * scaling.area**2,
        "observed_conservative_increment_covariance": _concatenate(
            observed_conservative_covariances
        )
        * scaling.area**2,
        "simulated_conservative_increment_covariance": _concatenate(
            simulated_conservative_covariances
        )
        * scaling.area**2,
        "observed_increment_total_covariance": _concatenate(
            observed_total_increment_covariances
        )
        * scaling.area**2,
        "simulated_increment_total_covariance": _concatenate(
            simulated_total_increment_covariances
        )
        * scaling.area**2,
        "covariance_offset": np.asarray(covariance_offsets, dtype=np.int64),
        "covariance_vector_offset": np.asarray(
            covariance_vector_offsets, dtype=np.int64
        ),
    }
    metrics: dict[str, float | int] = {
        **transfer_rate_discrepancy,
        "negative_area_count": negative,
        "generated_area_count": generated,
        "negative_area_fraction": float(negative / generated) if generated else 0.0,
        "terminated_path_count": terminated,
        "observed_total_mean": float(observed.mean() * scaling.area) if observed.size else float("nan"),
        "simulated_total_mean": float(simulated.mean() * scaling.area) if simulated.size else float("nan"),
        "observed_total_variance": float(observed.var() * scaling.area**2) if observed.size else float("nan"),
        "simulated_total_variance": float(simulated.var() * scaling.area**2) if simulated.size else float("nan"),
        "observed_area_mean": float(observed_area.mean() * scaling.area) if observed_area.size else float("nan"),
        "simulated_area_mean": float(simulated_area.mean() * scaling.area) if simulated_area.size else float("nan"),
        "observed_area_variance": float(observed_area.var() * scaling.area**2) if observed_area.size else float("nan"),
        "simulated_area_variance": float(simulated_area.var() * scaling.area**2) if simulated_area.size else float("nan"),
        "observed_conservative_variance": float(observed_difference.var() * scaling.area**2)
        if observed_difference.size
        else float("nan"),
        "simulated_conservative_variance": float(simulated_difference.var() * scaling.area**2)
        if simulated_difference.size
        else float("nan"),
        "observed_total_relaxation_time": _relaxation_time(
            observed_total_acf, observed_lag_times
        ),
        "simulated_total_relaxation_time": _relaxation_time(
            simulated_total_acf, simulated_lag_times
        ),
        "observed_conservative_relaxation_time": _relaxation_time(
            observed_conservative_acf, observed_lag_times
        ),
        "simulated_conservative_relaxation_time": _relaxation_time(
            simulated_conservative_acf, simulated_lag_times
        ),
    }
    arrays.update(
        {
            "observed_area": observed_path_area,
            "simulated_area": simulated_path_area,
            "observed_conservative": observed_difference * scaling.area,
            "simulated_conservative": simulated_difference * scaling.area,
            "observed_area_increment": _concatenate(observed_increment_values)
            * scaling.area,
            "simulated_area_increment": _concatenate(simulated_increment_values)
            * scaling.area,
        }
    )
    return arrays, metrics


def validate_posterior(
    segments: list[StableSegment],
    lag: int,
    scaling: Scaling,
    normalized_samples: np.ndarray,
    *,
    predictive_draws: int = 200,
    paths_per_segment: int = 200,
    seed: int = 0,
) -> ValidationResult:
    """Validate with frozen scaling and posterior; this function never refits."""
    posterior = _posterior_matrix(normalized_samples)
    one_step_arrays, one_step_metrics = _one_step(
        segments,
        lag,
        scaling,
        posterior,
        predictive_draws,
        seed,
    )
    path_arrays, path_metrics = _paths(
        segments,
        lag,
        scaling,
        posterior,
        paths_per_segment,
        seed + 2,
    )
    return ValidationResult(
        arrays={**one_step_arrays, **path_arrays},
        metrics={**one_step_metrics, **path_metrics},
    )
