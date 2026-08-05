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
    conservative_segment_slope,
    conservative_projection,
    persistent_innovations,
    transition,
    transition_log_density,
)
from ..detection.segments import StableSegment


# Defined once so their trace caches are shared across blocks, validation sets,
# and lags rather than rebuilt on every loop iteration.
_transition_rows = jax.vmap(transition, in_axes=(0, 0, None, None))
_transition_draws = jax.vmap(_transition_rows, in_axes=(None, None, 0, None))
_density_rows = jax.vmap(transition_log_density, in_axes=(0, 0, 0, None, None))
_density_draws = jax.vmap(_density_rows, in_axes=(None, None, None, 0, None))


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
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("posterior samples must have shape (..., 5)")
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
    sigma_b_squared: float,
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
            block.current, block.dt, selected_device, sigma_b_squared
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
                sigma_b_squared,
            )
        )
        predictive_mean = predictions.mean(axis=0)
        transition_draws = generator.integers(0, len(selected), size=len(current))
        predictive_sample = predictions[
            transition_draws, np.arange(len(current))
        ]
        low, high = np.quantile(predictions, [0.025, 0.975], axis=0)
        median_means, median_covariances = _transition_rows(
            block.current, block.dt, median_device, sigma_b_squared
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
        projection = conservative_projection(current.shape[1])
        observed_exchange_parts.append((observed_delta @ projection).ravel())
        predicted_exchange_parts.append((predicted_delta @ projection).ravel())
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
    """Pool total-mode and conservative observed-rate innovations without overlap."""
    series: list[np.ndarray] = []
    tau_p, kappa_total, diffusion_u, diffusion_total, area_star = parameters
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
                persistent_innovations(
                    sequence, jnp.asarray(tau_p), jnp.asarray(diffusion_u)
                )
            )
            series.extend(innovations[:, component] for component in range(innovations.shape[1]))
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


def _pooled_lag_times(series: list[np.ndarray]) -> np.ndarray:
    return _pooled_increment_statistic(
        series, lambda increments: float(np.median(increments))
    )


def _increment_statistics(
    increment: np.ndarray, n_bands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened band, conservative-mode, and band-total covariances."""
    if len(increment) > 1:
        band = np.cov(increment, rowvar=False)
        conservative = np.cov(
            increment @ conservative_projection(n_bands), rowvar=False
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


def _paths(
    segments: list[StableSegment],
    lag: int,
    scaling: Scaling,
    posterior: np.ndarray,
    sigma_b_squared: float,
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
        slope = conservative_segment_slope(normalized)
        if len(sampled_tau) > 1:
            transfer_rate = np.diff(observed_deviation, axis=0) / np.diff(
                sampled_tau
            )[:, None] - slope
            transfer_tau = 0.5 * (sampled_tau[1:] + sampled_tau[:-1])
            observed_transfer_rate_series.extend(
                transfer_rate[:, component] for component in range(segment.n_bands)
            )
            observed_transfer_rate_tau.extend(
                transfer_tau for _ in range(segment.n_bands)
            )
        observed_conservative.append(observed_deviation.ravel())
        observed_conservative_series.extend(
            observed_deviation[:, component] for component in range(segment.n_bands)
        )
        observed_increment = np.diff(sampled_areas, axis=0)
        observed_increment_values.append(observed_increment.ravel())
        projection = conservative_projection(segment.n_bands)
        band, conservative, total = _increment_statistics(
            observed_increment, segment.n_bands
        )
        observed_covariances.append(band)
        observed_conservative_covariances.append(conservative)
        observed_total_increment_covariances.append(total)
        simulated_segment_increments: list[np.ndarray] = []
        parameter_start = segment_index * paths_per_segment
        segment_parameters = selected[
            parameter_start : parameter_start + paths_per_segment
        ]
        paths = np.full(
            (paths_per_segment, *sampled_areas.shape), np.nan, dtype=np.float64
        )
        paths[:, 0] = sampled_areas[0]
        rates = generator.standard_normal(
            (paths_per_segment, segment.n_bands)
        ) @ projection
        rates *= np.sqrt(
            segment_parameters[:, 2] / segment_parameters[:, 0]
        )[:, None]
        slopes = generator.standard_normal(
            (paths_per_segment, segment.n_bands)
        ) @ projection
        slopes *= np.sqrt(sigma_b_squared)
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
            conservative = current[indices] - totals[:, None] / segment.n_bands
            following_totals = (
                totals
                - parameters[:, 1] * (totals - parameters[:, 4]) * step
                + np.sqrt(2.0 * parameters[:, 3] * step)
                * generator.standard_normal(len(indices))
            )
            following = (
                conservative
                + slopes[indices] * step
                + rates[indices] * step
                + following_totals[:, None] / segment.n_bands
            )
            decay = np.exp(-step / parameters[:, 0])
            rate_noise = generator.standard_normal(
                (len(indices), segment.n_bands)
            ) @ projection
            rates[indices] = (
                decay[:, None] * rates[indices]
                + np.sqrt(
                    parameters[:, 2]
                    / parameters[:, 0]
                    * (1.0 - decay**2)
                )[:, None]
                * rate_noise
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
            if len(valid) > 1:
                valid_tau = sampled_tau[: len(valid)]
                transfer_rate = np.diff(deviation, axis=0) / np.diff(
                    valid_tau
                )[:, None] - slopes[path_index]
                transfer_tau = 0.5 * (valid_tau[1:] + valid_tau[:-1])
                simulated_transfer_rate_series.extend(
                    transfer_rate[:, component]
                    for component in range(segment.n_bands)
                )
                simulated_transfer_rate_tau.extend(
                    transfer_tau for _ in range(segment.n_bands)
                )
            if len(valid) > 1:
                increment = np.diff(valid, axis=0)
                simulated_segment_increments.append(increment)
                simulated_increment_values.append(increment.ravel())
        combined_increment = (
            np.concatenate(simulated_segment_increments)
            if simulated_segment_increments
            else np.empty((0, segment.n_bands))
        )
        band, conservative, total = _increment_statistics(
            combined_increment, segment.n_bands
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
    sigma_b_squared: float = 0.0,
    predictive_draws: int = 200,
    paths_per_segment: int = 200,
    seed: int = 0,
) -> ValidationResult:
    """Validate with frozen scaling and posterior; this function never refits."""
    if sigma_b_squared < 0.0 or not np.isfinite(sigma_b_squared):
        raise ValueError("sigma_b_squared must be finite and nonnegative")
    posterior = _posterior_matrix(normalized_samples)
    one_step_arrays, one_step_metrics = _one_step(
        segments,
        lag,
        scaling,
        posterior,
        sigma_b_squared,
        predictive_draws,
        seed,
    )
    path_arrays, path_metrics = _paths(
        segments,
        lag,
        scaling,
        posterior,
        sigma_b_squared,
        paths_per_segment,
        seed + 2,
    )
    return ValidationResult(
        arrays={**one_step_arrays, **path_arrays},
        metrics={**one_step_metrics, **path_metrics},
    )
