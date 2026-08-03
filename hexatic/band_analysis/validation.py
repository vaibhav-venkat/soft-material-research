from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from .model import Scaling, build_transition_blocks, transition, transition_log_density
from .segments import StableSegment


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
    generator = np.random.default_rng(seed + 1)
    observed_parts: list[np.ndarray] = []
    mean_parts: list[np.ndarray] = []
    low_parts: list[np.ndarray] = []
    high_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    residual_covariances: list[np.ndarray] = []
    residual_covariance_offsets = [0]
    log_scores: list[np.ndarray] = []
    covered = 0
    values = 0
    for block in blocks:
        following = np.asarray(block.following)
        transition_rows = jax.vmap(transition, in_axes=(0, 0, None))
        means, covariances = jax.vmap(
            lambda parameters: transition_rows(block.current, block.dt, parameters)
        )(jnp.asarray(selected))
        means_np = np.asarray(means)
        cholesky = np.linalg.cholesky(np.asarray(covariances))
        noise = generator.standard_normal(means_np.shape)
        predictions = means_np + np.einsum("...ij,...j->...i", cholesky, noise)
        density_rows = jax.vmap(
            transition_log_density, in_axes=(0, 0, 0, None)
        )
        draw_logp = np.asarray(
            jax.vmap(
                lambda parameters: density_rows(
                    block.following, block.current, block.dt, parameters
                )
            )(jnp.asarray(selected))
        )
        predictive_mean = predictions.mean(axis=0)
        low, high = np.quantile(predictions, [0.025, 0.975], axis=0)
        median_parameters = np.median(selected, axis=0)
        median_means, median_covariances = transition_rows(
            block.current, block.dt, jnp.asarray(median_parameters)
        )
        block_residuals = np.linalg.solve(
            np.linalg.cholesky(np.asarray(median_covariances)),
            (following - np.asarray(median_means))[..., None],
        )[..., 0]
        residual_parts.append(block_residuals.ravel())
        residual_covariance = (
            np.cov(block_residuals, rowvar=False)
            if len(block_residuals) > 1
            else np.zeros((following.shape[1], following.shape[1]))
        )
        residual_covariances.append(np.atleast_2d(residual_covariance).ravel())
        residual_covariance_offsets.append(
            residual_covariance_offsets[-1] + following.shape[1] ** 2
        )
        observed_parts.append(following.ravel())
        mean_parts.append(predictive_mean.ravel())
        low_parts.append(low.ravel())
        high_parts.append(high.ravel())
        log_scores.append(
            logsumexp(draw_logp, axis=0)
            - np.log(draws)
            - following.shape[1] * np.log(scaling.area)
        )
        covered += int(np.count_nonzero((following >= low) & (following <= high)))
        values += following.size
    observed = np.concatenate(observed_parts) if observed_parts else np.empty(0)
    predicted = np.concatenate(mean_parts) if mean_parts else np.empty(0)
    whitened = np.concatenate(residual_parts) if residual_parts else np.empty(0)
    arrays = {
        "one_step_observed": observed * scaling.area,
        "one_step_mean": predicted * scaling.area,
        "one_step_low": (
            np.concatenate(low_parts) if low_parts else np.empty(0)
        )
        * scaling.area,
        "one_step_high": (
            np.concatenate(high_parts) if high_parts else np.empty(0)
        )
        * scaling.area,
        "whitened_residual": whitened,
        "whitened_residual_acf": _autocorrelation(whitened),
        "whitened_residual_covariance": (
            np.concatenate(residual_covariances)
            if residual_covariances
            else np.empty(0)
        ),
        "residual_covariance_offset": np.asarray(
            residual_covariance_offsets, dtype=np.int64
        ),
    }
    metrics = {
        "predictive_log_score": float(np.mean(np.concatenate(log_scores)))
        if log_scores
        else float("nan"),
        "coverage_95": float(covered / values) if values else float("nan"),
        "residual_mean": float(np.mean(whitened)) if whitened.size else float("nan"),
        "residual_variance": float(np.var(whitened)) if whitened.size else float("nan"),
        "residual_lag1_autocorrelation": (
            float(_autocorrelation(whitened)[1]) if whitened.size > 1 else float("nan")
        ),
        "one_step_rmse": float(np.sqrt(np.mean((observed - predicted) ** 2)) * scaling.area)
        if observed.size
        else float("nan"),
    }
    return arrays, metrics


def _autocorrelation(values: np.ndarray) -> np.ndarray:
    if not values.size:
        return np.empty(0, dtype=np.float64)
    centered = values - values.mean()
    variance = np.dot(centered, centered)
    if variance == 0.0:
        return np.zeros(len(values), dtype=np.float64)
    return np.correlate(centered, centered, mode="full")[len(values) - 1 :] / variance


def _paths(
    segments: list[StableSegment],
    scaling: Scaling,
    posterior: np.ndarray,
    paths_per_segment: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    generator = np.random.default_rng(seed)
    selected = posterior[
        _draw_indices(len(posterior), paths_per_segment * max(len(segments), 1), seed)
    ]
    path_values: list[np.ndarray] = []
    path_offsets = [0]
    negative = 0
    generated = 0
    terminated = 0
    observed_totals: list[np.ndarray] = []
    simulated_totals: list[np.ndarray] = []
    observed_acf: list[np.ndarray] = []
    simulated_acf: list[np.ndarray] = []
    observed_covariances: list[np.ndarray] = []
    simulated_covariances: list[np.ndarray] = []
    covariance_offsets = [0]
    path_band_counts: list[int] = []
    path_segment_indices: list[int] = []
    observed_area_values: list[np.ndarray] = []
    simulated_area_values: list[np.ndarray] = []
    observed_conservative: list[np.ndarray] = []
    simulated_conservative: list[np.ndarray] = []
    for segment_index, segment in enumerate(segments):
        normalized = scaling.normalize_segment(segment)
        observed_total = normalized.areas.sum(axis=1)
        observed_area_values.append(normalized.areas.ravel())
        observed_conservative.append(
            (normalized.areas - normalized.areas.mean(axis=1, keepdims=True)).ravel()
        )
        observed_totals.append(observed_total)
        observed_acf.append(_autocorrelation(observed_total))
        observed_increment = np.diff(normalized.areas, axis=0)
        observed_covariance = (
            np.cov(observed_increment, rowvar=False)
            if len(observed_increment) > 1
            else np.zeros((segment.n_bands, segment.n_bands))
        )
        observed_covariances.append(np.atleast_2d(observed_covariance).ravel())
        simulated_segment_increments: list[np.ndarray] = []
        parameter_start = segment_index * paths_per_segment
        segment_parameters = selected[
            parameter_start : parameter_start + paths_per_segment
        ]
        paths = np.full(
            (paths_per_segment, *normalized.areas.shape), np.nan, dtype=np.float64
        )
        paths[:, 0] = normalized.areas[0]
        stops = np.full(paths_per_segment, len(normalized.areas), dtype=np.int64)
        active = np.ones(paths_per_segment, dtype=bool)
        transition_paths = jax.vmap(transition, in_axes=(0, None, 0))
        for frame in range(len(normalized.areas) - 1):
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
            means, covariances = transition_paths(
                jnp.asarray(current[indices]),
                normalized.tau[frame + 1] - normalized.tau[frame],
                jnp.asarray(segment_parameters[indices]),
            )
            try:
                cholesky = np.linalg.cholesky(np.asarray(covariances))
            except np.linalg.LinAlgError:
                good: list[int] = []
                factors: list[np.ndarray] = []
                for local, covariance in enumerate(np.asarray(covariances)):
                    try:
                        factors.append(np.linalg.cholesky(covariance))
                        good.append(local)
                    except np.linalg.LinAlgError:
                        failed = indices[local]
                        active[failed] = False
                        stops[failed] = frame + 1
                        terminated += 1
                indices = indices[np.asarray(good, dtype=np.int64)]
                means = means[jnp.asarray(good, dtype=jnp.int64)]
                cholesky = np.asarray(factors)
            if not len(indices):
                continue
            noise = generator.standard_normal(np.asarray(means).shape)
            following = np.asarray(means) + np.einsum(
                "...ij,...j->...i", cholesky, noise
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
            path_values.append(valid.ravel() * scaling.area)
            path_offsets.append(path_offsets[-1] + valid.size)
            path_band_counts.append(segment.n_bands)
            path_segment_indices.append(segment_index)
            simulated_area_values.append(valid.ravel())
            simulated_conservative.append(
                (valid - valid.mean(axis=1, keepdims=True)).ravel()
            )
            total = valid.sum(axis=1)
            simulated_totals.append(total)
            simulated_acf.append(_autocorrelation(total))
            if len(valid) > 1:
                simulated_segment_increments.append(np.diff(valid, axis=0))
        combined_increment = (
            np.concatenate(simulated_segment_increments)
            if simulated_segment_increments
            else np.empty((0, segment.n_bands))
        )
        simulated_covariance = (
            np.cov(combined_increment, rowvar=False)
            if len(combined_increment) > 1
            else np.zeros((segment.n_bands, segment.n_bands))
        )
        simulated_covariances.append(np.atleast_2d(simulated_covariance).ravel())
        covariance_offsets.append(covariance_offsets[-1] + segment.n_bands**2)
    observed = np.concatenate(observed_totals) if observed_totals else np.empty(0)
    simulated = np.concatenate(simulated_totals) if simulated_totals else np.empty(0)
    arrays = {
        "path_area": np.concatenate(path_values) if path_values else np.empty(0),
        "path_offset": np.asarray(path_offsets, dtype=np.int64),
        "path_band_count": np.asarray(path_band_counts, dtype=np.int64),
        "path_segment_index": np.asarray(path_segment_indices, dtype=np.int64),
        "observed_total": observed * scaling.area,
        "simulated_total": simulated * scaling.area,
        "observed_total_acf": np.concatenate(observed_acf) if observed_acf else np.empty(0),
        "simulated_total_acf": np.concatenate(simulated_acf) if simulated_acf else np.empty(0),
        "observed_increment_covariance": (
            np.concatenate(observed_covariances) * scaling.area**2
            if observed_covariances
            else np.empty(0)
        ),
        "simulated_increment_covariance": (
            np.concatenate(simulated_covariances) * scaling.area**2
            if simulated_covariances
            else np.empty(0)
        ),
        "covariance_offset": np.asarray(covariance_offsets, dtype=np.int64),
    }
    observed_area = (
        np.concatenate(observed_area_values) if observed_area_values else np.empty(0)
    )
    simulated_area = (
        np.concatenate(simulated_area_values) if simulated_area_values else np.empty(0)
    )
    observed_difference = (
        np.concatenate(observed_conservative) if observed_conservative else np.empty(0)
    )
    simulated_difference = (
        np.concatenate(simulated_conservative) if simulated_conservative else np.empty(0)
    )
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
    }
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
        segments, lag, scaling, posterior, predictive_draws, seed
    )
    path_arrays, path_metrics = _paths(
        segments, scaling, posterior, paths_per_segment, seed + 2
    )
    return ValidationResult(
        arrays={**one_step_arrays, **path_arrays},
        metrics={**one_step_metrics, **path_metrics},
    )
