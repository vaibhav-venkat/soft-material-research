from __future__ import annotations

from functools import partial
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from .common import (
    area_diffusion_model,
    area_drift_constant_model,
    area_drift_cubic_model,
)

@partial(jax.jit, static_argnames=("lag_count", "bin_count"))
def _area_bin_perimeter_reductions(
    area: jax.Array,
    perimeter: jax.Array,
    stable: jax.Array,
    segments: jax.Array,
    edges: jax.Array,
    *,
    lag_count: int,
    bin_count: int,
) -> tuple[jax.Array, jax.Array]:
    n_time = area.shape[0]
    starts = jnp.arange(n_time)
    lags = jnp.arange(1, lag_count + 1)[:, jnp.newaxis]
    ends = starts[jnp.newaxis, :] + lags
    valid_time = ends < n_time
    ends = jnp.minimum(ends, n_time - 1)
    valid = (
        valid_time[..., jnp.newaxis]
        & stable[jnp.newaxis, :, :]
        & stable[ends]
        & (segments[jnp.newaxis, :, :] == segments[ends])
        & jnp.isfinite(area[jnp.newaxis, :, :])
        & jnp.isfinite(area[ends])
        & jnp.isfinite(perimeter[jnp.newaxis, :, :])
    )
    bins = jnp.searchsorted(edges, area, side="right") - 1
    bins = jnp.clip(bins, 0, bin_count - 1)
    offsets = jnp.arange(lag_count)[:, jnp.newaxis, jnp.newaxis] * bin_count
    group_count = lag_count * bin_count
    groups = jnp.where(
        valid, offsets + bins[jnp.newaxis, :, :], group_count
    ).ravel()
    counts = jnp.bincount(groups, length=group_count + 1)[:-1].reshape(
        lag_count, bin_count
    )
    sums = jnp.bincount(
        groups,
        weights=jnp.where(
            valid,
            perimeter[jnp.newaxis, :, :],
            0.0,
        ).ravel(),
        length=group_count + 1,
    )[:-1].reshape(lag_count, bin_count)
    return counts, sums


def add_area_diffusion_fit(tensors: dict[str, np.ndarray]) -> None:
    """Compare four conditional area-diffusion models at every lag."""
    centers = tensors["dynamics_area_bin_center"]
    diffusion = tensors["dynamics_area_diffusion"]
    counts = tensors["dynamics_area_count"]
    mean_lags = tensors["dynamics_area_mean_physical_lag"]
    lag_count = diffusion.shape[0]
    d0_values = np.full(lag_count, np.nan)
    gamma_values = np.full(lag_count, np.nan)
    alpha_values = np.full(lag_count, np.nan)
    r_squared = np.full(lag_count, np.nan)
    full_rmse = np.full(lag_count, np.nan)
    physical_lags = np.full(lag_count, np.nan)
    fit_valid = np.zeros(lag_count, dtype=bool)
    constant_valid = np.zeros(lag_count, dtype=bool)
    constant_d0 = np.full(lag_count, np.nan)
    constant_rmse = np.full(lag_count, np.nan)
    sqrt_valid = np.zeros(lag_count, dtype=bool)
    sqrt_gamma = np.full(lag_count, np.nan)
    sqrt_rmse = np.full(lag_count, np.nan)
    perimeter_valid = np.zeros(lag_count, dtype=bool)
    perimeter_gamma = np.full(lag_count, np.nan)
    perimeter_rmse = np.full(lag_count, np.nan)
    mean_perimeter = np.full(diffusion.shape, np.nan)

    finite_edges = tensors["dynamics_area_bin_edges"]
    finite_edges = finite_edges[np.isfinite(finite_edges)]
    if lag_count and finite_edges.size >= 2:
        bin_count = len(finite_edges) - 1
        perimeter_reduced = jax.device_get(
            _area_bin_perimeter_reductions(
                jnp.asarray(tensors["area"]),
                jnp.asarray(tensors["interface_length"]),
                jnp.asarray(tensors["stable"]),
                jnp.asarray(tensors["segment_id"]),
                jnp.asarray(finite_edges),
                lag_count=lag_count,
                bin_count=bin_count,
            )
        )
        perimeter_counts = np.asarray(perimeter_reduced[0])
        perimeter_sums = np.asarray(perimeter_reduced[1])
        mean_perimeter[:, :bin_count] = np.where(
            perimeter_counts > 0,
            perimeter_sums / np.maximum(perimeter_counts, 1),
            np.nan,
        )

    for lag_index in range(lag_count):
        usable = (
            np.isfinite(centers)
            & np.isfinite(diffusion[lag_index])
            & np.isfinite(mean_lags[lag_index])
            & (centers > 0.0)
            & (counts[lag_index] > 0)
        )
        if not np.any(usable):
            continue
        area = centers[usable]
        observed = diffusion[lag_index, usable]
        weights = counts[lag_index, usable].astype(float)
        physical_lags[lag_index] = float(
            np.average(mean_lags[lag_index, usable], weights=weights)
        )

        fitted_d0 = float(np.average(observed, weights=weights))
        constant_valid[lag_index] = True
        constant_d0[lag_index] = fitted_d0
        constant_rmse[lag_index] = _weighted_rmse(
            observed, np.full(observed.shape, fitted_d0), weights
        )

        sqrt_area = np.sqrt(area)
        sqrt_denominator = float(np.sum(weights * sqrt_area**2))
        if sqrt_denominator > 0.0:
            fitted_gamma = float(
                np.sum(weights * sqrt_area * observed) / sqrt_denominator
            )
            sqrt_valid[lag_index] = True
            sqrt_gamma[lag_index] = fitted_gamma
            sqrt_rmse[lag_index] = _weighted_rmse(
                observed, fitted_gamma * sqrt_area, weights
            )

        perimeter = mean_perimeter[lag_index, usable]
        perimeter_usable = np.isfinite(perimeter)
        if np.any(perimeter_usable):
            perimeter_weights = weights[perimeter_usable]
            perimeter_values = perimeter[perimeter_usable]
            perimeter_observed = observed[perimeter_usable]
            denominator = float(
                np.sum(perimeter_weights * perimeter_values**2)
            )
            if denominator > 0.0:
                fitted_gamma_p = float(
                    np.sum(
                        perimeter_weights
                        * perimeter_values
                        * perimeter_observed
                    )
                    / denominator
                )
                perimeter_valid[lag_index] = True
                perimeter_gamma[lag_index] = fitted_gamma_p
                perimeter_rmse[lag_index] = _weighted_rmse(
                    perimeter_observed,
                    fitted_gamma_p * perimeter_values,
                    perimeter_weights,
                )

        if np.count_nonzero(usable) < 4:
            continue
        area_scale = float(np.median(area))

        def scaled_model(
            values: np.ndarray,
            d0: float,
            scaled_gamma: float,
            alpha: float,
        ) -> np.ndarray:
            return d0 + scaled_gamma * (values / area_scale) ** alpha

        d0_guess = max(0.0, 0.5 * float(np.min(observed)))
        gamma_guess = max(
            np.finfo(float).eps, float(np.max(observed) - d0_guess)
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                parameters, _ = curve_fit(
                    scaled_model,
                    area,
                    observed,
                    p0=(d0_guess, gamma_guess, 1.0),
                    sigma=1.0 / np.sqrt(weights),
                    absolute_sigma=False,
                    bounds=((0.0, 0.0, -5.0), (np.inf, np.inf, 5.0)),
                    maxfev=50_000,
                )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        d0, scaled_gamma, alpha = (float(value) for value in parameters)
        gamma = scaled_gamma / area_scale**alpha
        predicted = area_diffusion_model(area, d0, gamma, alpha)
        residual_sum = float(np.sum((observed - predicted) ** 2))
        total_sum = float(np.sum((observed - np.mean(observed)) ** 2))
        d0_values[lag_index] = d0
        gamma_values[lag_index] = gamma
        alpha_values[lag_index] = alpha
        r_squared[lag_index] = (
            1.0 - residual_sum / total_sum if total_sum > 0.0 else np.nan
        )
        full_rmse[lag_index] = _weighted_rmse(observed, predicted, weights)
        fit_valid[lag_index] = True

    tensors["area_diffusion_fit_valid"] = fit_valid
    tensors["area_diffusion_fit_d0"] = d0_values
    tensors["area_diffusion_fit_gamma"] = gamma_values
    tensors["area_diffusion_fit_alpha"] = alpha_values
    tensors["area_diffusion_fit_r_squared"] = r_squared
    tensors["area_diffusion_fit_rmse"] = full_rmse
    tensors["area_diffusion_fit_physical_lag"] = physical_lags
    tensors["area_diffusion_bin_mean_perimeter"] = mean_perimeter
    tensors["area_diffusion_constant_fit_valid"] = constant_valid
    tensors["area_diffusion_constant_fit_d0"] = constant_d0
    tensors["area_diffusion_constant_fit_rmse"] = constant_rmse
    tensors["area_diffusion_sqrt_fit_valid"] = sqrt_valid
    tensors["area_diffusion_sqrt_fit_gamma"] = sqrt_gamma
    tensors["area_diffusion_sqrt_fit_rmse"] = sqrt_rmse
    tensors["area_diffusion_perimeter_fit_valid"] = perimeter_valid
    tensors["area_diffusion_perimeter_fit_gamma_p"] = perimeter_gamma
    tensors["area_diffusion_perimeter_fit_rmse"] = perimeter_rmse


def _weighted_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sqrt(np.average((observed - predicted) ** 2, weights=weights)))


def add_area_geometry_fits(tensors: dict[str, np.ndarray]) -> None:
    """Fit stable-band area-width and perimeter-area linear relationships."""
    stable = tensors["stable"]
    area = tensors["area"]
    width = tensors["mean_width"]
    perimeter = tensors["interface_length"]

    width_usable = stable & np.isfinite(area) & np.isfinite(width)
    width_count = int(np.count_nonzero(width_usable))
    width_valid = width_count >= 2 and np.ptp(width[width_usable]) > 0.0
    area_a0 = area_a1 = area_rmse = np.nan
    if width_valid:
        area_a1, area_a0 = (
            float(value)
            for value in np.polyfit(width[width_usable], area[width_usable], 1)
        )
        predicted = area_a0 + area_a1 * width[width_usable]
        area_rmse = float(np.sqrt(np.mean((area[width_usable] - predicted) ** 2)))

    perimeter_usable = stable & np.isfinite(area) & np.isfinite(perimeter)
    perimeter_count = int(np.count_nonzero(perimeter_usable))
    perimeter_valid = (
        perimeter_count >= 2 and np.ptp(area[perimeter_usable]) > 0.0
    )
    perimeter_p0 = perimeter_ba = perimeter_rmse = np.nan
    if perimeter_valid:
        perimeter_ba, perimeter_p0 = (
            float(value)
            for value in np.polyfit(
                area[perimeter_usable], perimeter[perimeter_usable], 1
            )
        )
        predicted = perimeter_p0 + perimeter_ba * area[perimeter_usable]
        perimeter_rmse = float(
            np.sqrt(np.mean((perimeter[perimeter_usable] - predicted) ** 2))
        )

    tensors["area_width_fit_valid"] = np.asarray(width_valid)
    tensors["area_width_fit_count"] = np.asarray(width_count, dtype=np.int64)
    tensors["area_width_fit_a0"] = np.asarray(area_a0)
    tensors["area_width_fit_a1"] = np.asarray(area_a1)
    tensors["area_width_fit_rmse"] = np.asarray(area_rmse)
    tensors["perimeter_area_fit_valid"] = np.asarray(perimeter_valid)
    tensors["perimeter_area_fit_count"] = np.asarray(
        perimeter_count, dtype=np.int64
    )
    tensors["perimeter_area_fit_p0"] = np.asarray(perimeter_p0)
    tensors["perimeter_area_fit_ba"] = np.asarray(perimeter_ba)
    tensors["perimeter_area_fit_rmse"] = np.asarray(perimeter_rmse)


def add_area_increment_conservation(tensors: dict[str, np.ndarray]) -> None:
    """Compute lag-1 spatial covariance and the lag-2 conservation ratio."""
    area = tensors["area"]
    valid = tensors["valid"]
    segments = tensors["segment_id"]
    positions = tensors["wrapped_axial_position"]
    event_count = tensors["frame_event_count"]

    def grouped_increments(frame_lag: int) -> dict[int, np.ndarray]:
        grouped: dict[int, list[np.ndarray]] = {}
        for time_index in range(max(0, area.shape[0] - frame_lag)):
            end_index = time_index + frame_lag
            if np.any(event_count[time_index + 1 : end_index + 1] != 0):
                continue
            selected = np.flatnonzero(valid[time_index])
            if selected.size < 2:
                continue
            interval_valid = True
            for sampled_index in range(time_index + 1, end_index + 1):
                interval_valid &= (
                    np.count_nonzero(valid[sampled_index]) == selected.size
                    and np.all(valid[sampled_index, selected])
                    and np.array_equal(
                        segments[time_index, selected],
                        segments[sampled_index, selected],
                    )
                )
            if not interval_valid:
                continue
            ordered = selected[np.argsort(positions[time_index, selected])]
            increments = area[end_index, ordered] - area[time_index, ordered]
            if np.all(np.isfinite(increments)):
                grouped.setdefault(int(selected.size), []).append(increments)
        return {
            band_count: np.stack(samples)
            for band_count, samples in grouped.items()
            if len(samples) >= 2
        }

    covariance_groups = grouped_increments(1)
    conservation_groups = grouped_increments(2)
    maximum_band_count = max(covariance_groups, default=0)
    covariance_sum = np.zeros(maximum_band_count)
    covariance_count = np.zeros(maximum_band_count, dtype=np.int64)
    for band_count, group in covariance_groups.items():
        samples = jnp.asarray(group)
        centered = samples - jnp.mean(samples)
        covariance_values = np.asarray(
            jax.device_get(
                jnp.stack(
                    [
                        jnp.mean(
                            centered
                            * jnp.roll(centered, -separation, axis=1)
                        )
                        for separation in range(band_count)
                    ]
                )
            )
        )
        pair_count = int(samples.shape[0] * samples.shape[1])
        covariance_sum[:band_count] += covariance_values * pair_count
        covariance_count[:band_count] += pair_count

    band_counts = np.asarray(sorted(conservation_groups), dtype=np.int64)
    r_cons_by_count = np.full(len(band_counts), np.nan)
    sample_count = np.zeros(len(band_counts), dtype=np.int64)
    total_numerator = 0.0
    total_denominator = 0.0
    for group_index, band_count in enumerate(band_counts):
        samples = jnp.asarray(conservation_groups[int(band_count)])
        numerator, denominator = jax.device_get(
            (
                jnp.var(jnp.sum(samples, axis=1)),
                jnp.sum(jnp.var(samples, axis=0)),
            )
        )
        sample_count[group_index] = int(samples.shape[0])
        if float(denominator) > 0.0:
            r_cons_by_count[group_index] = float(numerator / denominator)
            total_numerator += float(samples.shape[0] * numerator)
            total_denominator += float(samples.shape[0] * denominator)

    covariance = np.where(
        covariance_count > 0,
        covariance_sum / np.maximum(covariance_count, 1),
        np.nan,
    )
    tensors["area_increment_covariance_frame_lag"] = np.asarray(
        1, dtype=np.int64
    )
    tensors["area_increment_r_cons_frame_lag"] = np.asarray(2, dtype=np.int64)
    tensors["area_increment_covariance_separation"] = np.arange(
        maximum_band_count, dtype=np.int64
    )
    tensors["area_increment_covariance"] = covariance
    tensors["area_increment_covariance_count"] = covariance_count
    tensors["area_increment_c1"] = np.asarray(
        covariance[1] if covariance.size > 1 else np.nan
    )
    tensors["area_increment_r_cons"] = np.asarray(
        total_numerator / total_denominator
        if total_denominator > 0.0
        else np.nan
    )
    tensors["area_increment_r_cons_band_count"] = band_counts
    tensors["area_increment_r_cons_by_band_count"] = r_cons_by_count
    tensors["area_increment_r_cons_sample_count"] = sample_count


def add_area_drift_fits(tensors: dict[str, np.ndarray]) -> None:
    """Fit the constant and cubic area-drift models."""
    centers = tensors["dynamics_area_bin_center"]
    drift = tensors["dynamics_area_drift"]
    counts = tensors["dynamics_area_count"]
    mean_lags = tensors["dynamics_area_mean_physical_lag"]
    lag_count = drift.shape[0]

    outputs = {
        "area_drift_fit_physical_lag": np.full(lag_count, np.nan),
        "area_drift_constant_fit_valid": np.zeros(lag_count, dtype=bool),
        "area_drift_constant_fit_nu": np.full(lag_count, np.nan),
        "area_drift_constant_fit_rmse": np.full(lag_count, np.nan),
        "area_drift_cubic_fit_valid": np.zeros(lag_count, dtype=bool),
        "area_drift_cubic_fit_c1": np.full(lag_count, np.nan),
        "area_drift_cubic_fit_c3": np.full(lag_count, np.nan),
        "area_drift_cubic_fit_area_star": np.full(lag_count, np.nan),
        "area_drift_cubic_fit_rmse": np.full(lag_count, np.nan),
    }

    for lag_index in range(lag_count):
        usable = (
            np.isfinite(centers)
            & np.isfinite(drift[lag_index])
            & np.isfinite(mean_lags[lag_index])
            & (counts[lag_index] > 0)
        )
        if not np.any(usable):
            continue
        area = centers[usable]
        observed = drift[lag_index, usable]
        weights = counts[lag_index, usable].astype(float)
        sigma = 1.0 / np.sqrt(weights)
        outputs["area_drift_fit_physical_lag"][lag_index] = float(
            np.average(mean_lags[lag_index, usable], weights=weights)
        )

        nu = -float(np.average(observed, weights=weights))
        constant_prediction = area_drift_constant_model(area, nu)
        outputs["area_drift_constant_fit_valid"][lag_index] = True
        outputs["area_drift_constant_fit_nu"][lag_index] = nu
        outputs["area_drift_constant_fit_rmse"][lag_index] = _weighted_rmse(
            observed, constant_prediction, weights
        )

        if area.size < 3 or np.ptp(area) <= 0.0:
            continue
        slope, _ = (
            float(value)
            for value in np.polyfit(area, observed, 1, w=np.sqrt(weights))
        )
        area_scale = float(np.ptp(area))
        initial_c1 = -slope
        initial_area_star = float(np.average(area, weights=weights))

        def scaled_cubic_model(
            values: np.ndarray,
            scaled_c1: float,
            scaled_c3: float,
            area_star: float,
        ) -> np.ndarray:
            centered = (values - area_star) / area_scale
            return -scaled_c1 * centered - scaled_c3 * centered**3

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                cubic_parameters, _ = curve_fit(
                    scaled_cubic_model,
                    area,
                    observed,
                    p0=(initial_c1 * area_scale, 0.0, initial_area_star),
                    sigma=sigma,
                    absolute_sigma=False,
                    method="trf",
                    x_scale="jac",
                    max_nfev=50_000,
                )
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        scaled_c1, scaled_c3, area_star = (
            float(value) for value in cubic_parameters
        )
        c1 = scaled_c1 / area_scale
        c3 = scaled_c3 / area_scale**3
        cubic_prediction = area_drift_cubic_model(area, c1, c3, area_star)
        outputs["area_drift_cubic_fit_valid"][lag_index] = True
        outputs["area_drift_cubic_fit_c1"][lag_index] = c1
        outputs["area_drift_cubic_fit_c3"][lag_index] = c3
        outputs["area_drift_cubic_fit_area_star"][lag_index] = area_star
        outputs["area_drift_cubic_fit_rmse"][lag_index] = _weighted_rmse(
            observed, cubic_prediction, weights
        )

    tensors.update(outputs)


