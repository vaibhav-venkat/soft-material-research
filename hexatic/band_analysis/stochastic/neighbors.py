from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .common import (
    MAX_FRAME_LAG,
    MIN_BIN_SAMPLES,
    PREFERRED_BIN_SAMPLES,
    TARGET_BINS,
    _conditional_reductions,
    _quantile_edges,
    _weighted_rmse,
)

def _neighbor_area_contrast_and_mean(
    area: np.ndarray,
    wrapped_positions: np.ndarray,
    stable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    contrast = np.full(area.shape, np.nan)
    local_average = np.full(area.shape, np.nan)
    for time_index in range(area.shape[0]):
        selected = np.flatnonzero(
            stable[time_index]
            & np.isfinite(area[time_index])
            & np.isfinite(wrapped_positions[time_index])
        )
        if selected.size < 2:
            continue
        ordered = selected[
            np.argsort(wrapped_positions[time_index, selected])
        ]
        previous = np.roll(ordered, 1)
        following = np.roll(ordered, -1)
        contrast[time_index, ordered] = area[time_index, ordered] - 0.5 * (
            area[time_index, previous] + area[time_index, following]
        )
        local_average[time_index, ordered] = (
            area[time_index, previous]
            + area[time_index, ordered]
            + area[time_index, following]
        ) / 3.0
    return contrast, local_average


def neighbor_relative_area(
    area: np.ndarray,
    wrapped_positions: np.ndarray,
    stable: np.ndarray,
) -> np.ndarray:
    """Return each stable band's raw area contrast with cyclic neighbors."""
    contrast, _ = _neighbor_area_contrast_and_mean(
        area, wrapped_positions, stable
    )
    return contrast


def normalized_neighbor_relative_area(
    area: np.ndarray,
    wrapped_positions: np.ndarray,
    stable: np.ndarray,
) -> np.ndarray:
    """Return each stable band's dimensionless contrast with cyclic neighbors."""
    contrast, local_average = _neighbor_area_contrast_and_mean(
        area, wrapped_positions, stable
    )
    relative = np.full(area.shape, np.nan)
    usable = np.isfinite(contrast) & (local_average != 0.0)
    relative[usable] = contrast[usable] / local_average[usable]
    return relative


def _cyclic_neighbor_slots(
    wrapped_positions: np.ndarray,
    stable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every stable band to its previous and next wrapped-order neighbors."""
    previous = np.full(stable.shape, -1, dtype=np.int64)
    following = np.full(stable.shape, -1, dtype=np.int64)
    for time_index in range(stable.shape[0]):
        selected = np.flatnonzero(
            stable[time_index]
            & np.isfinite(wrapped_positions[time_index])
        )
        if selected.size < 2:
            continue
        ordered = selected[
            np.argsort(wrapped_positions[time_index, selected])
        ]
        previous[time_index, ordered] = np.roll(ordered, 1)
        following[time_index, ordered] = np.roll(ordered, -1)
    return previous, following


@partial(jax.jit, static_argnames=("lag_count", "bin_count"))
def _neighbor_relative_area_change_reductions(
    area: jax.Array,
    conditioning: jax.Array,
    stable: jax.Array,
    segments: jax.Array,
    times: jax.Array,
    previous: jax.Array,
    following: jax.Array,
    edges: jax.Array,
    *,
    lag_count: int,
    bin_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reduce Delta(delta A) rates for every lag and conditioning bin."""
    n_time, n_tracks = area.shape
    starts = jnp.arange(n_time)
    lags = jnp.arange(1, lag_count + 1)[:, jnp.newaxis]
    ends = starts[jnp.newaxis, :] + lags
    valid_time = ends < n_time
    ends = jnp.minimum(ends, n_time - 1)

    safe_previous = jnp.maximum(previous, 0)
    safe_following = jnp.maximum(following, 0)
    initial_previous_area = jnp.take_along_axis(area, safe_previous, axis=1)
    initial_following_area = jnp.take_along_axis(area, safe_following, axis=1)
    final_area = area[ends]
    final_previous_area = jnp.take_along_axis(
        final_area, safe_previous[jnp.newaxis, :, :], axis=2
    )
    final_following_area = jnp.take_along_axis(
        final_area, safe_following[jnp.newaxis, :, :], axis=2
    )

    final_stable = stable[ends]
    final_previous_stable = jnp.take_along_axis(
        final_stable, safe_previous[jnp.newaxis, :, :], axis=2
    )
    final_following_stable = jnp.take_along_axis(
        final_stable, safe_following[jnp.newaxis, :, :], axis=2
    )
    initial_previous_segments = jnp.take_along_axis(
        segments, safe_previous, axis=1
    )
    initial_following_segments = jnp.take_along_axis(
        segments, safe_following, axis=1
    )
    final_segments = segments[ends]
    final_previous_segments = jnp.take_along_axis(
        final_segments, safe_previous[jnp.newaxis, :, :], axis=2
    )
    final_following_segments = jnp.take_along_axis(
        final_segments, safe_following[jnp.newaxis, :, :], axis=2
    )

    delta_tau = times[ends] - times[jnp.newaxis, :]
    valid = (
        valid_time[..., jnp.newaxis]
        & stable[jnp.newaxis, :, :]
        & (previous[jnp.newaxis, :, :] >= 0)
        & (following[jnp.newaxis, :, :] >= 0)
        & final_stable
        & final_previous_stable
        & final_following_stable
        & (
            segments[jnp.newaxis, :, :]
            == final_segments
        )
        & (
            initial_previous_segments[jnp.newaxis, :, :]
            == final_previous_segments
        )
        & (
            initial_following_segments[jnp.newaxis, :, :]
            == final_following_segments
        )
        & jnp.isfinite(conditioning[jnp.newaxis, :, :])
        & (delta_tau[..., jnp.newaxis] > 0.0)
    )
    central_increment = final_area - area[jnp.newaxis, :, :]
    previous_increment = (
        final_previous_area - initial_previous_area[jnp.newaxis, :, :]
    )
    following_increment = (
        final_following_area - initial_following_area[jnp.newaxis, :, :]
    )
    relative_increment = central_increment - 0.5 * (
        previous_increment + following_increment
    )
    valid &= jnp.isfinite(relative_increment)

    bins = jnp.searchsorted(edges, conditioning, side="right") - 1
    bins = jnp.clip(bins, 0, bin_count - 1)
    lag_offsets = (
        jnp.arange(lag_count)[:, jnp.newaxis, jnp.newaxis] * bin_count
    )
    group_count = lag_count * bin_count
    groups = jnp.where(
        valid,
        lag_offsets + bins[jnp.newaxis, :, :],
        group_count,
    ).ravel()
    counts = jnp.bincount(groups, length=group_count + 1)[:-1].reshape(
        lag_count, bin_count
    )

    def reduce(weights: jax.Array) -> jax.Array:
        return jnp.bincount(
            groups,
            weights=jnp.where(valid, weights, 0.0).ravel(),
            length=group_count + 1,
        )[:-1].reshape(lag_count, bin_count)

    safe_lags = jnp.where(valid, delta_tau[..., jnp.newaxis], 1.0)
    sum_rate = reduce(relative_increment / safe_lags)
    sum_lag = reduce(
        jnp.broadcast_to(delta_tau[..., jnp.newaxis], valid.shape)
    )
    return counts, sum_rate, sum_lag


def _add_neighbor_conditioned_area_drift(
    tensors: dict[str, np.ndarray],
    conditioning: np.ndarray,
    prefix: str,
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    target_bins: int = TARGET_BINS,
    minimum_samples: int = MIN_BIN_SAMPLES,
    preferred_bin_samples: int = PREFERRED_BIN_SAMPLES,
) -> None:
    if (
        max_frame_lag < 1
        or target_bins < 1
        or minimum_samples < 1
        or preferred_bin_samples < minimum_samples
    ):
        raise ValueError("neighbor-area controls must be positive")
    area = tensors["area"]
    stable = tensors["stable"]
    edges = _quantile_edges(
        conditioning[stable & np.isfinite(conditioning)],
        target_bins,
        preferred_bin_samples,
    )
    lag_count = min(max_frame_lag, max(0, area.shape[0] - 1))
    stored_edges = np.full(target_bins + 1, np.nan)
    centers = np.full(target_bins, np.nan)
    drift = np.full((lag_count, target_bins), np.nan)
    counts = np.zeros((lag_count, target_bins), dtype=np.int64)
    mean_lag = np.full((lag_count, target_bins), np.nan)
    if edges.size:
        bin_count = len(edges) - 1
        stored_edges[: len(edges)] = edges
        centers[:bin_count] = 0.5 * (edges[:-1] + edges[1:])
        if lag_count:
            reduced = jax.device_get(
                _conditional_reductions(
                    jnp.asarray(area),
                    jnp.asarray(conditioning),
                    jnp.asarray(stable),
                    jnp.asarray(tensors["segment_id"]),
                    jnp.asarray(tensors["physical_time"]),
                    jnp.asarray(edges),
                    lag_count=lag_count,
                    bin_count=bin_count,
                )
            )
            reduced_counts = np.asarray(reduced[0], dtype=np.int64)
            sum_rate = np.asarray(reduced[1])
            sum_lag = np.asarray(reduced[4])
            usable = reduced_counts >= minimum_samples
            safe_counts = np.maximum(reduced_counts, 1)
            counts[:, :bin_count] = reduced_counts
            drift[:, :bin_count] = np.where(
                usable,
                sum_rate / safe_counts,
                np.nan,
            )
            mean_lag[:, :bin_count] = np.where(
                usable,
                sum_lag / safe_counts,
                np.nan,
            )
    tensors[f"{prefix}_bin_edges"] = stored_edges
    tensors[f"{prefix}_bin_center"] = centers
    tensors[f"{prefix}_drift"] = drift
    tensors[f"{prefix}_count"] = counts
    tensors[f"{prefix}_mean_physical_lag"] = mean_lag


def add_neighbor_relative_area_drifts(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    target_bins: int = TARGET_BINS,
    minimum_samples: int = MIN_BIN_SAMPLES,
    preferred_bin_samples: int = PREFERRED_BIN_SAMPLES,
) -> None:
    """Estimate area drift conditioned on raw and normalized neighbor contrast."""
    area = tensors["area"]
    positions = tensors["wrapped_axial_position"]
    stable = tensors["stable"]
    contrast, local_average = _neighbor_area_contrast_and_mean(
        area, positions, stable
    )
    normalized = np.full(area.shape, np.nan)
    usable = np.isfinite(contrast) & (local_average != 0.0)
    normalized[usable] = contrast[usable] / local_average[usable]
    tensors["neighbor_relative_area"] = contrast
    tensors["normalized_neighbor_relative_area"] = normalized
    controls = {
        "max_frame_lag": max_frame_lag,
        "target_bins": target_bins,
        "minimum_samples": minimum_samples,
        "preferred_bin_samples": preferred_bin_samples,
    }
    _add_neighbor_conditioned_area_drift(
        tensors,
        contrast,
        "dynamics_neighbor_relative_area",
        **controls,
    )
    _add_neighbor_conditioned_area_drift(
        tensors,
        normalized,
        "dynamics_normalized_neighbor_relative_area",
        **controls,
    )


def add_neighbor_relative_area_fit(tensors: dict[str, np.ndarray]) -> None:
    """Fit ``F_deltaA = lambda * deltaA`` through the origin for each lag."""
    prefix = "dynamics_neighbor_relative_area"
    centers = tensors[f"{prefix}_bin_center"]
    drift = tensors[f"{prefix}_drift"]
    counts = tensors[f"{prefix}_count"]
    mean_lags = tensors[f"{prefix}_mean_physical_lag"]
    lag_count = drift.shape[0]
    valid = np.zeros(lag_count, dtype=bool)
    slope = np.full(lag_count, np.nan)
    rmse = np.full(lag_count, np.nan)
    physical_lag = np.full(lag_count, np.nan)
    for lag_index in range(lag_count):
        usable = (
            np.isfinite(centers)
            & np.isfinite(drift[lag_index])
            & np.isfinite(mean_lags[lag_index])
            & (counts[lag_index] > 0)
        )
        if np.count_nonzero(usable) < 2:
            continue
        values = centers[usable]
        observed = drift[lag_index, usable]
        weights = counts[lag_index, usable].astype(float)
        denominator = float(np.sum(weights * values**2))
        if denominator <= 0.0:
            continue
        fitted_slope = float(np.sum(weights * values * observed) / denominator)
        predicted = fitted_slope * values
        valid[lag_index] = True
        slope[lag_index] = fitted_slope
        rmse[lag_index] = _weighted_rmse(observed, predicted, weights)
        physical_lag[lag_index] = float(
            np.average(mean_lags[lag_index, usable], weights=weights)
        )
    tensors["neighbor_relative_area_fit_valid"] = valid
    tensors["neighbor_relative_area_fit_lambda"] = slope
    tensors["neighbor_relative_area_fit_rmse"] = rmse
    tensors["neighbor_relative_area_fit_physical_lag"] = physical_lag


def add_neighbor_relative_area_change_drift(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    minimum_samples: int = MIN_BIN_SAMPLES,
) -> None:
    """Estimate ``<Delta(delta A_i) | delta A_i> / Delta tau``."""
    if max_frame_lag < 1 or minimum_samples < 1:
        raise ValueError("relative-area change controls must be positive")
    area = tensors["area"]
    lag_count = min(max_frame_lag, max(0, area.shape[0] - 1))
    stored_edges = tensors["dynamics_neighbor_relative_area_bin_edges"]
    edges = stored_edges[np.isfinite(stored_edges)]
    target_bins = len(tensors["dynamics_neighbor_relative_area_bin_center"])
    drift = np.full((lag_count, target_bins), np.nan)
    counts = np.zeros((lag_count, target_bins), dtype=np.int64)
    mean_lag = np.full((lag_count, target_bins), np.nan)
    if edges.size >= 2 and lag_count:
        bin_count = len(edges) - 1
        previous, following = _cyclic_neighbor_slots(
            tensors["wrapped_axial_position"], tensors["stable"]
        )
        reduced = jax.device_get(
            _neighbor_relative_area_change_reductions(
                jnp.asarray(area),
                jnp.asarray(tensors["neighbor_relative_area"]),
                jnp.asarray(tensors["stable"]),
                jnp.asarray(tensors["segment_id"]),
                jnp.asarray(tensors["physical_time"]),
                jnp.asarray(previous),
                jnp.asarray(following),
                jnp.asarray(edges),
                lag_count=lag_count,
                bin_count=bin_count,
            )
        )
        reduced_counts = np.asarray(reduced[0], dtype=np.int64)
        sum_rate = np.asarray(reduced[1])
        sum_lag = np.asarray(reduced[2])
        usable = reduced_counts >= minimum_samples
        safe_counts = np.maximum(reduced_counts, 1)
        counts[:, :bin_count] = reduced_counts
        drift[:, :bin_count] = np.where(
            usable,
            sum_rate / safe_counts,
            np.nan,
        )
        mean_lag[:, :bin_count] = np.where(
            usable,
            sum_lag / safe_counts,
            np.nan,
        )
    prefix = "dynamics_neighbor_relative_area_change"
    tensors[f"{prefix}_bin_center"] = tensors[
        "dynamics_neighbor_relative_area_bin_center"
    ].copy()
    tensors[f"{prefix}_drift"] = drift
    tensors[f"{prefix}_count"] = counts
    tensors[f"{prefix}_mean_physical_lag"] = mean_lag


def add_neighbor_relative_area_change_fit(
    tensors: dict[str, np.ndarray],
) -> None:
    """Fit ``G_deltaA = Lambda * deltaA`` through the origin for each lag."""
    prefix = "dynamics_neighbor_relative_area_change"
    centers = tensors[f"{prefix}_bin_center"]
    drift = tensors[f"{prefix}_drift"]
    counts = tensors[f"{prefix}_count"]
    mean_lags = tensors[f"{prefix}_mean_physical_lag"]
    lag_count = drift.shape[0]
    valid = np.zeros(lag_count, dtype=bool)
    slope = np.full(lag_count, np.nan)
    rmse = np.full(lag_count, np.nan)
    physical_lag = np.full(lag_count, np.nan)
    for lag_index in range(lag_count):
        usable = (
            np.isfinite(centers)
            & np.isfinite(drift[lag_index])
            & np.isfinite(mean_lags[lag_index])
            & (counts[lag_index] > 0)
        )
        if np.count_nonzero(usable) < 2:
            continue
        values = centers[usable]
        observed = drift[lag_index, usable]
        weights = counts[lag_index, usable].astype(float)
        denominator = float(np.sum(weights * values**2))
        if denominator <= 0.0:
            continue
        fitted_slope = float(np.sum(weights * values * observed) / denominator)
        predicted = fitted_slope * values
        valid[lag_index] = True
        slope[lag_index] = fitted_slope
        rmse[lag_index] = _weighted_rmse(observed, predicted, weights)
        physical_lag[lag_index] = float(
            np.average(mean_lags[lag_index, usable], weights=weights)
        )
    tensors["neighbor_relative_area_change_fit_valid"] = valid
    tensors["neighbor_relative_area_change_fit_lambda"] = slope
    tensors["neighbor_relative_area_change_fit_rmse"] = rmse
    tensors["neighbor_relative_area_change_fit_physical_lag"] = physical_lag



