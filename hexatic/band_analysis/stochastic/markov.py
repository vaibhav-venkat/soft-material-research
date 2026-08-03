from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .common import MAX_FRAME_LAG, MIN_BIN_SAMPLES, TARGET_BINS, _quantile_edges

@partial(
    jax.jit,
    static_argnames=("lag_count", "bin_count", "minimum_samples"),
)
def _chapman_kolmogorov_reductions(
    area: jax.Array,
    stable: jax.Array,
    segments: jax.Array,
    times: jax.Array,
    edges: jax.Array,
    *,
    lag_count: int,
    bin_count: int,
    minimum_samples: int,
) -> tuple[jax.Array, ...]:
    """Count one- and two-lag area transitions for every tested lag."""
    n_time, n_tracks = area.shape
    starts = jnp.arange(n_time)
    lags = jnp.arange(1, lag_count + 1)[:, jnp.newaxis]
    middle = jnp.minimum(starts[jnp.newaxis, :] + lags, n_time - 1)
    ends = jnp.minimum(starts[jnp.newaxis, :] + 2 * lags, n_time - 1)
    valid_middle_time = starts[jnp.newaxis, :] + lags < n_time
    valid_end_time = starts[jnp.newaxis, :] + 2 * lags < n_time

    def bins(values: jax.Array) -> jax.Array:
        result = jnp.searchsorted(edges, values, side="right") - 1
        return jnp.clip(result, 0, bin_count - 1)

    start_bins = bins(area)[jnp.newaxis, :, :]
    middle_bins = bins(area[middle])
    end_bins = bins(area[ends])
    start_stable = stable[jnp.newaxis, :, :]
    start_segments = segments[jnp.newaxis, :, :]
    finite_start = jnp.isfinite(area)[jnp.newaxis, :, :]
    valid_one = (
        valid_middle_time[..., jnp.newaxis]
        & start_stable
        & stable[middle]
        & (start_segments == segments[middle])
        & finite_start
        & jnp.isfinite(area[middle])
    )
    valid_two = (
        valid_end_time[..., jnp.newaxis]
        & start_stable
        & stable[ends]
        & (start_segments == segments[ends])
        & finite_start
        & jnp.isfinite(area[ends])
    )
    matrix_size = bin_count * bin_count
    lag_offsets = (
        jnp.arange(lag_count)[:, jnp.newaxis, jnp.newaxis] * matrix_size
    )

    def transition_counts(
        initial: jax.Array, final: jax.Array, valid: jax.Array
    ) -> jax.Array:
        groups = jnp.where(
            valid,
            lag_offsets + initial * bin_count + final,
            lag_count * matrix_size,
        )
        return jnp.bincount(
            groups.ravel(), length=lag_count * matrix_size + 1
        )[:-1].reshape(lag_count, bin_count, bin_count)

    one_counts = transition_counts(start_bins, middle_bins, valid_one)
    two_counts = transition_counts(start_bins, end_bins, valid_two)
    one_row_counts = jnp.sum(one_counts, axis=2)
    two_row_counts = jnp.sum(two_counts, axis=2)
    one_valid = one_row_counts >= minimum_samples
    two_valid = two_row_counts >= minimum_samples
    one = one_counts / jnp.maximum(one_row_counts[..., jnp.newaxis], 1)
    direct = two_counts / jnp.maximum(two_row_counts[..., jnp.newaxis], 1)
    composed = one @ one
    retained_mass = jnp.sum(one * one_valid[:, jnp.newaxis, :], axis=2)
    row_valid = one_valid & two_valid & (retained_mass >= 1.0 - 1.0e-6)
    row_error = 0.5 * jnp.sum(jnp.abs(direct - composed), axis=2)
    row_error = jnp.where(row_valid, row_error, jnp.nan)
    weights = jnp.where(row_valid, two_row_counts, 0)
    weight_sum = jnp.sum(weights, axis=1)
    discrepancy = jnp.where(
        weight_sum > 0,
        jnp.nansum(row_error * weights, axis=1) / jnp.maximum(weight_sum, 1),
        jnp.nan,
    )
    base_lag = times[middle] - times[jnp.newaxis, :]
    direct_lag = times[ends] - times[jnp.newaxis, :]
    mean_base_lag = jnp.sum(
        jnp.where(valid_one, base_lag[..., jnp.newaxis], 0.0), axis=(1, 2)
    ) / jnp.maximum(jnp.sum(valid_one, axis=(1, 2)), 1)
    mean_direct_lag = jnp.sum(
        jnp.where(valid_two, direct_lag[..., jnp.newaxis], 0.0), axis=(1, 2)
    ) / jnp.maximum(jnp.sum(valid_two, axis=(1, 2)), 1)
    return (
        one_counts,
        two_counts,
        jnp.where(one_valid[..., jnp.newaxis], one, jnp.nan),
        jnp.where(row_valid[..., jnp.newaxis], direct, jnp.nan),
        jnp.where(row_valid[..., jnp.newaxis], composed, jnp.nan),
        row_valid,
        row_error,
        discrepancy,
        mean_base_lag,
        mean_direct_lag,
    )


def add_area_chapman_kolmogorov(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    minimum_samples: int = MIN_BIN_SAMPLES,
) -> None:
    """Compare direct two-lag area transitions with composed one-lag ones."""
    if max_frame_lag < 2 or minimum_samples < 1:
        raise ValueError("Chapman-Kolmogorov controls are invalid")
    area = tensors["area"]
    stable = tensors["stable"]
    n_time = area.shape[0]
    lag_count = min(max_frame_lag // 2, max(0, (n_time - 1) // 2))
    stored_bin_count = tensors["dynamics_area_bin_center"].size
    finite_edges = tensors["dynamics_area_bin_edges"]
    edges = finite_edges[np.isfinite(finite_edges)]
    bin_count = max(0, edges.size - 1)
    shape = (lag_count, stored_bin_count, stored_bin_count)
    one_probability = np.full(shape, np.nan)
    direct_probability = np.full(shape, np.nan)
    composed_probability = np.full(shape, np.nan)
    row_error = np.full((lag_count, stored_bin_count), np.nan)
    one_counts_stored = np.zeros(shape, dtype=np.int64)
    direct_counts_stored = np.zeros(shape, dtype=np.int64)
    row_valid_stored = np.zeros((lag_count, stored_bin_count), dtype=bool)
    discrepancy = np.full(lag_count, np.nan)
    base_physical_lag = np.full(lag_count, np.nan)
    direct_physical_lag = np.full(lag_count, np.nan)

    if lag_count and bin_count:
        reduced = jax.device_get(
            _chapman_kolmogorov_reductions(
                jnp.asarray(area),
                jnp.asarray(stable),
                jnp.asarray(tensors["segment_id"]),
                jnp.asarray(tensors["physical_time"]),
                jnp.asarray(edges),
                lag_count=lag_count,
                bin_count=bin_count,
                minimum_samples=minimum_samples,
            )
        )
        one_counts = np.asarray(reduced[0], dtype=np.int64)
        direct_counts = np.asarray(reduced[1], dtype=np.int64)
        one_probability[:, :bin_count, :bin_count] = reduced[2]
        direct_probability[:, :bin_count, :bin_count] = reduced[3]
        composed_probability[:, :bin_count, :bin_count] = reduced[4]
        row_valid_stored[:, :bin_count] = reduced[5]
        row_error[:, :bin_count] = reduced[6]
        discrepancy = np.asarray(reduced[7])
        one_counts_stored[:, :bin_count, :bin_count] = one_counts
        direct_counts_stored[:, :bin_count, :bin_count] = direct_counts
        base_physical_lag = np.asarray(reduced[8])
        direct_physical_lag = np.asarray(reduced[9])

    tensors["area_ck_frame_lag"] = np.arange(1, lag_count + 1, dtype=np.int64)
    tensors["area_ck_base_physical_lag"] = base_physical_lag
    tensors["area_ck_direct_physical_lag"] = direct_physical_lag
    tensors["area_ck_one_step_count"] = one_counts_stored
    tensors["area_ck_direct_count"] = direct_counts_stored
    tensors["area_ck_one_step_probability"] = one_probability
    tensors["area_ck_direct_probability"] = direct_probability
    tensors["area_ck_composed_probability"] = composed_probability
    tensors["area_ck_row_valid"] = row_valid_stored
    tensors["area_ck_row_total_variation"] = row_error
    tensors["area_ck_weighted_total_variation"] = discrepancy



