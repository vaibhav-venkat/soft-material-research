from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .common import (
    DYNAMIC_PROPERTIES,
    MAX_FRAME_LAG,
    MIN_BIN_SAMPLES,
    PREFERRED_BIN_SAMPLES,
    TARGET_BINS,
    ProgressCallback,
    _conditional_reductions,
    _conditioning_values,
    _property_values,
    _quantile_edges,
)

def add_conditional_dynamics(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    target_bins: int = TARGET_BINS,
    minimum_samples: int = MIN_BIN_SAMPLES,
    preferred_bin_samples: int = PREFERRED_BIN_SAMPLES,
    progress: ProgressCallback | None = None,
) -> None:
    if (
        max_frame_lag < 1
        or target_bins < 1
        or minimum_samples < 1
        or preferred_bin_samples < minimum_samples
    ):
        raise ValueError("dynamics controls must be positive")
    stable = tensors["stable"]
    segments = tensors["segment_id"]
    times = tensors["physical_time"]
    n_time = len(times)
    lag_count = min(max_frame_lag, max(0, n_time - 1))
    tensors["dynamics_frame_lag"] = np.arange(1, lag_count + 1, dtype=np.int64)

    for property_index, prop in enumerate(DYNAMIC_PROPERTIES, start=1):
        values = _property_values(tensors, prop)
        conditioning = _conditioning_values(tensors, prop)
        edges = _quantile_edges(
            conditioning[stable], target_bins, preferred_bin_samples
        )
        stored_edges = np.full(target_bins + 1, np.nan)
        centers = np.full(target_bins, np.nan)
        drift = np.full((lag_count, target_bins), np.nan)
        diffusion = np.full((lag_count, target_bins), np.nan)
        counts = np.zeros((lag_count, target_bins), dtype=np.int64)
        mean_lag = np.full((lag_count, target_bins), np.nan)
        if edges.size:
            bin_count = len(edges) - 1
            stored_edges[: len(edges)] = edges
            centers[:bin_count] = 0.5 * (edges[:-1] + edges[1:])
            if lag_count:
                reduced = jax.device_get(
                    _conditional_reductions(
                        jnp.asarray(values),
                        jnp.asarray(conditioning),
                        jnp.asarray(stable),
                        jnp.asarray(segments),
                        jnp.asarray(times),
                        jnp.asarray(edges),
                        lag_count=lag_count,
                        bin_count=bin_count,
                    )
                )
                reduced_counts = np.asarray(reduced[0], dtype=np.int64)
                sum_rate = np.asarray(reduced[1])
                sum_square_over_lag = np.asarray(reduced[2])
                sum_increment = np.asarray(reduced[3])
                sum_lag = np.asarray(reduced[4])
                counts[:, :bin_count] = reduced_counts
                usable = reduced_counts >= minimum_samples
                safe_counts = np.maximum(reduced_counts, 1)
                estimates = sum_rate / safe_counts
                diffusion_values = (
                    sum_square_over_lag
                    - 2.0 * estimates * sum_increment
                    + estimates**2 * sum_lag
                ) / (2.0 * safe_counts)
                drift[:, :bin_count] = np.where(usable, estimates, np.nan)
                diffusion[:, :bin_count] = np.where(
                    usable, np.maximum(diffusion_values, 0.0), np.nan
                )
                mean_lag[:, :bin_count] = np.where(
                    usable, sum_lag / safe_counts, np.nan
                )

        prefix = f"dynamics_{prop.slug}"
        tensors[f"{prefix}_bin_edges"] = stored_edges
        tensors[f"{prefix}_bin_center"] = centers
        tensors[f"{prefix}_drift"] = drift
        tensors[f"{prefix}_diffusion"] = diffusion
        tensors[f"{prefix}_count"] = counts
        tensors[f"{prefix}_mean_physical_lag"] = mean_lag
        if progress is not None:
            progress(
                "stage=stochastic conditional="
                f"{property_index}/{len(DYNAMIC_PROPERTIES)} property={prop.slug}"
            )


@partial(jax.jit, static_argnames=("lag_count", "bin_count"))
def _fixed_n_event_free_scalar_reductions(
    values: jax.Array,
    band_count: jax.Array,
    frame_event_count: jax.Array,
    times: jax.Array,
    edges: jax.Array,
    *,
    lag_count: int,
    bin_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reduce scalar increments over fixed-count, event-free intervals."""
    n_time = values.shape[0]
    starts = jnp.arange(n_time)
    lags = jnp.arange(1, lag_count + 1)[:, jnp.newaxis]
    ends = starts[jnp.newaxis, :] + lags
    valid_time = ends < n_time
    ends = jnp.minimum(ends, n_time - 1)
    initial = values[jnp.newaxis, :]
    final = values[ends]
    initial_count = band_count[jnp.newaxis, :]
    valid = (
        valid_time
        & (initial_count > 0)
        & jnp.isfinite(initial)
        & jnp.isfinite(final)
    )
    for offset in range(1, lag_count + 1):
        sampled = jnp.minimum(starts + offset, n_time - 1)
        included = offset <= lags
        valid &= (~included) | (
            (band_count[sampled][jnp.newaxis, :] == initial_count)
            & (frame_event_count[sampled][jnp.newaxis, :] == 0)
        )
    delta_tau = times[ends] - times[jnp.newaxis, :]
    valid &= delta_tau > 0.0
    bins = jnp.searchsorted(edges, values, side="right") - 1
    bins = jnp.clip(bins, 0, bin_count - 1)
    lag_offsets = jnp.arange(lag_count)[:, jnp.newaxis] * bin_count
    group_count = lag_count * bin_count
    groups = jnp.where(
        valid,
        lag_offsets + bins[jnp.newaxis, :],
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

    safe_lags = jnp.where(valid, delta_tau, 1.0)
    sum_rate = reduce((final - initial) / safe_lags)
    sum_lag = reduce(delta_tau)
    return counts, sum_rate, sum_lag


def _add_fixed_n_event_free_drift(
    tensors: dict[str, np.ndarray],
    values_key: str,
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
        raise ValueError("scalar dynamics controls must be positive")
    values = tensors[values_key]
    finite = values[np.isfinite(values)]
    edges = _quantile_edges(finite, target_bins, preferred_bin_samples)
    lag_count = min(max_frame_lag, max(0, len(values) - 1))
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
                _fixed_n_event_free_scalar_reductions(
                    jnp.asarray(values),
                    jnp.asarray(tensors["instantaneous_band_count"]),
                    jnp.asarray(tensors["frame_event_count"]),
                    jnp.asarray(tensors["physical_time"]),
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
    tensors[f"{prefix}_bin_edges"] = stored_edges
    tensors[f"{prefix}_bin_center"] = centers
    tensors[f"{prefix}_drift"] = drift
    tensors[f"{prefix}_count"] = counts
    tensors[f"{prefix}_mean_physical_lag"] = mean_lag


def add_area_cv_squared_drift(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    target_bins: int = TARGET_BINS,
    minimum_samples: int = MIN_BIN_SAMPLES,
    preferred_bin_samples: int = PREFERRED_BIN_SAMPLES,
) -> None:
    """Estimate F_C(C) on fixed-band-count, event-free intervals."""
    _add_fixed_n_event_free_drift(
        tensors,
        "instantaneous_area_cv_squared",
        "dynamics_area_cv_squared_fixed_n",
        max_frame_lag=max_frame_lag,
        target_bins=target_bins,
        minimum_samples=minimum_samples,
        preferred_bin_samples=preferred_bin_samples,
    )


def add_total_area_drift(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    target_bins: int = TARGET_BINS,
    minimum_samples: int = MIN_BIN_SAMPLES,
    preferred_bin_samples: int = PREFERRED_BIN_SAMPLES,
) -> None:
    """Estimate F_T(A_T) on fixed-band-count, event-free intervals."""
    _add_fixed_n_event_free_drift(
        tensors,
        "instantaneous_total_area",
        "dynamics_total_area_fixed_n",
        max_frame_lag=max_frame_lag,
        target_bins=target_bins,
        minimum_samples=minimum_samples,
        preferred_bin_samples=preferred_bin_samples,
    )

