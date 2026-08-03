from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from .characterization import N_MODES


MAX_FRAME_LAG = 2
TARGET_BINS = 20
MIN_BIN_SAMPLES = 50
PREFERRED_BIN_SAMPLES = 100
ProgressCallback = Callable[[str], None]


def area_diffusion_model(
    area: np.ndarray | float,
    d0: float,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    return np.asarray(d0 + gamma * np.asarray(area) ** alpha)


def area_drift_constant_model(
    area: np.ndarray | float,
    nu: float,
) -> np.ndarray:
    return np.broadcast_to(-nu, np.asarray(area).shape)


def area_drift_cubic_model(
    area: np.ndarray | float,
    c1: float,
    c3: float,
    area_star: float,
) -> np.ndarray:
    centered = np.asarray(area) - area_star
    return np.asarray(-c1 * centered - c3 * centered**3)


@dataclass(frozen=True)
class DynamicProperty:
    slug: str
    label: str
    tensor: str
    mode: int | None = None
    conditioning_tensor: str | None = None


DYNAMIC_PROPERTIES = (
    DynamicProperty(
        "axial_position",
        r"$X_{\rm wrapped}$",
        "unwrapped_axial_position",
        conditioning_tensor="wrapped_axial_position",
    ),
    DynamicProperty("area", r"$A$", "area"),
    DynamicProperty("mean_width", r"$\bar w$", "mean_width"),
    DynamicProperty("width_variance", r"$\sigma_w^2$", "width_variance"),
    DynamicProperty(
        "centerline_variance", r"$\sigma_h^2$", "centerline_variance"
    ),
    DynamicProperty("interface_length", r"$P$", "interface_length"),
    *tuple(
        DynamicProperty(
            f"centerline_mode_{mode}",
            rf"$H_{{{mode}}}$",
            "centerline_mode_amplitude",
            mode - 1,
        )
        for mode in range(1, N_MODES + 1)
    ),
    *tuple(
        DynamicProperty(
            f"width_mode_{mode}",
            rf"$|\widetilde w_{{{mode}}}|$",
            "width_mode_amplitude",
            mode - 1,
        )
        for mode in range(1, N_MODES + 1)
    ),
)


def _property_values(
    tensors: dict[str, np.ndarray], prop: DynamicProperty
) -> np.ndarray:
    values = tensors[prop.tensor]
    return values if prop.mode is None else values[..., prop.mode]


def _conditioning_values(
    tensors: dict[str, np.ndarray], prop: DynamicProperty
) -> np.ndarray:
    if prop.conditioning_tensor is None:
        return _property_values(tensors, prop)
    return tensors[prop.conditioning_tensor]


def _quantile_edges(
    values: np.ndarray,
    target_bins: int,
    preferred_bin_samples: int,
) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.empty(0, dtype=float)
    bin_count = min(target_bins, max(1, finite.size // preferred_bin_samples))
    edges = np.unique(
        np.asarray(
            jax.device_get(
                jnp.quantile(
                    jnp.asarray(finite),
                    jnp.linspace(0.0, 1.0, bin_count + 1),
                )
            )
        )
    )
    if edges.size == 1:
        padding = max(1.0, abs(float(edges[0]))) * 1.0e-12
        edges = np.asarray([edges[0] - padding, edges[0] + padding])
    return edges


@partial(jax.jit, static_argnames=("lag_count", "bin_count"))
def _conditional_reductions(
    values: jax.Array,
    conditioning: jax.Array,
    stable: jax.Array,
    segments: jax.Array,
    times: jax.Array,
    edges: jax.Array,
    *,
    lag_count: int,
    bin_count: int,
) -> tuple[jax.Array, ...]:
    """Reduce every lag and bin without host synchronization."""
    n_time, n_tracks = values.shape
    starts = jnp.arange(n_time)
    lags = jnp.arange(1, lag_count + 1)[:, jnp.newaxis]
    ends = starts[jnp.newaxis, :] + lags
    valid_time = ends < n_time
    ends = jnp.minimum(ends, n_time - 1)

    initial_values = jnp.broadcast_to(
        values[jnp.newaxis, :, :], (lag_count, n_time, n_tracks)
    )
    final_values = values[ends]
    initial_conditioning = jnp.broadcast_to(
        conditioning[jnp.newaxis, :, :], (lag_count, n_time, n_tracks)
    )
    final_stable = stable[ends]
    final_segments = segments[ends]
    delta_tau = times[ends] - times[jnp.newaxis, :]
    pair_lags = delta_tau[..., jnp.newaxis]
    valid = (
        valid_time[..., jnp.newaxis]
        & stable[jnp.newaxis, :, :]
        & final_stable
        & (segments[jnp.newaxis, :, :] == final_segments)
        & jnp.isfinite(initial_values)
        & jnp.isfinite(final_values)
        & jnp.isfinite(initial_conditioning)
        & (pair_lags > 0.0)
    )

    bins = jnp.searchsorted(edges, initial_conditioning, side="right") - 1
    bins = jnp.clip(bins, 0, bin_count - 1)
    lag_offsets = (
        jnp.arange(lag_count)[:, jnp.newaxis, jnp.newaxis] * bin_count
    )
    group_count = lag_count * bin_count
    groups = jnp.where(valid, lag_offsets + bins, group_count).ravel()
    increments = final_values - initial_values
    safe_lags = jnp.where(valid, pair_lags, 1.0)

    def reduce(weights: jax.Array) -> jax.Array:
        return jnp.bincount(
            groups,
            weights=jnp.where(valid, weights, 0.0).ravel(),
            length=group_count + 1,
        )[:-1].reshape(lag_count, bin_count)

    counts = jnp.bincount(groups, length=group_count + 1)[:-1].reshape(
        lag_count, bin_count
    )
    sum_rate = reduce(increments / safe_lags)
    sum_square_over_lag = reduce(increments**2 / safe_lags)
    sum_increment = reduce(increments)
    sum_lag = reduce(jnp.broadcast_to(pair_lags, valid.shape))
    return counts, sum_rate, sum_square_over_lag, sum_increment, sum_lag


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


def add_area_cv_squared_drift(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    target_bins: int = TARGET_BINS,
    minimum_samples: int = MIN_BIN_SAMPLES,
    preferred_bin_samples: int = PREFERRED_BIN_SAMPLES,
) -> None:
    """Estimate F_C(C) on fixed-band-count, event-free intervals."""
    if (
        max_frame_lag < 1
        or target_bins < 1
        or minimum_samples < 1
        or preferred_bin_samples < minimum_samples
    ):
        raise ValueError("CV-area dynamics controls must be positive")
    values = tensors["instantaneous_area_cv_squared"]
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
    prefix = "dynamics_area_cv_squared_fixed_n"
    tensors[f"{prefix}_bin_edges"] = stored_edges
    tensors[f"{prefix}_bin_center"] = centers
    tensors[f"{prefix}_drift"] = drift
    tensors[f"{prefix}_count"] = counts
    tensors[f"{prefix}_mean_physical_lag"] = mean_lag


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
    """Compute spatial increment covariance and conservation ratio at lag one."""
    area = tensors["area"]
    valid = tensors["valid"]
    segments = tensors["segment_id"]
    positions = tensors["wrapped_axial_position"]
    event_count = tensors["frame_event_count"]
    grouped: dict[int, list[np.ndarray]] = {}
    for time_index in range(max(0, area.shape[0] - 1)):
        if event_count[time_index + 1] != 0:
            continue
        selected = np.flatnonzero(valid[time_index])
        if selected.size < 2 or np.count_nonzero(valid[time_index + 1]) != selected.size:
            continue
        if not np.all(valid[time_index + 1, selected]):
            continue
        if not np.array_equal(
            segments[time_index, selected],
            segments[time_index + 1, selected],
        ):
            continue
        ordered = selected[np.argsort(positions[time_index, selected])]
        increments = area[time_index + 1, ordered] - area[time_index, ordered]
        if np.all(np.isfinite(increments)):
            grouped.setdefault(int(selected.size), []).append(increments)

    usable_groups = {
        band_count: np.stack(samples)
        for band_count, samples in grouped.items()
        if len(samples) >= 2
    }
    maximum_band_count = max(usable_groups, default=0)
    covariance_sum = np.zeros(maximum_band_count)
    covariance_count = np.zeros(maximum_band_count, dtype=np.int64)
    band_counts = np.asarray(sorted(usable_groups), dtype=np.int64)
    r_cons_by_count = np.full(len(band_counts), np.nan)
    sample_count = np.zeros(len(band_counts), dtype=np.int64)
    total_numerator = 0.0
    total_denominator = 0.0
    for group_index, band_count in enumerate(band_counts):
        samples = jnp.asarray(usable_groups[int(band_count)])
        centered = samples - jnp.mean(samples)
        covariances = jnp.stack(
            [
                jnp.mean(centered * jnp.roll(centered, -separation, axis=1))
                for separation in range(int(band_count))
            ]
        )
        total_increment = jnp.sum(samples, axis=1)
        numerator = jnp.var(total_increment)
        denominator = jnp.sum(jnp.var(samples, axis=0))
        covariance_values, numerator_value, denominator_value = jax.device_get(
            (covariances, numerator, denominator)
        )
        pair_count = int(samples.shape[0] * samples.shape[1])
        covariance_sum[: int(band_count)] += (
            np.asarray(covariance_values) * pair_count
        )
        covariance_count[: int(band_count)] += pair_count
        sample_count[group_index] = int(samples.shape[0])
        if float(denominator_value) > 0.0:
            r_cons_by_count[group_index] = float(
                numerator_value / denominator_value
            )
            total_numerator += float(samples.shape[0] * numerator_value)
            total_denominator += float(samples.shape[0] * denominator_value)

    covariance = np.where(
        covariance_count > 0,
        covariance_sum / np.maximum(covariance_count, 1),
        np.nan,
    )
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


@jax.jit
def _batched_position_msd(
    values: jax.Array,
    stable: jax.Array,
    segments: jax.Array,
    times: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_time, n_tracks = values.shape
    lag_count = n_time - 1
    starts = jnp.arange(n_time)
    lags = jnp.arange(1, n_time)[:, jnp.newaxis]
    ends = starts[jnp.newaxis, :] + lags
    valid_time = ends < n_time
    ends = jnp.minimum(ends, n_time - 1)
    final_values = values[ends]
    delta_tau = times[ends] - times[jnp.newaxis, :]
    valid = (
        valid_time[..., jnp.newaxis]
        & stable[jnp.newaxis, :, :]
        & stable[ends]
        & (segments[jnp.newaxis, :, :] == segments[ends])
        & (delta_tau[..., jnp.newaxis] > 0.0)
    )
    displacement = final_values - values[jnp.newaxis, :, :]
    counts = jnp.sum(valid, axis=(1, 2))
    safe_counts = jnp.maximum(counts, 1)
    msd = jnp.sum(jnp.where(valid, displacement**2, 0.0), axis=(1, 2)) / safe_counts
    physical_lag = (
        jnp.sum(
            jnp.where(valid, delta_tau[..., jnp.newaxis], 0.0),
            axis=(1, 2),
        )
        / safe_counts
    )
    usable = counts > 0
    return (
        jnp.where(usable, physical_lag, jnp.nan),
        jnp.where(usable, msd, jnp.nan),
        counts,
    )


def add_position_msd(
    tensors: dict[str, np.ndarray],
    *,
    progress: ProgressCallback | None = None,
) -> None:
    values = tensors["unwrapped_axial_position"]
    stable = tensors["stable"]
    segments = tensors["segment_id"]
    times = tensors["physical_time"]
    lag_count = max(0, len(times) - 1)
    frame_lags = np.arange(1, lag_count + 1, dtype=np.int64)
    if lag_count:
        physical_lags, msd, counts = jax.device_get(
            _batched_position_msd(
                jnp.asarray(values),
                jnp.asarray(stable),
                jnp.asarray(segments),
                jnp.asarray(times),
            )
        )
        counts = np.asarray(counts, dtype=np.int64)
    else:
        physical_lags = np.empty(0)
        msd = np.empty(0)
        counts = np.empty(0, dtype=np.int64)
    if progress is not None:
        progress(f"stage=stochastic msd_lags={lag_count}/{lag_count}")
    tensors["position_msd_frame_lag"] = frame_lags
    tensors["position_msd_physical_lag"] = physical_lags
    tensors["position_msd"] = msd
    tensors["position_msd_count"] = counts


def _neighbor_pair_slots(
    positions: np.ndarray, stable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Store unique periodic neighbor pairs in compact per-frame slots."""
    n_time = stable.shape[0]
    frame_pairs: list[list[tuple[int, int]]] = []
    for time_index in range(n_time):
        selected = np.flatnonzero(stable[time_index])
        if len(selected) < 2:
            frame_pairs.append([])
            continue
        ordered = selected[np.argsort(positions[time_index, selected])]
        if len(ordered) == 2:
            first, second = int(ordered[0]), int(ordered[1])
            frame_pairs.append([(min(first, second), max(first, second))])
            continue
        following = np.roll(ordered, -1)
        frame_pairs.append(
            sorted(
                {
                    (min(int(first), int(second)), max(int(first), int(second)))
                    for first, second in zip(ordered, following, strict=True)
                }
            )
        )
    max_pairs = max((len(pairs) for pairs in frame_pairs), default=0)
    first_slots = np.full((n_time, max_pairs), -1, dtype=np.int64)
    second_slots = np.full((n_time, max_pairs), -1, dtype=np.int64)
    for time_index, pairs in enumerate(frame_pairs):
        for pair_index, (first, second) in enumerate(pairs):
            first_slots[time_index, pair_index] = first
            second_slots[time_index, pair_index] = second
    return first_slots, second_slots


@jax.jit
def _batched_neighbor_covariance(
    centered: jax.Array,
    first_slots: jax.Array,
    second_slots: jax.Array,
    segments: jax.Array,
    times: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n_time = centered.shape[0]
    starts = jnp.arange(n_time)
    lags = jnp.arange(1, n_time)[:, jnp.newaxis]
    ends = starts[jnp.newaxis, :] + lags
    valid_time = ends < n_time
    ends = jnp.minimum(ends, n_time - 1)
    safe_first = jnp.maximum(first_slots, 0)
    safe_second = jnp.maximum(second_slots, 0)
    end_first = first_slots[ends]
    end_second = second_slots[ends]
    safe_end_first = jnp.maximum(end_first, 0)
    safe_end_second = jnp.maximum(end_second, 0)

    start_first_values = jnp.take_along_axis(centered, safe_first, axis=1)
    start_second_values = jnp.take_along_axis(centered, safe_second, axis=1)
    end_centered = centered[ends]
    end_first_values = jnp.take_along_axis(
        end_centered, safe_end_first, axis=2
    )
    end_second_values = jnp.take_along_axis(
        end_centered, safe_end_second, axis=2
    )
    start_first_segments = jnp.take_along_axis(segments, safe_first, axis=1)
    start_second_segments = jnp.take_along_axis(segments, safe_second, axis=1)
    end_segments = segments[ends]
    end_first_segments = jnp.take_along_axis(
        end_segments, safe_end_first, axis=2
    )
    end_second_segments = jnp.take_along_axis(
        end_segments, safe_end_second, axis=2
    )

    pair_valid = (
        valid_time[..., jnp.newaxis, jnp.newaxis]
        & (first_slots[jnp.newaxis, :, :, jnp.newaxis] >= 0)
        & (end_first[..., jnp.newaxis, :] >= 0)
        & (
            first_slots[jnp.newaxis, :, :, jnp.newaxis]
            == end_first[..., jnp.newaxis, :]
        )
        & (
            second_slots[jnp.newaxis, :, :, jnp.newaxis]
            == end_second[..., jnp.newaxis, :]
        )
        & (
            start_first_segments[jnp.newaxis, :, :, jnp.newaxis]
            == end_first_segments[..., jnp.newaxis, :]
        )
        & (
            start_second_segments[jnp.newaxis, :, :, jnp.newaxis]
            == end_second_segments[..., jnp.newaxis, :]
        )
    )
    products = (
        start_first_values[jnp.newaxis, :, :, jnp.newaxis]
        * end_second_values[..., jnp.newaxis, :]
        + start_second_values[jnp.newaxis, :, :, jnp.newaxis]
        * end_first_values[..., jnp.newaxis, :]
    )
    delta_tau = times[ends] - times[jnp.newaxis, :]
    pair_valid &= delta_tau[..., jnp.newaxis, jnp.newaxis] > 0.0
    matched_pairs = jnp.sum(pair_valid, axis=(1, 2, 3))
    counts = 2 * matched_pairs
    safe_counts = jnp.maximum(counts, 1)
    covariance = (
        jnp.sum(jnp.where(pair_valid, products, 0.0), axis=(1, 2, 3))
        / safe_counts
    )
    physical_lag = (
        jnp.sum(
            jnp.where(
                pair_valid,
                2.0 * delta_tau[..., jnp.newaxis, jnp.newaxis],
                0.0,
            ),
            axis=(1, 2, 3),
        )
        / safe_counts
    )
    usable = counts > 0
    return (
        jnp.where(usable, physical_lag, jnp.nan),
        jnp.where(usable, covariance, jnp.nan),
        counts,
    )


@jax.jit
def _area_time_series(
    area: jax.Array,
    stable: jax.Array,
    segments: jax.Array,
    times: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    n_time = area.shape[0]
    stable_total = jnp.where(
        jnp.any(stable, axis=1),
        jnp.sum(jnp.where(stable, area, 0.0), axis=1),
        jnp.nan,
    )
    mean_absolute_rate = jnp.full(n_time, jnp.nan)
    if n_time > 1:
        adjacent_dt = times[1:] - times[:-1]
        adjacent_valid = (
            stable[:-1]
            & stable[1:]
            & (segments[:-1] == segments[1:])
            & (adjacent_dt[:, jnp.newaxis] > 0.0)
        )
        adjacent_count = jnp.sum(adjacent_valid, axis=1)
        adjacent_sum = jnp.sum(
            jnp.where(adjacent_valid, jnp.abs(area[1:] - area[:-1]), 0.0),
            axis=1,
        )
        adjacent_rate = jnp.where(
            adjacent_count > 0,
            adjacent_sum / adjacent_count / adjacent_dt,
            jnp.nan,
        )
        mean_absolute_rate = mean_absolute_rate.at[1:].set(adjacent_rate)
    return stable_total, mean_absolute_rate


def add_area_coupling(
    tensors: dict[str, np.ndarray],
    *,
    progress: ProgressCallback | None = None,
) -> None:
    area = tensors["area"]
    positions = tensors["wrapped_axial_position"]
    stable = tensors["stable"]
    segments = tensors["segment_id"]
    times = tensors["physical_time"]
    n_time = len(times)

    centered = np.full_like(area, np.nan)
    for segment_id in np.unique(segments[stable]):
        selected = stable & (segments == segment_id)
        centered[selected] = area[selected] - float(np.mean(area[selected]))

    first_slots, second_slots = _neighbor_pair_slots(positions, stable)
    lag_count = max(0, n_time - 1)
    if lag_count:
        physical_lag, covariance, counts = jax.device_get(
            _batched_neighbor_covariance(
                jnp.asarray(centered),
                jnp.asarray(first_slots),
                jnp.asarray(second_slots),
                jnp.asarray(segments),
                jnp.asarray(times),
            )
        )
        counts = np.asarray(counts, dtype=np.int64)
    else:
        physical_lag = np.empty(0)
        covariance = np.empty(0)
        counts = np.empty(0, dtype=np.int64)
    if progress is not None:
        progress(
            f"stage=stochastic area_coupling_lags={lag_count}/{lag_count}"
        )

    stable_total, mean_absolute_rate = jax.device_get(
        _area_time_series(
            jnp.asarray(area),
            jnp.asarray(stable),
            jnp.asarray(segments),
            jnp.asarray(times),
        )
    )

    tensors["neighbor_area_covariance_frame_lag"] = np.arange(
        1, lag_count + 1, dtype=np.int64
    )
    tensors["neighbor_area_covariance_physical_lag"] = physical_lag
    tensors["neighbor_area_covariance"] = covariance
    tensors["neighbor_area_covariance_count"] = counts
    tensors["stable_total_area"] = stable_total
    tensors["stable_mean_absolute_area_rate"] = mean_absolute_rate


def add_stochastic_statistics(
    tensors: dict[str, np.ndarray],
    *,
    max_frame_lag: int = MAX_FRAME_LAG,
    progress: ProgressCallback | None = None,
) -> None:
    if progress is not None:
        progress("stage=stochastic conditional_start")
    add_conditional_dynamics(
        tensors, max_frame_lag=max_frame_lag, progress=progress
    )
    if progress is not None:
        progress("stage=stochastic area_cv_squared_fixed_n start")
    add_area_cv_squared_drift(
        tensors,
        max_frame_lag=max_frame_lag,
    )
    if progress is not None:
        progress("stage=stochastic area_cv_squared_fixed_n complete")
        progress("stage=stochastic neighbor_relative_area start")
    add_neighbor_relative_area_drifts(
        tensors,
        max_frame_lag=max_frame_lag,
    )
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area change_start")
    add_neighbor_relative_area_change_drift(
        tensors,
        max_frame_lag=max_frame_lag,
    )
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area change_fit")
    add_neighbor_relative_area_change_fit(tensors)
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area fits_start")
    add_neighbor_relative_area_fit(tensors)
    if progress is not None:
        progress("stage=stochastic neighbor_relative_area complete")
        progress("stage=stochastic area_drift_fits start")
    add_area_drift_fits(tensors)
    if progress is not None:
        progress("stage=stochastic area_drift_fits complete")
        progress("stage=stochastic area_diffusion_fit start")
    add_area_diffusion_fit(tensors)
    if progress is not None:
        progress("stage=stochastic area_diffusion_fit complete")
        progress("stage=stochastic area_geometry_fits start")
    add_area_geometry_fits(tensors)
    if progress is not None:
        progress("stage=stochastic area_geometry_fits complete")
        progress("stage=stochastic chapman_kolmogorov start")
    add_area_chapman_kolmogorov(tensors, max_frame_lag=max_frame_lag)
    if progress is not None:
        progress("stage=stochastic chapman_kolmogorov complete")
        progress("stage=stochastic msd_start")
    add_position_msd(tensors, progress=progress)
    if progress is not None:
        progress("stage=stochastic area_coupling_start")
    add_area_coupling(tensors, progress=progress)
    if progress is not None:
        progress("stage=stochastic area_increment_conservation start")
    add_area_increment_conservation(tensors)
    if progress is not None:
        progress("stage=stochastic area_increment_conservation complete")
        progress("stage=stochastic complete")
