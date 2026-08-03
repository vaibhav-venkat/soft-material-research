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


MAX_FRAME_LAG = 10
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


def add_area_diffusion_fit(tensors: dict[str, np.ndarray]) -> None:
    """Fit ``D_A(A) = D0 + Gamma A**alpha`` independently for each lag."""
    centers = tensors["dynamics_area_bin_center"]
    diffusion = tensors["dynamics_area_diffusion"]
    counts = tensors["dynamics_area_count"]
    mean_lags = tensors["dynamics_area_mean_physical_lag"]
    lag_count = diffusion.shape[0]
    d0_values = np.full(lag_count, np.nan)
    gamma_values = np.full(lag_count, np.nan)
    alpha_values = np.full(lag_count, np.nan)
    r_squared = np.full(lag_count, np.nan)
    physical_lags = np.full(lag_count, np.nan)
    fit_valid = np.zeros(lag_count, dtype=bool)

    for lag_index in range(lag_count):
        usable = (
            np.isfinite(centers)
            & np.isfinite(diffusion[lag_index])
            & np.isfinite(mean_lags[lag_index])
            & (centers > 0.0)
            & (counts[lag_index] > 0)
        )
        if np.count_nonzero(usable) < 4:
            continue
        area = centers[usable]
        observed = diffusion[lag_index, usable]
        weights = counts[lag_index, usable].astype(float)
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
        physical_lags[lag_index] = float(
            np.average(mean_lags[lag_index, usable], weights=weights)
        )
        fit_valid[lag_index] = True

    tensors["area_diffusion_fit_valid"] = fit_valid
    tensors["area_diffusion_fit_d0"] = d0_values
    tensors["area_diffusion_fit_gamma"] = gamma_values
    tensors["area_diffusion_fit_alpha"] = alpha_values
    tensors["area_diffusion_fit_r_squared"] = r_squared
    tensors["area_diffusion_fit_physical_lag"] = physical_lags


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
    progress: ProgressCallback | None = None,
) -> None:
    if progress is not None:
        progress("stage=stochastic conditional_start")
    add_conditional_dynamics(tensors, progress=progress)
    if progress is not None:
        progress("stage=stochastic area_diffusion_fit start")
    add_area_diffusion_fit(tensors)
    if progress is not None:
        progress("stage=stochastic area_diffusion_fit complete")
        progress("stage=stochastic chapman_kolmogorov start")
    add_area_chapman_kolmogorov(tensors)
    if progress is not None:
        progress("stage=stochastic chapman_kolmogorov complete")
        progress("stage=stochastic msd_start")
    add_position_msd(tensors, progress=progress)
    if progress is not None:
        progress("stage=stochastic area_coupling_start")
    add_area_coupling(tensors, progress=progress)
    if progress is not None:
        progress("stage=stochastic complete")
