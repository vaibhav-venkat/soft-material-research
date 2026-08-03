from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .characterization import N_MODES


MAX_FRAME_LAG = 10
TARGET_BINS = 20
MIN_BIN_SAMPLES = 50
PREFERRED_BIN_SAMPLES = 100


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


def add_conditional_dynamics(
    tensors: dict[str, np.ndarray],
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
        raise ValueError("dynamics controls must be positive")
    stable = tensors["stable"]
    segments = tensors["segment_id"]
    times = tensors["physical_time"]
    n_time = len(times)
    lag_count = min(max_frame_lag, max(0, n_time - 1))
    tensors["dynamics_frame_lag"] = np.arange(1, lag_count + 1, dtype=np.int64)

    for prop in DYNAMIC_PROPERTIES:
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
            stored_edges[: len(edges)] = edges
            centers[: len(edges) - 1] = 0.5 * (edges[:-1] + edges[1:])
            values_device = jnp.asarray(values)
            conditioning_device = jnp.asarray(conditioning)
            stable_device = jnp.asarray(stable)
            segments_device = jnp.asarray(segments)
            times_device = jnp.asarray(times)
            for lag_index, lag in enumerate(range(1, lag_count + 1)):
                delta_tau = times_device[lag:] - times_device[:-lag]
                valid = (
                    stable_device[:-lag]
                    & stable_device[lag:]
                    & (segments_device[:-lag] == segments_device[lag:])
                    & jnp.isfinite(values_device[:-lag])
                    & jnp.isfinite(values_device[lag:])
                )
                valid &= delta_tau[:, np.newaxis] > 0.0
                initial = conditioning_device[:-lag]
                increments = values_device[lag:] - values_device[:-lag]
                pair_lags = jnp.broadcast_to(delta_tau[:, np.newaxis], valid.shape)
                for bin_index in range(len(edges) - 1):
                    upper_comparison = (
                        initial <= edges[bin_index + 1]
                        if bin_index == len(edges) - 2
                        else initial < edges[bin_index + 1]
                    )
                    selected = (
                        valid
                        & (initial >= edges[bin_index])
                        & upper_comparison
                    )
                    count = int(jax.device_get(jnp.sum(selected)))
                    counts[lag_index, bin_index] = count
                    if count < minimum_samples:
                        continue
                    dz = jnp.where(selected, increments, 0.0)
                    dt = jnp.where(selected, pair_lags, 1.0)
                    estimate_device = jnp.sum(dz / dt) / count
                    estimate = float(jax.device_get(estimate_device))
                    drift[lag_index, bin_index] = estimate
                    diffusion[lag_index, bin_index] = float(
                        jax.device_get(
                            jnp.sum(
                                jnp.where(
                                    selected,
                                    (increments - estimate_device * pair_lags) ** 2
                                    / (2.0 * pair_lags),
                                    0.0,
                                )
                            )
                            / count
                        )
                    )
                    mean_lag[lag_index, bin_index] = float(
                        jax.device_get(jnp.sum(jnp.where(selected, pair_lags, 0.0)) / count)
                    )

        prefix = f"dynamics_{prop.slug}"
        tensors[f"{prefix}_bin_edges"] = stored_edges
        tensors[f"{prefix}_bin_center"] = centers
        tensors[f"{prefix}_drift"] = drift
        tensors[f"{prefix}_diffusion"] = diffusion
        tensors[f"{prefix}_count"] = counts
        tensors[f"{prefix}_mean_physical_lag"] = mean_lag


def add_position_msd(tensors: dict[str, np.ndarray]) -> None:
    values = jnp.asarray(tensors["unwrapped_axial_position"])
    stable = jnp.asarray(tensors["stable"])
    segments = jnp.asarray(tensors["segment_id"])
    times = jnp.asarray(tensors["physical_time"])
    lag_count = max(0, len(times) - 1)
    frame_lags = np.arange(1, lag_count + 1, dtype=np.int64)
    physical_lags = np.full(lag_count, np.nan)
    msd = np.full(lag_count, np.nan)
    counts = np.zeros(lag_count, dtype=np.int64)
    for lag_index, lag in enumerate(frame_lags):
        delta_tau = times[lag:] - times[:-lag]
        valid = (
            stable[:-lag]
            & stable[lag:]
            & (segments[:-lag] == segments[lag:])
        )
        valid &= delta_tau[:, np.newaxis] > 0.0
        count = int(jax.device_get(jnp.sum(valid)))
        counts[lag_index] = count
        if not count:
            continue
        displacement = values[lag:] - values[:-lag]
        pair_lags = jnp.broadcast_to(delta_tau[:, np.newaxis], valid.shape)
        msd[lag_index] = float(
            jax.device_get(jnp.sum(jnp.where(valid, displacement**2, 0.0)) / count)
        )
        physical_lags[lag_index] = float(
            jax.device_get(jnp.sum(jnp.where(valid, pair_lags, 0.0)) / count)
        )
    tensors["position_msd_frame_lag"] = frame_lags
    tensors["position_msd_physical_lag"] = physical_lags
    tensors["position_msd"] = msd
    tensors["position_msd_count"] = counts


def _neighbor_pairs(positions: np.ndarray, selected: np.ndarray) -> set[tuple[int, int]]:
    if len(selected) < 2:
        return set()
    ordered = selected[np.argsort(positions[selected])]
    if len(ordered) == 2:
        first, second = int(ordered[0]), int(ordered[1])
        return {(min(first, second), max(first, second))}
    pairs: set[tuple[int, int]] = set()
    for first_value, second_value in zip(
        ordered, np.roll(ordered, -1), strict=True
    ):
        first, second = int(first_value), int(second_value)
        pairs.add((min(first, second), max(first, second)))
    return pairs


def add_area_coupling(tensors: dict[str, np.ndarray]) -> None:
    area = tensors["area"]
    area_device = jnp.asarray(area)
    positions = tensors["wrapped_axial_position"]
    stable = tensors["stable"]
    segments = tensors["segment_id"]
    times = tensors["physical_time"]
    n_time = len(times)

    centered_device = jnp.full_like(area_device, jnp.nan)
    for segment_id in np.unique(segments[stable]):
        selected = stable & (segments == segment_id)
        selected_device = jnp.asarray(selected)
        mean_area = jnp.sum(jnp.where(selected_device, area_device, 0.0)) / jnp.sum(
            selected_device
        )
        centered_device = jnp.where(
            selected_device, area_device - mean_area, centered_device
        )

    neighbors = [
        _neighbor_pairs(positions[time_index], np.flatnonzero(stable[time_index]))
        for time_index in range(n_time)
    ]
    lag_count = max(0, n_time - 1)
    covariance = np.full(lag_count, np.nan)
    physical_lag = np.full(lag_count, np.nan)
    counts = np.zeros(lag_count, dtype=np.int64)
    for lag_index, lag in enumerate(range(1, lag_count + 1)):
        first_times: list[int] = []
        second_times: list[int] = []
        first_tracks: list[int] = []
        second_tracks: list[int] = []
        pair_lags: list[float] = []
        for first_time in range(n_time - lag):
            second_time = first_time + lag
            dt = float(times[second_time] - times[first_time])
            if dt <= 0.0:
                continue
            for first, second in neighbors[first_time] & neighbors[second_time]:
                if (
                    segments[first_time, first] != segments[second_time, first]
                    or segments[first_time, second] != segments[second_time, second]
                ):
                    continue
                first_times.extend((first_time, first_time))
                second_times.extend((second_time, second_time))
                first_tracks.extend((first, second))
                second_tracks.extend((second, first))
                pair_lags.extend((dt, dt))
        counts[lag_index] = len(first_times)
        if first_times:
            products_device = (
                centered_device[jnp.asarray(first_times), jnp.asarray(first_tracks)]
                * centered_device[
                    jnp.asarray(second_times), jnp.asarray(second_tracks)
                ]
            )
            covariance[lag_index] = float(
                jax.device_get(jnp.mean(products_device))
            )
            physical_lag[lag_index] = float(
                jax.device_get(jnp.mean(jnp.asarray(pair_lags)))
            )

    stable_device = jnp.asarray(stable)
    segments_device = jnp.asarray(segments)
    stable_total_device = jnp.where(
        jnp.any(stable_device, axis=1),
        jnp.sum(jnp.where(stable_device, area_device, 0.0), axis=1),
        jnp.nan,
    )
    mean_absolute_rate_device = jnp.full(n_time, jnp.nan)
    if n_time > 1:
        adjacent_valid = (
            stable_device[:-1]
            & stable_device[1:]
            & (segments_device[:-1] == segments_device[1:])
        )
        adjacent_dt = jnp.asarray(times[1:] - times[:-1])
        adjacent_valid &= adjacent_dt[:, np.newaxis] > 0.0
        adjacent_count = jnp.sum(adjacent_valid, axis=1)
        adjacent_sum = jnp.sum(
            jnp.where(
                adjacent_valid,
                jnp.abs(area_device[1:] - area_device[:-1]),
                0.0,
            ),
            axis=1,
        )
        adjacent_rate = jnp.where(
            adjacent_count > 0,
            adjacent_sum / adjacent_count / adjacent_dt,
            jnp.nan,
        )
        mean_absolute_rate_device = mean_absolute_rate_device.at[1:].set(
            adjacent_rate
        )
    stable_total = np.asarray(jax.device_get(stable_total_device))
    mean_absolute_rate = np.asarray(
        jax.device_get(mean_absolute_rate_device)
    )

    tensors["neighbor_area_covariance_frame_lag"] = np.arange(
        1, lag_count + 1, dtype=np.int64
    )
    tensors["neighbor_area_covariance_physical_lag"] = physical_lag
    tensors["neighbor_area_covariance"] = covariance
    tensors["neighbor_area_covariance_count"] = counts
    tensors["stable_total_area"] = stable_total
    tensors["stable_mean_absolute_area_rate"] = mean_absolute_rate


def add_stochastic_statistics(tensors: dict[str, np.ndarray]) -> None:
    add_conditional_dynamics(tensors)
    add_position_msd(tensors)
    add_area_coupling(tensors)
