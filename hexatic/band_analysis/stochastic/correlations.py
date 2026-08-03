from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .common import ProgressCallback

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



