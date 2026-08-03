from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..characterization import N_MODES

MAX_FRAME_LAG = 2
TARGET_BINS = 20
MIN_BIN_SAMPLES = 50
PREFERRED_BIN_SAMPLES = 100
ProgressCallback = Callable[[str], None]


def _weighted_rmse(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(np.sqrt(np.average((observed - predicted) ** 2, weights=weights)))


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


