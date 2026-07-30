"""Velocity, orientation, and inferred wall-force correlation curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve
import seaborn as sns

from .plotting import _save_figure


def _finite_difference(
    values: NDArray[np.float64],
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Second-order three-point finite difference on non-uniform coordinates.

    This replicates the Zig ``finiteDifference`` formula exactly so that
    the velocity signal matches the big_lx_analysis results.
    """
    n = len(values)
    if n != len(coords):
        raise ValueError("values and coordinates must have the same length")
    if n < 2:
        raise ValueError("need at least two samples for derivative")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain non-finite samples")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates contain non-finite samples")
    if np.any(np.diff(coords) <= 0.0):
        raise ValueError("coordinates must be strictly increasing")
    derivative = np.empty(n, dtype=np.float64)

    if n == 2:
        slope = (values[1] - values[0]) / (coords[1] - coords[0])
        derivative[0] = slope
        derivative[1] = slope
        return derivative

    # Interior points: three-point Lagrange interpolation derivative.
    for i in range(1, n - 1):
        h_before = coords[i] - coords[i - 1]
        h_after = coords[i + 1] - coords[i]
        a = -h_after / (h_before * (h_before + h_after))
        b = (h_after - h_before) / (h_before * h_after)
        c = h_before / (h_after * (h_before + h_after))
        derivative[i] = a * values[i - 1] + b * values[i] + c * values[i + 1]

    # First point: forward three-point.
    h1 = coords[1] - coords[0]
    h2 = coords[2] - coords[1]
    derivative[0] = (
        -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * values[0]
        + (h1 + h2) / (h1 * h2) * values[1]
        - h1 / (h2 * (h1 + h2)) * values[2]
    )

    # Last point: backward three-point.
    h_before = coords[n - 2] - coords[n - 3]
    h_last = coords[n - 1] - coords[n - 2]
    derivative[n - 1] = (
        h_last / (h_before * (h_before + h_last)) * values[n - 3]
        - (h_last + h_before) / (h_before * h_last) * values[n - 2]
        + (2.0 * h_last + h_before) / (h_last * (h_before + h_last)) * values[n - 1]
    )
    return derivative


def _finite_difference_vector(
    values: NDArray[np.float64],
    coords: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Component-wise finite difference of a (frames, components) series."""
    return np.stack(
        [_finite_difference(values[:, axis], coords) for axis in range(values.shape[1])],
        axis=1,
    )


def _normalized_autocorrelation(
    series: NDArray[np.float64],
    max_lag: int,
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return the unbiased autocorrelation of a scalar or vector series.

    ``series`` is either (frames,) or (frames, components). For the vector
    case, products are dot products summed over components. When ``normalize``
    is true, divide the result by its lag-zero value.
    """
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D series, got shape {series.shape}")
    n = len(series)
    if n < 3:
        raise ValueError("need at least three samples")
    if not np.all(np.isfinite(series)):
        raise ValueError("series contains non-finite samples")
    if max_lag < 0:
        raise ValueError("max_lag must be nonnegative")
    if max_lag > n - 2:
        raise ValueError(
            f"max_lag ({max_lag}) exceeds available lags "
            f"({n - 2} for {n} samples)"
        )

    correlation = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        sample_count = n - lag
        left = series[:sample_count]
        right = series[lag : lag + sample_count]
        correlation[lag] = float(np.sum(left * right)) / sample_count
    if not normalize:
        return correlation
    if correlation[0] == 0.0:
        raise ValueError("cannot normalize an autocorrelation with zero lag value")
    return correlation / correlation[0]


def _distinct_particle_correlation(
    q: NDArray[np.float32],
    max_lag: int,
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return the unbiased distinct-particle orientation correlation.

    Returns the j != i part of the double sum,

        1/N^2 sum_i sum_{j != i} q_i(t) . q_j(t + lag),

    obtained as the full double sum minus the self (i == j) term. The full
    term factorizes through the per-frame mean and is cheap; the self term
    does not factorize, so it needs one FFT per particle chunk.

    When ``normalize`` is true, divide the result by its lag-zero value.
    """
    n_frames, n_particles, _ = q.shape
    if max_lag < 0 or max_lag > n_frames - 2:
        raise ValueError(f"max_lag {max_lag} out of range for {n_frames} frames")

    pair_counts = np.arange(
        n_frames, n_frames - max_lag - 1, -1, dtype=np.float64
    )

    mean_q = q.mean(axis=1, dtype=np.float64)
    full = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        count = n_frames - lag
        full[lag] = np.sum(mean_q[:count] * mean_q[lag:])
    full /= pair_counts

    # Chunk the particles so a long run does not need one huge FFT temporary.
    self_total = np.zeros(max_lag + 1, dtype=np.float64)
    for start in range(0, n_particles, 512):
        chunk = q[:, start : start + 512].astype(np.float64)
        raw = fftconvolve(chunk, chunk[::-1], mode="full", axes=0)
        self_total += raw[n_frames - 1 : n_frames + max_lag].sum(axis=(1, 2))
    self_term = self_total / (pair_counts * n_particles**2)

    distinct = full - self_term
    if not normalize:
        return distinct
    if distinct[0] == 0.0:
        raise ValueError("distinct-particle correlation has zero lag-0 value")
    return distinct / distinct[0]


def _self_particle_orientation_correlation(
    q: NDArray[np.float32],
    max_lag: int,
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Return the unbiased autocorrelation of each particle with itself.

    The averaged products are ``q_i(t) . q_i(t + lag)`` with no cross-particle
    terms. FFT particle chunks compute all requested lags together. When
    ``normalize`` is true, divide the result by its lag-zero value.
    """
    n_frames, n_particles, _ = q.shape
    if max_lag < 0 or max_lag > n_frames - 2:
        raise ValueError(f"max_lag {max_lag} out of range for {n_frames} frames")

    total = np.zeros(max_lag + 1, dtype=np.float64)
    for start in range(0, n_particles, 512):
        chunk = q[:, start : start + 512].astype(np.float64)
        raw = fftconvolve(chunk, chunk[::-1], mode="full", axes=0)
        total += raw[n_frames - 1 : n_frames + max_lag].sum(axis=(1, 2))
    pair_counts = np.arange(
        n_frames, n_frames - max_lag - 1, -1, dtype=np.float64
    )
    correlation = total / (pair_counts * n_particles)
    if not normalize:
        return correlation
    if correlation[0] == 0.0:
        raise ValueError("self-particle autocorrelation has zero lag-0 value")
    return correlation / correlation[0]


def _lag_times(
    elapsed_time: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Return the native big-Lx lag grid after validating uniform spacing."""
    if len(elapsed_time) < 3:
        raise ValueError("at least three frames are required")
    spacing = float(elapsed_time[1] - elapsed_time[0])
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("invalid simulation time spacing")
    actual_spacing = np.diff(elapsed_time)
    tolerance = np.maximum(
        1.0e-12,
        1.0e-10 * np.maximum(np.abs(spacing), np.abs(actual_spacing)),
    )
    if (
        not np.all(np.isfinite(elapsed_time))
        or np.any(np.abs(actual_spacing - spacing) > tolerance)
    ):
        raise ValueError("simulation frames must have uniform, increasing spacing")
    return np.arange(max_lag + 1, dtype=np.float64) * spacing


def _correlation_curves(
    com: NDArray[np.float64],
    q: NDArray[np.float32],
    elapsed: NDArray[np.float64],
    max_lag: int,
    u0: float,
    gamma: float,
    cylindrical: bool,
    planar: bool,
    active_particle_ids: NDArray[np.int64] | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Reduce one seed to velocity, orientation, and force correlations.

    The wall-force correlation is divided by ``gamma**2`` so that every
    unnormalized curve has velocity-squared units.
    """
    velocity = _finite_difference_vector(com, elapsed)
    if cylindrical:
        velocity_signal = velocity[:, :1]
        orientation_signal = q[:, :, :1]
    elif planar:
        velocity_signal = velocity
        orientation_signal = q[:, :, :2]
    else:
        velocity_signal = velocity
        orientation_signal = q
    if active_particle_ids is None:
        active_particle_ids = np.arange(q.shape[1], dtype=np.int64)
    if (
        active_particle_ids.ndim != 1
        or len(active_particle_ids) == 0
        or np.any(active_particle_ids < 0)
        or np.any(active_particle_ids >= q.shape[1])
        or len(np.unique(active_particle_ids)) != len(active_particle_ids)
    ):
        raise ValueError("active particle IDs must be unique and in range")
    active_polarization = np.sum(
        orientation_signal[:, active_particle_ids], axis=1, dtype=np.float64
    ) / q.shape[1]
    wall_force = gamma * (velocity_signal - u0 * active_polarization)
    velocity_correlation = _normalized_autocorrelation(
        velocity_signal, max_lag, normalize=False
    )
    self_correlation = _self_particle_orientation_correlation(
        orientation_signal, max_lag, normalize=False
    )
    self_correlation *= u0**2 / q.shape[1]
    distinct_correlation = _distinct_particle_correlation(
        orientation_signal, max_lag, normalize=False
    )
    distinct_correlation *= u0**2
    wall_force_correlation = _normalized_autocorrelation(
        wall_force, max_lag, normalize=False
    )
    wall_force_correlation /= gamma**2
    return (
        velocity_correlation,
        self_correlation,
        distinct_correlation,
        wall_force_correlation,
    )


def _plot_correlation(
    lag_time: NDArray[np.float64],
    velocity_x: NDArray[np.float64],
    orientation_self_x: NDArray[np.float64],
    orientation_distinct_x: NDArray[np.float64],
    wall_force_x: NDArray[np.float64],
    normalize: bool,
    cylindrical: bool,
    planar: bool,
    label: str,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    palette = sns.color_palette("colorblind")
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)

    velocity_name = r"$C_{V_x}(\tau)$" if cylindrical else r"$C_{\mathbf{V}}(\tau)$"
    self_name = (
        r"$(U_0^2/N)C_{q_x}^{\mathrm{self}}(\tau)$"
        if cylindrical
        else r"$(U_0^2/N)C_{\mathbf{q}}^{\mathrm{self}}(\tau)$"
    )
    distinct_name = (
        r"$(U_0^2/N^2)\sum_{i\ne j}C_{q_{i,x}q_{j,x}}(\tau)$"
        if cylindrical
        else r"$(U_0^2/N^2)\sum_{i\ne j}C_{\mathbf{q}_i\mathbf{q}_j}(\tau)$"
    )
    force_name = (
        r"$\gamma^{-2}C_{F_{\mathrm{wall},x}}(\tau)$"
        if cylindrical
        else r"$\gamma^{-2}C_{\mathbf{F}_{\mathrm{wall}}}(\tau)$"
    )
    curves = (
        (velocity_x, palette[0], "-", velocity_name),
        (
            orientation_self_x,
            palette[1],
            "--",
            self_name,
        ),
        (
            orientation_distinct_x,
            palette[2],
            ":",
            distinct_name,
        ),
        (
            wall_force_x,
            palette[3],
            "-.",
            force_name,
        ),
    )
    for correlation, color, style, name in curves:
        axis.plot(lag_time, correlation, color=color, ls=style, lw=2.0, label=name)

    axis.axhline(0.0, color="0.65", lw=0.9)
    title = (
        "Normalized autocorrelation"
        if normalize
        else "Unnormalized autocorrelation"
    )
    geometry = (
        "2D Cartesian"
        if planar
        else "cylindrical"
        if cylindrical
        else "3D Cartesian"
    )
    axis.set_title(f"{title} — {label} ({geometry})")
    axis.set_xlabel("Lag time")
    axis.set_ylabel(title)
    axis.set_xlim(0.0, lag_time[-1])
    axis.grid(axis="y", color="0.9", lw=0.7)
    axis.legend(frameon=False, ncol=2)
    sns.despine(ax=axis)

    _save_figure(figure, output)
