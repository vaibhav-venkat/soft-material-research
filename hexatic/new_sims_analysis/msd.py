"""Center-of-mass mean squared displacement and instantaneous exponent plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from .plotting import _save_figure


def _msd_from_com(
    com: NDArray[np.float64],
    max_lag: int,
) -> NDArray[np.float64]:
    """Return one seed's lag-averaged COM mean squared displacement.

    ``MSD(dt) = <(COM(t0 + dt) - COM(t0))^2>`` averaged over every available
    time origin ``t0``, which is what makes the curve smooth and essentially
    monotonic; a single origin gives one sample per lag and wanders instead.
    Sample counts fall off as the lag approaches the trajectory length, so the
    tail is still the noisiest part.
    """
    n_frames = len(com)
    if max_lag < 0 or max_lag > n_frames - 1:
        raise ValueError(f"max_lag {max_lag} out of range for {n_frames} frames")
    msd = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        displacement = com[lag:] - com[: n_frames - lag]
        msd[lag] = np.mean(np.sum(displacement**2, axis=1))
    return msd


def _msd_variants(
    com: NDArray[np.float64],
    cylindrical: bool,
) -> list[tuple[str, str, NDArray[np.float64]]]:
    """Return ``(suffix, title, coordinates)`` MSD variants.

    Cartesian cases give one variant. Cylindrical cases give two: the COM
    mapped back to Cartesian ``(x, y, z)``, whose transverse part is bounded by
    the cylinder radius, and the unrolled surface coordinates ``(x, r theta)``,
    which keep the unwrapped azimuthal drift as an arc length. Consequently,
    cylindrical Cartesian transport has only one asymptotically unbounded
    direction, while the unrolled surface has two.
    """
    if not cylindrical:
        return [("", "", com)]
    x, theta, r = com[:, 0], com[:, 1], com[:, 2]
    cartesian = np.stack(
        [x, r * np.cos(theta), r * np.sin(theta)], axis=1
    )
    unrolled = np.stack([x, r * theta], axis=1)
    return [
        ("_cartesian", r" — Cartesian $(x, y, z)$", cartesian),
        ("_unrolled", r" — surface $(x, r\theta)$", unrolled),
    ]


def _instantaneous_msd_exponent(
    tau: NDArray[np.float64],
    msd: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(tau, d log(MSD) / d log(tau))`` on positive samples."""
    usable = np.isfinite(tau) & np.isfinite(msd) & (tau > 0.0) & (msd > 0.0)
    if np.count_nonzero(usable) < 2:
        raise ValueError("need at least two positive finite MSD samples")
    tau_positive = tau[usable]
    if np.any(np.diff(tau_positive) <= 0.0):
        raise ValueError("positive lag times must be strictly increasing")
    log_tau = np.log(tau_positive)
    log_msd = np.log(msd[usable])
    edge_order = 2 if len(log_tau) >= 3 else 1
    alpha = np.gradient(log_msd, log_tau, edge_order=edge_order)
    return tau_positive, np.asarray(alpha, dtype=np.float64)


def _plot_msd(
    tau: NDArray[np.float64],
    msd: NDArray[np.float64],
    n_seeds: int,
    n_particles: int,
    label: str,
    output: Path,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    palette = sns.color_palette("colorblind")
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)

    # Log axes cannot show zero lag or a zero/negative MSD, so drop those
    # points rather than letting matplotlib clip them silently.
    visible = (tau > 0.0) & (msd > 0.0)
    if not np.any(visible):
        raise ValueError("no positive MSD samples to plot on log-log axes")
    axis.plot(
        tau[visible],
        msd[visible],
        color=palette[0],
        lw=2.0,
        label=(
            rf"$N\,\langle (\mathrm{{COM}}(t_0+\Delta t)"
            rf"-\mathrm{{COM}}(t_0))^2 \rangle_{{t_0}}$, $N = {n_particles}$"
        ),
    )

    alpha_tau, alpha = _instantaneous_msd_exponent(tau, msd)
    alpha_axis = axis.twinx()
    alpha_axis.plot(
        alpha_tau,
        alpha,
        color=palette[1],
        lw=1.5,
        alpha=0.9,
        label=r"instantaneous $\alpha(\Delta t)$",
    )
    alpha_axis.set_ylabel(
        r"$\alpha(\Delta t)=d\log(\mathrm{MSD})/d\log(\Delta t)$",
        color=palette[1],
    )
    alpha_axis.tick_params(axis="y", colors=palette[1])

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(f"Center-of-mass MSD — {label} ({n_seeds} seeds)")
    axis.set_xlabel(r"lag time $\Delta t$")
    axis.set_ylabel("MSD (per-particle equivalent)")
    axis.set_xlim(float(tau[visible][0]), float(tau[visible][-1]))
    axis.grid(which="major", color="0.85", lw=0.7)
    axis.grid(which="minor", color="0.94", lw=0.5)
    lines = axis.get_lines() + alpha_axis.get_lines()
    axis.legend(lines, [str(line.get_label()) for line in lines], frameon=False)
    sns.despine(ax=axis, right=False)
    sns.despine(ax=alpha_axis, left=True, right=False)

    _save_figure(figure, output)
