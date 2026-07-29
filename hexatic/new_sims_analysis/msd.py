"""Center-of-mass mean squared displacement, its power-law fit, and its plot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns


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
) -> list[tuple[str, str, NDArray[np.float64], int]]:
    """Return ``(suffix, title, coordinates, transport_dimensions)`` variants.

    Cartesian cases give one variant. Cylindrical cases give two: the COM
    mapped back to Cartesian ``(x, y, z)``, whose transverse part is bounded by
    the cylinder radius, and the unrolled surface coordinates ``(x, r theta)``,
    which keep the unwrapped azimuthal drift as an arc length. Consequently,
    cylindrical Cartesian transport is asymptotically one-dimensional (only
    ``x`` is unbounded), while unrolled surface transport is two-dimensional.
    """
    if not cylindrical:
        return [("", "", com, com.shape[1])]
    x, theta, r = com[:, 0], com[:, 1], com[:, 2]
    cartesian = np.stack(
        [x, r * np.cos(theta), r * np.sin(theta)], axis=1
    )
    unrolled = np.stack([x, r * theta], axis=1)
    return [
        ("_cartesian", r" — Cartesian $(x, y, z)$", cartesian, 1),
        ("_unrolled", r" — surface $(x, r\theta)$", unrolled, 2),
    ]


@dataclass(frozen=True)
class MsdFit:
    """Fits of the MSD over the lag window ``[tau_min, tau_max]``.

    Two fits are carried side by side: the free power law ``A tau^alpha``, and
    a forced-linear fit that assumes alpha = 1 regardless of what the power law
    actually found. The linear fit uses the conventional ``MSD = 2 d D tau``,
    so ``diffusion`` is the raw slope over ``2 d`` and is directly comparable
    with a literature diffusion constant.
    """

    tau_min: float
    tau_max: float
    alpha: float
    amplitude: float
    dimensions: int
    slope: float
    diffusion: float
    tau: NDArray[np.float64]
    msd: NDArray[np.float64]
    msd_linear: NDArray[np.float64]


def _fit_msd_power_law(
    tau: NDArray[np.float64],
    msd: NDArray[np.float64],
    tau_min: float,
    tau_max: float,
    dimensions: int,
) -> MsdFit:
    """Fit ``MSD = A tau^alpha`` over the closed lag window ``[tau_min, tau_max]``.

    The window only selects points; the regression runs against the lag time
    itself. Shifting the origin to ``tau - tau_min`` would bias alpha badly,
    because ``A tau^alpha`` is scale-free only about tau = 0: a purely diffusive
    ``MSD = c tau`` re-expressed as ``c (t + tau_min)`` is concave in ``t`` and
    fits as alpha well below one.

    The fit is a least-squares line through ``log MSD = log A + alpha log tau``,
    so both ``A`` and ``alpha`` come from every point in the window.
    """
    if not np.isfinite(tau_min) or tau_min <= 0.0:
        raise ValueError(f"tau_min must be finite and positive, got {tau_min}")
    if tau_max <= tau_min:
        raise ValueError(
            f"tau_max ({tau_max}) must exceed tau_min ({tau_min})"
        )
    usable = (tau >= tau_min) & (tau <= tau_max) & (tau > 0.0) & (msd > 0.0)
    if np.count_nonzero(usable) < 2:
        raise ValueError(
            "need at least two positive MSD samples in the lag window "
            f"[{tau_min}, {tau_max}]; the available lags span "
            f"[{tau[tau > 0.0].min() if np.any(tau > 0.0) else 0.0}, {tau.max()}]"
        )
    if dimensions < 1:
        raise ValueError(f"dimensions must be at least 1, got {dimensions}")
    tau_fit = tau[usable]
    msd_fit = msd[usable]
    alpha, log_a = np.polyfit(np.log(tau_fit), np.log(msd_fit), 1)
    amplitude = float(np.exp(log_a))

    # Forced alpha = 1: least squares through the origin, since MSD(0) = 0 is
    # exact rather than a fitted quantity. Converted with MSD = 2 d D tau.
    slope = float(np.dot(tau_fit, msd_fit) / np.dot(tau_fit, tau_fit))
    return MsdFit(
        tau_min=float(tau_min),
        tau_max=float(tau_max),
        alpha=float(alpha),
        amplitude=amplitude,
        dimensions=int(dimensions),
        slope=slope,
        diffusion=slope / (2.0 * float(dimensions)),
        tau=tau_fit,
        msd=amplitude * tau_fit ** float(alpha),
        msd_linear=slope * tau_fit,
    )


def _plot_msd(
    tau: NDArray[np.float64],
    msd: NDArray[np.float64],
    fits: Sequence[MsdFit],
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
    # On log-log a "slight offset" has to be multiplicative: a constant factor
    # is a constant vertical shift, so the dashed line stays parallel to the
    # data instead of being sheared the way an additive offset would.
    for index, fit in enumerate(fits):
        color = palette[(index + 3) % len(palette)]
        offset = 1.18 * 1.12**index
        # Anchor at the geometric middle so the labels sit mid-line.
        anchor = int(
            np.argmin(np.abs(np.log(fit.tau) - np.mean(np.log(fit.tau))))
        )

        above = fit.msd * offset
        axis.plot(
            fit.tau,
            above,
            color=color,
            ls="--",
            lw=1.8,
            label=(
                rf"$\alpha = {fit.alpha:.3f}$  "
                rf"($\Delta t \in [{fit.tau_min:g}, {fit.tau_max:g}]$)"
            ),
        )
        axis.annotate(
            rf"$\alpha = {fit.alpha:.3f}$",
            xy=(fit.tau[anchor], above[anchor]),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            color=color,
        )

        # The forced-linear fit gets the reciprocal offset, so it sits the same
        # distance below the data as the power law sits above it.
        below = fit.msd_linear / offset
        axis.plot(
            fit.tau,
            below,
            color=color,
            ls="-.",
            lw=1.8,
            label=(
                rf"$D = {fit.diffusion:.4g}$  "
                rf"($\mathrm{{MSD}} = 2d_\mathrm{{eff}}D\,\Delta t$, "
                rf"$d_\mathrm{{eff}} = {fit.dimensions}$)"
            ),
        )
        axis.annotate(
            rf"$D = {fit.diffusion:.4g}$",
            xy=(fit.tau[anchor], below[anchor]),
            # The fit line rises to the right, so below-right clears it just as
            # above-left clears it for the alpha label.
            xytext=(8, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            color=color,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(f"Center-of-mass MSD — {label} ({n_seeds} seeds)")
    axis.set_xlabel(r"lag time $\Delta t$")
    axis.set_ylabel("MSD (per-particle equivalent)")
    axis.set_xlim(float(tau[visible][0]), float(tau[visible][-1]))
    axis.grid(which="major", color="0.85", lw=0.7)
    axis.grid(which="minor", color="0.94", lw=0.5)
    axis.legend(frameon=False)
    sns.despine(ax=axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.new_sims_analysis"},
    )
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
