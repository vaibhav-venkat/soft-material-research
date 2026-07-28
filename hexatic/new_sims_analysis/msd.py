"""Center-of-mass mean squared displacement, its power-law fit, and its plot."""

from __future__ import annotations

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
) -> list[tuple[str, str, NDArray[np.float64]]]:
    """Return the ``(suffix, title, coordinates)`` MSD variants for one seed.

    Cartesian cases give one variant. Cylindrical cases give two: the COM
    mapped back to Cartesian ``(x, y, z)``, whose transverse part is bounded by
    the cylinder radius, and the unrolled surface coordinates ``(x, r theta)``,
    which keep the unwrapped azimuthal drift as an arc length.
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


def _fit_msd_power_law(
    tau: NDArray[np.float64],
    msd: NDArray[np.float64],
    tau_start: float,
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Fit ``MSD = A tau^alpha`` over the lag window ``tau > tau_start``.

    ``tau_start`` only selects the window; the regression runs against the lag
    time itself. Shifting the origin to ``tau - tau_start`` would bias alpha
    badly, because ``A tau^alpha`` is scale-free only about tau = 0: a purely
    diffusive ``MSD = c tau`` re-expressed as ``c (t + tau_start)`` is concave
    in ``t`` and fits as alpha well below one.

    The fit is a least-squares line through ``log MSD = log A + alpha log tau``,
    so both ``A`` and ``alpha`` come from every point in the window rather than
    pinning ``A`` to a single frame. Returns ``(alpha, A, tau_fit, msd_fit)``.
    """
    usable = (tau > tau_start) & (tau > 0.0) & (msd > 0.0)
    if np.count_nonzero(usable) < 2:
        raise ValueError(
            f"need at least two positive MSD samples after tau={tau_start}"
        )
    tau_fit = tau[usable]
    alpha, log_a = np.polyfit(np.log(tau_fit), np.log(msd[usable]), 1)
    amplitude = float(np.exp(log_a))
    return float(alpha), amplitude, tau_fit, amplitude * tau_fit**alpha


def _plot_msd(
    tau: NDArray[np.float64],
    msd: NDArray[np.float64],
    tau_fit: NDArray[np.float64],
    msd_fit: NDArray[np.float64],
    alpha: float,
    n_seeds: int,
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
            r"$\langle (\mathrm{COM}(t_0+\Delta t)"
            r"-\mathrm{COM}(t_0))^2 \rangle_{t_0}$"
        ),
    )
    # TEMPORARY: power-law fit overlay disabled; MSD curve only.
    # axis.plot(
    #     tau_fit,
    #     msd_fit,
    #     color=palette[3],
    #     ls="--",
    #     lw=1.8,
    # )
    # # Anchor at the geometric middle so the label sits mid-line on log axes.
    # anchor = int(
    #     np.argmin(np.abs(np.log(tau_fit) - np.mean(np.log(tau_fit))))
    # )
    # axis.annotate(
    #     rf"$\alpha = {alpha:.3f}$",
    #     xy=(tau_fit[anchor], msd_fit[anchor]),
    #     xytext=(-8, 12),
    #     textcoords="offset points",
    #     ha="right",
    #     color=palette[3],
    # )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(f"Center-of-mass MSD — {label} ({n_seeds} seeds)")
    axis.set_xlabel(r"lag time $\Delta t$")
    axis.set_ylabel("MSD")
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
