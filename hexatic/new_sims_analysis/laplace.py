"""Complex Laplace transform of the velocity correlation, and its heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns


def _simpson_weights(sample_count: int, spacing: float) -> NDArray[np.float64]:
    """Match the Laplace-analysis Simpson rule, including an even-size tail."""
    if sample_count < 2 or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("Laplace transform needs at least two uniform samples")
    weights = np.zeros(sample_count, dtype=np.float64)
    if sample_count == 2:
        weights[:] = spacing / 2.0
        return weights
    simpson_count = sample_count - 1 if sample_count % 2 == 0 else sample_count
    weights[:simpson_count:2] = 2.0 * spacing / 3.0
    weights[1:simpson_count:2] = 4.0 * spacing / 3.0
    weights[0] = spacing / 3.0
    weights[simpson_count - 1] = spacing / 3.0
    if sample_count % 2 == 0:
        weights[-1] += 5.0 * spacing / 12.0
        weights[-2] += 2.0 * spacing / 3.0
        weights[-3] -= spacing / 12.0
    return weights


def _laplace_transform(
    lag_time: NDArray[np.float64],
    correlation: NDArray[np.float64],
    r_points: int,
    omega_points: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128],
]:
    """Evaluate the same complex Laplace grid as big_lx_analysis."""
    if len(lag_time) != len(correlation) or len(lag_time) < 2:
        raise ValueError("Laplace inputs must have the same length of at least two")
    spacing = float(lag_time[1] - lag_time[0])
    duration = float(lag_time[-1] - lag_time[0])
    if duration <= 0.0:
        raise ValueError("Laplace correlation must span positive lag time")
    omega_max = min(np.pi / spacing, 20.0 * np.pi / duration)
    r = np.linspace(-10.0 / duration, 0.0, r_points, dtype=np.float64)
    omega = np.linspace(
        -omega_max, omega_max, omega_points, dtype=np.float64
    )
    weights = _simpson_weights(len(lag_time), spacing)
    weighted_correlation = weights * correlation
    envelope = np.exp(r[:, None] * lag_time[None, :])
    phase = np.exp(1j * omega[:, None] * lag_time[None, :])
    values = (phase * weighted_correlation[None, :]) @ envelope.T
    return r, omega, np.asarray(values, dtype=np.complex128)


def _plot_laplace(
    r: NDArray[np.float64],
    omega: NDArray[np.float64],
    values: NDArray[np.complex128],
    output: Path,
) -> None:
    """Copy the big_lx_analysis three-panel Laplace heatmap."""
    sns.set_theme(context="paper", style="ticks")
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 3.4),
        squeeze=False,
        constrained_layout=True,
    )
    fields = (
        np.log10(np.maximum(np.abs(values), np.finfo(float).tiny)),
        values.real,
        values.imag,
    )
    titles = (
        r"$\log_{10}|L(r,\omega)|$",
        r"$\Re L(r,\omega)$",
        r"$\Im L(r,\omega)$",
    )
    for column, (field, title) in enumerate(zip(fields, titles, strict=True)):
        axis = axes[0, column]
        image = axis.pcolormesh(
            r, omega, field, shading="auto", cmap="viridis"
        )
        figure.colorbar(image, ax=axis, pad=0.02)
        axis.set_xlabel(r"decay coordinate $r$")
        if column == 0:
            axis.set_ylabel(r"angular frequency $\omega$")
        axis.set_title(title)
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
