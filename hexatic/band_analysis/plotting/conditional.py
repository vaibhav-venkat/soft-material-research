from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..dynamics import DYNAMIC_PROPERTIES, area_diffusion_model

def _plot_conditional_property(
    tensors: dict[str, np.ndarray],
    *,
    slug: str,
    label: str,
    output: Path,
    include_msd: bool,
) -> None:
    panel_count = 3 if include_msd else 2
    figure, axes = plt.subplots(1, panel_count, figsize=(6.2 * panel_count, 4.7))
    prefix = f"dynamics_{slug}"
    centers = tensors[f"{prefix}_bin_center"]
    drift = tensors[f"{prefix}_drift"]
    diffusion = tensors[f"{prefix}_diffusion"]
    mean_lags = tensors[f"{prefix}_mean_physical_lag"]
    frame_lags = tensors["dynamics_frame_lag"]
    colors = matplotlib.colormaps["viridis"](
        np.linspace(0.05, 0.95, max(1, len(frame_lags)))
    )
    for lag_index, (frame_lag, color) in enumerate(
        zip(frame_lags, colors, strict=True)
    ):
        usable = np.isfinite(centers) & np.isfinite(drift[lag_index])
        if not np.any(usable):
            continue
        physical_lag = float(np.nanmean(mean_lags[lag_index, usable]))
        legend = rf"$\Delta m={int(frame_lag)}$, $\Delta\tau={physical_lag:.4g}$"
        axes[0].plot(centers[usable], drift[lag_index, usable], "o-", ms=3, color=color, label=legend)
        usable_diffusion = np.isfinite(centers) & np.isfinite(diffusion[lag_index])
        axes[1].plot(
            centers[usable_diffusion],
            diffusion[lag_index, usable_diffusion],
            "o-",
            ms=3,
            color=color,
            label=legend,
        )
        if (
            slug == "area"
            and tensors["area_diffusion_fit_valid"][lag_index]
            and np.any(usable_diffusion)
        ):
            fit_area = np.linspace(
                float(np.min(centers[usable_diffusion])),
                float(np.max(centers[usable_diffusion])),
                200,
            )
            axes[1].plot(
                fit_area,
                area_diffusion_model(
                    fit_area,
                    float(tensors["area_diffusion_fit_d0"][lag_index]),
                    float(tensors["area_diffusion_fit_gamma"][lag_index]),
                    float(tensors["area_diffusion_fit_alpha"][lag_index]),
                ),
                "--",
                color=color,
                linewidth=1.2,
            )
    axes[0].axhline(0.0, color="0.65", linewidth=0.8)
    axes[0].set(xlabel=label, ylabel=rf"$F_{{{label.strip('$')}}}$")
    axes[1].set(xlabel=label, ylabel=rf"$D_{{{label.strip('$')}}}$")
    for axis in axes[:2]:
        if axis.lines:
            axis.legend(fontsize=7)
        axis.grid(alpha=0.25)
    if include_msd:
        usable = (
            np.isfinite(tensors["position_msd_physical_lag"])
            & np.isfinite(tensors["position_msd"])
        )
        axes[2].plot(
            tensors["position_msd_physical_lag"][usable],
            tensors["position_msd"][usable],
            "o-",
            color="black",
            ms=3,
        )
        axes[2].set(xlabel=r"lag $\tau$", ylabel=r"$\mathrm{MSD}_X(\tau)$")
        axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)



