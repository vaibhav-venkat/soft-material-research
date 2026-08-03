from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dynamics import DYNAMIC_PROPERTIES, area_diffusion_model


_LEGACY_NET_PLOTS = (
    "net_band_area.png",
    "net_mean_width.png",
    "net_axial_displacement.png",
    "net_width_variance.png",
    "net_centerline_variance.png",
    "net_interface_length.png",
    *(f"net_centerline_mode_{mode}.png" for mode in range(1, 7)),
    *(f"net_width_mode_{mode}.png" for mode in range(1, 7)),
)


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


def _plot_area_coupling(tensors: dict[str, np.ndarray], output_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    covariance_file = "neighbor_area_covariance.png"
    usable = (
        np.isfinite(tensors["neighbor_area_covariance_physical_lag"])
        & np.isfinite(tensors["neighbor_area_covariance"])
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(
        tensors["neighbor_area_covariance_physical_lag"][usable],
        tensors["neighbor_area_covariance"][usable],
        "o-",
        color="black",
        ms=3,
    )
    axis.axhline(0.0, color="0.65", linewidth=0.8)
    axis.set(xlabel=r"lag $\tau$", ylabel=r"neighbor $C^A(\tau)$")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / covariance_file, dpi=180)
    plt.close(figure)
    outputs["neighbor_area_covariance"] = covariance_file

    redistribution_file = "stable_area_redistribution.png"
    figure, area_axis = plt.subplots(figsize=(8.0, 4.7))
    rate_axis = area_axis.twinx()
    time = tensors["physical_time"]
    area_axis.plot(time, tensors["stable_total_area"], color="black", label=r"$A_{\rm total}$")
    rate_axis.plot(
        time,
        tensors["stable_mean_absolute_area_rate"],
        color="tab:red",
        alpha=0.8,
        label=r"$\langle |\Delta A/\Delta\tau|\rangle$",
    )
    area_axis.set(xlabel=r"physical time $\tau$", ylabel=r"stable $A_{\rm total}$")
    rate_axis.set_ylabel(r"mean $|\Delta A/\Delta\tau|$")
    area_axis.grid(alpha=0.25)
    lines = area_axis.lines + rate_axis.lines
    area_axis.legend(lines, [str(line.get_label()) for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(output_dir / redistribution_file, dpi=180)
    plt.close(figure)
    outputs["stable_area_redistribution"] = redistribution_file
    return outputs


def _plot_area_diagnostics(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    perimeter_file = "stable_interface_length_vs_area.png"
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    stable = tensors["stable"]
    area = tensors["area"]
    perimeter = tensors["interface_length"]
    track_ids = tensors["track_id"]
    for track_column, track_id in enumerate(track_ids):
        usable = (
            stable[:, track_column]
            & np.isfinite(area[:, track_column])
            & np.isfinite(perimeter[:, track_column])
        )
        if np.any(usable):
            axis.scatter(
                area[usable, track_column],
                perimeter[usable, track_column],
                s=16,
                alpha=0.65,
                label=f"track {int(track_id)}",
            )
    axis.set(xlabel=r"band area $A$", ylabel=r"total boundary length $P$")
    axis.grid(alpha=0.25)
    if axis.collections:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / perimeter_file, dpi=180)
    plt.close(figure)
    outputs["stable_interface_length_vs_area"] = perimeter_file

    alpha_file = "area_diffusion_alpha.png"
    valid = tensors["area_diffusion_fit_valid"]
    physical_lag = tensors["area_diffusion_fit_physical_lag"]
    alpha = tensors["area_diffusion_fit_alpha"]
    usable = valid & np.isfinite(physical_lag) & np.isfinite(alpha)
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(physical_lag[usable], alpha[usable], "o-", color="black")
    for point_index, (lag, exponent) in enumerate(
        zip(physical_lag[usable], alpha[usable], strict=True)
    ):
        axis.annotate(
            rf"$\alpha={float(exponent):.3g}$",
            (float(lag), float(exponent)),
            xytext=(5, 8 if point_index % 2 == 0 else -14),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        xlabel=r"physical lag $\Delta\tau$",
        ylabel=r"fitted exponent $\alpha$",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / alpha_file, dpi=180)
    plt.close(figure)
    outputs["area_diffusion_alpha"] = alpha_file
    return outputs


def plot_characterization(
    tensors: dict[str, np.ndarray],
    output_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in _LEGACY_NET_PLOTS:
        (output_dir / filename).unlink(missing_ok=True)

    outputs: dict[str, str] = {}
    for plot_index, prop in enumerate(DYNAMIC_PROPERTIES, start=1):
        filename = f"stochastic_{prop.slug}.png"
        _plot_conditional_property(
            tensors,
            slug=prop.slug,
            label=prop.label,
            output=output_dir / filename,
            include_msd=prop.slug == "axial_position",
        )
        outputs[prop.slug] = filename
        if progress is not None:
            progress(
                f"stage=plots stochastic={plot_index}/{len(DYNAMIC_PROPERTIES)} "
                f"property={prop.slug}"
            )
    outputs.update(_plot_area_coupling(tensors, output_dir))
    outputs.update(_plot_area_diagnostics(tensors, output_dir))
    if progress is not None:
        progress("stage=plots coupling=2/2 complete")
        progress("stage=plots area_diagnostics=2/2 complete")
    return outputs
