from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dynamics import (
    DYNAMIC_PROPERTIES,
    area_diffusion_model,
    area_drift_constant_model,
    area_drift_cubic_model,
)
from .tracking import EventCode


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


def _plot_area_heterogeneity_events(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "area_heterogeneity_events.png"
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 10.0), sharex=True)
    time = tensors["physical_time"]
    metrics = (
        (
            tensors["instantaneous_area_variance"],
            r"$V_A(t)$",
        ),
        (
            tensors["instantaneous_area_cv_squared"],
            r"$CV_A^2(t)$",
        ),
        (
            tensors["instantaneous_maximum_area_fraction"],
            r"$f_{\max}(t)$",
        ),
    )
    for axis, (values, ylabel) in zip(axes, metrics, strict=True):
        axis.plot(time, values, color="black", linewidth=1.4)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    event_styles = (
        (EventCode.BIRTH, "birth", "tab:green", "^"),
        (EventCode.DISAPPEARANCE, "death", "tab:red", "v"),
        (EventCode.MERGE, "merge", "tab:orange", "X"),
        (EventCode.SPLIT, "split", "tab:blue", "P"),
    )
    frame_event_code = tensors["frame_event_code"]
    for event_code, label, color, marker in event_styles:
        occurrences = np.sum(frame_event_code == int(event_code), axis=1)
        event_indices = np.flatnonzero(occurrences > 0)
        if not event_indices.size:
            continue
        for axis, (values, _) in zip(axes, metrics, strict=True):
            for event_index in event_indices:
                axis.axvline(
                    time[event_index],
                    color=color,
                    linewidth=0.8,
                    alpha=0.18,
                    zorder=1,
                )
            finite_events = event_indices[np.isfinite(values[event_indices])]
            if finite_events.size:
                axis.scatter(
                    time[finite_events],
                    values[finite_events],
                    s=45.0 + 12.0 * (occurrences[finite_events] - 1),
                    color=color,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                    label=label if axis is axes[0] else None,
                )
        if not np.any(np.isfinite(metrics[0][0][event_indices])):
            axes[0].axvline(
                time[event_indices[0]],
                color=color,
                linewidth=0.8,
                alpha=0.5,
                label=label,
            )

    handles, _ = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(ncol=4, fontsize=8, loc="best")
    axes[-1].set_xlabel(r"physical time $\tau$")
    figure.suptitle(
        "Instantaneous band-area heterogeneity and tracking events"
    )
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"area_heterogeneity_events": filename}


def _plot_area_cv_squared_drift(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "area_cv_squared_drift_fixed_n.png"
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    prefix = "dynamics_area_cv_squared_fixed_n"
    centers = tensors[f"{prefix}_bin_center"]
    drift = tensors[f"{prefix}_drift"]
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
        axis.plot(
            centers[usable],
            drift[lag_index, usable],
            "o-",
            ms=3,
            color=color,
            label=(
                rf"$\Delta m={int(frame_lag)}$, "
                rf"$\Delta\tau={physical_lag:.4g}$"
            ),
        )
    axis.axhline(0.0, color="0.65", linewidth=0.8)
    axis.set(
        xlabel=r"$C=CV_A^2$",
        ylabel=r"$F_C(C)=\langle\Delta C\mid C\rangle/\Delta\tau$",
        title="Fixed band count; event-free intervals only",
    )
    axis.grid(alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"area_cv_squared_drift_fixed_n": filename}


def _plot_normalized_neighbor_relative_area_drift(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "normalized_neighbor_relative_area_drift.png"
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    prefix = "dynamics_normalized_neighbor_relative_area"
    centers = tensors[f"{prefix}_bin_center"]
    drift = tensors[f"{prefix}_drift"]
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
        axis.plot(
            centers[usable],
            drift[lag_index, usable],
            "o-",
            ms=3,
            color=color,
            label=(
                rf"$\Delta m={int(frame_lag)}$, "
                rf"$\Delta\tau={physical_lag:.4g}$"
            ),
        )
    axis.axhline(0.0, color="0.65", linewidth=0.8)
    axis.set(
        xlabel=(
            r"$\widetilde{\delta A}_i="
            r"\frac{A_i-(A_{i-1}+A_{i+1})/2}"
            r"{(A_{i-1}+A_i+A_{i+1})/3}$"
        ),
        ylabel=(
            r"$F_{\widetilde{\delta A}}(\widetilde{\delta A})="
            r"\langle\Delta A_i\mid\widetilde{\delta A}_i\rangle/\Delta\tau$"
        ),
    )
    axis.grid(alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"normalized_neighbor_relative_area_drift": filename}


def _plot_neighbor_relative_area_drift(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "neighbor_relative_area_drift.png"
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    prefix = "dynamics_neighbor_relative_area"
    centers = tensors[f"{prefix}_bin_center"]
    drift = tensors[f"{prefix}_drift"]
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
        axis.plot(
            centers[usable],
            drift[lag_index, usable],
            "o-",
            ms=3,
            color=color,
            label=(
                rf"$\Delta m={int(frame_lag)}$, "
                rf"$\Delta\tau={physical_lag:.4g}$"
            ),
        )
        if tensors["neighbor_relative_area_fit_valid"][lag_index]:
            fitted_lambda = float(
                tensors["neighbor_relative_area_fit_lambda"][lag_index]
            )
            rmse = float(tensors["neighbor_relative_area_fit_rmse"][lag_index])
            fit_delta = np.linspace(
                min(0.0, float(np.min(centers[usable]))),
                max(0.0, float(np.max(centers[usable]))),
                300,
            )
            axis.plot(
                fit_delta,
                fitted_lambda * fit_delta,
                "--",
                color=color,
                linewidth=1.5,
                label=(
                    rf"fit $\lambda={fitted_lambda:.4g}$; "
                    rf"RMSE={rmse:.4g}"
                ),
            )
    axis.axhline(0.0, color="0.65", linewidth=0.8)
    axis.set(
        xlabel=r"$\delta A_i=A_i-(A_{i-1}+A_{i+1})/2$",
        ylabel=(
            r"$F_{\delta A}(\delta A)="
            r"\langle\Delta A_i\mid\delta A_i\rangle/\Delta\tau$"
        ),
    )
    axis.grid(alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"neighbor_relative_area_drift": filename}


def _plot_neighbor_relative_area_change_drift(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "neighbor_relative_area_change_drift.png"
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    prefix = "dynamics_neighbor_relative_area_change"
    centers = tensors[f"{prefix}_bin_center"]
    drift = tensors[f"{prefix}_drift"]
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
        axis.plot(
            centers[usable],
            drift[lag_index, usable],
            "o-",
            ms=3,
            color=color,
            label=(
                rf"$\Delta m={int(frame_lag)}$, "
                rf"$\Delta\tau={physical_lag:.4g}$"
            ),
        )
        if tensors["neighbor_relative_area_change_fit_valid"][lag_index]:
            fitted_lambda = float(
                tensors["neighbor_relative_area_change_fit_lambda"][lag_index]
            )
            rmse = float(
                tensors["neighbor_relative_area_change_fit_rmse"][lag_index]
            )
            fit_delta = np.linspace(
                min(0.0, float(np.min(centers[usable]))),
                max(0.0, float(np.max(centers[usable]))),
                300,
            )
            axis.plot(
                fit_delta,
                fitted_lambda * fit_delta,
                "--",
                color=color,
                linewidth=1.5,
                label=(
                    rf"fit $\Lambda={fitted_lambda:.4g}$; "
                    rf"RMSE={rmse:.4g}"
                ),
            )
    axis.axhline(0.0, color="0.65", linewidth=0.8)
    axis.set(
        xlabel=r"$\delta A_i=A_i-(A_{i-1}+A_{i+1})/2$",
        ylabel=(
            r"$G_{\delta A}(\delta A)="
            r"\langle\Delta\delta A_i\mid\delta A_i\rangle/\Delta\tau$"
        ),
    )
    axis.grid(alpha=0.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"neighbor_relative_area_change_drift": filename}


def _plot_area_diffusion_fits(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "area_diffusion_fits.png"
    frame_lags = tensors["dynamics_frame_lag"]
    lag_count = len(frame_lags)
    figure, axes = plt.subplots(
        max(1, lag_count),
        1,
        figsize=(9.0, max(4.8, 4.2 * lag_count)),
        squeeze=False,
    )
    centers = tensors["dynamics_area_bin_center"]
    diffusion = tensors["dynamics_area_diffusion"]
    physical_lags = tensors["area_diffusion_fit_physical_lag"]
    for lag_index, frame_lag in enumerate(frame_lags):
        axis = axes[lag_index, 0]
        usable = np.isfinite(centers) & np.isfinite(diffusion[lag_index])
        axis.plot(
            centers[usable], diffusion[lag_index, usable], "o", color="black",
            label=r"measured $D_A$",
        )
        if np.any(usable):
            fit_area = np.linspace(
                float(np.min(centers[usable])),
                float(np.max(centers[usable])),
                300,
            )
            if tensors["area_diffusion_constant_fit_valid"][lag_index]:
                d0 = float(tensors["area_diffusion_constant_fit_d0"][lag_index])
                rmse = float(tensors["area_diffusion_constant_fit_rmse"][lag_index])
                axis.plot(
                    fit_area,
                    np.full(fit_area.shape, d0),
                    "--",
                    label=rf"$D_0={d0:.4g}$; RMSE={rmse:.4g}",
                )
            if tensors["area_diffusion_sqrt_fit_valid"][lag_index]:
                gamma = float(tensors["area_diffusion_sqrt_fit_gamma"][lag_index])
                rmse = float(tensors["area_diffusion_sqrt_fit_rmse"][lag_index])
                axis.plot(
                    fit_area,
                    gamma * np.sqrt(fit_area),
                    "-.",
                    label=rf"$\Gamma\sqrt{{A}}$: $\Gamma={gamma:.4g}$; RMSE={rmse:.4g}",
                )
            if tensors["area_diffusion_perimeter_fit_valid"][lag_index]:
                gamma_p = float(
                    tensors["area_diffusion_perimeter_fit_gamma_p"][lag_index]
                )
                rmse = float(
                    tensors["area_diffusion_perimeter_fit_rmse"][lag_index]
                )
                perimeter = tensors["area_diffusion_bin_mean_perimeter"][lag_index]
                perimeter_usable = usable & np.isfinite(perimeter)
                axis.plot(
                    centers[perimeter_usable],
                    gamma_p * perimeter[perimeter_usable],
                    ":",
                    linewidth=2.0,
                    label=rf"$\Gamma_P P$: $\Gamma_P={gamma_p:.4g}$; RMSE={rmse:.4g}",
                )
            if tensors["area_diffusion_fit_valid"][lag_index]:
                d0 = float(tensors["area_diffusion_fit_d0"][lag_index])
                gamma = float(tensors["area_diffusion_fit_gamma"][lag_index])
                alpha = float(tensors["area_diffusion_fit_alpha"][lag_index])
                rmse = float(tensors["area_diffusion_fit_rmse"][lag_index])
                axis.plot(
                    fit_area,
                    area_diffusion_model(fit_area, d0, gamma, alpha),
                    "-",
                    linewidth=1.5,
                    label=(
                        rf"$D_0+\Gamma A^\alpha$: $D_0={d0:.4g}$, "
                        rf"$\Gamma={gamma:.4g}$, $\alpha={alpha:.4g}$; "
                        rf"RMSE={rmse:.4g}"
                    ),
                )
        axis.set(
            xlabel=r"initial band area $A$",
            ylabel=r"$D_A(A;\Delta\tau)$",
            title=(
                rf"frame lag {int(frame_lag)}; "
                rf"$\Delta\tau={float(physical_lags[lag_index]):.4g}$"
            ),
        )
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7)
    if not lag_count:
        axes[0, 0].set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"area_diffusion_fits": filename}


def _plot_area_geometry_fits(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "area_geometry_fits.png"
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    stable = tensors["stable"]
    area = tensors["area"]
    width = tensors["mean_width"]
    perimeter = tensors["interface_length"]

    usable = stable & np.isfinite(area) & np.isfinite(width)
    axes[0].scatter(width[usable], area[usable], s=12, alpha=0.45)
    if bool(tensors["area_width_fit_valid"]):
        a0 = float(tensors["area_width_fit_a0"])
        a1 = float(tensors["area_width_fit_a1"])
        rmse = float(tensors["area_width_fit_rmse"])
        fit_width = np.linspace(float(np.min(width[usable])), float(np.max(width[usable])), 300)
        axes[0].plot(
            fit_width,
            a0 + a1 * fit_width,
            color="black",
            label=rf"$a_0={a0:.4g}$, $a_1={a1:.4g}$; RMSE={rmse:.4g}",
        )
    axes[0].set(xlabel=r"mean width $\bar w$", ylabel=r"area $A(\bar w)$")

    usable = stable & np.isfinite(area) & np.isfinite(perimeter)
    axes[1].scatter(area[usable], perimeter[usable], s=12, alpha=0.45)
    if bool(tensors["perimeter_area_fit_valid"]):
        p0 = float(tensors["perimeter_area_fit_p0"])
        ba = float(tensors["perimeter_area_fit_ba"])
        rmse = float(tensors["perimeter_area_fit_rmse"])
        fit_area = np.linspace(float(np.min(area[usable])), float(np.max(area[usable])), 300)
        axes[1].plot(
            fit_area,
            p0 + ba * fit_area,
            color="black",
            label=rf"$P_0={p0:.4g}$, $b_A={ba:.4g}$; RMSE={rmse:.4g}",
        )
    axes[1].set(xlabel=r"area $A$", ylabel=r"total boundary length $P(A)$")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"area_geometry_fits": filename}


def _plot_area_increment_conservation(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    filename = "area_increment_conservation.png"
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    separation = tensors["area_increment_covariance_separation"]
    covariance = tensors["area_increment_covariance"]
    usable = np.isfinite(covariance)
    axis.plot(separation[usable], covariance[usable], "o-", color="black")
    c1 = float(tensors["area_increment_c1"])
    r_cons = float(tensors["area_increment_r_cons"])
    axis.axhline(0.0, color="0.65", linewidth=0.8)
    axis.set(
        xlabel=r"cyclic band separation $r$",
        ylabel=r"$C_r=\operatorname{Cov}(\Delta A_i,\Delta A_{i+r})$",
        title=(
            "Fixed band count; event-free one-frame increments\n"
            rf"$C_1={c1:.6g}$; $r_{{\rm cons}}={r_cons:.6g}$"
        ),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"area_increment_conservation": filename}


def _plot_area_diagnostics(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    drift_fit_file = "area_drift_fits.png"
    frame_lags = tensors["dynamics_frame_lag"]
    lag_count = len(frame_lags)
    figure, axes = plt.subplots(
        max(1, lag_count),
        1,
        figsize=(9.0, max(4.8, 4.2 * lag_count)),
        squeeze=False,
    )
    centers = tensors["dynamics_area_bin_center"]
    drift = tensors["dynamics_area_drift"]
    physical_lags = tensors["area_drift_fit_physical_lag"]
    for lag_index, frame_lag in enumerate(frame_lags):
        axis = axes[lag_index, 0]
        usable = np.isfinite(centers) & np.isfinite(drift[lag_index])
        axis.plot(
            centers[usable],
            drift[lag_index, usable],
            "o",
            color="black",
            label=r"measured $F_A$",
        )
        if np.any(usable):
            fit_area = np.linspace(
                float(np.min(centers[usable])),
                float(np.max(centers[usable])),
                300,
            )
            if tensors["area_drift_constant_fit_valid"][lag_index]:
                nu = float(tensors["area_drift_constant_fit_nu"][lag_index])
                rmse = float(tensors["area_drift_constant_fit_rmse"][lag_index])
                axis.plot(
                    fit_area,
                    area_drift_constant_model(fit_area, nu),
                    "--",
                    label=rf"$-\nu$: $\nu={nu:.4g}$; RMSE={rmse:.4g}",
                )
            if tensors["area_drift_cubic_fit_valid"][lag_index]:
                c1 = float(tensors["area_drift_cubic_fit_c1"][lag_index])
                c3 = float(tensors["area_drift_cubic_fit_c3"][lag_index])
                area_star = float(
                    tensors["area_drift_cubic_fit_area_star"][lag_index]
                )
                rmse = float(tensors["area_drift_cubic_fit_rmse"][lag_index])
                axis.plot(
                    fit_area,
                    area_drift_cubic_model(fit_area, c1, c3, area_star),
                    ":",
                    linewidth=2.0,
                    label=(
                        rf"cubic: $c_1={c1:.4g}$, $c_3={c3:.4g}$, "
                        rf"$A_\ast={area_star:.4g}$; RMSE={rmse:.4g}"
                    ),
                )
        physical_lag = float(physical_lags[lag_index])
        axis.set(
            xlabel=r"initial band area $A$",
            ylabel=r"$F_A(A;\Delta\tau)$",
            title=(
                rf"frame lag {int(frame_lag)}; "
                rf"$\Delta\tau={physical_lag:.4g}$"
            ),
        )
        axis.axhline(0.0, color="0.65", linewidth=0.8)
        axis.grid(alpha=0.25)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    if not lag_count:
        axes[0, 0].text(
            0.5,
            0.5,
            "no conditional lags available",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / drift_fit_file, dpi=180)
    plt.close(figure)
    outputs["area_drift_fits"] = drift_fit_file

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

    ck_file = "area_chapman_kolmogorov.png"
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    lag = tensors["area_ck_base_physical_lag"]
    discrepancy = tensors["area_ck_weighted_total_variation"]
    usable_lag = np.isfinite(lag) & np.isfinite(discrepancy)
    axes[0, 0].plot(lag[usable_lag], discrepancy[usable_lag], "o-", color="black")
    axes[0, 0].set(
        xlabel=r"base physical lag $\Delta\tau$",
        ylabel="weighted total-variation discrepancy",
    )
    axes[0, 0].set_ylim(bottom=0.0)
    axes[0, 0].grid(alpha=0.25)

    valid_lags = np.flatnonzero(usable_lag)
    if valid_lags.size:
        lag_index = int(valid_lags[-1])
        direct = tensors["area_ck_direct_probability"][lag_index]
        composed = tensors["area_ck_composed_probability"][lag_index]
        difference = np.abs(direct - composed)
        finite_probability = np.concatenate(
            (direct[np.isfinite(direct)], composed[np.isfinite(composed)])
        )
        probability_max = (
            float(np.max(finite_probability)) if finite_probability.size else 1.0
        )
        panels = (
            (direct, r"direct $P_{2\Delta\tau}$", probability_max),
            (composed, r"composed $P_{\Delta\tau}^2$", probability_max),
            (difference, "absolute difference", None),
        )
        centers = tensors["dynamics_area_bin_center"]
        finite_centers = np.flatnonzero(np.isfinite(centers))
        last_bin = int(finite_centers[-1] + 1) if finite_centers.size else 0
        for axis, (matrix, title, vmax) in zip(
            (axes[0, 1], axes[1, 0], axes[1, 1]), panels, strict=True
        ):
            image = axis.imshow(
                matrix[:last_bin, :last_bin],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=0.0,
                vmax=vmax,
            )
            axis.set(
                xlabel="final area bin",
                ylabel="initial area bin",
                title=title,
            )
            figure.colorbar(image, ax=axis, shrink=0.85)
        figure.suptitle(
            "Area Chapman–Kolmogorov test: "
            rf"$\Delta\tau={float(lag[lag_index]):.4g}$; stable segments only"
        )
    else:
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.text(
                0.5,
                0.5,
                "insufficient rows with at least 50 increments",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / ck_file, dpi=180)
    plt.close(figure)
    outputs["area_chapman_kolmogorov"] = ck_file
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
    outputs.update(_plot_area_heterogeneity_events(tensors, output_dir))
    outputs.update(_plot_area_cv_squared_drift(tensors, output_dir))
    outputs.update(_plot_area_diffusion_fits(tensors, output_dir))
    outputs.update(_plot_area_geometry_fits(tensors, output_dir))
    outputs.update(_plot_area_increment_conservation(tensors, output_dir))
    outputs.update(_plot_neighbor_relative_area_drift(tensors, output_dir))
    outputs.update(
        _plot_neighbor_relative_area_change_drift(tensors, output_dir)
    )
    outputs.update(
        _plot_normalized_neighbor_relative_area_drift(tensors, output_dir)
    )
    outputs.update(_plot_area_diagnostics(tensors, output_dir))
    if progress is not None:
        progress("stage=plots coupling=2/2 complete")
        progress("stage=plots area_diagnostics=12/12 complete")
    return outputs
