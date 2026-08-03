from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..tracking import EventCode

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



