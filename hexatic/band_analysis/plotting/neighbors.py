from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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



