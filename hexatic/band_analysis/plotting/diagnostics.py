from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..dynamics import area_drift_constant_model, area_drift_cubic_model

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



