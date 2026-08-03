from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..dynamics import area_diffusion_model

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
    covariance_lag = int(tensors["area_increment_covariance_frame_lag"])
    conservation_lag = int(tensors["area_increment_r_cons_frame_lag"])
    axis.axhline(0.0, color="0.65", linewidth=0.8)
    axis.set(
        xlabel=r"cyclic band separation $r$",
        ylabel=r"$C_r=\operatorname{Cov}(\Delta A_i,\Delta A_{i+r})$",
        title=(
            "Fixed band count; event-free intervals\n"
            rf"$C_1={c1:.6g}$ (lag {covariance_lag}); "
            rf"$r_{{\rm cons}}={r_cons:.6g}$ (lag {conservation_lag})"
        ),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / filename, dpi=180)
    plt.close(figure)
    return {"area_increment_conservation": filename}


