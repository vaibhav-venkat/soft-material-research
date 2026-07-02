"""Cache and report outputs for rho fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cache import write_npz_atomic
from .config import RhoFittingConfig
from .geometry import surface_lengths
from .io import ActiveMatterArrays
from .plots import write_density_plots
from .regression import StabilityResult
from .report import density_report_lines, write_report


def write_mechanical_outputs(
    active: ActiveMatterArrays,
    coarse: dict[str, np.ndarray],
    spectral: dict[str, np.ndarray],
    fit_payload: dict[str, np.ndarray | StabilityResult],
    config: RhoFittingConfig,
) -> tuple[Path, Path]:
    cache_path = config.output_dir / f"{config.case_id}_fit_result.npz"
    report_path = config.output_dir / f"{config.case_id}_rho_fitting_report.md"
    y_rho_fit = fit_payload["Y_rho_fit"]
    y_p_fit = fit_payload["Y_P_fit"]
    y_q_fit = fit_payload["Y_Q_fit"]
    assert isinstance(y_rho_fit, StabilityResult)
    assert isinstance(y_p_fit, StabilityResult)
    assert isinstance(y_q_fit, StabilityResult)
    write_npz_atomic(
        cache_path,
        overwrite=config.overwrite,
        metadata=cache_metadata(active, config) | {"analysis": "global_mechanical_moment_fits"},
        raw_rho=coarse["rho"],
        raw_P=coarse["P"],
        raw_Q=coarse["Q"],
        raw_A=coarse["A"],
        raw_J_rho=coarse["J_rho"],
        raw_J_P=coarse["J_P"],
        raw_J_Q=coarse["J_Q"],
        rho=spectral["rho"],
        P=spectral["P"],
        Q=spectral["Q"],
        A=spectral["A"],
        J_density=spectral["J_density"],
        J_rho=spectral["J_rho"],
        J_P=spectral["J_P"],
        J_Q=spectral["J_Q"],
        Y_rho=spectral["Y_rho"],
        Y_P=spectral["Y_P"],
        Y_Q=spectral["Y_Q"],
        F_rho_prediction=fit_payload["F_rho_prediction"],
        partial_t_rho=spectral["partial_t_rho"],
        temporal_power=spectral["temporal_power"],
        cheb_times=spectral["cheb_times"],
        cheb_scaled_times=spectral["cheb_scaled_times"],
        sample_indices=fit_payload["sample_indices"],
        Y_rho_rows=fit_payload["Y_rho_rows"],
        Y_P_rows=fit_payload["Y_P_rows"],
        Y_Q_rows=fit_payload["Y_Q_rows"],
        Y_rho_row_index=fit_payload["Y_rho_row_index"],
        Y_P_row_index=fit_payload["Y_P_row_index"],
        Y_Q_row_index=fit_payload["Y_Q_row_index"],
        Y_rho_library=fit_payload["Y_rho_library"],
        Y_P_library=fit_payload["Y_P_library"],
        Y_Q_library=fit_payload["Y_Q_library"],
        Y_rho_names=fit_payload["Y_rho_names"],
        Y_P_names=fit_payload["Y_P_names"],
        Y_Q_names=fit_payload["Y_Q_names"],
        Y_rho_coefficients=y_rho_fit.coefficients,
        Y_P_coefficients=y_p_fit.coefficients,
        Y_Q_coefficients=y_q_fit.coefficients,
        Y_rho_importance=y_rho_fit.importance,
        Y_P_importance=y_p_fit.importance,
        Y_Q_importance=y_q_fit.importance,
        Y_rho_active=y_rho_fit.active,
        Y_P_active=y_p_fit.active,
        Y_Q_active=y_q_fit.active,
        Y_rho_rmse=np.asarray(y_rho_fit.rmse),
        Y_P_rmse=np.asarray(y_p_fit.rmse),
        Y_Q_rmse=np.asarray(y_q_fit.rmse),
        Y_rho_r2=np.asarray(y_rho_fit.r2),
        Y_P_r2=np.asarray(y_p_fit.r2),
        Y_Q_r2=np.asarray(y_q_fit.r2),
    )
    write_report(
        report_path,
        mechanical_report_lines(
            case_id=config.case_id,
            nd=fit_payload["sample_indices"].shape[0],
            frames=active.coords.shape[0],
            grid_shape=(active.x_centers.size, active.theta_centers.size),
            sigma=config.settings.sigma,
            cheb_cutoff=config.settings.cheb_cutoff,
            fits={"Y_rho": y_rho_fit, "Y_P": y_p_fit, "Y_Q": y_q_fit},
        ),
        overwrite=config.overwrite,
    )
    return cache_path, report_path


def mechanical_report_lines(
    *,
    case_id: str,
    nd: int,
    frames: int,
    grid_shape: tuple[int, int],
    sigma: float,
    cheb_cutoff: int,
    fits: dict[str, StabilityResult],
) -> list[str]:
    lines = [
        f"# Rho fitting report: {case_id}",
        "",
        "## Settings",
        f"- frames: {frames}",
        f"- grid: {grid_shape[0]} x {grid_shape[1]}",
        f"- samples: {nd}",
        f"- sigma: {sigma:.8g}",
        f"- cheb_cutoff: {cheb_cutoff}",
        "- analysis: global mechanical moment fits with y = R theta",
        "",
    ]
    for target, fit in fits.items():
        lines.extend(
            [
                f"## {target}",
                f"- rmse: {fit.rmse:.8g}",
                f"- r2: {fit.r2:.8g}",
                f"- tau_index: {fit.tau_index if fit.tau_index is not None else 'none'}",
                "",
                "| term | active | coefficient | importance | raw corr |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, active, coefficient, importance, raw_correlation in zip(
            fit.labels,
            fit.active,
            fit.coefficients,
            fit.importance,
            fit.raw_correlations,
            strict=True,
        ):
            lines.append(
                f"| {label} | {int(active)} | {coefficient:.10g} | "
                f"{importance:.4f} | {format_float(raw_correlation)} |"
            )
        lines.append("")
    return lines


def write_density_outputs(
    active: ActiveMatterArrays,
    coarse: dict[str, np.ndarray],
    spectral: dict[str, np.ndarray],
    fit_payload: dict[str, np.ndarray | StabilityResult],
    config: RhoFittingConfig,
) -> tuple[Path, Path]:
    fit = fit_payload["fit"]
    assert isinstance(fit, StabilityResult), "fit payload is missing StabilityResult"
    cache_path = config.output_dir / f"{config.case_id}_fit_result.npz"
    report_path = config.output_dir / f"{config.case_id}_rho_fitting_report.md"
    write_npz_atomic(
        cache_path,
        overwrite=config.overwrite,
        metadata=cache_metadata(active, config),
        raw_rho=coarse["rho"],
        raw_J_density=coarse["J_density"],
        rho=spectral["rho"],
        J_density=spectral["J_density"],
        J_grad_rho=spectral["J_grad_rho"],
        J_grad_lap_rho=spectral["J_grad_lap_rho"],
        J_lap_rho_grad_rho=spectral["J_lap_rho_grad_rho"],
        J_grad_rho_cubed=spectral["J_grad_rho_cubed"],
        partial_t_rho=spectral["partial_t_rho"],
        temporal_power=spectral["temporal_power"],
        cheb_times=spectral["cheb_times"],
        cheb_scaled_times=spectral["cheb_scaled_times"],
        sample_indices=fit_payload["sample_indices"],
        X_density=fit_payload["X_density"],
        y_density=fit_payload["y_density"],
        y_pred=fit.y_pred,
        term_names=fit_payload["term_names"],
        term_labels=fit_payload["term_labels"],
        term_fluxes=fit_payload["term_fluxes"],
        coefficients=fit.coefficients,
        importance=fit.importance,
        importance_path=fit.importance_path,
        tau_values=fit.tau_values,
        tau_index=np.asarray(-1 if fit.tau_index is None else fit.tau_index),
        active=fit.active,
        raw_correlations=fit.raw_correlations,
        rmse=np.asarray(fit.rmse),
        r2=np.asarray(fit.r2),
    )
    write_report(
        report_path,
        density_report_lines(
            case_id=config.case_id,
            nd=fit_payload["sample_indices"].shape[0],
            frames=active.coords.shape[0],
            grid_shape=(active.x_centers.size, active.theta_centers.size),
            sigma=config.settings.sigma,
            cheb_cutoff=config.settings.cheb_cutoff,
            fit=fit,
        ),
        overwrite=config.overwrite,
    )
    if config.make_plots:
        lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
        write_density_plots(
            config.output_dir,
            config.case_id,
            fit,
            fit_payload["y_density"],
            spectral["rho"],
            spectral["partial_t_rho"],
            spectral["temporal_power"],
            config.settings.cheb_cutoff,
            lx,
            ly,
        )
    return cache_path, report_path


def cache_metadata(active: ActiveMatterArrays, config: RhoFittingConfig) -> dict[str, object]:
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    return {
        "case_id": config.case_id,
        "active_fields_path": str(config.paths.active_fields_path),
        "active_fields_mtime": config.paths.active_fields_path.stat().st_mtime,
        "Nx": int(active.x_centers.size),
        "Ny": int(active.theta_centers.size),
        "sigma": float(config.settings.sigma),
        "lx": float(lx),
        "ly": float(ly),
        "radius": float(active.radius),
        "cheb_cutoff": int(config.settings.cheb_cutoff),
        "nd": int(config.settings.nd),
        "seed": int(config.settings.seed),
        "replace": bool(config.settings.replace),
        "analysis": "global_density_flux_divergence",
    }


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"
