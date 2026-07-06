"""Cache and report outputs for rho fitting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from .cache import write_npz_atomic
from .config import RhoFittingConfig
from .geometry import surface_lengths
from .io import ActiveMatterArrays
from .regression import StabilityResult
from .report import write_report


def write_mechanical_outputs(
    active: ActiveMatterArrays,
    coarse: dict[str, np.ndarray],
    spectral: dict[str, np.ndarray],
    fit_payload: Mapping[str, np.ndarray | StabilityResult],
    config: RhoFittingConfig,
) -> tuple[Path, Path]:
    """Write mechanical fit cache arrays and the markdown summary report."""
    cache_path = config.output_dir / f"{config.case_id}_fit_result.npz"
    report_path = config.output_dir / f"{config.case_id}_rho_fitting_report.md"
    y_rho_fit = fit_payload["Y_rho_fit"]
    y_p_fit = fit_payload["Y_P_fit"]
    y_q_fit = fit_payload["Y_Q_fit"]
    assert isinstance(y_rho_fit, StabilityResult)
    assert isinstance(y_p_fit, StabilityResult)
    assert isinstance(y_q_fit, StabilityResult)
    arrays = cast(Mapping[str, np.ndarray], fit_payload)
    assert config.settings is not None, "rho fitting settings were not initialized"
    write_npz_atomic(
        cache_path,
        overwrite=config.overwrite,
        metadata=cache_metadata(active, config) | {"analysis": "global_mechanical_moment_fits"},
        raw_rho=coarse["rho"],
        raw_P=coarse["P"],
        raw_Q=coarse["Q"],
        raw_A=coarse["A"],
        raw_psi6_sq=coarse["psi6_sq"],
        raw_J_rho=coarse["J_rho"],
        raw_J_P=coarse["J_P"],
        raw_J_Q=coarse["J_Q"],
        r_edges=coarse["r_edges"],
        r_centers=coarse["r_centers"],
        rho=spectral["rho"],
        P=spectral["P"],
        Q=spectral["Q"],
        A=spectral["A"],
        psi6_sq=spectral["psi6_sq"],
        J_density=spectral["J_density"],
        J_rho=spectral["J_rho"],
        J_P=spectral["J_P"],
        J_Q=spectral["J_Q"],
        Y_rho=spectral["Y_rho"],
        Y_P=spectral["Y_P"],
        Y_Q=spectral["Y_Q"],
        F_rho_prediction=arrays["F_rho_prediction"],
        partial_t_rho=spectral["partial_t_rho"],
        temporal_power=spectral["temporal_power"],
        cheb_times=spectral["cheb_times"],
        cheb_scaled_times=spectral["cheb_scaled_times"],
        sample_indices=arrays["sample_indices"],
        Y_rho_rows=arrays["Y_rho_rows"],
        Y_P_rows=arrays["Y_P_rows"],
        Y_Q_rows=arrays["Y_Q_rows"],
        Y_rho_row_index=arrays["Y_rho_row_index"],
        Y_P_row_index=arrays["Y_P_row_index"],
        Y_Q_row_index=arrays["Y_Q_row_index"],
        Y_rho_library=arrays["Y_rho_library"],
        Y_P_library=arrays["Y_P_library"],
        Y_Q_library=arrays["Y_Q_library"],
        Y_rho_names=arrays["Y_rho_names"],
        Y_P_names=arrays["Y_P_names"],
        Y_Q_names=arrays["Y_Q_names"],
        Y_rho_coefficients=y_rho_fit.coefficients,
        Y_P_coefficients=y_p_fit.coefficients,
        Y_Q_coefficients=y_q_fit.coefficients,
        Y_rho_importance=y_rho_fit.importance,
        Y_P_importance=y_p_fit.importance,
        Y_Q_importance=y_q_fit.importance,
        Y_rho_importance_path=y_rho_fit.importance_path,
        Y_P_importance_path=y_p_fit.importance_path,
        Y_Q_importance_path=y_q_fit.importance_path,
        Y_rho_tau_values=y_rho_fit.tau_values,
        Y_P_tau_values=y_p_fit.tau_values,
        Y_Q_tau_values=y_q_fit.tau_values,
        Y_rho_tau_index=np.asarray(-1 if y_rho_fit.tau_index is None else y_rho_fit.tau_index),
        Y_P_tau_index=np.asarray(-1 if y_p_fit.tau_index is None else y_p_fit.tau_index),
        Y_Q_tau_index=np.asarray(-1 if y_q_fit.tau_index is None else y_q_fit.tau_index),
        Y_rho_active=y_rho_fit.active,
        Y_P_active=y_p_fit.active,
        Y_Q_active=y_q_fit.active,
        Y_rho_rmse=np.asarray(y_rho_fit.rmse),
        Y_P_rmse=np.asarray(y_p_fit.rmse),
        Y_Q_rmse=np.asarray(y_q_fit.rmse),
        Y_rho_r2=np.asarray(y_rho_fit.r2),
        Y_P_r2=np.asarray(y_p_fit.r2),
        Y_Q_r2=np.asarray(y_q_fit.r2),
        Y_rho_flux_rmse=np.asarray(np.nan if y_rho_fit.auxiliary_rmse is None else y_rho_fit.auxiliary_rmse),
        Y_P_flux_rmse=np.asarray(np.nan if y_p_fit.auxiliary_rmse is None else y_p_fit.auxiliary_rmse),
        Y_Q_flux_rmse=np.asarray(np.nan if y_q_fit.auxiliary_rmse is None else y_q_fit.auxiliary_rmse),
        Y_rho_flux_r2=np.asarray(np.nan if y_rho_fit.auxiliary_r2 is None else y_rho_fit.auxiliary_r2),
        Y_P_flux_r2=np.asarray(np.nan if y_p_fit.auxiliary_r2 is None else y_p_fit.auxiliary_r2),
        Y_Q_flux_r2=np.asarray(np.nan if y_q_fit.auxiliary_r2 is None else y_q_fit.auxiliary_r2),
    )
    write_report(
        report_path,
        mechanical_report_lines(
            case_id=config.case_id,
            nd=arrays["sample_indices"].shape[0],
            frames=active.coords.shape[0],
            grid_shape=(active.x_centers.size, active.theta_centers.size, coarse["r_centers"].size),
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
    grid_shape: tuple[int, int, int],
    sigma: float,
    cheb_cutoff: int,
    fits: dict[str, StabilityResult],
) -> list[str]:
    """Build markdown report lines for divergence-first mechanical closure fits."""
    lines = [
        f"# Rho fitting report: {case_id}",
        "",
        "## Settings",
        f"- frames: {frames}",
        f"- grid: {grid_shape[0]} x {grid_shape[1]} x {grid_shape[2]}",
        f"- samples: {nd}",
        f"- sigma: {sigma:.8g}",
        f"- cheb_cutoff: {cheb_cutoff}",
        "- analysis: divergence-first global cylindrical 3D mechanical moment fits",
        "",
    ]
    for target, fit in fits.items():
        lines.extend(
            [
                f"## {target}",
                f"- divergence rmse: {fit.rmse:.8g}",
                f"- divergence r2: {fit.r2:.8g}",
                f"- flux rmse: {_optional_float(fit.auxiliary_rmse)}",
                f"- flux r2: {_optional_float(fit.auxiliary_r2)}",
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
                f"| {_markdown_cell(label)} | {int(active)} | {coefficient:.10g} | "
                f"{importance:.4f} | {format_float(raw_correlation)} |"
            )
        lines.append("")
        if fit.tau_values.size:
            lines.extend(
                [
                    f"### {target} tau importance path",
                    "",
                    "| tau index | tau | " + " | ".join(_markdown_cell(label) for label in fit.labels) + " |",
                    "|---:|---:|" + "---:|" * len(fit.labels),
                ]
            )
            for tau_index, tau in enumerate(fit.tau_values):
                values = " | ".join(f"{value:.4f}" for value in fit.importance_path[tau_index])
                lines.append(f"| {tau_index} | {tau:.6g} | {values} |")
            lines.append("")
    return lines


def cache_metadata(active: ActiveMatterArrays, config: RhoFittingConfig) -> dict[str, object]:
    """Collect run metadata stored alongside rho-fitting NPZ outputs."""
    assert config.settings is not None, "rho fitting settings were not initialized"
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    if config.settings.radial_range is None:
        finite_r = active.coords[..., 2][np.isfinite(active.coords[..., 2])]
        r_min = float(np.min(finite_r))
        r_max = float(np.max(finite_r))
    else:
        r_min, r_max = config.settings.radial_range
    r_edges = np.linspace(r_min, r_max, config.settings.radial_bins + 1, dtype=float)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    return {
        "case_id": config.case_id,
        "coordinate_system": "cylindrical_3d",
        "active_fields_path": str(config.paths.active_fields_path),
        "active_fields_mtime": config.paths.active_fields_path.stat().st_mtime,
        "Nx": int(active.x_centers.size),
        "Ntheta": int(active.theta_centers.size),
        "Nr": int(r_centers.size),
        "sigma": float(config.settings.sigma),
        "lx": float(lx),
        "ly": float(ly),
        "theta_period": float(active.theta_edges[-1] - active.theta_edges[0]),
        "r_edges": [float(value) for value in r_edges],
        "r_centers": [float(value) for value in r_centers],
        "radius": float(active.radius),
        "cheb_cutoff": int(config.settings.cheb_cutoff),
        "nd": int(config.settings.nd),
        "seed": int(config.settings.seed),
        "replace": bool(config.settings.replace),
        "tau_count": int(config.settings.tau_count),
        "tau_eps": float(config.settings.tau_eps),
        "mechanical_flux_weight": float(config.settings.mechanical_flux_weight),
        "analysis": "global_density_flux_divergence",
    }


def format_float(value: float) -> str:
    """Format finite floats for compact report tables."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _markdown_cell(value: str) -> str:
    """Escape markdown table separators inside a cell value."""
    return value.replace("|", "\\|")


def _optional_float(value: float | None) -> str:
    """Format an optional float for report text."""
    if value is None:
        return "none"
    return f"{value:.8g}"
