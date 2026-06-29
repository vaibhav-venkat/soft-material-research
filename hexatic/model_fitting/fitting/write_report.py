"""Text report writer for hydrodynamic model fitting results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import operators as ops
from .fit import FittingResult


def write_model_report(
    result: FittingResult,
    output_dir: str | Path,
    *,
    case_id: str = "radius_15D",
) -> Path:
    """Write a density/current-focused model report .txt."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{case_id}_model_report.txt"
    residual_diag = dest / f"{case_id}_residual_flux_split_diagnostics.npz"
    residual_summary = _load_residual_split_summary(residual_diag)

    lines: list[str] = []
    _add = lines.append
    _add(f"Density / Current Report  —  case: {case_id}")
    _add(f"{'=' * 72}")

    _add("\nEquation Summary")
    _add("-" * 48)
    _add("  Measured continuity:")
    _add("    ∂_t ρ = -∇·J_m + S_cross")
    _add("  Deterministic plus stochastic residual-flux form:")
    _add("    ∂_t ρ = -∇·[J_fit + J_sys] + S_cross - ∇·ξ")
    _add("  with")
    _add("    J_fit = a1 ρP + a2 χ(ρP)_perp + a3 f")
    _add("          + a4 DρP + a5 Dχ(ρP)_perp + a6 Df")
    _add("          + a7(-∇ρ) + a8(-∇|ψ6|) + a9(-∇D)")
    _add("    J_res = J_m - J_EOM,    J_res = J_sys + ξ")

    _add("\nConstants in J_fit")
    _add("-" * 48)
    coefficient_names = ("a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9")
    for cname, label, coef, active in zip(
        coefficient_names,
        result.density.labels,
        result.density.coefficients,
        result.density.active,
        strict=False,
    ):
        flag = " [active]" if active else ""
        _add(f"  {cname:>3s} = {coef: .8e}    multiplies {label}{flag}")

    _add("\nTerm Definitions")
    _add("-" * 48)
    _add("  ρ           Gaussian-smoothed surface number density.")
    _add("  P           local tangent polarization vector; ρP is polar material flux.")
    _add("  χ           chirality scalar.")
    _add("  V_perp      90-degree tangent rotation: (V_x, V_y)_perp = (-V_y, V_x).")
    _add("  f           Gaussian-smoothed tangent force-density field from particle data.")
    _add("  D           defect/disorder scalar, D = (6 - neighbor_count)^2.")
    _add("  |ψ6|        local hexatic order magnitude.")
    _add("  J_sys       persistent residual current, estimated in diagnostics as mean_t J_res(x,y,t).")
    _add("  ξ           colored stochastic/unresolved current after removing J_sys.")

    _add("\nHow J_m and S_cross Are Calculated")
    _add("-" * 48)
    _add("  J_m uses the same particles, shell support, midpoint positions, and Gaussian kernel as ρ.")
    _add("  For particles inside the shell at both ends of a transition:")
    _add("    J_m(x,θ,t+1/2) = Σ_i K[x,θ; r_i(t+1/2)] (Δx_i/Δt, R Δθ_i/Δt)")
    _add("  where Δx and Δθ use minimum-image displacement on the periodic surface.")
    _add("  S_cross is the matched Gaussian entry/exit source:")
    _add("    S_cross = (Gaussian density entering shell - Gaussian density exiting shell) / Δt")

    _add("\nRegression Settings")
    _add("-" * 48)
    _add(f"  ridge_alpha              = {result.ridge_alpha}")
    _add(f"  stlsq_threshold          = {result.stlsq_threshold}")
    _add(f"  stlsq_max_iter           = {result.stlsq_max_iter}")
    _add(f"  coarse_grain             = {result.coarse_grain_transitions} transitions")
    _add(f"  transition dt after CG   = {result.dt}")
    _add(f"  pocket_radius            = {result.pocket_radius}")

    masked = int(np.count_nonzero(result.mask))
    total = result.mask.size
    _add("\nShared Mask")
    _add("-" * 48)
    _add(f"  {masked} / {total} valid space-time samples")

    rho_stats = _field_stats(result.fields.partial_t_rho, result.mask)
    target_stats = _field_stats(result.density_target, result.mask)
    source_stats = _field_stats(result.fields.S_cross, result.mask)
    _add("\nDensity Time-Derivative Summary")
    _add("-" * 48)
    _add("  Fields are on the coarse-grained transition grid used by the fit.")
    _add("  partial_t ρ statistics:")
    _add(_format_stats(rho_stats))
    _add("  S_cross statistics:")
    _add(_format_stats(source_stats))
    _add("  Y_ρ = partial_t ρ - S_cross statistics:")
    _add(_format_stats(target_stats))

    continuity = _measured_continuity_metrics(result)
    _add("\nMeasured Continuity Numbers")
    _add("-" * 48)
    _add("  Test: Y_ρ = ∂_t ρ − S_cross ≈ -∇·J_m")
    for key in ("correlation", "r2", "normalized_mae", "rms_error", "target_std", "prediction_std"):
        _add(f"  {key:>30s}  {continuity[key]:.6g}")

    _add("\nDeterministic Current-Closure Numbers")
    _add("-" * 48)
    _add("  Fit target: J_m. Density prediction: Y_ρ ≈ -∇·J_fit.")
    _add(f"  {'rows used':>30s}  {result.density.rows_used}")
    for label, key in (
        ("density_correlation", "correlation"),
        ("density_r2", "r2"),
        ("density_normalized_mae", "normalized_mae"),
    ):
        val = result.density.metrics.get(key, float("nan"))
        _add(f"  {label:>30s}  {val:.6g}")
    for key in (
        "current_r2_x",
        "current_r2_y",
        "current_normalized_mae_x",
        "current_normalized_mae_y",
    ):
        val = result.density.metrics.get(key, float("nan"))
        _add(f"  {key:>30s}  {val:.6g}")

    _add("\nResidual / Stochastic Flux Numbers")
    _add("-" * 48)
    if residual_summary is None:
        _add("  residual split diagnostics were not found; run the residual audit to populate this section.")
        _add(f"  expected file: {residual_diag}")
    else:
        _add(f"  residual diagnostic window = {residual_summary['window']} transitions")
        _add("  Residual audit uses particle-level midpoint EOM current:")
        _add("    J_EOM = J_active + J_pair + J_wall")
        _add("    J_res = J_m - J_EOM")
        _add("    J_sys(x,y) = mean_t J_res(x,y,t)")
        _add("    ξ = J_res - J_sys")
        _add("  Numerical value of J_sys in this report is represented by its density-relevant divergence:")
        _add("    -∇·J_sys, because only current divergence enters the density equation.")
        _add(f"  {'R²(-div J_m, Yρ)':>30s}  {residual_summary['measured_R2']:.6g}")
        _add(f"  {'R²(-div J_EOM, Yρ)':>30s}  {residual_summary['eom_R2']:.6g}")
        _add(f"  {'R²(-div J_res, Y_res)':>30s}  {residual_summary['res_R2']:.6g}")
        _add(f"  {'R²(-div J_sys, Y_res)':>30s}  {residual_summary['sys_R2']:.6g}")
        _add(f"  {'corr(-div J_sys, Y_res)':>30s}  {residual_summary['sys_corr']:.6g}")
        _add(f"  {'R²(-div(J_EOM+J_sys), Yρ)':>30s}  {residual_summary['eom_plus_sys_R2']:.6g}")
        _add(f"  {'rms(-div J_sys)':>30s}  {residual_summary['rms_div_jsys']:.6g}")
        _add(f"  {'rms(-div J_res)':>30s}  {residual_summary['rms_div_jres']:.6g}")
        _add(f"  {'rms(-div ξ)':>30s}  {residual_summary['rms_div_xi']:.6g}")
        _add(f"  {'rms reduction':>30s}  {residual_summary['rms_reduction']:.6g}")
        _add(f"  {'residual diag file':>30s}  {residual_diag}")

    _add("\nInterpretation")
    _add("-" * 48)
    _add("  1. The measured continuity equation with J_m is accurate.")
    _add("  2. The deterministic current library only partially reproduces J_m.")
    _add("  3. The missing density dynamics are isolated as -∇·J_res.")
    _add("  4. The current best closed form is stochastic/hidden-variable, not purely deterministic.")

    _add("\nSaved Outputs")
    _add("-" * 48)
    _add(f"  {'fit cache':>30s}  {_result_path(output_dir, case_id, 'fitting.npz')}")
    _add(f"  {'hydrodynamic fields':>30s}  {_result_path(output_dir, case_id, 'hydrodynamic_fields.npz')}")
    _add(f"  {'gaussian fields':>30s}  {_result_path(output_dir, case_id, 'gaussian_fields.npz')}")
    _add(f"  {'text report':>30s}  {path}")
    if residual_diag.exists():
        _add(f"  {'residual split diag':>30s}  {residual_diag}")

    _add(f"\n{'=' * 72}\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[fitting] Report saved: {path}")
    return path


def _measured_continuity_metrics(result: FittingResult) -> dict[str, float]:
    fields = result.fields
    ly = fields.cylinder_radius * (fields.theta_edges[-1] - fields.theta_edges[0])
    kx, ky = ops.build_k_vectors(
        fields.x_centers.size,
        fields.theta_centers.size,
        fields.lx,
        ly,
    )
    prediction = -ops.fft_divergence(fields.material_current, kx, ky)
    return _scalar_metrics(result.density_target, prediction, result.mask)


def _field_stats(field: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(field)
    values = np.asarray(field, dtype=float)[valid]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _format_stats(stats: dict[str, float]) -> str:
    return (
        f"    mean={stats['mean']:.6g}, std={stats['std']:.6g}, "
        f"rms={stats['rms']:.6g}, min={stats['min']:.6g}, max={stats['max']:.6g}"
    )


def _load_residual_split_summary(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as arrays:
        metrics = arrays["metrics"].item()
        window = int(np.asarray(arrays["window"]))

    rms_div_jres = float(metrics["res_bookkeeping"]["rms"])
    rms_div_jsys = float(metrics["spatial_sys_to_yres"]["rms"])
    # The spatial-systematic split stores rms(Y_res - -div J_sys) as rms_err,
    # which is rms(-div xi) up to the small measured-continuity error.
    rms_div_xi = float(metrics["spatial_sys_to_yres"]["rms_err"])
    return {
        "window": window,
        "measured_R2": float(metrics["measured"]["R2"]),
        "eom_R2": float(metrics["eom"]["R2"]),
        "res_R2": float(metrics["res_bookkeeping"]["R2"]),
        "sys_R2": float(metrics["spatial_sys_to_yres"]["R2"]),
        "sys_corr": float(metrics["spatial_sys_to_yres"]["corr"]),
        "eom_plus_sys_R2": float(metrics["eom_plus_spatial_sys"]["R2"]),
        "rms_div_jsys": rms_div_jsys,
        "rms_div_jres": rms_div_jres,
        "rms_div_xi": rms_div_xi,
        "rms_reduction": 1.0 - rms_div_xi / rms_div_jres,
    }


def _scalar_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    y = target[valid]
    p = prediction[valid]
    if y.size == 0:
        return {
            "correlation": float("nan"),
            "r2": float("nan"),
            "normalized_mae": float("nan"),
            "rms_error": float("nan"),
            "target_std": float("nan"),
            "prediction_std": float("nan"),
        }
    corr = float("nan")
    if y.size >= 2 and np.std(y) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(y, p)[0, 1])
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - float(np.sum((y - p) ** 2)) / ss_tot
    scale = float(np.mean(np.abs(y)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return {
        "correlation": corr,
        "r2": r2,
        "normalized_mae": float(np.mean(np.abs(y - p)) / scale),
        "rms_error": float(np.sqrt(np.mean((y - p) ** 2))),
        "target_std": float(np.std(y)),
        "prediction_std": float(np.std(p)),
    }


def _result_path(output_dir: str | Path, case_id: str, suffix: str) -> Path:
    return Path(output_dir) / f"{case_id}_{suffix}"
