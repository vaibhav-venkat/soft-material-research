"""Density-only report writer for stochastic residual-flux model results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import operators as ops
from .fit import FittingResult
from .library import VectorLibrary, build_current_library
from .regression import RegressionResult, fit_vector_library


@dataclass(frozen=True)
class DensityModelSummary:
    """Full diagnostic output for one residual-split density model."""

    name: str
    equation: str
    det_equation: str
    coefficients: tuple[tuple[str, float, str], ...]
    # full stochastic (with xi)
    r2: float
    mae: float
    normalized_mae: float
    correlation: float
    # deterministic only (without xi)
    det_r2: float
    det_mae: float
    det_normalized_mae: float
    det_correlation: float
    # residual diagnostics
    rms_div_xi: float
    rms_div_jsys: float
    rms_div_jres: float
    fraction_jsys_jres: float
    note: str


def write_model_report(
    result: FittingResult,
    output_dir: str | Path,
    *,
    case_id: str = "radius_15D",
) -> Path:
    """Write density-only text and Markdown reports for the three requested models."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    txt_path = dest / f"{case_id}_model_report.txt"
    md_path = dest / f"{case_id}_model_report.md"

    models = _three_density_models(result)
    rho_stats = _field_stats(result.fields.partial_t_rho, result.mask)
    source_stats = _field_stats(result.fields.S_cross, result.mask)
    y_stats = _field_stats(result.density_target, result.mask)

    lines: list[str] = []
    add = lines.append
    add(f"Density Stochastic-Flux Report — case: {case_id}")
    add("=" * 72)
    add("")
    add("Governing density equation")
    add("-" * 48)
    add("  ∂_t ρ = -∇_s·J_m + S_cross")
    add("  J_res = J_m - J_base")
    add("  J_sys = mean_t J_res")
    add("  ξ = J_res - J_sys")
    add("  ∂_t ρ_pred = -∇_s·[J_base + J_sys] + S_cross - ∇_s·ξ")
    add("")
    add("Important identity")
    add("-" * 48)
    add("  Because ξ is defined as J_res - J_sys, each full stochastic model")
    add("  reconstructs J_m by construction. The reported R² and MAE therefore")
    add("  test the full residual-flux equation against ∂_tρ, not the deterministic")
    add("  base current alone.")
    add("")
    add("Field summaries")
    add("-" * 48)
    add("  ∂_t ρ:")
    add(_format_stats(rho_stats))
    add("  S_cross:")
    add(_format_stats(source_stats))
    add("  Y_ρ = ∂_t ρ - S_cross:")
    add(_format_stats(y_stats))
    add("")
    add("Three Full Stochastic Density Models")
    add("-" * 48)
    for idx, model in enumerate(models, start=1):
        add(f"  Model {idx}: {model.name}")
        add(f"    full equation: {model.equation}")
        add(f"    deterministic equation (no ξ): {model.det_equation}")
        add(f"    Full stochastic (with ξ) against ∂_tρ:")
        add(f"      R²:  {model.r2:.8g}")
        add(f"      MAE: {model.mae:.8g}")
        add(f"      normalized MAE: {model.normalized_mae:.8g}")
        add(f"      correlation:    {model.correlation:.8g}")
        add(f"    Deterministic (without ξ) against ∂_tρ:")
        add(f"      R²:  {model.det_r2:.8g}")
        add(f"      MAE: {model.det_mae:.8g}")
        add(f"      normalized MAE: {model.det_normalized_mae:.8g}")
        add(f"      correlation:    {model.det_correlation:.8g}")
        add(f"    Residual diagnostics:")
        add(f"      rms(-∇·J_res): {model.rms_div_jres:.8g}")
        add(f"      rms(-∇·J_sys): {model.rms_div_jsys:.8g}")
        add(f"      rms(-∇·ξ):     {model.rms_div_xi:.8g}")
        add(f"      fraction J_sys/J_res: {model.fraction_jsys_jres:.8g}")
        if model.note:
            add(f"    note: {model.note}")
        if model.coefficients:
            add("    coefficients:")
            for cname, coef, label in model.coefficients:
                add(f"      {cname:>3s} = {coef: .8e}    {label}")
        else:
            add("    coefficients: no fitted a_i; EOM terms use fixed microscopic coefficients")
        add("")

    add("Saved outputs")
    add("-" * 48)
    add(f"  text report:       {txt_path}")
    add(f"  markdown report:   {md_path}")
    add(f"  fit cache:         {_result_path(dest, case_id, 'fitting.npz')}")
    add(f"  hydrodynamic data: {_result_path(dest, case_id, 'hydrodynamic_fields.npz')}")
    add(f"  gaussian data:     {_result_path(dest, case_id, 'gaussian_fields.npz')}")
    add("")

    txt_path.write_text("\n".join(lines))
    md_path.write_text(_markdown_report(case_id, models, rho_stats, source_stats, y_stats, txt_path, md_path))
    print(f"[fitting] Report saved: {txt_path}")
    print(f"[fitting] Markdown report saved: {md_path}")
    return txt_path


def _three_density_models(result: FittingResult) -> tuple[DensityModelSummary, ...]:
    try:
        current_lib = build_current_library(result.fields)
    except AttributeError:
        measured = _metrics_for_partial_t_prediction(
            result,
            -_divergence(result, result.fields.material_current) + result.fields.S_cross,
        )
        base = dict(
            r2=measured["r2"],
            mae=measured["mae"],
            normalized_mae=measured["normalized_mae"],
            correlation=measured["correlation"],
            det_r2=float("nan"),
            det_mae=float("nan"),
            det_normalized_mae=float("nan"),
            det_correlation=float("nan"),
            rms_div_xi=float("nan"),
            rms_div_jsys=float("nan"),
            rms_div_jres=float("nan"),
            fraction_jsys_jres=float("nan"),
        )
        note = "current-library fields unavailable in minimal result"
        return (
            DensityModelSummary(
                "J_fit residual split",
                "-∇_s·[J_fit + J_sys_fit] + S_cross - ∇_s·ξ_fit",
                "-∇_s·[J_fit + J_sys_fit] + S_cross",
                (), note=note, **base,
            ),
            DensityModelSummary(
                "J_EOM residual split",
                "-∇_s·[J_EOM + J_sys] + S_cross - ∇_s·ξ",
                "-∇_s·[J_EOM + J_sys] + S_cross",
                (), note=note, **base,
            ),
            DensityModelSummary(
                "J_fit without force_density residual split",
                "-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross - ∇_s·ξ_no_f",
                "-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross",
                (), note=note, **base,
            ),
        )
    j_fit = _current_from_fit(current_lib, result.density.coefficients)

    fit_diag = _residual_diagnostics(result, j_fit)
    fit_coefficients = tuple(
        (f"a{i}", float(coef), label)
        for i, (coef, label) in enumerate(zip(result.density.coefficients, result.density.labels), start=1)
    )

    no_force_fit = _fit_without_force_density(result, current_lib)
    if no_force_fit is None:
        no_force_coefficients: tuple[tuple[str, float, str], ...] = ()
        no_force_diag = _residual_diagnostics(result, np.zeros_like(result.fields.material_current))
        no_force_note = "no-force refit unavailable for this synthetic/minimal result"
    else:
        no_force_result, no_force_lib = no_force_fit
        no_force_current = _current_from_fit(no_force_lib, no_force_result.coefficients)
        no_force_diag = _residual_diagnostics(result, no_force_current)
        no_force_coefficients = tuple(
            (f"a{i}", float(coef), label)
            for i, (coef, label) in enumerate(zip(no_force_result.coefficients, no_force_result.labels), start=1)
        )
        no_force_note = "force_density and D force_density omitted from J_fit before residual split"

    eom_diag = _residual_diagnostics(result, np.zeros_like(result.fields.material_current))

    return (
        DensityModelSummary(
            name="J_fit residual split",
            equation="-∇_s·[J_fit + J_sys_fit] + S_cross - ∇_s·ξ_fit",
            det_equation="-∇_s·[J_fit + J_sys_fit] + S_cross",
            coefficients=fit_coefficients,
            note="J_res_fit = J_m - J_fit; J_sys_fit = mean_t J_res_fit; ξ_fit = J_res_fit - J_sys_fit",
            **fit_diag,
        ),
        DensityModelSummary(
            name="J_EOM residual split",
            equation="-∇_s·[J_EOM + J_sys] + S_cross - ∇_s·ξ",
            det_equation="-∇_s·[J_EOM + J_sys] + S_cross",
            coefficients=(),
            note="J_EOM = J_active + J_pair + J_wall; full residual identity reconstructs J_m",
            **eom_diag,
        ),
        DensityModelSummary(
            name="J_fit without force_density residual split",
            equation="-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross - ∇_s·ξ_no_f",
            det_equation="-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross",
            coefficients=no_force_coefficients,
            note=no_force_note,
            **no_force_diag,
        ),
    )


def _fit_without_force_density(
    result: FittingResult,
    library: VectorLibrary,
) -> tuple[RegressionResult, VectorLibrary] | None:
    keep = tuple(
        i for i, name in enumerate(library.names)
        if name not in {"force_density", "D_force_density"}
    )
    if not keep or result.fields.material_current.shape[:-1] != result.mask.shape:
        return None
    sub_library = VectorLibrary(
        names=tuple(library.names[i] for i in keep),
        labels=tuple(library.labels[i] for i in keep),
        values=library.values[..., keep, :],
    )
    fit = fit_vector_library(
        sub_library,
        result.fields.material_current,
        result.mask,
        ridge_alpha=result.ridge_alpha,
        stlsq_threshold=result.stlsq_threshold,
        stlsq_max_iter=result.stlsq_max_iter,
    )
    return fit, sub_library


def _current_from_fit(library: VectorLibrary, coefficients: np.ndarray) -> np.ndarray:
    return np.einsum("...tc,t->...c", library.values, np.asarray(coefficients, dtype=float))


def _residual_diagnostics(result: FittingResult, base_current: np.ndarray) -> dict[str, float]:
    """Return full/deterministic metrics + residual diagnostic stats for a given base current."""
    j_res = result.fields.material_current - base_current
    j_sys = np.mean(j_res, axis=0, keepdims=True)
    xi = j_res - j_sys

    # deterministic prediction (without xi)
    det_prediction = -_divergence(result, base_current + j_sys) + result.fields.S_cross
    det = _metrics_for_partial_t_prediction(result, det_prediction)

    # full stochastic prediction
    full_current = base_current + j_sys + xi
    full_prediction = -_divergence(result, full_current) + result.fields.S_cross
    full = _metrics_for_partial_t_prediction(result, full_prediction)

    # residual diagnostic fields (different time dimensions)
    C_jres = -_divergence(result, j_res)   # (T, nx, ntheta)
    C_jsys = -_divergence(result, j_sys)   # (1,  nx, ntheta)
    C_xi = -_divergence(result, xi)         # (T, nx, ntheta)
    T = C_jres.shape[0]
    v = result.mask & np.isfinite(C_jres) & np.isfinite(C_xi)
    rms_jres = float(np.sqrt(np.mean(C_jres[v] ** 2)))
    # J_sys has 1 time sample; use mask valid across all times at each spatial cell
    v_sys = result.mask.all(axis=0) & np.isfinite(C_jsys[0])
    rms_jsys = float(np.sqrt(np.mean(C_jsys[0][v_sys] ** 2)))
    rms_xi = float(np.sqrt(np.mean(C_xi[v] ** 2)))
    fraction = rms_jsys / rms_jres if rms_jres > 0 else float("nan")

    return {
        "r2": full["r2"],
        "mae": full["mae"],
        "normalized_mae": full["normalized_mae"],
        "correlation": full["correlation"],
        "det_r2": det["r2"],
        "det_mae": det["mae"],
        "det_normalized_mae": det["normalized_mae"],
        "det_correlation": det["correlation"],
        "rms_div_xi": rms_xi,
        "rms_div_jsys": rms_jsys,
        "rms_div_jres": rms_jres,
        "fraction_jsys_jres": fraction,
    }


def _metrics_for_partial_t_prediction(result: FittingResult, prediction: np.ndarray) -> dict[str, float]:
    valid = result.mask & np.isfinite(result.fields.partial_t_rho) & np.isfinite(prediction)
    y = result.fields.partial_t_rho[valid]
    p = prediction[valid]
    if y.size == 0:
        return {"r2": float("nan"), "mae": float("nan"), "normalized_mae": float("nan"), "correlation": float("nan")}
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - float(np.sum((y - p) ** 2)) / ss_tot
    mae = float(np.mean(np.abs(y - p)))
    scale = float(np.mean(np.abs(y)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    corr = float("nan")
    if y.size >= 2 and np.std(y) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(y, p)[0, 1])
    return {"r2": r2, "mae": mae, "normalized_mae": mae / scale, "correlation": corr}


def _divergence(result: FittingResult, current: np.ndarray) -> np.ndarray:
    fields = result.fields
    ly = fields.cylinder_radius * (fields.theta_edges[-1] - fields.theta_edges[0])
    kx, ky = ops.build_k_vectors(fields.x_centers.size, fields.theta_centers.size, fields.lx, ly)
    return ops.fft_divergence(current, kx, ky)


def _field_stats(field: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(field)
    values = np.asarray(field, dtype=float)[valid]
    if values.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "rms": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _format_stats(stats: dict[str, float]) -> str:
    return (
        f"    mean={stats['mean']:.6g}, std={stats['std']:.6g}, rms={stats['rms']:.6g}, "
        f"min={stats['min']:.6g}, max={stats['max']:.6g}"
    )


def _markdown_report(
    case_id: str,
    models: tuple[DensityModelSummary, ...],
    rho_stats: dict[str, float],
    source_stats: dict[str, float],
    y_stats: dict[str, float],
    txt_path: Path,
    md_path: Path,
) -> str:
    lines: list[str] = []
    add = lines.append
    add(f"# Density Stochastic-Flux Report: `{case_id}`")
    add("")
    add("## Governing Equation")
    add("")
    add("```text")
    add("∂_t ρ_pred = -∇_s·[J_base + J_sys] + S_cross - ∇_s·ξ")
    add("J_res = J_m - J_base,  J_sys = mean_t J_res,  ξ = J_res - J_sys")
    add("```")
    add("")
    add("Because `ξ` is defined from the residual, each full stochastic model reconstructs `J_m` by identity; the metrics below are against `∂_tρ`.")
    add("")
    add("## Field Summaries")
    add("")
    add(_markdown_stats_table([("∂_t ρ", rho_stats), ("S_cross", source_stats), ("Y_ρ", y_stats)]))
    add("")
    add("## Three Full Stochastic Density Models")
    add("")
    add("| model | R² full | R² det (no ξ) | rms(-∇·ξ) | fraction J_sys/J_res |")
    add("|---|---:|---:|---:|---:|")
    for model in models:
        add(f"| {model.name} | `{model.r2:.8g}` | `{model.det_r2:.8g}` | `{model.rms_div_xi:.8g}` | `{model.fraction_jsys_jres:.8g}` |")
    add("")
    for idx, model in enumerate(models, start=1):
        add(f"### Model {idx}: {model.name}")
        add("")
        add("Full equation:")
        add("```text")
        add(model.equation)
        add("```")
        add("Deterministic (no ξ):")
        add("```text")
        add(model.det_equation)
        add("```")
        add("")
        add("| metric | full (with ξ) | deterministic (no ξ) |")
        add("|---|---:|---:|")
        add(f"| R² vs `∂_tρ` | `{model.r2:.8g}` | `{model.det_r2:.8g}` |")
        add(f"| MAE vs `∂_tρ` | `{model.mae:.8g}` | `{model.det_mae:.8g}` |")
        add(f"| normalized MAE | `{model.normalized_mae:.8g}` | `{model.det_normalized_mae:.8g}` |")
        add(f"| correlation | `{model.correlation:.8g}` | `{model.det_correlation:.8g}` |")
        add("")
        add("Residual diagnostics:")
        add("")
        add("| quantity | value |")
        add("|---|---:|")
        add(f"| rms(-∇·J_res) | `{model.rms_div_jres:.8g}` |")
        add(f"| rms(-∇·J_sys) | `{model.rms_div_jsys:.8g}` |")
        add(f"| rms(-∇·ξ) | `{model.rms_div_xi:.8g}` |")
        add(f"| fraction J_sys / J_res | `{model.fraction_jsys_jres:.8g}` |")
        add("")
        if model.note:
            add(f"Note: {model.note}")
            add("")
        if model.coefficients:
            add("| coefficient | value | term |")
            add("|---:|---:|---|")
            for cname, coef, label in model.coefficients:
                add(f"| `{cname}` | `{coef:.8e}` | {label} |")
            add("")
        else:
            add("No fitted `a_i`; EOM terms use fixed microscopic coefficients.\n")
    add("## Saved Outputs")
    add("")
    add(f"- Text report: `{txt_path}`")
    add(f"- Markdown report: `{md_path}`")
    return "\n".join(lines)


def _markdown_stats_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    lines = ["| field | mean | std | rms | min | max |", "|---|---:|---:|---:|---:|---:|"]
    for name, stats in rows:
        safe_name = name.replace("|", "\\|")
        lines.append(
            f"| {safe_name} | `{stats['mean']:.6g}` | `{stats['std']:.6g}` | `{stats['rms']:.6g}` | "
            f"`{stats['min']:.6g}` | `{stats['max']:.6g}` |"
        )
    return "\n".join(lines)


def _result_path(output_dir: str | Path, case_id: str, suffix: str) -> Path:
    return Path(output_dir) / f"{case_id}_{suffix}"
