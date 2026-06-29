"""Density-only report writer for stochastic residual-flux model results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import operators as ops
from .fit import FittingResult
from .library import VectorLibrary, build_current_library, build_no_force_low_k_library
from .regression import RegressionResult, fit_vector_library
from .stochastic import (
    StochasticMechanismSummary,
    compute_stochastic_mechanism,
    markdown_report_lines as stochastic_markdown_report_lines,
    text_report_lines as stochastic_text_report_lines,
)


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
    current_r2: float
    current_r2_x: float
    current_r2_y: float
    stochastic: StochasticMechanismSummary | None
    source_stochastic: StochasticMechanismSummary | None
    note: str


def write_model_report(
    result: FittingResult,
    output_dir: str | Path,
    *,
    case_id: str = "radius_15D",
    drop_no_force_low_k_terms: tuple[str, ...] = (),
) -> Path:
    """Write density-only text and Markdown reports for the three requested models."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    txt_path = dest / f"{case_id}_model_report.txt"
    md_path = dest / f"{case_id}_model_report.md"

    models = _three_density_models(
        result,
        drop_no_force_low_k_terms=drop_no_force_low_k_terms,
    )
    rho_stats = _field_stats(result.fields.partial_t_rho, result.mask)
    source_stats = _field_stats(result.fields.S_cross, result.mask)
    source_pred_stats = _field_stats(result.source.prediction, result.mask)
    y_stats = _field_stats(result.density_target, result.mask)

    lines: list[str] = []
    add = lines.append
    add(f"Density Stochastic-Flux Report — case: {case_id}")
    add("=" * 72)
    add("")
    model3 = models[2] if len(models) >= 3 else None
    if model3 is not None and model3.stochastic is not None:
        add("Headline Model 3 Result")
        add("-" * 48)
        add("  partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross + eta_AR1")
        add("  Metrics against actual partial_t rho:")
        add(f"    R²:  {model3.stochastic.r2:.8g}")
        add(f"    MAE: {model3.stochastic.mae:.8g}")
        add(f"    normalized MAE: {model3.stochastic.normalized_mae:.8g}")
        add(f"    correlation:    {model3.stochastic.correlation:.8g}")
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
    add("  S_cross_pred:")
    add(_format_stats(source_pred_stats))
    add("  Y_ρ = ∂_t ρ - S_cross:")
    add(_format_stats(y_stats))
    add("")
    add("Fitted S_cross source model")
    add("-" * 48)
    add(f"  equation: {_source_equation(result)}")
    add("  Metrics against actual S_cross:")
    add(f"    R²:  {result.source.metrics.get('r2', np.nan):.8g}")
    add(f"    MAE: {result.source.metrics.get('mae', np.nan):.8g}")
    add(f"    normalized MAE: {result.source.metrics.get('normalized_mae', np.nan):.8g}")
    add(f"    correlation:    {result.source.metrics.get('correlation', np.nan):.8g}")
    add("  coefficients:")
    for cname, coef, label in _source_coefficients(result):
        add(f"    {cname:>3s} = {coef: .8e}    {label}")
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
        if np.isfinite(model.current_r2):
            add("    Current fit diagnostics against J_m:")
            add(f"      R² combined: {model.current_r2:.8g}")
            add(f"      R² x:        {model.current_r2_x:.8g}")
            add(f"      R² theta:    {model.current_r2_y:.8g}")
        if model.stochastic is not None:
            lines.extend(stochastic_text_report_lines(model.stochastic))
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

    _plot_jsys(result, dest, case_id, models)

    txt_path.write_text("\n".join(lines))
    md_path.write_text(
        _markdown_report(
            case_id,
            models,
            result,
            rho_stats,
            source_stats,
            source_pred_stats,
            y_stats,
            txt_path,
            md_path,
        )
    )
    print(f"[fitting] Report saved: {txt_path}")
    print(f"[fitting] Markdown report saved: {md_path}")
    print(f"[fitting] J_sys plot saved: {dest / f'{case_id}_jsys.png'}")
    return txt_path


def _three_density_models(
    result: FittingResult,
    *,
    drop_no_force_low_k_terms: tuple[str, ...] = (),
) -> tuple[DensityModelSummary, ...]:
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
            current_r2=float("nan"),
            current_r2_x=float("nan"),
            current_r2_y=float("nan"),
            stochastic=None,
            source_stochastic=None,
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

    no_force_fit = _fit_without_force_density(
        result,
        current_lib,
        drop_low_k_terms=drop_no_force_low_k_terms,
    )
    if no_force_fit is None:
        no_force_coefficients: tuple[tuple[str, float, str], ...] = ()
        no_force_diag = _residual_diagnostics(result, np.zeros_like(result.fields.material_current))
        no_force_diag["source_stochastic"] = None
        no_force_note = "no-force refit unavailable for this synthetic/minimal result"
    else:
        no_force_result, no_force_lib = no_force_fit
        no_force_current = _current_from_fit(no_force_lib, no_force_result.coefficients)
        no_force_diag = _residual_diagnostics(
            result,
            no_force_current,
            stochastic_as_full=True,
        )
        no_force_diag["source_stochastic"] = None
        no_force_coefficients = tuple(
            (f"a{i}", float(coef), label)
            for i, (coef, label) in enumerate(zip(no_force_result.coefficients, no_force_result.labels), start=1)
        )
        no_force_note = (
            "force_density and D force_density omitted from J_fit before residual split"
        )

    eom_diag = _residual_diagnostics(result, np.zeros_like(result.fields.material_current))
    eom_diag.update(
        current_r2=float("nan"),
        current_r2_x=float("nan"),
        current_r2_y=float("nan"),
        source_stochastic=None,
    )
    fit_diag["source_stochastic"] = None

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
            name="J_fit without force_density residual split + 85% η-power AR(1)",
            equation="-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross + η_AR1",
            det_equation="-∇_s·[J_fit_no_f + J_sys_no_f] + S_cross",
            coefficients=no_force_coefficients,
            note=(
                no_force_note
                + "; final model uses seeded adaptive ξ_85 selected from empirical η power"
            ),
            **no_force_diag,
        ),
    )


def _fit_without_force_density(
    result: FittingResult,
    library: VectorLibrary,
    *,
    drop_low_k_terms: tuple[str, ...] = (),
) -> tuple[RegressionResult, VectorLibrary] | None:
    keep = tuple(
        i for i, name in enumerate(library.names)
        if name not in {"force_density", "D_force_density"}
    )
    if not keep or result.fields.material_current.shape[:-1] != result.mask.shape:
        return None
    names = tuple(library.names[i] for i in keep)
    labels = tuple(library.labels[i] for i in keep)
    values = library.values[..., keep, :]
    low_k_library = build_no_force_low_k_library(
        result.fields,
        drop_terms=drop_low_k_terms,
    )
    if low_k_library.values.shape[-2] > 0:
        names = names + low_k_library.names
        labels = labels + low_k_library.labels
        values = np.concatenate((values, low_k_library.values), axis=-2)
    sub_library = VectorLibrary(names=names, labels=labels, values=values)
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


def _residual_diagnostics(
    result: FittingResult,
    base_current: np.ndarray,
    *,
    source_prediction: np.ndarray | None = None,
    stochastic_as_full: bool = False,
) -> dict[str, float]:
    """Return full/deterministic metrics + residual diagnostic stats for a given base current."""
    j_res = result.fields.material_current - base_current
    j_sys = _masked_time_mean(j_res, result.mask)
    xi = j_res - j_sys
    current_metrics = _metrics_for_current_prediction(result, base_current)
    source = result.fields.S_cross if source_prediction is None else source_prediction

    # deterministic prediction (without xi)
    det_prediction = -_divergence(result, base_current + j_sys) + source
    det = _metrics_for_partial_t_prediction(result, det_prediction)

    # full stochastic prediction
    full_current = base_current + j_sys + xi
    full_prediction = -_divergence(result, full_current) + source
    full = _metrics_for_partial_t_prediction(result, full_prediction)
    stochastic = compute_stochastic_mechanism(
        result,
        base_current,
        source_prediction=source_prediction,
    )
    if stochastic_as_full and stochastic is not None:
        full = {
            "r2": stochastic.r2,
            "mae": stochastic.mae,
            "normalized_mae": stochastic.normalized_mae,
            "correlation": stochastic.correlation,
        }

    # residual diagnostic fields (different time dimensions)
    C_jres = -_divergence(result, j_res)   # (T, nx, ntheta)
    C_jsys = -_divergence(result, j_sys)   # (1,  nx, ntheta)
    C_xi = -_divergence(result, xi)         # (T, nx, ntheta)
    T = C_jres.shape[0]
    v = result.mask & np.isfinite(C_jres) & np.isfinite(C_xi)
    rms_jres = _rms_where(C_jres, v)
    v_sys = result.mask.any(axis=0) & np.isfinite(C_jsys[0])
    rms_jsys = _rms_where(C_jsys[0], v_sys)
    rms_xi = _rms_where(C_xi, v)
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
        "current_r2": current_metrics["r2"],
        "current_r2_x": current_metrics["r2_x"],
        "current_r2_y": current_metrics["r2_y"],
        "stochastic": stochastic,
}


def _masked_time_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    assert values.shape[:-1] == mask.shape
    weights = mask[..., None]
    count = np.sum(weights, axis=0, keepdims=True)
    total = np.sum(np.where(weights, values, 0.0), axis=0, keepdims=True)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def _rms_where(values: np.ndarray, valid: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    selected = values[valid]
    return float(np.sqrt(np.mean(selected * selected)))


def _metrics_for_current_prediction(result: FittingResult, prediction: np.ndarray) -> dict[str, float]:
    valid = (
        result.mask
        & np.all(np.isfinite(result.fields.material_current), axis=-1)
        & np.all(np.isfinite(prediction), axis=-1)
    )
    target = result.fields.material_current
    tx = target[..., 0][valid]
    ty = target[..., 1][valid]
    px = prediction[..., 0][valid]
    py = prediction[..., 1][valid]
    return {
        "r2": _r2(np.concatenate((tx, ty)), np.concatenate((px, py))),
        "r2_x": _r2(tx, px),
        "r2_y": _r2(ty, py),
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


def _source_coefficients(result: FittingResult) -> tuple[tuple[str, float, str], ...]:
    return tuple(
        (f"c{i}", float(coef), label)
        for i, (coef, label) in enumerate(
            zip(result.source.coefficients, result.source.labels),
        )
    )


def _source_equation(result: FittingResult) -> str:
    terms = [f"c{i} {label}" for i, label in enumerate(result.source.labels)]
    return "S_cross_pred = " + " + ".join(terms)


def _markdown_report(
    case_id: str,
    models: tuple[DensityModelSummary, ...],
    result: FittingResult,
    rho_stats: dict[str, float],
    source_stats: dict[str, float],
    source_pred_stats: dict[str, float],
    y_stats: dict[str, float],
    txt_path: Path,
    md_path: Path,
) -> str:
    lines: list[str] = []
    add = lines.append
    model3 = models[2] if len(models) >= 3 else None
    add(f"# Density Stochastic-Flux Report: `{case_id}`")
    add("")
    if model3 is not None and model3.stochastic is not None:
        add("## Headline Model 3 Result")
        add("")
        add("```text")
        add("partial_t rho_pred = -div(J_fit_no_force + J_sys_no_force) + S_cross + eta_AR1")
        add("```")
        add("")
        add("| metric vs `partial_t rho` | value |")
        add("|---|---:|")
        add(f"| R² | `{model3.stochastic.r2:.8g}` |")
        add(f"| MAE | `{model3.stochastic.mae:.8g}` |")
        add(f"| normalized MAE | `{model3.stochastic.normalized_mae:.8g}` |")
        add(f"| correlation | `{model3.stochastic.correlation:.8g}` |")
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
    add(_markdown_stats_table([
        ("∂_t ρ", rho_stats),
        ("S_cross", source_stats),
        ("S_cross_pred", source_pred_stats),
        ("Y_ρ", y_stats),
    ]))
    add("")
    add("## Fitted S_cross Source Model")
    add("")
    add("```text")
    add(_source_equation(result))
    add("```")
    add("")
    add("| metric vs actual `S_cross` | value |")
    add("|---|---:|")
    add(f"| R² | `{result.source.metrics.get('r2', np.nan):.8g}` |")
    add(f"| MAE | `{result.source.metrics.get('mae', np.nan):.8g}` |")
    add(f"| normalized MAE | `{result.source.metrics.get('normalized_mae', np.nan):.8g}` |")
    add(f"| correlation | `{result.source.metrics.get('correlation', np.nan):.8g}` |")
    add("")
    add("| coefficient | value | term |")
    add("|---:|---:|---|")
    for cname, coef, label in _source_coefficients(result):
        add(f"| `{cname}` | `{coef:.8e}` | {label} |")
    add("")
    add("## Three Full Stochastic Density Models")
    add("")
    add("| model | R² AR(1)/full | R² det (no ξ) | rms(-∇·ξ) | fraction J_sys/J_res |")
    add("|---|---:|---:|---:|---:|")
    for model in models:
        add(
            f"| {model.name} | `{model.r2:.8g}` | "
            f"`{model.det_r2:.8g}` | `{model.rms_div_xi:.8g}` | "
            f"`{model.fraction_jsys_jres:.8g}` |"
        )
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
        if np.isfinite(model.current_r2):
            add("Current fit against `J_m`:")
            add("")
            add("| quantity | R² |")
            add("|---|---:|")
            add(f"| combined components | `{model.current_r2:.8g}` |")
            add(f"| x component | `{model.current_r2_x:.8g}` |")
            add(f"| theta component | `{model.current_r2_y:.8g}` |")
            add("")
        if model.stochastic is not None:
            lines.extend(stochastic_markdown_report_lines(model.stochastic))
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


def _r2(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    finite = np.isfinite(target) & np.isfinite(prediction)
    target = target[finite]
    prediction = prediction[finite]
    if target.size == 0:
        return float("nan")
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((target - prediction) ** 2)) / ss_tot


def _plot_jsys(result: FittingResult, dest: Path, case_id: str, models: tuple[DensityModelSummary, ...]) -> Path | None:
    """Save a quiver plot of J_sys for each stochastic model."""
    # try/except for matplotlib backend issues in headless environments
    try:
        current_lib = build_current_library(result.fields)
    except (AttributeError, Exception):
        return None
    import matplotlib
    matplotlib.use("Agg")

    base_currents = []
    j_fit = _current_from_fit(current_lib, result.density.coefficients)
    base_currents.append(j_fit)
    base_currents.append(np.zeros_like(result.fields.material_current))  # EOM = J_m proxy

    no_force_fit = _fit_without_force_density(result, current_lib)
    if no_force_fit is not None:
        no_force_result, no_force_lib = no_force_fit
        base_currents.append(_current_from_fit(no_force_lib, no_force_result.coefficients))
    else:
        base_currents.append(np.zeros_like(result.fields.material_current))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
    titles = [
        "Model 1: J_fit residual\nJ_sys(x,θ)",
        "Model 2: J_EOM residual\nJ_sys(x,θ)",
        "Model 3: no force residual\nJ_sys(x,θ)",
    ]

    x_centers = result.fields.x_centers
    theta_centers = result.fields.theta_centers
    X, Theta = np.meshgrid(x_centers, theta_centers, indexing="ij")

    for idx, (base, ax, title) in enumerate(zip(base_currents, axes[0], titles)):
        j_res = result.fields.material_current - base
        j_sys = np.mean(j_res, axis=0)  # (nx, ntheta, 2)
        mag = np.sqrt(j_sys[..., 0] ** 2 + j_sys[..., 1] ** 2)

        # subsample for readability
        step = max(1, x_centers.size // 15)
        s = slice(None, None, step)

        vx = j_sys[s, s, 0]
        vy = j_sys[s, s, 1]
        vmag = mag[s, s]
        # per-panel adaptive scale: map 95th percentile magnitude to ~0.5 inches
        p95 = float(np.percentile(vmag[vmag > 0], 95)) if np.any(vmag > 0) else 1.0
        scale = max(p95 / 0.5, 1e-12)

        q = ax.quiver(
            X[s, s], Theta[s, s],
            vx, vy,
            vmag, cmap="viridis",
            scale=scale, scale_units="inches",
            width=0.005,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("θ")
        ax.set_title(title, fontsize=10)
        plt.colorbar(q, ax=ax, label="|J_sys|")

    fig.suptitle(f"Persistent residual current J_sys — case {case_id}", fontsize=12)
    fig.tight_layout()
    path = dest / f"{case_id}_jsys.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _result_path(output_dir: str | Path, case_id: str, suffix: str) -> Path:
    return Path(output_dir) / f"{case_id}_{suffix}"
