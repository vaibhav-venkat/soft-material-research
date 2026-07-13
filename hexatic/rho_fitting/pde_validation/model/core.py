"""Rust IMEX rollout for fitted cylindrical moment closures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hexatic.rho_fitting import _rho_fitting_core, _rho_fitting_core_import_error
from hexatic.rho_fitting.constants import DEFAULT_PDE_DT_MAX
from ..cache import ValidationInputs, load_validation_inputs
from .types import Array, ValidationOptions, ValidationResult


def run_validation(inputs: ValidationInputs, options: ValidationOptions | None = None) -> ValidationResult:
    """Run the dealiased Rust IMEX rollout and return cache-grid results."""
    options = ValidationOptions() if options is None else options
    frames = inputs.rho.shape[0] if options.max_frames is None else min(inputs.rho.shape[0], options.max_frames)
    assert frames >= 2
    times = inputs.times[:frames]
    assert options.solver == "rk4", f"unsupported PDE solver: {options.solver}"
    dt_max = DEFAULT_PDE_DT_MAX if options.dt is None else float(options.dt)
    assert dt_max > 0.0
    if _rho_fitting_core is None:
        raise ImportError(f"rho-fitting Rust core is unavailable: {_rho_fitting_core_import_error}")
    mode = {"full": 0, "rho-only": 1, "p-only": 2, "q-only": 3}[options.mode]
    ubar_source = {"cached": 0, "fitted": 1}[options.ubar_source]
    payload = _rho_fitting_core.run_pde_validation(
        np.ascontiguousarray(inputs.rho[:frames], dtype=np.float64),
        np.ascontiguousarray(inputs.p[:frames], dtype=np.float64),
        np.ascontiguousarray(inputs.q[:frames], dtype=np.float64),
        np.ascontiguousarray(inputs.psi6_sq[:frames], dtype=np.float64),
        np.ascontiguousarray(inputs.y_p[:frames], dtype=np.float64),
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(inputs.y_rho_coefficients, dtype=np.float64),
        np.ascontiguousarray(inputs.y_p_coefficients, dtype=np.float64),
        np.ascontiguousarray(inputs.y_q_coefficients, dtype=np.float64),
        np.ascontiguousarray(inputs.r_centers, dtype=np.float64),
        np.ascontiguousarray(inputs.r_edges, dtype=np.float64),
        float(inputs.lx),
        float(inputs.theta_period),
        float(inputs.u0),
        float(inputs.gamma),
        int(frames),
        float(dt_max),
        int(mode),
        int(ubar_source),
    )
    rho_fit = np.asarray(payload["rho"])
    p_fit = np.asarray(payload["P"])
    q_fit = np.asarray(payload["Q"])
    return _result(options.mode, rho_fit, p_fit, q_fit, inputs, times)


def _result(mode: str, rho_fit: Array, p_fit: Array, q_fit: Array, inputs: ValidationInputs, times: Array) -> ValidationResult:
    rho_true, p_true, q_true = inputs.rho[:times.size], inputs.p[:times.size], inputs.q[:times.size]
    fit, truth, axes = validation_metric_arrays(mode, rho_fit, p_fit, q_fit, rho_true, p_true, q_true)
    residual = fit - truth
    rmse = np.sqrt(np.mean(residual * residual, axis=axes))
    centered = truth - np.mean(truth, axis=axes, keepdims=True)
    r2 = 1.0 - np.divide(np.sum(residual * residual, axis=axes), np.sum(centered * centered, axis=axes), out=np.full(times.size, np.nan), where=np.sum(centered * centered, axis=axes) > 0.0)
    return ValidationResult(rho_fit, p_fit, q_fit, rho_true, p_true, q_true, times, rmse, r2)


def validation_metric_arrays(mode: str, rho_fit: Array, p_fit: Array, q_fit: Array, rho_true: Array, p_true: Array, q_true: Array) -> tuple[Array, Array, tuple[int, ...]]:
    if mode in {"full", "rho-only"}: return rho_fit, rho_true, (1, 2, 3)
    if mode == "p-only": return p_fit, p_true, (1, 2, 3, 4)
    if mode == "q-only": return q_fit, q_true, (1, 2, 3, 4, 5)
    raise AssertionError(f"unknown validation mode: {mode}")


def run_validation_from_cache(cache_path: Path, options: ValidationOptions | None = None) -> tuple[ValidationInputs, ValidationResult]:
    inputs = load_validation_inputs(cache_path)
    return inputs, run_validation(inputs, options)
