"""Shenfun IMEX rollout for fitted cylindrical moment closures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hexatic.rho_fitting.spectral import CylindricalSpectralOperators, barycentric_matrix, cached_cylindrical_operators, transfer_radial

from ..cache import ValidationInputs, load_validation_inputs
from ..operators import alignment_tensor, closure_fields, divergence_vector, estimate_ubar
from .interpolation import interpolated_cached_fields, interpolated_fields
from .types import P_RELAXATION_COEFFICIENT, Q_RELAXATION_COEFFICIENT, Array, ValidationOptions, ValidationResult


def run_validation(inputs: ValidationInputs, options: ValidationOptions | None = None) -> ValidationResult:
    """Run the dealiased Shenfun IMEX rollout and return cache-grid results."""
    options = ValidationOptions() if options is None else options
    frames = inputs.rho.shape[0] if options.max_frames is None else min(inputs.rho.shape[0], options.max_frames)
    assert frames >= 2
    times = inputs.times[:frames]
    shape = (int(inputs.rho.shape[1]), int(inputs.rho.shape[2]), int(inputs.rho.shape[3]))
    operators = cached_cylindrical_operators(
        inputs.lx,
        inputs.theta_period,
        float(inputs.r_edges[0]),
        float(inputs.r_edges[-1]),
        *shape,
    )
    to_spectral = barycentric_matrix(inputs.r_centers, operators.radial_nodes())
    to_cache = barycentric_matrix(operators.radial_nodes(), inputs.r_centers)
    rho = transfer_radial(inputs.rho[0], to_spectral, 2)
    p = transfer_radial(inputs.p[0], to_spectral, 2)
    q = transfer_radial(inputs.q[0], to_spectral, 2)
    psi6 = transfer_radial(inputs.psi6_sq[0], to_spectral, 2)
    rho_fit = np.empty((frames,) + shape, dtype=np.float64)
    p_fit = np.empty((frames,) + shape + (3,), dtype=np.float64)
    q_fit = np.empty((frames,) + shape + (3, 3), dtype=np.float64)
    rho_fit[0], p_fit[0], q_fit[0] = inputs.rho[0], inputs.p[0], inputs.q[0]
    dt_max = 5.0e-3 if options.dt is None else float(options.dt)
    assert dt_max > 0.0
    print(f"[rho_fitting.pde_validation] spectral rollout frames={frames} dt_max={dt_max:.6g}", flush=True)
    for frame in range(frames - 1):
        interval = float(times[frame + 1] - times[frame])
        steps = max(1, int(np.ceil(interval / dt_max)))
        dt = interval / steps
        print(f"[rho_fitting.pde_validation] frame={frame + 1}/{frames - 1} substeps={steps}", flush=True)
        for substep in range(steps):
            time = float(times[frame]) + substep * dt
            rho, p, q = _imex_step(inputs, operators, rho, p, q, psi6, time, dt, options.mode, options.ubar_source, to_spectral)
            rho = operators.filter_two_thirds(rho)
            p = operators.filter_two_thirds(p)
            q = operators.filter_two_thirds(q)
            assert np.all(np.isfinite(rho)) and np.all(np.isfinite(p)) and np.all(np.isfinite(q)), "spectral rollout became non-finite"
        rho_fit[frame + 1] = transfer_radial(rho, to_cache, 2)
        p_fit[frame + 1] = transfer_radial(p, to_cache, 2)
        q_fit[frame + 1] = transfer_radial(q, to_cache, 2)
        print(f"[rho_fitting.pde_validation] frame={frame + 1}/{frames - 1} complete", flush=True)
    return _result(options.mode, rho_fit, p_fit, q_fit, inputs, times)


def _imex_step(inputs: ValidationInputs, operators: CylindricalSpectralOperators, rho: Array, p: Array, q: Array, psi6: Array, time: float, dt: float, mode: str, ubar_source: str, to_spectral: Array) -> tuple[Array, Array, Array]:
    rho_ref, p_ref, q_ref, _a_ref, y_p_ref = interpolated_cached_fields(inputs, time)
    rho_ref, p_ref, q_ref, _a_ref, y_p_ref = (transfer_radial(value, to_spectral, 2) for value in (rho_ref, p_ref, q_ref, _a_ref, y_p_ref))
    a_ref = alignment_tensor(rho_ref, q_ref)
    rho_eval = rho if mode in {"full", "rho-only"} else rho_ref
    p_eval = p if mode in {"full", "p-only"} else p_ref
    q_eval = q if mode in {"full", "q-only"} else q_ref
    assert ubar_source in {"cached", "fitted"}
    ubar = estimate_ubar(y_p_ref, a_ref) if ubar_source == "cached" else None
    closures = closure_fields(rho_eval, p_eval, q_eval, psi6, inputs.y_rho_coefficients, inputs.y_p_coefficients, inputs.y_q_coefficients, operators, ubar)
    d_rho = -divergence_vector(inputs.u0 * p_eval + closures.f_rho / inputs.gamma, operators)
    d_p = -inputs.u0 * divergence_vector(closures.f_p, operators)
    d_q = -divergence_vector(closures.f_q, operators) + closures.s_q
    if mode in {"full", "rho-only"}:
        rho = rho + dt * d_rho
    else:
        rho = rho_ref
    if mode in {"full", "p-only"}:
        p = (p + dt * d_p) / (1.0 - dt * P_RELAXATION_COEFFICIENT)
    else:
        p = p_ref
    if mode in {"full", "q-only"}:
        q = (q + dt * d_q) / (1.0 - dt * Q_RELAXATION_COEFFICIENT)
    else:
        q = q_ref
    return rho, p, q


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
