"""Coupled py-pde model for validating rho-fitting closures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pde import CartesianGrid, FieldCollection, MemoryStorage, PDEBase, ScalarField
from scipy.ndimage import gaussian_filter

from .cache import ValidationInputs, load_validation_inputs
from .operators import closure_fields, divergence_surface_flux, divergence_vector, estimate_ubar


Array = NDArray[Any]
D = 3
COMPONENTS = 13
RELAXATION_COEFFICIENT = -0.01
VALIDATION_BOUND_SCALE = 10.0


@dataclass(frozen=True)
class ValidationResult:
    """Predicted and reference rollout fields with per-frame validation metrics."""

    rho_fit: Array
    p_fit: Array
    q_fit: Array
    rho_true: Array
    p_true: Array
    q_true: Array
    times: Array
    rmse_t: Array
    r2_t: Array


@dataclass(frozen=True)
class ValidationOptions:
    """Solver, filtering, frame-count, and validation-mode controls for PDE rollouts."""

    max_frames: int | None = None
    solver: str = "filtered-euler"
    dt: float | None = None
    filter_sigma: float | None = None
    mode: str = "full"


class RhoFitPDE(PDEBase):
    """py-pde model that advances fitted rho, P, and Q closure equations."""

    def __init__(self, inputs: ValidationInputs, filter_sigma: float | None = None):
        """Initialize grid spacing, fixed inputs, filters, and projection bounds."""
        super().__init__()
        self.inputs = inputs
        self.dx = inputs.lx / inputs.rho.shape[1]
        self.dtheta = inputs.theta_period / inputs.rho.shape[2]
        self.psi6_sq_fixed = np.asarray(inputs.psi6_sq[0], dtype=np.float64)
        self.filter_sigma = filter_sigma
        self.rho_min, self.rho_max = _scalar_bounds(inputs.rho)
        self.p_limit = _symmetric_bound(inputs.p)
        self.q_limit = _symmetric_bound(inputs.q)

    def evolution_rate(self, state: FieldCollection, t: float = 0.0) -> FieldCollection:
        """Return the coupled PDE time derivative for the current state and time."""
        rho, p, q = unpack_state(state)
        rho, p, q = self.project_fields(rho, p, q)
        rho_eval, p_eval, q_eval = self.filtered_fields(rho, p, q)
        rho_eval, p_eval, q_eval = self.project_fields(rho_eval, p_eval, q_eval)
        ubar = self.fit_time_ubar(t)
        closures = closure_fields(
            rho_eval,
            p_eval,
            q_eval,
            self.psi6_sq_fixed,
            self.inputs.y_rho_coefficients,
            self.inputs.y_p_coefficients,
            self.inputs.y_q_coefficients,
            self.dx,
            self.dtheta,
            self.inputs.r_centers,
            ubar_override=ubar,
        )

        rho_flux = self.inputs.u0 * p_eval + closures.f_rho / self.inputs.gamma
        d_rho = -divergence_vector(rho_flux, self.dx, self.dtheta, self.inputs.r_centers)
        d_p = -self.inputs.u0 * divergence_surface_flux(closures.f_p, self.dx, self.dtheta, self.inputs.r_centers)
        d_p += RELAXATION_COEFFICIENT * p
        d_q = -divergence_surface_flux(closures.f_q, self.dx, self.dtheta, self.inputs.r_centers)
        d_q += RELAXATION_COEFFICIENT * q
        return pack_state(state.grid, d_rho, d_p, d_q)

    def project_fields(self, rho: Array, p: Array, q: Array) -> tuple[Array, Array, Array]:
        """Clip non-finite and runaway fields back into validation-derived bounds."""
        rho_out = np.nan_to_num(rho, nan=self.rho_min, posinf=self.rho_max, neginf=self.rho_min)
        p_out = np.nan_to_num(p, nan=0.0, posinf=self.p_limit, neginf=-self.p_limit)
        q_out = np.nan_to_num(q, nan=0.0, posinf=self.q_limit, neginf=-self.q_limit)
        return (
            np.clip(rho_out, self.rho_min, self.rho_max),
            np.clip(p_out, -self.p_limit, self.p_limit),
            np.clip(q_out, -self.q_limit, self.q_limit),
        )

    def project_state(self, state: FieldCollection) -> None:
        """Apply field projection in-place to a packed py-pde state."""
        rho, p, q = self.project_fields(*unpack_state(state))
        state.data[...] = pack_state(state.grid, rho, p, q).data

    def stable_rate_data(self, rate: FieldCollection, step_dt: float) -> Array:
        """Clip rate data so one explicit step cannot jump outside projection bounds."""
        assert step_dt > 0.0, "step_dt must be positive"
        values = np.nan_to_num(rate.data, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.empty_like(values, dtype=np.float64)
        rho_limit = (self.rho_max - self.rho_min) / step_dt
        out[0] = np.clip(values[0], -rho_limit, rho_limit)
        p_rate_limit = (2.0 * self.p_limit) / step_dt
        q_rate_limit = (2.0 * self.q_limit) / step_dt
        out[1:4] = np.clip(values[1:4], -p_rate_limit, p_rate_limit)
        out[4:] = np.clip(values[4:], -q_rate_limit, q_rate_limit)
        return out

    def filtered_fields(self, rho: Array, p: Array, q: Array) -> tuple[Array, Array, Array]:
        """Optionally smooth fields with periodic x/theta filtering and nonperiodic radial filtering."""
        if self.filter_sigma is None or self.filter_sigma <= 0.0:
            return rho, p, q
        sx = self.filter_sigma / self.dx
        stheta = self.filter_sigma / (float(np.mean(self.inputs.r_centers)) * self.dtheta)
        sr = self.filter_sigma / float(np.mean(np.diff(self.inputs.r_centers)))
        rho_out = gaussian_filter(rho, sigma=(sx, stheta, sr), mode=("wrap", "wrap", "nearest"))
        p_out = gaussian_filter(p, sigma=(sx, stheta, sr, 0.0), mode=("wrap", "wrap", "nearest", "nearest"))
        q_out = gaussian_filter(q, sigma=(sx, stheta, sr, 0.0, 0.0), mode=("wrap", "wrap", "nearest", "nearest", "nearest"))
        return rho_out, p_out, q_out

    def fit_time_ubar(self, t: float) -> Array:
        """Estimate cached-time transport speed by interpolating ``A`` and ``Y_P``."""
        _, _, _, a, y_p = interpolated_cached_fields(self.inputs, t)
        return estimate_ubar(y_p, a)


def make_grid(inputs: ValidationInputs) -> CartesianGrid:
    """Create a 3D storage grid for validation fields."""
    r_min = float(inputs.r_centers[0])
    r_max = float(inputs.r_centers[-1])
    if inputs.r_centers.size > 1:
        padding = 0.5 * float(np.mean(np.diff(inputs.r_centers)))
        r_min -= padding
        r_max += padding
    return CartesianGrid(
        [(0.0, inputs.lx), (0.0, inputs.theta_period), (r_min, r_max)],
        inputs.rho.shape[1:],
        periodic=[True, True, False],
    )


def pack_state(grid: Any, rho: Array, p: Array, q: Array) -> FieldCollection:
    """Pack rho, three P components, and nine Q components into a FieldCollection."""
    fields = [ScalarField(grid, rho, label="rho")]
    fields.extend(ScalarField(grid, p[..., component], label=f"P{component}") for component in range(3))
    fields.extend(ScalarField(grid, q[..., a, b], label=f"Q{a}{b}") for a in range(3) for b in range(3))
    return FieldCollection(fields)


def unpack_state(state: FieldCollection) -> tuple[Array, Array, Array]:
    """Unpack a FieldCollection into ``rho``, ``P``, and ``Q`` arrays."""
    assert len(state) == COMPONENTS, f"expected {COMPONENTS} fields"
    rho = np.asarray(state[0].data, dtype=np.float64)
    p = np.stack([np.asarray(state[1 + component].data, dtype=np.float64) for component in range(3)], axis=-1)
    offset = 4
    q = np.empty(rho.shape + (3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            q[..., a, b] = np.asarray(state[offset + 3 * a + b].data, dtype=np.float64)
    return rho, p, q


def _scalar_bounds(values: Array) -> tuple[float, float]:
    """Return padded scalar bounds derived from finite validation input values."""
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    assert finite.size > 0, "validation input has no finite scalar values"
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = max(maximum - minimum, abs(maximum), abs(minimum), 1.0)
    padding = VALIDATION_BOUND_SCALE * span
    return minimum - padding, maximum + padding


def _symmetric_bound(values: Array) -> float:
    """Return a symmetric absolute bound derived from finite tensor input values."""
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    assert finite.size > 0, "validation input has no finite tensor values"
    return max(float(np.max(np.abs(finite))) * VALIDATION_BOUND_SCALE, 1.0)


def run_validation(
    inputs: ValidationInputs,
    options: ValidationOptions | None = None,
) -> ValidationResult:
    """Run a PDE rollout from cached frame zero and compare against cached fields.

    Parameters:
        inputs: Validated fields, coefficients, geometry, and constants from a fit cache.
        options: Solver settings, optional frame limit, smoothing, and metric mode.

    Returns:
        ``ValidationResult`` containing fitted fields, reference fields, times, RMSE, and
        R^2 arrays over the rollout frames.

    Examples:
        ``result = run_validation(inputs, ValidationOptions(max_frames=20, mode="rho-only"))``

    Edge cases:
        Single-field modes hard-reset the other fields to interpolated cached references
        before each step, so they test one equation conditionally rather than the full
        coupled rollout.
    """
    options = ValidationOptions() if options is None else options
    frame_count = (
        inputs.rho.shape[0]
        if options.max_frames is None
        else min(int(options.max_frames), inputs.rho.shape[0])
    )
    assert frame_count >= 2, "validation needs at least two frames"
    times = inputs.times[:frame_count]
    grid = make_grid(inputs)
    pde = RhoFitPDE(inputs, filter_sigma=_validation_filter_sigma(inputs, options))
    state = pack_state(grid, inputs.rho[0], inputs.p[0], inputs.q[0])
    pde.project_state(state)
    rho_fit, p_fit, q_fit = _run_solver(pde, state, times, options)
    assert np.all(np.isfinite(rho_fit)), "rho_fit became non-finite; try a smaller --dt or inspect closure stability"
    assert np.all(np.isfinite(p_fit)), "P_fit became non-finite; try a smaller --dt or inspect closure stability"
    assert np.all(np.isfinite(q_fit)), "Q_fit became non-finite; try a smaller --dt or inspect closure stability"

    rho_true = inputs.rho[:frame_count]
    p_true = inputs.p[:frame_count]
    q_true = inputs.q[:frame_count]
    metric_fit, metric_true, metric_axes = validation_metric_arrays(
        options.mode,
        rho_fit,
        p_fit,
        q_fit,
        rho_true,
        p_true,
        q_true,
    )
    residual = metric_fit - metric_true
    rmse_t = np.sqrt(np.mean(residual * residual, axis=metric_axes))
    centered = metric_true - np.mean(metric_true, axis=metric_axes, keepdims=True)
    total = np.sum(centered * centered, axis=metric_axes)
    error = np.sum(residual * residual, axis=metric_axes)
    r2_t = 1.0 - np.divide(error, total, out=np.full_like(total, np.nan), where=total > 0.0)
    return ValidationResult(
        rho_fit=rho_fit,
        p_fit=p_fit,
        q_fit=q_fit,
        rho_true=rho_true,
        p_true=p_true,
        q_true=q_true,
        times=times,
        rmse_t=rmse_t,
        r2_t=r2_t,
    )


def validation_metric_arrays(
    mode: str,
    rho_fit: Array,
    p_fit: Array,
    q_fit: Array,
    rho_true: Array,
    p_true: Array,
    q_true: Array,
) -> tuple[Array, Array, tuple[int, ...]]:
    """Select fitted/reference arrays and reduction axes for a validation mode."""
    if mode in {"full", "rho-only"}:
        return rho_fit, rho_true, (1, 2, 3)
    if mode == "p-only":
        return p_fit, p_true, (1, 2, 3, 4)
    if mode == "q-only":
        return q_fit, q_true, (1, 2, 3, 4, 5)
    raise AssertionError(f"unknown validation mode: {mode}")


def _validation_filter_sigma(inputs: ValidationInputs, options: ValidationOptions) -> float | None:
    """Choose the Gaussian filter width used by the filtered explicit Euler solver."""
    if options.solver != "filtered-euler":
        return None
    if options.filter_sigma is not None:
        return options.filter_sigma
    dx = inputs.lx / inputs.rho.shape[1]
    dtheta_length = float(np.mean(inputs.r_centers)) * inputs.theta_period / inputs.rho.shape[2]
    dr = float(np.mean(np.diff(inputs.r_centers)))
    return float(inputs.metadata.get("sigma", min(dx, dtheta_length, dr)))


def _run_solver(
    pde: RhoFitPDE,
    state: FieldCollection,
    times: Array,
    options: ValidationOptions,
) -> tuple[Array, Array, Array]:
    """Dispatch validation rollout to the requested solver implementation."""
    if options.solver == "scipy":
        assert options.mode == "full", "single-equation modes currently support euler and filtered-euler solvers"
        return _run_scipy(pde, state, times, options.dt)
    if options.solver in {"euler", "filtered-euler"}:
        dt = 5.0e-3 if options.dt is None else options.dt
        return _run_euler(pde, state, times, dt, options.mode)
    raise AssertionError(f"unknown validation solver: {options.solver}")


def _run_scipy(pde: RhoFitPDE, state: FieldCollection, times: Array, dt: float | None) -> tuple[Array, Array, Array]:
    """Run the full coupled PDE with py-pde's SciPy BDF solver."""
    storage = MemoryStorage()
    kwargs: dict[str, Any] = {"method": "BDF"}
    if dt is not None:
        assert dt > 0.0, "dt must be positive"
        kwargs["max_step"] = float(dt)
    pde.solve(
        state,
        t_range=(float(times[0]), float(times[-1])),
        tracker=storage.tracker(times),
        solver="scipy",
        **kwargs,
    )
    assert len(storage.data) == times.size, "py-pde did not store every requested frame"
    rho_fit = np.asarray([frame[0] for frame in storage.data], dtype=np.float64)
    p_fit = np.asarray([np.moveaxis(frame[1:4], 0, -1) for frame in storage.data], dtype=np.float64)
    q_fit = np.asarray([np.moveaxis(frame[4:].reshape(3, 3, *frame.shape[1:]), (0, 1), (-2, -1)) for frame in storage.data], dtype=np.float64)
    return rho_fit, p_fit, q_fit


def interpolation_index_weight(times: Array, t: float) -> tuple[int, float]:
    """Return bracketing time index and linear interpolation weight for ``t``."""
    index = int(np.searchsorted(times, t, side="right") - 1)
    index = max(0, min(index, times.size - 2))
    t0 = float(times[index])
    t1 = float(times[index + 1])
    weight = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return index, weight


def interpolate_time_series(values: Array, times: Array, t: float) -> Array:
    """Linearly interpolate a time-indexed array at physical time ``t``."""
    index, weight = interpolation_index_weight(times, t)
    return (1.0 - weight) * values[index] + weight * values[index + 1]


def interpolated_fields(inputs: ValidationInputs, t: float) -> tuple[Array, Array, Array]:
    """Return interpolated cached rho, P, and Q fields at physical time ``t``."""
    rho, p, q, _, _ = interpolated_cached_fields(inputs, t)
    return rho, p, q


def interpolated_cached_fields(inputs: ValidationInputs, t: float) -> tuple[Array, Array, Array, Array, Array]:
    """Return interpolated cached rho, P, Q, A, and Y_P fields at physical time ``t``."""
    rho = interpolate_time_series(inputs.rho, inputs.times, t)
    p = interpolate_time_series(inputs.p, inputs.times, t)
    q = interpolate_time_series(inputs.q, inputs.times, t)
    a = interpolate_time_series(inputs.a, inputs.times, t)
    y_p = interpolate_time_series(inputs.y_p, inputs.times, t)
    return rho, p, q, a, y_p


def interpolated_p_q(inputs: ValidationInputs, t: float) -> tuple[Array, Array]:
    """Return interpolated cached P and Q fields at physical time ``t``."""
    _, p, q = interpolated_fields(inputs, t)
    return p, q


def _run_euler(
    pde: RhoFitPDE,
    state: FieldCollection,
    times: Array,
    dt: float,
    mode: str,
) -> tuple[Array, Array, Array]:
    """Run explicit Euler validation with optional single-equation modes."""
    assert mode in {"full", "rho-only", "p-only", "q-only"}, f"unknown validation mode: {mode}"
    rho_fit = np.empty((times.size,) + state.grid.shape, dtype=np.float64)
    p_fit = np.empty((times.size,) + state.grid.shape + (3,), dtype=np.float64)
    q_fit = np.empty((times.size,) + state.grid.shape + (3, 3), dtype=np.float64)
    rho_fit[0], p_fit[0], q_fit[0] = unpack_state(state)
    assert dt > 0.0, "dt must be positive"
    for index in range(times.size - 1):
        frame_dt = float(times[index + 1] - times[index])
        assert frame_dt > 0.0, "validation times must be strictly increasing"
        substeps = max(1, int(np.ceil(frame_dt / dt)))
        step_dt = frame_dt / substeps
        for substep in range(substeps):
            t = float(times[index]) + substep * step_dt
            if mode == "full":
                _step_full_state(pde, state, t, step_dt)
            else:
                _step_single_field_state(pde, state, t, step_dt, mode)
            assert np.all(np.isfinite(state.data)), "validation state became non-finite"
        rho_fit[index + 1], p_fit[index + 1], q_fit[index + 1] = unpack_state(state)
    return rho_fit, p_fit, q_fit


def _step_full_state(
    pde: RhoFitPDE,
    state: FieldCollection,
    t: float,
    step_dt: float,
) -> None:
    """Advance all packed fields by one projected explicit Euler step."""
    rate = pde.evolution_rate(state, t)
    state.data[...] = state.data + step_dt * pde.stable_rate_data(rate, step_dt)
    pde.project_state(state)


def _step_single_field_state(
    pde: RhoFitPDE,
    state: FieldCollection,
    t: float,
    step_dt: float,
    mode: str,
) -> None:
    """Advance only the selected field while resetting other fields to cached references."""
    hard_state = _single_field_reference_state(pde, state, t, mode)
    rate = pde.evolution_rate(hard_state, t)
    rate_data = pde.stable_rate_data(rate, step_dt)
    if mode == "rho-only":
        state.data[0] = state.data[0] + step_dt * rate_data[0]
        state.data[1:] = hard_state.data[1:]
    elif mode == "p-only":
        state.data[0] = hard_state.data[0]
        state.data[1:4] = state.data[1:4] + step_dt * rate_data[1:4]
        state.data[4:] = hard_state.data[4:]
    else:
        state.data[:4] = hard_state.data[:4]
        state.data[4:] = state.data[4:] + step_dt * rate_data[4:]
    pde.project_state(state)


def _single_field_reference_state(
    pde: RhoFitPDE,
    state: FieldCollection,
    t: float,
    mode: str,
) -> FieldCollection:
    """Build a state mixing the live selected field with cached reference fields."""
    rho_data, p_data, q_data = interpolated_fields(pde.inputs, t)
    rho, p, q = unpack_state(state)
    rho_eval = rho if mode == "rho-only" else rho_data
    p_eval = p if mode == "p-only" else p_data
    q_eval = q if mode == "q-only" else q_data
    return pack_state(state.grid, rho_eval, p_eval, q_eval)


def run_validation_from_cache(
    cache_path: Path,
    options: ValidationOptions | None = None,
) -> tuple[ValidationInputs, ValidationResult]:
    """Load validation inputs from a cache path and run one validation rollout."""
    inputs = load_validation_inputs(cache_path)
    return inputs, run_validation(inputs, options)
