from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import curve_fit

from hexatic.active_matter_cylinder.common import _theta_bin_indices, _x_bin_indices
from hexatic.active_matter_cylinder.config import ACTIVE_DATA_DIR, ActiveMatterFields
from hexatic.constants import cylinder

from .common import CYLINDER_PATHS, _format_theta_axis
from .series import (
    _outer_shell_xtheta_velocity_frames,
    center_of_mass_series,
    disclination_center_of_mass_series,
    dislocation_summary_series,
    first_trajectory_step,
    theta_com_velocity_series,
    x_center_of_mass_velocity_series,
)


ACTIVE_MATTER_FIELDS_FILE = ACTIVE_DATA_DIR / "active_matter_fields.npz"


@dataclass(frozen=True)
class RelaxationParams:
    v_inf: float
    amplitude: float
    tau: float
    amplitude_2: float | None = None
    tau_2: float | None = None

    @property
    def stage_count(self) -> int:
        return 1 if self.amplitude_2 is None or self.tau_2 is None else 2

    def as_tuple(self) -> tuple[float, ...]:
        if self.stage_count == 1:
            return (self.v_inf, self.amplitude, self.tau)
        assert self.amplitude_2 is not None
        assert self.tau_2 is not None
        return (self.v_inf, self.amplitude, self.tau, self.amplitude_2, self.tau_2)


@dataclass(frozen=True)
class RelaxationFit:
    stage_count: int
    params: RelaxationParams
    aicc: float
    steps: np.ndarray
    modeled_values: np.ndarray
    residual_noise: float
    rms_residual: float
    fit_mode: str


@dataclass(frozen=True)
class ShellPxChangeDecomposition:
    steps: np.ndarray
    orientation_term: np.ndarray
    membership_term: np.ndarray
    reconstructed_delta: np.ndarray
    discrete_delta: np.ndarray


@dataclass(frozen=True)
class PolarTangentPopulationSeries:
    steps: np.ndarray
    p_tangent_all: np.ndarray
    p_tangent_shell: np.ndarray
    p_tangent_core: np.ndarray
    alpha_all: np.ndarray
    alpha_shell: np.ndarray
    alpha_core: np.ndarray


@dataclass(frozen=True)
class PolarSourceResidualSeries:
    steps: np.ndarray
    source_x_shell: np.ndarray
    source_theta_shell: np.ndarray
    source_x_core: np.ndarray
    source_theta_core: np.ndarray
    tau_rot_shell: float
    tau_rot_core: float
    lambda_shell: float
    lambda_core: float


@dataclass(frozen=True)
class XResidualSeries:
    steps: np.ndarray
    residuals: np.ndarray
    mean: float
    rms: float
    autocorrelation_lags: np.ndarray
    autocorrelation: np.ndarray


@dataclass(frozen=True)
class OrientationAutocorrelationSeries:
    label: str
    lag_steps: np.ndarray
    lag_times: np.ndarray
    c_p: np.ndarray
    c_px: np.ndarray
    tau_p: float
    tau_px: float


@dataclass(frozen=True)
class InitialCylinderMaps:
    step: int
    x_edges: np.ndarray
    theta_edges: np.ndarray
    px: np.ndarray
    rho: np.ndarray
    counts: np.ndarray
    px_contribution: np.ndarray
    psi6: np.ndarray
    defects: np.ndarray
    chirality: np.ndarray
    stress_divergence_x: np.ndarray


def _single_relaxation(time: np.ndarray, v_inf: float, amplitude: float, tau: float):
    return v_inf + amplitude * np.exp(-time / tau)


def _double_relaxation(
    time: np.ndarray,
    v_inf: float,
    amplitude_1: float,
    tau_1: float,
    amplitude_2: float,
    tau_2: float,
):
    return (
        v_inf
        + amplitude_1 * np.exp(-time / tau_1)
        + amplitude_2 * np.exp(-time / tau_2)
    )


def _aicc(residuals: np.ndarray, parameter_count: int) -> float:
    n_points = residuals.size
    rss = float(np.sum(residuals**2))
    if n_points <= parameter_count + 1:
        return np.inf
    rss = max(rss, np.finfo(float).tiny)
    return (
        n_points * np.log(rss / n_points)
        + 2 * parameter_count
        + (2 * parameter_count * (parameter_count + 1))
        / (n_points - parameter_count - 1)
    )


def _fit_relaxation_series(
    steps: np.ndarray,
    values: np.ndarray,
    fit_mode: str = "auto",
) -> RelaxationFit | None:
    if fit_mode not in {"auto", "single", "double"}:
        raise ValueError("fit_mode must be 'auto', 'single', or 'double'.")

    finite = np.isfinite(steps) & np.isfinite(values)
    fit_steps = np.asarray(steps[finite], dtype=np.float64)
    fit_values = np.asarray(values[finite], dtype=np.float64)
    if fit_steps.size < 4:
        return None

    time = (fit_steps - fit_steps[0]) * float(cylinder.TIMESTEP)
    time_span = float(np.ptp(time))
    if not np.isfinite(time_span) or time_span <= 0.0:
        return None

    tail_start = max(0, int(0.75 * fit_values.size))
    v_inf_guess = float(np.nanmedian(fit_values[tail_start:]))
    amplitude_guess = float(fit_values[0] - v_inf_guess)
    if amplitude_guess == 0.0:
        amplitude_guess = float(np.nanmax(fit_values) - np.nanmin(fit_values))
    if amplitude_guess == 0.0:
        amplitude_guess = 1.0

    tau_min = max(np.finfo(float).eps, time_span / 1.0e6)
    fits = []

    if fit_mode in {"auto", "single"}:
        try:
            single_params, _ = curve_fit(
                _single_relaxation,
                time,
                fit_values,
                p0=(v_inf_guess, amplitude_guess, max(time_span / 3.0, tau_min)),
                bounds=([-np.inf, -np.inf, tau_min], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
            single_model = _single_relaxation(time, *single_params)
            fits.append(
                {
                    "params": RelaxationParams(
                        *tuple(float(value) for value in single_params)
                    ),
                    "aicc": _aicc(fit_values - single_model, 3),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError):
            pass

    if fit_mode in {"auto", "double"} and fit_steps.size >= 7:
        try:
            double_params, _ = curve_fit(
                _double_relaxation,
                time,
                fit_values,
                p0=(
                    v_inf_guess,
                    0.6 * amplitude_guess,
                    max(time_span / 10.0, tau_min),
                    0.4 * amplitude_guess,
                    max(time_span / 2.0, tau_min),
                ),
                bounds=(
                    [-np.inf, -np.inf, tau_min, -np.inf, tau_min],
                    [np.inf, np.inf, np.inf, np.inf, np.inf],
                ),
                maxfev=40000,
            )
            v_inf, amplitude_1, tau_1, amplitude_2, tau_2 = [
                float(value) for value in double_params
            ]
            if tau_1 > tau_2:
                amplitude_1, tau_1, amplitude_2, tau_2 = (
                    amplitude_2,
                    tau_2,
                    amplitude_1,
                    tau_1,
                )
            params = RelaxationParams(
                v_inf=v_inf,
                amplitude=amplitude_1,
                tau=tau_1,
                amplitude_2=amplitude_2,
                tau_2=tau_2,
            )
            double_model = _double_relaxation(time, *params.as_tuple())
            fits.append(
                {
                    "params": params,
                    "aicc": _aicc(fit_values - double_model, 5),
                }
            )
        except (RuntimeError, ValueError, FloatingPointError):
            pass

    if not fits:
        return None

    best_fit = min(fits, key=lambda fit: fit["aicc"])
    params = best_fit["params"]
    if params.stage_count == 1:
        fitted = _single_relaxation(time, *params.as_tuple())
    else:
        fitted = _double_relaxation(time, *params.as_tuple())

    residuals = fit_values - fitted
    late_start = max(0, int(0.67 * residuals.size))
    return RelaxationFit(
        stage_count=params.stage_count,
        params=params,
        aicc=float(best_fit["aicc"]),
        steps=fit_steps,
        modeled_values=fitted,
        residual_noise=float(np.nanstd(residuals[late_start:])),
        rms_residual=float(np.sqrt(np.nanmean(residuals**2))),
        fit_mode=fit_mode,
    )


def _print_x_velocity_relaxation_fit(
    fit: RelaxationFit | None,
    shell_only: bool,
) -> None:
    population = "outer-shell" if shell_only else "all-particle"
    if fit is None:
        print(f"x velocity relaxation fit ({population}): no stable fit found.")
        return

    print(f"x velocity relaxation fit ({population}):")
    print(f"  requested mode = {fit.fit_mode}")
    print(f"  stages = {fit.stage_count}")
    print(f"  V_inf = {fit.params.v_inf:.8g}")
    if fit.stage_count == 1:
        print(f"  A = {fit.params.amplitude:.8g}")
        print(f"  tau = {fit.params.tau:.8g}")
    else:
        print(f"  A_1 = {fit.params.amplitude:.8g}")
        print(f"  tau_1 = {fit.params.tau:.8g}")
        print(f"  A_2 = {fit.params.amplitude_2:.8g}")
        print(f"  tau_2 = {fit.params.tau_2:.8g}")
    print(f"  residual noise = {fit.residual_noise:.8g}")
    print(f"  RMS residual = {fit.rms_residual:.8g}")


def _load_active_matter_fields(filename: str | Path) -> ActiveMatterFields | None:
    input_path = Path(filename)
    if not input_path.exists():
        print(f"shell P_x series: active matter fields file not found at {input_path}.")
        return None

    with np.load(input_path) as data:
        return ActiveMatterFields(
            steps=np.asarray(data["steps"]),
            x_edges=np.asarray(data["x_edges"]),
            x_centers=np.asarray(data["x_centers"]),
            theta_edges=np.asarray(data["theta_edges"]),
            theta_centers=np.asarray(data["theta_centers"]),
            coords=np.asarray(data["coords"]),
            shell_mask=np.asarray(data["shell_mask"]),
            rho=np.asarray(data["rho"]),
            active_direction=np.asarray(data["active_direction"]),
            direction_cylindrical=np.asarray(data["direction_cylindrical"]),
            polar_mean=np.asarray(data["polar_mean"]),
            polar_cylindrical=np.asarray(data["polar_cylindrical"]),
            flux_cylindrical=np.asarray(data["flux_cylindrical"]),
            force_density=np.asarray(data["force_density"]),
            force_density_cylindrical=np.asarray(data["force_density_cylindrical"]),
        )


def _shell_binned_px_series(fields: ActiveMatterFields) -> tuple[np.ndarray, np.ndarray]:
    px_values = np.full(len(fields.steps), np.nan, dtype=np.float64)
    n_x_bins = len(fields.x_centers)
    n_theta_bins = len(fields.theta_centers)
    n_bins = n_x_bins * n_theta_bins

    for frame_idx in range(len(fields.steps)):
        mask = fields.shell_mask[frame_idx]
        if not np.any(mask):
            continue

        coords = fields.coords[frame_idx, mask]
        px = fields.polar_cylindrical[frame_idx, mask, 0]
        finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1]) & np.isfinite(px)
        if not np.any(finite):
            continue

        box_length_x = float(fields.x_edges[-1] - fields.x_edges[0])
        x_indices = _x_bin_indices(coords[finite, 0], box_length_x, n_x_bins)
        theta_indices = _theta_bin_indices(coords[finite, 1], n_theta_bins)
        groups = x_indices * n_theta_bins + theta_indices
        counts = np.bincount(groups, minlength=n_bins)
        sums = np.bincount(groups, weights=px[finite], minlength=n_bins)
        occupied = counts > 0
        if np.any(occupied):
            bin_means = sums[occupied] / counts[occupied]
            px_values[frame_idx] = float(np.mean(bin_means))

    return fields.steps, px_values


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    finite_mask = mask & np.isfinite(values)
    if not np.any(finite_mask):
        return np.nan
    return float(np.mean(values[finite_mask]))


def _shell_discrete_px_series(fields: ActiveMatterFields) -> tuple[np.ndarray, np.ndarray]:
    px = np.asarray(fields.direction_cylindrical[..., 0], dtype=np.float64)
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    px_values = np.full(len(fields.steps), np.nan, dtype=np.float64)

    for frame_idx in range(len(fields.steps)):
        px_values[frame_idx] = _masked_mean(px[frame_idx], shell_mask[frame_idx])

    return fields.steps, px_values


def _polar_tangent_population_series(
    fields: ActiveMatterFields,
) -> PolarTangentPopulationSeries:
    orientation = np.asarray(fields.direction_cylindrical, dtype=np.float64)
    px = orientation[..., 0]
    ptheta = orientation[..., 2]
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    px_all = np.full(len(fields.steps), np.nan, dtype=np.float64)
    ptheta_all = np.full(len(fields.steps), np.nan, dtype=np.float64)
    px_shell = np.full(len(fields.steps), np.nan, dtype=np.float64)
    ptheta_shell = np.full(len(fields.steps), np.nan, dtype=np.float64)
    px_core = np.full(len(fields.steps), np.nan, dtype=np.float64)
    ptheta_core = np.full(len(fields.steps), np.nan, dtype=np.float64)

    for frame_idx in range(len(fields.steps)):
        finite = np.isfinite(px[frame_idx]) & np.isfinite(ptheta[frame_idx])
        if np.any(finite):
            px_all[frame_idx] = float(np.mean(px[frame_idx, finite]))
            ptheta_all[frame_idx] = float(np.mean(ptheta[frame_idx, finite]))
        px_shell[frame_idx] = _masked_mean(px[frame_idx], shell_mask[frame_idx])
        ptheta_shell[frame_idx] = _masked_mean(ptheta[frame_idx], shell_mask[frame_idx])
        px_core[frame_idx] = _masked_mean(px[frame_idx], ~shell_mask[frame_idx])
        ptheta_core[frame_idx] = _masked_mean(ptheta[frame_idx], ~shell_mask[frame_idx])

    return PolarTangentPopulationSeries(
        steps=np.asarray(fields.steps, dtype=np.int64),
        p_tangent_all=np.sqrt(px_all**2 + ptheta_all**2),
        p_tangent_shell=np.sqrt(px_shell**2 + ptheta_shell**2),
        p_tangent_core=np.sqrt(px_core**2 + ptheta_core**2),
        alpha_all=np.arctan2(ptheta_all, px_all),
        alpha_shell=np.arctan2(ptheta_shell, px_shell),
        alpha_core=np.arctan2(ptheta_core, px_core),
    )


def _tangent_mean_components(
    fields: ActiveMatterFields,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orientation = np.asarray(fields.direction_cylindrical, dtype=np.float64)
    px = orientation[..., 0]
    ptheta = orientation[..., 2]
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    n_frames = len(fields.steps)
    px_all = np.full(n_frames, np.nan, dtype=np.float64)
    ptheta_all = np.full(n_frames, np.nan, dtype=np.float64)
    px_shell = np.full(n_frames, np.nan, dtype=np.float64)
    ptheta_shell = np.full(n_frames, np.nan, dtype=np.float64)
    px_core = np.full(n_frames, np.nan, dtype=np.float64)
    ptheta_core = np.full(n_frames, np.nan, dtype=np.float64)

    for frame_idx in range(n_frames):
        finite = np.isfinite(px[frame_idx]) & np.isfinite(ptheta[frame_idx])
        masks = [
            (finite, px_all, ptheta_all),
            (finite & shell_mask[frame_idx], px_shell, ptheta_shell),
            (finite & ~shell_mask[frame_idx], px_core, ptheta_core),
        ]
        for mask, px_out, ptheta_out in masks:
            if np.any(mask):
                px_out[frame_idx] = float(np.mean(px[frame_idx, mask]))
                ptheta_out[frame_idx] = float(np.mean(ptheta[frame_idx, mask]))

    return px_all, ptheta_all, px_shell, ptheta_shell, px_core, ptheta_core


def _polar_source_residual_series(
    fields: ActiveMatterFields,
) -> PolarSourceResidualSeries:
    steps = np.asarray(fields.steps, dtype=np.int64)
    (
        _px_all,
        _ptheta_all,
        px_shell,
        ptheta_shell,
        px_core,
        ptheta_core,
    ) = _tangent_mean_components(fields)
    shell_correlation = _orientation_autocorrelation_series(
        fields,
        "shell",
        "shell at t",
    )
    core_correlation = _orientation_autocorrelation_series(
        fields,
        "core",
        "core at t",
    )
    tau_shell = float(shell_correlation.tau_p)
    tau_core = float(core_correlation.tau_p)
    lambda_shell = (
        1.0 / tau_shell
        if np.isfinite(tau_shell) and tau_shell > 0.0
        else np.nan
    )
    lambda_core = (
        1.0 / tau_core
        if np.isfinite(tau_core) and tau_core > 0.0
        else np.nan
    )
    dt = np.diff(steps).astype(np.float64) * float(cylinder.TIMESTEP)

    def source_component(values: np.ndarray, tau_rot: float) -> np.ndarray:
        source = np.full(max(0, values.size - 1), np.nan, dtype=np.float64)
        valid_tau = np.isfinite(tau_rot) and tau_rot > 0.0
        if not valid_tau:
            return source
        finite = np.isfinite(values[:-1]) & np.isfinite(values[1:]) & np.isfinite(dt)
        lambda_dt = 1.0 - np.exp(-dt / tau_rot)
        source[finite] = values[1:][finite] - values[:-1][finite] + (
            lambda_dt[finite] * values[:-1][finite]
        )
        return source

    return PolarSourceResidualSeries(
        steps=steps[:-1],
        source_x_shell=source_component(px_shell, tau_shell),
        source_theta_shell=source_component(ptheta_shell, tau_shell),
        source_x_core=source_component(px_core, tau_core),
        source_theta_core=source_component(ptheta_core, tau_core),
        tau_rot_shell=tau_shell,
        tau_rot_core=tau_core,
        lambda_shell=lambda_shell,
        lambda_core=lambda_core,
    )


def _xtheta_group_indices(
    coords: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    n_x = len(x_edges) - 1
    n_theta = len(theta_edges) - 1
    x_indices = _x_bin_indices(coords[:, 0], float(x_edges[-1] - x_edges[0]), n_x)
    theta_indices = _theta_bin_indices(coords[:, 1], n_theta)
    return x_indices * n_theta + theta_indices, n_x, n_theta, n_x * n_theta


def _shell_xtheta_mean_map(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    finite = shell_mask & np.isfinite(values)
    result = np.full((len(x_edges) - 1, len(theta_edges) - 1), np.nan, dtype=np.float64)
    if not np.any(finite):
        return result

    groups, n_x, n_theta, n_bins = _xtheta_group_indices(
        coords[finite],
        x_edges,
        theta_edges,
    )
    counts = np.bincount(groups, minlength=n_bins)
    sums = np.bincount(groups, weights=values[finite], minlength=n_bins)
    occupied = counts > 0
    flat = np.full(n_bins, np.nan, dtype=np.float64)
    flat[occupied] = sums[occupied] / counts[occupied]
    return flat.reshape((n_x, n_theta))


def _shell_xtheta_px_contribution_map(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    px_values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    finite = shell_mask & np.isfinite(px_values)
    result = np.full((len(x_edges) - 1, len(theta_edges) - 1), np.nan, dtype=np.float64)
    n_shell = int(np.count_nonzero(finite))
    if n_shell == 0:
        return result

    groups, n_x, n_theta, n_bins = _xtheta_group_indices(
        coords[finite],
        x_edges,
        theta_edges,
    )
    counts = np.bincount(groups, minlength=n_bins)
    sums = np.bincount(groups, weights=px_values[finite], minlength=n_bins)
    occupied = counts > 0
    flat = np.full(n_bins, np.nan, dtype=np.float64)
    flat[occupied] = sums[occupied] / float(n_shell)
    return flat.reshape((n_x, n_theta))


def _shell_xtheta_density_map(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    counts = _shell_xtheta_count_map(coords, shell_mask, x_edges, theta_edges)
    dx = float(np.mean(np.diff(x_edges)))
    dtheta = float(np.mean(np.diff(theta_edges)))
    bin_area = dx * float(cylinder.CYLINDER_RADIUS) * dtheta
    if bin_area <= 0.0:
        return counts.astype(np.float64)
    return counts.astype(np.float64) / bin_area


def _shell_xtheta_count_map(
    coords: np.ndarray,
    shell_mask: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(x_edges) - 1, len(theta_edges) - 1), dtype=np.int64)
    if not np.any(shell_mask):
        return result

    groups, n_x, n_theta, n_bins = _xtheta_group_indices(
        coords[shell_mask],
        x_edges,
        theta_edges,
    )
    return np.bincount(groups, minlength=n_bins).reshape((n_x, n_theta))


def _closest_index(steps: np.ndarray, target_step: int) -> int:
    return int(np.argmin(np.abs(np.asarray(steps, dtype=np.int64) - int(target_step))))


def _point_xtheta_mean_map(
    coords: np.ndarray,
    values: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    theta_column: int = 1,
) -> np.ndarray:
    result = np.full((len(x_edges) - 1, len(theta_edges) - 1), np.nan, dtype=np.float64)
    finite = (
        np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, theta_column])
        & np.isfinite(values)
    )
    if not np.any(finite):
        return result

    group_coords = np.column_stack((coords[finite, 0], coords[finite, theta_column]))
    groups, n_x, n_theta, n_bins = _xtheta_group_indices(
        group_coords,
        x_edges,
        theta_edges,
    )
    counts = np.bincount(groups, minlength=n_bins)
    sums = np.bincount(groups, weights=values[finite], minlength=n_bins)
    occupied = counts > 0
    flat = np.full(n_bins, np.nan, dtype=np.float64)
    flat[occupied] = sums[occupied] / counts[occupied]
    return flat.reshape((n_x, n_theta))


def _initial_hexatic_maps(
    hexatic_gsd: str | Path,
    target_step: int,
    coords: np.ndarray,
    shell_mask: np.ndarray,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    empty = np.full((len(x_edges) - 1, len(theta_edges) - 1), np.nan, dtype=np.float64)
    path = Path(hexatic_gsd)
    if not path.exists():
        return empty, empty

    with gsd.hoomd.open(name=str(path), mode="r") as source:
        if len(source) == 0:
            return empty, empty
        steps = np.asarray(
            [int(frame.configuration.step) for frame in source],
            dtype=np.int64,
        )
        frame = source[_closest_index(steps, target_step)]
        velocity = getattr(frame.particles, "velocity", None)
        if velocity is None:
            return empty, empty
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.ndim != 2 or velocity.shape[1] < 3:
            return empty, empty

    psi6 = velocity[:, 0]
    charge = velocity[:, 2]
    defect_indicator = np.where(np.isfinite(charge), np.abs(charge) > 0.0, np.nan)
    return (
        _shell_xtheta_mean_map(coords, shell_mask, psi6, x_edges, theta_edges),
        _shell_xtheta_mean_map(
            coords,
            shell_mask,
            defect_indicator.astype(np.float64),
            x_edges,
            theta_edges,
        ),
    )


def _initial_chirality_map(
    chirality_npz: str | Path,
    target_step: int,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    metric_name: str = "instant_helix_relative",
) -> np.ndarray:
    result = np.full((len(x_edges) - 1, len(theta_edges) - 1), np.nan, dtype=np.float64)
    path = Path(chirality_npz)
    if not path.exists():
        return result

    with np.load(path) as data:
        if "xtheta_values" not in data or "steps" not in data or "metric_names" not in data:
            return result
        metric_names = tuple(str(name) for name in data["metric_names"])
        if metric_name in metric_names:
            metric_idx = metric_names.index(metric_name)
        else:
            metric_idx = 0
        steps = np.asarray(data["steps"], dtype=np.int64)
        frame_idx = _closest_index(steps, target_step)
        values = np.asarray(data["xtheta_values"], dtype=np.float64)[
            metric_idx,
            frame_idx,
        ]
        source_x_centers = np.asarray(data["x_centers"], dtype=np.float64)
        source_theta_centers = np.asarray(data["theta_centers"], dtype=np.float64)

    if values.shape == result.shape:
        return values

    x_grid, theta_grid = np.meshgrid(
        source_x_centers,
        source_theta_centers,
        indexing="ij",
    )
    coords = np.column_stack((x_grid.ravel(), theta_grid.ravel()))
    return _point_xtheta_mean_map(
        coords,
        values.ravel(),
        x_edges,
        theta_edges,
        theta_column=1,
    )


def _initial_stress_divergence_x_map(
    stress_npz: str | Path,
    target_step: int,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    field_name: str = "div_sigma_full",
) -> np.ndarray:
    result = np.full((len(x_edges) - 1, len(theta_edges) - 1), np.nan, dtype=np.float64)
    path = Path(stress_npz)
    if not path.exists():
        return result

    with np.load(path) as data:
        if field_name not in data or "grid_coords" not in data:
            return result
        if "steps" in data:
            steps = np.asarray(data["steps"], dtype=np.int64)
            frame_idx = _closest_index(steps, target_step)
            coords = np.asarray(data["grid_coords"], dtype=np.float64)
            values = np.asarray(data[field_name], dtype=np.float64)
            if coords.ndim == 3:
                coords = coords[frame_idx]
            if values.ndim == 3:
                values = values[frame_idx]
        else:
            coords = np.asarray(data["grid_coords"], dtype=np.float64)
            values = np.asarray(data[field_name], dtype=np.float64)

    if values.ndim != 2 or values.shape[1] < 1:
        return result
    return _point_xtheta_mean_map(
        coords,
        values[:, 0],
        x_edges,
        theta_edges,
        theta_column=2,
    )


def _initial_cylinder_maps(
    active_fields: ActiveMatterFields,
    frame_idx: int = 0,
    hexatic_gsd: str | Path = CYLINDER_PATHS.out_gsd,
    chirality_npz: str | Path = CYLINDER_PATHS.in_gsd.parent / "chirality_fields.npz",
    stress_npz: str | Path = CYLINDER_PATHS.in_gsd.parent
    / "shear_flux_decomposition_series.npz",
) -> InitialCylinderMaps:
    steps = np.asarray(active_fields.steps, dtype=np.int64)
    if steps.size == 0:
        raise ValueError("No active matter frames available for initial maps.")
    frame_idx = int(np.clip(frame_idx, 0, steps.size - 1))

    coords = np.asarray(active_fields.coords[frame_idx], dtype=np.float64)
    shell_mask = np.asarray(active_fields.shell_mask[frame_idx], dtype=bool)
    px_values = np.asarray(
        active_fields.direction_cylindrical[frame_idx, :, 0],
        dtype=np.float64,
    )
    x_edges = np.asarray(active_fields.x_edges, dtype=np.float64)
    theta_edges = np.asarray(active_fields.theta_edges, dtype=np.float64)
    step = int(steps[frame_idx])
    psi6_map, defect_map = _initial_hexatic_maps(
        hexatic_gsd,
        step,
        coords,
        shell_mask,
        x_edges,
        theta_edges,
    )

    return InitialCylinderMaps(
        step=step,
        x_edges=x_edges,
        theta_edges=theta_edges,
        px=_shell_xtheta_mean_map(coords, shell_mask, px_values, x_edges, theta_edges),
        rho=_shell_xtheta_density_map(coords, shell_mask, x_edges, theta_edges),
        counts=_shell_xtheta_count_map(coords, shell_mask, x_edges, theta_edges),
        px_contribution=_shell_xtheta_px_contribution_map(
            coords,
            shell_mask,
            px_values,
            x_edges,
            theta_edges,
        ),
        psi6=psi6_map,
        defects=defect_map,
        chirality=_initial_chirality_map(chirality_npz, step, x_edges, theta_edges),
        stress_divergence_x=_initial_stress_divergence_x_map(
            stress_npz,
            step,
            x_edges,
            theta_edges,
        ),
    )


def _shell_px_change_decomposition(
    fields: ActiveMatterFields,
) -> ShellPxChangeDecomposition:
    px = np.asarray(fields.direction_cylindrical[..., 0], dtype=np.float64)
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    steps = np.asarray(fields.steps, dtype=np.int64)

    n_intervals = max(0, steps.size - 1)
    orientation_term = np.full(n_intervals, np.nan, dtype=np.float64)
    membership_term = np.full(n_intervals, np.nan, dtype=np.float64)
    reconstructed_delta = np.full(n_intervals, np.nan, dtype=np.float64)
    discrete_delta = np.full(n_intervals, np.nan, dtype=np.float64)

    for interval_idx in range(n_intervals):
        shell_at_t = shell_mask[interval_idx]
        shell_at_next = shell_mask[interval_idx + 1]
        px_at_t = px[interval_idx]
        px_at_next = px[interval_idx + 1]

        mean_a_t = _masked_mean(px_at_t, shell_at_t)
        mean_a_next = _masked_mean(px_at_next, shell_at_t)
        mean_b_next = _masked_mean(px_at_next, shell_at_next)

        orientation_term[interval_idx] = mean_a_next - mean_a_t
        membership_term[interval_idx] = mean_b_next - mean_a_next
        reconstructed_delta[interval_idx] = (
            orientation_term[interval_idx] + membership_term[interval_idx]
        )
        discrete_delta[interval_idx] = mean_b_next - mean_a_t

    return ShellPxChangeDecomposition(
        steps=steps[1:],
        orientation_term=orientation_term,
        membership_term=membership_term,
        reconstructed_delta=reconstructed_delta,
        discrete_delta=discrete_delta,
    )


def _autocorrelation(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_values = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite_values.size == 0:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float64),
        )

    centered = finite_values - float(np.mean(finite_values))
    denominator = float(np.dot(centered, centered))
    lags = np.arange(finite_values.size, dtype=np.int64)
    if denominator <= 0.0:
        autocorrelation = np.full(finite_values.size, np.nan, dtype=np.float64)
        autocorrelation[0] = 1.0
        return lags, autocorrelation

    autocorrelation = np.asarray(
        [
            float(np.dot(centered[: finite_values.size - lag], centered[lag:]))
            / denominator
            for lag in lags
        ],
        dtype=np.float64,
    )
    return lags, autocorrelation


def _x_residual_series(
    input_gsd: str | Path,
    active_fields: ActiveMatterFields,
    shell_only: bool = True,
) -> XResidualSeries:
    velocity_series = x_center_of_mass_velocity_series(input_gsd, shell_only=shell_only)
    px_steps, px_values = _shell_discrete_px_series(active_fields)
    common_steps, velocity_indices, px_indices = np.intersect1d(
        velocity_series.steps,
        px_steps,
        assume_unique=False,
        return_indices=True,
    )
    residuals = (
        velocity_series.x_velocities[velocity_indices]
        - float(cylinder.U0) * px_values[px_indices]
    )
    finite = np.isfinite(residuals)
    finite_residuals = residuals[finite]
    if finite_residuals.size:
        mean = float(np.mean(finite_residuals))
        rms = float(np.sqrt(np.mean(finite_residuals**2)))
    else:
        mean = np.nan
        rms = np.nan
    autocorrelation_lags, autocorrelation = _autocorrelation(finite_residuals)

    return XResidualSeries(
        steps=common_steps,
        residuals=residuals,
        mean=mean,
        rms=rms,
        autocorrelation_lags=autocorrelation_lags,
        autocorrelation=autocorrelation,
    )


def _orientation_population_mask(
    shell_mask: np.ndarray,
    lag: int,
    population: str,
) -> np.ndarray:
    start_shell = shell_mask[: shell_mask.shape[0] - lag]
    if population == "all":
        return np.ones_like(start_shell, dtype=bool)
    if population == "shell":
        return start_shell
    if population == "shell_remain":
        next_shell = shell_mask[lag:]
        return start_shell & next_shell
    if population == "core":
        return ~start_shell
    raise ValueError(f"Unknown orientation population: {population}")


def _fit_autocorrelation_tau(
    lag_steps: np.ndarray,
    values: np.ndarray,
) -> float:
    fit = _fit_relaxation_series(lag_steps, values, fit_mode="single")
    if fit is None:
        return np.nan
    return float(fit.params.tau)


def _orientation_autocorrelation_series(
    fields: ActiveMatterFields,
    population: str,
    label: str,
) -> OrientationAutocorrelationSeries:
    directions = np.asarray(fields.active_direction, dtype=np.float64)
    px = np.asarray(fields.direction_cylindrical[..., 0], dtype=np.float64)
    shell_mask = np.asarray(fields.shell_mask, dtype=bool)
    steps = np.asarray(fields.steps, dtype=np.int64)
    n_frames = steps.size

    lag_steps = np.empty(n_frames, dtype=np.int64)
    c_p = np.full(n_frames, np.nan, dtype=np.float64)
    c_px = np.full(n_frames, np.nan, dtype=np.float64)

    for lag in range(n_frames):
        lag_steps[lag] = int(
            np.round(np.mean(steps[lag:] - steps[: n_frames - lag]))
        )
        start_directions = directions[: n_frames - lag]
        lagged_directions = directions[lag:]
        start_px = px[: n_frames - lag]
        lagged_px = px[lag:]
        mask = _orientation_population_mask(shell_mask, lag, population)

        dot_values = np.sum(start_directions * lagged_directions, axis=-1)
        axial_values = start_px * lagged_px
        finite_dot = mask & np.isfinite(dot_values)
        finite_axial = mask & np.isfinite(axial_values)

        if np.any(finite_dot):
            c_p[lag] = float(np.mean(dot_values[finite_dot]))
        if np.any(finite_axial):
            c_px[lag] = float(np.mean(axial_values[finite_axial]))

    lag_times = lag_steps.astype(np.float64) * float(cylinder.TIMESTEP)
    return OrientationAutocorrelationSeries(
        label=label,
        lag_steps=lag_steps,
        lag_times=lag_times,
        c_p=c_p,
        c_px=c_px,
        tau_p=_fit_autocorrelation_tau(lag_steps, c_p),
        tau_px=_fit_autocorrelation_tau(lag_steps, c_px),
    )


def _print_px_relaxation_fit(fit: RelaxationFit | None) -> None:
    if fit is None:
        print("shell P_x relaxation fit: no stable fit found.")
        return

    print("shell P_x relaxation fit:")
    print(f"  requested mode = {fit.fit_mode}")
    print(f"  stages = {fit.stage_count}")
    print(f"  P_inf = {fit.params.v_inf:.8g}")
    if fit.stage_count == 1:
        print(f"  A = {fit.params.amplitude:.8g}")
        print(f"  tau = {fit.params.tau:.8g}")
    else:
        print(f"  A_1 = {fit.params.amplitude:.8g}")
        print(f"  tau_1 = {fit.params.tau:.8g}")
        print(f"  A_2 = {fit.params.amplitude_2:.8g}")
        print(f"  tau_2 = {fit.params.tau_2:.8g}")
    print(f"  residual noise = {fit.residual_noise:.8g}")
    print(f"  RMS residual = {fit.rms_residual:.8g}")


def _default_restart_initial_gsd(comparison_gsd: str | Path) -> Path:
    comparison_path = Path(comparison_gsd)
    if comparison_path.name.startswith("trajectory_"):
        initial_name = comparison_path.name.replace("trajectory_", "initial_", 1)
    else:
        initial_name = f"initial_{comparison_path.name}"
    return comparison_path.parent / "initial" / initial_name


def _restart_alignment_step(
    comparison_gsd: str | Path,
    restart_initial_gsd: str | Path | None = None,
) -> int:
    initial_path = (
        Path(restart_initial_gsd)
        if restart_initial_gsd is not None
        else _default_restart_initial_gsd(comparison_gsd)
    )
    if initial_path.exists():
        return first_trajectory_step(initial_path)
    return first_trajectory_step(comparison_gsd)


def plot_restart_comparison_velocity_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    comparison_gsd: str | Path = (
        CYLINDER_PATHS.in_gsd.parent
        / "restart_ensemble"
        / "trajectory_cylinder_C.gsd"
    ),
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_x_com_velocity_restart_comparison.png"
    ),
    restart_initial_gsd: str | Path | None = None,
    shell_only: bool = False,
) -> None:
    restart_step = _restart_alignment_step(
        comparison_gsd,
        restart_initial_gsd=restart_initial_gsd,
    )
    regular_series = x_center_of_mass_velocity_series(input_gsd, shell_only=shell_only)
    comparison_series = x_center_of_mass_velocity_series(
        comparison_gsd,
        shell_only=shell_only,
    )

    regular_mask = regular_series.steps >= restart_step
    if not np.any(regular_mask):
        raise ValueError(
            f"No regular-trajectory velocity samples found at or after step {restart_step}."
        )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        regular_series.steps[regular_mask],
        regular_series.x_velocities[regular_mask],
        color="tab:blue",
        label="regular trajectory",
    )
    axis.plot(
        comparison_series.steps,
        comparison_series.x_velocities,
        color="tab:green",
        label=Path(comparison_gsd).stem,
    )
    axis.axvline(
        restart_step,
        color="black",
        linestyle=":",
        linewidth=1.4,
        alpha=0.65,
        label="restart frame",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("x center-of-mass velocity")
    population = "outer-shell" if shell_only else "all-particle"
    axis.set_title(f"Cylinder {population} x velocity restart comparison")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def _animate_outer_shell_xtheta_velocity(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.shell_xtheta_x_velocity_movie,
    start_frame: int = 100,
    n_x_bins: int = 80,
    n_theta_bins: int = 48,
    fps: int = 12,
    velocity_component: str = "x",
) -> None:
    if velocity_component not in {"x", "theta"}:
        raise ValueError("velocity_component must be 'x' or 'theta'.")

    frames = _outer_shell_xtheta_velocity_frames(
        input_gsd,
        start_frame=start_frame,
        n_x_bins=n_x_bins,
        n_theta_bins=n_theta_bins,
        velocity_component=velocity_component,
    )
    if not frames:
        raise ValueError(
            f"No consecutive frames available at or after frame {start_frame}."
        )

    all_velocities = np.concatenate(
        [
            np.asarray(frame["velocity"], dtype=np.float64)
            for frame in frames
            if np.asarray(frame["velocity"]).size
        ]
    )
    if all_velocities.size:
        vmax = float(np.nanpercentile(np.abs(all_velocities), 98.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.nanmax(np.abs(all_velocities)))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
    else:
        vmax = 1.0

    first_box_length_x = float(frames[0]["box_length_x"])
    fig, axis = plt.subplots(figsize=(11, 5.8))
    scatter = axis.scatter(
        [],
        [],
        c=[],
        s=[],
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        edgecolors="none",
    )
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
    if velocity_component == "x":
        colorbar.set_label(r"binned discrete $v_x = \Delta x / \Delta t$")
        title_prefix = "Outer-shell binned x velocity"
    else:
        colorbar.set_label(
            r"binned discrete $\omega_\theta = \Delta \theta / \Delta t$"
        )
        title_prefix = "Outer-shell binned theta velocity"
    axis.set_xlim(-0.5 * first_box_length_x, 0.5 * first_box_length_x)
    axis.set_ylim(0.0, 2.0 * np.pi)
    axis.set_xlabel("x")
    axis.set_ylabel(r"$\theta$")
    axis.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: {
                0.0: "0",
                0.5: r"$\pi/2$",
                1.0: r"$\pi$",
                1.5: r"$3\pi/2$",
                2.0: r"$2\pi$",
            }.get(round(value / np.pi, 1), "")
        )
    )
    axis.grid(True, ls="--", alpha=0.25)
    title = axis.set_title("")

    def update(animation_idx: int):
        frame = frames[animation_idx]
        x = np.asarray(frame["x"], dtype=np.float64)
        theta = np.asarray(frame["theta"], dtype=np.float64)
        velocity = np.asarray(frame["velocity"], dtype=np.float64)
        counts = np.asarray(frame["counts"], dtype=np.float64)
        scatter.set_offsets(np.column_stack((x, theta)) if x.size else np.empty((0, 2)))
        scatter.set_array(velocity)
        scatter.set_sizes(18.0 + 5.0 * np.sqrt(counts) if counts.size else [])
        axis.set_xlim(
            -0.5 * float(frame["box_length_x"]),
            0.5 * float(frame["box_length_x"]),
        )
        title.set_text(
            f"{title_prefix} on x-theta grid "
            f"(frame {int(frame['frame_idx'])}, step {int(frame['step'])})"
        )
        return scatter, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".gif":
            animation.save(output_path, writer="pillow", fps=fps)
        else:
            animation.save(output_path, writer="ffmpeg", fps=fps, dpi=160)
        plt.close(fig)


def animate_outer_shell_xtheta_x_velocity(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.shell_xtheta_x_velocity_movie,
    start_frame: int = 100,
    n_x_bins: int = 80,
    n_theta_bins: int = 48,
    fps: int = 12,
) -> None:
    _animate_outer_shell_xtheta_velocity(
        input_gsd=input_gsd,
        filename=filename,
        start_frame=start_frame,
        n_x_bins=n_x_bins,
        n_theta_bins=n_theta_bins,
        fps=fps,
        velocity_component="x",
    )


def animate_outer_shell_xtheta_theta_velocity(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.shell_xtheta_theta_velocity_movie,
    start_frame: int = 100,
    n_x_bins: int = 80,
    n_theta_bins: int = 48,
    fps: int = 12,
) -> None:
    _animate_outer_shell_xtheta_velocity(
        input_gsd=input_gsd,
        filename=filename,
        start_frame=start_frame,
        n_x_bins=n_x_bins,
        n_theta_bins=n_theta_bins,
        fps=fps,
        velocity_component="theta",
    )


def plot_velocity_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot,
    shell_only: bool = False,
    relaxation_fit_mode: str = "single",
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    align_with_px: bool = False,
) -> None:
    series = x_center_of_mass_velocity_series(input_gsd, shell_only=shell_only)
    vx_fit = _fit_relaxation_series(
        series.steps,
        series.x_velocities,
        fit_mode=relaxation_fit_mode,
    )
    assert vx_fit is not None
    _print_x_velocity_relaxation_fit(vx_fit, shell_only)
    if align_with_px:
        active_fields = _load_active_matter_fields(active_matter_fields_file)
        px_steps = px_values = px_fit = None
        if active_fields is not None:
            px_steps, px_values = _shell_discrete_px_series(active_fields)
            px_fit = _fit_relaxation_series(
                px_steps,
                px_values,
                fit_mode=relaxation_fit_mode,
            )
        _print_px_relaxation_fit(px_fit)
        assert px_fit is not None
        u0_eff = vx_fit.params.amplitude / px_fit.params.amplitude

        print(f"U0_eff = {u0_eff}")
        print(f"U0 actual = {cylinder.U0}")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.x_velocities,
        color="tab:blue",
        label="velocity_x",
    )
    if vx_fit is not None:
        fit_label = (
            "velocity_x best-fit relaxation"
            if vx_fit.stage_count == 1
            else "velocity_x best-fit two-stage relaxation"
        )
        axis.plot(
            vx_fit.steps,
            vx_fit.modeled_values,
            color="tab:blue",
            linestyle="--",
            linewidth=2.0,
            label=fit_label,
        )
    if align_with_px:
        px_axis = axis.twinx()
        if px_steps is not None and px_values is not None:
            px_axis.plot(
                px_steps,
                px_values,
                color="purple",
                label=r"$P_{x,\mathrm{shell}}$",
            )
        if px_fit is not None:
            px_fit_label = (
                r"$P_x$ best-fit relaxation"
                if px_fit.stage_count == 1
                else r"$P_x$ best-fit two-stage relaxation"
            )
            px_axis.plot(
                px_fit.steps,
                px_fit.modeled_values,
                color="purple",
                linestyle="--",
                linewidth=2.0,
                label=px_fit_label,
            )
        px_axis.axhline(0.0, color="purple", linewidth=1.0, alpha=0.2)
        px_axis.set_ylabel(r"shell mean $P_x$", color="purple")
        px_axis.tick_params(axis="y", labelcolor="purple")
        px_lines, px_labels = px_axis.get_legend_handles_labels()
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("x center-of-mass velocity", color="tab:blue")
    axis.tick_params(axis="y", labelcolor="tab:blue")
    population = "outer-shell" if shell_only else "all-particle"
    title = f"Cylinder {population} x velocity and shell mean $P_x$" if align_with_px else f"Cylinder {population} x velocity"
    axis.set_title(title)
    axis.grid(True, ls="--", alpha=0.35)
    lines, labels = axis.get_legend_handles_labels()
    l1 = lines + px_lines if align_with_px else lines
    l2 = labels + px_labels if align_with_px else labels
    axis.legend(l1, l2, loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_shell_px_change_decomposition(
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_shell_px_change_decomposition.png"
    ),
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    px_change = _shell_px_change_decomposition(active_fields)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        px_change.steps,
        px_change.orientation_term,
        color="tab:red",
        linewidth=1.5,
        alpha=0.9,
        label=(
            r"$\langle p_x(t+\Delta t)\rangle_A"
            r"-\langle p_x(t)\rangle_A$"
        ),
    )
    axis.plot(
        px_change.steps,
        px_change.membership_term,
        color="tab:orange",
        linewidth=1.5,
        alpha=0.9,
        label=(
            r"$\langle p_x(t+\Delta t)\rangle_B"
            r"-\langle p_x(t+\Delta t)\rangle_A$"
        ),
    )
    # axis.plot(
    #     px_change.steps,
    #     px_change.reconstructed_delta,
    #     color="black",
    #     linestyle="--",
    #     linewidth=1.7,
    #     alpha=0.8,
    #     label=r"$\Delta P_x$ terms sum",
    # )
    # axis.plot(
    #     px_change.steps,
    #     px_change.discrete_delta,
    #     color="tab:cyan",
    #     linestyle=":",
    #     linewidth=2.0,
    #     alpha=0.95,
    #     label=r"discrete $\Delta P_x$",
    # )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel(r"$\Delta P_x$")
    axis.set_title(r"Shell $P_x$ change decomposition")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_shell_px_change_cumsum(
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_shell_px_change_cumsum.png"
    ),
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    px_change = _shell_px_change_decomposition(active_fields)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        px_change.steps,
        np.cumsum(px_change.reconstructed_delta),
        color="black",
        linewidth=1.8,
        label=r"$\mathrm{cumsum}(\Delta P_{\mathrm{total}})$",
    )
    axis.plot(
        px_change.steps,
        np.cumsum(px_change.orientation_term),
        color="tab:red",
        linewidth=1.6,
        alpha=0.9,
        label=r"$\mathrm{cumsum}(\Delta P_{\mathrm{orient}})$",
    )
    axis.plot(
        px_change.steps,
        np.cumsum(px_change.membership_term),
        color="tab:orange",
        linewidth=1.6,
        alpha=0.9,
        label=r"$\mathrm{cumsum}(\Delta P_{\mathrm{member}})$",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel(r"Cumulative $\Delta P_x$")
    axis.set_title(r"Cumulative shell $P_x$ change decomposition")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_polar_tangent_population_series(
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_polar_tangent_population_series.png"
    ),
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    series = _polar_tangent_population_series(active_fields)

    fig, (magnitude_axis, angle_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
    )
    magnitude_axis.plot(
        series.steps,
        series.p_tangent_shell,
        color="tab:blue",
        linewidth=1.6,
        label=r"$P_{\mathrm{tangent,shell}}(t)$",
    )
    magnitude_axis.plot(
        series.steps,
        series.p_tangent_core,
        color="tab:orange",
        linewidth=1.6,
        label=r"$P_{\mathrm{tangent,core}}(t)$",
    )
    magnitude_axis.set_ylabel(r"$P_{\mathrm{tangent}}$")
    magnitude_axis.set_title(r"Mean tangent-plane polarization magnitude")
    magnitude_axis.grid(True, ls="--", alpha=0.35)
    magnitude_axis.legend(loc="best")

    angle_axis.plot(
        series.steps,
        series.alpha_shell,
        color="tab:blue",
        linewidth=1.6,
        label=r"$\alpha_{\mathrm{shell}}(t)$",
    )
    angle_axis.plot(
        series.steps,
        series.alpha_core,
        color="tab:orange",
        linewidth=1.6,
        label=r"$\alpha_{\mathrm{core}}(t)$",
    )
    angle_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    angle_axis.set_xlabel("Simulation step")
    angle_axis.set_ylabel(r"$\alpha = \mathrm{atan2}(P_\theta, P_x)$")
    angle_axis.set_title(r"Mean tangent-plane polarization angle")
    angle_axis.grid(True, ls="--", alpha=0.35)
    angle_axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_polar_source_residual_series(
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_polar_source_residual_series.png"
    ),
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    source = _polar_source_residual_series(active_fields)
    shell_magnitude = np.sqrt(source.source_x_shell**2 + source.source_theta_shell**2)
    core_magnitude = np.sqrt(source.source_x_core**2 + source.source_theta_core**2)

    print("Polar source residual decorrelation rates:")
    print(
        f"  shell tau_rot = {source.tau_rot_shell:.8g}, "
        f"lambda = {source.lambda_shell:.8g}"
    )
    print(
        f"  core tau_rot = {source.tau_rot_core:.8g}, "
        f"lambda = {source.lambda_core:.8g}"
    )

    fig, (x_axis, theta_axis, magnitude_axis) = plt.subplots(
        3,
        1,
        figsize=(10, 8.2),
        sharex=True,
    )
    x_axis.plot(
        source.steps,
        source.source_x_shell,
        color="tab:blue",
        linewidth=1.5,
        label=r"$S_{x,\mathrm{shell}}(t)$",
    )
    x_axis.plot(
        source.steps,
        source.source_x_core,
        color="tab:orange",
        linewidth=1.5,
        label=r"$S_{x,\mathrm{core}}(t)$",
    )
    x_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    x_axis.set_ylabel(r"$S_x$")
    x_axis.set_title(
        r"Polar source residual, "
        r"$S_T=\Delta P_T+(1-e^{-\Delta t/\tau_{\rm rot}})P_T$"
    )
    x_axis.grid(True, ls="--", alpha=0.35)
    x_axis.legend(loc="best")

    theta_axis.plot(
        source.steps,
        source.source_theta_shell,
        color="tab:blue",
        linewidth=1.5,
        label=r"$S_{\theta,\mathrm{shell}}(t)$",
    )
    theta_axis.plot(
        source.steps,
        source.source_theta_core,
        color="tab:orange",
        linewidth=1.5,
        label=r"$S_{\theta,\mathrm{core}}(t)$",
    )
    theta_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    theta_axis.set_ylabel(r"$S_\theta$")
    theta_axis.grid(True, ls="--", alpha=0.35)
    theta_axis.legend(loc="best")

    magnitude_axis.plot(
        source.steps,
        shell_magnitude,
        color="tab:blue",
        linewidth=1.5,
        label=r"$|S_T|_{\mathrm{shell}}(t)$",
    )
    magnitude_axis.plot(
        source.steps,
        core_magnitude,
        color="tab:orange",
        linewidth=1.5,
        label=r"$|S_T|_{\mathrm{core}}(t)$",
    )
    magnitude_axis.set_xlabel("Simulation step")
    magnitude_axis.set_ylabel(r"$|S_T|$")
    magnitude_axis.grid(True, ls="--", alpha=0.35)
    magnitude_axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_x_residual_diagnostics(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_x_residual_diagnostics.png"
    ),
    shell_only: bool = True,
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    residual_series = _x_residual_series(
        input_gsd,
        active_fields,
        shell_only=shell_only,
    )

    fig, (residual_axis, stats_axis, autocorr_axis) = plt.subplots(
        3,
        1,
        figsize=(10, 8.4),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.6]},
    )
    residual_axis.plot(
        residual_series.steps,
        residual_series.residuals,
        color="tab:blue",
        linewidth=1.5,
        label=r"$R_x = V_x - U_0 P_x$",
    )
    residual_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    residual_axis.axhline(
        residual_series.mean,
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label=fr"mean = {residual_series.mean:.4g}",
    )
    residual_axis.set_xlabel("Simulation step")
    residual_axis.set_ylabel(r"$R_x$")
    population = "outer-shell" if shell_only else "all-particle"
    residual_axis.set_title(f"Cylinder {population} x residual after active-polar removal")
    residual_axis.grid(True, ls="--", alpha=0.35)
    residual_axis.legend(loc="best")

    stats_axis.bar(
        ["mean", "RMS"],
        [residual_series.mean, residual_series.rms],
        color=["tab:red", "tab:purple"],
        alpha=0.8,
    )
    stats_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    stats_axis.set_ylabel(r"$R_x$ statistic")
    stats_axis.set_title(
        fr"mean = {residual_series.mean:.4g}, RMS = {residual_series.rms:.4g}"
    )
    stats_axis.grid(True, axis="y", ls="--", alpha=0.35)

    autocorr_axis.plot(
        residual_series.autocorrelation_lags,
        residual_series.autocorrelation,
        color="tab:green",
        linewidth=1.5,
        label=r"$R_x$ autocorrelation",
    )
    autocorr_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    autocorr_axis.set_xlabel("Lag index")
    autocorr_axis.set_ylabel("Autocorrelation")
    autocorr_axis.grid(True, ls="--", alpha=0.35)
    autocorr_axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_orientation_autocorrelation_diagnostics(
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_orientation_autocorrelation_tau.png"
    ),
    relaxation_fit_mode: str = "single",
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    populations = [
        ("all", "all particles"),
        ("shell", "shell at t"),
        ("shell_remain", "remain in shell"),
        ("core", "core at t"),
    ]
    correlations = [
        _orientation_autocorrelation_series(active_fields, key, label)
        for key, label in populations
    ]

    px_steps, px_values = _shell_discrete_px_series(active_fields)
    shell_px_fit = _fit_relaxation_series(
        px_steps,
        px_values,
        fit_mode=relaxation_fit_mode,
    )
    shell_px_tau = np.nan if shell_px_fit is None else float(shell_px_fit.params.tau)

    fig, (cp_axis, cpx_axis, tau_axis) = plt.subplots(
        3,
        1,
        figsize=(11, 9.5),
        sharex=False,
        gridspec_kw={"height_ratios": [2.0, 2.0, 1.3]},
    )
    for correlation in correlations:
        cp_axis.plot(
            correlation.lag_times,
            correlation.c_p,
            linewidth=1.5,
            label=correlation.label,
        )
        cpx_axis.plot(
            correlation.lag_times,
            correlation.c_px,
            linewidth=1.5,
            label=correlation.label,
        )

    cp_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    cp_axis.set_xlabel(r"Lag time $\tau$")
    cp_axis.set_ylabel(r"$C_p(\tau)$")
    cp_axis.set_title(r"Orientation autocorrelation $\langle p_i(t)\cdot p_i(t+\tau)\rangle$")
    cp_axis.grid(True, ls="--", alpha=0.35)
    cp_axis.legend(loc="best")

    cpx_axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    cpx_axis.set_xlabel(r"Lag time $\tau$")
    cpx_axis.set_ylabel(r"$C_{p_x}(\tau)$")
    cpx_axis.set_title(r"Axial orientation autocorrelation $\langle p_{x,i}(t)p_{x,i}(t+\tau)\rangle$")
    cpx_axis.grid(True, ls="--", alpha=0.35)
    cpx_axis.legend(loc="best")

    tau_labels: list[str] = []
    tau_values: list[float] = []
    tau_colors: list[str] = []
    for correlation in correlations:
        tau_labels.append(f"{correlation.label}\nC_p")
        tau_values.append(correlation.tau_p)
        tau_colors.append("tab:blue")
        tau_labels.append(f"{correlation.label}\nC_px")
        tau_values.append(correlation.tau_px)
        tau_colors.append("tab:orange")

    tau_axis.bar(
        np.arange(len(tau_values)),
        tau_values,
        color=tau_colors,
        alpha=0.8,
    )
    if np.isfinite(shell_px_tau):
        tau_axis.axhline(
            shell_px_tau,
            color="purple",
            linestyle="--",
            linewidth=1.6,
            label=fr"$\tau_P$ = {shell_px_tau:.4g}",
        )
    tau_axis.set_xticks(np.arange(len(tau_labels)))
    tau_axis.set_xticklabels(tau_labels, rotation=35, ha="right")
    tau_axis.set_ylabel(r"Fitted $\tau$")
    tau_axis.set_title(r"$\tau_P$ and fitted orientation decorrelation times")
    tau_axis.grid(True, axis="y", ls="--", alpha=0.35)
    if np.isfinite(shell_px_tau):
        tau_axis.legend(loc="best")

    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def _map_norm(values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin < 0.0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    return None


def _plot_xtheta_map(
    axis,
    x_edges: np.ndarray,
    theta_edges: np.ndarray,
    values: np.ndarray,
    title: str,
    colorbar_label: str,
) -> None:
    norm = _map_norm(values)
    mesh = axis.pcolormesh(
        x_edges,
        theta_edges,
        values.T,
        shading="auto",
        cmap="coolwarm" if norm is not None else "viridis",
        norm=norm,
    )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel(r"$\theta$")
    axis.set_yticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: {
                0.0: "0",
                0.5: r"$\pi/2$",
                1.0: r"$\pi$",
                1.5: r"$3\pi/2$",
                2.0: r"$2\pi$",
            }.get(round(value / np.pi, 1), "")
        )
    )
    axis.grid(False)
    axis.figure.colorbar(mesh, ax=axis, label=colorbar_label)


def _map_correlation(
    reference: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    reference_flat = np.asarray(reference, dtype=np.float64).ravel()
    values_flat = np.asarray(values, dtype=np.float64).ravel()
    finite = np.isfinite(reference_flat) & np.isfinite(values_flat)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool).ravel()
    if np.count_nonzero(finite) < 2:
        return np.nan

    reference_values = reference_flat[finite]
    comparison_values = values_flat[finite]
    if np.nanstd(reference_values) == 0.0 or np.nanstd(comparison_values) == 0.0:
        return np.nan
    return float(np.corrcoef(reference_values, comparison_values)[0, 1])


def _print_initial_stress_divergence_correlations(
    maps: InitialCylinderMaps,
) -> None:
    mask = (
        np.isfinite(maps.px_contribution)
        & np.isfinite(maps.stress_divergence_x)
    )
    n_bins = int(np.count_nonzero(mask))
    px_corr = _map_correlation(maps.px, maps.stress_divergence_x, mask=mask)
    contribution_corr = _map_correlation(
        maps.px_contribution,
        maps.stress_divergence_x,
        mask=mask,
    )

    print(
        "Initial x-theta stress-divergence bin correlations "
        f"(step {maps.step}, bins={n_bins}):"
    )
    print(f"  corr(P_x,b, stress_div_b) = {px_corr:.8g}")
    print(f"  corr(contribution_b, stress_div_b) = {contribution_corr:.8g}")


def _print_initial_px_means(
    active_fields: ActiveMatterFields,
    frame_idx: int,
) -> None:
    steps = np.asarray(active_fields.steps, dtype=np.int64)
    if steps.size == 0:
        return
    frame_idx = int(np.clip(frame_idx, 0, steps.size - 1))
    shell_mask = np.asarray(active_fields.shell_mask[frame_idx], dtype=bool)
    orientation = np.asarray(
        active_fields.direction_cylindrical[frame_idx],
        dtype=np.float64,
    )
    core_mask = ~shell_mask

    def component_means(mask: np.ndarray | None = None) -> tuple[float, float, float]:
        if mask is None:
            return tuple(float(np.nanmean(orientation[:, idx])) for idx in range(3))
        return tuple(_masked_mean(orientation[:, idx], mask) for idx in range(3))

    all_means = component_means()
    shell_means = component_means(shell_mask)
    core_means = component_means(core_mask)

    print(f"Initial cylindrical orientation means (step {int(steps[frame_idx])}):")
    print(
        "  "
        f"P_x_all = {all_means[0]:.8g}, "
        f"P_theta_all = {all_means[2]:.8g}, "
        f"P_r_all = {all_means[1]:.8g}"
    )
    print(
        "  "
        f"P_x_shell = {shell_means[0]:.8g}, "
        f"P_theta_shell = {shell_means[2]:.8g}, "
        f"P_r_shell = {shell_means[1]:.8g}"
    )
    print(
        "  "
        f"P_x_core = {core_means[0]:.8g}, "
        f"P_theta_core = {core_means[2]:.8g}, "
        f"P_r_core = {core_means[1]:.8g}"
    )


def plot_initial_cylinder_spatial_maps(
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_initial_spatial_maps.png"
    ),
    frame_idx: int = 0,
    hexatic_gsd: str | Path = CYLINDER_PATHS.out_gsd,
    chirality_npz: str | Path = CYLINDER_PATHS.in_gsd.parent / "chirality_fields.npz",
    stress_npz: str | Path = CYLINDER_PATHS.in_gsd.parent
    / "shear_flux_decomposition_series.npz",
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    maps = _initial_cylinder_maps(
        active_fields,
        frame_idx=frame_idx,
        hexatic_gsd=hexatic_gsd,
        chirality_npz=chirality_npz,
        stress_npz=stress_npz,
    )
    _print_initial_px_means(active_fields, frame_idx)
    _print_initial_stress_divergence_correlations(maps)
    panels = [
        (maps.px, r"$P_x(x,\theta)$", r"$P_x$"),
        (maps.rho, r"$\rho(x,\theta)$", r"shell surface density"),
        (
            maps.px_contribution,
            r"$[N_b/N_{\mathrm{shell}}]P_{x,b}$",
            r"contribution$_b$",
        ),
        (maps.psi6, r"$\psi_6(x,\theta)$", r"$\psi_6$"),
        (maps.defects, r"defects$(x,\theta)$", "defect fraction"),
        (maps.chirality, r"chirality$(x,\theta)$", "chirality"),
        (
            maps.stress_divergence_x,
            r"stress divergence$_x(x,\theta)$",
            r"$(\nabla\cdot\sigma)_x$",
        ),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, (values, title, colorbar_label) in zip(flat_axes, panels):
        _plot_xtheta_map(
            axis,
            maps.x_edges,
            maps.theta_edges,
            values,
            title,
            colorbar_label,
        )
    flat_axes[-1].axis("off")
    fig.suptitle(f"Initial cylinder x-theta maps (step {maps.step})")

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_initial_local_balance_maps(
    active_matter_fields_file: str | Path = ACTIVE_MATTER_FIELDS_FILE,
    filename: str | Path | None = CYLINDER_PATHS.x_com_velocity_plot.with_name(
        "cylinder_initial_local_balance_maps.png"
    ),
    frame_idx: int = 0,
    hexatic_gsd: str | Path = CYLINDER_PATHS.out_gsd,
    chirality_npz: str | Path = CYLINDER_PATHS.in_gsd.parent / "chirality_fields.npz",
    stress_npz: str | Path = CYLINDER_PATHS.in_gsd.parent
    / "shear_flux_decomposition_series.npz",
) -> None:
    active_fields = _load_active_matter_fields(active_matter_fields_file)
    if active_fields is None:
        raise FileNotFoundError(active_matter_fields_file)

    maps = _initial_cylinder_maps(
        active_fields,
        frame_idx=frame_idx,
        hexatic_gsd=hexatic_gsd,
        chirality_npz=chirality_npz,
        stress_npz=stress_npz,
    )
    _print_initial_px_means(active_fields, frame_idx)
    local_active = float(cylinder.U0) * maps.px
    local_stress = maps.stress_divergence_x / float(cylinder.GAMMA)
    local_sum = local_active + local_stress
    panels = [
        (local_active, r"local active$_b = U_0 P_{x,b}$", "local active"),
        (
            local_stress,
            r"local stress$_b = \gamma^{-1}$ stress div$_b$",
            "local stress",
        ),
        (
            local_sum,
            r"local active$_b$ + local stress$_b$",
            "local balance",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, (values, title, colorbar_label) in zip(axes, panels):
        _plot_xtheta_map(
            axis,
            maps.x_edges,
            maps.theta_edges,
            values,
            title,
            colorbar_label,
        )
    fig.suptitle(f"Initial local active-stress balance maps (step {maps.step})")

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_theta_center_of_mass_velocity_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.theta_com_velocity_plot,
    shell_only: bool = False,
) -> None:
    series = theta_com_velocity_series(input_gsd, shell_only=shell_only)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.theta_velocities,
        color="tab:orange",
        label=r"$\langle \Delta \theta_i\rangle / \Delta t$",
    )
    axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("theta center-of-mass angular velocity")
    population = "outer-shell" if shell_only else "all-particle"
    axis.set_title(f"Cylinder {population} theta center-of-mass velocity")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    filename: str | Path | None = CYLINDER_PATHS.com_plot,
) -> None:
    series = center_of_mass_series(input_gsd)

    fig, x_axis = plt.subplots(figsize=(9, 5))
    x_line = x_axis.plot(
        series.steps,
        series.x_centers,
        color="tab:blue",
        label="x center of mass",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")

    theta_axis = x_axis.twinx()
    theta_line = theta_axis.plot(
        series.steps,
        series.theta_centers,
        color="tab:orange",
        label="theta center of mass",
    )
    _format_theta_axis(theta_axis)

    lines = x_line + theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(lines, labels, loc="best")
    x_axis.set_title("Cylinder center of mass")
    x_axis.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_disclination_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.disclination_com_plot,
) -> None:
    series = disclination_center_of_mass_series(input_gsd, neighbor_count_txt)

    fig, x_axis = plt.subplots(figsize=(12, 6))
    plus_x_line = x_axis.plot(
        series.steps,
        series.plus_x_centers,
        color="#1f77b4",
        linewidth=1.8,
        alpha=0.9,
        label="+1 x COM",
    )
    minus_x_line = x_axis.plot(
        series.steps,
        series.minus_x_centers,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.8,
        alpha=0.65,
        label="-1 x COM",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")
    x_axis.margins(x=0.02, y=0.15)

    theta_axis = x_axis.twinx()
    plus_theta_line = theta_axis.plot(
        series.steps,
        series.plus_theta_centers,
        color="#ff7f0e",
        linewidth=1.8,
        alpha=0.9,
        label="+1 theta COM",
    )
    minus_theta_line = theta_axis.plot(
        series.steps,
        series.minus_theta_centers,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.8,
        alpha=0.65,
        label="-1 theta COM",
    )
    _format_theta_axis(theta_axis)

    lines = plus_x_line + plus_theta_line + minus_x_line + minus_theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(
        lines,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
    )
    x_axis.set_title("Cylinder disclination center of mass")
    x_axis.grid(True, ls="--", alpha=0.28)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

def plot_dislocation_center_of_mass_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.dislocation_com_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, x_axis = plt.subplots(figsize=(10, 5))
    x_line = x_axis.plot(
        series.steps,
        series.x_centers,
        color="tab:blue",
        label="dislocation x COM",
    )
    x_axis.set_xlabel("Simulation step")
    x_axis.set_ylabel("Center of mass x", color="tab:blue")
    x_axis.tick_params(axis="y", labelcolor="tab:blue")
    x_axis.margins(x=0.02, y=0.15)

    theta_axis = x_axis.twinx()
    theta_line = theta_axis.plot(
        series.steps,
        series.theta_centers,
        color="tab:orange",
        label="dislocation theta COM",
    )
    _format_theta_axis(theta_axis)

    lines = x_line + theta_line
    labels = [line.get_label() for line in lines]
    x_axis.legend(lines, labels, loc="best")
    x_axis.set_title("Cylinder dislocation center of mass")
    x_axis.grid(True, ls="--", alpha=0.35)
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_dislocation_count_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.dislocation_count_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.dislocation_counts,
        color="tab:purple",
        label="dislocation particles",
    )
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Number of dislocation particles")
    axis.set_title("Cylinder dislocation count")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_disclination_count_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.disclination_count_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.plus_disclination_counts,
        color="tab:orange",
        label="+1 disclinations",
    )
    axis.plot(
        series.steps,
        series.minus_disclination_counts,
        color="tab:red",
        linestyle="--",
        label="-1 disclinations",
    )
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Number of disclinations")
    axis.set_title("Cylinder disclination counts")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def plot_net_disclination_charge_series(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    filename: str | Path | None = CYLINDER_PATHS.net_charge_plot,
) -> None:
    series = dislocation_summary_series(
        input_gsd,
        neighbor_count_txt,
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        series.steps,
        series.net_disclination_charges,
        color="tab:green",
        label=r"net charge $\sum_i(6 - n_i)$",
    )
    axis.axhline(0, color="black", linewidth=1.0, alpha=0.45)
    axis.set_xlabel("Simulation step")
    axis.set_ylabel("Net disclination charge")
    axis.set_title("Cylinder net disclination charge")
    axis.grid(True, ls="--", alpha=0.35)
    axis.legend(loc="best")
    fig.tight_layout()

    if filename is None:
        plt.show()
    else:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
