"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _rho_fitting_core
from .basis import ChebyshevTimeResult, chebyshev_filter_and_derivative, temporal_power_spectrum
from .cache import write_npz_atomic
from .config import RhoFittingConfig, radius_from_case_id
from .geometry import minimum_image, surface_lengths, tangential_particle_vectors, theta_to_y
from .io import ActiveMatterArrays, load_active_matter_npz, load_gsd_orientations, validate_step_alignment
from .library import flux_names, mechanical_labels, term_labels, term_names
from .plots import write_density_plots, write_temporal_power_plots
from .regression import StabilityResult, stability_selection
from .report import density_report_lines, write_report


@dataclass(frozen=True)
class RhoFittingResult:
    case_id: str
    status: str
    nd: int
    frames: int
    particles: int
    grid_shape: tuple[int, int]
    sample_count: int
    active_terms: tuple[str, ...]
    cache_path: Path
    report_path: Path

    def summary(self) -> str:
        active = ",".join(self.active_terms) if self.active_terms else "none"
        return (
            f"[rho_fitting] case={self.case_id} status={self.status} nd={self.nd} "
            f"frames={self.frames} particles={self.particles} grid={self.grid_shape} "
            f"samples={self.sample_count} active={active}"
        )


def run(config: RhoFittingConfig) -> RhoFittingResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    active = load_active_matter_npz(config.paths.active_fields_path, radius_from_case_id(config.case_id))
    _progress(
        f"loaded frames={active.coords.shape[0]} particles={active.coords.shape[1]} "
        f"grid=({active.x_centers.size}, {active.theta_centers.size})"
    )

    coarse = coarse_grain_active_fields(active, config)
    _progress(f"mechanical fields coarse-grained rho shape={coarse['rho'].shape}")
    spectral = spectral_active_fields(active, coarse, config)
    if config.make_plots:
        for path in write_temporal_power_plots(
            config.output_dir,
            config.case_id,
            spectral["temporal_power"],
            config.settings.cheb_cutoff,
        ):
            _progress(f"temporal power plot written {path}")

    fit_payload = fit_mechanical(active, spectral, config)
    cache_path, report_path = write_mechanical_outputs(active, coarse, spectral, fit_payload, config)
    active_terms = tuple(
        f"{target}:{name}"
        for target in ("Y_rho", "Y_P", "Y_Q")
        for name, keep in zip(
            fit_payload[f"{target}_fit"].names,
            fit_payload[f"{target}_fit"].active,
            strict=True,
        )
        if keep
    )
    return RhoFittingResult(
        case_id=config.case_id,
        status="mechanical-fit",
        nd=config.settings.nd,
        frames=active.coords.shape[0],
        particles=active.coords.shape[1],
        grid_shape=(active.x_centers.size, active.theta_centers.size),
        sample_count=fit_payload["sample_indices"].shape[0],
        active_terms=active_terms,
        cache_path=cache_path,
        report_path=report_path,
    )


def coarse_grain_active_fields(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> dict[str, np.ndarray]:
    assert _rho_fitting_core is not None, "rho_fitting extension is not built"
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    coords = np.ascontiguousarray(active.coords, dtype=np.float64)
    y_centers = np.ascontiguousarray(theta_to_y(active.theta_centers, active.radius), dtype=np.float64)
    all_particles = np.ones_like(active.shell_mask, dtype=bool)
    directions = particle_tangent_directions(active, config)
    velocities = particle_surface_velocities(active, config)
    fields = _rho_fitting_core.build_mechanical_fields(
        coords,
        np.ascontiguousarray(directions, dtype=np.float64),
        np.ascontiguousarray(velocities, dtype=np.float64),
        all_particles,
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        y_centers,
        lx,
        ly,
        active.radius,
        float(config.settings.sigma),
        float(config.settings.gamma),
        float(config.settings.u0),
    )
    out = {name: np.asarray(value) for name, value in fields.items()}
    out["J_density"] = out["J_rho"]
    return out


def particle_tangent_directions(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    orientation = None
    if active.direction_cylindrical is None and active.active_direction is None:
        gsd = load_gsd_orientations(config.paths.gsd_path)
        validate_step_alignment(active, gsd)
        orientation = gsd.orientation
    return tangential_particle_vectors(
        active.coords,
        direction_cylindrical=active.direction_cylindrical,
        active_direction=active.active_direction,
        orientation=orientation,
    )


def particle_surface_velocities(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    coords = np.asarray(active.coords, dtype=np.float64)
    velocities = np.zeros((*coords.shape[:2], 2), dtype=np.float64)
    if coords.shape[0] < 2:
        return velocities

    lx, _ = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    times = (np.asarray(active.steps, dtype=np.float64) - float(active.steps[0])) * config.settings.timestep
    for frame in range(coords.shape[0]):
        left = max(0, frame - 1)
        right = min(coords.shape[0] - 1, frame + 1)
        dt = times[right] - times[left]
        assert dt > 0.0, "steps must increase over time"
        velocities[frame, :, 0] = minimum_image(coords[right, :, 0] - coords[left, :, 0], lx) / dt
        velocities[frame, :, 1] = (
            active.radius * minimum_image(coords[right, :, 1] - coords[left, :, 1], 2.0 * np.pi) / dt
        )
    return np.ascontiguousarray(velocities)


def spectral_active_fields(
    active: ActiveMatterArrays,
    coarse: dict[str, np.ndarray],
    config: RhoFittingConfig,
) -> dict[str, np.ndarray]:
    assert _rho_fitting_core is not None, "rho_fitting extension is not built"
    rho_time = chebyshev_filter_and_derivative(
        coarse["rho"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    filtered = {"rho": rho_time.filtered}
    coefficients = [rho_time.coefficients]
    for name in ("P", "Q", "A", "J_rho", "J_P", "J_Q"):
        result = chebyshev_filter_and_derivative(
            coarse[name],
            active.steps,
            config.settings.timestep,
            config.settings.cheb_cutoff,
        )
        _validate_temporal_alignment(rho_time, result)
        filtered[name] = result.filtered
        coefficients.append(result.coefficients)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    fluxes = _rho_fitting_core.build_density_fluxes(
        np.ascontiguousarray(rho_time.filtered, dtype=np.float64),
        lx,
        ly,
    )
    targets = _rho_fitting_core.build_mechanical_targets(
        np.ascontiguousarray(filtered["P"], dtype=np.float64),
        np.ascontiguousarray(filtered["J_rho"], dtype=np.float64),
        np.ascontiguousarray(filtered["J_P"], dtype=np.float64),
        np.ascontiguousarray(filtered["J_Q"], dtype=np.float64),
        float(config.settings.gamma),
        float(config.settings.u0),
    )
    min_rho = float(np.min(rho_time.filtered))
    if min_rho < -1.0e-8:
        _progress(f"warning: filtered rho has negative values; min={min_rho:.6g}")
    return {
        "rho": rho_time.filtered,
        "P": filtered["P"],
        "Q": filtered["Q"],
        "A": filtered["A"],
        "J_density": filtered["J_rho"],
        "J_rho": filtered["J_rho"],
        "J_P": filtered["J_P"],
        "J_Q": filtered["J_Q"],
        "Y_rho": np.asarray(targets["Y_rho"]),
        "Y_P": np.asarray(targets["Y_P"]),
        "Y_Q": np.asarray(targets["Y_Q"]),
        "J_grad_rho": np.asarray(fluxes["grad_rho"]),
        "J_grad_lap_rho": np.asarray(fluxes["grad_lap_rho"]),
        "J_lap_rho_grad_rho": np.asarray(fluxes["lap_rho_grad_rho"]),
        "J_grad_rho_cubed": np.asarray(fluxes["grad_rho_cubed"]),
        "partial_t_rho": rho_time.derivative,
        "temporal_power": temporal_power_spectrum(*coefficients),
        "cheb_times": rho_time.times,
        "cheb_scaled_times": rho_time.scaled_times,
    }


def fit_mechanical(
    active: ActiveMatterArrays,
    spectral: dict[str, np.ndarray],
    config: RhoFittingConfig,
) -> dict[str, np.ndarray | StabilityResult]:
    valid_mask = _mechanical_valid_mask(spectral)
    sample_indices = np.asarray(
        _rho_fitting_core.sample_rows(
            np.ascontiguousarray(valid_mask),
            config.settings.nd,
            config.settings.seed,
            config.settings.replace,
        )
    )
    warnings = _sampling_warnings(config.settings.nd, sample_indices.shape[0], config.settings.replace)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    libraries = _mechanical_libraries(spectral, spectral["Y_rho"], lx, ly)
    y_rho_fit, y_rho_rows, y_rho_index = _fit_component_target("Y_rho", spectral["Y_rho"], libraries["Y_rho"], libraries["Y_rho_names"], sample_indices, config)
    f_rho_pred = np.tensordot(y_rho_fit.coefficients, libraries["Y_rho"], axes=(0, 0))
    libraries = _mechanical_libraries(spectral, f_rho_pred, lx, ly)
    y_p_fit, y_p_rows, y_p_index = _fit_component_target("Y_P", spectral["Y_P"], libraries["Y_P"], libraries["Y_P_names"], sample_indices, config)
    y_q_fit, y_q_rows, y_q_index = _fit_component_target("Y_Q", spectral["Y_Q"], libraries["Y_Q"], libraries["Y_Q_names"], sample_indices, config)
    return {
        "sample_indices": sample_indices,
        "warnings": np.asarray(warnings),
        "Y_rho_fit": y_rho_fit,
        "Y_P_fit": y_p_fit,
        "Y_Q_fit": y_q_fit,
        "Y_rho_rows": y_rho_rows,
        "Y_P_rows": y_p_rows,
        "Y_Q_rows": y_q_rows,
        "Y_rho_row_index": y_rho_index,
        "Y_P_row_index": y_p_index,
        "Y_Q_row_index": y_q_index,
        "Y_rho_library": libraries["Y_rho"],
        "Y_P_library": libraries["Y_P"],
        "Y_Q_library": libraries["Y_Q"],
        "Y_rho_names": np.asarray(libraries["Y_rho_names"]),
        "Y_P_names": np.asarray(libraries["Y_P_names"]),
        "Y_Q_names": np.asarray(libraries["Y_Q_names"]),
        "F_rho_prediction": f_rho_pred,
    }


def _mechanical_libraries(
    spectral: dict[str, np.ndarray],
    f_rho: np.ndarray,
    lx: float,
    ly: float,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    libs = _rho_fitting_core.build_mechanical_libraries(
        np.ascontiguousarray(spectral["rho"], dtype=np.float64),
        np.ascontiguousarray(spectral["P"], dtype=np.float64),
        np.ascontiguousarray(spectral["A"], dtype=np.float64),
        np.ascontiguousarray(f_rho, dtype=np.float64),
        lx,
        ly,
    )
    return {
        "Y_rho": np.asarray(libs["Y_rho"]),
        "Y_P": np.asarray(libs["Y_P"]),
        "Y_Q": np.asarray(libs["Y_Q"]),
        "Y_rho_names": tuple(str(name) for name in libs["Y_rho_names"]),
        "Y_P_names": tuple(str(name) for name in libs["Y_P_names"]),
        "Y_Q_names": tuple(str(name) for name in libs["Y_Q_names"]),
    }


def _fit_component_target(
    target_name: str,
    target: np.ndarray,
    library: np.ndarray,
    names: tuple[str, ...],
    sample_indices: np.ndarray,
    config: RhoFittingConfig,
) -> tuple[StabilityResult, np.ndarray, np.ndarray]:
    row_payload = _rho_fitting_core.sample_component_rows(
        np.ascontiguousarray(target, dtype=np.float64),
        np.ascontiguousarray(library, dtype=np.float64),
        np.ascontiguousarray(sample_indices, dtype=np.int64),
    )
    rows = np.asarray(row_payload["rows"])
    row_index = np.asarray(row_payload["row_index"])
    y = rows[:, 0]
    X = rows[:, 1:]
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[finite]
    X = X[finite]
    rows = rows[finite]
    row_index = row_index[finite]
    labels = mechanical_labels(names)
    _progress(f"{target_name} fit rows={X.shape[0]} terms={', '.join(names)}")
    fit = stability_selection(
        X,
        y,
        names,
        labels,
        seed=config.settings.seed,
        tau_eps=config.settings.tau_eps,
        subsamples=config.settings.subsamples,
        importance_threshold=config.settings.importance_threshold,
        alpha=config.settings.alpha,
        max_iter=config.settings.stlsq_max_iter,
    )
    return fit, rows, row_index


def _mechanical_valid_mask(spectral: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.isfinite(spectral["rho"])
    for name in ("P", "A", "J_rho", "J_P", "J_Q", "Y_rho", "Y_P", "Y_Q"):
        axes = tuple(range(3, spectral[name].ndim))
        valid &= np.all(np.isfinite(spectral[name]), axis=axes)
    return valid


def fit_density(
    active: ActiveMatterArrays,
    spectral: dict[str, np.ndarray],
    config: RhoFittingConfig,
) -> dict[str, np.ndarray | StabilityResult]:
    assert _rho_fitting_core is not None, "rho_fitting extension is not built"
    names = term_names()
    labels = term_labels()
    valid_mask = np.isfinite(spectral["rho"]) & np.isfinite(spectral["partial_t_rho"])
    sample_indices = np.asarray(
        _rho_fitting_core.sample_rows(
            np.ascontiguousarray(valid_mask),
            config.settings.nd,
            config.settings.seed,
            config.settings.replace,
        )
    )
    warnings = _sampling_warnings(config.settings.nd, sample_indices.shape[0], config.settings.replace)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    X = np.asarray(
        _rho_fitting_core.build_density_library(
            np.ascontiguousarray(spectral["rho"], dtype=np.float64),
            np.ascontiguousarray(sample_indices, dtype=np.int64),
            list(names),
            lx,
            ly,
        )
    )
    y = _sample_scalar(spectral["partial_t_rho"], sample_indices)
    finite_rows = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite_rows]
    y = y[finite_rows]
    sample_indices = sample_indices[finite_rows]
    _progress(f"fit rows={X.shape[0]} terms={', '.join(names)}")
    fit = stability_selection(
        X,
        y,
        names,
        labels,
        seed=config.settings.seed,
        tau_eps=config.settings.tau_eps,
        subsamples=config.settings.subsamples,
        importance_threshold=config.settings.importance_threshold,
        alpha=config.settings.alpha,
        max_iter=config.settings.stlsq_max_iter,
    )
    return {
        "X_density": X,
        "y_density": y,
        "sample_indices": sample_indices,
        "term_names": np.asarray(names),
        "term_labels": np.asarray(labels),
        "term_fluxes": np.asarray(flux_names()),
        "warnings": np.asarray(warnings),
        "fit": fit,
    }


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
        metadata=_cache_metadata(active, config) | {"analysis": "global_mechanical_moment_fits"},
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
        warnings=fit_payload["warnings"],
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
            warnings=tuple(str(warning) for warning in fit_payload["warnings"]),
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
    warnings: tuple[str, ...],
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
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
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
                f"{importance:.4f} | {_format_float(raw_correlation)} |"
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
        metadata=_cache_metadata(active, config),
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
        warnings=fit_payload["warnings"],
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
            warnings=tuple(str(warning) for warning in fit_payload["warnings"]),
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


def _coarse_grain(
    coords: np.ndarray,
    vectors: np.ndarray,
    mask: np.ndarray,
    active: ActiveMatterArrays,
    y_centers: np.ndarray,
    lx: float,
    ly: float,
    config: RhoFittingConfig,
):
    return _rho_fitting_core.coarse_grain_fields(
        coords,
        np.ascontiguousarray(vectors, dtype=np.float64),
        np.ascontiguousarray(mask),
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        y_centers,
        lx,
        ly,
        active.radius,
        float(config.settings.sigma),
    )


def _validate_temporal_alignment(
    rho_time: ChebyshevTimeResult,
    j_time: ChebyshevTimeResult,
) -> None:
    assert rho_time.filtered.shape == rho_time.derivative.shape
    assert j_time.filtered.shape[:3] == rho_time.filtered.shape
    assert np.array_equal(rho_time.times, j_time.times)
    assert np.array_equal(rho_time.scaled_times, j_time.scaled_times)


def _sample_scalar(field: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    return field[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]]


def _cache_metadata(active: ActiveMatterArrays, config: RhoFittingConfig) -> dict[str, object]:
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


def _sampling_warnings(requested: int, actual: int, replace: bool) -> tuple[str, ...]:
    if not replace and actual < requested:
        return (f"requested {requested} rows without replacement, but only {actual} valid rows were available",)
    return ()


def _progress(message: str) -> None:
    print(f"[rho_fitting] {message}", flush=True)


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"
