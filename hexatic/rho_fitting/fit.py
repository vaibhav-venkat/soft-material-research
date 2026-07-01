"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _rho_fitting_core
from .basis import ChebyshevTimeResult, chebyshev_filter_and_derivative, temporal_power_spectrum
from .cache import write_npz_atomic
from .config import RhoFittingConfig, radius_from_case_id
from .geometry import minimum_image, surface_lengths, theta_to_y
from .io import ActiveMatterArrays, load_active_matter_npz
from .library import flux_names, term_labels, term_names
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
    _progress(f"global rho coarse-grained shape={coarse['rho'].shape}")
    spectral = spectral_active_fields(active, coarse, config)
    if config.make_plots:
        for path in write_temporal_power_plots(
            config.output_dir,
            config.case_id,
            spectral["temporal_power"],
            config.settings.cheb_cutoff,
        ):
            _progress(f"temporal power plot written {path}")

    fit_payload = fit_density(active, spectral, config)
    cache_path, report_path = write_density_outputs(active, coarse, spectral, fit_payload, config)
    fit = fit_payload["fit"]
    assert isinstance(fit, StabilityResult)
    active_terms = tuple(name for name, keep in zip(fit.names, fit.active, strict=True) if keep)
    return RhoFittingResult(
        case_id=config.case_id,
        status="density-fit",
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
    zero_vectors = np.zeros((*active.coords.shape[:2], 2), dtype=np.float64)
    rho_result = _coarse_grain(coords, zero_vectors, all_particles, active, y_centers, lx, ly, config)
    j_result = _coarse_grain(
        coords,
        particle_surface_velocities(active, config),
        all_particles,
        active,
        y_centers,
        lx,
        ly,
        config,
    )
    return {
        "rho": np.asarray(rho_result["rho"]),
        "J_density": np.asarray(j_result["P_density"]),
    }


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
    j_time = chebyshev_filter_and_derivative(
        coarse["J_density"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    _validate_temporal_alignment(rho_time, j_time)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    fluxes = _rho_fitting_core.build_density_fluxes(
        np.ascontiguousarray(rho_time.filtered, dtype=np.float64),
        lx,
        ly,
    )
    min_rho = float(np.min(rho_time.filtered))
    if min_rho < -1.0e-8:
        _progress(f"warning: filtered rho has negative values; min={min_rho:.6g}")
    return {
        "rho": rho_time.filtered,
        "J_density": j_time.filtered,
        "J_grad_rho": np.asarray(fluxes["grad_rho"]),
        "J_grad_lap_rho": np.asarray(fluxes["grad_lap_rho"]),
        "J_lap_rho_grad_rho": np.asarray(fluxes["lap_rho_grad_rho"]),
        "J_grad_rho_cubed": np.asarray(fluxes["grad_rho_cubed"]),
        "partial_t_rho": rho_time.derivative,
        "temporal_power": temporal_power_spectrum(rho_time.coefficients),
        "cheb_times": rho_time.times,
        "cheb_scaled_times": rho_time.scaled_times,
    }


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
        tau_count=config.settings.tau_count,
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
