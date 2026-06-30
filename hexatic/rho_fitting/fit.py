"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _rho_fitting_core
from .basis import chebyshev_filter_and_derivative, temporal_power_spectrum
from .cache import write_npz_atomic
from .config import RhoFittingConfig, radius_from_case_id
from .geometry import surface_lengths, theta_to_y
from .io import ActiveMatterArrays, load_active_matter_npz
from .library import density_terms, term_labels, term_names
from .plots import write_density_plots
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
    coarse_shape: tuple[int, int, int] | None = None
    temporal_shape: tuple[int, int, int] | None = None
    sample_count: int = 0
    active_terms: tuple[str, ...] = ()
    cache_path: Path | None = None
    report_path: Path | None = None

    def summary(self) -> str:
        summary = (
            f"[rho_fitting] case={self.case_id} status={self.status} nd={self.nd} "
            f"frames={self.frames} particles={self.particles} grid={self.grid_shape}"
        )
        if self.coarse_shape is not None:
            summary += f" rho={self.coarse_shape}"
        if self.temporal_shape is not None:
            summary += f" partial_t_rho={self.temporal_shape}"
        if self.sample_count:
            summary += f" samples={self.sample_count}"
        if self.active_terms:
            summary += f" active={','.join(self.active_terms)}"
        return summary


def run(config: RhoFittingConfig) -> RhoFittingResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"start case={config.case_id}")
    _progress("load active matter arrays")
    active = _load_case(config)
    active = _limit_frames(config, active)
    _progress(
        f"loaded frames={active.coords.shape[0]} particles={active.coords.shape[1]} "
        f"grid=({active.x_centers.size}, {active.theta_centers.size})"
    )

    coarse_shape = None
    temporal_shape = None
    sample_count = 0
    active_terms: tuple[str, ...] = ()
    cache_path = None
    report_path = None
    if config.coarse_grain:
        _progress("coarse-grain density")
        coarse = coarse_grain_active_fields(active, config.settings.sigma)
        coarse_shape = tuple(coarse["rho"].shape)
        _progress(f"coarse-grain complete rho={coarse_shape}")
        _progress("chebyshev filter and partial_t rho")
        spectral = spectral_active_fields(active, coarse, config)
        temporal_shape = tuple(spectral["partial_t_rho"].shape)
        _progress(f"chebyshev complete partial_t_rho={temporal_shape}")
        _progress("sample rows and fit density model")
        fit_payload = fit_density(active, spectral, config)
        sample_count = fit_payload["sample_indices"].shape[0]
        fit = fit_payload["fit"]
        active_terms = tuple(name for name, active_flag in zip(fit.names, fit.active, strict=True) if active_flag)
        _progress(f"fit complete samples={sample_count} active_terms={len(active_terms)}")
        _progress("write cache/report/plots")
        cache_path, report_path = write_density_outputs(active, spectral, fit_payload, config)
        _progress("outputs complete")

    return RhoFittingResult(
        case_id=config.case_id,
        status="density-fit" if config.coarse_grain else "data-ready",
        nd=config.settings.nd,
        frames=active.coords.shape[0],
        particles=active.coords.shape[1],
        grid_shape=(active.x_centers.size, active.theta_centers.size),
        coarse_shape=coarse_shape,
        temporal_shape=temporal_shape,
        sample_count=sample_count,
        active_terms=active_terms,
        cache_path=cache_path,
        report_path=report_path,
    )


def coarse_grain_active_fields(
    active: ActiveMatterArrays,
    sigma: float,
) -> dict[str, np.ndarray]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")

    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    y_centers = theta_to_y(active.theta_centers, active.radius)
    p_particles = np.zeros((*active.coords.shape[:2], 2), dtype=np.float64)
    result = _rho_fitting_core.coarse_grain_fields(
        np.ascontiguousarray(active.coords, dtype=np.float64),
        p_particles,
        np.ascontiguousarray(active.shell_mask),
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        np.ascontiguousarray(y_centers, dtype=np.float64),
        lx,
        ly,
        active.radius,
        float(sigma),
    )
    return {"rho": np.asarray(result["rho"])}


def spectral_active_fields(
    active: ActiveMatterArrays,
    coarse: dict[str, np.ndarray],
    config: RhoFittingConfig,
) -> dict[str, np.ndarray]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")

    _progress("chebyshev rho 1/1")
    rho_time = chebyshev_filter_and_derivative(
        coarse["rho"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    power = temporal_power_spectrum(rho_time.coefficients)
    return {
        "rho": rho_time.filtered,
        "partial_t_rho": rho_time.derivative,
        "temporal_power": power,
        "cheb_times": rho_time.times,
        "cheb_scaled_times": rho_time.scaled_times,
    }


def fit_density(
    active: ActiveMatterArrays,
    spectral: dict[str, np.ndarray],
    config: RhoFittingConfig,
) -> dict[str, np.ndarray | tuple[str, ...] | StabilityResult]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")

    terms = density_terms(config.settings.n_rho_power, config.settings.n_rho_lap_power)
    names = term_names(terms)
    labels = term_labels(terms)
    _progress(f"density terms={len(names)} ({', '.join(names)})")
    _progress("build valid mask")
    valid_mask = np.isfinite(spectral["rho"]) & np.isfinite(spectral["partial_t_rho"])
    _progress(f"sample rows target={config.settings.nd}")
    sample_indices = np.asarray(
        _rho_fitting_core.sample_rows(
            np.ascontiguousarray(valid_mask),
            config.settings.nd,
            config.settings.seed,
            config.settings.replace,
        )
    )
    _progress(f"sample rows complete actual={sample_indices.shape[0]}")
    warnings = _sampling_warnings(
        requested=config.settings.nd,
        actual=sample_indices.shape[0],
        replace=config.settings.replace,
    )
    for warning in warnings:
        _progress(f"warning: {warning}")
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    _progress("build density library")
    X = np.asarray(
        _rho_fitting_core.build_density_library(
            np.ascontiguousarray(spectral["rho"], dtype=np.float64),
            np.ascontiguousarray(sample_indices, dtype=np.int64),
            list(names),
            lx,
            ly,
        )
    )
    _progress(f"density library complete shape={X.shape}")
    _progress("sample target partial_t_rho")
    y = _sample_scalar(spectral["partial_t_rho"], sample_indices)
    finite_rows = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[finite_rows]
    y = y[finite_rows]
    sample_indices = sample_indices[finite_rows]
    _progress(f"finite regression rows={X.shape[0]}")
    _progress(
        f"stability selection taus={config.settings.tau_count} "
        f"subsamples={config.settings.subsamples}"
    )
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
        "warnings": np.asarray(warnings),
        "fit": fit,
    }


def write_density_outputs(
    active: ActiveMatterArrays,
    spectral: dict[str, np.ndarray],
    fit_payload: dict[str, np.ndarray | StabilityResult],
    config: RhoFittingConfig,
) -> tuple[Path, Path]:
    fit = fit_payload["fit"]
    if not isinstance(fit, StabilityResult):
        raise TypeError("fit payload is missing StabilityResult")
    cache_path = config.output_dir / f"{config.case_id}_fit_result.npz"
    report_path = config.output_dir / f"{config.case_id}_rho_fitting_report.md"
    metadata = _cache_metadata(active, config)
    write_npz_atomic(
        cache_path,
        overwrite=config.overwrite,
        metadata=metadata,
        rho=spectral["rho"],
        partial_t_rho=spectral["partial_t_rho"],
        temporal_power=spectral["temporal_power"],
        cheb_times=spectral["cheb_times"],
        sample_indices=fit_payload["sample_indices"],
        X_density=fit_payload["X_density"],
        y_density=fit_payload["y_density"],
        y_pred=fit.y_pred,
        term_names=fit_payload["term_names"],
        term_labels=fit_payload["term_labels"],
        coefficients=fit.coefficients,
        importance=fit.importance,
        importance_path=fit.importance_path,
        tau_values=fit.tau_values,
        active=fit.active,
        warnings=fit_payload["warnings"],
        rmse=np.asarray(fit.rmse),
        r2=np.asarray(fit.r2),
    )
    _progress(f"cache written {cache_path}")
    lines = density_report_lines(
        case_id=config.case_id,
        nd=fit_payload["sample_indices"].shape[0],
        frames=active.coords.shape[0],
        grid_shape=(active.x_centers.size, active.theta_centers.size),
        sigma=config.settings.sigma,
        cheb_cutoff=config.settings.cheb_cutoff,
        n_rho_power=config.settings.n_rho_power,
        n_rho_lap_power=config.settings.n_rho_lap_power,
        fit=fit,
        warnings=tuple(str(warning) for warning in fit_payload["warnings"]),
    )
    write_report(report_path, lines, overwrite=config.overwrite)
    _progress(f"report written {report_path}")
    if config.make_plots:
        _progress("write plots")
        write_density_plots(config.output_dir, config.case_id, fit, fit_payload["y_density"])
        _progress("plots written")
    return cache_path, report_path


def _load_case(config: RhoFittingConfig) -> ActiveMatterArrays:
    fallback_radius = radius_from_case_id(config.case_id)
    active = load_active_matter_npz(config.paths.active_fields_path, fallback_radius)
    return active


def _limit_frames(
    config: RhoFittingConfig,
    active: ActiveMatterArrays,
) -> ActiveMatterArrays:
    if config.max_frames is None or active.coords.shape[0] <= config.max_frames:
        return active

    n = config.max_frames
    return ActiveMatterArrays(
        steps=active.steps[:n],
        coords=active.coords[:n],
        shell_mask=active.shell_mask[:n],
        x_edges=active.x_edges,
        x_centers=active.x_centers,
        theta_edges=active.theta_edges,
        theta_centers=active.theta_centers,
        active_direction=None if active.active_direction is None else active.active_direction[:n],
        direction_cylindrical=None
        if active.direction_cylindrical is None
        else active.direction_cylindrical[:n],
        radius=active.radius,
    )


def _sample_scalar(field: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    return field[
        sample_indices[:, 0],
        sample_indices[:, 1],
        sample_indices[:, 2],
    ]


def _cache_metadata(active: ActiveMatterArrays, config: RhoFittingConfig) -> dict[str, object]:
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    paths = config.paths
    return {
        "case_id": config.case_id,
        "active_fields_path": str(paths.active_fields_path),
        "active_fields_mtime": paths.active_fields_path.stat().st_mtime,
        "Nx": int(active.x_centers.size),
        "Ny": int(active.theta_centers.size),
        "sigma": float(config.settings.sigma),
        "lx": float(lx),
        "ly": float(ly),
        "radius": float(active.radius),
        "cheb_cutoff": int(config.settings.cheb_cutoff),
        "n_rho_power": int(config.settings.n_rho_power),
        "n_rho_lap_power": int(config.settings.n_rho_lap_power),
        "nd": int(config.settings.nd),
        "seed": int(config.settings.seed),
        "replace": bool(config.settings.replace),
    }


def _sampling_warnings(requested: int, actual: int, replace: bool) -> tuple[str, ...]:
    if not replace and actual < requested:
        return (
            f"requested {requested} rows without replacement, but only {actual} valid rows were available",
        )
    return ()


def _progress(message: str) -> None:
    print(f"[rho_fitting] {message}", flush=True)
