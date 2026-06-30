"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import _rho_fitting_core
from .basis import ChebyshevTimeResult, chebyshev_filter_and_derivative, temporal_power_spectrum
from .cache import write_npz_atomic
from .config import RhoFittingConfig, radius_from_case_id
from .geometry import minimum_image, surface_lengths, tangential_particle_vectors, theta_to_y
from .io import ActiveMatterArrays, load_active_matter_npz
from .library import density_terms, term_labels, term_names
from .plots import write_density_plots, write_temporal_power_plot
from .regression import StabilityResult, raw_feature_correlations, stability_selection
from .report import density_report_lines, write_report


@dataclass(frozen=True)
class RhoFittingResult:
    case_id: str
    status: str
    nd: int
    frames: int
    particles: int | None
    grid_shape: tuple[int, int]
    coarse_shape: tuple[int, int, int] | None = None
    temporal_shape: tuple[int, int, int] | None = None
    sample_count: int = 0
    active_terms: tuple[str, ...] = ()
    cache_path: Path | None = None
    report_path: Path | None = None

    def summary(self) -> str:
        particle_text = "" if self.particles is None else f" particles={self.particles}"
        summary = (
            f"[rho_fitting] case={self.case_id} status={self.status} nd={self.nd} "
            f"frames={self.frames}{particle_text} grid={self.grid_shape}"
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
        coarse = coarse_grain_active_fields(active, config)
        coarse_shape = tuple(coarse["rho"].shape)
        _progress(
            f"coarse-grain complete rho={coarse_shape} "
            f"P={coarse['P_density'].shape} J={coarse['J_density'].shape} "
            f"S={coarse['S_cross'].shape}"
        )
        _progress("chebyshev filter and partial_t rho")
        spectral = spectral_active_fields(active, coarse, config)
        temporal_shape = tuple(spectral["partial_t_rho"].shape)
        _progress(f"chebyshev complete partial_t_rho={temporal_shape}")
        if config.make_plots:
            path = write_temporal_power_plot(
                config.output_dir,
                config.case_id,
                spectral["temporal_power"],
                config.settings.cheb_cutoff,
            )
            _progress(f"temporal power plot written {path}")
        _progress("sample rows and fit density model")
        fit_payload = fit_density(active, spectral, config)
        sample_count = fit_payload["sample_indices"].shape[0]
        fit = fit_payload["fit"]
        active_terms = tuple(name for name, active_flag in zip(fit.names, fit.active, strict=True) if active_flag)
        _progress(f"fit complete samples={sample_count} active_terms={len(active_terms)}")
        _progress("write cache/report/plots")
        cache_path, report_path = write_density_outputs(active, coarse, spectral, fit_payload, config)
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
    config: RhoFittingConfig,
) -> dict[str, np.ndarray]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")

    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    y_centers = theta_to_y(active.theta_centers, active.radius)
    p_particles = tangential_particle_vectors(
        active.coords,
        direction_cylindrical=active.direction_cylindrical,
        active_direction=active.active_direction,
    )
    j_particles = particle_surface_velocities(active, config)
    source_particles = particle_shell_source(active, config)
    all_particles = np.ones_like(active.shell_mask, dtype=bool)
    result = _rho_fitting_core.coarse_grain_fields(
        np.ascontiguousarray(active.coords, dtype=np.float64),
        p_particles,
        np.ascontiguousarray(active.shell_mask),
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        np.ascontiguousarray(y_centers, dtype=np.float64),
        lx,
        ly,
        active.radius,
        float(config.settings.sigma),
    )
    j_result = _rho_fitting_core.coarse_grain_fields(
        np.ascontiguousarray(active.coords, dtype=np.float64),
        j_particles,
        np.ascontiguousarray(active.shell_mask),
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        np.ascontiguousarray(y_centers, dtype=np.float64),
        lx,
        ly,
        active.radius,
        float(config.settings.sigma),
    )
    source_result = _rho_fitting_core.coarse_grain_fields(
        np.ascontiguousarray(active.coords, dtype=np.float64),
        source_particles,
        np.ascontiguousarray(all_particles),
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        np.ascontiguousarray(y_centers, dtype=np.float64),
        lx,
        ly,
        active.radius,
        float(config.settings.sigma),
    )
    return {
        "rho": np.asarray(result["rho"]),
        "P_density": np.asarray(result["P_density"]),
        "J_density": np.asarray(j_result["P_density"]),
        "S_cross": np.asarray(source_result["P_density"])[..., 0],
    }


def particle_surface_velocities(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    coords = np.asarray(active.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return np.zeros((*coords.shape[:2], 2), dtype=np.float64)

    lx, _ = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    times = (np.asarray(active.steps, dtype=np.float64) - float(active.steps[0])) * float(
        config.settings.timestep
    )
    velocities = np.zeros((*coords.shape[:2], 2), dtype=np.float64)
    for frame in range(coords.shape[0]):
        if frame == 0:
            left = 0
            right = 1
        elif frame == coords.shape[0] - 1:
            left = coords.shape[0] - 2
            right = coords.shape[0] - 1
        else:
            left = frame - 1
            right = frame + 1

        dt = times[right] - times[left]
        if dt <= 0.0:
            raise ValueError("steps must increase over time")
        dx = minimum_image(coords[right, :, 0] - coords[left, :, 0], lx)
        dtheta = minimum_image(coords[right, :, 1] - coords[left, :, 1], 2.0 * np.pi)
        velocities[frame, :, 0] = dx / dt
        velocities[frame, :, 1] = active.radius * dtheta / dt
    return np.ascontiguousarray(velocities)


def particle_shell_source(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> np.ndarray:
    mask = np.asarray(active.shell_mask, dtype=np.float64)
    if mask.shape[0] < 2:
        return np.zeros((*mask.shape, 2), dtype=np.float64)

    times = (np.asarray(active.steps, dtype=np.float64) - float(active.steps[0])) * float(
        config.settings.timestep
    )
    source = np.zeros((*mask.shape, 2), dtype=np.float64)
    for frame in range(mask.shape[0]):
        if frame == 0:
            left = 0
            right = 1
        elif frame == mask.shape[0] - 1:
            left = mask.shape[0] - 2
            right = mask.shape[0] - 1
        else:
            left = frame - 1
            right = frame + 1

        dt = times[right] - times[left]
        if dt <= 0.0:
            raise ValueError("steps must increase over time")
        source[frame, :, 0] = (mask[right] - mask[left]) / dt
    return np.ascontiguousarray(source)


def spectral_active_fields(
    active: ActiveMatterArrays,
    coarse: dict[str, np.ndarray],
    config: RhoFittingConfig,
) -> dict[str, np.ndarray]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")

    _progress("chebyshev rho 1/4")
    rho_time = chebyshev_filter_and_derivative(
        coarse["rho"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    _progress("chebyshev P_density 2/4")
    p_time = chebyshev_filter_and_derivative(
        coarse["P_density"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    _progress("chebyshev J_density 3/4")
    j_time = chebyshev_filter_and_derivative(
        coarse["J_density"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    _progress("chebyshev S_cross 4/4")
    source_time = chebyshev_filter_and_derivative(
        coarse["S_cross"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    _validate_temporal_alignment(rho_time, p_time, j_time, source_time)
    power = temporal_power_spectrum(
        rho_time.coefficients,
        p_time.coefficients[..., 0],
        p_time.coefficients[..., 1],
        j_time.coefficients[..., 0],
        j_time.coefficients[..., 1],
        source_time.coefficients,
    )
    min_rho = float(np.min(rho_time.filtered))
    if min_rho < -1.0e-8:
        _progress(f"warning: filtered rho has negative values; min={min_rho:.6g}")
    return {
        "rho": rho_time.filtered,
        "P_density": p_time.filtered,
        "J_density": j_time.filtered,
        "S_cross": source_time.filtered,
        "partial_t_rho": rho_time.derivative,
        "temporal_power": power,
        "cheb_times": rho_time.times,
        "cheb_scaled_times": rho_time.scaled_times,
    }


def _validate_temporal_alignment(
    rho_time: ChebyshevTimeResult,
    p_time: ChebyshevTimeResult,
    j_time: ChebyshevTimeResult,
    source_time: ChebyshevTimeResult,
) -> None:
    if rho_time.filtered.shape != rho_time.derivative.shape:
        raise ValueError("rho and partial_t_rho must share the same time-space shape")
    if p_time.filtered.shape[:3] != rho_time.filtered.shape:
        raise ValueError("rho, P_density, and partial_t_rho must share frame/grid axes")
    if j_time.filtered.shape[:3] != rho_time.filtered.shape:
        raise ValueError("rho, J_density, and partial_t_rho must share frame/grid axes")
    if source_time.filtered.shape != rho_time.filtered.shape:
        raise ValueError("rho, S_cross, and partial_t_rho must share frame/grid axes")
    if not np.array_equal(rho_time.times, p_time.times):
        raise ValueError("rho and P_density Chebyshev times are not aligned")
    if not np.array_equal(rho_time.times, j_time.times):
        raise ValueError("rho and J_density Chebyshev times are not aligned")
    if not np.array_equal(rho_time.times, source_time.times):
        raise ValueError("rho and S_cross Chebyshev times are not aligned")
    if not np.array_equal(rho_time.scaled_times, p_time.scaled_times):
        raise ValueError("rho and P_density scaled Chebyshev times are not aligned")
    if not np.array_equal(rho_time.scaled_times, j_time.scaled_times):
        raise ValueError("rho and J_density scaled Chebyshev times are not aligned")
    if not np.array_equal(rho_time.scaled_times, source_time.scaled_times):
        raise ValueError("rho and S_cross scaled Chebyshev times are not aligned")


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
    valid_mask = (
        np.isfinite(spectral["rho"])
        & np.isfinite(spectral["partial_t_rho"])
        & np.all(np.isfinite(spectral["P_density"]), axis=-1)
        & np.all(np.isfinite(spectral["J_density"]), axis=-1)
        & np.isfinite(spectral["S_cross"])
    )
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
            np.ascontiguousarray(spectral["P_density"], dtype=np.float64),
            np.ascontiguousarray(spectral["J_density"], dtype=np.float64),
            np.ascontiguousarray(spectral["S_cross"], dtype=np.float64),
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
    coarse: dict[str, np.ndarray],
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
        raw_rho=coarse["rho"],
        raw_P_density=coarse["P_density"],
        raw_J_density=coarse["J_density"],
        raw_S_cross=coarse["S_cross"],
        rho=spectral["rho"],
        P_density=spectral["P_density"],
        J_density=spectral["J_density"],
        S_cross=spectral["S_cross"],
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
        tau_index=np.asarray(-1 if fit.tau_index is None else fit.tau_index),
        active=fit.active,
        raw_correlations=fit.raw_correlations,
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
        lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
        write_density_plots(
            config.output_dir,
            config.case_id,
            fit,
            fit_payload["y_density"],
            spectral["rho"],
            spectral["P_density"],
            spectral["J_density"],
            spectral["S_cross"],
            spectral["partial_t_rho"],
            spectral["temporal_power"],
            config.settings.cheb_cutoff,
            lx,
            ly,
        )
        _progress("plots written")
    return cache_path, report_path


def write_density_report_from_cache(config: RhoFittingConfig) -> RhoFittingResult:
    cache_path = config.output_dir / f"{config.case_id}_fit_result.npz"
    report_path = config.output_dir / f"{config.case_id}_rho_fitting_report.md"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    with np.load(cache_path, allow_pickle=False) as data:
        metadata = _metadata_from_cache(data)
        fit = _fit_from_cache(data)
        warnings = _strings_from_array(data["warnings"]) if "warnings" in data.files else ()
        sample_count = int(np.asarray(data["sample_indices"]).shape[0])
        rho_shape = tuple(int(value) for value in np.asarray(data["rho"]).shape)

    grid_shape = _metadata_grid_shape(metadata, rho_shape)
    frames = int(rho_shape[0]) if rho_shape else int(metadata.get("frames", 0))
    lines = density_report_lines(
        case_id=str(metadata.get("case_id", config.case_id)),
        nd=sample_count,
        frames=frames,
        grid_shape=grid_shape,
        sigma=float(metadata.get("sigma", config.settings.sigma)),
        cheb_cutoff=int(metadata.get("cheb_cutoff", config.settings.cheb_cutoff)),
        n_rho_power=int(metadata.get("n_rho_power", config.settings.n_rho_power)),
        n_rho_lap_power=int(metadata.get("n_rho_lap_power", config.settings.n_rho_lap_power)),
        fit=fit,
        warnings=warnings,
    )
    write_report(report_path, lines, overwrite=config.overwrite)
    _progress(f"report written {report_path}")
    active_terms = tuple(name for name, active_flag in zip(fit.names, fit.active, strict=True) if active_flag)
    return RhoFittingResult(
        case_id=config.case_id,
        status="report-only",
        nd=sample_count,
        frames=frames,
        particles=None,
        grid_shape=grid_shape,
        sample_count=sample_count,
        active_terms=active_terms,
        cache_path=cache_path,
        report_path=report_path,
    )


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
        flux_cylindrical=None
        if active.flux_cylindrical is None
        else active.flux_cylindrical[:n],
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


def _fit_from_cache(data: np.lib.npyio.NpzFile) -> StabilityResult:
    names = _strings_from_array(data["term_names"])
    labels = _strings_from_array(data["term_labels"])
    importance = np.asarray(data["importance"], dtype=np.float64)
    importance_path = np.asarray(data["importance_path"], dtype=np.float64)
    tau_values = np.asarray(data["tau_values"], dtype=np.float64)
    coefficients = np.asarray(data["coefficients"], dtype=np.float64)
    active = np.asarray(data["active"], dtype=bool)
    y_pred = np.asarray(data["y_pred"], dtype=np.float64)
    y = np.asarray(data["y_density"], dtype=np.float64)
    raw_correlations = _raw_correlations_from_cache(data)
    tau_index = _tau_index_from_cache(data, importance_path, importance)
    rmse = float(np.asarray(data["rmse"]).reshape(-1)[0])
    r2 = float(np.asarray(data["r2"]).reshape(-1)[0])
    return StabilityResult(
        names=names,
        labels=labels,
        coefficients=coefficients,
        importance=importance,
        raw_correlations=raw_correlations,
        importance_path=importance_path,
        tau_values=tau_values,
        active=active,
        tau_index=tau_index,
        y_pred=y_pred,
        rmse=rmse,
        r2=r2,
    )


def _raw_correlations_from_cache(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "raw_correlations" in data.files:
        return np.asarray(data["raw_correlations"], dtype=np.float64)
    if "X_density" not in data.files or "y_density" not in data.files:
        raise ValueError("cache is missing X_density/y_density needed for raw correlations")
    return raw_feature_correlations(
        np.asarray(data["X_density"], dtype=np.float64),
        np.asarray(data["y_density"], dtype=np.float64),
    )


def _tau_index_from_cache(
    data: np.lib.npyio.NpzFile,
    importance_path: np.ndarray,
    importance: np.ndarray,
) -> int | None:
    if "tau_index" in data.files:
        value = int(np.asarray(data["tau_index"]).reshape(-1)[0])
        return None if value < 0 else value
    for index, row in enumerate(importance_path):
        if row.shape == importance.shape and np.allclose(row, importance, equal_nan=True):
            return int(index)
    return None


def _metadata_from_cache(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in data.files:
        return {}
    return json.loads(str(np.asarray(data["metadata_json"]).item()))


def _metadata_grid_shape(
    metadata: dict[str, object],
    rho_shape: tuple[int, ...],
) -> tuple[int, int]:
    if "Nx" in metadata and "Ny" in metadata:
        return int(metadata["Nx"]), int(metadata["Ny"])
    if len(rho_shape) >= 3:
        return int(rho_shape[1]), int(rho_shape[2])
    return 0, 0


def _strings_from_array(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(values).reshape(-1))


def _progress(message: str) -> None:
    print(f"[rho_fitting] {message}", flush=True)
