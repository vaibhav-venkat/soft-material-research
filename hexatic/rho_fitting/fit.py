"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from . import _rho_fitting_core
from .basis import ChebyshevTimeResult, chebyshev_filter_and_derivative, temporal_power_spectrum
from .config import NumericalSettings, RhoFittingConfig, radius_from_case_id
from .geometry import surface_lengths, theta_to_y
from .io import ActiveMatterArrays, load_active_matter_npz
from .library import flux_names, mechanical_labels, term_labels, term_names
from .outputs import mechanical_report_lines, write_density_outputs, write_mechanical_outputs
from .particles import particle_surface_velocities, particle_tangent_directions
from .plots import write_temporal_power_plots
from .regression import StabilityResult, stability_selection

Array: TypeAlias = NDArray[Any]


class MechanicalLibraries(TypedDict):
    Y_rho: Array
    Y_P: Array
    Y_Q: Array
    Y_rho_names: tuple[str, ...]
    Y_P_names: tuple[str, ...]
    Y_Q_names: tuple[str, ...]


class MechanicalFitPayload(TypedDict):
    sample_indices: Array
    Y_rho_fit: StabilityResult
    Y_P_fit: StabilityResult
    Y_Q_fit: StabilityResult
    Y_rho_rows: Array
    Y_P_rows: Array
    Y_Q_rows: Array
    Y_rho_row_index: Array
    Y_P_row_index: Array
    Y_Q_row_index: Array
    Y_rho_library: Array
    Y_P_library: Array
    Y_Q_library: Array
    Y_rho_names: Array
    Y_P_names: Array
    Y_Q_names: Array
    F_rho_prediction: Array


class DensityFitPayload(TypedDict):
    X_density: Array
    y_density: Array
    sample_indices: Array
    term_names: Array
    term_labels: Array
    term_fluxes: Array
    fit: StabilityResult


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
    settings = _settings(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    active = load_active_matter_npz(config.paths.active_fields_path, radius_from_case_id(config.case_id))
    _log(
        f"loaded frames={active.coords.shape[0]} particles={active.coords.shape[1]} "
        f"grid=({active.x_centers.size}, {active.theta_centers.size})"
    )

    if config.fit_only:
        coarse, spectral = _load_mechanical_fit_cache(active, config)
        _log(f"loaded cached mechanical fields from {_mechanical_cache_path(config)}")
    else:
        coarse = coarse_grain_active_fields(active, config)
        _log(f"mechanical fields coarse-grained rho shape={coarse['rho'].shape}")
        spectral = spectral_active_fields(active, coarse, config)
        if config.make_plots and not config.correlations_only:
            for path in write_temporal_power_plots(
                config.output_dir,
                config.case_id,
                spectral["temporal_power"],
                settings.cheb_cutoff,
            ):
                _log(f"temporal power plot written {path}")

    if config.correlations_only:
        sample_indices = print_mechanical_correlations(active, spectral, config)
        return RhoFittingResult(
            case_id=config.case_id,
            status="mechanical-correlations",
            nd=settings.nd,
            frames=active.coords.shape[0],
            particles=active.coords.shape[1],
            grid_shape=(active.x_centers.size, active.theta_centers.size),
            sample_count=sample_indices.shape[0],
            active_terms=(),
            cache_path=_mechanical_cache_path(config) if config.fit_only else config.output_dir,
            report_path=config.output_dir,
        )

    fit_payload = fit_mechanical(active, spectral, config)
    output_payload = cast(Mapping[str, np.ndarray | StabilityResult], fit_payload)
    output_config = _overwrite_config(config) if config.fit_only else config
    cache_path, report_path = write_mechanical_outputs(active, coarse, spectral, output_payload, output_config)
    active_terms = tuple(
        f"{target}:{name}"
        for target, fit in (
            ("Y_rho", fit_payload["Y_rho_fit"]),
            ("Y_P", fit_payload["Y_P_fit"]),
            ("Y_Q", fit_payload["Y_Q_fit"]),
        )
        for name, keep in zip(fit.names, fit.active, strict=True)
        if keep
    )
    return RhoFittingResult(
        case_id=config.case_id,
        status="mechanical-fit",
        nd=settings.nd,
        frames=active.coords.shape[0],
        particles=active.coords.shape[1],
        grid_shape=(active.x_centers.size, active.theta_centers.size),
        sample_count=fit_payload["sample_indices"].shape[0],
        active_terms=active_terms,
        cache_path=cache_path,
        report_path=report_path,
    )


def _settings(config: RhoFittingConfig) -> NumericalSettings:
    assert config.settings is not None, "rho fitting settings were not initialized"
    return config.settings


def _overwrite_config(config: RhoFittingConfig) -> RhoFittingConfig:
    return RhoFittingConfig(
        case_id=config.case_id,
        overwrite=True,
        make_plots=config.make_plots,
        correlations_only=config.correlations_only,
        fit_only=config.fit_only,
        output_dir=config.output_dir,
        settings=config.settings,
    )


def _mechanical_cache_path(config: RhoFittingConfig) -> Path:
    return config.output_dir / f"{config.case_id}_fit_result.npz"


def _load_mechanical_fit_cache(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> tuple[dict[str, Array], dict[str, Array]]:
    path = _mechanical_cache_path(config)
    assert path.exists(), f"fit-only cache does not exist: {path}"
    with np.load(path, allow_pickle=False) as cache:
        required = (
            "raw_rho",
            "raw_P",
            "raw_Q",
            "raw_A",
            "raw_psi6_sq",
            "raw_J_rho",
            "raw_J_P",
            "raw_J_Q",
            "rho",
            "P",
            "Q",
            "A",
            "psi6_sq",
            "J_rho",
            "J_P",
            "J_Q",
            "Y_rho",
            "Y_P",
            "Y_Q",
            "J_density",
            "partial_t_rho",
            "temporal_power",
            "cheb_times",
            "cheb_scaled_times",
        )
        missing = [name for name in required if name not in cache.files]
        assert not missing, f"fit-only cache is missing fields: {', '.join(missing)}"
        coarse = {
            "rho": np.asarray(cache["raw_rho"]),
            "P": np.asarray(cache["raw_P"]),
            "Q": np.asarray(cache["raw_Q"]),
            "A": np.asarray(cache["raw_A"]),
            "psi6_sq": np.asarray(cache["raw_psi6_sq"]),
            "J_rho": np.asarray(cache["raw_J_rho"]),
            "J_P": np.asarray(cache["raw_J_P"]),
            "J_Q": np.asarray(cache["raw_J_Q"]),
        }
        coarse["J_density"] = coarse["J_rho"]
        spectral = {name: np.asarray(cache[name]) for name in required if not name.startswith("raw_")}
    _validate_cached_fields(active, coarse, spectral, config)
    return coarse, spectral


def _validate_cached_fields(
    active: ActiveMatterArrays,
    coarse: dict[str, Array],
    spectral: dict[str, Array],
    config: RhoFittingConfig,
) -> None:
    settings = _settings(config)
    grid_shape = (active.coords.shape[0], active.x_centers.size, active.theta_centers.size)
    assert coarse["rho"].shape == grid_shape, "cached raw rho shape does not match active grid"
    assert spectral["rho"].shape == grid_shape, "cached rho shape does not match active grid"
    assert spectral["P"].shape == grid_shape + (3,), "cached P must use 3D orientation axes"
    assert spectral["Q"].shape == grid_shape + (3, 3), "cached Q must use 3D orientation axes"
    assert spectral["A"].shape == grid_shape + (3, 3), "cached A must use 3D orientation axes"
    assert spectral["J_rho"].shape == grid_shape + (2,), "cached J_rho must use 2D surface flux axes"
    assert spectral["J_P"].shape == grid_shape + (2, 3), "cached J_P must use 2D flux and 3D moment axes"
    assert spectral["J_Q"].shape == grid_shape + (2, 3, 3), "cached J_Q must use 2D flux and 3D moment axes"
    assert spectral["psi6_sq"].shape == grid_shape, "cached psi6_sq shape does not match active grid"
    assert spectral["Y_rho"].shape == grid_shape + (2,), "cached Y_rho shape is invalid"
    assert spectral["Y_P"].shape == grid_shape + (2, 3), "cached Y_P shape is invalid"
    assert spectral["Y_Q"].shape == grid_shape + (2, 3, 3), "cached Y_Q shape is invalid"
    assert int(np.asarray(spectral["cheb_times"]).shape[0]) == active.steps.size, "cached time axis length is invalid"
    assert settings.cheb_cutoff > 0, "cheb_cutoff must be positive"


def _core() -> Any:
    assert _rho_fitting_core is not None, "rho_fitting extension is not built"
    return _rho_fitting_core


def coarse_grain_active_fields(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> dict[str, Array]:
    core = _core()
    settings = _settings(config)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    coords = np.ascontiguousarray(active.coords, dtype=np.float64)
    y_centers = np.ascontiguousarray(theta_to_y(active.theta_centers, active.radius), dtype=np.float64)
    all_particles = np.ones_like(active.shell_mask, dtype=bool)
    directions = particle_tangent_directions(active, config)
    velocities = particle_surface_velocities(active, config)
    psi6_abs = _load_hexatic_abs_frames(config.paths.hexatic_order_path, active)
    fields = core.build_mechanical_fields(
        coords,
        np.ascontiguousarray(directions, dtype=np.float64),
        np.ascontiguousarray(velocities, dtype=np.float64),
        np.ascontiguousarray(psi6_abs, dtype=np.float64),
        all_particles,
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        y_centers,
        lx,
        ly,
        active.radius,
        float(settings.sigma),
        float(settings.gamma),
        float(settings.u0),
    )
    out = {name: np.asarray(value) for name, value in fields.items()}
    out["J_density"] = out["J_rho"]
    return out

def spectral_active_fields(
    active: ActiveMatterArrays,
    coarse: dict[str, Array],
    config: RhoFittingConfig,
) -> dict[str, Array]:
    core = _core()
    settings = _settings(config)
    rho_time = chebyshev_filter_and_derivative(
        coarse["rho"],
        active.steps,
        settings.timestep,
        settings.cheb_cutoff,
    )
    filtered = {"rho": rho_time.filtered}
    coefficients = [rho_time.coefficients]
    for name in ("P", "Q", "A", "J_rho", "J_P", "J_Q", "psi6_sq"):
        result = chebyshev_filter_and_derivative(
            coarse[name],
            active.steps,
            settings.timestep,
            settings.cheb_cutoff,
        )
        _validate_temporal_alignment(rho_time, result)
        filtered[name] = result.filtered
        coefficients.append(result.coefficients)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    fluxes = core.build_density_fluxes(
        np.ascontiguousarray(rho_time.filtered, dtype=np.float64),
        lx,
        ly,
    )
    targets = core.build_mechanical_targets(
        np.ascontiguousarray(filtered["P"], dtype=np.float64),
        np.ascontiguousarray(filtered["J_rho"], dtype=np.float64),
        np.ascontiguousarray(filtered["J_P"], dtype=np.float64),
        np.ascontiguousarray(filtered["J_Q"], dtype=np.float64),
        float(settings.gamma),
        float(settings.u0),
    )
    min_rho = float(np.min(rho_time.filtered))
    if min_rho < -1.0e-8:
        _log(f"warning: filtered rho has negative values; min={min_rho:.6g}")
    return {
        "rho": rho_time.filtered,
        "P": filtered["P"],
        "Q": filtered["Q"],
        "A": filtered["A"],
        "psi6_sq": filtered["psi6_sq"],
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
    spectral: dict[str, Array],
    config: RhoFittingConfig,
) -> MechanicalFitPayload:
    sample_indices = _mechanical_sample_indices(spectral, config)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    libraries = _mechanical_libraries(spectral, lx, ly)
    y_rho_fit, y_rho_rows, y_rho_index = _fit_divergence_primary_target("Y_rho", spectral["Y_rho"], libraries["Y_rho"], libraries["Y_rho_names"], sample_indices, config, lx, ly)
    f_rho_pred = np.tensordot(y_rho_fit.coefficients, libraries["Y_rho"], axes=(0, 0))
    y_p_fit, y_p_rows, y_p_index = _fit_divergence_primary_target("Y_P", spectral["Y_P"], libraries["Y_P"], libraries["Y_P_names"], sample_indices, config, lx, ly)
    y_q_fit, y_q_rows, y_q_index = _fit_divergence_primary_target("Y_Q", spectral["Y_Q"], libraries["Y_Q"], libraries["Y_Q_names"], sample_indices, config, lx, ly)
    return {
        "sample_indices": sample_indices,
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


def print_mechanical_correlations(
    active: ActiveMatterArrays,
    spectral: dict[str, Array],
    config: RhoFittingConfig,
) -> Array:
    sample_indices = _mechanical_sample_indices(spectral, config)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    libraries = _mechanical_libraries(spectral, lx, ly)
    for target_name, target, library, names in (
        ("Y_rho", spectral["Y_rho"], libraries["Y_rho"], libraries["Y_rho_names"]),
        ("Y_P", spectral["Y_P"], libraries["Y_P"], libraries["Y_P_names"]),
        ("Y_Q", spectral["Y_Q"], libraries["Y_Q"], libraries["Y_Q_names"]),
    ):
        rows, _row_index, y, X = _sample_component_matrix(
            target,
            library,
            sample_indices,
        )
        labels = mechanical_labels(names)
        _log(f"{target_name} correlation rows={X.shape[0]} terms={', '.join(names)}")
        _log("raw feature correlations with target")
        for label, correlation in zip(labels, _raw_feature_correlations(X, y), strict=True):
            _log(f"  {label}: {_format_correlation(correlation)}")
        assert rows.shape[0] == y.shape[0], "sampled row payload is inconsistent"
    return sample_indices


def _mechanical_sample_indices(spectral: dict[str, Array], config: RhoFittingConfig) -> Array:
    core = _core()
    settings = _settings(config)
    valid_mask = _mechanical_valid_mask(spectral)
    sample_indices = np.asarray(
        core.sample_rows(
            np.ascontiguousarray(valid_mask),
            settings.nd,
            settings.seed,
            settings.replace,
        )
    )
    _validate_sample_count(settings.nd, sample_indices.shape[0], settings.replace)
    return sample_indices


def _mechanical_libraries(
    spectral: dict[str, Array],
    lx: float,
    ly: float,
) -> MechanicalLibraries:
    core = _core()
    libs = core.build_mechanical_libraries(
        np.ascontiguousarray(spectral["rho"], dtype=np.float64),
        np.ascontiguousarray(spectral["P"], dtype=np.float64),
        np.ascontiguousarray(spectral["Q"], dtype=np.float64),
        np.ascontiguousarray(spectral["A"], dtype=np.float64),
        np.ascontiguousarray(spectral["psi6_sq"], dtype=np.float64),
        np.ascontiguousarray(spectral["Y_P"], dtype=np.float64),
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
    target: Array,
    library: Array,
    names: tuple[str, ...],
    sample_indices: Array,
    config: RhoFittingConfig,
) -> tuple[StabilityResult, Array, Array]:
    settings = _settings(config)
    rows, row_index, y, X = _sample_component_matrix(target, library, sample_indices)
    labels = mechanical_labels(names)
    _log(f"{target_name} fit rows={X.shape[0]} terms={', '.join(names)}")
    fit = stability_selection(
        X,
        y,
        names,
        labels,
        seed=settings.seed,
        tau_count=settings.tau_count,
        tau_eps=settings.tau_eps,
        subsamples=settings.subsamples,
        importance_threshold=settings.importance_threshold,
        alpha=settings.alpha,
        max_iter=settings.stlsq_max_iter,
    )
    return fit, rows, row_index


def _fit_divergence_primary_target(
    target_name: str,
    target: Array,
    library: Array,
    names: tuple[str, ...],
    sample_indices: Array,
    config: RhoFittingConfig,
    lx: float,
    ly: float,
) -> tuple[StabilityResult, Array, Array]:
    settings = _settings(config)
    rows, row_index, y_flux, X_flux = _sample_component_matrix(target, library, sample_indices)
    y_div, X_div = _sample_divergence_matrix(target, library, sample_indices, lx, ly)
    labels = mechanical_labels(names)
    weight = float(settings.mechanical_flux_weight)
    _log(
        f"{target_name} divergence fit rows={X_div.shape[0]} flux_rows={X_flux.shape[0]} "
        f"flux_weight={weight:.6g} terms={', '.join(names)}"
    )
    if weight > 0.0:
        X_fit = np.vstack((X_div, weight * X_flux))
        y_fit = np.concatenate((y_div, weight * y_flux))
    else:
        X_fit = X_div
        y_fit = y_div
    fit = stability_selection(
        X_fit,
        y_fit,
        names,
        labels,
        seed=settings.seed,
        tau_count=settings.tau_count,
        tau_eps=settings.tau_eps,
        subsamples=settings.subsamples,
        importance_threshold=settings.importance_threshold,
        alpha=settings.alpha,
        max_iter=settings.stlsq_max_iter,
        evaluation_X=X_div,
        evaluation_y=y_div,
        auxiliary_X=X_flux,
        auxiliary_y=y_flux,
    )
    return fit, rows, row_index


def _sample_component_matrix(
    target: Array,
    library: Array,
    sample_indices: Array,
) -> tuple[Array, Array, Array, Array]:
    core = _core()
    row_payload = core.sample_component_rows(
        np.ascontiguousarray(target, dtype=np.float64),
        np.ascontiguousarray(library, dtype=np.float64),
        np.ascontiguousarray(sample_indices, dtype=np.int64),
    )
    rows = np.asarray(row_payload["rows"])
    row_index = np.asarray(row_payload["row_index"])
    y = rows[:, 0]
    X = rows[:, 1:]
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return rows[finite], row_index[finite], y[finite], X[finite]


def _sample_divergence_matrix(
    target: Array,
    library: Array,
    sample_indices: Array,
    lx: float,
    ly: float,
) -> tuple[Array, Array]:
    target_div = _divergence_surface_flux_field(target, lx, ly)
    div_terms = np.asarray([_divergence_surface_flux_field(term, lx, ly) for term in library], dtype=np.float64)
    if target_div.ndim == 3:
        y_div = _sample_scalar(target_div, sample_indices)
        X_div = np.stack([_sample_scalar(term, sample_indices) for term in div_terms], axis=1)
    else:
        _rows, _row_index, y_div, X_div = _sample_component_matrix(target_div, div_terms, sample_indices)
    finite = np.isfinite(y_div) & np.all(np.isfinite(X_div), axis=1)
    return y_div[finite], X_div[finite]


def _divergence_surface_flux_field(field: Array, lx: float, ly: float) -> Array:
    field = np.asarray(field, dtype=np.float64)
    assert field.ndim >= 4 and field.shape[3] == 2, "surface flux field must have shape (T,Nx,Ny,2,...)"
    nx = field.shape[1]
    ny = field.shape[2]
    dx = lx / nx
    dy = ly / ny
    return (
        (np.roll(field[:, :, :, 0, ...], -1, axis=1) - np.roll(field[:, :, :, 0, ...], 1, axis=1)) / (2.0 * dx)
        + (np.roll(field[:, :, :, 1, ...], -1, axis=2) - np.roll(field[:, :, :, 1, ...], 1, axis=2)) / (2.0 * dy)
    )


def _raw_feature_correlations(X: Array, y: Array) -> Array:
    y_centered = y - np.mean(y)
    y_norm = float(np.linalg.norm(y_centered))
    correlations = np.full(X.shape[1], np.nan, dtype=np.float64)
    for feature in range(X.shape[1]):
        x_centered = X[:, feature] - np.mean(X[:, feature])
        denominator = float(np.linalg.norm(x_centered) * y_norm)
        if denominator > 0.0:
            correlations[feature] = float(np.dot(x_centered, y_centered) / denominator)
    return correlations


def _format_correlation(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _mechanical_valid_mask(spectral: dict[str, Array]) -> Array:
    valid = np.isfinite(spectral["rho"])
    for name in ("P", "A", "J_rho", "J_P", "J_Q", "Y_rho", "Y_P", "Y_Q"):
        axes = tuple(range(3, spectral[name].ndim))
        valid &= np.all(np.isfinite(spectral[name]), axis=axes)
    valid &= np.isfinite(spectral["psi6_sq"])
    return valid


def _load_hexatic_abs_frames(path: Path, active: ActiveMatterArrays) -> Array:
    table = np.loadtxt(path, dtype=np.float64)
    assert table.ndim == 2 and table.shape[1] >= 6, "hexatic order table must have at least 6 columns"
    frame_indices = table[:, 0].astype(np.int64)
    table_steps = table[:, 1].astype(np.int64)
    particle_indices = table[:, 2].astype(np.int64)
    assert np.all((0 <= frame_indices) & (frame_indices < active.steps.size))
    assert np.all((0 <= particle_indices) & (particle_indices < active.coords.shape[1]))
    assert np.array_equal(table_steps, active.steps[frame_indices])
    values = np.full(active.coords.shape[:2], np.nan, dtype=np.float64)
    values[frame_indices, particle_indices] = table[:, 5]
    assert np.all(np.isfinite(values)), "hexatic order table is incomplete"
    return values


def fit_density(
    active: ActiveMatterArrays,
    spectral: dict[str, Array],
    config: RhoFittingConfig,
) -> DensityFitPayload:
    core = _core()
    settings = _settings(config)
    names = term_names()
    labels = term_labels()
    valid_mask = np.isfinite(spectral["rho"]) & np.isfinite(spectral["partial_t_rho"])
    sample_indices = np.asarray(
        core.sample_rows(
            np.ascontiguousarray(valid_mask),
            settings.nd,
            settings.seed,
            settings.replace,
        )
    )
    _validate_sample_count(settings.nd, sample_indices.shape[0], settings.replace)
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    X = np.asarray(
        core.build_density_library(
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
    _log(f"fit rows={X.shape[0]} terms={', '.join(names)}")
    fit = stability_selection(
        X,
        y,
        names,
        labels,
        seed=settings.seed,
        tau_count=settings.tau_count,
        tau_eps=settings.tau_eps,
        subsamples=settings.subsamples,
        importance_threshold=settings.importance_threshold,
        alpha=settings.alpha,
        max_iter=settings.stlsq_max_iter,
    )
    return {
        "X_density": X,
        "y_density": y,
        "sample_indices": sample_indices,
        "term_names": np.asarray(names),
        "term_labels": np.asarray(labels),
        "term_fluxes": np.asarray(flux_names()),
        "fit": fit,
    }


def _validate_temporal_alignment(
    rho_time: ChebyshevTimeResult,
    j_time: ChebyshevTimeResult,
) -> None:
    assert rho_time.filtered.shape == rho_time.derivative.shape
    assert j_time.filtered.shape[:3] == rho_time.filtered.shape
    assert np.array_equal(rho_time.times, j_time.times)
    assert np.array_equal(rho_time.scaled_times, j_time.scaled_times)


def _sample_scalar(field: Array, sample_indices: Array) -> Array:
    return field[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2]]


def _validate_sample_count(requested: int, actual: int, replace: bool) -> None:
    if not replace:
        assert actual == requested, (
            f"requested {requested} rows without replacement, but only {actual} valid rows were available"
        )


def _log(message: str) -> None:
    print(f"[rho_fitting] {message}", flush=True)
