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
from .geometry import surface_lengths
from .io import ActiveMatterArrays, load_active_matter_npz
from .library import mechanical_labels
from .outputs import mechanical_report_lines, write_mechanical_outputs
from .particles import particle_surface_velocities, particle_tangent_directions
from .plots import write_temporal_power_plots
from .regression import StabilityResult, stability_selection

Array: TypeAlias = NDArray[Any]


class MechanicalLibraries(TypedDict):
    """Mechanical candidate libraries and names returned by the Rust extension."""

    Y_rho: Array
    Y_P: Array
    Y_Q: Array
    Y_rho_names: tuple[str, ...]
    Y_P_names: tuple[str, ...]
    Y_Q_names: tuple[str, ...]


Y_RHO_NAMES = ("grad_rho", "grad_lap_rho", "Q_dot_grad_rho")
Y_P_NAMES = ("A", "rho_A", "psi6sq_A", "grad_P", "rho_grad_P", "grad_lap_P")
Y_Q_NAMES = (
    "Ubar_P_dot_alpha_traceless",
    "grad_P_symmetric_traceless",
    "grad_Q",
    "rho_grad_Q",
    "grad_lap_Q",
)


class MechanicalFitPayload(TypedDict):
    """Mechanical fit matrices, row indices, libraries, and fitted model results."""

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


@dataclass(frozen=True)
class RhoFittingResult:
    """Summary of one rho-fitting workflow run and its generated artifacts."""

    case_id: str
    status: str
    nd: int
    frames: int
    particles: int
    grid_shape: tuple[int, int, int]
    sample_count: int
    active_terms: tuple[str, ...]
    cache_path: Path
    report_path: Path

    def summary(self) -> str:
        """Return a single-line status summary suitable for CLI output."""
        active = ",".join(self.active_terms) if self.active_terms else "none"
        return (
            f"[rho_fitting] case={self.case_id} status={self.status} nd={self.nd} "
            f"frames={self.frames} particles={self.particles} grid={self.grid_shape} "
            f"samples={self.sample_count} active={active}"
        )


def run(config: RhoFittingConfig) -> RhoFittingResult:
    """Run the rho-fitting workflow for one configured case.

    Parameters:
        config: Case id, output controls, and numerical settings for coarse-graining,
            filtering, sampling, and regression.

    Returns:
        ``RhoFittingResult`` describing whether correlations or full mechanical fits were
        produced and where the cache/report were written.

    Examples:
        ``result = run(RhoFittingConfig(case_id="radius_15D", fit_only=True))``

    Edge cases:
        ``fit_only`` requires an existing mechanical cache; it skips coarse-graining but
        still rewrites fit outputs with overwrite enabled.
    """
    settings = _settings(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    active = load_active_matter_npz(config.paths.active_fields_path, radius_from_case_id(config.case_id))
    r_edges, r_centers = radial_grid(active, settings)
    _log(
        f"loaded frames={active.coords.shape[0]} particles={active.coords.shape[1]} "
        f"grid=({active.x_centers.size}, {active.theta_centers.size}, {r_centers.size})"
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
            grid_shape=(active.x_centers.size, active.theta_centers.size, r_centers.size),
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
        grid_shape=(active.x_centers.size, active.theta_centers.size, r_centers.size),
        sample_count=fit_payload["sample_indices"].shape[0],
        active_terms=active_terms,
        cache_path=cache_path,
        report_path=report_path,
    )


def _settings(config: RhoFittingConfig) -> NumericalSettings:
    """Return initialized numerical settings from a validated config."""
    assert config.settings is not None, "rho fitting settings were not initialized"
    return config.settings


def _overwrite_config(config: RhoFittingConfig) -> RhoFittingConfig:
    """Return a copy of a config that permits replacing generated output files."""
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
    """Return the standard mechanical fit NPZ cache path for a case."""
    return config.output_dir / f"{config.case_id}_fit_result.npz"


def radial_grid(active: ActiveMatterArrays, settings: NumericalSettings) -> tuple[Array, Array]:
    """Return global radial bin edges and centers for cylindrical 3D fitting."""
    if settings.radial_range is None:
        radii = np.asarray(active.coords[..., 2], dtype=np.float64)
        finite = radii[np.isfinite(radii)]
        assert finite.size > 0, "cannot infer radial grid from non-finite coordinates"
        r_min = float(np.min(finite))
        r_max = float(np.max(finite))
        if r_min == r_max:
            padding = max(abs(r_min), 1.0) * 1.0e-6
            r_min -= padding
            r_max += padding
    else:
        r_min, r_max = settings.radial_range
    assert r_min >= 0.0 and r_min < r_max, "radial grid must be non-negative and ordered"
    edges = np.linspace(r_min, r_max, settings.radial_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    assert np.all(centers > 0.0), "radial centers must be positive for cylindrical derivatives"
    return edges, centers


def _load_mechanical_fit_cache(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Load raw and filtered mechanical fields from an existing fit cache.

    Parameters:
        active: Current active-matter input arrays used to validate cached grid shapes.
        config: Run configuration whose output directory and case id locate the cache.

    Returns:
        ``(coarse, spectral)`` dictionaries matching the outputs normally produced by
        coarse-graining and Chebyshev filtering.

    Edge cases:
        The cache must contain both raw and filtered fields; missing keys are reported
        before shape validation.
    """
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
            "r_edges",
            "r_centers",
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
            "r_edges": np.asarray(cache["r_edges"]),
            "r_centers": np.asarray(cache["r_centers"]),
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
    """Validate cached mechanical field shapes against the active input grid."""
    settings = _settings(config)
    _, r_centers = radial_grid(active, settings)
    grid_shape = (active.coords.shape[0], active.x_centers.size, active.theta_centers.size, r_centers.size)
    assert np.allclose(coarse["r_centers"], r_centers), "cached radial centers do not match current settings"
    assert coarse["rho"].shape == grid_shape, "cached raw rho shape does not match active grid"
    assert spectral["rho"].shape == grid_shape, "cached rho shape does not match active grid"
    assert spectral["P"].shape == grid_shape + (3,), "cached P must use 3D orientation axes"
    assert spectral["Q"].shape == grid_shape + (3, 3), "cached Q must use 3D orientation axes"
    assert spectral["A"].shape == grid_shape + (3, 3), "cached A must use 3D orientation axes"
    assert spectral["J_rho"].shape == grid_shape + (3,), "cached J_rho must use 3D cylinder-basis flux axes"
    assert spectral["J_P"].shape == grid_shape + (3, 3), "cached J_P must use 3D flux and 3D moment axes"
    assert spectral["J_Q"].shape == grid_shape + (3, 3, 3), "cached J_Q must use 3D flux and 3D moment axes"
    assert spectral["psi6_sq"].shape == grid_shape, "cached psi6_sq shape does not match active grid"
    assert spectral["Y_rho"].shape == grid_shape + (3,), "cached Y_rho shape is invalid"
    assert spectral["Y_P"].shape == grid_shape + (3, 3), "cached Y_P shape is invalid"
    assert spectral["Y_Q"].shape == grid_shape + (3, 3, 3), "cached Y_Q shape is invalid"
    assert int(np.asarray(spectral["cheb_times"]).shape[0]) == active.steps.size, "cached time axis length is invalid"
    assert settings.cheb_cutoff > 0, "cheb_cutoff must be positive"


def _core() -> Any:
    """Return the compiled Rust extension or fail with a clear build message."""
    assert _rho_fitting_core is not None, "rho_fitting extension is not built"
    return _rho_fitting_core


def coarse_grain_active_fields(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> dict[str, Array]:
    """Build raw coarse-grained density, moments, currents, and mechanical fields.

    Parameters:
        active: Particle positions, masks, directions, and grid metadata.
        config: Numerical settings and paths for hexatic order and fallback orientations.

    Returns:
        Dictionary of raw grid fields including ``rho``, ``P``, ``Q``, ``A``, ``psi6_sq``,
        and current tensors with surface flux axes.

    Examples:
        ``coarse = coarse_grain_active_fields(active, config)``

    Edge cases:
        Uses all particles rather than ``shell_mask`` for the current mechanical pass, and
        expects hexatic-order text rows to cover every frame/particle pair.
    """
    core = _core()
    settings = _settings(config)
    lx, _ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    theta_period = float(active.theta_edges[-1] - active.theta_edges[0])
    r_edges, r_centers = radial_grid(active, settings)
    coords = np.ascontiguousarray(active.coords, dtype=np.float64)
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
        np.ascontiguousarray(active.theta_centers, dtype=np.float64),
        np.ascontiguousarray(r_centers, dtype=np.float64),
        lx,
        theta_period,
        float(settings.sigma),
        float(settings.gamma),
        float(settings.u0),
    )
    out = {name: np.asarray(value) for name, value in fields.items()}
    out["J_density"] = out["J_rho"]
    out["r_edges"] = r_edges
    out["r_centers"] = r_centers
    return out

def spectral_active_fields(
    active: ActiveMatterArrays,
    coarse: dict[str, Array],
    config: RhoFittingConfig,
) -> dict[str, Array]:
    """Filter raw coarse fields over time and build spectral targets/libraries.

    Parameters:
        active: Input arrays providing time steps and grid geometry.
        coarse: Raw coarse-grained field dictionary from ``coarse_grain_active_fields``.
        config: Numerical settings controlling Chebyshev filtering and physical constants.

    Returns:
        Dictionary of filtered fields, mechanical targets ``Y_*``, density flux candidates,
        ``partial_t_rho``, and Chebyshev diagnostic arrays.

    Edge cases:
        Negative filtered density is only logged as a warning because polynomial filtering
        can overshoot near sharp temporal features.
    """
    core = _core()
    settings = _settings(config)
    rho_time = chebyshev_filter_and_derivative(
        coarse["rho"],
        active.steps,
        settings.timestep,
        settings.cheb_cutoff,
    )
    coefficients = [rho_time.coefficients]
    p_time = _filter_coarse_field("P", coarse, active, settings, rho_time)
    q_time = _filter_coarse_field("Q", coarse, active, settings, rho_time)
    a_time = _filter_coarse_field("A", coarse, active, settings, rho_time)
    j_rho_time = _filter_coarse_field("J_rho", coarse, active, settings, rho_time)
    j_p_time = _filter_coarse_field("J_P", coarse, active, settings, rho_time)
    j_q_time = _filter_coarse_field("J_Q", coarse, active, settings, rho_time)
    psi6_sq_time = _filter_coarse_field("psi6_sq", coarse, active, settings, rho_time)
    coefficients.extend(
        (
            p_time.coefficients,
            q_time.coefficients,
            a_time.coefficients,
            j_rho_time.coefficients,
            j_p_time.coefficients,
            j_q_time.coefficients,
            psi6_sq_time.coefficients,
        )
    )
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    targets = core.build_mechanical_targets(
        np.ascontiguousarray(p_time.filtered, dtype=np.float64),
        np.ascontiguousarray(j_rho_time.filtered, dtype=np.float64),
        np.ascontiguousarray(j_p_time.filtered, dtype=np.float64),
        np.ascontiguousarray(j_q_time.filtered, dtype=np.float64),
        float(settings.gamma),
        float(settings.u0),
    )
    min_rho = float(np.min(rho_time.filtered))
    if min_rho < -1.0e-8:
        _log(f"warning: filtered rho has negative values; min={min_rho:.6g}")
    return {
        "rho": rho_time.filtered,
        "P": p_time.filtered,
        "Q": q_time.filtered,
        "A": a_time.filtered,
        "psi6_sq": psi6_sq_time.filtered,
        "J_density": j_rho_time.filtered,
        "J_rho": j_rho_time.filtered,
        "J_P": j_p_time.filtered,
        "J_Q": j_q_time.filtered,
        "Y_rho": np.asarray(targets["Y_rho"]),
        "Y_P": np.asarray(targets["Y_P"]),
        "Y_Q": np.asarray(targets["Y_Q"]),
        "partial_t_rho": rho_time.derivative,
        "temporal_power": temporal_power_spectrum(*coefficients),
        "cheb_times": rho_time.times,
        "cheb_scaled_times": rho_time.scaled_times,
    }


def _filter_coarse_field(
    name: str,
    coarse: dict[str, Array],
    active: ActiveMatterArrays,
    settings: NumericalSettings,
    reference: ChebyshevTimeResult,
) -> ChebyshevTimeResult:
    """Filter one coarse field and assert it shares the reference time grid."""
    result = chebyshev_filter_and_derivative(
        coarse[name],
        active.steps,
        settings.timestep,
        settings.cheb_cutoff,
    )
    _validate_temporal_alignment(reference, result)
    return result


def fit_mechanical(
    active: ActiveMatterArrays,
    spectral: dict[str, Array],
    config: RhoFittingConfig,
) -> MechanicalFitPayload:
    """Fit divergence-aware closures for density, P, and Q mechanical flux targets.

    Parameters:
        active: Input grid metadata used for periodic divergence lengths.
        spectral: Filtered fields and target arrays from ``spectral_active_fields``.
        config: Sampling, weighting, and sparse-regression settings.

    Returns:
        Payload containing sampled rows, fitted models, full libraries, term names, and
        the reconstructed ``F_rho`` prediction field.

    Examples:
        ``payload = fit_mechanical(active, spectral, config)``

    Edge cases:
        Primary fit metrics are computed on divergence rows; flux rows are auxiliary and
        may be weighted into the fit without becoming the reported PDE metric.
    """
    sample_indices = _mechanical_sample_indices(spectral, config)
    geometry = _cylindrical_geometry(active, config)
    y_rho_fit, y_rho_rows, y_rho_index = _fit_divergence_primary_target("Y_rho", spectral, sample_indices, config, geometry)
    y_p_fit, y_p_rows, y_p_index = _fit_divergence_primary_target("Y_P", spectral, sample_indices, config, geometry)
    y_q_fit, y_q_rows, y_q_index = _fit_divergence_primary_target("Y_Q", spectral, sample_indices, config, geometry)
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
        "Y_rho_library": np.empty((0,), dtype=np.float64),
        "Y_P_library": np.empty((0,), dtype=np.float64),
        "Y_Q_library": np.empty((0,), dtype=np.float64),
        "Y_rho_names": np.asarray(Y_RHO_NAMES),
        "Y_P_names": np.asarray(Y_P_NAMES),
        "Y_Q_names": np.asarray(Y_Q_NAMES),
        "F_rho_prediction": np.empty((0,), dtype=np.float64),
    }


def print_mechanical_correlations(
    active: ActiveMatterArrays,
    spectral: dict[str, Array],
    config: RhoFittingConfig,
) -> Array:
    """Print raw target-feature correlations for each mechanical library."""
    sample_indices = _mechanical_sample_indices(spectral, config)
    geometry = _cylindrical_geometry(active, config)
    for target_name, names in (
        ("Y_rho", Y_RHO_NAMES),
        ("Y_P", Y_P_NAMES),
        ("Y_Q", Y_Q_NAMES),
    ):
        y, X, _row_index = _sample_divergence_design(target_name, spectral, sample_indices, geometry)
        labels = mechanical_labels(names)
        _log(f"{target_name} correlation rows={X.shape[0]} terms={', '.join(names)}")
        _log("raw feature correlations with target")
        for label, correlation in zip(labels, _raw_feature_correlations(X, y), strict=True):
            _log(f"  {label}: {_format_correlation(correlation)}")
    return sample_indices


def _mechanical_sample_indices(spectral: dict[str, Array], config: RhoFittingConfig) -> Array:
    """Sample valid ``(time, x, theta, r)`` grid indices for mechanical regression."""
    settings = _settings(config)
    valid_mask = _mechanical_valid_mask(spectral)
    flat_valid = np.flatnonzero(valid_mask.reshape(-1))
    assert flat_valid.size > 0, "no valid rows to sample"
    rng = np.random.default_rng(settings.seed)
    if settings.replace:
        chosen = rng.choice(flat_valid, size=settings.nd, replace=True)
    else:
        chosen = rng.choice(flat_valid, size=min(settings.nd, flat_valid.size), replace=False)
    unraveled = np.unravel_index(chosen, valid_mask.shape)
    sample_indices = np.stack(unraveled, axis=1).astype(np.int64, copy=False)
    _validate_sample_count(settings.nd, sample_indices.shape[0], settings.replace)
    return sample_indices


def _cylindrical_geometry(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> tuple[float, float, Array]:
    """Return ``(lx, theta_period, r_centers)`` for 3D cylindrical operators."""
    settings = _settings(config)
    lx, _ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    theta_period = float(active.theta_edges[-1] - active.theta_edges[0])
    _r_edges, r_centers = radial_grid(active, settings)
    return lx, theta_period, r_centers


def _mechanical_libraries(
    spectral: dict[str, Array],
    lx: float,
    ly: float,
) -> MechanicalLibraries:
    """Build mechanical candidate libraries from filtered fields using the Rust core."""
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


def _fit_divergence_primary_target(
    target_name: str,
    spectral: dict[str, Array],
    sample_indices: Array,
    config: RhoFittingConfig,
    geometry: tuple[float, float, Array],
) -> tuple[StabilityResult, Array, Array]:
    """Fit one mechanical target with divergence rows as the primary evaluation target.

    Parameters:
        target_name: Report label for progress logging.
        spectral: Filtered fields and target fluxes.
        sample_indices: Sampled ``(T, Nx, Ntheta, Nr)`` locations.
        config: Regression settings including optional flux-row weight.
        geometry: Cylindrical grid geometry.

    Returns:
        ``(fit, rows, row_index)`` where ``fit`` is evaluated on divergence rows and
        ``rows``/``row_index`` preserve the sampled flux-row payload for reports.

    Edge cases:
        Setting ``mechanical_flux_weight`` to zero removes flux rows from the fit entirely;
        finite filtering is applied separately to flux and divergence rows.
    """
    settings = _settings(config)
    names = _target_names(target_name)
    y_div, X_div, row_index = _sample_divergence_design(target_name, spectral, sample_indices, geometry)
    rows = np.column_stack((y_div, X_div))
    labels = mechanical_labels(names)
    _log(
        f"{target_name} divergence fit rows={X_div.shape[0]} flux_rows=0 "
        f"flux_weight=0 terms={', '.join(names)}"
    )
    fit = stability_selection(
        X_div,
        y_div,
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
        non_positive_names=("grad_rho", "grad_P", "grad_Q"),
    )
    return fit, rows, row_index


def _target_names(target_name: str) -> tuple[str, ...]:
    """Return coefficient names for one mechanical target."""
    if target_name == "Y_rho":
        return Y_RHO_NAMES
    if target_name == "Y_P":
        return Y_P_NAMES
    if target_name == "Y_Q":
        return Y_Q_NAMES
    raise AssertionError(f"unknown mechanical target: {target_name}")


def _sample_divergence_design(
    target_name: str,
    spectral: dict[str, Array],
    sample_indices: Array,
    geometry: tuple[float, float, Array],
) -> tuple[Array, Array, Array]:
    """Sample divergence target rows and candidate divergence columns."""
    lx, theta_period, r_centers = geometry
    target_div = _divergence_cylindrical_flux(spectral[target_name], lx, theta_period, r_centers)
    y = _sample_field_components(target_div, sample_indices)
    row_index = _component_row_index(sample_indices, target_div.shape[4:])
    columns = [
        _sample_field_components(term, sample_indices)
        for term in _divergence_candidate_terms(target_name, spectral, lx, theta_period, r_centers)
    ]
    X = np.stack(columns, axis=1)
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return y[finite], X[finite], row_index[finite]


def _divergence_candidate_terms(
    target_name: str,
    spectral: dict[str, Array],
    lx: float,
    theta_period: float,
    r_centers: Array,
) -> list[Array]:
    """Return divergence-space candidate fields for one target."""
    rho = spectral["rho"]
    p = spectral["P"]
    q = spectral["Q"]
    a = spectral["A"]
    psi6_sq = spectral["psi6_sq"]
    if target_name == "Y_rho":
        grad_rho = _gradient_cylindrical_scalar(rho, lx, theta_period, r_centers)
        lap_rho = _laplacian_cylindrical_scalar(rho, lx, theta_period, r_centers)
        grad_lap_rho = _gradient_cylindrical_scalar(lap_rho, lx, theta_period, r_centers)
        q_grad_rho = np.einsum("...ka,...a->...k", q, grad_rho)
        return [
            _divergence_cylindrical_flux(grad_rho, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(grad_lap_rho, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(q_grad_rho, lx, theta_period, r_centers),
        ]
    if target_name == "Y_P":
        grad_p = _gradient_cylindrical_vector(p, lx, theta_period, r_centers)
        lap_p = _laplacian_cylindrical_vector(p, lx, theta_period, r_centers)
        grad_lap_p = _gradient_cylindrical_vector(lap_p, lx, theta_period, r_centers)
        return [
            _divergence_cylindrical_flux(a, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(rho[..., None, None] * a, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(psi6_sq[..., None, None] * a, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(grad_p, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(rho[..., None, None] * grad_p, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(grad_lap_p, lx, theta_period, r_centers),
        ]
    if target_name == "Y_Q":
        grad_q = _gradient_cylindrical_rank2(q, lx, theta_period, r_centers)
        lap_q = _laplacian_cylindrical_rank2(q, lx, theta_period, r_centers)
        grad_lap_q = _gradient_cylindrical_rank2(lap_q, lx, theta_period, r_centers)
        ubar = _estimate_ubar(spectral["Y_P"], a)
        return [
            _divergence_cylindrical_flux(ubar[..., None, None, None] * _p_dot_alpha_traceless(p), lx, theta_period, r_centers),
            _divergence_cylindrical_flux(_grad_p_symmetric_traceless(p, lx, theta_period, r_centers), lx, theta_period, r_centers),
            _divergence_cylindrical_flux(grad_q, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(rho[..., None, None, None] * grad_q, lx, theta_period, r_centers),
            _divergence_cylindrical_flux(grad_lap_q, lx, theta_period, r_centers),
        ]
    raise AssertionError(f"unknown mechanical target: {target_name}")


def _gradient_cylindrical_scalar(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Return cylindrical gradient components `(x, theta, r)` for a scalar field."""
    values = np.asarray(values, dtype=np.float64)
    dx = lx / values.shape[1]
    dtheta = theta_period / values.shape[2]
    dr = _radial_spacing(r_centers)
    out = np.empty(values.shape + (3,), dtype=np.float64)
    out[..., 0] = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * dx)
    theta_derivative = (np.roll(values, -1, axis=2) - np.roll(values, 1, axis=2)) / (2.0 * dtheta)
    out[..., 1] = theta_derivative / r_centers[None, None, None, :]
    out[..., 2] = np.gradient(values, dr, axis=3, edge_order=1)
    return out


def _divergence_cylindrical_flux(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Return cylindrical divergence of a flux field whose axis 4 stores `(x, theta, r)`."""
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim >= 5 and values.shape[4] == 3, "flux must be (T,Nx,Ntheta,Nr,3,...)"
    dx = lx / values.shape[1]
    dtheta = theta_period / values.shape[2]
    dr = _radial_spacing(r_centers)
    flux_x = np.take(values, 0, axis=4)
    flux_theta = np.take(values, 1, axis=4)
    flux_r = np.take(values, 2, axis=4)
    div_x = (np.roll(flux_x, -1, axis=1) - np.roll(flux_x, 1, axis=1)) / (2.0 * dx)
    div_theta = (
        (np.roll(flux_theta, -1, axis=2) - np.roll(flux_theta, 1, axis=2))
        / (2.0 * dtheta)
    ) / r_centers[(None,) * 3 + (slice(None),) + (None,) * (values.ndim - 5)]
    weighted = r_centers[(None,) * 3 + (slice(None),) + (None,) * (values.ndim - 5)] * flux_r
    faces = np.zeros(weighted.shape[:3] + (weighted.shape[3] + 1,) + weighted.shape[4:], dtype=np.float64)
    faces[:, :, :, 1:-1, ...] = 0.5 * (weighted[:, :, :, :-1, ...] + weighted[:, :, :, 1:, ...])
    div_r = (faces[:, :, :, 1:, ...] - faces[:, :, :, :-1, ...]) / dr
    div_r = div_r / r_centers[(None,) * 3 + (slice(None),) + (None,) * (values.ndim - 5)]
    return div_x + div_theta + div_r


def _laplacian_cylindrical_scalar(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Return scalar cylindrical Laplacian using divergence of gradient."""
    return _divergence_cylindrical_flux(_gradient_cylindrical_scalar(values, lx, theta_period, r_centers), lx, theta_period, r_centers)


def _gradient_cylindrical_vector(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Differentiate every vector component over the 3D cylindrical grid."""
    out = np.empty(values.shape[:-1] + (3, values.shape[-1]), dtype=np.float64)
    for component in range(values.shape[-1]):
        out[..., :, component] = _gradient_cylindrical_scalar(values[..., component], lx, theta_period, r_centers)
    return out


def _gradient_cylindrical_rank2(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Differentiate every rank-2 tensor component over the 3D cylindrical grid."""
    out = np.empty(values.shape[:-2] + (3, values.shape[-2], values.shape[-1]), dtype=np.float64)
    for row in range(values.shape[-2]):
        for col in range(values.shape[-1]):
            out[..., :, row, col] = _gradient_cylindrical_scalar(values[..., row, col], lx, theta_period, r_centers)
    return out


def _laplacian_cylindrical_vector(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Apply the cylindrical scalar Laplacian to every vector component."""
    out = np.empty_like(values, dtype=np.float64)
    for component in range(values.shape[-1]):
        out[..., component] = _laplacian_cylindrical_scalar(values[..., component], lx, theta_period, r_centers)
    return out


def _laplacian_cylindrical_rank2(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Apply the cylindrical scalar Laplacian to every rank-2 component."""
    out = np.empty_like(values, dtype=np.float64)
    for row in range(values.shape[-2]):
        for col in range(values.shape[-1]):
            out[..., row, col] = _laplacian_cylindrical_scalar(values[..., row, col], lx, theta_period, r_centers)
    return out


def _estimate_ubar(y_p: Array, a: Array) -> Array:
    """Project measured/fitted P flux onto all three rows of A."""
    denominator = np.sum(a * a, axis=(-2, -1))
    numerator = np.sum(y_p * a, axis=(-2, -1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)


def _p_dot_alpha_traceless(p: Array) -> Array:
    """Build symmetric traceless P-alignment flux tensor over all 3 directions."""
    out = np.zeros(p.shape[:-1] + (3, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for k in range(3):
        for a in range(3):
            for b in range(3):
                out[..., k, a, b] = p[..., a] * float(k == b) + p[..., b] * float(k == a) - (2.0 / 3.0) * p[..., k] * identity[a, b]
    return out


def _grad_p_symmetric_traceless(p: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Build symmetric traceless gradient-of-P tensor over all 3 directions."""
    grad_p = _gradient_cylindrical_vector(p, lx, theta_period, r_centers)
    out = np.zeros(p.shape[:-1] + (3, 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for k in range(3):
        trace_part = (2.0 / 3.0) * grad_p[..., k, k]
        for a in range(3):
            for b in range(3):
                out[..., k, a, b] = grad_p[..., k, a] * float(k == b) + grad_p[..., k, b] * float(k == a) - trace_part * identity[a, b]
    return out


def _radial_spacing(r_centers: Array) -> float:
    """Return the uniform radial spacing."""
    assert r_centers.size >= 2, "at least two radial bins are required for 3D derivatives"
    spacing = np.diff(r_centers)
    assert np.allclose(spacing, spacing[0]), "radial centers must be uniformly spaced"
    return float(spacing[0])


def _sample_field_components(field: Array, sample_indices: Array) -> Array:
    """Sample all non-grid components at sampled `(T,Nx,Ntheta,Nr)` rows."""
    values = field[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2], sample_indices[:, 3]]
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _component_row_index(sample_indices: Array, component_shape: tuple[int, ...]) -> Array:
    """Build row metadata matching `_sample_field_components` flattening order."""
    component_count = int(np.prod(component_shape, dtype=np.int64)) if component_shape else 1
    rows = np.repeat(sample_indices, component_count, axis=0)
    if not component_shape:
        return rows
    component_indices = np.array(np.unravel_index(np.arange(component_count), component_shape)).T
    tiled = np.tile(component_indices, (sample_indices.shape[0], 1))
    return np.hstack((rows, tiled.astype(np.int64, copy=False)))


def _sample_component_matrix(
    target: Array,
    library: Array,
    sample_indices: Array,
) -> tuple[Array, Array, Array, Array]:
    """Sample all tensor components at grid locations into ``[target, features...]`` rows."""
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
    """Sample divergence of a flux target and all candidate flux terms."""
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
    """Compute centered periodic divergence from the first two surface flux components."""
    field = np.asarray(field, dtype=np.float64)
    assert field.ndim >= 4 and field.shape[3] >= 2, "flux field must have shape (T,Nx,Ny,>=2,...)"
    nx = field.shape[1]
    ny = field.shape[2]
    dx = lx / nx
    dy = ly / ny
    return (
        (np.roll(field[:, :, :, 0, ...], -1, axis=1) - np.roll(field[:, :, :, 0, ...], 1, axis=1)) / (2.0 * dx)
        + (np.roll(field[:, :, :, 1, ...], -1, axis=2) - np.roll(field[:, :, :, 1, ...], 1, axis=2)) / (2.0 * dy)
    )


def _raw_feature_correlations(X: Array, y: Array) -> Array:
    """Compute Pearson correlations between each feature column and the target."""
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
    """Format finite correlations compactly and preserve non-finite values as ``nan``."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _mechanical_valid_mask(spectral: dict[str, Array]) -> Array:
    """Return grid points where all mechanical target and candidate inputs are finite."""
    valid = np.isfinite(spectral["rho"])
    for name in ("P", "A", "J_rho", "J_P", "J_Q", "Y_rho", "Y_P", "Y_Q"):
        axes = tuple(range(4, spectral[name].ndim))
        valid &= np.all(np.isfinite(spectral[name]), axis=axes)
    valid &= np.isfinite(spectral["psi6_sq"])
    return valid


def _load_hexatic_abs_frames(path: Path, active: ActiveMatterArrays) -> Array:
    """Load per-particle ``|psi6|`` values from the hexatic text table.

    Parameters:
        path: Text table with frame, step, particle, and hexatic-order columns.
        active: Active-matter arrays that define expected frame steps and particle count.

    Returns:
        Dense ``(frames, particles)`` array of hexatic magnitudes.

    Edge cases:
        The table must be complete and step-aligned; missing particle rows remain NaN and
        cause validation to fail.
    """
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


def _validate_temporal_alignment(
    rho_time: ChebyshevTimeResult,
    j_time: ChebyshevTimeResult,
) -> None:
    """Assert that a filtered field uses the same time coordinates as rho."""
    assert rho_time.filtered.shape == rho_time.derivative.shape
    assert j_time.filtered.shape[:4] == rho_time.filtered.shape
    assert np.array_equal(rho_time.times, j_time.times)
    assert np.array_equal(rho_time.scaled_times, j_time.scaled_times)


def _sample_scalar(field: Array, sample_indices: Array) -> Array:
    """Sample scalar field values at ``(time, x, theta, r)`` integer indices."""
    return field[sample_indices[:, 0], sample_indices[:, 1], sample_indices[:, 2], sample_indices[:, 3]]


def _validate_sample_count(requested: int, actual: int, replace: bool) -> None:
    """Reject undersampling when sampling without replacement."""
    if not replace:
        assert actual == requested, (
            f"requested {requested} rows without replacement, but only {actual} valid rows were available"
        )


def _log(message: str) -> None:
    """Print a flushed rho-fitting progress message."""
    print(f"[rho_fitting] {message}", flush=True)
