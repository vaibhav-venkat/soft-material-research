"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from . import _rho_fitting_core, _rho_fitting_core_import_error
from .basis import chebyshev_filter_fields, temporal_power_spectrum
from .config import NumericalSettings, RhoFittingConfig, radius_from_case_id
from .field_types import MechanicalRawFields, MechanicalSpectralFields, MechanicalTargets
from .geometry import surface_lengths
from .io import ActiveMatterArrays, load_active_matter_npz
from .library import mechanical_labels
from .outputs import mechanical_report_lines, write_mechanical_outputs
from .particles import particle_surface_velocities, particle_tangent_directions
from .spectral import CylindricalSpectralOperators, barycentric_matrix, cached_cylindrical_operators, transfer_radial
from .plots import write_temporal_power_plots
from .regression import StabilityResult, stability_selection

Array: TypeAlias = NDArray[Any]


Y_RHO_NAMES = (
    "grad_rho",
    "A_dot_grad_rho",
    "P",
)
Y_P_NAMES = ("A", "psi6sq_A", "grad_P")
Y_Q_NAMES = (
    "tangential_projected_Ubar_P_alpha",
    "radial_projected_Ubar_P_alpha",
    "radial_grad_Q",
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
class TargetDesignFields:
    """One target's reusable flux and divergence fields."""

    target_flux: Array
    candidate_fluxes: tuple[Array, ...]
    target_divergence: Array
    candidate_divergences: tuple[Array, ...]


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
        _log(f"mechanical fields coarse-grained rho shape={coarse.rho.shape}")
        spectral = spectral_active_fields(active, coarse, config)
        if config.make_plots and not config.correlations_only:
            for path in write_temporal_power_plots(
                config.output_dir,
                config.case_id,
                spectral.temporal_power,
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
) -> tuple[MechanicalRawFields, MechanicalSpectralFields]:
    """Load raw and filtered mechanical fields from an existing fit cache.

    Parameters:
        active: Current active-matter input arrays used to validate cached grid shapes.
        config: Run configuration whose output directory and case id locate the cache.

    Returns:
        Typed raw and filtered field bundles matching the normal workflow outputs.

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
        coarse = MechanicalRawFields(
            rho=np.asarray(cache["raw_rho"]),
            P=np.asarray(cache["raw_P"]),
            Q=np.asarray(cache["raw_Q"]),
            A=np.asarray(cache["raw_A"]),
            psi6_sq=np.asarray(cache["raw_psi6_sq"]),
            J_rho=np.asarray(cache["raw_J_rho"]),
            J_P=np.asarray(cache["raw_J_P"]),
            J_Q=np.asarray(cache["raw_J_Q"]),
            r_edges=np.asarray(cache["r_edges"]),
            r_centers=np.asarray(cache["r_centers"]),
        )
        spectral = MechanicalSpectralFields(
            rho=np.asarray(cache["rho"]),
            P=np.asarray(cache["P"]),
            Q=np.asarray(cache["Q"]),
            A=np.asarray(cache["A"]),
            psi6_sq=np.asarray(cache["psi6_sq"]),
            J_rho=np.asarray(cache["J_rho"]),
            J_P=np.asarray(cache["J_P"]),
            J_Q=np.asarray(cache["J_Q"]),
            Y_rho=np.asarray(cache["Y_rho"]),
            Y_P=np.asarray(cache["Y_P"]),
            Y_Q=np.asarray(cache["Y_Q"]),
            partial_t_rho=np.asarray(cache["partial_t_rho"]),
            temporal_power=np.asarray(cache["temporal_power"]),
            cheb_times=np.asarray(cache["cheb_times"]),
            cheb_scaled_times=np.asarray(cache["cheb_scaled_times"]),
        )
    _validate_cached_fields(active, coarse, spectral, config)
    return coarse, spectral


def _validate_cached_fields(
    active: ActiveMatterArrays,
    coarse: MechanicalRawFields,
    spectral: MechanicalSpectralFields,
    config: RhoFittingConfig,
) -> None:
    """Validate cached mechanical field shapes against the active input grid."""
    settings = _settings(config)
    _, r_centers = radial_grid(active, settings)
    grid_shape = (active.coords.shape[0], active.x_centers.size, active.theta_centers.size, r_centers.size)
    assert np.allclose(coarse.r_centers, r_centers), "cached radial centers do not match current settings"
    assert coarse.rho.shape == grid_shape, "cached raw rho shape does not match active grid"
    assert spectral.rho.shape == grid_shape, "cached rho shape does not match active grid"
    assert spectral.P.shape == grid_shape + (3,), "cached P must use 3D orientation axes"
    assert spectral.Q.shape == grid_shape + (3, 3), "cached Q must use 3D orientation axes"
    assert spectral.A.shape == grid_shape + (3, 3), "cached A must use 3D orientation axes"
    assert spectral.J_rho.shape == grid_shape + (3,), "cached J_rho must use 3D cylinder-basis flux axes"
    assert spectral.J_P.shape == grid_shape + (3, 3), "cached J_P must use 3D flux and 3D moment axes"
    assert spectral.J_Q.shape == grid_shape + (3, 3, 3), "cached J_Q must use 3D flux and 3D moment axes"
    assert spectral.psi6_sq.shape == grid_shape, "cached psi6_sq shape does not match active grid"
    assert spectral.Y_rho.shape == grid_shape + (3,), "cached Y_rho shape is invalid"
    assert spectral.Y_P.shape == grid_shape + (3, 3), "cached Y_P shape is invalid"
    assert spectral.Y_Q.shape == grid_shape + (3, 3, 3), "cached Y_Q shape is invalid"
    assert int(np.asarray(spectral.cheb_times).shape[0]) == active.steps.size, "cached time axis length is invalid"
    assert settings.cheb_cutoff > 0, "cheb_cutoff must be positive"


def _core() -> Any:
    """Return the compiled Rust extension or fail with a clear build message."""
    if _rho_fitting_core is None:
        detail = (
            f": {_rho_fitting_core_import_error}"
            if _rho_fitting_core_import_error is not None
            else ""
        )
        raise RuntimeError(
            "rho_fitting extension is not importable"
            f"{detail}. Build/install it with `pixi run rho-fitting-build`."
        )
    return _rho_fitting_core


def coarse_grain_active_fields(
    active: ActiveMatterArrays,
    config: RhoFittingConfig,
) -> MechanicalRawFields:
    """Build raw coarse-grained density, moments, currents, and mechanical fields.

    Parameters:
        active: Particle positions, masks, directions, and grid metadata.
        config: Numerical settings and paths for hexatic order and fallback orientations.

    Returns:
        A typed raw field bundle with canonical grid and component axes.

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
    return MechanicalRawFields(
        rho=np.asarray(fields["rho"]),
        P=np.asarray(fields["P"]),
        Q=np.asarray(fields["Q"]),
        A=np.asarray(fields["A"]),
        psi6_sq=np.asarray(fields["psi6_sq"]),
        J_rho=np.asarray(fields["J_rho"]),
        J_P=np.asarray(fields["J_P"]),
        J_Q=np.asarray(fields["J_Q"]),
        r_edges=r_edges,
        r_centers=r_centers,
    )

def spectral_active_fields(
    active: ActiveMatterArrays,
    coarse: MechanicalRawFields,
    config: RhoFittingConfig,
) -> MechanicalSpectralFields:
    """Filter raw coarse fields over time and build spectral targets/libraries.

    Parameters:
        active: Input arrays providing time steps and grid geometry.
        coarse: Raw coarse-grained field dictionary from ``coarse_grain_active_fields``.
        config: Numerical settings controlling Chebyshev filtering and physical constants.

    Returns:
        A typed bundle of filtered fields, mechanical targets, and Chebyshev diagnostics.

    Edge cases:
        Negative filtered density is only logged as a warning because polynomial filtering
        can overshoot near sharp temporal features.
    """
    core = _core()
    settings = _settings(config)
    time_results = chebyshev_filter_fields(
        (
            coarse.rho,
            coarse.P,
            coarse.Q,
            coarse.A,
            coarse.J_rho,
            coarse.J_P,
            coarse.J_Q,
            coarse.psi6_sq,
        ),
        active.steps,
        settings.timestep,
        settings.cheb_cutoff,
    )
    (
        rho_time,
        p_time,
        q_time,
        a_time,
        j_rho_time,
        j_p_time,
        j_q_time,
        psi6_sq_time,
    ) = time_results
    coefficients = [rho_time.coefficients]
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
    target_payload = core.build_mechanical_targets(
        np.ascontiguousarray(p_time.filtered, dtype=np.float64),
        np.ascontiguousarray(j_rho_time.filtered, dtype=np.float64),
        np.ascontiguousarray(j_p_time.filtered, dtype=np.float64),
        np.ascontiguousarray(j_q_time.filtered, dtype=np.float64),
        float(settings.gamma),
        float(settings.u0),
    )
    targets = MechanicalTargets(
        Y_rho=np.asarray(target_payload["Y_rho"]),
        Y_P=np.asarray(target_payload["Y_P"]),
        Y_Q=np.asarray(target_payload["Y_Q"]),
    )
    min_rho = float(np.min(rho_time.filtered))
    if min_rho < -1.0e-8:
        _log(f"warning: filtered rho has negative values; min={min_rho:.6g}")
    return MechanicalSpectralFields(
        rho=rho_time.filtered,
        P=p_time.filtered,
        Q=q_time.filtered,
        A=a_time.filtered,
        psi6_sq=psi6_sq_time.filtered,
        J_rho=j_rho_time.filtered,
        J_P=j_p_time.filtered,
        J_Q=j_q_time.filtered,
        Y_rho=targets.Y_rho,
        Y_P=targets.Y_P,
        Y_Q=targets.Y_Q,
        partial_t_rho=rho_time.derivative,
        temporal_power=temporal_power_spectrum(*coefficients),
        cheb_times=rho_time.times,
        cheb_scaled_times=rho_time.scaled_times,
    )


def fit_mechanical(
    active: ActiveMatterArrays,
    spectral: MechanicalSpectralFields,
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
    _log(f"mechanical fit sampled {sample_indices.shape[0]} grid cells; starting spectral libraries")
    _log("mechanical fit target=Y_rho")
    y_rho_fit, y_rho_rows, y_rho_index = _fit_divergence_primary_target("Y_rho", spectral, sample_indices, config, geometry)
    _log("mechanical fit target=Y_P")
    y_p_fit, y_p_rows, y_p_index = _fit_divergence_primary_target("Y_P", spectral, sample_indices, config, geometry)
    _log("mechanical fit target=Y_Q")
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
    spectral: MechanicalSpectralFields,
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
        fields = _target_design_fields(target_name, spectral, geometry)
        assembled = _assemble_regression_rows(fields, sample_indices, 0.0)
        y = assembled["divergence_y"]
        X = assembled["divergence_X"]
        assert y is not None and X is not None
        labels = mechanical_labels(names)
        _log(f"{target_name} correlation rows={X.shape[0]} terms={', '.join(names)}")
        _log("raw feature correlations with target")
        for label, correlation in zip(labels, _raw_feature_correlations(X, y), strict=True):
            _log(f"  {label}: {_format_correlation(correlation)}")
    return sample_indices


def _mechanical_sample_indices(spectral: MechanicalSpectralFields, config: RhoFittingConfig) -> Array:
    """Sample valid ``(time, x, theta, r)`` grid indices for mechanical regression."""
    settings = _settings(config)
    valid_mask = _mechanical_valid_mask(spectral)
    sample_indices = np.asarray(
        _core().sample_grid_rows(
            np.ascontiguousarray(valid_mask, dtype=bool),
            int(settings.nd),
            int(settings.seed),
            bool(settings.replace),
        )
    )
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


def _fit_divergence_primary_target(
    target_name: str,
    spectral: MechanicalSpectralFields,
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
        ``rows``/``row_index`` preserve the sampled divergence payload for reports.

    Edge cases:
        Setting ``mechanical_flux_weight`` to zero removes flux rows from the fit entirely;
        finite filtering is applied separately to flux and divergence rows.
    """
    settings = _settings(config)
    names = _target_names(target_name)
    fields = _target_design_fields(target_name, spectral, geometry)
    flux_weight = float(settings.mechanical_flux_weight)
    assembled = _assemble_regression_rows(fields, sample_indices, flux_weight)
    y_div = assembled["divergence_y"]
    X_div = assembled["divergence_X"]
    row_index = assembled["row_index"]
    y_fit = assembled["y"]
    X_fit = assembled["X"]
    y_flux = assembled["flux_y"]
    X_flux = assembled["flux_X"]
    rows = assembled["divergence_rows"]
    assert (
        y_div is not None
        and X_div is not None
        and row_index is not None
        and y_fit is not None
        and X_fit is not None
        and rows is not None
    )
    labels = mechanical_labels(names)
    _log(
        f"{target_name} fit rows={X_fit.shape[0]} divergence_rows={X_div.shape[0]} "
        f"flux_rows={0 if X_flux is None else X_flux.shape[0]} "
        f"flux_weight={flux_weight:.6g} terms={', '.join(names)}"
    )
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
        non_positive_names=("grad_rho", "A_dot_grad_rho", "grad_P", "radial_grad_Q"),
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


def _target_design_fields(
    target_name: str,
    spectral: MechanicalSpectralFields,
    geometry: tuple[float, float, Array],
) -> TargetDesignFields:
    """Build each target/candidate flux and divergence exactly once."""
    lx, theta_period, r_centers = geometry
    _log(f"{target_name}: building reusable spectral flux/divergence fields")
    target_flux = getattr(spectral, target_name)
    candidate_fluxes = tuple(_candidate_flux_terms(target_name, spectral, lx, theta_period, r_centers))
    target_divergence = _divergence_cylindrical_flux(target_flux, lx, theta_period, r_centers)
    candidate_divergences = tuple(
        _divergence_cylindrical_flux(field, lx, theta_period, r_centers)
        for field in candidate_fluxes
    )
    return TargetDesignFields(target_flux, candidate_fluxes, target_divergence, candidate_divergences)


def _assemble_regression_rows(
    fields: TargetDesignFields,
    sample_indices: Array,
    flux_weight: float,
) -> dict[str, Array | None]:
    """Assemble finite divergence/flux regression rows in the Rust core."""
    core = _core()
    divergence_library = np.ascontiguousarray(
        np.stack(fields.candidate_divergences, axis=0),
        dtype=np.float64,
    )
    target_flux: Array | None = None
    flux_library: Array | None = None
    if flux_weight > 0.0:
        target_flux = np.ascontiguousarray(fields.target_flux, dtype=np.float64)
        flux_library = np.ascontiguousarray(
            np.stack(fields.candidate_fluxes, axis=0),
            dtype=np.float64,
        )
    payload = core.assemble_regression_rows(
        np.ascontiguousarray(fields.target_divergence, dtype=np.float64),
        divergence_library,
        np.ascontiguousarray(sample_indices, dtype=np.int64),
        target_flux,
        flux_library,
        flux_weight,
    )
    return {name: np.asarray(value) if value is not None else None for name, value in payload.items()}


def _candidate_flux_terms(
    target_name: str,
    spectral: MechanicalSpectralFields,
    lx: float,
    theta_period: float,
    r_centers: Array,
) -> list[Array]:
    """Return flux-space candidate fields for one target."""
    rho = spectral.rho
    p = spectral.P
    q = spectral.Q
    psi6_sq = spectral.psi6_sq
    core = _core()
    a = np.asarray(
        core.build_alignment_tensor(
            np.ascontiguousarray(rho, dtype=np.float64),
            np.ascontiguousarray(q, dtype=np.float64),
        )
    )
    if target_name == "Y_rho":
        grad_rho = _gradient_cylindrical_scalar(rho, lx, theta_period, r_centers)
        a_grad_rho = np.asarray(
            core.contract_alignment_gradient(
                np.ascontiguousarray(a, dtype=np.float64),
                np.ascontiguousarray(grad_rho, dtype=np.float64),
            )
        )
        return [
            grad_rho,
            a_grad_rho,
            p,
        ]
    if target_name == "Y_P":
        grad_p = _gradient_cylindrical_vector(p, lx, theta_period, r_centers)
        psi6_a = np.asarray(
            core.scale_by_scalar(
                np.ascontiguousarray(psi6_sq, dtype=np.float64),
                np.ascontiguousarray(a, dtype=np.float64),
            )
        )
        return [a, psi6_a, grad_p]
    if target_name == "Y_Q":
        grad_q = _gradient_cylindrical_rank2(q, lx, theta_period, r_centers)
        ubar = np.asarray(
            core.estimate_ubar(
                np.ascontiguousarray(spectral.Y_P, dtype=np.float64),
                np.ascontiguousarray(a, dtype=np.float64),
            )
        )
        p_alignment = np.asarray(
            core.build_p_alignment(np.ascontiguousarray(p, dtype=np.float64))
        )
        ubar_p_alpha = np.asarray(
            core.scale_by_scalar(
                np.ascontiguousarray(ubar, dtype=np.float64),
                np.ascontiguousarray(p_alignment, dtype=np.float64),
            )
        )
        return [
            np.asarray(core.project_flux_directions(ubar_p_alpha, 0)),
            np.asarray(core.project_flux_directions(ubar_p_alpha, 1)),
            np.asarray(core.project_flux_directions(np.ascontiguousarray(grad_q), 1)),
        ]
    raise AssertionError(f"unknown mechanical target: {target_name}")


def _gradient_cylindrical_scalar(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Return cylindrical gradient components `(x, theta, r)` for a scalar field."""
    values = np.asarray(values, dtype=np.float64)
    operators = _spectral_operators(values, lx, theta_period, r_centers)
    to_spectral, to_cache = _radial_transfer_matrices(r_centers, operators)
    spectral_values = transfer_radial(values, to_spectral, 3)
    return transfer_radial(operators.gradient_scalar_frames(spectral_values, label="gradient"), to_cache, 3)


def _divergence_cylindrical_flux(values: Array, lx: float, theta_period: float, r_centers: Array) -> Array:
    """Return cylindrical divergence of a flux field whose axis 4 stores `(x, theta, r)`."""
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim >= 5 and values.shape[4] == 3, "flux must be (T,Nx,Ntheta,Nr,3,...)"
    operators = _spectral_operators(values, lx, theta_period, r_centers)
    to_spectral, to_cache = _radial_transfer_matrices(r_centers, operators)
    spectral_values = transfer_radial(values, to_spectral, 3)
    return transfer_radial(operators.divergence_frames(spectral_values, label="divergence"), to_cache, 3)


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


def _spectral_operators(values: Array, lx: float, theta_period: float, r_centers: Array) -> CylindricalSpectralOperators:
    """Create the shared annular Shenfun operator for a time-indexed grid."""
    dr = _radial_spacing(r_centers)
    return cached_cylindrical_operators(
        lx,
        theta_period,
        float(r_centers[0] - 0.5 * dr),
        float(r_centers[-1] + 0.5 * dr),
        int(values.shape[1]),
        int(values.shape[2]),
        int(values.shape[3]),
    )


def _radial_transfer_matrices(
    r_centers: Array,
    operators: CylindricalSpectralOperators,
) -> tuple[Array, Array]:
    """Return cached-grid to Shenfun-grid and reverse radial interpolation matrices."""
    nodes = operators.radial_nodes()
    return barycentric_matrix(r_centers, nodes), barycentric_matrix(nodes, r_centers)


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


def _mechanical_valid_mask(spectral: MechanicalSpectralFields) -> Array:
    """Return grid points where all mechanical target and candidate inputs are finite."""
    fields = [
        spectral.rho,
        spectral.P,
        spectral.A,
        spectral.J_rho,
        spectral.J_P,
        spectral.J_Q,
        spectral.Y_rho,
        spectral.Y_P,
        spectral.Y_Q,
        spectral.psi6_sq,
    ]
    return np.asarray(_core().finite_grid_mask(fields), dtype=bool)


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


def _validate_sample_count(requested: int, actual: int, replace: bool) -> None:
    """Reject undersampling when sampling without replacement."""
    if not replace:
        assert actual == requested, (
            f"requested {requested} rows without replacement, but only {actual} valid rows were available"
        )


def _log(message: str) -> None:
    """Print a flushed rho-fitting progress message."""
    print(f"[rho_fitting] {message}", flush=True)
