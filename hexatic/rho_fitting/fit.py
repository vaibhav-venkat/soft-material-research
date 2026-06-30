"""Workflow orchestration for rho fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _rho_fitting_core
from .basis import chebyshev_filter_and_derivative, temporal_power_spectrum
from .config import RhoFittingConfig, radius_from_case_id
from .geometry import surface_lengths, tangential_particle_vectors, theta_to_y
from .io import (
    ActiveMatterArrays,
    load_active_matter_npz,
    load_gsd_orientations,
    validate_step_alignment,
)


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
    derivative_keys: tuple[str, ...] = ()

    def summary(self) -> str:
        summary = (
            f"[rho_fitting] case={self.case_id} status={self.status} nd={self.nd} "
            f"frames={self.frames} particles={self.particles} grid={self.grid_shape}"
        )
        if self.coarse_shape is not None:
            summary += f" rho={self.coarse_shape}"
        if self.temporal_shape is not None:
            summary += f" partial_t_rho={self.temporal_shape}"
        if self.derivative_keys:
            summary += f" derivatives={','.join(self.derivative_keys)}"
        return summary


def run(config: RhoFittingConfig) -> RhoFittingResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    active = _load_case(config)
    p_particles = _particle_vectors(config, active)
    active, p_particles = _limit_frames(config, active, p_particles)

    coarse_shape = None
    temporal_shape = None
    derivative_keys: tuple[str, ...] = ()
    if config.coarse_grain:
        coarse = coarse_grain_active_fields(active, p_particles, config.settings.sigma)
        coarse_shape = tuple(coarse["rho"].shape)
        spectral = spectral_active_fields(active, coarse, config)
        temporal_shape = tuple(spectral["partial_t_rho"].shape)
        derivative_keys = tuple(spectral["derivatives"])

    return RhoFittingResult(
        case_id=config.case_id,
        status="coarse-grained" if config.coarse_grain else "data-ready",
        nd=config.settings.nd,
        frames=active.coords.shape[0],
        particles=active.coords.shape[1],
        grid_shape=(active.x_centers.size, active.theta_centers.size),
        coarse_shape=coarse_shape,
        temporal_shape=temporal_shape,
        derivative_keys=derivative_keys,
    )


def coarse_grain_active_fields(
    active: ActiveMatterArrays,
    p_particles: np.ndarray,
    sigma: float,
) -> dict[str, np.ndarray]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")

    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    y_centers = theta_to_y(active.theta_centers, active.radius)
    result = _rho_fitting_core.coarse_grain_fields(
        np.ascontiguousarray(active.coords, dtype=np.float64),
        np.ascontiguousarray(p_particles, dtype=np.float64),
        np.ascontiguousarray(active.shell_mask),
        np.ascontiguousarray(active.x_centers, dtype=np.float64),
        np.ascontiguousarray(y_centers, dtype=np.float64),
        lx,
        ly,
        active.radius,
        float(sigma),
    )
    return {
        "rho": np.asarray(result["rho"]),
        "P_density": np.asarray(result["P_density"]),
    }


def spectral_active_fields(
    active: ActiveMatterArrays,
    coarse: dict[str, np.ndarray],
    config: RhoFittingConfig,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")

    rho_time = chebyshev_filter_and_derivative(
        coarse["rho"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    p_time = chebyshev_filter_and_derivative(
        coarse["P_density"],
        active.steps,
        config.settings.timestep,
        config.settings.cheb_cutoff,
    )
    lx, ly = surface_lengths(active.x_edges, active.theta_edges, active.radius)
    requested = ("grad_rho", "lap_rho", "div_p", "lap_p", "grad_div_p")
    derivatives = spatial_derivatives(rho_time.filtered, p_time.filtered, lx, ly, requested)
    power = temporal_power_spectrum(
        rho_time.coefficients,
        p_time.coefficients[..., 0],
        p_time.coefficients[..., 1],
    )
    return {
        "rho": rho_time.filtered,
        "P_density": p_time.filtered,
        "partial_t_rho": rho_time.derivative,
        "partial_t_P_density": p_time.derivative,
        "temporal_power": power,
        "cheb_times": rho_time.times,
        "cheb_scaled_times": rho_time.scaled_times,
        "derivatives": tuple(derivatives),
        **derivatives,
    }


def spatial_derivatives(
    rho: np.ndarray,
    p_density: np.ndarray,
    lx: float,
    ly: float,
    requested: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if _rho_fitting_core is None:
        raise RuntimeError("rho_fitting extension is not built")
    result = _rho_fitting_core.spatial_derivatives(
        np.ascontiguousarray(rho, dtype=np.float64),
        np.ascontiguousarray(p_density, dtype=np.float64),
        float(lx),
        float(ly),
        list(requested),
    )
    return {name: np.asarray(result[name]) for name in requested}


def _load_case(config: RhoFittingConfig) -> ActiveMatterArrays:
    fallback_radius = radius_from_case_id(config.case_id)
    active = load_active_matter_npz(config.paths.active_fields_path, fallback_radius)
    return active


def _particle_vectors(config: RhoFittingConfig, active: ActiveMatterArrays) -> np.ndarray:
    gsd = load_gsd_orientations(config.paths.gsd_path)
    validate_step_alignment(active, gsd)

    return tangential_particle_vectors(
        active.coords,
        direction_cylindrical=active.direction_cylindrical,
        active_direction=active.active_direction,
        orientation=gsd.orientation,
    )


def _limit_frames(
    config: RhoFittingConfig,
    active: ActiveMatterArrays,
    p_particles: np.ndarray,
) -> tuple[ActiveMatterArrays, np.ndarray]:
    if config.max_frames is None or active.coords.shape[0] <= config.max_frames:
        return active, p_particles

    n = config.max_frames
    return (
        ActiveMatterArrays(
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
        ),
        p_particles[:n],
    )
