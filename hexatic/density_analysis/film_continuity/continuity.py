from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .binning import accumulate_counts_and_sums
from .config import (
    FilmContinuityConfig,
    FilmContinuityScalars,
    scalars_from_active_fields,
)
from .fields import (
    J_film_from_face_crossings,
    S_cross,
    neg_div_J_from_face_crossings,
    partial_t_rho,
    rho_film,
)
from .io_cache import load_active_matter_fields
from .velocity import compute_velocities


@dataclass(frozen=True)
class FilmContinuityResult:
    transition_steps: np.ndarray
    dt: float
    cylinder_radius: float
    lx: float
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    area_bin: np.ndarray
    rho_film_b: np.ndarray
    partial_t_rho_film_b: np.ndarray
    J_film_b: np.ndarray
    neg_div_J_film_b: np.ndarray
    S_cross_b: np.ndarray
    residual_b: np.ndarray
    n_film_frame: np.ndarray
    film_count_per_bin_frame: np.ndarray
    film_count_per_bin_b: np.ndarray
    n_in_b: np.ndarray
    n_out_b: np.ndarray
    x_face_crossings_b: np.ndarray
    theta_face_crossings_b: np.ndarray
    total_b: np.ndarray
    sum_vx_b: np.ndarray
    sum_vy_b: np.ndarray
    residual_mean_abs: float
    residual_median_abs: float
    normalized_residual_mean_abs: float
    normalized_residual_median_abs: float
    particle_diameter: float | None = None
    pocket_radius: float | None = None

    def as_cache_arrays(self) -> dict[str, Any]:
        return {
            "transition_steps": self.transition_steps,
            "dt": self.dt,
            "cylinder_radius": self.cylinder_radius,
            "lx": self.lx,
            "x_edges": self.x_edges,
            "x_centers": self.x_centers,
            "theta_edges": self.theta_edges,
            "theta_centers": self.theta_centers,
            "A_bin": self.area_bin,
            "rho_film_b": self.rho_film_b,
            "partial_t_rho_film_b": self.partial_t_rho_film_b,
            "J_film_b": self.J_film_b,
            "neg_div_J_film_b": self.neg_div_J_film_b,
            "S_cross_b": self.S_cross_b,
            "residual_b": self.residual_b,
            "total_b": self.neg_div_J_film_b + self.S_cross_b,
            "n_film_frame": self.n_film_frame,
            "film_count_per_bin_frame": self.film_count_per_bin_frame,
            "film_count_per_bin_b": self.film_count_per_bin_b,
            "n_in_b": self.n_in_b,
            "n_out_b": self.n_out_b,
            "x_face_crossings_b": self.x_face_crossings_b,
            "theta_face_crossings_b": self.theta_face_crossings_b,
            "sum_vx_b": self.sum_vx_b,
            "sum_vy_b": self.sum_vy_b,
            "residual_mean_abs": self.residual_mean_abs,
            "residual_median_abs": self.residual_median_abs,
            "normalized_residual_mean_abs": self.normalized_residual_mean_abs,
            "normalized_residual_median_abs": self.normalized_residual_median_abs,
            "particle_diameter": np.nan
            if self.particle_diameter is None
            else self.particle_diameter,
            "pocket_radius": np.nan if self.pocket_radius is None else self.pocket_radius,
        }

    @classmethod
    def from_cache_arrays(cls, arrays: dict[str, Any]) -> "FilmContinuityResult":
        kwargs = dict(arrays)
        if "A_bin" in kwargs and "area_bin" not in kwargs:
            kwargs["area_bin"] = kwargs.pop("A_bin")
        for key in (
            "dt",
            "cylinder_radius",
            "lx",
            "residual_mean_abs",
            "residual_median_abs",
            "normalized_residual_mean_abs",
            "normalized_residual_median_abs",
        ):
            if key in kwargs:
                kwargs[key] = float(np.asarray(kwargs[key]))
        for key in ("particle_diameter", "pocket_radius"):
            if key in kwargs:
                value = float(np.asarray(kwargs[key]))
                kwargs[key] = None if np.isnan(value) else value
        return cls(**kwargs)

    def residual_summary(self) -> str:
        return (
            "residual mean abs="
            f"{self.residual_mean_abs:.6g}, "
            "median abs="
            f"{self.residual_median_abs:.6g}, "
            "normalized mean abs="
            f"{self.normalized_residual_mean_abs:.6g}, "
            "normalized median abs="
            f"{self.normalized_residual_median_abs:.6g}"
        )


def compute_film_continuity(
    config: FilmContinuityConfig,
    scalars: FilmContinuityScalars | None = None,
) -> FilmContinuityResult:
    active = load_active_matter_fields(config.active_matter_path)
    if scalars is None:
        scalars = scalars_from_active_fields(
            config,
            steps=active.steps,
            x_edges=active.x_edges,
            theta_edges=active.theta_edges,
            pocket_radius=active.pocket_radius,
        )

    vx, vy = compute_velocities(
        active.coords,
        active.shell_mask,
        scalars.lx,
        scalars.cylinder_radius,
        scalars.dt,
    )
    binned = accumulate_counts_and_sums(
        active.coords,
        active.shell_mask,
        vx,
        vy,
        scalars.x_edges,
        scalars.theta_edges,
    )

    rho_frames = rho_film(binned.film_count_per_bin_frame, scalars.area_bin)
    rho_transition = rho_frames[:-1]
    partial_t = partial_t_rho(rho_frames, scalars.dt)
    j_film = J_film_from_face_crossings(
        binned.x_face_crossings,
        binned.theta_face_crossings,
        scalars.x_edges,
        scalars.theta_edges,
        scalars.cylinder_radius,
        scalars.dt,
    )
    neg_div = neg_div_J_from_face_crossings(
        binned.x_face_crossings,
        binned.theta_face_crossings,
        scalars.area_bin,
        scalars.dt,
    )
    source = S_cross(binned.n_in, binned.n_out, scalars.area_bin, scalars.dt)
    total = neg_div + source
    residual_b = residual(partial_t, neg_div, source)
    stats = residual_diagnostics(
        residual_b,
        partial_t,
        counts=binned.counts,
        min_count=config.min_count,
    )

    return FilmContinuityResult(
        transition_steps=np.stack((active.steps[:-1], active.steps[1:]), axis=1),
        dt=scalars.dt,
        cylinder_radius=scalars.cylinder_radius,
        lx=scalars.lx,
        x_edges=scalars.x_edges,
        x_centers=scalars.x_centers,
        theta_edges=scalars.theta_edges,
        theta_centers=scalars.theta_centers,
        area_bin=scalars.area_bin,
        rho_film_b=rho_transition,
        partial_t_rho_film_b=partial_t,
        J_film_b=j_film,
        neg_div_J_film_b=neg_div,
        S_cross_b=source,
        residual_b=residual_b,
        total_b=total,
        n_film_frame=np.sum(active.shell_mask, axis=1),
        film_count_per_bin_frame=binned.film_count_per_bin_frame,
        film_count_per_bin_b=binned.counts,
        n_in_b=binned.n_in,
        n_out_b=binned.n_out,
        x_face_crossings_b=binned.x_face_crossings,
        theta_face_crossings_b=binned.theta_face_crossings,
        sum_vx_b=binned.sum_vx,
        sum_vy_b=binned.sum_vy,
        residual_mean_abs=stats["residual_mean_abs"],
        residual_median_abs=stats["residual_median_abs"],
        normalized_residual_mean_abs=stats["normalized_residual_mean_abs"],
        normalized_residual_median_abs=stats["normalized_residual_median_abs"],
        particle_diameter=scalars.particle_diameter,
        pocket_radius=scalars.pocket_radius,
    )


def residual(
    partial_t_rho_film_b: np.ndarray,
    neg_div_J_film_b: np.ndarray,
    S_cross_b: np.ndarray,
) -> np.ndarray:
    return partial_t_rho_film_b - neg_div_J_film_b - S_cross_b


def residual_diagnostics(
    residual_b: np.ndarray,
    partial_t_rho_film_b: np.ndarray,
    *,
    counts: np.ndarray | None = None,
    min_count: int = 0,
) -> dict[str, float]:
    if counts is not None and min_count > 0:
        mask = np.asarray(counts) >= min_count
    else:
        mask = np.ones(residual_b.shape, dtype=bool)
    if not np.any(mask):
        mask = np.ones(residual_b.shape, dtype=bool)

    residual_abs = np.abs(np.asarray(residual_b, dtype=float)[mask])
    scale_abs = np.abs(np.asarray(partial_t_rho_film_b, dtype=float)[mask])
    mean_scale = float(np.mean(scale_abs))
    median_scale = float(np.median(scale_abs))
    mean_residual = float(np.mean(residual_abs))
    median_residual = float(np.median(residual_abs))
    return {
        "residual_mean_abs": mean_residual,
        "residual_median_abs": median_residual,
        "normalized_residual_mean_abs": _safe_ratio(mean_residual, mean_scale),
        "normalized_residual_median_abs": _safe_ratio(median_residual, median_scale),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)
