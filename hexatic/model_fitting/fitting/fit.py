"""Orchestrate fitting of film fluxes to hydrodynamic fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FittingConfig
from .fields import HydrodynamicFields, load_or_compute_fields
from .library import (
    build_density_library,
    build_polarization_library,
    density_target,
    polarization_target,
)
from .regression import RegressionResult, fit_scalar_library, fit_vector_library


CACHE_VERSION = 17


@dataclass(frozen=True)
class FittingResult:
    transition_steps: np.ndarray
    dt: float
    cylinder_radius: float
    lx: float
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    fields: HydrodynamicFields
    mask: np.ndarray
    density_target: np.ndarray
    polarization_target: np.ndarray
    density: RegressionResult
    polarization: RegressionResult
    smoothing_bins: float = 0.0
    ridge_alpha: float = 1.0e-6
    stlsq_threshold: float = 1.0e-8
    stlsq_max_iter: int = 20
    pocket_radius: float | None = None

    @property
    def rho(self) -> np.ndarray:
        return self.fields.rho

    def as_cache_arrays(self) -> dict[str, Any]:
        arrays: dict[str, Any] = {
            "cache_version": CACHE_VERSION,
            "transition_steps": self.transition_steps,
            "dt": self.dt,
            "cylinder_radius": self.cylinder_radius,
            "lx": self.lx,
            "x_edges": self.x_edges,
            "x_centers": self.x_centers,
            "theta_edges": self.theta_edges,
            "theta_centers": self.theta_centers,
            "mask": self.mask,
            "density_target": self.density_target,
            "polarization_target": self.polarization_target,
            "density_names": np.asarray(self.density.names),
            "density_labels": np.asarray(self.density.labels),
            "density_coefficients": self.density.coefficients,
            "density_prediction": self.density.prediction,
            "density_residual": self.density.residual,
            "density_scales": self.density.scales,
            "density_active": self.density.active,
            "density_rows_used": self.density.rows_used,
            "polarization_names": np.asarray(self.polarization.names),
            "polarization_labels": np.asarray(self.polarization.labels),
            "polarization_coefficients": self.polarization.coefficients,
            "polarization_prediction": self.polarization.prediction,
            "polarization_residual": self.polarization.residual,
            "polarization_scales": self.polarization.scales,
            "polarization_active": self.polarization.active,
            "polarization_rows_used": self.polarization.rows_used,
            "smoothing_bins": self.smoothing_bins,
            "ridge_alpha": self.ridge_alpha,
            "stlsq_threshold": self.stlsq_threshold,
            "stlsq_max_iter": self.stlsq_max_iter,
            "pocket_radius": np.nan if self.pocket_radius is None else self.pocket_radius,
        }
        arrays.update(_flatten_float_dict(self.density.metrics, "density_metrics"))
        arrays.update(_flatten_float_dict(self.polarization.metrics, "polarization_metrics"))
        return arrays

    @classmethod
    def from_cache_arrays(
        cls, arrays: dict[str, Any], fields: HydrodynamicFields,
    ) -> FittingResult:
        kwargs = dict(arrays)
        cache_version = int(np.asarray(kwargs.pop("cache_version", 0)))
        if cache_version < CACHE_VERSION:
            raise ValueError(
                "Cached fitting result is from an older layout; recompute it."
            )
        kwargs.pop("fields", None)
        density = _regression_from_cache(kwargs, "density")
        polarization = _regression_from_cache(kwargs, "polarization")
        for key in (
            "dt", "cylinder_radius", "lx", "smoothing_bins",
            "ridge_alpha", "stlsq_threshold",
        ):
            if key in kwargs:
                kwargs[key] = float(np.asarray(kwargs[key]))
        if "stlsq_max_iter" in kwargs:
            kwargs["stlsq_max_iter"] = int(np.asarray(kwargs["stlsq_max_iter"]))
        pocket_radius_val = float(np.asarray(kwargs.get("pocket_radius", np.nan)))
        kwargs["pocket_radius"] = None if np.isnan(pocket_radius_val) else pocket_radius_val
        if "mask" in kwargs:
            kwargs["mask"] = np.asarray(kwargs["mask"], dtype=bool)
        if "density_target" in kwargs:
            kwargs["density_target"] = np.asarray(kwargs["density_target"], dtype=float)
        if "polarization_target" in kwargs:
            kwargs["polarization_target"] = np.asarray(
                kwargs["polarization_target"], dtype=float
            )
        for prefix in ("density", "polarization"):
            for key in tuple(kwargs):
                if key.startswith(f"{prefix}_") and key != f"{prefix}_target":
                    kwargs.pop(key)
        return cls(fields=fields, density=density, polarization=polarization, **kwargs)

    def summary(self) -> str:
        masked_bins = int(np.count_nonzero(self.mask)) if self.mask is not None else 0
        total_bins = self.mask.size if self.mask is not None else 0
        return (
            f"mask: {masked_bins}/{total_bins} valid samples, "
            f"density_r2={self.density.metrics.get('r2', np.nan):.6g}, "
            f"polarization_r2=({self.polarization.metrics.get('r2_x', np.nan):.6g}, "
            f"{self.polarization.metrics.get('r2_y', np.nan):.6g})"
        )


def compute_fitting(config: FittingConfig) -> FittingResult:
    fields = load_or_compute_fields(config)
    density_lib = build_density_library(fields)
    polarization_lib = build_polarization_library(fields)
    y_density = density_target(fields)
    y_polarization = polarization_target(fields)
    density_fit = fit_scalar_library(
        density_lib,
        y_density,
        fields.mask,
        ridge_alpha=config.ridge_alpha,
        stlsq_threshold=config.stlsq_threshold,
        stlsq_max_iter=config.stlsq_max_iter,
    )
    polarization_fit = fit_vector_library(
        polarization_lib,
        y_polarization,
        fields.mask,
        ridge_alpha=config.ridge_alpha,
        stlsq_threshold=config.stlsq_threshold,
        stlsq_max_iter=config.stlsq_max_iter,
    )
    return FittingResult(
        transition_steps=fields.transition_steps,
        dt=fields.dt,
        cylinder_radius=fields.cylinder_radius,
        lx=fields.lx,
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        theta_edges=fields.theta_edges,
        theta_centers=fields.theta_centers,
        fields=fields,
        mask=fields.mask,
        density_target=y_density,
        polarization_target=y_polarization,
        density=density_fit,
        polarization=polarization_fit,
        smoothing_bins=config.smoothing_bins,
        ridge_alpha=config.ridge_alpha,
        stlsq_threshold=config.stlsq_threshold,
        stlsq_max_iter=config.stlsq_max_iter,
        pocket_radius=fields.pocket_radius,
    )


def stlsq(
    design: np.ndarray,
    measured: np.ndarray,
    *,
    threshold: float,
    max_iter: int,
) -> np.ndarray:
    from .regression import _fit_rows

    coefficients, *_ = _fit_rows(
        design, measured,
        ridge_alpha=0.0,
        stlsq_threshold=threshold,
        stlsq_max_iter=max_iter,
    )
    return coefficients


def _flatten_float_dict(values: dict[str, float], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_names": np.asarray(tuple(values)),
        f"{prefix}_values": np.asarray([values[name] for name in values], dtype=float),
    }


def _reconstruct_float_dict(arrays: dict[str, Any], prefix: str) -> dict[str, float]:
    names_key = f"{prefix}_names"
    values_key = f"{prefix}_values"
    if names_key not in arrays or values_key not in arrays:
        return {}
    names = tuple(str(name) for name in np.asarray(arrays[names_key]))
    values = np.asarray(arrays[values_key], dtype=float)
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _regression_from_cache(arrays: dict[str, Any], prefix: str) -> RegressionResult:
    return RegressionResult(
        names=tuple(str(x) for x in np.asarray(arrays[f"{prefix}_names"])),
        labels=tuple(str(x) for x in np.asarray(arrays[f"{prefix}_labels"])),
        coefficients=np.asarray(arrays[f"{prefix}_coefficients"], dtype=float),
        prediction=np.asarray(arrays[f"{prefix}_prediction"], dtype=float),
        residual=np.asarray(arrays[f"{prefix}_residual"], dtype=float),
        metrics=_reconstruct_float_dict(arrays, f"{prefix}_metrics"),
        scales=np.asarray(arrays[f"{prefix}_scales"], dtype=float),
        active=np.asarray(arrays[f"{prefix}_active"], dtype=bool),
        rows_used=int(np.asarray(arrays[f"{prefix}_rows_used"])),
    )
