"""Orchestrate hydrodynamic density/polarization model fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import FittingConfig
from .fields import HydrodynamicFields, load_or_compute_fields
from .library import (
    build_current_library,
    build_polarization_library,
    build_s_cross_library,
    density_target,
    current_target,
    polarization_target,
)
from . import operators as ops
from .regression import RegressionResult, fit_scalar_library, fit_vector_library


CACHE_VERSION = 26


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
    source: RegressionResult
    density: RegressionResult
    polarization: RegressionResult
    source_contributions: np.ndarray
    density_contributions: np.ndarray
    polarization_contributions: np.ndarray
    curl_residual: np.ndarray
    ridge_alpha: float = 1.0e-6
    stlsq_threshold: float = 1.0e-8
    stlsq_max_iter: int = 20
    coarse_grain_transitions: int = 20
    pocket_radius: float | None = None

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
            "source_names": np.asarray(self.source.names),
            "source_labels": np.asarray(self.source.labels),
            "source_coefficients": self.source.coefficients,
            "source_prediction": self.source.prediction,
            "source_residual": self.source.residual,
            "source_scales": self.source.scales,
            "source_active": self.source.active,
            "source_rows_used": self.source.rows_used,
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
            "source_contributions": self.source_contributions,
            "density_contributions": self.density_contributions,
            "polarization_contributions": self.polarization_contributions,
            "curl_residual": self.curl_residual,
            "ridge_alpha": self.ridge_alpha,
            "stlsq_threshold": self.stlsq_threshold,
            "stlsq_max_iter": self.stlsq_max_iter,
            "coarse_grain_transitions": self.coarse_grain_transitions,
            "pocket_radius": np.nan if self.pocket_radius is None else self.pocket_radius,
        }
        arrays.update(_flatten_float_dict(self.source.metrics, "source_metrics"))
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
        source = _regression_from_cache(kwargs, "source")
        density = _regression_from_cache(kwargs, "density")
        polarization = _regression_from_cache(kwargs, "polarization")
        for key in (
            "dt", "cylinder_radius", "lx",
            "ridge_alpha", "stlsq_threshold",
        ):
            if key in kwargs:
                kwargs[key] = float(np.asarray(kwargs[key]))
        if "stlsq_max_iter" in kwargs:
            kwargs["stlsq_max_iter"] = int(np.asarray(kwargs["stlsq_max_iter"]))
        if "coarse_grain_transitions" in kwargs:
            kwargs["coarse_grain_transitions"] = int(np.asarray(kwargs["coarse_grain_transitions"]))
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
        for prefix in ("source", "density", "polarization"):
            for key in tuple(kwargs):
                if key.startswith(f"{prefix}_") and key not in (
                    f"{prefix}_target", f"{prefix}_contributions",
                ):
                    kwargs.pop(key)
        return cls(
            fields=fields,
            source=source,
            density=density,
            polarization=polarization,
            **kwargs,
        )

    def summary(self) -> str:
        masked_bins = int(np.count_nonzero(self.mask)) if self.mask is not None else 0
        total_bins = self.mask.size if self.mask is not None else 0
        curl_stats = ""
        if self.curl_residual is not None:
            c = self.curl_residual[self.mask]
            if c.size > 0:
                curl_stats = f", curl_resid: mean={np.nanmean(c):.4g}, rms={np.sqrt(np.nanmean(c**2)):.4g}"
        return (
            f"mask: {masked_bins}/{total_bins} valid samples, "
            f"source_r2={self.source.metrics.get('r2', np.nan):.6g}, "
            f"source_nmae={100.0 * self.source.metrics.get('normalized_mae', np.nan):.4g}%, "
            f"density_r2={self.density.metrics.get('r2', np.nan):.6g}, "
            f"density_nmae={100.0 * self.density.metrics.get('normalized_mae', np.nan):.4g}%, "
            f"current_r2=({self.density.metrics.get('current_r2_x', np.nan):.6g}, "
            f"{self.density.metrics.get('current_r2_y', np.nan):.6g}), "
            f"polarization_r2=({self.polarization.metrics.get('r2_x', np.nan):.6g}, "
            f"{self.polarization.metrics.get('r2_y', np.nan):.6g}), "
            f"polarization_nmae=("
            f"{100.0 * self.polarization.metrics.get('normalized_mae_x', np.nan):.4g}%, "
            f"{100.0 * self.polarization.metrics.get('normalized_mae_y', np.nan):.4g}%)"
            f"{curl_stats}"
        )


def compute_fitting(config: FittingConfig) -> FittingResult:
    """Fit current and polarization models, then evaluate density conservatively."""
    raw_fields = load_or_compute_fields(config)
    fields = _coarse_grain_fields(raw_fields, config.coarse_grain_transitions)
    source_lib = build_s_cross_library(fields)
    current_lib = build_current_library(fields)
    polarization_lib = build_polarization_library(fields)
    y_density = density_target(fields)
    y_current = current_target(fields)
    y_polarization = polarization_target(fields)

    source_fit = fit_scalar_library(
        source_lib,
        fields.S_cross,
        fields.mask,
        ridge_alpha=config.ridge_alpha,
        stlsq_threshold=config.stlsq_threshold,
        stlsq_max_iter=config.stlsq_max_iter,
    )
    current_fit = fit_vector_library(
        current_lib,
        y_current,
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

    kx, ky = _field_wavenumbers(fields)
    density_fit = _current_fit_as_density_result(
        current_fit=current_fit,
        density_target_values=y_density,
        mask=fields.mask,
        kx=kx,
        ky=ky,
    )
    density_contributions = np.stack(
        [
            -ops.fft_divergence(
                current_lib.values[..., term_idx, :] * current_fit.coefficients[term_idx],
                kx,
                ky,
            )
            for term_idx in range(current_lib.values.shape[-2])
        ],
        axis=-1,
    )
    source_contributions = source_lib.values * source_fit.coefficients[None, None, None, :]
    polarization_contributions = (
        polarization_lib.values * polarization_fit.coefficients[None, None, None, :, None]
    )

    # curl of polarization residual
    residual = y_polarization - polarization_fit.prediction
    curl_residual = ops.fft_curl(residual, kx, ky)

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
        source=source_fit,
        density=density_fit,
        polarization=polarization_fit,
        source_contributions=source_contributions,
        density_contributions=density_contributions,
        polarization_contributions=polarization_contributions,
        curl_residual=curl_residual,
        ridge_alpha=config.ridge_alpha,
        stlsq_threshold=config.stlsq_threshold,
        stlsq_max_iter=config.stlsq_max_iter,
        coarse_grain_transitions=config.coarse_grain_transitions,
        pocket_radius=fields.pocket_radius,
    )


def _field_wavenumbers(fields: HydrodynamicFields) -> tuple[np.ndarray, np.ndarray]:
    """Return spectral wave numbers for the cylinder surface grid."""
    ly = fields.cylinder_radius * (fields.theta_edges[-1] - fields.theta_edges[0])
    return ops.build_k_vectors(fields.x_centers.size, fields.theta_centers.size, fields.lx, ly)


def _current_fit_as_density_result(
    *,
    current_fit: RegressionResult,
    density_target_values: np.ndarray,
    mask: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
) -> RegressionResult:
    """Wrap a vector current fit as the density result via -div J_fit."""
    density_prediction = -ops.fft_divergence(current_fit.prediction, kx, ky)
    density_residual = density_target_values - density_prediction
    density_metrics = _density_metrics(density_target_values, density_prediction, mask)
    density_metrics.update({f"current_{k}": v for k, v in current_fit.metrics.items()})
    return RegressionResult(
        names=current_fit.names,
        labels=current_fit.labels,
        coefficients=current_fit.coefficients,
        prediction=density_prediction,
        residual=density_residual,
        metrics=density_metrics,
        scales=current_fit.scales,
        active=current_fit.active,
        rows_used=current_fit.rows_used,
    )


def _coarse_grain_fields(fields: HydrodynamicFields, window: int) -> HydrodynamicFields:
    """Average adjacent transition fields for slower, lower-noise fitting."""
    if window <= 1:
        return fields

    def avg(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        usable = (values.shape[0] // window) * window
        assert usable > 0
        return values[:usable].reshape(-1, window, *values.shape[1:]).mean(axis=1)

    def all_valid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=bool)
        usable = (values.shape[0] // window) * window
        assert usable > 0
        return values[:usable].reshape(-1, window, *values.shape[1:]).all(axis=1)

    usable = (fields.transition_steps.shape[0] // window) * window
    steps = fields.transition_steps[:usable].reshape(-1, window, 2)
    transition_steps = np.stack((steps[:, 0, 0], steps[:, -1, 1]), axis=1)
    return HydrodynamicFields(
        transition_steps=transition_steps,
        dt=fields.dt * window,
        cylinder_radius=fields.cylinder_radius,
        lx=fields.lx,
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        theta_edges=fields.theta_edges,
        theta_centers=fields.theta_centers,
        rho=fields.rho,
        P=fields.P,
        chirality=fields.chirality,
        D=fields.D,
        hexatic_order=fields.hexatic_order,
        S_cross=avg(fields.S_cross),
        partial_t_rho=avg(fields.partial_t_rho),
        partial_t_P=avg(fields.partial_t_P),
        grad_rho=avg(fields.grad_rho),
        grad_D=avg(fields.grad_D),
        grad_hexatic_order=avg(fields.grad_hexatic_order),
        div_P=avg(fields.div_P),
        div_chiral_P_perp=avg(fields.div_chiral_P_perp),
        laplacian_rho=avg(fields.laplacian_rho),
        laplacian_D=avg(fields.laplacian_D),
        laplacian_hexatic_order=avg(fields.laplacian_hexatic_order),
        P_dot_grad_P=avg(fields.P_dot_grad_P),
        P_perp_dot_grad_P=avg(fields.P_perp_dot_grad_P),
        laplacian_P=avg(fields.laplacian_P),
        laplacian_P_perp=avg(fields.laplacian_P_perp),
        material_current=avg(fields.material_current),
        mid_rho=avg(fields.mid_rho),
        mid_chirality=avg(fields.mid_chirality),
        mid_D=avg(fields.mid_D),
        mid_hexatic_order=avg(fields.mid_hexatic_order),
        mid_h=avg(fields.mid_h),
        mid_P_r=avg(fields.mid_P_r),
        mid_P=avg(fields.mid_P),
        mid_force_density=avg(fields.mid_force_density),
        mask=all_valid(fields.mask),
        pocket_radius=fields.pocket_radius,
    )


def _density_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    y = target[valid]
    p = prediction[valid]
    if y.size == 0:
        return {
            "correlation": float("nan"),
            "r2": float("nan"),
            "normalized_mae": float("nan"),
        }
    corr = float("nan")
    if y.size >= 2 and np.std(y) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(y, p)[0, 1])
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - float(np.sum((y - p) ** 2)) / ss_tot
    scale = float(np.mean(np.abs(y)))
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0
    return {
        "correlation": corr,
        "r2": r2,
        "normalized_mae": float(np.mean(np.abs(y - p)) / scale),
    }


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
