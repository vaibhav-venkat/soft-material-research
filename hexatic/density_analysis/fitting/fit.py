from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg

from .config import FittingConfig
from .fields import load_or_compute_fields


CACHE_VERSION = 2


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
    rho: np.ndarray
    J: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    grad_x_mid: np.ndarray
    grad_y_mid: np.ndarray
    c_x: np.ndarray
    c_y: np.ndarray
    c_x_global: float
    c_y_global: float
    fitted_x: np.ndarray
    fitted_y: np.ndarray
    residual_x: np.ndarray
    residual_y: np.ndarray
    mask: np.ndarray
    counts: np.ndarray | None
    residual_x_mean_abs: float
    residual_y_mean_abs: float
    residual_x_median_abs: float
    residual_y_median_abs: float
    particle_diameter: float | None = None
    pocket_radius: float | None = None

    def as_cache_arrays(self) -> dict[str, Any]:
        return {
            "cache_version": CACHE_VERSION,
            "transition_steps": self.transition_steps,
            "dt": self.dt,
            "cylinder_radius": self.cylinder_radius,
            "lx": self.lx,
            "x_edges": self.x_edges,
            "x_centers": self.x_centers,
            "theta_edges": self.theta_edges,
            "theta_centers": self.theta_centers,
            "rho": self.rho,
            "J": self.J,
            "grad_x": self.grad_x,
            "grad_y": self.grad_y,
            "grad_x_mid": self.grad_x_mid,
            "grad_y_mid": self.grad_y_mid,
            "c_x": self.c_x,
            "c_y": self.c_y,
            "c_x_global": self.c_x_global,
            "c_y_global": self.c_y_global,
            "fitted_x": self.fitted_x,
            "fitted_y": self.fitted_y,
            "residual_x": self.residual_x,
            "residual_y": self.residual_y,
            "mask": self.mask,
            "counts": np.asarray([])
            if self.counts is None
            else np.asarray(self.counts),
            "residual_x_mean_abs": self.residual_x_mean_abs,
            "residual_y_mean_abs": self.residual_y_mean_abs,
            "residual_x_median_abs": self.residual_x_median_abs,
            "residual_y_median_abs": self.residual_y_median_abs,
            "particle_diameter": np.nan
            if self.particle_diameter is None
            else self.particle_diameter,
            "pocket_radius": np.nan if self.pocket_radius is None else self.pocket_radius,
        }

    @classmethod
    def from_cache_arrays(cls, arrays: dict[str, Any]) -> "FittingResult":
        kwargs = dict(arrays)
        cache_version = int(np.asarray(kwargs.pop("cache_version", 0)))
        if cache_version < CACHE_VERSION:
            raise ValueError(
                "Cached fitting result is from an older coefficient layout; recompute it."
            )
        for key in (
            "dt",
            "cylinder_radius",
            "lx",
            "c_x_global",
            "c_y_global",
            "residual_x_mean_abs",
            "residual_y_mean_abs",
            "residual_x_median_abs",
            "residual_y_median_abs",
        ):
            if key in kwargs:
                kwargs[key] = float(np.asarray(kwargs[key]))
        for key in ("particle_diameter", "pocket_radius"):
            if key in kwargs:
                value = float(np.asarray(kwargs[key]))
                kwargs[key] = None if np.isnan(value) else value
        if "counts" in kwargs and np.asarray(kwargs["counts"]).size == 0:
            kwargs["counts"] = None
        if "mask" in kwargs:
            kwargs["mask"] = np.asarray(kwargs["mask"], dtype=bool)
        return cls(**kwargs)

    def summary(self) -> str:
        residual_stats = _normalized_residual_summary(
            self.residual_x,
            self.residual_y,
            self.J,
            self.mask,
        )
        return (
            f"c_x_global={self.c_x_global:.6g}, "
            f"c_y_global={self.c_y_global:.6g}, "
            f"mean_abs_J_x={residual_stats['mean_abs_J_x']:.6g}, "
            f"mean_abs_J_y={residual_stats['mean_abs_J_y']:.6g}, "
            f"normalized_residual_x mean={residual_stats['residual_x_mean']:.6g}, "
            f"median={residual_stats['residual_x_median']:.6g}, "
            f"mean abs={residual_stats['residual_x_mean_abs']:.6g}, "
            f"normalized_residual_y mean={residual_stats['residual_y_mean']:.6g}, "
            f"median={residual_stats['residual_y_median']:.6g}, "
            f"mean abs={residual_stats['residual_y_mean_abs']:.6g}"
        )


def compute_fitting(config: FittingConfig) -> FittingResult:
    print(f"[fitting] Starting fit for case {config.case_id!r}.")
    fields = load_or_compute_fields(config)
    print(f"[fitting] Building fit mask with min_count={config.min_count}...")
    mask = _fit_mask(fields.counts, fields.grad_x_mid.shape, config.min_count)
    print(f"[fitting] Fit mask keeps {np.count_nonzero(mask)}/{mask.size} bins.")
    print("[fitting] Solving local least-squares coefficient maps over transitions...")
    c_x = _coefficient_map(fields.grad_x_mid, fields.J[..., 0], mask)
    c_y = _coefficient_map(fields.grad_y_mid, fields.J[..., 1], mask)
    print(
        "[fitting] Local coefficient maps ready: "
        f"c_x finite {np.count_nonzero(np.isfinite(c_x))}/{c_x.size}, "
        f"c_y finite {np.count_nonzero(np.isfinite(c_y))}/{c_y.size}."
    )
    print("[fitting] Solving global least-squares coefficients...")
    c_x_global = _coefficient(fields.grad_x_mid[mask], fields.J[..., 0][mask])
    c_y_global = _coefficient(fields.grad_y_mid[mask], fields.J[..., 1][mask])
    print(
        "[fitting] Global coefficients: "
        f"c_x={c_x_global:.6g}, c_y={c_y_global:.6g}."
    )

    print("[fitting] Computing fitted fields and residual diagnostics...")
    fitted_x = c_x[None, :, :] * fields.grad_x_mid
    fitted_y = c_y[None, :, :] * fields.grad_y_mid
    residual_x = fields.J[..., 0] - fitted_x
    residual_y = fields.J[..., 1] - fitted_y
    diagnostics = _residual_diagnostics(residual_x, residual_y, mask)
    residual_stats = _normalized_residual_summary(residual_x, residual_y, fields.J, mask)
    print(
        "[fitting] Residual summary: "
        f"mean_abs_J_x={residual_stats['mean_abs_J_x']:.6g}, "
        f"normalized_residual_x mean={residual_stats['residual_x_mean']:.6g}, "
        f"median={residual_stats['residual_x_median']:.6g}, "
        f"mean abs={residual_stats['residual_x_mean_abs']:.6g}; "
        f"mean_abs_J_y={residual_stats['mean_abs_J_y']:.6g}, "
        f"normalized_residual_y mean={residual_stats['residual_y_mean']:.6g}, "
        f"median={residual_stats['residual_y_median']:.6g}, "
        f"mean abs={residual_stats['residual_y_mean_abs']:.6g}."
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
        rho=fields.rho,
        J=fields.J,
        grad_x=fields.grad_x,
        grad_y=fields.grad_y,
        grad_x_mid=fields.grad_x_mid,
        grad_y_mid=fields.grad_y_mid,
        c_x=c_x,
        c_y=c_y,
        c_x_global=c_x_global,
        c_y_global=c_y_global,
        fitted_x=fitted_x,
        fitted_y=fitted_y,
        residual_x=residual_x,
        residual_y=residual_y,
        mask=mask,
        counts=fields.counts,
        residual_x_mean_abs=diagnostics["residual_x_mean_abs"],
        residual_y_mean_abs=diagnostics["residual_y_mean_abs"],
        residual_x_median_abs=diagnostics["residual_x_median_abs"],
        residual_y_median_abs=diagnostics["residual_y_median_abs"],
        pocket_radius=fields.pocket_radius,
    )


def _coefficient_map(
    gradient: np.ndarray,
    measured: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float)
    measured = np.asarray(measured, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if gradient.shape != measured.shape or gradient.shape != mask.shape:
        raise ValueError("gradient, measured, and mask must have matching shapes.")

    finite = mask & np.isfinite(gradient) & np.isfinite(measured)
    weighted_gradient = np.where(finite, gradient, 0.0)
    weighted_measured = np.where(finite, measured, 0.0)
    numerator = np.sum(weighted_gradient * weighted_measured, axis=0)
    denominator = np.sum(weighted_gradient * weighted_gradient, axis=0)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0.0,
    )


def _coefficient(gradient: np.ndarray, measured: np.ndarray) -> float:
    gradient = np.asarray(gradient, dtype=float).ravel()
    measured = np.asarray(measured, dtype=float).ravel()
    finite = np.isfinite(gradient) & np.isfinite(measured)
    gradient = gradient[finite]
    measured = measured[finite]
    if gradient.size == 0 or np.allclose(gradient, 0.0):
        return float("nan")
    coefficient, *_ = linalg.lstsq(gradient[:, None], measured)
    return float(coefficient[0])


def _nan_stat(values: np.ndarray, function) -> float:
    if values.size == 0:
        return float("nan")
    return float(function(values))


def _fit_mask(
    counts: np.ndarray | None,
    shape: tuple[int, int, int],
    min_count: int,
) -> np.ndarray:
    if counts is None or min_count <= 0:
        return np.ones(shape, dtype=bool)
    counts = np.asarray(counts)
    if counts.shape != shape:
        raise ValueError(f"counts shape {counts.shape} does not match {shape}.")
    mask = counts >= min_count
    if not np.any(mask):
        return np.ones(shape, dtype=bool)
    return mask


def _residual_diagnostics(
    residual_x: np.ndarray,
    residual_y: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    residual_x = np.asarray(residual_x, dtype=float)
    residual_y = np.asarray(residual_y, dtype=float)
    x_mask = mask & np.isfinite(residual_x)
    y_mask = mask & np.isfinite(residual_y)
    x_abs = np.abs(residual_x[x_mask])
    y_abs = np.abs(residual_y[y_mask])
    return {
        "residual_x_mean_abs": float(np.nanmean(x_abs)),
        "residual_y_mean_abs": float(np.nanmean(y_abs)),
        "residual_x_median_abs": float(np.nanmedian(x_abs)),
        "residual_y_median_abs": float(np.nanmedian(y_abs)),
    }


def _normalized_residual_summary(
    residual_x: np.ndarray,
    residual_y: np.ndarray,
    measured_J: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    residual_x = np.asarray(residual_x, dtype=float)
    residual_y = np.asarray(residual_y, dtype=float)
    measured_J = np.asarray(measured_J, dtype=float)
    x_mask = mask & np.isfinite(residual_x) & np.isfinite(measured_J[..., 0])
    y_mask = mask & np.isfinite(residual_y) & np.isfinite(measured_J[..., 1])
    mean_abs_J_x = _mean_abs_scale(measured_J[..., 0], x_mask)
    mean_abs_J_y = _mean_abs_scale(measured_J[..., 1], y_mask)
    x_values = residual_x[x_mask] / mean_abs_J_x
    y_values = residual_y[y_mask] / mean_abs_J_y
    return {
        "mean_abs_J_x": mean_abs_J_x,
        "mean_abs_J_y": mean_abs_J_y,
        "residual_x_mean": _nan_stat(x_values, np.nanmean),
        "residual_x_median": _nan_stat(x_values, np.nanmedian),
        "residual_x_mean_abs": _nan_stat(np.abs(x_values), np.nanmean),
        "residual_x_median_abs": _nan_stat(np.abs(x_values), np.nanmedian),
        "residual_y_mean": _nan_stat(y_values, np.nanmean),
        "residual_y_median": _nan_stat(y_values, np.nanmedian),
        "residual_y_mean_abs": _nan_stat(np.abs(y_values), np.nanmean),
        "residual_y_median_abs": _nan_stat(np.abs(y_values), np.nanmedian),
    }


def _mean_abs_scale(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 1.0
    scale = float(np.nanmean(np.abs(np.asarray(values, dtype=float)[mask])))
    if not np.isfinite(scale) or scale == 0.0:
        return 1.0
    return scale
