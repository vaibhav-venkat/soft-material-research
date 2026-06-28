"""Orchestrate fitting of film fluxes to hydrodynamic fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter

from .config import FittingConfig
from .fields import load_or_compute_fields
from .io_cache import flatten_array_dict, reconstruct_array_dict


CACHE_VERSION = 16
SMOOTH_WEIGHT_FLOOR = 1.0e-12


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
    J: np.ndarray
    frame_fields: dict[str, np.ndarray]
    mid_fields: dict[str, np.ndarray]
    mask: np.ndarray
    counts: np.ndarray | None
    smoothing_bins: float = 0.0
    particle_diameter: float | None = None
    pocket_radius: float | None = None

    @property
    def rho(self) -> np.ndarray:
        return self.frame_fields["rho"]

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
            "J": self.J,
            "mask": self.mask,
            "counts": np.asarray([])
            if self.counts is None
            else np.asarray(self.counts),
            "smoothing_bins": self.smoothing_bins,
            "particle_diameter": np.nan
            if self.particle_diameter is None
            else self.particle_diameter,
            "pocket_radius": np.nan if self.pocket_radius is None else self.pocket_radius,
        }
        arrays.update(flatten_array_dict(self.frame_fields, "frame_fields"))
        arrays.update(flatten_array_dict(self.mid_fields, "mid_fields"))
        return arrays

    @classmethod
    def from_cache_arrays(cls, arrays: dict[str, Any]) -> FittingResult:
        kwargs = dict(arrays)
        cache_version = int(np.asarray(kwargs.pop("cache_version", 0)))
        if cache_version < CACHE_VERSION:
            raise ValueError(
                "Cached fitting result is from an older layout; recompute it."
            )
        frame_fields = reconstruct_array_dict(kwargs, "frame_fields")
        mid_fields = reconstruct_array_dict(kwargs, "mid_fields")
        for prefix in ("frame_fields", "mid_fields"):
            for key in tuple(kwargs):
                if key.startswith(f"{prefix}__"):
                    kwargs.pop(key)

        for key in ("dt", "cylinder_radius", "lx", "smoothing_bins"):
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
        return cls(
            **kwargs,
            frame_fields=frame_fields,
            mid_fields=mid_fields,
        )

    def summary(self) -> str:
        masked_bins = int(np.count_nonzero(self.mask)) if self.mask is not None else 0
        total_bins = self.mask.size if self.mask is not None else 0
        return (
            f"mask: {masked_bins}/{total_bins} bins, "
            f"smoothing_bins={self.smoothing_bins}"
        )


def compute_fitting(config: FittingConfig) -> FittingResult:
    fields = load_or_compute_fields(config)
    mask = _fit_mask(fields.counts, fields.J.shape[:3])
    print(f"[fitting] Fit mask: {int(np.count_nonzero(mask))}/{mask.size} bins.")

    return FittingResult(
        transition_steps=fields.transition_steps,
        dt=fields.dt,
        cylinder_radius=fields.cylinder_radius,
        lx=fields.lx,
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        theta_edges=fields.theta_edges,
        theta_centers=fields.theta_centers,
        J=fields.J,
        frame_fields=fields.frame_fields,
        mid_fields=fields.mid_fields,
        mask=mask,
        counts=fields.counts,
        smoothing_bins=config.smoothing_bins,
        pocket_radius=fields.pocket_radius,
    )


def stlsq(
    design: np.ndarray,
    measured: np.ndarray,
    *,
    threshold: float,
    max_iter: int,
) -> np.ndarray:
    """STLSQ: iteratively zero out coefficients below threshold."""
    design = np.asarray(design, dtype=float)
    measured = np.asarray(measured, dtype=float)
    if design.ndim != 2:
        raise ValueError("design must be two-dimensional.")
    if measured.ndim != 1 or measured.shape[0] != design.shape[0]:
        raise ValueError("measured must be one-dimensional and match design rows.")

    finite = np.isfinite(measured) & np.all(np.isfinite(design), axis=1)
    design = design[finite]
    measured = measured[finite]
    n_features = design.shape[1]
    coefficients = np.zeros(n_features, dtype=float)
    if design.shape[0] == 0 or n_features == 0:
        return coefficients

    active = np.any(np.abs(design) > 0.0, axis=0)
    if not np.any(active):
        return coefficients

    for _ in range(max_iter):
        active_indices = np.flatnonzero(active)
        solved, *_ = linalg.lstsq(design[:, active_indices], measured)
        next_coefficients = np.zeros(n_features, dtype=float)
        next_coefficients[active_indices] = solved
        next_active = np.abs(next_coefficients) >= threshold
        next_active &= np.any(np.abs(design) > 0.0, axis=0)
        if np.array_equal(next_active, active):
            coefficients = next_coefficients
            break
        coefficients = next_coefficients
        active = next_active
        if not np.any(active):
            coefficients[:] = 0.0
            break
    return coefficients


def normalized_stlsq_physical(
    design: np.ndarray,
    measured: np.ndarray,
    *,
    threshold: float,
    max_iter: int,
) -> np.ndarray:
    """RMS-normalize design columns, fit, then rescale coefficients."""
    normalized_design, scales = _normalize_design_columns(design)
    normalized_coefficients = stlsq(
        normalized_design,
        measured,
        threshold=threshold,
        max_iter=max_iter,
    )
    return normalized_coefficients / scales


def _normalize_design_columns(
    design: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(design, dtype=float)
    finite = np.all(np.isfinite(design), axis=1)
    scales = np.ones(design.shape[1], dtype=float)
    if np.any(finite):
        rms = np.sqrt(np.nanmean(design[finite] ** 2, axis=0))
        valid_scales = np.isfinite(rms) & (rms > 0.0)
        scales[valid_scales] = rms[valid_scales]
    return design / scales[None, :], scales


def _fit_mask(
    counts: np.ndarray | None,
    shape: tuple[int, int, int],
) -> np.ndarray:
    if counts is None:
        return np.ones(shape, dtype=bool)
    counts = np.asarray(counts)
    if counts.shape != shape:
        raise ValueError(f"counts shape {counts.shape} does not match {shape}.")
    return counts > 0
