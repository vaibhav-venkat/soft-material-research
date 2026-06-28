from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter

from .config import FittingConfig
from .fields import compute_scalar_modifiers, load_or_compute_fields
from .io_cache import flatten_array_dict, reconstruct_array_dict


CACHE_VERSION = 15
COMPONENTS = ("x", "y")
G_MODIFIER_NAMES = ("G_0", "G_1", "G_2", "G_3", "G_4")
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
    candidate_names: tuple[str, ...]
    coef_map: dict[str, np.ndarray]
    fitted: np.ndarray
    residual: np.ndarray
    mask: np.ndarray
    counts: np.ndarray | None
    residual_x_mean_abs: float
    residual_y_mean_abs: float
    residual_x_median_abs: float
    residual_y_median_abs: float
    stlsq_threshold: float
    stlsq_max_iter: int
    smoothing_bins: float = 0.0
    particle_diameter: float | None = None
    pocket_radius: float | None = None

    @property
    def rho(self) -> np.ndarray:
        return self.frame_fields["rho"]

    @property
    def residual_x(self) -> np.ndarray:
        return self.residual[..., 0]

    @property
    def residual_y(self) -> np.ndarray:
        return self.residual[..., 1]

    @property
    def fitted_x(self) -> np.ndarray:
        return self.fitted[..., 0]

    @property
    def fitted_y(self) -> np.ndarray:
        return self.fitted[..., 1]

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
            "candidate_names": np.asarray(self.candidate_names),
            "fitted": self.fitted,
            "residual": self.residual,
            "mask": self.mask,
            "counts": np.asarray([])
            if self.counts is None
            else np.asarray(self.counts),
            "residual_x_mean_abs": self.residual_x_mean_abs,
            "residual_y_mean_abs": self.residual_y_mean_abs,
            "residual_x_median_abs": self.residual_x_median_abs,
            "residual_y_median_abs": self.residual_y_median_abs,
            "stlsq_threshold": self.stlsq_threshold,
            "stlsq_max_iter": self.stlsq_max_iter,
            "smoothing_bins": self.smoothing_bins,
            "particle_diameter": np.nan
            if self.particle_diameter is None
            else self.particle_diameter,
            "pocket_radius": np.nan if self.pocket_radius is None else self.pocket_radius,
        }
        arrays.update(flatten_array_dict(self.frame_fields, "frame_fields"))
        arrays.update(flatten_array_dict(self.mid_fields, "mid_fields"))
        arrays.update(flatten_array_dict(self.coef_map, "coef_map"))
        return arrays

    @classmethod
    def from_cache_arrays(cls, arrays: dict[str, Any]) -> FittingResult:
        kwargs = dict(arrays)
        cache_version = int(np.asarray(kwargs.pop("cache_version", 0)))
        if cache_version < CACHE_VERSION:
            raise ValueError(
                "Cached fitting result is from an older coefficient layout; recompute it."
            )
        frame_fields = reconstruct_array_dict(kwargs, "frame_fields")
        mid_fields = reconstruct_array_dict(kwargs, "mid_fields")
        coef_map = reconstruct_array_dict(kwargs, "coef_map")
        for prefix in ("frame_fields", "mid_fields", "coef_map"):
            for key in tuple(kwargs):
                if key.startswith(f"{prefix}__"):
                    kwargs.pop(key)

        for key in (
            "dt",
            "cylinder_radius",
            "lx",
            "residual_x_mean_abs",
            "residual_y_mean_abs",
            "residual_x_median_abs",
            "residual_y_median_abs",
            "stlsq_threshold",
            "smoothing_bins",
        ):
            if key in kwargs:
                kwargs[key] = float(np.asarray(kwargs[key]))
        if "stlsq_max_iter" in kwargs:
            kwargs["stlsq_max_iter"] = int(np.asarray(kwargs["stlsq_max_iter"]))
        for key in ("particle_diameter", "pocket_radius"):
            if key in kwargs:
                value = float(np.asarray(kwargs[key]))
                kwargs[key] = None if np.isnan(value) else value
        if "counts" in kwargs and np.asarray(kwargs["counts"]).size == 0:
            kwargs["counts"] = None
        if "mask" in kwargs:
            kwargs["mask"] = np.asarray(kwargs["mask"], dtype=bool)
        kwargs["candidate_names"] = tuple(str(name) for name in kwargs["candidate_names"])
        return cls(
            **kwargs,
            frame_fields=frame_fields,
            mid_fields=mid_fields,
            coef_map=coef_map,
        )

    def summary(self) -> str:
        residual_stats = _normalized_residual_summary(
            self.residual_x,
            self.residual_y,
            self.J,
            self.mask,
        )
        return (
            f"normalized_residual_x "
            f"mean={residual_stats['residual_x_mean']:.6g}, "
            f"mean_abs={residual_stats['residual_x_mean_abs']:.6g}, "
            f"median={residual_stats['residual_x_median']:.6g}; "
            f"normalized_residual_y "
            f"mean={residual_stats['residual_y_mean']:.6g}, "
            f"mean_abs={residual_stats['residual_y_mean_abs']:.6g}, "
            f"median={residual_stats['residual_y_median']:.6g}"
        )


def compute_fitting(config: FittingConfig) -> FittingResult:
    fields = load_or_compute_fields(config)
    candidate_names = config.selected_candidate_names
    _validate_candidates(fields.mid_fields, candidate_names, fields.J.shape)
    mask = _fit_mask(fields.counts, fields.J.shape[:3], config.min_count)
    print(f"[fitting] Fit mask: {int(np.count_nonzero(mask))}/{mask.size} bins.")
    J_fit = fields.J
    mid_fields_fit = fields.mid_fields
    g_modifier_indices = config.g_modifier_indices
    if not np.any(mask):
        print("[fitting] Mask empty: returning zero fit.")
        coef_map = _zero_coefficients(candidate_names, g_modifier_indices)
        fitted = np.zeros_like(J_fit, dtype=float)
    else:
        if config.smoothing_bins > 0.0:
            print(f"[fitting] Smoothing with sigma={config.smoothing_bins:g} bins...")
            J_fit, mid_fields_fit = _smooth_fitting_arrays(
                fields.J,
                fields.mid_fields,
                mask,
                candidate_names,
                sigma_bins=config.smoothing_bins,
            )
            scalar_names = ("rho", "D", "hexatic_order")
            for name in scalar_names:
                mid_fields_fit[name] = _smooth_scalar_field(
                    mid_fields_fit[name], mask, config.smoothing_bins
                )
        print("[fitting] Computing scalar modifiers...")
        num, G_modifiers = compute_scalar_modifiers(mid_fields_fit, mask)
        print("[fitting] Solving global STLSQ coefficients...")
        coef_map = _global_coefficients(
            mid_fields_fit,
            J_fit,
            mask,
            candidate_names,
            g_modifier_indices=g_modifier_indices,
            G_modifiers=G_modifiers,
            threshold=config.stlsq_threshold,
            max_iter=config.stlsq_max_iter,
        )
        fitted = _fitted_from_coef(
            mid_fields_fit,
            coef_map,
            candidate_names,
            G_modifiers=G_modifiers,
            g_modifier_indices=g_modifier_indices,
        )
    residual = J_fit - fitted
    diagnostics = _residual_diagnostics(residual[..., 0], residual[..., 1], mask)

    return FittingResult(
        transition_steps=fields.transition_steps,
        dt=fields.dt,
        cylinder_radius=fields.cylinder_radius,
        lx=fields.lx,
        x_edges=fields.x_edges,
        x_centers=fields.x_centers,
        theta_edges=fields.theta_edges,
        theta_centers=fields.theta_centers,
        J=J_fit,
        frame_fields=fields.frame_fields,
        mid_fields=mid_fields_fit,
        candidate_names=candidate_names,
        coef_map=coef_map,
        fitted=fitted,
        residual=residual,
        mask=mask,
        counts=fields.counts,
        residual_x_mean_abs=diagnostics["residual_x_mean_abs"],
        residual_y_mean_abs=diagnostics["residual_y_mean_abs"],
        residual_x_median_abs=diagnostics["residual_x_median_abs"],
        residual_y_median_abs=diagnostics["residual_y_median_abs"],
        stlsq_threshold=config.stlsq_threshold,
        stlsq_max_iter=config.stlsq_max_iter,
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


def _smooth_fitting_arrays(
    J: np.ndarray,
    mid_fields: dict[str, np.ndarray],
    mask: np.ndarray,
    candidate_names: tuple[str, ...],
    *,
    sigma_bins: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if sigma_bins <= 0.0:
        return J, mid_fields
    J_smoothed = _smooth_vector_field(J, mask, sigma_bins)
    smoothed_fields = dict(mid_fields)
    for name in candidate_names:
        smoothed_fields[name] = _smooth_vector_field(
            mid_fields[name],
            mask,
            sigma_bins,
        )
    return J_smoothed, smoothed_fields


def _smooth_vector_field(
    values: np.ndarray,
    mask: np.ndarray,
    sigma_bins: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape[:3] != mask.shape or values.shape[-1] != 2:
        raise ValueError("values must have shape mask + (2,).")
    smoothed = np.empty_like(values, dtype=float)
    for component in range(2):
        smoothed[..., component] = _smooth_scalar_field(
            values[..., component],
            mask,
            sigma_bins,
        )
    return smoothed


def _smooth_scalar_field(
    values: np.ndarray,
    mask: np.ndarray,
    sigma_bins: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"values shape {values.shape} does not match mask {mask.shape}.")
    finite = mask & np.isfinite(values)
    weights = finite.astype(float)
    weighted = np.where(finite, values, 0.0)
    sigma = (0.0, float(sigma_bins), float(sigma_bins))
    numerator = gaussian_filter(weighted, sigma=sigma, mode="wrap")
    denominator = gaussian_filter(weights, sigma=sigma, mode="wrap")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(values, np.nan, dtype=float),
        where=denominator > SMOOTH_WEIGHT_FLOOR,
    )


def _global_coefficients(
    mid_fields: dict[str, np.ndarray],
    measured: np.ndarray,
    mask: np.ndarray,
    candidate_names: tuple[str, ...],
    *,
    g_modifier_indices: tuple[int, ...],
    G_modifiers: dict[str, np.ndarray],
    threshold: float,
    max_iter: int,
) -> dict[str, np.ndarray]:
    coef_map = _zero_coefficients(candidate_names, g_modifier_indices)
    coef_keys = tuple(coef_map)
    if not np.any(mask):
        return coef_map
    for component in range(2):
        columns = []
        for name in candidate_names:
            V = mid_fields[name][..., component]
            for m_idx in g_modifier_indices:
                G = G_modifiers[f"G_{m_idx}"]
                columns.append((V * G)[mask])
        design = np.column_stack(columns)
        target = measured[..., component][mask]
        coefficients = normalized_stlsq_physical(
            design, target, threshold=threshold, max_iter=max_iter,
        )
        for col_idx, key in enumerate(coef_keys):
            coef_map[key][component] = coefficients[col_idx]
    return coef_map


def _zero_coefficients(
    candidate_names: tuple[str, ...],
    g_modifier_indices: tuple[int, ...],
) -> dict[str, np.ndarray]:
    return {
        f"{name}_m{m_idx}": np.zeros(2, dtype=float)
        for name in candidate_names
        for m_idx in g_modifier_indices
    }


def _fitted_from_coef(
    mid_fields: dict[str, np.ndarray],
    coef_map: dict[str, np.ndarray],
    candidate_names: tuple[str, ...],
    *,
    G_modifiers: dict[str, np.ndarray],
    g_modifier_indices: tuple[int, ...],
) -> np.ndarray:
    first = mid_fields[candidate_names[0]]
    fitted = np.zeros_like(first, dtype=float)
    for name in candidate_names:
        V = mid_fields[name]
        for m_idx in g_modifier_indices:
            G = G_modifiers[f"G_{m_idx}"]
            c = coef_map[f"{name}_m{m_idx}"]
            fitted += c[None, None, None, :] * G[..., None] * V
    return fitted


def normalized_stlsq_physical(
    design: np.ndarray,
    measured: np.ndarray,
    *,
    threshold: float,
    max_iter: int,
) -> np.ndarray:
    normalized_design, scales = _normalize_design_columns(design)
    normalized_coefficients = stlsq(
        normalized_design,
        measured,
        threshold=threshold,
        max_iter=max_iter,
    )
    return normalized_coefficients / scales


def _normalize_design_columns(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(design, dtype=float)
    finite = np.all(np.isfinite(design), axis=1)
    scales = np.ones(design.shape[1], dtype=float)
    if np.any(finite):
        rms = np.sqrt(np.nanmean(design[finite] ** 2, axis=0))
        valid_scales = np.isfinite(rms) & (rms > 0.0)
        scales[valid_scales] = rms[valid_scales]
    return design / scales[None, :], scales


def _validate_candidates(
    mid_fields: dict[str, np.ndarray],
    candidate_names: tuple[str, ...],
    measured_shape: tuple[int, ...],
) -> None:
    for name in candidate_names:
        if name not in mid_fields:
            raise KeyError(f"Candidate field {name!r} was not computed.")
        if mid_fields[name].shape != measured_shape:
            raise ValueError(
                f"Candidate {name!r} shape {mid_fields[name].shape} "
                f"does not match measured shape {measured_shape}."
            )


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
    if not np.any(x_mask) and not np.any(y_mask):
        return {
            "residual_x_mean_abs": float("nan"),
            "residual_y_mean_abs": float("nan"),
            "residual_x_median_abs": float("nan"),
            "residual_y_median_abs": float("nan"),
        }
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
