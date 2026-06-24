from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class FitCurve:
    radii: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ExponentialFit:
    y_inf: float
    amplitude: float
    length_scale: float
    radii: np.ndarray
    values: np.ndarray
    fit_radii: np.ndarray
    fit_values: np.ndarray
    rms_residual: float

    def as_payload(self, prefix: str = "fit") -> dict[str, np.ndarray | float | int]:
        return {
            f"{prefix}_success": 1,
            f"{prefix}_y_inf": self.y_inf,
            f"{prefix}_amplitude": self.amplitude,
            f"{prefix}_length_scale": self.length_scale,
            f"{prefix}_radii": self.fit_radii,
            f"{prefix}_values": self.fit_values,
            f"{prefix}_rms_residual": self.rms_residual,
        }


def exponential_model(
    radius: np.ndarray,
    y_inf: float,
    amplitude: float,
    length_scale: float,
) -> np.ndarray:
    return y_inf + amplitude * np.exp(-radius / length_scale)


def fit_exponential_radius_trend(
    radii: np.ndarray,
    values: np.ndarray,
    n_curve_points: int = 200,
) -> ExponentialFit | None:
    radii = np.asarray(radii, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(radii) & np.isfinite(values)
    fit_radii = radii[finite]
    fit_values = values[finite]
    if fit_radii.size < 4 or np.ptp(fit_radii) <= 0.0:
        return None

    order = np.argsort(fit_radii)
    fit_radii = fit_radii[order]
    fit_values = fit_values[order]

    tail_start = max(0, int(0.75 * fit_values.size))
    y_inf_guess = float(np.median(fit_values[tail_start:]))
    amplitude_guess = float(fit_values[0] - y_inf_guess)
    if amplitude_guess == 0.0:
        amplitude_guess = float(np.max(fit_values) - np.min(fit_values))
    if amplitude_guess == 0.0:
        amplitude_guess = 1.0

    radius_span = float(np.ptp(fit_radii))
    length_min = max(np.finfo(float).eps, radius_span / 1.0e6)
    length_guess = max(radius_span / 2.0, length_min)

    try:
        params, _ = curve_fit(
            exponential_model,
            fit_radii,
            fit_values,
            p0=(y_inf_guess, amplitude_guess, length_guess),
            bounds=([-np.inf, -np.inf, length_min], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    y_inf, amplitude, length_scale = (float(value) for value in params)
    modeled = exponential_model(fit_radii, y_inf, amplitude, length_scale)
    residuals = fit_values - modeled
    curve_radii = np.linspace(float(fit_radii.min()), float(fit_radii.max()), n_curve_points)
    curve_values = exponential_model(curve_radii, y_inf, amplitude, length_scale)
    return ExponentialFit(
        y_inf=y_inf,
        amplitude=amplitude,
        length_scale=length_scale,
        radii=curve_radii,
        values=curve_values,
        fit_radii=fit_radii,
        fit_values=fit_values,
        rms_residual=float(np.sqrt(np.mean(residuals**2))),
    )


def fit_payload(
    radii: np.ndarray,
    named_values: dict[str, np.ndarray],
) -> tuple[dict[str, ExponentialFit | None], dict[str, np.ndarray | float | int]]:
    fits: dict[str, ExponentialFit | None] = {}
    payload: dict[str, np.ndarray | float | int] = {}
    for name, values in named_values.items():
        fit = fit_exponential_radius_trend(radii, values)
        fits[name] = fit
        prefix = f"fit_{name}"
        if fit is None:
            payload[f"{prefix}_success"] = 0
            continue
        payload.update(fit.as_payload(prefix=prefix))
    return fits, payload
