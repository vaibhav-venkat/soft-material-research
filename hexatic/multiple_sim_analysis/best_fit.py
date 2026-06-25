from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

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


def _pysr_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _pysr_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _format_equation_table(equations) -> str:
    preferred_columns = [
        column
        for column in ("complexity", "loss", "score", "equation")
        if column in equations.columns
    ]
    if preferred_columns:
        equations = equations[preferred_columns]
    sort_columns = [
        column for column in ("loss", "complexity") if column in equations.columns
    ]
    if sort_columns:
        equations = equations.sort_values(sort_columns, ascending=True)
    return equations.to_string(index=False)


def symbolic_regression_report(
    x_values: np.ndarray,
    named_values: dict[str, np.ndarray],
    output_txt: str | Path,
    title: str,
    x_label: str,
) -> None:
    output_path = Path(output_txt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        title,
        "=" * len(title),
        "",
        f"x variable: {x_label}",
        "model: PySRRegressor",
        "plotting: disabled; equations are written here only",
        "",
    ]
    try:
        from pysr import PySRRegressor
    except Exception as exc:
        lines.append(f"PySR import/setup failed: {type(exc).__name__}: {exc}")
        output_path.write_text("\n".join(lines) + "\n")
        return

    niterations = _pysr_int_env("MULTIPLE_SIM_ANALYSIS_PYSR_ITERATIONS", 100)
    maxsize = _pysr_int_env("MULTIPLE_SIM_ANALYSIS_PYSR_MAXSIZE", 20)
    timeout_seconds = _pysr_float_env(
        "MULTIPLE_SIM_ANALYSIS_PYSR_TIMEOUT_SECONDS",
        60.0,
    )

    lines.extend(
        [
            f"niterations: {niterations}",
            f"maxsize: {maxsize}",
            f"timeout_seconds: {timeout_seconds}",
            "sorted by: loss, then complexity",
            "",
        ]
    )

    x_values = np.asarray(x_values, dtype=np.float64)
    for series_name, series_values in named_values.items():
        series_values = np.asarray(series_values, dtype=np.float64)
        finite = np.isfinite(x_values) & np.isfinite(series_values)
        x_fit = x_values[finite]
        y_fit = series_values[finite]
        lines.extend([f"[{series_name}]", f"n_points: {x_fit.size}"])
        if x_fit.size < 3 or np.ptp(x_fit) <= 0.0:
            lines.extend(["skipped: fewer than 3 finite points or zero x span", ""])
            continue

        try:
            model = PySRRegressor(
                niterations=niterations,
                maxsize=maxsize,
                timeout_in_seconds=timeout_seconds,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "sqrt", "square"],
                model_selection="best",
                progress=False,
                verbosity=0,
            )
            model.fit(x_fit.reshape(-1, 1), y_fit, variable_names=[x_label])
            lines.append(_format_equation_table(model.equations_))
        except Exception as exc:
            lines.append(f"PySR fit failed: {type(exc).__name__}: {exc}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
