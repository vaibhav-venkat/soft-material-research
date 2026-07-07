"""Plotly outputs for rho-fitting PDE validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import plotly.graph_objects as go

from .cache import ValidationInputs
from .model import ValidationResult


Array = NDArray[Any]


def write_rho_animation(
    path: Path,
    inputs: ValidationInputs,
    result: ValidationResult,
    *,
    stride: int = 1,
    overwrite: bool = False,
) -> None:
    write_scalar_animation(
        path,
        inputs,
        result.times,
        result.rho_fit,
        result.rho_true,
        fit_label="rho_fit",
        true_label="rho",
        colorbar_title="rho",
        overwrite=overwrite,
        stride=stride,
    )


def write_p_animation(
    path: Path,
    inputs: ValidationInputs,
    result: ValidationResult,
    *,
    stride: int = 1,
    overwrite: bool = False,
) -> None:
    fields = [
        ("|P|", np.linalg.norm(result.p_fit, axis=-1), np.linalg.norm(result.p_true, axis=-1)),
        ("P_x", result.p_fit[..., 0], result.p_true[..., 0]),
        ("P_theta", result.p_fit[..., 1], result.p_true[..., 1]),
        ("P_radial", result.p_fit[..., 2], result.p_true[..., 2]),
    ]
    write_component_animation(
        path,
        inputs,
        result.times,
        fields,
        title="P_fit vs P projected over radius",
        overwrite=overwrite,
        stride=stride,
    )


def write_q_animation(
    path: Path,
    inputs: ValidationInputs,
    result: ValidationResult,
    *,
    stride: int = 1,
    overwrite: bool = False,
) -> None:
    q_fit = result.q_fit
    q_true = result.q_true
    fields = [
        ("|Q|", np.linalg.norm(q_fit, axis=(-2, -1)), np.linalg.norm(q_true, axis=(-2, -1))),
        ("Q_xx", q_fit[..., 0, 0], q_true[..., 0, 0]),
        ("Q_tt", q_fit[..., 1, 1], q_true[..., 1, 1]),
        ("Q_rr", q_fit[..., 2, 2], q_true[..., 2, 2]),
        ("Q_xt", q_fit[..., 0, 1], q_true[..., 0, 1]),
        ("Q_xr", q_fit[..., 0, 2], q_true[..., 0, 2]),
        ("Q_tr", q_fit[..., 1, 2], q_true[..., 1, 2]),
    ]
    write_component_animation(
        path,
        inputs,
        result.times,
        fields,
        title="Q_fit vs Q projected over radius",
        overwrite=overwrite,
        stride=stride,
    )


def write_component_animation(
    path: Path,
    inputs: ValidationInputs,
    times: Array,
    fields: list[tuple[str, Array, Array]],
    *,
    title: str,
    stride: int = 1,
    overwrite: bool = False,
) -> None:
    assert fields, "at least one component field is required"
    assert stride > 0, "plot stride must be positive"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [(name, radial_projection(fit_values), radial_projection(true_values)) for name, fit_values, true_values in fields]
    frame_indices = np.arange(0, times.size, stride, dtype=int)
    x, y = surface_axes(inputs.lx, inputs.theta_period, fields[0][1].shape[1:])
    rows = len(fields)

    def heatmaps(index: int) -> list[go.Heatmap]:
        traces: list[go.Heatmap] = []
        for row, (name, fit_values, true_values) in enumerate(fields, start=1):
            cmin = float(np.nanmin([np.nanmin(fit_values), np.nanmin(true_values)]))
            cmax = float(np.nanmax([np.nanmax(fit_values), np.nanmax(true_values)]))
            traces.append(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=fit_values[index].T,
                    zmin=cmin,
                    zmax=cmax,
                    colorscale="Viridis",
                    name=f"{name} fit",
                    showscale=True,
                    colorbar={"title": name},
                    xaxis=f"x{2 * row - 1 if row > 1 else ''}",
                    yaxis=f"y{2 * row - 1 if row > 1 else ''}",
                )
            )
            traces.append(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=true_values[index].T,
                    zmin=cmin,
                    zmax=cmax,
                    colorscale="Viridis",
                    name=name,
                    showscale=False,
                    xaxis=f"x{2 * row}",
                    yaxis=f"y{2 * row}",
                )
            )
        return traces

    fig = go.Figure(data=heatmaps(int(frame_indices[0])))
    fig.frames = [
        go.Frame(data=heatmaps(int(index)), name=str(int(index)))
        for index in frame_indices
    ]
    first_frame = str(int(frame_indices[0]))
    layout: dict[str, Any] = {
        "title": title,
        "height": max(360, rows * 260),
        "grid": {"rows": rows, "columns": 2, "pattern": "independent"},
        "updatemenus": [
            {
                "type": "buttons",
                "buttons": animation_buttons(first_frame),
            }
        ],
    }
    for row, (name, _, _) in enumerate(fields, start=1):
        left_axis = "" if row == 1 else str(2 * row - 1)
        right_axis = str(2 * row)
        layout[f"xaxis{left_axis}"] = {"title": "x"}
        layout[f"yaxis{left_axis}"] = {"title": f"{name} fit, theta"}
        layout[f"xaxis{right_axis}"] = {"title": "x"}
        layout[f"yaxis{right_axis}"] = {"title": f"{name}, theta"}
    fig.update_layout(**layout)
    fig.write_html(path)


def write_scalar_animation(
    path: Path,
    inputs: ValidationInputs,
    times: Array,
    fit_values: Array,
    true_values: Array,
    *,
    fit_label: str,
    true_label: str,
    colorbar_title: str,
    stride: int = 1,
    overwrite: bool = False,
) -> None:
    assert stride > 0, "plot stride must be positive"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fit_values, x, y, scalar_title, y_title = scalar_plot_values(inputs, fit_values)
    true_values, _, _, _, _ = scalar_plot_values(inputs, true_values)
    error_values = fit_values - true_values
    frame_indices = np.arange(0, times.size, stride, dtype=int)
    global_cmin = float(np.nanmin([np.nanmin(fit_values), np.nanmin(true_values)]))
    global_cmax = float(np.nanmax([np.nanmax(fit_values), np.nanmax(true_values)]))
    error_abs = float(np.nanmax(np.abs(error_values)))

    def heatmap_pair(index: int) -> list[go.Heatmap]:
        cmin, cmax = robust_frame_range(fit_values[index], true_values[index], global_cmin, global_cmax)
        err_max = max(error_abs, 1.0e-12)
        return [
            go.Heatmap(
                x=x,
                y=y,
                z=fit_values[index].T,
                zmin=cmin,
                zmax=cmax,
                colorscale="Viridis",
                zsmooth="best",
                name=fit_label,
                showscale=True,
                colorbar={"title": colorbar_title},
                xaxis="x",
                yaxis="y",
            ),
            go.Heatmap(
                x=x,
                y=y,
                z=true_values[index].T,
                zmin=cmin,
                zmax=cmax,
                colorscale="Viridis",
                zsmooth="best",
                name=true_label,
                showscale=False,
                xaxis="x2",
                yaxis="y2",
            ),
            go.Heatmap(
                x=x,
                y=y,
                z=error_values[index].T,
                zmin=-err_max,
                zmax=err_max,
                colorscale="RdBu",
                reversescale=True,
                zsmooth="best",
                name=f"{fit_label} - {true_label}",
                showscale=True,
                colorbar={"title": "error", "x": 1.03},
                xaxis="x3",
                yaxis="y3",
            ),
        ]

    fig = go.Figure(data=heatmap_pair(int(frame_indices[0])))
    fig.frames = [
        go.Frame(data=heatmap_pair(int(index)), name=str(int(index)))
        for index in frame_indices
    ]
    first_frame = str(int(frame_indices[0]))
    fig.update_layout(
        title=f"{fit_label} vs {true_label} {scalar_title}",
        grid={"rows": 1, "columns": 3, "pattern": "independent"},
        xaxis={"title": "x", "domain": [0.0, 0.30]},
        yaxis={"title": y_title},
        xaxis2={"title": "x", "domain": [0.35, 0.65]},
        yaxis2={"title": y_title, "anchor": "x2"},
        xaxis3={"title": "x", "domain": [0.70, 1.0]},
        yaxis3={"title": y_title, "anchor": "x3"},
        annotations=[
            {"text": fit_label, "x": 0.15, "y": 1.08, "xref": "paper", "yref": "paper", "showarrow": False},
            {"text": true_label, "x": 0.50, "y": 1.08, "xref": "paper", "yref": "paper", "showarrow": False},
            {"text": f"{fit_label} - {true_label}", "x": 0.85, "y": 1.08, "xref": "paper", "yref": "paper", "showarrow": False},
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": animation_buttons(first_frame),
            }
        ],
    )
    fig.write_html(path)


def scalar_plot_values(inputs: ValidationInputs, values: Array) -> tuple[Array, Array, Array, str, str]:
    """Return scalar values and axes for the rho validation plot."""
    projected = radial_projection(values)
    x, theta = surface_axes(inputs.lx, inputs.theta_period, projected.shape[1:])
    r_ref = float(np.mean(inputs.r_centers))
    return projected, x, r_ref * theta, f"projected over radius as (x, {r_ref:.3g} theta)", "r_ref theta"


def robust_frame_range(fit_frame: Array, true_frame: Array, global_cmin: float, global_cmax: float) -> tuple[float, float]:
    """Return a shared per-frame color range that keeps small rho variations visible."""
    values = np.concatenate((np.ravel(fit_frame), np.ravel(true_frame)))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return global_cmin, global_cmax
    cmin, cmax = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
        center = float(np.nanmean(finite))
        span = max(global_cmax - global_cmin, abs(center), 1.0) * 1.0e-6
        return center - span, center + span
    padding = 0.02 * (cmax - cmin)
    return float(cmin - padding), float(cmax + padding)


def animation_buttons(first_frame: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
        },
        {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
        },
        {
            "label": "Reset",
            "method": "animate",
            "args": [[first_frame], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        },
    ]


def radial_projection(values: Array) -> Array:
    """Project time-indexed 3D scalar/component fields to `(T,Nx,Ntheta)` for plots."""
    if values.ndim >= 4:
        return np.nanmean(values, axis=3)
    return values


def surface_axes(lx: float, theta_period: float, shape: tuple[int, int]) -> tuple[Array, Array]:
    nx, ny = shape
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, theta_period, ny, endpoint=False)
    return x, y
