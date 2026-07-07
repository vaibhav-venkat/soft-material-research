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
SHELL_SLICE_OFFSET = 0


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
        title="P_fit vs P near outer shell",
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
        title="Q_fit vs Q near outer shell",
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

    radial_index = near_shell_radial_index(inputs.r_centers)
    fields = [
        (
            name,
            radial_slice_projection(fit_values, radial_index),
            radial_slice_projection(true_values, radial_index),
        )
        for name, fit_values, true_values in fields
    ]
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
                    colorbar={"title": name, "x": 1.02, "len": min(0.9, 0.75 / max(rows, 1))},
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
        "width": 1500,
        "height": max(520, rows * 320),
        "margin": {"l": 70, "r": 180, "t": 95, "b": 70},
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
    error_abs = padded_error_range(error_values, padding_fraction=0.50)

    def heatmap_pair(index: int) -> list[go.Heatmap]:
        cmin, cmax = padded_frame_range(fit_values[index], true_values[index], padding_fraction=0.20)
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
                colorbar={"title": colorbar_title, "x": 1.01, "len": 0.82, "y": 0.50},
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
                colorbar={"title": "error", "x": 1.10, "len": 0.82, "y": 0.50},
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
        width=1900,
        height=980,
        margin={"l": 70, "r": 260, "t": 95, "b": 70},
        xaxis={"title": "x", "domain": [0.00, 0.285]},
        yaxis={"title": y_title},
        xaxis2={"title": "x", "domain": [0.335, 0.620]},
        yaxis2={"title": y_title, "anchor": "x2"},
        xaxis3={"title": "x", "domain": [0.670, 0.955]},
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
    radial_index = near_shell_radial_index(inputs.r_centers)
    projected = radial_slice_projection(values, radial_index)
    x, theta = surface_axes(inputs.lx, inputs.theta_period, projected.shape[1:])
    radius = float(inputs.r_centers[radial_index])
    return projected, x, theta, f"outer-shell slice r={radius:.3g} as (x, theta)", "theta"


def padded_frame_range(fit_frame: Array, true_frame: Array, *, padding_fraction: float) -> tuple[float, float]:
    """Return a widened color range shared by fit and true fields for one frame."""
    values = np.concatenate((np.ravel(fit_frame), np.ravel(true_frame)))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    cmin, cmax = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
        center = float(np.nanmean(finite))
        span = max(abs(center), 1.0) * 1.0e-6
        return center - span, center + span
    padding = float(padding_fraction) * (cmax - cmin)
    return float(cmin - padding), float(cmax + padding)


def padded_error_range(error_values: Array, *, padding_fraction: float) -> float:
    """Return one widened symmetric error range shared across all animation frames."""
    finite = np.asarray(error_values[np.isfinite(error_values)], dtype=np.float64)
    if finite.size == 0:
        return 1.0
    high = float(np.percentile(np.abs(finite), 99.5))
    if not np.isfinite(high) or high <= 0.0:
        high = float(np.nanmax(np.abs(finite)))
    return max((1.0 + float(padding_fraction)) * high, 1.0e-12)


def animation_buttons(first_frame: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 250, "redraw": True}, "fromcurrent": True}],
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


def radial_slice_projection(values: Array, radial_index: int) -> Array:
    """Select one radial slice from time-indexed 3D scalar/component fields."""
    if values.ndim >= 4:
        return np.take(values, radial_index, axis=3)
    return values


def near_shell_radial_index(r_centers: Array) -> int:
    """Return the default near-shell radial bin index used by validation plots."""
    if r_centers.size <= 1:
        return 0
    return max(0, int(r_centers.size) - 1 - SHELL_SLICE_OFFSET)


def surface_axes(lx: float, theta_period: float, shape: tuple[int, int]) -> tuple[Array, Array]:
    nx, ny = shape
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, theta_period, ny, endpoint=False)
    return x, y
