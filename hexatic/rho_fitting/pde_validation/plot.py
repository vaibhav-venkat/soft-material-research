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
        title="P_fit vs P on the unwrapped cylinder surface",
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
        title="Q_fit vs Q on the unwrapped cylinder surface",
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

    frame_indices = np.arange(0, times.size, stride, dtype=int)
    x, y = surface_axes(inputs.lx, inputs.ly, fields[0][1].shape[1:])
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
        layout[f"yaxis{left_axis}"] = {"title": f"{name} fit, R theta"}
        layout[f"xaxis{right_axis}"] = {"title": "x"}
        layout[f"yaxis{right_axis}"] = {"title": f"{name}, R theta"}
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

    frame_indices = np.arange(0, times.size, stride, dtype=int)
    x, y = surface_axes(inputs.lx, inputs.ly, fit_values.shape[1:])
    cmin = float(np.nanmin([np.nanmin(fit_values), np.nanmin(true_values)]))
    cmax = float(np.nanmax([np.nanmax(fit_values), np.nanmax(true_values)]))

    def heatmap_pair(index: int) -> list[go.Heatmap]:
        return [
            go.Heatmap(
                x=x,
                y=y,
                z=fit_values[index].T,
                zmin=cmin,
                zmax=cmax,
                colorscale="Viridis",
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
                name=true_label,
                showscale=False,
                xaxis="x2",
                yaxis="y2",
            ),
        ]

    fig = go.Figure(data=heatmap_pair(int(frame_indices[0])))
    fig.frames = [
        go.Frame(data=heatmap_pair(int(index)), name=str(int(index)))
        for index in frame_indices
    ]
    first_frame = str(int(frame_indices[0]))
    fig.update_layout(
        title=f"{fit_label} vs {true_label} on the unwrapped cylinder surface",
        grid={"rows": 1, "columns": 2, "pattern": "independent"},
        xaxis={"title": "x", "domain": [0.0, 0.47]},
        yaxis={"title": "R theta"},
        xaxis2={"title": "x", "domain": [0.53, 1.0]},
        yaxis2={"title": "R theta", "anchor": "x2"},
        annotations=[
            {"text": fit_label, "x": 0.235, "y": 1.08, "xref": "paper", "yref": "paper", "showarrow": False},
            {"text": true_label, "x": 0.765, "y": 1.08, "xref": "paper", "yref": "paper", "showarrow": False},
        ],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": animation_buttons(first_frame),
            }
        ],
    )
    fig.write_html(path)


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


def surface_axes(lx: float, ly: float, shape: tuple[int, int]) -> tuple[Array, Array]:
    nx, ny = shape
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    return x, y
