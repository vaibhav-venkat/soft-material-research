from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go

from .continuity import FilmContinuityResult


PLOT_QUANTITIES = (
    "partial_t_rho_film_b",
    "neg_div_J_film_b",
    "S_cross_b",
    "residual_b",
    "total_b"
)

QUANTITY_LABELS = {
    "partial_t_rho_film_b": "partial_t rho_film",
    "neg_div_J_film_b": "-div J_film",
    "S_cross_b": "S_cross",
    "residual_b": "continuity residual",
    "total_b": "Total"
}


def write_quantity_map(
    result: FilmContinuityResult,
    quantity: str,
    output_path: str | Path,
    *,
    frame_idx: int | None = None,
    quiver_stride: int = 4,
) -> Path:
    values = _quantity_values(result, quantity)
    heatmap_values = _select_transition_values(values, frame_idx)
    current = _select_transition_values(result.J_film_b, frame_idx)
    title_suffix = _frame_title_suffix(result, frame_idx)

    x_centers = np.asarray(result.x_centers, dtype=float)
    theta_centers = np.asarray(result.theta_centers, dtype=float)
    signed_z = np.asarray(heatmap_values, dtype=float).T
    magnitude_z = np.abs(signed_z)

    signed_limit = float(np.nanmax(np.abs(signed_z))) if signed_z.size else 0.0
    if not np.isfinite(signed_limit) or signed_limit == 0.0:
        signed_limit = 1.0
    magnitude_limit = float(np.nanmax(magnitude_z)) if magnitude_z.size else 0.0
    if not np.isfinite(magnitude_limit) or magnitude_limit == 0.0:
        magnitude_limit = 1.0

    signed_heatmap = go.Heatmap(
        x=x_centers,
        y=theta_centers,
        z=signed_z,
        colorscale="RdBu_r",
        zmid=0.0,
        zmin=-signed_limit,
        zmax=signed_limit,
        colorbar={"title": QUANTITY_LABELS.get(quantity, quantity)},
        name="signed",
        visible=True,
    )
    magnitude_heatmap = go.Heatmap(
        x=x_centers,
        y=theta_centers,
        z=magnitude_z,
        colorscale="Viridis",
        zmin=0.0,
        zmax=magnitude_limit,
        colorbar={"title": f"|{QUANTITY_LABELS.get(quantity, quantity)}|"},
        name="magnitude",
        visible=False,
    )

    figure = go.Figure(data=[signed_heatmap, magnitude_heatmap])
    for trace in _quiver_traces(
        x_centers,
        theta_centers,
        current,
        cylinder_radius=float(result.cylinder_radius),
        stride=quiver_stride,
    ):
        figure.add_trace(trace)

    n_quiver_traces = len(figure.data) - 2
    figure.update_layout(
        title=f"{QUANTITY_LABELS.get(quantity, quantity)} {title_suffix}",
        xaxis_title="x",
        yaxis_title="theta",
        template="plotly_white",
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.0,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "signed",
                        "method": "update",
                        "args": [
                            {"visible": [True, False] + [True] * n_quiver_traces},
                            {"title": f"{QUANTITY_LABELS.get(quantity, quantity)} {title_suffix}"},
                        ],
                    },
                    {
                        "label": "magnitude",
                        "method": "update",
                        "args": [
                            {"visible": [False, True] + [True] * n_quiver_traces},
                            {
                                "title": (
                                    f"|{QUANTITY_LABELS.get(quantity, quantity)}| "
                                    f"{title_suffix}"
                                )
                            },
                        ],
                    },
                ],
            }
        ],
    )
    figure.update_yaxes(range=[theta_centers[0], theta_centers[-1]])

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(destination)
    return destination


def write_all_maps(
    result: FilmContinuityResult,
    output_dir: str | Path,
    *,
    frame_idx: int | None = None,
    quantities: Iterable[str] = PLOT_QUANTITIES,
    case_id: str = "radius_15D",
) -> list[Path]:
    destination_dir = Path(output_dir)
    written: list[Path] = []
    for quantity in quantities:
        output_path = (
            destination_dir / f"{case_id}_film_continuity_map_{quantity}.html"
        )
        written.append(
            write_quantity_map(
                result,
                quantity,
                output_path,
                frame_idx=frame_idx,
            )
        )
    return written


def _quantity_values(result: FilmContinuityResult, quantity: str) -> np.ndarray:
    if quantity not in PLOT_QUANTITIES:
        raise ValueError(f"Unsupported film-continuity quantity: {quantity}")
    return np.asarray(getattr(result, quantity), dtype=float)


def _select_transition_values(values: np.ndarray, frame_idx: int | None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if frame_idx is None:
        return np.nanmean(values, axis=0)
    if frame_idx < 0 or frame_idx >= values.shape[0]:
        raise IndexError(
            f"frame_idx={frame_idx} is outside transition range 0..{values.shape[0] - 1}."
        )
    return values[frame_idx]


def _frame_title_suffix(
    result: FilmContinuityResult,
    frame_idx: int | None,
) -> str:
    if frame_idx is None:
        return "(transition mean)"
    steps = np.asarray(result.transition_steps)
    if steps.ndim == 2 and frame_idx < steps.shape[0]:
        return f"(steps {steps[frame_idx, 0]} -> {steps[frame_idx, 1]})"
    return f"(transition {frame_idx})"


def _quiver_traces(
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
    current: np.ndarray,
    *,
    cylinder_radius: float,
    stride: int,
) -> list[go.Scatter]:
    if stride < 1:
        raise ValueError("quiver_stride must be at least 1.")
    if current.shape != (x_centers.size, theta_centers.size, 2):
        raise ValueError("J_film_b must resolve to shape (nx, ntheta, 2).")

    sampled_x = x_centers[::stride]
    sampled_theta = theta_centers[::stride]
    sampled_current = current[::stride, ::stride, :]
    grid_x, grid_theta = np.meshgrid(sampled_x, sampled_theta, indexing="ij")
    u = sampled_current[..., 0]
    v_theta = sampled_current[..., 1] / cylinder_radius
    speed = np.hypot(u, v_theta)
    max_speed = float(np.nanmax(speed)) if speed.size else 0.0
    if not np.isfinite(max_speed) or max_speed == 0.0:
        return []

    dx = _typical_spacing(x_centers) * stride
    dtheta = _typical_spacing(theta_centers) * stride
    scale = 0.45 * min(dx, dtheta) / max_speed
    quiver = ff.create_quiver(
        grid_x.ravel(),
        grid_theta.ravel(),
        (u * scale).ravel(),
        (v_theta * scale).ravel(),
        scale=1.0,
        arrow_scale=0.28,
        line={"color": "rgba(20, 20, 20, 0.65)", "width": 1},
        name="J_film",
    )
    for trace in quiver.data:
        trace.showlegend = False
        trace.hoverinfo = "skip"
    return list(quiver.data)


def _typical_spacing(centers: np.ndarray) -> float:
    if centers.size < 2:
        return 1.0
    spacing = np.diff(np.asarray(centers, dtype=float))
    return float(np.nanmedian(spacing))
