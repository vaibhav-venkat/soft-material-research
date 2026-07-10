"""Write one per-case plot comparing ring tilt definitions and net velocity.

Usage::

    pixi run python -m hexatic.unwrapped_analysis.plot_ring_tilt_velocity
    pixi run python -m hexatic.unwrapped_analysis.plot_ring_tilt_velocity --interval 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hexatic.constants import cylinder

from .cases import ANALYSIS_DIR, UnwrappedCase, all_cases
from .plot_dynamic_ring import reconstruct_ring_metrics


OUTPUT_DIR = ANALYSIS_DIR / "output"


@dataclass(frozen=True)
class CaseFrameMetrics:
    frame_index: int
    step: int
    initial_row_tilt_deg: float
    instantaneous_tilt_deg: float
    instantaneous_ring_count: int
    net_axial_velocity: float
    net_abs_axial_velocity: float


def _initial_row_tilt(positions: np.ndarray, case: UnwrappedCase) -> float:
    """Return the mean fitted-plane tilt of the initializer's particle-ID rows."""
    positions = np.asarray(positions, dtype=np.float64)
    assert positions.shape == (case.n_particles, 3)
    rings = positions.reshape(case.n_x, case.n_theta, 3)
    tilts = np.empty(case.n_x, dtype=np.float64)
    for ring_index, ring in enumerate(rings):
        centered = ring - np.mean(ring, axis=0)
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        dot_x = float(np.clip(abs(normal[0]), 0.0, 1.0))
        tilts[ring_index] = float(np.degrees(np.arccos(dot_x)))
    return float(np.mean(tilts))


def _net_axial_velocity(
    previous_positions: np.ndarray | None,
    previous_step: int | None,
    positions: np.ndarray,
    step: int,
    box_length_x: float,
) -> tuple[float, float]:
    """Return signed and absolute mean axial velocities from periodic displacement."""
    if previous_positions is None or previous_step is None:
        return float("nan"), float("nan")
    dt = (step - previous_step) * cylinder.TIMESTEP
    assert dt > 0.0
    dx = positions[:, 0] - previous_positions[:, 0]
    dx -= box_length_x * np.round(dx / box_length_x)
    return float(np.mean(dx) / dt), float(np.mean(np.abs(dx)) / dt)


def measure_case(case: UnwrappedCase, interval: int) -> list[CaseFrameMetrics]:
    """Measure both ring definitions and net x velocity at selected frames."""
    results: list[CaseFrameMetrics] = []
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as trajectory:
        frame_indices = list(range(0, len(trajectory), interval))
        if frame_indices[-1] != len(trajectory) - 1:
            frame_indices.append(len(trajectory) - 1)

        previous_positions: np.ndarray | None = None
        previous_step: int | None = None
        for frame_index in frame_indices:
            frame = trajectory[frame_index]
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            step = int(frame.configuration.step)
            box_length_x = float(frame.configuration.box[0])
            ring_count, instantaneous_tilt, _ = reconstruct_ring_metrics(
                positions,
                case,
                box_length_x=box_length_x,
            )
            net_velocity, net_abs_velocity = _net_axial_velocity(
                previous_positions,
                previous_step,
                positions,
                step,
                box_length_x,
            )
            results.append(
                CaseFrameMetrics(
                    frame_index=frame_index,
                    step=step,
                    initial_row_tilt_deg=_initial_row_tilt(positions, case),
                    instantaneous_tilt_deg=instantaneous_tilt,
                    instantaneous_ring_count=ring_count,
                    net_axial_velocity=net_velocity,
                    net_abs_axial_velocity=net_abs_velocity,
                )
            )
            previous_positions = positions
            previous_step = step
    return results


def build_figure(
    series: dict[str, tuple[UnwrappedCase, list[CaseFrameMetrics]]],
) -> go.Figure:
    """Build vertically stacked tilt and velocity panels for every case."""
    n_cases = len(series)

    figure = make_subplots(
        rows=2 * n_cases,
        cols=1,
        vertical_spacing=0.025,
        subplot_titles=tuple(
            title
            for case_id in series
            for title in (
                f"{case_id}: mean ring tilt",
                f"{case_id}: net axial system velocity",
            )
        ),
    )
    for case_index, (case_id, (_, rows)) in enumerate(series.items()):
        tilt_row = 2 * case_index + 1
        velocity_row = tilt_row + 1
        frame_indices = np.asarray([row.frame_index for row in rows])
        steps = np.asarray([row.step for row in rows])
        initial_tilt = np.asarray([row.initial_row_tilt_deg for row in rows])
        instantaneous_tilt = np.asarray([row.instantaneous_tilt_deg for row in rows])
        ring_count = np.asarray([row.instantaneous_ring_count for row in rows])
        velocity = np.asarray([row.net_axial_velocity for row in rows])
        abs_velocity = np.asarray([row.net_abs_axial_velocity for row in rows])
        show_legend = case_index == 0

        figure.add_trace(
            go.Scatter(
                x=frame_indices, y=initial_tilt, mode="lines+markers",
                name="initial particle-ID rows", legendgroup="initial", showlegend=show_legend,
                line=dict(color="#2563eb", width=2), marker=dict(size=4, color="#2563eb"),
                customdata=steps,
                hovertemplate="frame=%{x:d}<br>initial-row tilt=%{y:.6g} deg<br>step=%{customdata:d}<extra></extra>",
            ), row=tilt_row, col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame_indices, y=instantaneous_tilt, mode="lines+markers",
                name="instantaneous reconstructed rings", legendgroup="instantaneous", showlegend=show_legend,
                line=dict(color="#dc2626", width=2), marker=dict(size=4, color="#dc2626"),
                customdata=np.column_stack((steps, ring_count)),
                hovertemplate="frame=%{x:d}<br>instantaneous-ring tilt=%{y:.6g} deg<br>step=%{customdata[0]:d}<br>detected rings=%{customdata[1]:d}<extra></extra>",
            ), row=tilt_row, col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame_indices, y=velocity, mode="lines+markers",
                name="net ⟨v_x⟩", legendgroup="velocity", showlegend=show_legend,
                line=dict(color="#16a34a", width=2), marker=dict(size=4, color="#16a34a"),
                customdata=steps,
                hovertemplate="frame=%{x:d}<br>net ⟨v_x⟩=%{y:.6g}<br>step=%{customdata:d}<extra></extra>",
            ), row=velocity_row, col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame_indices, y=abs_velocity, mode="lines+markers",
                name="net ⟨|v_x|⟩", legendgroup="abs_velocity", showlegend=show_legend,
                line=dict(color="#9333ea", width=2), marker=dict(size=4, color="#9333ea"),
                customdata=steps,
                hovertemplate="frame=%{x:d}<br>net ⟨|v_x|⟩=%{y:.6g}<br>step=%{customdata:d}<extra></extra>",
            ), row=velocity_row, col=1,
        )
        figure.add_hline(y=0.0, row=velocity_row, col=1, line=dict(color="#6b7280", dash="dot"))
        figure.update_xaxes(
            title_text="trajectory frame index", row=velocity_row, col=1,
            showgrid=True, gridcolor="rgba(39,49,61,0.10)", zeroline=False,
        )
        figure.update_yaxes(
            title_text="mean tilt (degrees)", row=tilt_row, col=1,
            showgrid=True, gridcolor="rgba(39,49,61,0.10)", zeroline=False,
        )
        figure.update_yaxes(
            title_text="velocity (distance / simulation time)", row=velocity_row, col=1,
            showgrid=True, gridcolor="rgba(39,49,61,0.10)", zeroline=False,
        )
    figure.update_layout(
        title=dict(
            text="Ring tilt and net axial velocity by circumference case",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=1080,
        height=560 * n_cases,
        hovermode="x unified",
        legend=dict(title_text="quantity"),
        margin=dict(l=95, r=40, t=95, b=75),
    )
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a stacked all-case ring-tilt and net-velocity HTML plot."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Analyze every Nth frame and always include the final frame (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "ring_tilt_velocity_by_case.html",
        help="Output HTML path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert args.interval > 0, "--interval must be positive"
    series: dict[str, tuple[UnwrappedCase, list[CaseFrameMetrics]]] = {}
    for case in all_cases():
        if not case.case_id.startswith("circ_"):
            continue
        if not case.trajectory_gsd.exists():
            print(f"[skip] missing {case.trajectory_gsd}")
            continue
        rows = measure_case(case, args.interval)
        series[case.case_id] = (case, rows)
        print(f"{case.case_id}: frames={len(rows)}")

    assert series, "no unwrapped trajectory GSD files found"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_figure(series).write_html(output, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
