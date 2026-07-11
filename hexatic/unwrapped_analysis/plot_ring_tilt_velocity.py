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
from .plot_ring import _fit_ring_plane_metrics


OUTPUT_DIR = ANALYSIS_DIR / "output"
DEFAULT_MAX_LAG = 30


@dataclass(frozen=True)
class CaseFrameMetrics:
    frame_index: int
    step: int
    initial_row_tilt_deg: float
    instantaneous_tilt_deg: float
    instantaneous_ring_count: int
    net_axial_velocity: float
    net_abs_axial_velocity: float
    net_abs_axial_polarization: float


def _net_abs_axial_polarization(case: UnwrappedCase) -> dict[int, float]:
    """Return mean absolute axial polarization indexed by simulation step."""
    path = ANALYSIS_DIR / "npz_fields" / f"{case.case_id}_active_matter_fields.npz"
    with np.load(path) as data:
        steps = np.asarray(data["steps"], dtype=np.int64)
        directions = np.asarray(data["direction_cylindrical"], dtype=np.float64)
    values = np.nanmean(np.abs(directions[:, :, 0]), axis=1)
    return dict(zip(steps.tolist(), values.tolist(), strict=True))


def _initial_ring_tilt(
    positions: np.ndarray, case: UnwrappedCase, box_length_x: float
) -> float:
    """Return mean tilt of geometrically reconstructed initial rings."""
    _, tilt_deg, _ = _fit_ring_plane_metrics(positions, case, box_length_x)
    return float(np.mean(tilt_deg))


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
    abs_axial_polarization = _net_abs_axial_polarization(case)
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
                    initial_row_tilt_deg=_initial_ring_tilt(
                        positions, case, box_length_x
                    ),
                    instantaneous_tilt_deg=instantaneous_tilt,
                    instantaneous_ring_count=ring_count,
                    net_axial_velocity=net_velocity,
                    net_abs_axial_velocity=net_abs_velocity,
                    net_abs_axial_polarization=abs_axial_polarization[step],
                )
            )
            previous_positions = positions
            previous_step = step
    return results


def _lag_pair(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair a(t) with b(t + lag)."""
    if lag < 0:
        return a[-lag:], b[:lag]
    if lag > 0:
        return a[:-lag], b[lag:]
    return a, b


def _pearson_lag_curve(
    source: np.ndarray,
    target: np.ndarray,
    lags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Pearson r and valid-pair count for each lag."""
    correlation = np.full(len(lags), np.nan, dtype=np.float64)
    sample_count = np.zeros(len(lags), dtype=np.int64)
    for index, lag in enumerate(lags):
        x, y = _lag_pair(source, target, int(lag))
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        sample_count[index] = len(x)
        if len(x) >= 3 and np.std(x) > 0.0 and np.std(y) > 0.0:
            correlation[index] = float(np.corrcoef(x, y)[0, 1])
    return correlation, sample_count


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
                name="geometric circumference rings", legendgroup="initial", showlegend=show_legend,
                line=dict(color="#2563eb", width=2), marker=dict(size=4, color="#2563eb"),
                customdata=steps,
                hovertemplate="frame=%{x:d}<br>circumference-ring tilt=%{y:.6g} deg<br>step=%{customdata:d}<extra></extra>",
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


def build_lag_figure(
    series: dict[str, tuple[UnwrappedCase, list[CaseFrameMetrics]]],
    max_lag: int,
) -> go.Figure:
    """Plot net |v_x| and |P_x| versus initial-row tilt at each lag."""
    n_cases = len(series)
    max_supported_lag = min(len(rows) - 2 for _, rows in series.values())
    max_lag = min(max_lag, max_supported_lag)
    assert max_lag > 0, "at least three sampled frames are required for lag correlation"
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    figure = make_subplots(
        rows=n_cases,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=tuple(
            f"{case_id}: correlation with circumference-ring tilt(t + lag)"
            for case_id in series
        ),
    )
    for row_index, (case_id, (_, rows)) in enumerate(series.items(), start=1):
        abs_velocity = np.asarray([row.net_abs_axial_velocity for row in rows])
        initial_tilt = np.asarray([row.initial_row_tilt_deg for row in rows])
        step = np.asarray([row.step for row in rows], dtype=float)
        lag_time = lags * float(np.median(np.diff(step))) * cylinder.TIMESTEP
        correlation, sample_count = _pearson_lag_curve(
            abs_velocity,
            initial_tilt,
            lags,
        )
        abs_polarization = np.asarray(
            [row.net_abs_axial_polarization for row in rows]
        )
        polarization_correlation, polarization_sample_count = _pearson_lag_curve(
            abs_polarization,
            initial_tilt,
            lags,
        )
        for name, values, counts, color, dash in (
            (f"{case_id} ⟨|v_x|⟩", correlation, sample_count, "#2563eb", "solid"),
            (
                f"{case_id} ⟨|P_x|⟩",
                polarization_correlation,
                polarization_sample_count,
                "#dc2626",
                "dash",
            ),
        ):
            figure.add_trace(
                go.Scatter(
                    x=lag_time,
                    y=values,
                    mode="lines+markers",
                    name=name,
                    legendgroup=name.rsplit(" ", 1)[-1],
                    showlegend=row_index == 1,
                    line=dict(color=color, width=2.2, dash=dash),
                    marker=dict(size=5.5, color=color, line=dict(width=0.5, color="white")),
                    customdata=np.column_stack((lags, counts)),
                    hovertemplate=(
                        "%{fullData.name}<br>"
                        "lag=%{customdata[0]:d} sampled frames<br>"
                        "lag time=%{x:.6g}<br>"
                        "valid pairs=%{customdata[1]:d}<br>"
                        "Pearson r=%{y:.6f}<extra></extra>"
                    ),
                ),
                row=row_index,
                col=1,
            )
        figure.add_hline(
            y=0.0,
            row=row_index,
            col=1,
            line=dict(color="#64748b", width=1, dash="dash"),
        )
        figure.add_vline(
            x=0.0,
            row=row_index,
            col=1,
            line=dict(color="#111827", width=1.2),
        )
        figure.update_yaxes(
            title_text="Pearson r",
            range=[-1.05, 1.05],
            row=row_index,
            col=1,
            showgrid=True,
            gridcolor="rgba(39,49,61,0.10)",
            zeroline=False,
        )

    figure.update_xaxes(
        title_text="lag time (simulation time)",
        row=n_cases,
        col=1,
        showgrid=True,
        gridcolor="rgba(39,49,61,0.10)",
        zeroline=False,
    )
    figure.update_layout(
        title=dict(
            text=(
                "Lag correlation: net ⟨|v_x|⟩(t) and ⟨|P_x|⟩(t) "
                "with circumference-ring tilt(t + lag), "
                f"lags ±{max_lag} sampled frames"
            ),
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=1080,
        height=330 * n_cases,
        hovermode="closest",
        legend=dict(title_text="source quantity"),
        margin=dict(l=95, r=40, t=100, b=75),
    )
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a stacked all-case ring-tilt and net-velocity HTML plot."
    )
    parser.add_argument(
        "--plot",
        choices=("timeseries", "lag"),
        default="timeseries",
        help="Write the time series or its lag-correlation diagnostic",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_MAX_LAG,
        help=f"Maximum lag in sampled frames for --plot lag (default: {DEFAULT_MAX_LAG})",
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
        default=None,
        help="Output HTML path (default depends on --plot)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert args.interval > 0, "--interval must be positive"
    assert args.max_lag > 0, "--max-lag must be positive"
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
    default_name = (
        "ring_tilt_velocity_by_case.html"
        if args.plot == "timeseries"
        else "initial_ring_tilt_abs_velocity_lag_correlation.html"
    )
    output = Path(args.output) if args.output is not None else OUTPUT_DIR / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    figure = (
        build_figure(series)
        if args.plot == "timeseries"
        else build_lag_figure(series, args.max_lag)
    )
    figure.write_html(output, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
