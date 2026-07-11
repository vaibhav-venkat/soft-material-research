"""Plot circumference-ring tilt versus C/D.

Each unwrapped initializer writes particles one axial row at a time. A row
contains ``n_theta`` particles and is a circumference-wrapping ring. This
module fits a plane to every such row in either the exact initial state or the
first saved trajectory frame and plots the axial component of its normal.

Usage::

    pixi run python -m hexatic.unwrapped_analysis.plot_ring
    pixi run python -m hexatic.unwrapped_analysis.plot_ring --source trajectory --frame 10
    pixi run python -m hexatic.unwrapped_analysis.plot_ring --source trajectory --all-frames
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import gsd.hoomd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hexatic.constants import cylinder

from .cases import ANALYSIS_DIR, UnwrappedCase, all_cases
from .simulate_case_perfect_hexatic import generate_unwrapped_lattice


OUTPUT_DIR = ANALYSIS_DIR / "output"
DEFAULT_SOURCE = "initial"


@dataclass(frozen=True)
class RingTiltCaseData:
    case_id: str
    c_over_d: float
    step: int
    ring_count: int
    mean_normal_dot_x: float
    std_normal_dot_x: float
    mean_tilt_deg: float
    std_tilt_deg: float
    mean_plane_rms: float
    max_plane_rms: float


def _fit_ring_plane_metrics(
    positions: np.ndarray,
    case: UnwrappedCase,
    box_length_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-ring metrics from geometric circumferential cycles.

    Perfect-supercell particles are not written in axial-row order, so a
    reshape into ``(n_x, n_theta)`` would mix unrelated lattice sites.
    """
    positions = np.asarray(positions, dtype=np.float64)
    assert positions.shape == (case.plot_n_particles, 3)
    metrics: list[tuple[float, float, float]] = []
    for ring in _selected_perfect_ring_cycles(case):
        ring_positions = positions[ring]
        centered = ring_positions - np.mean(ring_positions, axis=0)
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        dot_x = float(np.clip(abs(normal[0]), 0.0, 1.0))
        metrics.append(
            (
                dot_x,
                float(np.degrees(np.arccos(dot_x))),
                float(np.sqrt(np.mean((centered @ normal) ** 2))),
            )
        )

    if not metrics:
        raise ValueError(f"No perfect-supercell circumference paths found for {case.case_id}")
    values = np.asarray(metrics, dtype=np.float64)
    normal_dot_x, tilt_deg, plane_rms = values.T

    return normal_dot_x, tilt_deg, plane_rms


@lru_cache(maxsize=None)
def _perfect_supercell_ring_cycles(case: UnwrappedCase) -> tuple[np.ndarray, ...]:
    """Build circumference paths from the exact twisted lattice vectors."""
    positions, _ = generate_unwrapped_lattice(case)
    supercell = np.column_stack(
        (case.circumference_lattice_vector, case.axial_lattice_vector)
    ).astype(float)
    inverse = np.linalg.inv(supercell)
    theta = np.mod(np.arctan2(positions[:, 1], positions[:, 2]), 2.0 * np.pi)
    fractions = np.column_stack(
        (theta / (2.0 * np.pi), positions[:, 0] / case.perfect_hexatic_lx + 0.5)
    )
    lattice_coordinates = np.rint(fractions @ supercell.T).astype(np.int64)

    def key(coordinate: np.ndarray) -> tuple[float, float]:
        fraction = np.mod(inverse @ coordinate, 1.0)
        fraction[np.isclose(fraction, 1.0, atol=1.0e-8)] = 0.0
        return tuple(np.round(fraction, 8))

    particle_by_key = {
        key(coordinate): index
        for index, coordinate in enumerate(lattice_coordinates)
    }
    cycles: list[np.ndarray] = []
    for coordinate in lattice_coordinates:
        current = coordinate.copy()
        path: list[int] = []
        for _ in range(case.n_theta - 1):
            path.append(particle_by_key[key(current)])
            current += (0, 1)
        path.append(particle_by_key[key(current)])
        cycles.append(np.asarray(path, dtype=np.int64))
    return tuple(cycles)


@lru_cache(maxsize=None)
def _selected_perfect_ring_cycles(case: UnwrappedCase) -> tuple[np.ndarray, ...]:
    """Select a fixed, initially planar path sample for every later frame."""
    initial_positions, _ = generate_unwrapped_lattice(case)
    selected: list[np.ndarray] = []
    for ring in _perfect_supercell_ring_cycles(case):
        centered = initial_positions[ring] - np.mean(initial_positions[ring], axis=0)
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        plane_rms = float(np.sqrt(np.mean((centered @ normal) ** 2)))
        if plane_rms <= 0.5 * cylinder.PARTICLE_DIAMETER:
            selected.append(ring)
    if not selected:
        raise ValueError(f"No initially planar circumference paths for {case.case_id}")
    stride = max(1, len(selected) // 256)
    return tuple(selected[::stride])


def measure_case(
    case: UnwrappedCase,
    input_gsd: Path,
    frame_index: int,
) -> RingTiltCaseData:
    """Measure all initialized rings in one selected GSD frame."""
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as trajectory:
        assert 0 <= frame_index < len(trajectory), (
            f"frame {frame_index} outside [0, {len(trajectory)}) for {input_gsd}"
        )
        frame = trajectory[frame_index]

    positions = np.asarray(frame.particles.position, dtype=np.float64)
    normal_dot_x, tilt_deg, plane_rms = _fit_ring_plane_metrics(
        positions, case, float(frame.configuration.box[0])
    )
    return RingTiltCaseData(
        case_id=case.case_id,
        c_over_d=case.circumference / cylinder.PARTICLE_DIAMETER,
        step=int(frame.configuration.step),
        ring_count=len(normal_dot_x),
        mean_normal_dot_x=float(np.mean(normal_dot_x)),
        std_normal_dot_x=float(np.std(normal_dot_x)),
        mean_tilt_deg=float(np.mean(tilt_deg)),
        std_tilt_deg=float(np.std(tilt_deg)),
        mean_plane_rms=float(np.mean(plane_rms)),
        max_plane_rms=float(np.max(plane_rms)),
    )


def _hover_template(quantity: str) -> str:
    return (
        "case=%{text}<br>"
        "C/D=%{x:.5f}<br>"
        f"{quantity}=%{{y:.8g}}<br>"
        "ring count=%{customdata[0]:d}<br>"
        "frame step=%{customdata[1]:d}<br>"
        "ring std=%{customdata[2]:.5g}<br>"
        "mean plane RMS=%{customdata[3]:.5g}<br>"
        "max plane RMS=%{customdata[4]:.5g}<extra></extra>"
    )


def _add_metric_trace(
    figure: go.Figure,
    row: int,
    data: list[RingTiltCaseData],
    values: np.ndarray,
    std: np.ndarray,
    name: str,
    color: str,
) -> None:
    figure.add_trace(
        go.Scatter(
            x=np.asarray([item.c_over_d for item in data]),
            y=values,
            mode="lines+markers+text",
            name=name,
            text=[item.case_id for item in data],
            textposition="top center",
            textfont=dict(size=11, color="#374151"),
            marker=dict(size=11, color=color, line=dict(width=1.2, color="white")),
            line=dict(color=color, width=2),
            error_y=dict(type="data", array=std, visible=True, thickness=1.2),
            customdata=np.asarray(
                [
                    (
                        item.ring_count,
                        item.step,
                        item.std_normal_dot_x
                        if row == 1
                        else item.std_tilt_deg,
                        item.mean_plane_rms,
                        item.max_plane_rms,
                    )
                    for item in data
                ],
                dtype=float,
            ),
            hovertemplate=_hover_template(name),
        ),
        row=row,
        col=1,
    )


def build_figure(
    data: list[RingTiltCaseData],
    source: str,
    frame_index: int,
) -> go.Figure:
    """Build the two-panel ring-normal figure."""
    data = sorted(data, key=lambda item: item.c_over_d)
    frame_label = (
        "initial state (frame 0)"
        if source == "initial"
        else f"trajectory frame {frame_index}"
    )
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        subplot_titles=(
            f"Mean axial normal component, ⟨|n · x̂|⟩ ({frame_label})",
            f"Mean ring tilt, ⟨arccos(|n · x̂|)⟩ ({frame_label})",
        ),
    )

    _add_metric_trace(
        figure,
        row=1,
        data=data,
        values=np.asarray([item.mean_normal_dot_x for item in data]),
        std=np.asarray([item.std_normal_dot_x for item in data]),
        name="mean |n · x̂|",
        color="#2563eb",
    )
    _add_metric_trace(
        figure,
        row=2,
        data=data,
        values=np.asarray([item.mean_tilt_deg for item in data]),
        std=np.asarray([item.std_tilt_deg for item in data]),
        name="mean tilt (degrees)",
        color="#dc2626",
    )

    figure.add_hline(y=1.0, row=1, col=1, line=dict(color="#6b7280", dash="dot"))
    figure.add_hline(y=0.0, row=2, col=1, line=dict(color="#6b7280", dash="dot"))
    figure.update_xaxes(
        title_text="C / D  (circumference / particle diameter)",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="rgba(39,49,61,0.10)",
        zeroline=False,
    )
    for row in (1, 2):
        figure.update_yaxes(
            showgrid=True,
            gridcolor="rgba(39,49,61,0.10)",
            zeroline=False,
            row=row,
            col=1,
        )
    figure.update_yaxes(title_text="mean |n · x̂|", row=1, col=1)
    figure.update_yaxes(title_text="mean tilt (degrees)", row=2, col=1)
    figure.update_layout(
        title=dict(text=f"Circumference-ring tilt vs C/D ({frame_label})", x=0.5, xanchor="center"),
        template="plotly_white",
        width=1000,
        height=780,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=90, r=40, t=95, b=75),
    )
    return figure


def build_time_figure(
    series: dict[str, list[tuple[int, RingTiltCaseData]]],
) -> go.Figure:
    """Plot mean angular tilt and its time derivative over trajectory frames."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Mean circumference-ring tilt",
            "Angular rate of mean ring tilt",
        ),
    )
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    ordered = sorted(series.values(), key=lambda rows: rows[0][1].c_over_d)
    for color, indexed_rows in zip(colors, ordered):
        frame_indices = np.asarray([frame_index for frame_index, _ in indexed_rows])
        rows = [row for _, row in indexed_rows]
        tilt_deg = np.asarray([row.mean_tilt_deg for row in rows])
        time = np.asarray([row.step for row in rows], dtype=float) * cylinder.TIMESTEP
        angular_velocity = np.gradient(tilt_deg, time)
        figure.add_trace(
            go.Scatter(
                x=frame_indices,
                y=tilt_deg,
                mode="lines+markers",
                name=rows[0].case_id,
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                customdata=np.asarray(
                    [
                        (row.step, row.std_tilt_deg, row.mean_plane_rms)
                        for row in rows
                    ],
                    dtype=float,
                ),
                hovertemplate=(
                    "case=%{fullData.name}<br>"
                    "frame=%{x:d}<br>"
                    "mean tilt=%{y:.6g} deg<br>"
                    "step=%{customdata[0]:d}<br>"
                    "ring tilt std=%{customdata[1]:.5g} deg<br>"
                    "mean plane RMS=%{customdata[2]:.5g}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame_indices,
                y=angular_velocity,
                mode="lines+markers",
                name=rows[0].case_id,
                legendgroup=rows[0].case_id,
                showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                customdata=np.asarray([row.step for row in rows], dtype=float),
                hovertemplate=(
                    "case=%{fullData.name}<br>"
                    "frame=%{x:d}<br>"
                    "tilt angular rate=%{y:.6g} deg / simulation time<br>"
                    "step=%{customdata:d}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    figure.update_xaxes(
        title_text="trajectory frame index",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="rgba(39,49,61,0.10)",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="mean ring tilt (degrees)",
        row=1,
        col=1,
        showgrid=True,
        gridcolor="rgba(39,49,61,0.10)",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="d(mean tilt) / dt (degrees / simulation time)",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="rgba(39,49,61,0.10)",
        zeroline=True,
        zerolinecolor="rgba(39,49,61,0.35)",
    )
    figure.update_layout(
        title=dict(text="Circumference-ring tilt and angular rate across trajectory frames", x=0.5, xanchor="center"),
        template="plotly_white",
        width=1080,
        height=940,
        hovermode="x unified",
        legend=dict(title_text="case"),
        margin=dict(l=90, r=40, t=90, b=75),
    )
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fitted circumference-ring normals versus C/D."
    )
    parser.add_argument(
        "--source",
        choices=("initial", "trajectory"),
        default=DEFAULT_SOURCE,
        help="Read initial/*.gsd or frame 0 of gsd/*.gsd (default: initial)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based GSD frame index (default: 0)",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Plot mean angular tilt for every saved trajectory frame",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="When using --all-frames, analyze every Nth frame and always include the final frame (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default depends on --source)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert args.frame >= 0, "--frame must be non-negative"
    assert args.interval > 0, "--interval must be positive"
    if args.all_frames:
        assert args.source == "trajectory", "--all-frames requires --source trajectory"
    results: list[RingTiltCaseData] = []
    series: dict[str, list[tuple[int, RingTiltCaseData]]] = {}
    for case in all_cases():
        if not case.case_id.startswith("circ_"):
            continue
        input_gsd = case.initial_gsd if args.source == "initial" else case.trajectory_gsd
        if not input_gsd.exists():
            print(f"[skip] missing {input_gsd}")
            continue
        if args.all_frames:
            with gsd.hoomd.open(name=str(input_gsd), mode="r") as trajectory:
                frame_count = len(trajectory)
            frame_indices = list(range(0, frame_count, args.interval))
            if frame_indices[-1] != frame_count - 1:
                frame_indices.append(frame_count - 1)
            series[case.case_id] = [
                (frame_index, measure_case(case, input_gsd, frame_index))
                for frame_index in frame_indices
            ]
            print(
                f"{case.case_id}: measured {len(frame_indices)} of {frame_count} "
                f"trajectory frames (interval={args.interval})"
            )
            continue
        result = measure_case(case, input_gsd, args.frame)
        results.append(result)
        print(
            f"{result.case_id}: C/D={result.c_over_d:.5f} "
            f"rings={result.ring_count} "
            f"mean|n.x|={result.mean_normal_dot_x:.10f} "
            f"mean tilt={result.mean_tilt_deg:.8f} deg "
            f"mean plane RMS={result.mean_plane_rms:.5e}"
        )

    if args.all_frames:
        assert series, "no unwrapped trajectory GSD files found"
        output = (
            Path(args.output)
            if args.output is not None
            else OUTPUT_DIR / "trajectory_ring_tilt_over_frames.html"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        build_time_figure(series).write_html(output, include_plotlyjs="cdn", full_html=True)
        print(f"Wrote {output}")
        return

    assert results, "no unwrapped trajectory GSD files found"
    output = (
        Path(args.output)
        if args.output is not None
        else OUTPUT_DIR / f"{args.source}_frame_{args.frame}_ring_tilt_vs_c_over_d.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    build_figure(results, args.source, args.frame).write_html(
        output,
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
