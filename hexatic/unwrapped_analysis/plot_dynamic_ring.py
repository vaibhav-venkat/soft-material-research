"""Plot rings reconstructed independently from every trajectory frame.

The original initializer particle rows eventually cease to be geometric rings.
This module instead rebuilds a periodic six-neighbor graph in each frame,
follows each particle's most circumferential forward neighbor, and retains
closed paths that wrap once around the cylinder without axial winding.

Usage::

    pixi run python -m hexatic.unwrapped_analysis.plot_dynamic_ring
    pixi run python -m hexatic.unwrapped_analysis.plot_dynamic_ring --interval 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import gsd.hoomd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree  # type: ignore[unresolved-import]

from .cases import ANALYSIS_DIR, UnwrappedCase, all_cases


OUTPUT_DIR = ANALYSIS_DIR / "output"


@dataclass(frozen=True)
class DynamicRingFrame:
    frame_index: int
    step: int
    ring_count: int
    mean_tilt_deg: float
    mean_plane_rms: float


def _wrapped_delta(values: np.ndarray, period: float) -> np.ndarray:
    return values - period * np.round(values / period)


def _periodic_six_neighbors(positions: np.ndarray, box_length_x: float) -> np.ndarray:
    """Return six nearest particle IDs under periodicity along the cylinder axis."""
    n_particles = len(positions)
    assert n_particles > 6
    shifts = np.asarray((-box_length_x, 0.0, box_length_x))
    search_points = np.concatenate(
        [positions + np.array((shift, 0.0, 0.0)) for shift in shifts],
        axis=0,
    )
    source_ids = np.tile(np.arange(n_particles, dtype=np.int64), len(shifts))
    _, hits = cKDTree(search_points).query(positions, k=7)
    neighbors = source_ids[np.asarray(hits, dtype=np.int64)]
    assert neighbors.shape == (n_particles, 7)
    assert np.all(neighbors[:, 0] == np.arange(n_particles))
    return neighbors[:, 1:]


def _directed_circumferential_links(
    positions: np.ndarray,
    cylinder_radius: float,
    box_length_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose one positive-theta, most-circumferential neighbor per particle."""
    neighbors = _periodic_six_neighbors(positions, box_length_x)
    theta = np.mod(np.arctan2(positions[:, 1], positions[:, 2]), 2.0 * np.pi)
    dx = positions[neighbors, 0] - positions[:, np.newaxis, 0]
    dx = _wrapped_delta(dx, box_length_x)
    dtheta = theta[neighbors] - theta[:, np.newaxis]
    dtheta = _wrapped_delta(dtheta, 2.0 * np.pi)
    ds = cylinder_radius * dtheta

    valid = ds > 1.0e-10
    slope = np.full(ds.shape, np.inf, dtype=np.float64)
    slope[valid] = np.abs(dx[valid]) / ds[valid]
    choice = np.argmin(slope, axis=1)
    particle_ids = np.arange(len(positions))
    next_ids = neighbors[particle_ids, choice]
    link_dx = dx[particle_ids, choice]
    link_ds = ds[particle_ids, choice]
    next_ids[~np.isfinite(slope[particle_ids, choice])] = -1
    return next_ids, link_dx, link_ds


def _functional_cycles(next_ids: np.ndarray) -> list[np.ndarray]:
    """Find unique directed cycles in a graph with at most one outgoing edge."""
    state = np.zeros(len(next_ids), dtype=np.int8)
    cycles: list[np.ndarray] = []

    for start in range(len(next_ids)):
        if state[start] != 0:
            continue
        path: list[int] = []
        path_index: dict[int, int] = {}
        current = start
        while current >= 0 and state[current] == 0:
            state[current] = 1
            path_index[current] = len(path)
            path.append(current)
            current = int(next_ids[current])

        if current >= 0 and state[current] == 1 and current in path_index:
            cycles.append(np.asarray(path[path_index[current] :], dtype=np.int64))
        for particle_id in path:
            state[particle_id] = 2
    return cycles


def reconstruct_ring_metrics(
    positions: np.ndarray,
    case: UnwrappedCase,
    box_length_x: float,
) -> tuple[int, float, float]:
    """Return count, mean tilt, and mean plane RMS for instantaneous ring loops."""
    positions = np.asarray(positions, dtype=np.float64)
    assert positions.shape == (case.n_particles, 3)
    next_ids, link_dx, link_ds = _directed_circumferential_links(
        positions,
        cylinder_radius=case.radius,
        box_length_x=box_length_x,
    )
    min_ring_particles = max(12, case.n_theta // 2)
    tilts: list[float] = []
    residuals: list[float] = []
    for cycle in _functional_cycles(next_ids):
        winding = float(np.sum(link_ds[cycle]) / case.circumference)
        axial_drift = float(np.sum(link_dx[cycle]))
        if len(cycle) < min_ring_particles:
            continue
        if not 0.85 <= winding <= 1.15:
            continue
        if abs(axial_drift) > 0.25 * case.a:
            continue

        ring = positions[cycle]
        centered = ring - np.mean(ring, axis=0)
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        dot_x = float(np.clip(abs(normal[0]), 0.0, 1.0))
        tilts.append(float(np.degrees(np.arccos(dot_x))))
        residuals.append(float(np.sqrt(np.mean((centered @ normal) ** 2))))

    if not tilts:
        return 0, float("nan"), float("nan")
    return len(tilts), float(np.mean(tilts)), float(np.mean(residuals))


def measure_case(case: UnwrappedCase, interval: int) -> list[DynamicRingFrame]:
    """Reconstruct instantaneous rings over selected frames of one trajectory."""
    frames: list[DynamicRingFrame] = []
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as trajectory:
        frame_indices = list(range(0, len(trajectory), interval))
        if frame_indices[-1] != len(trajectory) - 1:
            frame_indices.append(len(trajectory) - 1)
        for frame_index in frame_indices:
            frame = trajectory[frame_index]
            ring_count, mean_tilt, mean_plane_rms = reconstruct_ring_metrics(
                frame.particles.position,
                case,
                box_length_x=float(frame.configuration.box[0]),
            )
            frames.append(
                DynamicRingFrame(
                    frame_index=frame_index,
                    step=int(frame.configuration.step),
                    ring_count=ring_count,
                    mean_tilt_deg=mean_tilt,
                    mean_plane_rms=mean_plane_rms,
                )
            )
    return frames


def build_figure(series: dict[str, list[DynamicRingFrame]]) -> go.Figure:
    """Build instantaneous ring tilt and detected-ring-count panels."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Mean tilt of instantaneous circumference-wrapping rings",
            "Detected instantaneous ring count",
        ),
    )
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    for color, (case_id, rows) in zip(colors, series.items()):
        frame_indices = np.asarray([row.frame_index for row in rows])
        steps = np.asarray([row.step for row in rows])
        ring_count = np.asarray([row.ring_count for row in rows])
        tilt = np.asarray([row.mean_tilt_deg for row in rows])
        residual = np.asarray([row.mean_plane_rms for row in rows])
        hover_data = np.column_stack((steps, ring_count, residual))
        figure.add_trace(
            go.Scatter(
                x=frame_indices,
                y=tilt,
                mode="lines+markers",
                name=case_id,
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                customdata=hover_data,
                hovertemplate=(
                    "case=%{fullData.name}<br>frame=%{x:d}<br>"
                    "mean reconstructed-ring tilt=%{y:.6g} deg<br>"
                    "step=%{customdata[0]:d}<br>"
                    "detected rings=%{customdata[1]:d}<br>"
                    "mean plane RMS=%{customdata[2]:.5g}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame_indices,
                y=ring_count,
                mode="lines+markers",
                name=case_id,
                legendgroup=case_id,
                showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                customdata=np.column_stack((steps, residual)),
                hovertemplate=(
                    "case=%{fullData.name}<br>frame=%{x:d}<br>"
                    "detected rings=%{y:d}<br>step=%{customdata[0]:d}<br>"
                    "mean plane RMS=%{customdata[1]:.5g}<extra></extra>"
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
        title_text="ring count",
        row=2,
        col=1,
        showgrid=True,
        gridcolor="rgba(39,49,61,0.10)",
        zeroline=False,
    )
    figure.update_layout(
        title=dict(
            text="Instantaneously reconstructed circumference rings",
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=1080,
        height=920,
        hovermode="x unified",
        legend=dict(title_text="case"),
        margin=dict(l=90, r=40, t=95, b=75),
    )
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot rings reconstructed independently from each trajectory frame."
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
        default=OUTPUT_DIR / "trajectory_dynamic_ring_tilt_over_frames.html",
        help="Output HTML path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    assert args.interval > 0, "--interval must be positive"
    series: dict[str, list[DynamicRingFrame]] = {}
    for case in all_cases():
        if not case.case_id.startswith("circ_"):
            continue
        if not case.trajectory_gsd.exists():
            print(f"[skip] missing {case.trajectory_gsd}")
            continue
        frames = measure_case(case, args.interval)
        series[case.case_id] = frames
        detected = [frame.ring_count for frame in frames]
        print(
            f"{case.case_id}: frames={len(frames)} "
            f"detected-rings=({min(detected)}, {max(detected)})"
        )

    assert series, "no unwrapped trajectory GSD files found"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_figure(series).write_html(output, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
