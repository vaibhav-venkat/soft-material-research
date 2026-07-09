"""Plot first-frame circumference-ring tilt versus C/D.

Each unwrapped initializer writes particles one axial row at a time.  A row
contains ``n_theta`` particles and is a circumference-wrapping ring.  This
module fits a plane to every such row in the first saved trajectory frame and
plots the axial component of its normal.

Usage::

    pixi run python -m hexatic.unwrapped_analysis.plot_ring
    pixi run python -m hexatic.unwrapped_analysis.plot_ring --output path/to/plot.html
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


OUTPUT_DIR = ANALYSIS_DIR / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "ring_tilt_vs_c_over_d.html"


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-ring |n.x|, tilt angle, and plane-fit RMS residual."""
    positions = np.asarray(positions, dtype=np.float64)
    assert positions.shape == (case.n_particles, 3)

    rings = positions.reshape(case.n_x, case.n_theta, 3)
    normal_dot_x = np.empty(case.n_x, dtype=np.float64)
    tilt_deg = np.empty(case.n_x, dtype=np.float64)
    plane_rms = np.empty(case.n_x, dtype=np.float64)

    for ring_idx, ring in enumerate(rings):
        centered = ring - np.mean(ring, axis=0)
        normal = np.linalg.svd(centered, full_matrices=False)[2][-1]
        dot_x = float(np.clip(abs(normal[0]), 0.0, 1.0))
        normal_dot_x[ring_idx] = dot_x
        tilt_deg[ring_idx] = float(np.degrees(np.arccos(dot_x)))
        plane_rms[ring_idx] = float(np.sqrt(np.mean((centered @ normal) ** 2)))

    return normal_dot_x, tilt_deg, plane_rms


def measure_case(case: UnwrappedCase) -> RingTiltCaseData:
    """Measure all initialized rings in the first saved frame for one case."""
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as trajectory:
        assert len(trajectory) > 0, f"empty trajectory: {case.trajectory_gsd}"
        frame = trajectory[0]

    positions = np.asarray(frame.particles.position, dtype=np.float64)
    normal_dot_x, tilt_deg, plane_rms = _fit_ring_plane_metrics(positions, case)
    return RingTiltCaseData(
        case_id=case.case_id,
        c_over_d=case.circumference / cylinder.PARTICLE_DIAMETER,
        step=int(frame.configuration.step),
        ring_count=case.n_x,
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


def build_figure(data: list[RingTiltCaseData]) -> go.Figure:
    """Build the two-panel first-frame ring-normal figure."""
    data = sorted(data, key=lambda item: item.c_over_d)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.13,
        subplot_titles=(
            "Mean axial normal component, ⟨|n · x̂|⟩ (first frame)",
            "Mean ring tilt, ⟨arccos(|n · x̂|)⟩ (first frame)",
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
        title=dict(text="First-frame circumference-ring tilt vs C/D", x=0.5, xanchor="center"),
        template="plotly_white",
        width=1000,
        height=780,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=90, r=40, t=95, b=75),
    )
    return figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot first-frame fitted circumference-ring normals versus C/D."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results: list[RingTiltCaseData] = []
    for case in all_cases():
        if not case.case_id.startswith("circ_"):
            continue
        if not case.trajectory_gsd.exists():
            print(f"[skip] missing {case.trajectory_gsd}")
            continue
        result = measure_case(case)
        results.append(result)
        print(
            f"{result.case_id}: C/D={result.c_over_d:.5f} "
            f"rings={result.ring_count} "
            f"mean|n.x|={result.mean_normal_dot_x:.10f} "
            f"mean tilt={result.mean_tilt_deg:.8f} deg "
            f"mean plane RMS={result.mean_plane_rms:.5e}"
        )

    assert results, "no unwrapped trajectory GSD files found"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_figure(results).write_html(output, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
