"""Plot properties vs C/D (circumference/diameter) across all cases.

Auto-discovers cases from ``hexatic_output/*_hexatic_order.txt`` and plots:
- Disclination density +1 (first frame)
- Disclination density -1 (first frame)
- Mean hexatic order |psi6| (first frame)
- Average net x-velocity over frames [v0..v1)  (finite-difference from npz)
- Average |x-velocity| over frames [v0..v1)
- Average net P_x over frames [v0..v1)  (from ``direction_cylindrical[:, :, 0]``)
- Average |P_x| over frames [v0..v1)

Velocity is computed as Δx / (Δstep · TIMESTEP) where Δstep is read from
the npz ``steps`` array and ``TIMESTEP`` comes from ``hexatic.constants.cylinder``.

Usage::

    python -m hexatic.unwrapped_analysis.plot_cir --help
    python -m hexatic.unwrapped_analysis.plot_cir
    python -m hexatic.unwrapped_analysis.plot_cir -v0 500 -v1 1000 --output path/to/output.html
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hexatic.constants import cylinder as _cyl_constants


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

ANALYSIS_DIR = Path(__file__).resolve().parent
HEXATIC_OUTPUT_DIR = ANALYSIS_DIR / "hexatic_output"
NPZ_FIELDS_DIR = ANALYSIS_DIR / "npz_fields"
OUTPUT_DIR = ANALYSIS_DIR / "output"

DEFAULT_V0 = 500
DEFAULT_V1 = 1000

# HOOMD timestep — multiply frame-step delta by this to get physical time.
_TIMESTEP: float = float(_cyl_constants.TIMESTEP)


# ---------------------------------------------------------------------------
# Case discovery and C/D parsing
# ---------------------------------------------------------------------------


def _discover_case_ids(hexatic_dir: Path) -> list[str]:
    """Return sorted unique case-ids found in *hexatic_output*."""
    ids: set[str] = set()
    for path in hexatic_dir.glob("*_hexatic_order.txt"):
        stem = path.name
        # stem:  "{case_id}_hexatic_order.txt"
        if stem.endswith("_hexatic_order.txt"):
            case_id = stem[: -len("_hexatic_order.txt")]
            ids.add(case_id)
    return sorted(ids, key=_sort_key_for_case_id)


def _sort_key_for_case_id(case_id: str) -> float:
    """Stable sort key: circ cases before radius cases, then by C/D."""
    cd = parse_c_over_d(case_id)
    return cd


def parse_c_over_d(case_id: str) -> float:
    """Extract circumference/diameter ratio from a case identifier.

    ``circ_60_0D``  → 60.0   (C = 60.0 D)
    ``circ_60_25D`` → 60.25
    ``circ_60_5D``  → 60.5
    ``circ_60_75D`` → 60.75
    ``radius_15D``  → 2 π × 15 ≈ 94.2478
    """
    # circ_<int>_<frac>D  e.g. circ_60_0D, circ_60_25D
    m = re.fullmatch(r"circ_(\d+)_(\d+)D", case_id)
    if m:
        integer_part = int(m.group(1))
        frac_part = m.group(2)
        # "0" → 0.0, "25" → 0.25, "5" → 0.5, "75" → 0.75
        frac_value = int(frac_part) / (10 ** len(frac_part))
        return float(integer_part) + frac_value

    # radius_<int>D  e.g. radius_15D
    m = re.fullmatch(r"radius_(\d+)D", case_id)
    if m:
        radius_mult = int(m.group(1))
        return 2.0 * math.pi * float(radius_mult)

    raise ValueError(
        f"Cannot parse C/D from case_id {case_id!r}. "
        f"Expected 'circ_<N>_<F>D' or 'radius_<N>D'."
    )


# ---------------------------------------------------------------------------
# Per-frame data helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CaseData:
    case_id: str
    c_over_d: float
    n_particles: int
    disclination_plus1: float   # fraction of particles
    disclination_minus1: float  # fraction of particles
    mean_psi6: float
    mean_vx: float              # avg net x-velocity over the velocity frame window
    mean_abs_vx: float          # avg |x-velocity| over the velocity frame window
    mean_px: float              # avg net P_x over the frame window
    mean_abs_px: float          # avg |P_x| over the frame window


def _mean_hexatic_first_frame(hexatic_order_path: Path) -> tuple[float, int]:
    """Return (mean |psi6|, n_particles) for frame 0."""
    raw = np.loadtxt(hexatic_order_path, comments="#", ndmin=2)
    if raw.size == 0:
        return float("nan"), 0
    frame_col = raw[:, 0].astype(int)
    psi_abs = raw[:, 5]
    mask = frame_col == 0
    vals = psi_abs[mask]
    return float(np.mean(vals)) if len(vals) > 0 else float("nan"), len(vals)


def _disclination_density_first_frame(
    neighbor_counts_path: Path,
) -> tuple[float, float, int]:
    """Return (fraction +1, fraction -1, n_particles) for frame 0."""
    raw = np.loadtxt(neighbor_counts_path, comments="#", ndmin=2)
    if raw.size == 0:
        return float("nan"), float("nan"), 0
    frame_col = raw[:, 0].astype(int)
    nc = raw[:, 3].astype(int)
    mask = frame_col == 0
    counts = nc[mask]
    n_total = len(counts)
    if n_total == 0:
        return float("nan"), float("nan"), 0
    plus1 = float(np.sum(counts == 5)) / n_total
    minus1 = float(np.sum(counts == 7)) / n_total
    return plus1, minus1, n_total


def _polarization_from_npz(
    npz_path: Path,
    frame_start: int = DEFAULT_V0,
    frame_end: int = DEFAULT_V1,
) -> tuple[float, float]:
    """Return (mean net P_x, mean |P_x|) over frames [frame_start, frame_end).

    P_x is read from ``direction_cylindrical[:, :, 0]`` (same convention as
    ``plot.py``: component 0 = x, 1 = R, 2 = θ).

    Returns (nan, nan) when the file is missing or the window is too small.
    """
    try:
        with np.load(npz_path) as data:
            p = np.asarray(data["direction_cylindrical"], dtype=float)  # (T, N, 3)
    except (FileNotFoundError, KeyError):
        return float("nan"), float("nan")

    n_frames = p.shape[0]
    f0 = max(0, frame_start)
    f1 = min(n_frames, frame_end)
    if f1 <= f0:
        return float("nan"), float("nan")

    px_window = p[f0:f1, :, 0]  # (W, N)

    per_frame_net = np.mean(px_window, axis=1)          # (W,)
    per_frame_abs = np.mean(np.abs(px_window), axis=1)  # (W,)

    mean_px = float(np.mean(per_frame_net))
    mean_abs_px = float(np.mean(per_frame_abs))

    print(
        f"  [P_x] {npz_path.name:50s}  "
        f"shape={p.shape}  window=[{f0}:{f1}]  "
        f"⟨P_x⟩={mean_px:.6e}  ⟨|P_x|⟩={mean_abs_px:.6e}"
    )
    return mean_px, mean_abs_px


def _velocity_from_npz(
    npz_path: Path,
    frame_start: int = DEFAULT_V0,
    frame_end: int = DEFAULT_V1,
) -> tuple[float, float]:
    """Return (mean net v_x, mean net |v_x|) over frames [frame_start, frame_end).

    v_x = Δx / (Δstep · TIMESTEP), where Δstep is the difference between
    consecutive entries in the npz ``steps`` array and TIMESTEP is the
    HOOMD integration timestep from ``hexatic.constants.cylinder``.

    Returns (nan, nan) when the file is missing or there are too few frames.
    """
    try:
        with np.load(npz_path) as data:
            coords = np.asarray(data["coords"], dtype=float)  # (T, N, 3)
            steps = np.asarray(data["steps"], dtype=float)    # (T,)
    except (FileNotFoundError, KeyError):
        return float("nan"), float("nan")

    n_frames = coords.shape[0]
    f0 = max(0, frame_start)
    f1 = min(n_frames, frame_end)
    if f1 <= f0 + 1:
        return float("nan"), float("nan")

    # physical time between trajectory writes:  (steps[1] - steps[0]) * TIMESTEP
    step_diffs = np.diff(steps)
    if len(step_diffs) == 0:
        return float("nan"), float("nan")
    dt_per_frame = float(np.median(step_diffs)) * _TIMESTEP
    if dt_per_frame <= 0:
        return float("nan"), float("nan")

    # forward difference: displacement per frame in the window
    dx = np.diff(coords[f0:f1, :, 0], axis=0)  # (W-1, N)
    vx_window = dx / dt_per_frame                # (W-1, N)  actual velocity

    # per-frame net: mean across particles for each frame-to-frame step
    per_frame_net = np.mean(vx_window, axis=1)          # (W-1,)
    per_frame_abs = np.mean(np.abs(vx_window), axis=1)  # (W-1,)

    # average those frame-wise nets into a single scalar
    mean_vx = float(np.mean(per_frame_net))
    mean_abs_vx = float(np.mean(per_frame_abs))

    print(
        f"  [v_x]  {npz_path.name:50s}  "
        f"shape={coords.shape}  window=[{f0}:{f1}]  "
        f"dt/frame={dt_per_frame:.6e}  "
        f"⟨v_x⟩={mean_vx:.6e}  ⟨|v_x|⟩={mean_abs_vx:.6e}"
    )
    return mean_vx, mean_abs_vx


def _gather_case_data(
    case_id: str,
    hexatic_dir: Path,
    npz_dir: Path,
    v0: int = DEFAULT_V0,
    v1: int = DEFAULT_V1,
) -> CaseData | None:
    """Collect metrics for a single case.

    Returns ``None`` when required files are missing or empty.
    """
    hexatic_path = hexatic_dir / f"{case_id}_hexatic_order.txt"
    neighbor_path = hexatic_dir / f"{case_id}_neighbor_counts.txt"
    npz_path = npz_dir / f"{case_id}_active_matter_fields.npz"

    if not hexatic_path.exists():
        print(f"  [skip] missing {hexatic_path.name}")
        return None
    if not neighbor_path.exists():
        print(f"  [skip] missing {neighbor_path.name}")
        return None

    mean_psi6, n_hex = _mean_hexatic_first_frame(hexatic_path)
    plus1, minus1, n_nbr = _disclination_density_first_frame(neighbor_path)
    mean_vx, mean_abs_vx = _velocity_from_npz(npz_path, v0, v1)
    mean_px, mean_abs_px = _polarization_from_npz(npz_path, v0, v1)

    n_particles = max(n_hex, n_nbr)

    return CaseData(
        case_id=case_id,
        c_over_d=parse_c_over_d(case_id),
        n_particles=n_particles,
        disclination_plus1=plus1,
        disclination_minus1=minus1,
        mean_psi6=mean_psi6,
        mean_vx=mean_vx,
        mean_abs_vx=mean_abs_vx,
        mean_px=mean_px,
        mean_abs_px=mean_abs_px,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _add_scatter(
    fig: go.Figure,
    row: int,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    name: str,
    color: str,
    marker_symbol: str = "circle",
) -> None:
    """Add a marker trace with case-label hover text."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            name=name,
            text=labels,
            textposition="top center",
            textfont=dict(size=11, color="#374151"),
            marker=dict(
                size=12,
                color=color,
                symbol=marker_symbol,
                line=dict(width=1.2, color="white"),
            ),
            hovertemplate=(
                "C/D=%{x:.3f}<br>"
                "%{fullData.name}=%{y:.5f}<br>"
                "%{text}<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )


def build_figure(rows_data: list[CaseData], v0: int = DEFAULT_V0, v1: int = DEFAULT_V1) -> go.Figure:
    """Build a five-panel Plotly figure from collected case data."""
    rows_data.sort(key=lambda d: d.c_over_d)

    cd = np.array([d.c_over_d for d in rows_data])
    labels = [d.case_id for d in rows_data]

    plus1 = np.array([d.disclination_plus1 for d in rows_data])
    minus1 = np.array([d.disclination_minus1 for d in rows_data])
    psi6 = np.array([d.mean_psi6 for d in rows_data])
    vx = np.array([d.mean_vx for d in rows_data])
    abs_vx = np.array([d.mean_abs_vx for d in rows_data])
    px = np.array([d.mean_px for d in rows_data])
    abs_px = np.array([d.mean_abs_px for d in rows_data])

    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Disclination density +1  (first frame)",
            "Disclination density −1  (first frame)",
            "Mean |ψ₆|  (first frame)",
            f"⟨v_x⟩  avg net x-velocity over frames {v0}–{v1}",
            f"⟨|v_x|⟩  avg |x-velocity| over frames {v0}–{v1}",
            f"⟨P_x⟩  avg net P_x over frames {v0}–{v1}",
            f"⟨|P_x|⟩  avg |P_x| over frames {v0}–{v1}",
        ),
        vertical_spacing=0.06,
    )

    _add_scatter(fig, 1, cd, plus1, labels, "+1 disclinations", "#dc2626", "triangle-up")
    _add_scatter(fig, 2, cd, minus1, labels, "−1 disclinations", "#2563eb", "triangle-down")
    _add_scatter(fig, 3, cd, psi6, labels, "mean |ψ₆|", "#7c3aed", "circle")
    _add_scatter(fig, 4, cd, vx, labels, "⟨v_x⟩", "#0891b2", "diamond")
    _add_scatter(fig, 5, cd, abs_vx, labels, "⟨|v_x|⟩", "#ea580c", "square")
    _add_scatter(fig, 6, cd, px, labels, "⟨P_x⟩", "#16a34a", "star")
    _add_scatter(fig, 7, cd, abs_px, labels, "⟨|P_x|⟩", "#d97706", "cross")

    # Axes formatting
    fig.update_xaxes(
        title_text="C / D  (circumference / particle diameter)",
        row=7,
        col=1,
        showgrid=True,
        gridcolor="rgba(39,49,61,0.10)",
        zeroline=False,
    )
    for row in range(1, 8):
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(39,49,61,0.10)",
            zeroline=False,
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text="fraction of particles", row=1, col=1)
    fig.update_yaxes(title_text="fraction of particles", row=2, col=1)
    fig.update_yaxes(title_text="⟨ |ψ₆| ⟩", row=3, col=1)
    fig.update_yaxes(title_text="v_x  (σ / τ)", row=4, col=1)
    fig.update_yaxes(title_text="|v_x|  (σ / τ)", row=5, col=1)
    fig.update_yaxes(title_text="P_x", row=6, col=1)
    fig.update_yaxes(title_text="|P_x|", row=7, col=1)

    # Zero reference lines for velocity and polarization panels
    fig.add_hline(y=0.0, row=4, col=1, line=dict(color="#9aa3ad", width=1, dash="dot"))
    fig.add_hline(y=0.0, row=6, col=1, line=dict(color="#9aa3ad", width=1, dash="dot"))

    fig.update_layout(
        title=dict(
            text="Properties vs C/D",
            x=0.5,
            xanchor="center",
            font=dict(size=22),
        ),
        template="plotly_white",
        width=960,
        height=1700,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=85, r=35, t=100, b=75),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot initial-condition properties vs C/D across all discovered cases."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output HTML (default: output/initial_vs_c_over_d.html)",
    )
    parser.add_argument(
        "-v0", "--v-frame-start",
        type=int,
        default=DEFAULT_V0,
        help=f"First frame index for velocity window (default: {DEFAULT_V0})",
    )
    parser.add_argument(
        "-v1", "--v-frame-end",
        type=int,
        default=DEFAULT_V1,
        help=f"One-past-end frame index for velocity window (default: {DEFAULT_V1})",
    )
    args = parser.parse_args()

    case_ids = _discover_case_ids(HEXATIC_OUTPUT_DIR)
    if not case_ids:
        print(f"No cases found in {HEXATIC_OUTPUT_DIR}")
        return

    print(f"Discovered {len(case_ids)} cases: {', '.join(case_ids)}")

    rows: list[CaseData] = []
    for cid in case_ids:
        data = _gather_case_data(cid, HEXATIC_OUTPUT_DIR, NPZ_FIELDS_DIR, args.v_frame_start, args.v_frame_end)
        if data is not None:
            rows.append(data)

    if not rows:
        print("No valid case data collected.")
        return

    print(f"Plotting {len(rows)} cases.")
    fig = build_figure(rows, args.v_frame_start, args.v_frame_end)

    output = args.output or OUTPUT_DIR / "initial_vs_c_over_d.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
