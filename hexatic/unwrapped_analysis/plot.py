from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CASE_ID = "circ_60_0D"
ANALYSIS_DIR = Path(__file__).resolve().parent
CASE_LABEL = "C = 60.0D"
DEFAULT_MAX_LAG = 30


def _mean_polarization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        steps = np.asarray(data["steps"], dtype=float)
        directions = np.asarray(data["direction_cylindrical"], dtype=float)
    return steps, np.nanmean(directions, axis=1)


def _mean_hexatic_order(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, comments="#")
    frame = raw[:, 0].astype(int)
    psi_abs = raw[:, 5]
    n_frames = int(frame.max()) + 1
    totals = np.bincount(frame, weights=psi_abs, minlength=n_frames)
    counts = np.bincount(frame, minlength=n_frames)
    return totals / counts


def _add_line(
    fig: go.Figure,
    row: int,
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    color: str,
    dash: str = "solid",
    showlegend: bool = True,
    secondary_y: bool = False,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            line=dict(color=color, width=2.4, dash=dash),
            hovertemplate="t=%{x:.3f}<br>%{fullData.name}=%{y:.5f}<extra></extra>",
        ),
        row=row,
        col=1,
        secondary_y=secondary_y,
    )


def _add_frame_lines(fig: go.Figure, t: np.ndarray) -> None:
    for row in range(1, 4):
        for idx, dash, color in ((17, "dot", "#555"), (18, "dash", "#111")):
            if len(t) > idx:
                fig.add_vline(
                    x=t[idx],
                    row=row,
                    col=1,
                    line=dict(color=color, width=1.5, dash=dash),
                )


def _padded_range(values: np.ndarray) -> list[float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = 0.08 * max(high - low, 1.0e-6)
    return [low - pad, high + pad]


def plot_case(output: Path) -> Path:
    steps, mean_p = _mean_polarization(
        ANALYSIS_DIR / "npz_fields" / f"{CASE_ID}_active_matter_fields.npz"
    )
    mean_psi6 = _mean_hexatic_order(
        ANALYSIS_DIR / "hexatic_output" / f"{CASE_ID}_hexatic_order.txt"
    )

    n = min(len(steps), len(mean_psi6), len(mean_p))
    t = (steps[:n] - steps[0]) / 1_000_000.0
    mean_p = mean_p[:n]
    mean_psi6 = mean_psi6[:n]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
        subplot_titles=(
            "P_x and mean hexatic order",
            "P_theta and mean hexatic order",
            "P_R and mean hexatic order",
        ),
        vertical_spacing=0.08,
    )

    components = (
        ("P_x", mean_p[:, 0], "#2ca02c"),
        ("P_theta", mean_p[:, 2], "#9467bd"),
        ("P_R", mean_p[:, 1], "#ff7f0e"),
    )
    for row, (name, values, color) in enumerate(components, start=1):
        _add_line(fig, row, t, values, name, color)
        _add_line(
            fig,
            row,
            t,
            mean_psi6,
            "mean |psi6|",
            "#111827",
            dash="dash",
            showlegend=row == 1,
            secondary_y=True,
        )
        fig.add_hline(y=0.0, row=row, col=1, line=dict(color="#9aa3ad", width=1))
        fig.update_yaxes(
            title_text="mean polarization",
            range=_padded_range(values),
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="mean |psi6|",
            row=row,
            col=1,
            secondary_y=True,
        )

    _add_frame_lines(fig, t)
    fig.update_xaxes(title_text="time since first frame (10^6 steps)", row=3, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(39,49,61,0.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(39,49,61,0.10)", zeroline=False)
    fig.update_layout(
        title=dict(
            text=f"Unwrapped {CASE_LABEL}: polarization and hexatic order",
            x=0.5,
            xanchor="center",
            font=dict(size=24),
        ),
        template="plotly_white",
        width=1120,
        height=960,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.75)",
        ),
        margin=dict(l=85, r=35, t=130, b=70),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def _lag_pair(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        return a[-lag:], b[:lag]
    if lag > 0:
        return a[:-lag], b[lag:]
    return a, b


def _pearson_lag_curve(
    a: np.ndarray,
    b: np.ndarray,
    lags: np.ndarray,
) -> np.ndarray:
    values = np.empty(len(lags), dtype=float)
    for idx, lag in enumerate(lags):
        x, y = _lag_pair(a, b, int(lag))
        values[idx] = float(np.corrcoef(x, y)[0, 1])
    return values


def plot_lag_correlations(output: Path, max_lag: int = DEFAULT_MAX_LAG) -> Path:
    steps, mean_p = _mean_polarization(
        ANALYSIS_DIR / "npz_fields" / f"{CASE_ID}_active_matter_fields.npz"
    )
    mean_psi6 = _mean_hexatic_order(
        ANALYSIS_DIR / "hexatic_output" / f"{CASE_ID}_hexatic_order.txt"
    )

    n = min(len(steps), len(mean_psi6), len(mean_p))
    max_lag = min(max_lag, n - 2)
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    lag_time = lags * float(np.median(np.diff(steps[:n]))) / 1_000_000.0
    sample_counts = n - np.abs(lags)

    px = mean_p[:n, 0]
    pr = mean_p[:n, 1]
    mean_psi6 = mean_psi6[:n]

    panels = (
        (
            "Pearson correlation: P_x(t0) with mean |psi6|(t0 + lag)",
            "P_x-H Pearson r",
            _pearson_lag_curve(px, mean_psi6, lags),
            "#16a34a",
        ),
        (
            "Pearson correlation: P_R(t0) with mean |psi6|(t0 + lag)",
            "P_R-H Pearson r",
            _pearson_lag_curve(pr, mean_psi6, lags),
            "#ea580c",
        ),
    )
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=tuple(title for title, _, _, _ in panels),
        vertical_spacing=0.12,
    )

    customdata = np.column_stack((lags, sample_counts))
    for row, (_, name, values, color) in enumerate(panels, start=1):
        fig.add_trace(
            go.Scatter(
                x=lag_time,
                y=values,
                customdata=customdata,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2.4),
                marker=dict(
                    size=6.5,
                    color=color,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=(
                    "lag=%{customdata[0]:.0f} frames<br>"
                    "paired samples=%{customdata[1]:.0f}<br>"
                    "lag time=%{x:.3f} x10^6 steps<br>"
                    "Pearson r=%{y:.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        fig.add_hline(
            y=0.0,
            row=row,
            col=1,
            line=dict(color="#64748b", width=1, dash="dash"),
        )
        fig.add_vline(
            x=0.0,
            row=row,
            col=1,
            line=dict(color="#111827", width=1.2),
        )
        fig.update_yaxes(
            title_text="Pearson r",
            range=[-1.05, 1.05],
            row=row,
            col=1,
        )

    fig.update_xaxes(title_text="lag t (10^6 steps)", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(39,49,61,0.10)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(39,49,61,0.10)", zeroline=False)
    fig.update_layout(
        title=dict(
            text=(
                f"Pearson lag correlations with mean |psi6|, {CASE_LABEL}, "
                f"lag -{max_lag}..{max_lag} frames"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=21),
        ),
        template="plotly_white",
        width=1080,
        height=760,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=80, r=40, t=125, b=70),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        choices=("timeseries", "lag"),
        default="timeseries",
    )
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.plot == "lag":
        output = args.output or ANALYSIS_DIR / "output" / f"{CASE_ID}_px_pr_hexatic_lag_correlation.html"
        print(plot_lag_correlations(output, max_lag=args.max_lag))
        return
    output = args.output or ANALYSIS_DIR / "output" / f"{CASE_ID}_polarization_hexatic.html"
    print(plot_case(output))


if __name__ == "__main__":
    main()
