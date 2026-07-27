"""Compare orientation autocorrelations for the two ideal 2D simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from hexatic.constants import cylinder
from hexatic.new_sims.cases import CaseKind

from .plot_ideal_orientation_autocorrelation import (
    ExponentialFit,
    _fit_exponential,
    _lag_times,
    _load_polarization,
    _orientation_autocorrelations,
)


CASE_LABELS = {
    CaseKind.IDEAL_2D_X_WALLS.value: r"Walls at $x=\pm L_x/2$",
    CaseKind.IDEAL_2D_PERIODIC.value: r"Periodic $x,y$",
}


def _load_manifest(
    shard_dir: Path,
    expected_case_id: str,
    expected_coordinate_order: tuple[str, ...],
) -> dict[str, Any]:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.new_sims.analysis.v1":
        raise ValueError(f"unsupported manifest schema in {manifest_path}")
    if manifest.get("complete") is not True:
        raise ValueError(f"analysis manifest is incomplete: {manifest_path}")
    case = manifest.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"missing case metadata in {manifest_path}")
    if case.get("case_id") != expected_case_id:
        raise ValueError(
            f"expected case {expected_case_id!r} in {manifest_path}, "
            f"got {case.get('case_id')!r}"
        )
    if manifest.get("coordinate_order") != list(expected_coordinate_order):
        raise ValueError(
            f"expected coordinate order {list(expected_coordinate_order)!r} "
            f"in {manifest_path}"
        )
    return manifest


def _load_correlation(
    shard_dir: Path,
    case_id: str,
    frame_limit: int,
    max_lag: int,
    minimum_correlation: float,
    coordinate_order: tuple[str, ...],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    ExponentialFit,
    ExponentialFit,
    int,
    int,
]:
    manifest = _load_manifest(shard_dir, case_id, coordinate_order)
    q, steps = _load_polarization(shard_dir, manifest, frame_limit)
    q_corr, qx_corr = _orientation_autocorrelations(q, max_lag)
    case = manifest["case"]
    timestep = float(case.get("timestep", cylinder.SIMULATION.timestep))
    tau_r = float(cylinder.SIMULATION.tau_r)
    if timestep <= 0.0 or tau_r <= 0.0:
        raise ValueError("timestep and tau_r must be positive")
    lag_time = _lag_times(steps, max_lag, timestep)
    return (
        lag_time,
        q_corr,
        qx_corr,
        _fit_exponential(lag_time, q_corr, tau_r, minimum_correlation),
        _fit_exponential(lag_time, qx_corr, tau_r, minimum_correlation),
        len(q),
        q.shape[1],
    )


def _plot(
    results: dict[
        str,
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            ExponentialFit,
            ExponentialFit,
            int,
            int,
        ],
    ],
    output: Path,
    case_labels: dict[str, str],
    title: str,
) -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    colors = sns.color_palette("colorblind", n_colors=len(results))
    figure, axes = plt.subplots(
        2, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    final_time = 0.0
    for case_index, (color, (case_id, result)) in enumerate(
        zip(colors, results.items(), strict=True)
    ):
        lag_time, q_corr, qx_corr, q_fit, qx_fit, _, _ = result
        label = case_labels[case_id]
        final_time = max(final_time, float(lag_time[-1]))
        for axis, correlation, fit in zip(
            axes, (q_corr, qx_corr), (q_fit, qx_fit), strict=True
        ):
            axis.plot(lag_time, correlation, color=color, linewidth=2.0, label=label)
            axis.plot(
                lag_time,
                fit.amplitude
                * np.exp(-fit.rate * lag_time / cylinder.SIMULATION.tau_r),
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"{label} fit",
            )
            axis.text(
                0.02 if case_index == 0 else 0.98,
                0.05,
                f"{label}\n"
                f"A = {fit.amplitude:.6g}\n"
                f"b = {fit.rate:.6g}\n"
                f"$R^2$ = {fit.r_squared:.6g}",
                color=color,
                transform=axis.transAxes,
                horizontalalignment="left" if case_index == 0 else "right",
                verticalalignment="bottom",
            )

    axes[0].set_title(title)
    axes[0].set_ylabel(r"$C_q(\Delta t)$")
    axes[1].set_title(r"Axial-orientation autocorrelation")
    axes[1].set_ylabel(r"$C_{q_x}(\Delta t)$")
    axes[1].set_xlabel(r"Lag time $\Delta t$")
    for axis in axes:
        axis.legend(frameon=False, loc="upper right")
    for axis in axes:
        axis.axhline(0.0, color="0.65", linewidth=0.8, zorder=0)
        axis.grid(axis="y", color="0.9", linewidth=0.7)
        axis.set_xlim(0.0, final_time)
        sns.despine(ax=axis)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format=output.suffix.lstrip(".") or "svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.new_sims_analysis"},
    )
    plt.close(figure)


def _write_report(
    results: dict[
        str,
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            ExponentialFit,
            ExponentialFit,
            int,
            int,
        ],
    ],
    report: Path,
    minimum_correlation: float,
    title: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        "The normalized, time-origin-averaged correlations use",
        "",
        r"```text",
        r"C_q(Δt) = <q_i(t) · q_i(t + Δt)> / C_q(0)",
        r"C_qx(Δt) = <q_ix(t) q_ix(t + Δt)> / C_qx(0)",
        r"fit: C(Δt) = A exp[-b Δt / τ_R]",
        r"```",
        "",
        (
            "The log-linear fit uses the initial contiguous decay while "
            f"`C > {minimum_correlation:g}`."
        ),
        "",
        "| Case | Frames | Particles | Correlation | A | b | R² |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for case_id, result in results.items():
        _, _, _, q_fit, qx_fit, frames, particles = result
        for correlation, fit in (("q", q_fit), ("q_x", qx_fit)):
            lines.append(
                f"| `{case_id}` | {frames} | {particles} | {correlation} | "
                f"{fit.amplitude:.8g} | {fit.rate:.8g} | "
                f"{fit.r_squared:.8g} |"
            )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare q_i(t) dot q_i(t + lag) for ideal walled and periodic "
            "2D active particles."
        )
    )
    parser.add_argument(
        "--walled-shard-dir",
        type=Path,
        default=None,
        help="Analysis directory for ideal_2d_x_walls.",
    )
    parser.add_argument(
        "--periodic-shard-dir",
        type=Path,
        default=None,
        help="Analysis directory for ideal_2d_periodic.",
    )
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=900)
    parser.add_argument(
        "--fit-min-correlation",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("2d_orientation_correlation.svg"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Markdown fit report path. By default, writes "
            "<output-stem>_report.md beside the plot."
        ),
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.max_lag < 0:
        parser.error("--max-lag must be nonnegative")
    if not 0.0 < args.fit_min_correlation < 1.0:
        parser.error("--fit-min-correlation must be between 0 and 1")
    if args.walled_shard_dir is None and args.periodic_shard_dir is None:
        parser.error(
            "provide --walled-shard-dir, --periodic-shard-dir, or both"
        )

    directories: dict[str, Path] = {}
    if args.walled_shard_dir is not None:
        directories[CaseKind.IDEAL_2D_X_WALLS.value] = args.walled_shard_dir
    if args.periodic_shard_dir is not None:
        directories[CaseKind.IDEAL_2D_PERIODIC.value] = args.periodic_shard_dir
    results = {
        case_id: _load_correlation(
            shard_dir,
            case_id,
            args.frames,
            args.max_lag,
            args.fit_min_correlation,
            ("x", "y"),
        )
        for case_id, shard_dir in directories.items()
    }
    _plot(
        results,
        args.output,
        CASE_LABELS,
        "2D orientation autocorrelation",
    )
    report = (
        args.report
        if args.report is not None
        else args.output.with_name(f"{args.output.stem}_report.md")
    )
    _write_report(
        results,
        report,
        args.fit_min_correlation,
        "Ideal 2D orientation-correlation fits",
    )
    print(f"wrote {args.output}")
    print(f"wrote {report}")
    for case_id, result in results.items():
        _, _, _, q_fit, qx_fit, frames, particles = result
        print(
            f"{case_id}: frames={frames} particles={particles} "
            f"q_rate={q_fit.rate:.8g} q_R2={q_fit.r_squared:.8g} "
            f"qx_rate={qx_fit.rate:.8g} qx_R2={qx_fit.r_squared:.8g}"
        )


if __name__ == "__main__":
    main()
