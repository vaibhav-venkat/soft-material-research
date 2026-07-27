"""Plot orientation correlations for ideal 3D bulk diffusion-trigger cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from hexatic.new_sims.cases import CaseKind

from .plot_2d_correlation import (
    _load_correlation,
    _plot,
    _write_report,
)


CASE_LABELS = {
    CaseKind.IDEAL_3D_BULK_PERIOD_10.value: "Diffusion trigger period 10",
    CaseKind.IDEAL_3D_BULK_PERIOD_1.value: "Diffusion trigger period 1",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare orientation autocorrelations for non-interacting, fully "
            "periodic 3D ABPs with diffusion trigger periods 10 and 1."
        )
    )
    parser.add_argument(
        "--period-10-shard-dir",
        type=Path,
        default=None,
        help="Analysis directory for ideal_3d_bulk_period_10.",
    )
    parser.add_argument(
        "--period-1-shard-dir",
        type=Path,
        default=None,
        help="Analysis directory for ideal_3d_bulk_period_1.",
    )
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--max-lag", type=int, default=900)
    parser.add_argument("--fit-min-correlation", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("3d_orientation_correlation.svg"),
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
    if args.period_10_shard_dir is None and args.period_1_shard_dir is None:
        parser.error(
            "provide --period-10-shard-dir, --period-1-shard-dir, or both"
        )

    directories: dict[str, Path] = {}
    if args.period_10_shard_dir is not None:
        directories[CaseKind.IDEAL_3D_BULK_PERIOD_10.value] = (
            args.period_10_shard_dir
        )
    if args.period_1_shard_dir is not None:
        directories[CaseKind.IDEAL_3D_BULK_PERIOD_1.value] = (
            args.period_1_shard_dir
        )
    results = {
        case_id: _load_correlation(
            shard_dir,
            case_id,
            args.frames,
            args.max_lag,
            args.fit_min_correlation,
            ("x", "y", "z"),
        )
        for case_id, shard_dir in directories.items()
    }
    _plot(
        results,
        args.output,
        CASE_LABELS,
        "Ideal 3D bulk orientation autocorrelation",
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
        "Ideal 3D bulk orientation-correlation fits",
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
