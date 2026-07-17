from __future__ import annotations

import argparse
from pathlib import Path

from .analyze_laplace import (
    center_of_mass_series,
    plot_center_series,
    scan_trajectories,
)
from .cases import all_cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot full-length unwrapped COM x-position and COM x-velocity "
            "directly from passive-dense trajectory GSDs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Passive-dense production root containing gsd/trajectory_*.gsd.",
    )
    parser.add_argument(
        "--case",
        "--cases",
        dest="case",
        action="extend",
        nargs="+",
        choices=tuple(case.case_id for case in all_cases()),
        help="Optional inclusive case list; by default plot every discovered case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG. Defaults to "
            "<input-dir>/gsd_com_analysis/com_x_velocity.png."
        ),
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.case and len(set(args.case)) != len(args.case):
        raise SystemExit("Each --case value must be unique")
    if args.dpi < 1:
        raise SystemExit("--dpi must be positive")
    output = (
        args.output.resolve()
        if args.output is not None
        else args.input_dir.resolve() / "gsd_com_analysis" / "com_x_velocity.png"
    )
    if output.suffix.lower() != ".png":
        raise SystemExit("--output must use a .png suffix")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Pass --overwrite to replace {output}")

    trajectories = scan_trajectories(args.input_dir, args.case)
    series_by_case = [center_of_mass_series(item) for item in trajectories]
    result = plot_center_series(series_by_case, output, args.dpi)
    for series in series_by_case:
        print(
            f"[passive_dense.com.case] case={series.trajectory.case_id} "
            f"frames={series.frames.size} first_step={series.steps[0]} "
            f"last_step={series.steps[-1]} gsd={series.trajectory.path}",
            flush=True,
        )
    print(
        f"[passive_dense.com] cases={len(series_by_case)} output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()

