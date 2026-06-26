from __future__ import annotations

import argparse

from . import disclination_order_fields, velocity
from .common import FRAME_START, FRAME_STOP, ensure_output_dirs, selected_cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--include-long-axis", action="store_true")
    parser.add_argument("--frame-start", type=int, default=FRAME_START)
    parser.add_argument("--frame-stop", type=int, default=FRAME_STOP)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute aggregate NPZ/plot outputs even when cached files exist.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=(
            "disclination_order_fields",
        ),
        help="Metric to skip. Can be passed multiple times.",
    )
    return parser.parse_args()


def run_all(args: argparse.Namespace) -> None:
    ensure_output_dirs()
    cases = selected_cases(
        args.case,
        include_all=args.all,
        include_long_axis=args.include_long_axis,
    )
    metrics = (
        ("disclination_velocity", velocity.run_disclination_velocity),
        ("disclination_order_fields", disclination_order_fields.run),
    )
    skipped = set(args.skip)
    for metric_name, runner in metrics:
        if metric_name in skipped:
            print(f"skipped {metric_name}")
            continue
        print(f"running {metric_name}")
        runner(
            cases,
            frame_start=args.frame_start,
            frame_stop=args.frame_stop,
            overwrite=args.overwrite,
        )
        print(f"finished {metric_name}")


def main() -> None:
    run_all(_parse_args())


if __name__ == "__main__":
    main()
