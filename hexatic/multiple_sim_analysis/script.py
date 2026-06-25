from __future__ import annotations

import argparse

from . import (
    chirality,
    density_profile,
    disclination,
    dislocation,
    force_density,
    polarization,
    radial_exchange_current,
    velocity,
    x_com,
)
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
            "velocity",
            "chirality",
            "density_profile",
            "disclination",
            "dislocation",
            "polarization",
            "x_com",
            "force_density",
            "radial_exchange_current",
        ),
        help="Metric to skip. Can be passed multiple times.",
    )
    parser.add_argument(
        "--radial-exchange-current-bin-width",
        type=float,
        default=radial_exchange_current.ACTIVE_RADIAL_BIN_WIDTH,
        help="Radial bin width for radial exchange current; defaults to 1 particle diameter.",
    )
    parser.add_argument(
        "--radial-exchange-current-kernel-radius",
        type=float,
        default=radial_exchange_current.LOCAL_POCKET_RADIUS,
        help="Gaussian kernel radius for radial exchange current; defaults to 2 particle diameters.",
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
        ("velocity", velocity.run),
        ("density_profile", density_profile.run),
        ("disclination", disclination.run),
        ("dislocation", dislocation.run),
        ("polarization", polarization.run),
        ("x_com", x_com.run),
        (
            "radial_exchange_current",
            lambda cases, frame_start, frame_stop, overwrite: radial_exchange_current.run(
                cases,
                frame_start=frame_start,
                frame_stop=frame_stop,
                overwrite=overwrite,
                radial_bin_width=args.radial_exchange_current_bin_width,
                kernel_radius=args.radial_exchange_current_kernel_radius,
            ),
        ),
        ("force_density", force_density.run),
        ("chirality", chirality.run),
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
