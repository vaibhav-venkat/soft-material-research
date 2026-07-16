from __future__ import annotations

import argparse
from pathlib import Path

from hexatic.big_lx.cases import DEFAULT_OUTPUT_ROOT, get_case

from .correlations import analyze_correlations, plot_correlations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot safetensor-only axial COM-velocity and hexatic-magnitude "
            "lag correlations for selected Big-Lx cases."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="Big-Lx case ID; repeat to compare multiple cases.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-origins", type=int, default=10)
    parser.add_argument("--max-lag", type=int)
    parser.add_argument("--particle-block-size", type=int, default=4096)
    parser.add_argument(
        "--psi6-mode",
        choices=("connected", "zoomed-magnitude"),
        default="connected",
        help=(
            "Plot mean-subtracted magnitude fluctuations on the shared axis, or "
            "raw magnitude correlations on a zoomed right-hand axis."
        ),
    )
    parser.add_argument(
        "--psi6-zoom-limits",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.9, 1.05),
        help="Right-axis limits used by --psi6-mode zoomed-magnitude.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Plot the absolute value of the normalized velocity correlation.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if len(set(args.case)) != len(args.case):
        raise SystemExit("Each --case value must be unique")
    if args.psi6_zoom_limits[0] >= args.psi6_zoom_limits[1]:
        raise SystemExit("--psi6-zoom-limits requires MIN < MAX")
    cases = [get_case(case_id) for case_id in args.case]
    psi6_correlation = (
        "connected-magnitude"
        if args.psi6_mode == "connected"
        else "magnitude"
    )
    series = [
        analyze_correlations(
            case,
            output_root=args.output_root,
            min_origins=args.min_origins,
            max_lag=args.max_lag,
            particle_block_size=args.particle_block_size,
            absolute=args.absolute,
            psi6_correlation=psi6_correlation,
        )
        for case in cases
    ]
    output = args.output or args.output_root / "plots" / "big_lx_correlations.png"
    result = plot_correlations(
        series,
        output,
        dpi=args.dpi,
        absolute=args.absolute,
        psi6_zoom_limits=tuple(args.psi6_zoom_limits),
    )
    print(
        f"[big_lx_analysis] cases={len(cases)} absolute={args.absolute} "
        f"psi6_mode={args.psi6_mode} "
        f"output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
