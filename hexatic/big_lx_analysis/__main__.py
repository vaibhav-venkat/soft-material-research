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
    cases = [get_case(case_id) for case_id in args.case]
    series = [
        analyze_correlations(
            case,
            output_root=args.output_root,
            min_origins=args.min_origins,
            max_lag=args.max_lag,
            absolute=args.absolute,
        )
        for case in cases
    ]
    output = args.output or args.output_root / "plots" / "big_lx_correlations.png"
    result = plot_correlations(
        series,
        output,
        dpi=args.dpi,
        absolute=args.absolute,
    )
    print(
        f"[big_lx_analysis] cases={len(cases)} absolute={args.absolute} "
        f"output={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
