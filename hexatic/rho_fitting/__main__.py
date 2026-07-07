"""CLI for the rho fitting workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, NumericalSettings, RhoFittingConfig
from .fit import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit rho mechanical dynamics.")
    parser.add_argument("--case", default="radius_15D")
    parser.add_argument("--nd", type=int, default=None)
    parser.add_argument("--radial-bins", type=int, default=None)
    parser.add_argument("--radial-range", type=float, nargs=2, default=None, metavar=("R_MIN", "R_MAX"))
    parser.add_argument("--mechanical-flux-weight", type=float, default=None)
    parser.add_argument(
        "--rho-divergence-weight",
        dest="mechanical_flux_weight",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--correlations-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = None
    if (
        args.nd is not None
        or args.mechanical_flux_weight is not None
        or args.radial_bins is not None
        or args.radial_range is not None
    ):
        base = NumericalSettings()
        kwargs = {}
        if args.nd is not None:
            kwargs["nd"] = args.nd
        if args.mechanical_flux_weight is not None:
            kwargs["mechanical_flux_weight"] = args.mechanical_flux_weight
        if args.radial_bins is not None:
            kwargs["radial_bins"] = args.radial_bins
        if args.radial_range is not None:
            kwargs["radial_range"] = tuple(args.radial_range)
        settings = replace(base, **kwargs)
    config = RhoFittingConfig(
        case_id=args.case,
        overwrite=args.overwrite,
        make_plots=not (args.no_plots or args.no_plot),
        correlations_only=args.correlations_only,
        fit_only=args.fit_only,
        output_dir=args.output_dir,
        settings=settings,
    )
    result = run(config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
