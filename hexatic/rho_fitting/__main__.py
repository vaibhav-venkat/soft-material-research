"""CLI for the rho fitting workflow."""

from __future__ import annotations

import argparse

from .config import NumericalSettings, RhoFittingConfig
from .fit import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit rho mechanical dynamics.")
    parser.add_argument("--case", default="radius_15D")
    parser.add_argument("--nd", type=int, default=None)
    parser.add_argument("--mechanical-flux-weight", type=float, default=None)
    parser.add_argument(
        "--rho-divergence-weight",
        dest="mechanical_flux_weight",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--correlations-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = None
    if args.nd is not None or args.mechanical_flux_weight is not None:
        kwargs = {}
        if args.nd is not None:
            kwargs["nd"] = args.nd
        if args.mechanical_flux_weight is not None:
            kwargs["mechanical_flux_weight"] = args.mechanical_flux_weight
        settings = NumericalSettings(**kwargs)
    config = RhoFittingConfig(
        case_id=args.case,
        overwrite=args.overwrite,
        make_plots=not (args.no_plots or args.no_plot),
        correlations_only=args.correlations_only,
        fit_only=args.fit_only,
        settings=settings,
    )
    result = run(config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
