"""CLI for the rho fitting workflow."""

from __future__ import annotations

import argparse

from .config import RhoFittingConfig
from .fit import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit rho and polarization dynamics.")
    parser.add_argument("--case", default="radius_15D")
    parser.add_argument("--nd", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RhoFittingConfig(
        case_id=args.case,
        nd=args.nd,
        seed=args.seed,
        overwrite=args.overwrite,
        make_plots=not args.no_plot,
    )
    result = run(config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
