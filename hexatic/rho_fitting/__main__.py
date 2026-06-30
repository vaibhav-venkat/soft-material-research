"""CLI for the rho fitting workflow."""

from __future__ import annotations

import argparse

from .config import NumericalSettings, RhoFittingConfig
from .fit import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit rho and polarization dynamics.")
    parser.add_argument("--case", default="radius_15D")
    parser.add_argument("--nd", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--cheb-cutoff", type=int, default=20)
    parser.add_argument("--timestep", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings_kwargs = {
        "sigma": args.sigma,
        "cheb_cutoff": args.cheb_cutoff,
        "nd": args.nd,
        "seed": args.seed,
    }
    if args.timestep is not None:
        settings_kwargs["timestep"] = args.timestep
    config = RhoFittingConfig(
        case_id=args.case,
        nd=args.nd,
        seed=args.seed,
        overwrite=args.overwrite,
        make_plots=not args.no_plot,
        max_frames=args.max_frames,
        settings=NumericalSettings(**settings_kwargs),
    )
    result = run(config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
