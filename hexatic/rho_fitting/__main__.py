"""CLI for the rho fitting workflow."""

from __future__ import annotations

import argparse

from .config import NumericalSettings, RhoFittingConfig
from .fit import run, write_density_report_from_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit density dynamics.")
    parser.add_argument("--case", default="radius_15D")
    parser.add_argument("--nd", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate the markdown report from the existing fit cache",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--cheb-cutoff", type=int, default=10)
    parser.add_argument("--timestep", type=float)
    parser.add_argument("--n-rho-power", type=int, default=5)
    parser.add_argument("--n-rho-lap-power", type=int, default=4)
    parser.add_argument("--subsamples", type=int, default=200)
    parser.add_argument("--tau-count", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings_kwargs = {
        "cheb_cutoff": args.cheb_cutoff,
        "n_rho_power": args.n_rho_power,
        "n_rho_lap_power": args.n_rho_lap_power,
        "nd": args.nd,
        "seed": args.seed,
        "subsamples": args.subsamples,
        "tau_count": args.tau_count,
    }
    if args.sigma is not None:
        settings_kwargs["sigma"] = args.sigma
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
    if args.report_only:
        result = write_density_report_from_cache(config)
    else:
        result = run(config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
