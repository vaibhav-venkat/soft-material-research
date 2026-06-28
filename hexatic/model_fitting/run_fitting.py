from __future__ import annotations

import argparse

from .fitting.config import DEFAULT_CASE_ID, FittingConfig
from .fitting.fields import load_or_compute_fields
from .fitting.fit import FittingResult, compute_fitting
from .fitting.io_cache import load_cache, write_cache
from .fitting.plots import write_all_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit film fluxes to FFT density-gradient fields."
    )
    parser.add_argument("--case", default=DEFAULT_CASE_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--bins",
        type=float,
        default=0.0,
        help="Spatial Gaussian smoothing sigma in grid bins for fitting arrays.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FittingConfig(
        case_id=args.case,
        smoothing_bins=args.bins,
    )
    print(f"[fitting] Case: {config.case_id}")

    if config.cache_path.exists() and not args.overwrite:
        print("[fitting] Loading cached fit result...")
        try:
            fields = load_or_compute_fields(config)
            result = FittingResult.from_cache_arrays(load_cache(config.cache_path), fields)
            print("[fitting] Cache loaded.")
        except ValueError:
            print("[fitting] Cache stale; recomputing...")
            result = compute_fitting(config)
            write_cache(config.cache_path, overwrite=True, **result.as_cache_arrays())
            print("[fitting] Cache written.")
    else:
        print("[fitting] Computing fit result...")
        result = compute_fitting(config)
        write_cache(config.cache_path, overwrite=args.overwrite, **result.as_cache_arrays())
        print("[fitting] Cache written.")

    if not args.no_plot:
        write_all_plots(result, config.output_dir, case_id=args.case)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
