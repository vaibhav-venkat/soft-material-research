"""CLI for the hydrodynamic model fitting workflow."""

from __future__ import annotations

import argparse

from .fitting.config import (
    DEFAULT_CASE_ID,
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_COARSE_GRAIN_TRANSITIONS,
    DEFAULT_STLSQ_MAX_ITER,
    DEFAULT_STLSQ_THRESHOLD,
    FittingConfig,
)
from .fitting.fields import load_or_compute_fields
from .fitting.fit import FittingResult, compute_fitting
from .fitting.io_cache import load_cache, write_cache
from .fitting.plots import write_all_plots
from .fitting.write_report import write_model_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit hydrodynamic density/polarization model."
    )
    parser.add_argument("--case", default=DEFAULT_CASE_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--stlsq-threshold",
        type=float,
        default=DEFAULT_STLSQ_THRESHOLD,
        help="STLSQ sparsification threshold.",
    )
    parser.add_argument(
        "--stlsq-max-iter",
        type=int,
        default=DEFAULT_STLSQ_MAX_ITER,
        help="Maximum STLSQ iterations.",
    )
    parser.add_argument(
        "--coarse-grain-transitions",
        type=int,
        default=DEFAULT_COARSE_GRAIN_TRANSITIONS,
        help="Number of adjacent transitions to average for current-closure fitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FittingConfig(
        case_id=args.case,
        ridge_alpha=args.ridge_alpha,
        stlsq_threshold=args.stlsq_threshold,
        stlsq_max_iter=args.stlsq_max_iter,
        coarse_grain_transitions=args.coarse_grain_transitions,
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

    # --- text reports ---
    write_model_report(result, config.output_dir, case_id=args.case)

    # --- plots ---
    if not args.no_plot:
        write_all_plots(result, config.output_dir, case_id=args.case)

    # --- terminal summary ---
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
