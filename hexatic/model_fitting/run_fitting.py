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
from .fitting.fit import FittingResult, compute_fitting, _coarse_grain_fields
from .fitting.io_cache import load_cache, write_cache
from .fitting.library import NO_FORCE_LOW_K_TERM_NAMES
from .fitting.movies import (
    MODEL_FITTING_MOVIE_FPS,
    MODEL_FITTING_MOVIE_SECONDS_PER_FRAME,
    write_model3_xtheta_movies,
)
from .fitting.plots import write_all_plots
from .fitting.write_report import write_model_report


DROP_NO_FORCE_LOW_K_TERMS: tuple[str, ...] = (
    "low_k_force_density",
    "low_k_grad_rho",
    "low_k_grad_hexatic_order",
    "low_k_grad_D",
    "low_k_D_P",
    "low_k_P_r_P",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit hydrodynamic density/polarization model."
    )
    parser.add_argument("--case", default=DEFAULT_CASE_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--movies",
        action="store_true",
        help="Write Model 3 x-theta comparison movies.",
    )
    parser.add_argument(
        "--movie-fps",
        type=int,
        default=MODEL_FITTING_MOVIE_FPS,
        help="Frames per second for Model 3 x-theta movies.",
    )
    parser.add_argument(
        "--movie-seconds-per-frame",
        type=float,
        default=MODEL_FITTING_MOVIE_SECONDS_PER_FRAME,
        help="Playback seconds to hold each coarse-grained Model 3 movie frame.",
    )
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
    _validate_drop_no_force_low_k_terms()
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
            fields = _coarse_grain_fields(
                load_or_compute_fields(config),
                config.coarse_grain_transitions,
            )
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
    write_model_report(
        result,
        config.output_dir,
        case_id=args.case,
        drop_no_force_low_k_terms=DROP_NO_FORCE_LOW_K_TERMS,
    )

    # --- plots ---
    if not args.no_plot:
        write_all_plots(result, config.output_dir, case_id=args.case)
        if args.movies:
            write_model3_xtheta_movies(
                result,
                config.output_dir,
                case_id=args.case,
                drop_no_force_low_k_terms=DROP_NO_FORCE_LOW_K_TERMS,
                fps=args.movie_fps,
                seconds_per_frame=args.movie_seconds_per_frame,
            )

    # --- terminal summary ---
    print(result.summary())
    return 0


def _validate_drop_no_force_low_k_terms() -> None:
    unknown = set(DROP_NO_FORCE_LOW_K_TERMS).difference(NO_FORCE_LOW_K_TERM_NAMES)
    if unknown:
        raise ValueError(
            "DROP_NO_FORCE_LOW_K_TERMS contains unknown low-k terms: "
            + ", ".join(sorted(unknown))
        )


if __name__ == "__main__":
    raise SystemExit(main())
