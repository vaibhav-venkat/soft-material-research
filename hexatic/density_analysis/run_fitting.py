from __future__ import annotations

import argparse

from .fitting.config import DEFAULT_CASE_ID, FittingConfig
from .fitting.fit import FittingResult, compute_fitting
from .fitting.io_cache import load_cache, write_cache
from .fitting.plots import write_all_plots
from .fitting.types import DEFAULT_CANDIDATES

CANDIDATES = DEFAULT_CANDIDATES
DROP_FIT_CANDIDATE: tuple[str, ...] = (
)


def fit_candidates() -> tuple[str, ...]:
    dropped = set(DROP_FIT_CANDIDATE)
    unknown = dropped.difference(CANDIDATES)
    if unknown:
        raise ValueError(
            "DROP_FIT_CANDIDATE contains unknown candidates: "
            + ", ".join(sorted(unknown))
        )
    candidates = tuple(name for name in CANDIDATES if name not in dropped)
    if not candidates:
        raise ValueError("DROP_FIT_CANDIDATE removed every fitting candidate.")
    return candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit film fluxes to FFT density-gradient fields."
    )
    parser.add_argument("--case", default=DEFAULT_CASE_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--gsd-path", default=None)
    parser.add_argument(
        "--bins",
        type=float,
        default=0.0,
        help="Spatial Gaussian smoothing sigma in grid bins for the actual fitting arrays.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_candidates = fit_candidates()
    config = FittingConfig(
        case_id=args.case,
        npz_path=args.npz_path,
        gsd_path=args.gsd_path,
        candidate_names=selected_candidates,
        smoothing_bins=args.bins,
    )

    print(f"[fitting] Case: {config.case_id}")
    print(f"[fitting] Output directory: {config.output_dir}")
    if config.cache_path.exists() and not args.no_cache and not args.overwrite:
        print(f"[fitting] Loading fitting cache from {config.cache_path}...")
        try:
            result = FittingResult.from_cache_arrays(load_cache(config.cache_path))
            if result.candidate_names != selected_candidates:
                raise ValueError(
                    "cached candidate set "
                    f"{result.candidate_names} does not match requested "
                    f"{selected_candidates}"
                )
        except ValueError as exc:
            print(f"[fitting] Cache is stale: {exc}")
            print("[fitting] Recomputing and replacing stale cache...")
            result = compute_fitting(config)
            write_cache(
                config.cache_path,
                overwrite=True,
                **result.as_cache_arrays(),
            )
            print("[fitting] Cache write complete.")
    else:
        if args.no_cache:
            print("[fitting] Cache disabled; computing result.")
        elif args.overwrite:
            print("[fitting] Overwrite requested; recomputing result.")
        else:
            print(f"[fitting] No fitting cache found at {config.cache_path}; computing result.")
        result = compute_fitting(config)
        if not args.no_cache:
            print(f"[fitting] Writing fitting cache to {config.cache_path}...")
            write_cache(
                config.cache_path,
                overwrite=args.overwrite,
                **result.as_cache_arrays(),
            )
            print("[fitting] Cache write complete.")
    if args.plot:
        print("[fitting] Plotting requested.")
        write_all_plots(
            result,
            config.output_dir,
            frame_idx=args.frame_idx,
            case_id=args.case,
        )
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
