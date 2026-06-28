from __future__ import annotations

import argparse

from .fitting.config import DEFAULT_CASE_ID, FittingConfig
from .fitting.fit import FittingResult, compute_fitting
from .fitting.io_cache import load_cache, write_cache
from .fitting.plots import write_all_plots
from .fitting.types import FIELD_REGISTRY, ROLE_CANDIDATE

DROP_FIT_CANDIDATE: tuple[str, ...] = (
    "D_P",
    "D_chiral_P_perp",
    "D_force_density"
)

DROP_G: tuple[str, ...] = (
    "G_4",
    "G_2",
    "G_3"
)

ALL_G_NAMES = ("G_0", "G_1", "G_2", "G_3", "G_4")


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
        help="Spatial Gaussian smoothing sigma in grid bins for the actual fitting arrays.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    all_candidates = FIELD_REGISTRY.names_for_role(ROLE_CANDIDATE)
    dropped = set(DROP_FIT_CANDIDATE)
    unknown = dropped.difference(all_candidates)
    if unknown:
        raise ValueError(
            "DROP_FIT_CANDIDATE contains unknown candidates: "
            + ", ".join(sorted(unknown))
        )
    selected = tuple(c for c in all_candidates if c not in DROP_FIT_CANDIDATE)
    if not selected:
        raise ValueError("DROP_FIT_CANDIDATE removed every fitting candidate.")

    unknown_g = set(DROP_G).difference(ALL_G_NAMES)
    if unknown_g:
        raise ValueError(
            "DROP_G contains unknown G modifier names: "
            + ", ".join(sorted(unknown_g))
        )
    dropped_g_indices = frozenset(
        int(name.split("_")[1]) for name in DROP_G
    )
    g_modifier_indices = tuple(
        i for i in range(5) if i not in dropped_g_indices
    )
    if not g_modifier_indices:
        raise ValueError("DROP_G removed every G modifier.")

    config = FittingConfig(
        case_id=args.case,
        candidate_names=selected,
        g_modifier_indices=g_modifier_indices,
        smoothing_bins=args.bins,
    )
    print(f"[fitting] Case: {config.case_id}")

    if config.cache_path.exists() and not args.overwrite:
        print("[fitting] Loading cached fit result...")
        try:
            result = FittingResult.from_cache_arrays(load_cache(config.cache_path))
            if result.candidate_names != selected:
                raise ValueError(
                    "cached candidate set "
                    f"{result.candidate_names} does not match requested "
                    f"{selected}"
                )
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
