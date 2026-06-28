from __future__ import annotations

import argparse

from .film_continuity.config import DEFAULT_CASE_ID, FilmContinuityConfig
from .film_continuity.continuity import FilmContinuityResult, compute_film_continuity
from .film_continuity.io_cache import load_cache, write_cache
from .film_continuity.plots import write_all_maps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and plot film density/flux continuity fields."
    )
    parser.add_argument("--case", default=DEFAULT_CASE_ID)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--gsd-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FilmContinuityConfig(
        case_id=args.case,
        npz_path=args.npz_path,
        gsd_path=args.gsd_path,
    )

    if config.cache_path.exists() and not args.no_cache and not args.overwrite:
        result = FilmContinuityResult.from_cache_arrays(load_cache(config.cache_path))
    else:
        result = compute_film_continuity(config)
        if not args.no_cache:
            write_cache(
                config.cache_path,
                overwrite=args.overwrite,
                **result.as_cache_arrays(),
            )
    if args.plot:
        write_all_maps(
            result,
            config.output_dir,
            frame_idx=args.frame_idx,
            case_id=args.case,
        )
    print(result.residual_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
