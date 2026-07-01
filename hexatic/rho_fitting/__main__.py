"""CLI for the rho fitting workflow."""

from __future__ import annotations

import argparse

from .config import RhoFittingConfig
from .fit import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit density dynamics.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RhoFittingConfig(
        overwrite=args.overwrite,
        make_plots=not args.no_plots,
    )
    result = run(config)
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
