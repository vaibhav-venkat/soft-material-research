"""CLI for the hydrodynamic model fitting workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .fitting.config import (
    DEFAULT_CASE_ID,
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_STLSQ_MAX_ITER,
    DEFAULT_STLSQ_THRESHOLD,
    FittingConfig,
)
from .fitting.fields import load_or_compute_fields
from .fitting.fit import FittingResult, compute_fitting
from .fitting.io_cache import load_cache, write_cache
from .fitting.plots import write_all_plots


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FittingConfig(
        case_id=args.case,
        ridge_alpha=args.ridge_alpha,
        stlsq_threshold=args.stlsq_threshold,
        stlsq_max_iter=args.stlsq_max_iter,
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
    _write_model_report(result, config.output_dir, case_id=args.case)

    # --- plots ---
    if not args.no_plot:
        write_all_plots(result, config.output_dir, case_id=args.case)

    # --- terminal summary ---
    print(result.summary())
    return 0


def _write_model_report(
    result: FittingResult,
    output_dir: str | Path,
    *,
    case_id: str = "radius_15D",
) -> Path:
    """Write a human-readable model report .txt."""
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{case_id}_model_report.txt"

    lines: list[str] = []
    _add = lines.append
    _add(f"Model Report  —  case: {case_id}")
    _add(f"{'=' * 60}")

    # Settings
    _add("\nRegression Settings")
    _add("-" * 40)
    _add(f"  ridge_alpha       = {result.ridge_alpha}")
    _add(f"  stlsq_threshold   = {result.stlsq_threshold}")
    _add(f"  stlsq_max_iter    = {result.stlsq_max_iter}")
    _add(f"  pocket_radius     = {result.pocket_radius}")

    # Mask
    masked = int(np.count_nonzero(result.mask))
    total = result.mask.size
    _add(f"\nShared Mask")
    _add("-" * 40)
    _add(f"  {masked} / {total} valid space-time samples")

    # Density
    _add(f"\nDensity Model  —  target = ∂_t ρ − S_cross")
    _add("-" * 40)
    for name, label, coef, active in zip(
        result.density.names,
        result.density.labels,
        result.density.coefficients,
        result.density.active,
    ):
        flag = " [active]" if active else ""
        _add(f"  {label:>30s}  {coef: .8e}{flag}")
    _add(f"  {'rows used':>30s}  {result.density.rows_used}")
    for key in ("correlation", "r2", "normalized_mae"):
        val = result.density.metrics.get(key, float("nan"))
        _add(f"  {key:>30s}  {val:.6g}")

    # Polarization
    _add(f"\nPolarization Model  —  target = ∂_t P")
    _add("-" * 40)
    for name, label, coef, active in zip(
        result.polarization.names,
        result.polarization.labels,
        result.polarization.coefficients,
        result.polarization.active,
    ):
        flag = " [active]" if active else ""
        _add(f"  {label:>30s}  {coef: .8e}{flag}")
    _add(f"  {'rows used':>30s}  {result.polarization.rows_used}")
    for key in ("correlation", "r2_x", "r2_y", "normalized_mae_x", "normalized_mae_y"):
        val = result.polarization.metrics.get(key, float("nan"))
        _add(f"  {key:>30s}  {val:.6g}")

    # Curl residual stats
    if result.curl_residual is not None:
        c = result.curl_residual[result.mask]
        if c.size > 0:
            _add(f"\nCurl of Polarization Residual")
            _add("-" * 40)
            _add(f"  {'mean':>30s}  {np.nanmean(c):.6g}")
            _add(f"  {'std':>30s}  {np.nanstd(c):.6g}")
            _add(f"  {'rms':>30s}  {np.sqrt(np.nanmean(c**2)):.6g}")
            _add(f"  {'max |curl|':>30s}  {np.nanmax(np.abs(c)):.6g}")

    _add(f"\n{'=' * 60}\n")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[fitting] Report saved: {path}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
