"""CLI for rho-fitting PDE validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hexatic.rho_fitting.cache import write_npz_atomic
from hexatic.rho_fitting.config import DEFAULT_OUTPUT_DIR

from .model import run_validation_from_cache
from .plot import write_p_animation, write_q_animation, write_rho_animation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate rho-fitting closures with py-pde.")
    parser.add_argument("--case", default="radius_15D")
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--solver", choices=("filtered-euler", "euler", "scipy"), default="filtered-euler")
    parser.add_argument("--dt", type=float, default=None, help="maximum solver step; also Euler substep size")
    parser.add_argument("--filter-sigma", type=float, default=None, help="physical Gaussian filter width for filtered Euler")
    parser.add_argument("--mode", choices=("all", "full", "rho-only", "p-only", "q-only"), default="all")
    parser.add_argument("--rho-only", action="store_true", help="evolve rho while using cached P and Q fields")
    parser.add_argument("--plot-stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_path = args.cache or args.output_dir / f"{args.case}_fit_result.npz"
    mode = "rho-only" if args.rho_only else args.mode
    modes = ("full", "rho-only", "p-only", "q-only") if mode == "all" else (mode,)
    html_dir = args.output_dir / "html"
    output_paths = []
    for run_mode in modes:
        inputs, result = run_validation_from_cache(
            cache_path,
            max_frames=args.max_frames,
            solver=args.solver,
            dt=args.dt,
            filter_sigma=args.filter_sigma,
            mode=run_mode,
        )
        output_path = args.output_dir / f"{args.case}_pde_validation_{run_mode}.npz"
        write_npz_atomic(
            output_path,
            overwrite=args.overwrite,
            metadata={
                "case_id": args.case,
                "cache_path": str(cache_path),
                "analysis": "rho_fitting_pde_validation",
                "mode": run_mode,
            },
            rho_fit=result.rho_fit,
            P_fit=result.p_fit,
            Q_fit=result.q_fit,
            rho_true=result.rho_true,
            P_true=result.p_true,
            Q_true=result.q_true,
            times=result.times,
            rmse_t=result.rmse_t,
            r2_t=result.r2_t,
        )
        output_paths.append(output_path)
        if not args.no_plot:
            write_mode_outputs(html_dir, args.case, run_mode, inputs, result, args.plot_stride, args.overwrite)
        print(
            "[rho_fitting.pde_validation] "
            f"mode={run_mode} frames={result.times.size} rmse_final={result.rmse_t[-1]:.6g} "
            f"r2_final={result.r2_t[-1]:.6g} output={output_path}",
            flush=True,
        )
    return 0


def write_mode_outputs(
    html_dir: Path,
    case: str,
    mode: str,
    inputs,
    result,
    stride: int,
    overwrite: bool,
) -> None:
    if mode == "full":
        write_rho_animation(html_dir / f"{case}_rho_full.html", inputs, result, stride=stride, overwrite=overwrite)
        write_p_animation(html_dir / f"{case}_P_full.html", inputs, result, stride=stride, overwrite=overwrite)
        write_q_animation(html_dir / f"{case}_Q_full.html", inputs, result, stride=stride, overwrite=overwrite)
    elif mode == "rho-only":
        write_rho_animation(html_dir / f"{case}_rho_rho_only.html", inputs, result, stride=stride, overwrite=overwrite)
    elif mode == "p-only":
        write_p_animation(html_dir / f"{case}_P_P_only.html", inputs, result, stride=stride, overwrite=overwrite)
    elif mode == "q-only":
        write_q_animation(html_dir / f"{case}_Q_Q_only.html", inputs, result, stride=stride, overwrite=overwrite)
    else:
        raise AssertionError(f"unknown output mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
