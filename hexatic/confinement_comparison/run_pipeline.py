from __future__ import annotations

import argparse
from pathlib import Path

from .analyze_case import LOCAL_POCKET_RADIUS
from .cases import DEFAULT_OUTPUT_ROOT
from .run_analysis import run_analysis
from .run_sweep import run_sweep


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run confinement simulations, then analysis after a barrier."
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--simulation-workers", type=int, default=3)
    parser.add_argument("--analysis-workers", type=int, default=3)
    parser.add_argument("--simulation-device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--backend", choices=("auto", "jax", "numpy"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pocket-radius", type=float, default=LOCAL_POCKET_RADIUS)
    parser.add_argument("--gaussian-cutoff-multiplier", type=float, default=5.0)
    parser.add_argument("--particle-block-size", type=int, default=2048)
    parser.add_argument("--target-shard-mib", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.require_gpu and args.simulation_device != "gpu":
        raise SystemExit("--require-gpu requires --simulation-device gpu")
    if args.overwrite and args.resume_analysis:
        raise SystemExit("--overwrite and --resume-analysis are mutually exclusive")
    common = {
        "all": args.all,
        "case": args.case,
        "output_root": args.output_root,
        "gpu_ids": args.gpu_ids,
        "overwrite": args.overwrite,
        "dry_run": args.dry_run,
    }
    simulation_args = argparse.Namespace(
        **common,
        workers=args.simulation_workers,
        device=args.simulation_device,
        run_steps=args.run_steps,
        trajectory_write_period=args.trajectory_write_period,
        seed=args.seed,
    )
    analysis_args = argparse.Namespace(
        **common,
        workers=args.analysis_workers,
        backend=args.backend,
        require_gpu=args.require_gpu,
        pocket_radius=args.pocket_radius,
        gaussian_cutoff_multiplier=args.gaussian_cutoff_multiplier,
        particle_block_size=args.particle_block_size,
        target_shard_mib=args.target_shard_mib,
        resume=args.resume_analysis,
    )
    print("[confinement.pipeline] stage=simulation", flush=True)
    run_sweep(simulation_args)
    print("[confinement.pipeline] simulation barrier complete; stage=analysis", flush=True)
    run_analysis(analysis_args)


if __name__ == "__main__":
    main()
