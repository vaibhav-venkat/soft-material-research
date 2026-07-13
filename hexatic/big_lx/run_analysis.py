from __future__ import annotations

import argparse
from pathlib import Path
import sys

from hexatic.constants import cylinder

from .cases import CasePaths, DEFAULT_OUTPUT_ROOT, select_cases
from .scheduler import ScheduledJob, parse_gpu_ids, run_jobs

LOCAL_POCKET_RADIUS = 2.0 * cylinder.ANALYSIS.particle_diameter


def analysis_jobs(args: argparse.Namespace) -> tuple[ScheduledJob, ...]:
    jobs = []
    for case in select_cases(args.all, args.case):
        command = [
            sys.executable,
            "-m",
            "hexatic.big_lx.analyze_case",
            "--case",
            case.case_id,
            "--output-root",
            str(args.output_root),
            "--backend",
            args.backend,
            "--pocket-radius",
            str(args.pocket_radius),
            "--gaussian-cutoff-multiplier",
            str(args.gaussian_cutoff_multiplier),
            "--particle-block-size",
            str(args.particle_block_size),
            "--target-shard-mib",
            str(args.target_shard_mib),
        ]
        if args.require_gpu:
            command.append("--require-gpu")
        if args.overwrite:
            command.append("--overwrite")
        jobs.append(
            ScheduledJob(
                case=case,
                command=tuple(command),
                log_path=CasePaths(case, args.output_root).analysis_log,
            )
        )
    return tuple(jobs)


def run_analysis(args: argparse.Namespace) -> None:
    run_jobs(
        analysis_jobs(args),
        workers=args.workers,
        gpu_ids=parse_gpu_ids(args.gpu_ids),
        use_gpu=args.backend != "numpy",
        dry_run=args.dry_run,
        stage_name="analysis",
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--backend", choices=("auto", "jax", "numpy"), default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--pocket-radius", type=float, default=LOCAL_POCKET_RADIUS)
    parser.add_argument("--gaussian-cutoff-multiplier", type=float, default=5.0)
    parser.add_argument("--particle-block-size", type=int, default=2048)
    parser.add_argument("--target-shard-mib", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded big-Lx analysis.")
    add_arguments(parser)
    return parser.parse_args()


def main() -> None:
    run_analysis(_parse_args())


if __name__ == "__main__":
    main()
