from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .cases import (
    DEFAULT_OUTPUT_ROOT,
    CasePaths,
    add_case_selection_arguments,
    select_cases,
)
from .scheduler import ScheduledJob, parse_gpu_ids, run_jobs


def simulation_jobs(args: argparse.Namespace) -> tuple[ScheduledJob, ...]:
    jobs = []
    for case in select_cases(args.all, args.case):
        command = [
            sys.executable,
            "-m",
            "hexatic.confinement_comparison.passive_dense.simulate_case",
            "--case",
            case.case_id,
            "--output-root",
            str(args.output_root),
            "--device",
            args.device,
        ]
        if args.device == "gpu":
            # CUDA_VISIBLE_DEVICES selects the physical GPU; HOOMD sees it as 0.
            command.extend(("--gpu-id", "0"))
        if args.run_steps is not None:
            command.extend(("--run-steps", str(args.run_steps)))
        if args.trajectory_write_period is not None:
            command.extend(
                ("--trajectory-write-period", str(args.trajectory_write_period))
            )
        if args.seed is not None:
            command.extend(("--seed", str(args.seed)))
        if args.overwrite:
            command.append("--overwrite")
        jobs.append(
            ScheduledJob(
                case=case,
                command=tuple(command),
                log_path=CasePaths(case, args.output_root).simulation_log,
            )
        )
    return tuple(jobs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the passive-cylinder and dense-2D simulation pipeline."
    )
    add_case_selection_arguments(parser)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_jobs(
        simulation_jobs(args),
        workers=args.workers,
        gpu_ids=parse_gpu_ids(args.gpu_ids),
        use_gpu=args.device == "gpu",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

