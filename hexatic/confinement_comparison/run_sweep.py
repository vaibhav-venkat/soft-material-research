from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .cases import (
    CasePaths,
    DEFAULT_OUTPUT_ROOT,
    GeometryKind,
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
            "hexatic.confinement_comparison.simulate_case",
            "--case",
            case.case_id,
            "--output-root",
            str(args.output_root),
            "--device",
            args.device,
        ]
        if args.device == "gpu":
            command.extend(("--gpu-id", "0"))
        if args.run_steps is not None:
            command.extend(("--run-steps", str(args.run_steps)))
        if args.trajectory_write_period is not None:
            command.extend(("--trajectory-write-period", str(args.trajectory_write_period)))
        if args.seed is not None:
            command.extend(("--seed", str(args.seed)))
        if args.overwrite:
            command.append("--overwrite")
        jobs.append(
            ScheduledJob(
                case=case,
                command=tuple(command),
                log_path=CasePaths(case, args.output_root).simulation_log,
                preferred_gpu_id=(
                    0
                    if case.kind
                    in {GeometryKind.PRISM_SURFACE_AREA, GeometryKind.TWO_DIMENSION}
                    else 1
                    if case.is_sandwich
                    else 1
                    if case.kind == GeometryKind.PRISM_VOLUME
                    else 0
                ),
            )
        )
    return tuple(jobs)


def run_sweep(args: argparse.Namespace) -> None:
    run_jobs(
        simulation_jobs(args),
        workers=args.workers,
        gpu_ids=parse_gpu_ids(args.gpu_ids),
        use_gpu=args.device == "gpu",
        dry_run=args.dry_run,
        stage_name="simulation",
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_case_selection_arguments(parser)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confinement simulations.")
    add_arguments(parser)
    run_sweep(parser.parse_args())


if __name__ == "__main__":
    main()
