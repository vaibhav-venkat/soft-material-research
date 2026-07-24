from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TextIO

from .cases import (
    FRAMES_PER_SHARD,
    CaseKind,
    CasePaths,
    DEFAULT_OUTPUT_ROOT,
    NewSimCase,
    get_case,
)


INITIAL_ASSIGNMENTS = (
    (CaseKind.IDEAL_ABP_CYLINDER.value, 0, 0),
    (CaseKind.X_WALLED_PRISM.value, 0, 1),
    (CaseKind.PERIODIC_2D_PLANE.value, 1, 0),
    (CaseKind.PERIODIC_3D_BULK.value, 1, 1),
)
TRACER_CASES = (
    CaseKind.SINGLE_ACTIVE_TRACER_CYLINDER.value,
    CaseKind.INVERSION_ACTIVE_PAIR_CYLINDER.value,
)


@dataclass
class RunningStage:
    case: NewSimCase
    gpu_id: int
    slot_id: int
    stage: str
    process: subprocess.Popen[bytes]
    log_handle: TextIO


def _parse_gpu_ids(value: str) -> tuple[int, int]:
    ids = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if len(ids) != 2 or len(set(ids)) != 2 or any(gpu_id < 0 for gpu_id in ids):
        raise ValueError("--gpu-ids must name exactly two distinct non-negative IDs")
    return ids[0], ids[1]


def _simulation_command(args: argparse.Namespace, case: NewSimCase) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "hexatic.new_sims.simulate_case",
        "--case",
        case.case_id,
        "--output-root",
        str(args.output_root),
        "--device",
        "gpu",
        "--gpu-id",
        "0",
    ]
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
    return command


def _analysis_command(args: argparse.Namespace, case: NewSimCase) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "hexatic.new_sims.analyze_case",
        "--case",
        case.case_id,
        "--output-root",
        str(args.output_root),
        "--backend",
        "jax",
        "--require-gpu",
        "--frames-per-shard",
        str(args.frames_per_shard),
    ]
    if args.overwrite:
        command.append("--overwrite")
    elif args.resume_analysis:
        command.append("--resume")
    return command


def _log_event(handle: TextIO, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"{timestamp} {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def _launch(
    args: argparse.Namespace,
    case: NewSimCase,
    *,
    physical_gpu_id: int,
    slot_id: int,
    stage: str,
    scheduler_log: TextIO,
) -> RunningStage:
    paths = CasePaths(case, args.output_root)
    paths.ensure_parent_dirs()
    command = (
        _simulation_command(args, case)
        if stage == "simulation"
        else _analysis_command(args, case)
    )
    log_path = paths.simulation_log if stage == "simulation" else paths.analysis_log
    log_handle = log_path.open("w")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    log_handle.write(
        f"$ {' '.join(command)}\n"
        f"CUDA_VISIBLE_DEVICES={physical_gpu_id}\n"
        f"pipeline_slot={slot_id}\n"
    )
    log_handle.flush()
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    _log_event(
        scheduler_log,
        f"event=started stage={stage} case={case.case_id} "
        f"pid={process.pid} gpu={physical_gpu_id} slot={slot_id} log={log_path}",
    )
    return RunningStage(
        case=case,
        gpu_id=physical_gpu_id,
        slot_id=slot_id,
        stage=stage,
        process=process,
        log_handle=log_handle,
    )


def _dry_run(gpu_ids: tuple[int, int]) -> None:
    print("[new_sims.pipeline dry-run] two resident slots per physical GPU")
    for case_id, gpu_index, slot_id in INITIAL_ASSIGNMENTS:
        case = get_case(case_id)
        print(
            f"gpu={gpu_ids[gpu_index]} slot={slot_id} "
            f"{case.case_id}: simulation -> immediate JAX analysis"
        )
    print(
        "after the first initial simulation+analysis pipeline completes: "
        f"launch {TRACER_CASES[0]} in that slot"
    )
    print(
        "after the next slot becomes available: "
        f"launch {TRACER_CASES[1]} there"
    )
    print("each tracer also runs simulation -> immediate JAX analysis")


def run_pipeline(args: argparse.Namespace) -> None:
    if args.overwrite and args.resume_analysis:
        raise ValueError("--overwrite and --resume-analysis are mutually exclusive")
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if args.dry_run:
        _dry_run(gpu_ids)
        return

    scheduler_path = args.output_root / "logs" / "pipeline.log"
    scheduler_path.parent.mkdir(parents=True, exist_ok=True)
    running: dict[tuple[int, int], RunningStage] = {}
    failures: list[tuple[str, str, int]] = []
    completed_initial = 0
    pending_tracers = [get_case(case_id) for case_id in TRACER_CASES]
    with scheduler_path.open("w") as scheduler_log:
        _log_event(
            scheduler_log,
            f"event=pipeline_started gpu_ids={gpu_ids} slots_per_gpu=2",
        )
        for case_id, gpu_index, slot_id in INITIAL_ASSIGNMENTS:
            physical_gpu = gpu_ids[gpu_index]
            running[(physical_gpu, slot_id)] = _launch(
                args,
                get_case(case_id),
                physical_gpu_id=physical_gpu,
                slot_id=slot_id,
                stage="simulation",
                scheduler_log=scheduler_log,
            )

        while running or pending_tracers:
            progress = False
            for key, job in tuple(running.items()):
                return_code = job.process.poll()
                if return_code is None:
                    continue
                progress = True
                job.log_handle.close()
                _log_event(
                    scheduler_log,
                    f"event=finished stage={job.stage} case={job.case.case_id} "
                    f"return_code={return_code} gpu={job.gpu_id} slot={job.slot_id}",
                )
                if return_code:
                    failures.append((job.case.case_id, job.stage, return_code))
                    del running[key]
                    continue
                if job.stage == "simulation":
                    running[key] = _launch(
                        args,
                        job.case,
                        physical_gpu_id=job.gpu_id,
                        slot_id=job.slot_id,
                        stage="analysis",
                        scheduler_log=scheduler_log,
                    )
                    continue
                del running[key]
                if not job.case.is_tracer:
                    completed_initial += 1

            if completed_initial:
                for physical_gpu in gpu_ids:
                    for slot_id in range(2):
                        key = (physical_gpu, slot_id)
                        if not pending_tracers or key in running:
                            continue
                        case = pending_tracers.pop(0)
                        running[key] = _launch(
                            args,
                            case,
                            physical_gpu_id=physical_gpu,
                            slot_id=slot_id,
                            stage="simulation",
                            scheduler_log=scheduler_log,
                        )
                        progress = True

            if not running and pending_tracers:
                raise RuntimeError(
                    "no initial case completed successfully; tracer cases cannot start"
                )
            if running and not progress:
                time.sleep(2.0)

        if failures:
            summary = "; ".join(
                f"{case}:{stage}:rc={code}" for case, stage, code in failures
            )
            _log_event(scheduler_log, f"event=pipeline_failed failures={summary}")
            raise SystemExit(f"new_sims pipeline failures: {summary}")
        _log_event(scheduler_log, "event=pipeline_complete")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete event-driven two-GPU new_sims pipeline."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frames-per-shard", type=int, default=FRAMES_PER_SHARD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_pipeline(_parse_args())


if __name__ == "__main__":
    main()
