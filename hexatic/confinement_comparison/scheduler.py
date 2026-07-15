from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import TextIO

from .cases import ComparisonCase


@dataclass(frozen=True)
class ScheduledJob:
    case: ComparisonCase
    command: tuple[str, ...]
    log_path: Path
    preferred_gpu_id: int | None = None


def parse_gpu_ids(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return 0, 1
    ids = tuple(int(token.strip()) for token in value.split(",") if token.strip())
    if not ids or any(device_id < 0 for device_id in ids):
        raise ValueError("--gpu-ids must contain non-negative comma-separated IDs")
    return ids


def run_jobs(
    jobs: tuple[ScheduledJob, ...],
    *,
    workers: int,
    gpu_ids: tuple[int, ...],
    use_gpu: bool,
    dry_run: bool,
    stage_name: str,
) -> None:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if use_gpu and not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    if use_gpu:
        unavailable = sorted(
            {
                job.preferred_gpu_id
                for job in jobs
                if job.preferred_gpu_id is not None
                and job.preferred_gpu_id not in gpu_ids
            }
        )
        if unavailable:
            raise ValueError(
                "scheduled jobs require GPU IDs that were not selected: "
                + ", ".join(str(device_id) for device_id in unavailable)
            )
    if dry_run:
        for index, job in enumerate(jobs):
            gpu_id = (
                job.preferred_gpu_id
                if use_gpu and job.preferred_gpu_id is not None
                else gpu_ids[index % len(gpu_ids)]
                if use_gpu
                else None
            )
            print(
                f"[dry-run] {stage_name} case={job.case.case_id} gpu={gpu_id} "
                f"log={job.log_path} command={' '.join(job.command)}"
            )
        return

    pending = list(jobs)
    running: list[tuple[ScheduledJob, subprocess.Popen, TextIO, int | None]] = []
    failures: list[tuple[ScheduledJob, int]] = []
    launch_count = 0
    while pending or running:
        while pending and len(running) < workers:
            job = pending.pop(0)
            gpu_id = (
                job.preferred_gpu_id
                if use_gpu and job.preferred_gpu_id is not None
                else gpu_ids[launch_count % len(gpu_ids)]
                if use_gpu
                else None
            )
            launch_count += 1
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = job.log_path.open("w")
            environment = os.environ.copy()
            if gpu_id is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            log_handle.write(
                f"$ {' '.join(job.command)}\nCUDA_VISIBLE_DEVICES={gpu_id}\n"
            )
            log_handle.flush()
            process = subprocess.Popen(
                job.command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            running.append((job, process, log_handle, gpu_id))
            print(
                f"started {stage_name} {job.case.case_id}: "
                f"pid={process.pid}, gpu={gpu_id}, log={job.log_path}",
                flush=True,
            )
        still_running = []
        for job, process, log_handle, gpu_id in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((job, process, log_handle, gpu_id))
                continue
            log_handle.close()
            if return_code:
                failures.append((job, return_code))
            print(
                f"finished {stage_name} {job.case.case_id}: return_code={return_code}",
                flush=True,
            )
        running = still_running
        if running:
            time.sleep(2.0)
    if failures:
        summary = "\n".join(
            f"{job.case.case_id}: return_code={code}, log={job.log_path}"
            for job, code in failures
        )
        raise SystemExit(f"One or more {stage_name} jobs failed:\n{summary}")
