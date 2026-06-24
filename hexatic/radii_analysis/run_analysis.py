from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .cases import RadiusCase, all_cases, ensure_output_dirs, get_case


def _selected_cases(args: argparse.Namespace) -> tuple[RadiusCase, ...]:
    if args.all:
        return all_cases(include_long_axis=args.include_long_axis)
    if args.case:
        return tuple(
            get_case(case_id, include_long_axis=args.include_long_axis)
            for case_id in args.case
        )
    raise SystemExit("Select --all or one or more --case values.")


def _command_for_case(case: RadiusCase, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "hexatic.radii_analysis.analyze_case",
        "--case",
        case.case_id,
        "--shear-series-stride",
        str(args.shear_series_stride),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.include_long_axis:
        command.append("--include-long-axis")
    if args.skip_shear_series:
        command.append("--skip-shear-series")
    return command


def run_cases(cases: tuple[RadiusCase, ...], args: argparse.Namespace) -> None:
    ensure_output_dirs()
    pending = list(cases)
    running: list[tuple[RadiusCase, subprocess.Popen, object]] = []
    failures: list[tuple[RadiusCase, int]] = []

    while pending or running:
        while pending and len(running) < args.workers:
            case = pending.pop(0)
            case.analysis_log.parent.mkdir(parents=True, exist_ok=True)
            log_handle = case.analysis_log.open("w")
            command = _command_for_case(case, args)
            log_handle.write(f"$ {' '.join(command)}\n")
            log_handle.flush()
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running.append((case, process, log_handle))
            print(f"started analysis {case.case_id}: pid={process.pid}, log={case.analysis_log}")

        still_running: list[tuple[RadiusCase, subprocess.Popen, object]] = []
        for case, process, log_handle in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((case, process, log_handle))
                continue
            log_handle.close()
            if return_code:
                failures.append((case, return_code))
            print(f"finished analysis {case.case_id}: return_code={return_code}")
        running = still_running
        if running:
            time.sleep(5.0)

    if failures:
        summary = "\n".join(
            f"{case.case_id}: return_code={return_code}, log={case.analysis_log}"
            for case, return_code in failures
        )
        raise SystemExit(f"One or more radius analyses failed:\n{summary}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shear-series-stride", type=int, default=1)
    parser.add_argument("--skip-shear-series", action="store_true")
    parser.add_argument(
        "--include-long-axis",
        action="store_true",
        help="Include the explicitly guarded Lx -> infinity case.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    return args


def main() -> None:
    args = _parse_args()
    cases = _selected_cases(args)
    run_cases(cases, args)


if __name__ == "__main__":
    main()
