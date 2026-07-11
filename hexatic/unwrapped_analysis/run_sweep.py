from __future__ import annotations

import argparse
import subprocess
import sys

from .cases import UnwrappedCase, all_cases, ensure_output_dirs, get_case


def _selected_cases(args: argparse.Namespace) -> tuple[UnwrappedCase, ...]:
    if args.all:
        return all_cases()
    if args.case:
        return tuple(get_case(case_id) for case_id in args.case)
    raise SystemExit("Select --all or one or more --case values.")


def _command_for_case(case: UnwrappedCase, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "hexatic.unwrapped_analysis.simulate_case_perfect_hexatic",
        "--case",
        case.case_id,
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.gpu_id is not None:
        command.extend(["--gpu-id", str(args.gpu_id)])
    return command


def run_cases(cases: tuple[UnwrappedCase, ...], args: argparse.Namespace) -> None:
    ensure_output_dirs()
    processes: list[tuple[UnwrappedCase, subprocess.Popen, object]] = []
    for case in cases:
        case.simulation_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = case.simulation_log.open("w")
        command = _command_for_case(case, args)
        log_handle.write(f"$ {' '.join(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((case, process, log_handle))
        print(f"started {case.case_id}: pid={process.pid}, log={case.simulation_log}")

    failures: list[tuple[UnwrappedCase, int]] = []
    for case, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append((case, return_code))
        print(f"finished {case.case_id}: return_code={return_code}")

    if failures:
        summary = "\n".join(
            f"{case.case_id}: return_code={return_code}, log={case.simulation_log}"
            for case, return_code in failures
        )
        raise SystemExit(f"One or more unwrapped simulations failed:\n{summary}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = _selected_cases(args)
    run_cases(cases, args)


if __name__ == "__main__":
    main()
