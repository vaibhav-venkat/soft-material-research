from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .cases import LONG_AXIS_CASE
    from .simulate_case import run_case
else:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from hexatic.radii_analysis.cases import LONG_AXIS_CASE
    from hexatic.radii_analysis.simulate_case import run_case


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded R=10D, Lx=500D limit script. This is intentionally "
            "not called by radii_analysis.sh."
        )
    )
    parser.add_argument("--yes-run-long-axis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.yes_run_long_axis:
        raise SystemExit(
            "This long-axis case is created for later use and is not part of "
            "the default radius sweep. Re-run with --yes-run-long-axis to execute it."
        )
    run_case(LONG_AXIS_CASE, overwrite=args.overwrite, gpu_id=args.gpu_id)


if __name__ == "__main__":
    main()
