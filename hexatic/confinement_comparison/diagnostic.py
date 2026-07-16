from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import gsd.hoomd
import numpy as np

from hexatic.big_lx.backend import select_backend
from hexatic.constants import cylinder

from .analyze_case import _logged_force, _project_tangent
from .cases import (
    CasePaths,
    DEFAULT_OUTPUT_ROOT,
    GeometryKind,
    add_case_selection_arguments,
    select_cases,
)
from .geometry import stored_to_logical
from .spatial import PeriodicTree
from .simulate_case import make_simulation, write_initial_state
from .storage import write_json_atomic


def run_diagnostic(
    case,
    *,
    output_root: Path,
    period: int,
    overwrite: bool,
) -> dict[str, object]:
    if period < 1:
        raise ValueError("period must be positive")
    paths = CasePaths(case, output_root)
    paths.ensure_parent_dirs()
    initial_path = paths.diagnostic_gsd.with_name(f"initial_{case.case_id}.gsd")
    outputs = (initial_path, paths.diagnostic_gsd, paths.diagnostic_json)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Diagnostic outputs exist; use --overwrite:\n{names}")
    if overwrite:
        for path in existing:
            path.unlink()
    initial_metrics = write_initial_state(case, initial_path)
    objects = make_simulation(
        case,
        initial_gsd=initial_path,
        trajectory_gsd=paths.diagnostic_gsd,
        write_period=period,
        device_name="cpu",
        gpu_id=None,
    )
    objects.simulation.run(period, write_at_start=True)
    objects.writer.flush()
    backend = select_backend("numpy")
    with gsd.hoomd.open(name=str(paths.diagnostic_gsd), mode="r") as trajectory:
        steps = [int(frame.configuration.step) for frame in trajectory]
        if steps != [0, period]:
            raise RuntimeError(f"expected diagnostic steps [0, {period}], got {steps}")
        finite = True
        residuals = []
        tangency = []
        direction_means = []
        minimum_distances = []
        for frame in trajectory:
            stored_positions = np.asarray(frame.particles.position, dtype=np.float32)
            positions = stored_to_logical(stored_positions, case)
            orientation = stored_to_logical(
                backend.directions(np.asarray(frame.particles.orientation)), case
            )
            active_force = stored_to_logical(
                _logged_force(frame, "active", case.n_particles), case
            )
            finite &= bool(
                np.all(np.isfinite(positions))
                and np.all(np.isfinite(orientation))
                and np.all(np.isfinite(active_force))
            )
            if case.is_constrained:
                residuals.append(
                    float(np.max(np.abs(np.sum(positions[:, 1:3] ** 2, axis=1) - case.radius**2)))
                )
                effective = (
                    _project_tangent(orientation, positions)
                    if case.kind == GeometryKind.CYLINDER_RATTLE
                    else active_force
                    / float(cylinder.SIMULATION.gamma * cylinder.SIMULATION.u0)
                )
                normals = np.zeros_like(positions)
                normals[:, 1:3] = positions[:, 1:3] / np.linalg.norm(
                    positions[:, 1:3], axis=1, keepdims=True
                )
                tangency.append(float(np.max(np.abs(np.sum(effective * normals, axis=1)))))
                direction_means.append(np.mean(effective, axis=0).tolist())
            tree = PeriodicTree.build(positions, case)
            _, bonds = tree.nearest_bonds(positions, 1)
            minimum_distances.append(
                float(np.min(np.linalg.norm(bonds[:, 0], axis=1)))
            )
    if not finite:
        raise RuntimeError(f"non-finite diagnostic state for {case.case_id}")
    payload: dict[str, object] = {
        "schema": "hexatic.confinement_comparison.diagnostic.v1",
        "case": case.as_metadata(),
        "trajectory_gsd": str(paths.diagnostic_gsd),
        "frame_count": 2,
        "steps": steps,
        "finite": finite,
        "minimum_pair_distance": minimum_distances,
        "constraint_residual_max": max(residuals) if residuals else None,
        "active_normal_component_max": max(tangency) if tangency else None,
        "effective_direction_means": direction_means,
        **initial_metrics,
    }
    write_json_atomic(paths.diagnostic_json, payload)
    print(
        f"diagnostic case={case.case_id} steps={steps} "
        f"min_pair={min(minimum_distances):.9g} finite={finite}",
        flush=True,
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-frame CPU diagnostics.")
    add_case_selection_arguments(parser)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--period", type=int, default=int(1e5))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for selected in select_cases(args.all, args.case):
        case = replace(selected, seed=args.seed) if args.seed is not None else selected
        run_diagnostic(
            case,
            output_root=args.output_root,
            period=args.period,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
