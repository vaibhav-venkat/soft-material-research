from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import NamedTuple

import gsd.hoomd
import hoomd
import numpy as np

from hexatic.constants import cylinder

from .cases import (
    DEFAULT_OUTPUT_ROOT,
    PASSIVE_KT,
    PASSIVE_STIFFNESS_MULTIPLIER,
    CasePaths,
    PassiveDenseCase,
    all_cases,
    get_case,
)
from .geometry import generate_initial_arrays


class SimulationObjects(NamedTuple):
    simulation: hoomd.Simulation
    writer: hoomd.write.GSD
    method: hoomd.md.methods.Method
    active: hoomd.md.force.Active | None


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def write_initial_state(case: PassiveDenseCase, path: Path) -> dict[str, object]:
    positions, orientations, directions = generate_initial_arrays(case)
    if len(positions) != case.n_particles:
        raise AssertionError(f"expected {case.n_particles} particles, got {len(positions)}")

    frame = gsd.hoomd.Frame()
    frame.configuration.step = 0
    frame.configuration.box = [*case.stored_box, 0, 0, 0]
    frame.configuration.dimensions = case.dimensions
    frame.particles.N = case.n_particles
    frame.particles.position = positions.astype(np.float32)
    frame.particles.diameter = np.full(
        case.n_particles,
        cylinder.ANALYSIS.particle_diameter,
        dtype=np.float32,
    )
    if case.is_dense_2d:
        moment_inertia = np.zeros((case.n_particles, 3), dtype=np.float32)
        moment_inertia[:, 2] = 1.0
    else:
        moment_inertia = np.ones((case.n_particles, 3), dtype=np.float32)
    frame.particles.moment_inertia = moment_inertia
    frame.particles.orientation = orientations.astype(np.float32)
    frame.particles.image = np.zeros((case.n_particles, 3), dtype=np.int32)
    frame.particles.typeid = np.zeros(case.n_particles, dtype=np.uint32)
    frame.particles.types = ["A"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=str(path), mode="w") as target:
        target.append(frame)

    return {
        "initial_direction_mean": np.mean(directions, axis=0).tolist(),
        "initial_direction_norm_min": float(np.min(np.linalg.norm(directions, axis=1))),
        "initial_direction_norm_max": float(np.max(np.linalg.norm(directions, axis=1))),
    }


def _pair_force(case: PassiveDenseCase) -> hoomd.md.pair.LJ:
    neighbor_list = hoomd.md.nlist.Cell(
        buffer=cylinder.SIMULATION.neighbor_list_buffer
    )
    pair = hoomd.md.pair.LJ(nlist=neighbor_list)
    epsilon = (
        PASSIVE_STIFFNESS_MULTIPLIER * PASSIVE_KT
        if case.is_passive_cylinder
        else cylinder.SIMULATION.interaction_epsilon_multiplier
        * cylinder.SIMULATION.gamma
        * cylinder.SIMULATION.u0
        * cylinder.ANALYSIS.sigma
    )
    pair.params[("A", "A")] = {
        "epsilon": epsilon,
        "sigma": cylinder.ANALYSIS.sigma,
    }
    pair.r_cut[("A", "A")] = cylinder.ANALYSIS.wall_cutoff
    return pair


def _wall_force(case: PassiveDenseCase) -> hoomd.md.external.wall.LJ:
    if case.is_passive_cylinder:
        walls = [
            hoomd.wall.Cylinder(
                radius=case.base.wall_radius,
                axis=(1, 0, 0),
                inside=True,
                open=True,
            )
        ]
        epsilon = PASSIVE_STIFFNESS_MULTIPLIER * PASSIVE_KT
    else:
        half_width = 0.5 * case.ly
        walls = [
            hoomd.wall.Plane(origin=(0, -half_width, 0), normal=(0, 1, 0)),
            hoomd.wall.Plane(origin=(0, half_width, 0), normal=(0, -1, 0)),
        ]
        epsilon = (
            cylinder.SIMULATION.interaction_epsilon_multiplier
            * cylinder.SIMULATION.gamma
            * cylinder.SIMULATION.u0
            * cylinder.ANALYSIS.sigma
        )
    parameters = {
        "sigma": cylinder.ANALYSIS.sigma,
        "epsilon": epsilon,
        "r_cut": cylinder.ANALYSIS.wall_cutoff,
    }
    if case.is_dense_2d:
        parameters["r_extrap"] = 0.98 * cylinder.ANALYSIS.wall_cutoff
    wall = hoomd.md.external.wall.LJ(walls=walls)
    wall.params["A"] = parameters
    return wall


def make_simulation(
    case: PassiveDenseCase,
    *,
    initial_gsd: Path,
    trajectory_gsd: Path,
    write_period: int,
    device_name: str,
    gpu_id: int | None,
) -> SimulationObjects:
    device = (
        hoomd.device.CPU()
        if device_name == "cpu"
        else hoomd.device.GPU(gpu_id=gpu_id)
    )
    simulation = hoomd.Simulation(device=device, seed=case.seed)
    simulation.create_state_from_gsd(filename=str(initial_gsd))
    all_particles = hoomd.filter.All()
    if case.is_passive_cylinder:
        method: hoomd.md.methods.Method = hoomd.md.methods.Brownian(
            filter=all_particles,
            kT=PASSIVE_KT,
            default_gamma=cylinder.SIMULATION.gamma,
        )
    else:
        method = hoomd.md.methods.OverdampedViscous(
            filter=all_particles,
            default_gamma=cylinder.SIMULATION.gamma,
        )
    integrator = hoomd.md.Integrator(
        dt=cylinder.SIMULATION.timestep,
        methods=[method],
    )
    simulation.operations.integrator = integrator

    pair = _pair_force(case)
    wall = _wall_force(case)
    integrator.forces.extend((pair, wall))

    active = None
    if case.is_dense_2d:
        active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
        active.use_orientation = True
        active.active_force["A"] = (
            cylinder.SIMULATION.gamma * cylinder.SIMULATION.u0,
            0.0,
            0.0,
        )
        active.active_torque["A"] = (0.0, 0.0, 0.0)
        integrator.forces.append(active)
        simulation.operations += active.create_diffusion_updater(
            trigger=hoomd.trigger.Periodic(
                cylinder.SIMULATION.rotational_diffusion_period
            ),
            rotational_diffusion=1.0 / cylinder.SIMULATION.tau_r,
        )

    logger = hoomd.logging.Logger(categories=["particle"])
    logger.add(pair, quantities=["forces"], user_name="pair")
    logger.add(wall, quantities=["forces"], user_name="wall")
    if active is not None:
        logger.add(active, quantities=["forces"], user_name="active")
    writer = hoomd.write.GSD(
        filename=str(trajectory_gsd),
        trigger=hoomd.trigger.Periodic(write_period),
        mode="wb",
        dynamic=["property", "particles/orientation"],
        logger=logger,
    )
    simulation.operations.writers.append(writer)
    return SimulationObjects(simulation, writer, method, active)


def _prepare_outputs(paths: CasePaths, overwrite: bool) -> bool:
    paths.ensure_parent_dirs()
    if paths.simulation_complete_json.exists() and not overwrite:
        print(f"skipping completed simulation {paths.case.case_id}")
        return False
    outputs = (
        paths.initial_gsd,
        paths.trajectory_gsd,
        paths.metadata_json,
        paths.simulation_complete_json,
    )
    existing = tuple(path for path in outputs if path.exists())
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to replace passive/dense outputs. "
            f"Pass --overwrite to replace them:\n{names}"
        )
    if overwrite:
        for path in existing:
            path.unlink()
    return True


def _validate_initial_on_cpu(path: Path, case: PassiveDenseCase) -> None:
    simulation = hoomd.Simulation(device=hoomd.device.CPU(), seed=case.seed)
    simulation.create_state_from_gsd(filename=str(path))
    snapshot = simulation.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        if snapshot.particles.N != case.n_particles:
            raise RuntimeError("CPU-loaded initial state has the wrong particle count")
        if simulation.state.box.dimensions != case.dimensions:
            raise RuntimeError("CPU-loaded initial state has the wrong dimensionality")


def write_initial_only(
    case: PassiveDenseCase,
    *,
    output_root: Path,
    overwrite: bool,
) -> Path:
    paths = CasePaths(case, output_root)
    paths.ensure_parent_dirs()
    existing = tuple(
        path for path in (paths.initial_gsd, paths.metadata_json) if path.exists()
    )
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Pass --overwrite to replace initial output(s):\n{names}")
    if overwrite:
        for path in existing:
            path.unlink()
    metadata = case.as_metadata()
    metadata.update(
        status="initial_only",
        initial_gsd=str(paths.initial_gsd),
        initial_validation_device="cpu",
    )
    metadata.update(write_initial_state(case, paths.initial_gsd))
    _validate_initial_on_cpu(paths.initial_gsd, case)
    _write_json_atomic(paths.metadata_json, metadata)
    print(
        f"wrote CPU-validated frame-0 initial state: {paths.initial_gsd}",
        flush=True,
    )
    return paths.initial_gsd


def run_case(
    case: PassiveDenseCase,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_steps: int | None = None,
    trajectory_write_period: int | None = None,
    overwrite: bool = False,
    device_name: str = "gpu",
    gpu_id: int | None = None,
) -> None:
    steps = case.run_steps if run_steps is None else int(run_steps)
    period = (
        case.trajectory_write_period
        if trajectory_write_period is None
        else int(trajectory_write_period)
    )
    if steps < 1 or period < 1:
        raise ValueError("run_steps and trajectory_write_period must be positive")
    if steps < period or steps % period:
        raise ValueError(
            "run_steps must be at least one period and divisible by "
            "trajectory_write_period"
        )
    paths = CasePaths(case, output_root)
    if not _prepare_outputs(paths, overwrite):
        return

    metadata = case.as_metadata()
    metadata.update(
        status="running",
        device=device_name,
        local_gpu_id=gpu_id,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        run_steps=steps,
        trajectory_write_period=period,
        expected_frame_count=steps // period,
        initial_gsd=str(paths.initial_gsd),
        trajectory_gsd=str(paths.trajectory_gsd),
    )
    metadata.update(write_initial_state(case, paths.initial_gsd))
    _write_json_atomic(paths.metadata_json, metadata)
    objects = make_simulation(
        case,
        initial_gsd=paths.initial_gsd,
        trajectory_gsd=paths.trajectory_gsd,
        write_period=period,
        device_name=device_name,
        gpu_id=gpu_id,
    )
    print(
        f"case={case.case_id} N={case.n_particles} box={case.stored_box} "
        f"steps={steps} period={period} device={device_name}",
        flush=True,
    )
    objects.simulation.run(steps)
    objects.writer.flush()
    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as trajectory:
        frame_count = len(trajectory)
        final_step = int(trajectory[-1].configuration.step) if frame_count else None
        finite = all(
            np.all(np.isfinite(np.asarray(frame.particles.position)))
            for frame in trajectory
        )
        planar = not case.is_dense_2d or all(
            np.allclose(np.asarray(frame.particles.position)[:, 2], 0.0, atol=1e-7)
            for frame in trajectory
        )
    if frame_count != steps // period:
        raise RuntimeError(f"expected {steps // period} frames, found {frame_count}")
    if not finite:
        raise RuntimeError("trajectory contains non-finite positions")
    if not planar:
        raise RuntimeError("2D trajectory contains nonzero z positions")

    metadata.update(status="complete", frame_count=frame_count, final_step=final_step)
    _write_json_atomic(paths.metadata_json, metadata)
    _write_json_atomic(
        paths.simulation_complete_json,
        {
            "schema": "hexatic.confinement_comparison.passive_dense.complete.v1",
            "case_id": case.case_id,
            "status": "complete",
            "frame_count": frame_count,
            "final_step": final_step,
            "seed": case.seed,
            "trajectory_gsd": str(paths.trajectory_gsd),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one passive/dense case.")
    parser.add_argument(
        "--case",
        required=True,
        choices=tuple(case.case_id for case in all_cases()),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--initial-only",
        action="store_true",
        help="Write the initial GSD, validate it on CPU, and exit without integrating.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case = get_case(args.case)
    if args.seed is not None:
        case = replace(case, seed=args.seed)
    if args.initial_only:
        write_initial_only(
            case,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
        return
    run_case(
        case,
        output_root=args.output_root,
        run_steps=args.run_steps,
        trajectory_write_period=args.trajectory_write_period,
        overwrite=args.overwrite,
        device_name=args.device,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
