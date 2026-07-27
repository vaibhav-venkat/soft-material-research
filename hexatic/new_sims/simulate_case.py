from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

import gsd.hoomd
import hoomd  # type: ignore
import numpy as np

from hexatic.constants import cylinder
from hexatic.confinement_comparison.storage import write_json_atomic

from .cases import (
    PASSIVE_KT,
    PASSIVE_STIFFNESS_MULTIPLIER,
    CaseKind,
    CasePaths,
    DEFAULT_OUTPUT_ROOT,
    NewSimCase,
    get_case,
)
from .geometry import generate_initial_arrays


def _prepare_outputs(paths: CasePaths, overwrite: bool) -> bool:
    paths.ensure_parent_dirs()
    if paths.simulation_complete_json.exists() and not overwrite:
        print(f"[new_sims.simulation] skipping complete case={paths.case.case_id}")
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
            "Refusing to reuse incomplete simulation output. "
            f"Pass --overwrite to replace it:\n{names}"
        )
    if overwrite:
        for path in existing:
            path.unlink()
    return True


def write_initial_state(case: NewSimCase, path: Path) -> dict[str, object]:
    positions, orientations, active_ids, metadata = generate_initial_arrays(case)
    if len(positions) != case.n_particles:
        raise AssertionError(f"expected {case.n_particles} particles, got {len(positions)}")

    frame = gsd.hoomd.Frame()
    frame.configuration.step = 0
    frame.configuration.box = [*case.stored_box, 0, 0, 0]
    frame.configuration.dimensions = case.dimensions
    frame.particles.N = case.n_particles
    frame.particles.position = positions.astype(np.float32)
    frame.particles.diameter = np.full(
        case.n_particles, cylinder.PARTICLE_DIAMETER, dtype=np.float32
    )
    moment_inertia = np.ones((case.n_particles, 3), dtype=np.float32)
    if case.is_2d:
        moment_inertia[:] = 0.0
        moment_inertia[:, 2] = 1.0
    frame.particles.moment_inertia = moment_inertia
    frame.particles.orientation = orientations.astype(np.float32)
    frame.particles.image = np.zeros((case.n_particles, 3), dtype=np.int32)
    if case.is_tracer:
        typeid = np.zeros(case.n_particles, dtype=np.uint32)
        typeid[active_ids] = 1
        frame.particles.types = ["passive", "active"]
        frame.particles.typeid = typeid
    else:
        frame.particles.types = ["A"]
        frame.particles.typeid = np.zeros(case.n_particles, dtype=np.uint32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=str(path), mode="w") as target:
        target.append(frame)
    return metadata


def _standard_epsilon() -> float:
    return (
        cylinder.SIMULATION.interaction_epsilon_multiplier
        * cylinder.SIMULATION.gamma
        * cylinder.SIMULATION.u0
        * cylinder.ANALYSIS.sigma
    )


def _pair_force(case: NewSimCase) -> hoomd.md.pair.LJ | None:
    if not case.has_pair_interaction:
        return None
    neighbor_list = hoomd.md.nlist.Cell(
        buffer=cylinder.SIMULATION.neighbor_list_buffer
    )
    pair = hoomd.md.pair.LJ(nlist=neighbor_list)
    epsilon = PASSIVE_EPSILON if case.is_tracer else _standard_epsilon()
    particle_types = ("passive", "active") if case.is_tracer else ("A",)
    for left_index, left in enumerate(particle_types):
        for right in particle_types[left_index:]:
            pair.params[(left, right)] = {
                "epsilon": epsilon,
                "sigma": cylinder.ANALYSIS.sigma,
            }
            pair.r_cut[(left, right)] = cylinder.ANALYSIS.wall_cutoff
    return pair


PASSIVE_EPSILON = PASSIVE_STIFFNESS_MULTIPLIER * PASSIVE_KT


def _wall_force(case: NewSimCase) -> hoomd.md.external.wall.LJ | None:
    if not case.wall_faces:
        return None
    if case.has_x_walls:
        half = 0.5 * case.lx
        walls = [
            hoomd.wall.Plane(origin=(-half, 0, 0), normal=(1, 0, 0)),
            hoomd.wall.Plane(origin=(half, 0, 0), normal=(-1, 0, 0)),
        ]
    else:
        walls = [
            hoomd.wall.Cylinder(
                radius=case.wall_radius,
                axis=(1, 0, 0),
                inside=True,
                open=True,
            )
        ]
    wall = hoomd.md.external.wall.LJ(walls=walls)
    epsilon = PASSIVE_EPSILON if case.is_tracer else _standard_epsilon()
    particle_types = ("passive", "active") if case.is_tracer else ("A",)
    for particle_type in particle_types:
        parameters = {
            "epsilon": epsilon,
            "sigma": cylinder.ANALYSIS.sigma,
            "r_cut": cylinder.ANALYSIS.wall_cutoff,
        }
        if case.has_x_walls:
            parameters["r_extrap"] = 0.98 * cylinder.ANALYSIS.wall_cutoff
        wall.params[particle_type] = parameters
    return wall


def make_simulation(
    case: NewSimCase,
    *,
    initial_gsd: Path,
    trajectory_gsd: Path,
    write_period: int,
    device_name: str,
    gpu_id: int | None,
) -> tuple[hoomd.Simulation, hoomd.write.GSD]:
    device = (
        hoomd.device.CPU()
        if device_name == "cpu"
        else hoomd.device.GPU(gpu_id=gpu_id)
    )
    simulation = hoomd.Simulation(device=device, seed=case.seed)
    simulation.create_state_from_gsd(filename=str(initial_gsd))
    integrator = hoomd.md.Integrator(dt=cylinder.SIMULATION.timestep)
    simulation.operations.integrator = integrator

    if case.is_tracer:
        passive = hoomd.filter.Type(["passive"])
        active_filter = hoomd.filter.Type(["active"])
        integrator.methods.extend(
            (
                hoomd.md.methods.Brownian(
                    filter=passive,
                    kT=PASSIVE_KT,
                    default_gamma=cylinder.SIMULATION.gamma,
                ),
                hoomd.md.methods.OverdampedViscous(
                    filter=active_filter,
                    default_gamma=cylinder.SIMULATION.gamma,
                ),
            )
        )
    else:
        active_filter = hoomd.filter.Type(["A"])
        integrator.methods.append(
            hoomd.md.methods.OverdampedViscous(
                filter=hoomd.filter.All(),
                default_gamma=cylinder.SIMULATION.gamma,
            )
        )

    pair = _pair_force(case)
    wall = _wall_force(case)
    if pair is not None:
        integrator.forces.append(pair)
    if wall is not None:
        integrator.forces.append(wall)

    active = hoomd.md.force.Active(filter=active_filter)
    active.use_orientation = True
    active_type = "active" if case.is_tracer else "A"
    active.active_force[active_type] = (
        cylinder.SIMULATION.gamma * cylinder.SIMULATION.u0,
        0.0,
        0.0,
    )
    active.active_torque[active_type] = (0.0, 0.0, 0.0)
    integrator.forces.append(active)
    simulation.operations += active.create_diffusion_updater(
        trigger=hoomd.trigger.Periodic(
            cylinder.SIMULATION.rotational_diffusion_period
        ),
        rotational_diffusion=1.0 / cylinder.SIMULATION.tau_r,
    )

    writer = hoomd.write.GSD(
        filename=str(trajectory_gsd),
        trigger=hoomd.trigger.Periodic(write_period),
        mode="wb",
        dynamic=["property", "particles/orientation", "particles/image"],
    )
    simulation.operations.writers.append(writer)
    return simulation, writer


def run_case(
    case: NewSimCase,
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
    if steps < period or steps % period:
        raise ValueError("run_steps must be positive and divisible by the write period")
    paths = CasePaths(case, output_root)
    if not _prepare_outputs(paths, overwrite):
        return

    metadata = case.as_metadata()
    metadata.update(
        status="running",
        run_steps=steps,
        trajectory_write_period=period,
        expected_frame_count=steps // period,
        device=device_name,
        local_gpu_id=gpu_id,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        initial_gsd=str(paths.initial_gsd),
        trajectory_gsd=str(paths.trajectory_gsd),
    )
    metadata.update(write_initial_state(case, paths.initial_gsd))
    write_json_atomic(paths.metadata_json, metadata)
    simulation, writer = make_simulation(
        case,
        initial_gsd=paths.initial_gsd,
        trajectory_gsd=paths.trajectory_gsd,
        write_period=period,
        device_name=device_name,
        gpu_id=gpu_id,
    )
    print(
        f"[new_sims.simulation] case={case.case_id} N={case.n_particles} "
        f"box={case.stored_box} steps={steps} period={period} "
        f"device={device_name} gpu={gpu_id}",
        flush=True,
    )
    simulation.run(steps)
    writer.flush()
    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as source:
        frame_count = len(source)
        final_step = int(source[-1].configuration.step) if frame_count else None
        finite = all(
            np.all(np.isfinite(np.asarray(frame.particles.position)))
            and np.all(np.isfinite(np.asarray(frame.particles.orientation)))
            for frame in source
        )
    expected = steps // period
    if frame_count != expected or final_step != steps:
        raise RuntimeError(
            f"trajectory completion mismatch: frames={frame_count}/{expected}, "
            f"step={final_step}/{steps}"
        )
    if not finite:
        raise RuntimeError(f"trajectory has non-finite particle data: {paths.trajectory_gsd}")
    metadata.update(status="complete", frame_count=frame_count, final_step=final_step)
    write_json_atomic(paths.metadata_json, metadata)
    write_json_atomic(
        paths.simulation_complete_json,
        {
            "schema": "hexatic.new_sims.simulation_complete.v1",
            "case_id": case.case_id,
            "status": "complete",
            "frame_count": frame_count,
            "final_step": final_step,
            "trajectory_gsd": str(paths.trajectory_gsd),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one new C=60.5D simulation.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case = get_case(args.case)
    if args.seed is not None:
        case = replace(case, seed=args.seed)
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
