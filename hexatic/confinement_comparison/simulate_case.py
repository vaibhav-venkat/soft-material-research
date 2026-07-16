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

from .cases import ComparisonCase, CasePaths, DEFAULT_OUTPUT_ROOT, GeometryKind, get_case
from .geometry import generate_cylinder_film, generate_planar_lattice, stored_to_logical


class SimulationObjects(NamedTuple):
    simulation: hoomd.Simulation
    writer: hoomd.write.GSD
    active: hoomd.md.force.Active


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _initial_arrays(case: ComparisonCase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case.is_cylinder:
        return generate_cylinder_film(case)
    return generate_planar_lattice(case)


def write_initial_state(case: ComparisonCase, path: Path) -> dict[str, object]:
    positions, orientations, directions = _initial_arrays(case)
    n_particles = len(positions)
    if n_particles != case.n_particles:
        raise AssertionError(f"expected {case.n_particles} particles, got {n_particles}")
    frame = gsd.hoomd.Frame()
    frame.particles.N = n_particles
    frame.particles.position = positions.astype(np.float32)
    frame.particles.diameter = np.full(
        n_particles, cylinder.ANALYSIS.particle_diameter, dtype=np.float32
    )
    if case.is_2d:
        moment_inertia = np.zeros((n_particles, 3), dtype=np.float32)
        moment_inertia[:, 2] = 1.0
    else:
        moment_inertia = np.ones((n_particles, 3), dtype=np.float32)
    frame.particles.moment_inertia = moment_inertia
    frame.particles.orientation = orientations.astype(np.float32)
    frame.particles.image = np.zeros((n_particles, 3), dtype=np.int32)
    frame.particles.typeid = np.zeros(n_particles, dtype=np.uint32)
    frame.particles.types = ["A"]
    frame.configuration.box = [*case.stored_box, 0, 0, 0]
    frame.configuration.dimensions = case.dimensions
    path.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=str(path), mode="w") as target:
        target.append(frame)

    logical_directions = stored_to_logical(directions, case)
    return {
        "initial_direction_mean": np.mean(logical_directions, axis=0).tolist(),
        "initial_direction_norm_min": float(
            np.min(np.linalg.norm(logical_directions, axis=1))
        ),
        "initial_direction_norm_max": float(
            np.max(np.linalg.norm(logical_directions, axis=1))
        ),
    }


def _wall_force(case: ComparisonCase) -> hoomd.md.external.wall.LJ:
    if case.is_prism:
        half = case.prism_wall_half_width
        walls = [
            hoomd.wall.Plane(origin=(0, -half, 0), normal=(0, 1, 0)),
            hoomd.wall.Plane(origin=(0, half, 0), normal=(0, -1, 0)),
            hoomd.wall.Plane(origin=(0, 0, -half), normal=(0, 0, 1)),
            hoomd.wall.Plane(origin=(0, 0, half), normal=(0, 0, -1)),
        ]
    elif case.is_sandwich:
        half = 0.5 * case.transverse_span
        walls = [
            hoomd.wall.Plane(origin=(0, 0, -half), normal=(0, 0, 1)),
            hoomd.wall.Plane(origin=(0, 0, half), normal=(0, 0, -1)),
        ]
    elif case.is_2d:
        half = 0.5 * case.transverse_span
        walls = [
            hoomd.wall.Plane(origin=(0, -half, 0), normal=(0, 1, 0)),
            hoomd.wall.Plane(origin=(0, half, 0), normal=(0, -1, 0)),
        ]
    else:
        walls = [
            hoomd.wall.Cylinder(
                radius=case.base.wall_radius,
                axis=(0, 0, 1),
                inside=True,
                open=True,
            )
        ]
    wall_pair = hoomd.md.external.wall.LJ(walls=walls)
    wall_pair.params["A"] = {
        "sigma": cylinder.ANALYSIS.sigma,
        "epsilon": (
            cylinder.SIMULATION.interaction_epsilon_multiplier
            * cylinder.SIMULATION.gamma
            * cylinder.SIMULATION.u0
            * cylinder.ANALYSIS.sigma
        ),
        "r_cut": cylinder.ANALYSIS.wall_cutoff,
    }
    return wall_pair


def make_simulation(
    case: ComparisonCase,
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
    sim = hoomd.Simulation(device=device, seed=case.seed)
    sim.create_state_from_gsd(filename=str(initial_gsd))
    all_particles = hoomd.filter.All()

    manifold = None
    if case.is_cylinder:
        manifold = hoomd.md.manifold.Cylinder(r=case.radius)
        method = hoomd.md.methods.rattle.OverdampedViscous(
            filter=all_particles,
            manifold_constraint=manifold,
            tolerance=1e-6,
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
    sim.operations.integrator = integrator

    neighbor_list = hoomd.md.nlist.Cell(
        buffer=cylinder.SIMULATION.neighbor_list_buffer
    )
    pair = hoomd.md.pair.LJ(nlist=neighbor_list)
    pair.params[("A", "A")] = {
        "epsilon": (
            cylinder.SIMULATION.interaction_epsilon_multiplier
            * cylinder.SIMULATION.gamma
            * cylinder.SIMULATION.u0
            * cylinder.ANALYSIS.sigma
        ),
        "sigma": cylinder.ANALYSIS.sigma,
    }
    pair.r_cut[("A", "A")] = cylinder.ANALYSIS.wall_cutoff
    integrator.forces.append(pair)

    wall_pair = _wall_force(case)
    integrator.forces.append(wall_pair)

    if case.kind == GeometryKind.CYLINDER_RATTLE_TANGENT:
        assert manifold is not None
        active = hoomd.md.force.ActiveOnManifold(
            filter=hoomd.filter.Type(["A"]),
            manifold_constraint=manifold,
        )
    else:
        active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
        active.use_orientation = True
    active.active_force["A"] = (
        cylinder.SIMULATION.gamma * cylinder.SIMULATION.u0,
        0.0,
        0.0,
    )
    active.active_torque["A"] = (0.0, 0.0, 0.0)
    integrator.forces.append(active)
    sim.operations += active.create_diffusion_updater(
        trigger=hoomd.trigger.Periodic(
            cylinder.SIMULATION.rotational_diffusion_period
        ),
        rotational_diffusion=1.0 / cylinder.SIMULATION.tau_r,
    )

    logger = hoomd.logging.Logger(categories=["particle"])
    logger.add(pair, quantities=["forces"], user_name="pair")
    logger.add(wall_pair, quantities=["forces"], user_name="wall")
    logger.add(active, quantities=["forces"], user_name="active")
    writer = hoomd.write.GSD(
        filename=str(trajectory_gsd),
        trigger=hoomd.trigger.Periodic(write_period),
        mode="wb",
        dynamic=["property", "particles/orientation"],
        logger=logger,
    )
    sim.operations.writers.append(writer)
    return SimulationObjects(sim, writer, active)


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
            "Refusing to reuse incomplete confinement outputs. "
            f"Pass --overwrite to replace them:\n{names}"
        )
    if overwrite:
        for path in existing:
            path.unlink()
    return True


def run_case(
    case: ComparisonCase,
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
    if steps < period:
        raise ValueError("run_steps must be at least one trajectory write period")
    if steps % period:
        raise ValueError(
            "run_steps must be divisible by trajectory_write_period so the "
            "final simulation step is present in the trajectory"
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
        f"case={case.case_id} geometry={case.kind.value} N={case.n_particles} "
        f"logical_Lx={case.lx:.12g} stored_box={case.stored_box} "
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
            and np.all(np.isfinite(np.asarray(frame.particles.orientation)))
            for frame in trajectory
        )
        planar_2d = not case.is_2d or all(
            np.allclose(np.asarray(frame.particles.position)[:, 2], 0.0, atol=1e-7)
            for frame in trajectory
        )
    expected = steps // period
    if frame_count != expected:
        raise RuntimeError(f"expected {expected} trajectory frames, found {frame_count}")
    if not finite:
        raise RuntimeError(f"non-finite trajectory data in {paths.trajectory_gsd}")
    if not planar_2d:
        raise RuntimeError(f"2D trajectory contains nonzero z positions: {paths.trajectory_gsd}")
    metadata.update(status="complete", frame_count=frame_count, final_step=final_step)
    _write_json_atomic(paths.metadata_json, metadata)
    _write_json_atomic(
        paths.simulation_complete_json,
        {
            "schema": "hexatic.confinement_comparison.simulation_complete.v1",
            "case_id": case.case_id,
            "status": "complete",
            "frame_count": frame_count,
            "final_step": final_step,
            "seed": case.seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "trajectory_gsd": str(paths.trajectory_gsd),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one confinement comparison case.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
