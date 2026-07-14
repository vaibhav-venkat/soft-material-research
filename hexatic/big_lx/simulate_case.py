from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import gsd.hoomd
import hoomd
import numpy as np

from hexatic.constants import cylinder

from .cases import BigLxCase, CasePaths, DEFAULT_OUTPUT_ROOT, get_case
from .lattice import generate_unwrapped_lattice, outward_normal_quaternions


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


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
        text = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to reuse incomplete big-Lx simulation outputs. "
            f"Pass --overwrite to replace them:\n{text}"
        )
    if overwrite:
        for path in existing:
            path.unlink()
    return True


def _write_initial_state(case: BigLxCase, path: Path) -> None:
    positions, theta = generate_unwrapped_lattice(case)
    n_particles = positions.shape[0]
    transverse_length = 2.0 * case.wall_radius

    frame = gsd.hoomd.Frame()
    frame.particles.N = n_particles
    frame.particles.position = positions
    frame.particles.diameter = np.full(
        n_particles,
        cylinder.ANALYSIS.particle_diameter,
        dtype=np.float32,
    )
    frame.particles.moment_inertia = np.ones((n_particles, 3), dtype=np.float32)
    frame.particles.orientation = outward_normal_quaternions(theta)
    frame.particles.image = np.zeros((n_particles, 3), dtype=np.int32)
    frame.particles.typeid = np.zeros(n_particles, dtype=np.uint32)
    frame.particles.types = ["A"]
    frame.configuration.box = [case.lx, transverse_length, transverse_length, 0, 0, 0]
    frame.configuration.dimensions = 3

    path.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=str(path), mode="w") as target:
        target.append(frame)


def run_case(
    case: BigLxCase,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_steps: int | None = None,
    trajectory_write_period: int | None = None,
    overwrite: bool = False,
    device_name: str = "gpu",
    gpu_id: int | None = None,
) -> None:
    steps = case.run_steps if run_steps is None else int(run_steps)
    write_period = (
        case.trajectory_write_period
        if trajectory_write_period is None
        else int(trajectory_write_period)
    )
    if steps < 0:
        raise ValueError("run_steps must be non-negative")
    if write_period <= 0:
        raise ValueError("trajectory_write_period must be positive")

    paths = CasePaths(case, output_root)
    if not _prepare_outputs(paths, overwrite):
        return

    metadata = case.as_metadata()
    metadata.update(
        status="running",
        run_steps=steps,
        trajectory_write_period=write_period,
        initial_gsd=str(paths.initial_gsd),
        trajectory_gsd=str(paths.trajectory_gsd),
        device=device_name,
        gpu_id=gpu_id,
    )
    _write_json_atomic(paths.metadata_json, metadata)
    _write_initial_state(case, paths.initial_gsd)

    analysis = cylinder.ANALYSIS
    simulation = cylinder.SIMULATION
    device = (
        hoomd.device.CPU()
        if device_name == "cpu"
        else hoomd.device.GPU(gpu_id=gpu_id)
    )
    sim = hoomd.Simulation(device=device, seed=case.seed)
    sim.create_state_from_gsd(filename=str(paths.initial_gsd))

    integrator = hoomd.md.Integrator(dt=simulation.timestep)
    integrator.methods.append(hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All()))
    sim.operations.integrator = integrator

    neighbor_list = hoomd.md.nlist.Cell(buffer=simulation.neighbor_list_buffer)
    pair = hoomd.md.pair.LJ(nlist=neighbor_list)
    pair.params[("A", "A")] = {
        "epsilon": (
            simulation.interaction_epsilon_multiplier
            * simulation.gamma
            * simulation.u0
            * analysis.sigma
        ),
        "sigma": analysis.sigma,
    }
    pair.r_cut[("A", "A")] = analysis.wall_cutoff
    integrator.forces.append(pair)

    wall = hoomd.wall.Cylinder(
        radius=case.wall_radius,
        axis=(1, 0, 0),
        inside=True,
        open=True,
    )
    wall_pair = hoomd.md.external.wall.LJ(walls=[wall])
    wall_pair.params["A"] = {
        "sigma": analysis.sigma,
        "epsilon": (
            simulation.interaction_epsilon_multiplier
            * simulation.gamma
            * simulation.u0
            * analysis.sigma
        ),
        "r_cut": analysis.wall_cutoff,
    }
    integrator.forces.append(wall_pair)

    active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
    active.use_orientation = True
    active.active_force["A"] = (simulation.gamma * simulation.u0, 0.0, 0.0)
    active.active_torque["A"] = (0.0, 0.0, 0.0)
    integrator.forces.append(active)
    sim.operations += active.create_diffusion_updater(
        trigger=hoomd.trigger.Periodic(simulation.rotational_diffusion_period),
        rotational_diffusion=1 / simulation.tau_r,
    )

    logger = hoomd.logging.Logger(categories=["particle"])
    logger.add(pair, quantities=["forces", "virials"])
    logger.add(wall_pair, quantities=["forces", "virials"])
    sim.operations.writers.append(
        hoomd.write.GSD(
            filename=str(paths.trajectory_gsd),
            trigger=hoomd.trigger.Periodic(write_period),
            mode="wb",
            dynamic=["property", "particles/orientation"],
            logger=logger,
        )
    )

    print(
        f"case={case.case_id} C={case.circumference:.12g} R={case.radius:.12g} "
        f"Lx={case.lx:.12g} multiplier={case.lx_multiplier} "
        f"N={case.n_particles} rho_volume={case.volume_density:.12g} "
        f"steps={steps} write_period={write_period} seed={case.seed} "
        f"device={device_name}"
    )
    sim.run(steps)

    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as trajectory:
        frame_count = len(trajectory)
        final_step = int(trajectory[-1].configuration.step) if frame_count else None
    metadata.update(status="complete", frame_count=frame_count, final_step=final_step)
    _write_json_atomic(paths.metadata_json, metadata)
    _write_json_atomic(
        paths.simulation_complete_json,
        {
            "case_id": case.case_id,
            "status": "complete",
            "frame_count": frame_count,
            "final_step": final_step,
            "seed": case.seed,
            "trajectory_gsd": str(paths.trajectory_gsd),
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one exact big-Lx supercell case.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--trajectory-write-period", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the case's default simulation seed.",
    )
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
