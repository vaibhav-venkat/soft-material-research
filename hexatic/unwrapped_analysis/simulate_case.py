from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gsd.hoomd
import hoomd
import numpy as np

from hexatic.constants import cylinder

from .cases import UnwrappedCase, ensure_output_dirs, get_case


def _ensure_can_write(paths: tuple[Path, ...], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        text = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing file(s):\n{text}")


def generate_unwrapped_lattice(case: UnwrappedCase) -> tuple[np.ndarray, np.ndarray]:
    positions = np.empty((case.n_particles, 3), dtype=np.float64)
    theta = np.empty(case.n_particles, dtype=np.float64)

    particle_idx = 0
    x0 = -0.5 * case.lx + 0.5 * case.h
    for j in range(case.n_x):
        x = x0 + j * case.h
        offset = 0.5 * (j % 2)
        for i in range(case.n_theta):
            surface_distance = ((i + offset) * case.a) % case.circumference
            angle = surface_distance / case.radius
            positions[particle_idx] = (
                x,
                case.radius * math.sin(angle),
                case.radius * math.cos(angle),
            )
            theta[particle_idx] = angle
            particle_idx += 1

    return positions, theta


def outward_normal_quaternions(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    half_turn = math.sqrt(0.5)
    quaternions = np.column_stack(
        (
            np.full(theta.shape, half_turn, dtype=np.float64),
            np.zeros(theta.shape, dtype=np.float64),
            -np.cos(theta) * half_turn,
            np.sin(theta) * half_turn,
        )
    )
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]
    return quaternions


def _write_initial_state(case: UnwrappedCase) -> None:
    analysis = cylinder.ANALYSIS

    np.random.seed(case.seed)
    position, theta = generate_unwrapped_lattice(case)
    n_particles = position.shape[0]

    transverse_length = 2.0 * case.wall_radius

    frame = gsd.hoomd.Frame()
    frame.particles.N = n_particles
    frame.particles.position = position
    frame.particles.diameter = [analysis.particle_diameter] * n_particles
    frame.particles.moment_inertia = [(1, 1, 1)] * n_particles
    frame.particles.orientation = outward_normal_quaternions(theta)
    frame.particles.image = np.zeros((n_particles, 3), dtype=np.int32)
    frame.particles.typeid = [0] * n_particles
    frame.configuration.box = [case.lx, transverse_length, transverse_length, 0, 0, 0]
    frame.configuration.dimensions = 3
    frame.particles.types = ["A"]

    case.initial_gsd.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=str(case.initial_gsd), mode="w") as target:
        target.append(frame)


def run_case(
    case: UnwrappedCase,
    overwrite: bool = False,
    gpu_id: int | None = None,
) -> None:
    ensure_output_dirs()
    _ensure_can_write(
        (case.initial_gsd, case.trajectory_gsd, case.metadata_json),
        overwrite=overwrite,
    )

    case.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    case.metadata_json.write_text(json.dumps(case.as_metadata(), indent=2) + "\n")
    _write_initial_state(case)

    analysis = cylinder.ANALYSIS
    simulation = cylinder.SIMULATION
    device = hoomd.device.GPU(gpu_id=gpu_id)
    sim = hoomd.Simulation(device=device, seed=case.seed)
    sim.create_state_from_gsd(filename=str(case.initial_gsd))

    integrator = hoomd.md.Integrator(dt=simulation.timestep)
    sim.operations.integrator = integrator
    filter_all = hoomd.filter.All()

    integrator.methods.append(hoomd.md.methods.OverdampedViscous(filter=filter_all))

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[("A", "A")] = dict(
        epsilon=50 * simulation.gamma * simulation.u0 * analysis.sigma,
        sigma=analysis.sigma,
    )
    lj.r_cut[("A", "A")] = analysis.wall_cutoff
    integrator.forces.append(lj)

    walls = [
        hoomd.wall.Cylinder(
            radius=case.wall_radius,
            axis=(1, 0, 0),
            inside=True,
            open=True,
        )
    ]
    lj_wall = hoomd.md.external.wall.LJ(walls=walls)
    lj_wall.params["A"] = {
        "sigma": analysis.sigma,
        "epsilon": 50 * simulation.gamma * simulation.u0 * analysis.sigma,
        "r_cut": analysis.wall_cutoff,
    }
    integrator.forces.append(lj_wall)

    active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
    active.use_orientation = True
    active.active_force["A"] = (simulation.gamma * simulation.u0, 0.0, 0.0)
    active.active_torque["A"] = (0.0, 0.0, 0.0)
    integrator.forces.append(active)

    rot_diff = active.create_diffusion_updater(
        trigger=hoomd.trigger.Periodic(simulation.rotational_diffusion_period),
        rotational_diffusion=1 / simulation.tau_r,
    )
    sim.operations += rot_diff

    logger = hoomd.logging.Logger(categories=["particle"])
    logger.add(lj, quantities=["forces", "virials"])
    logger.add(lj_wall, quantities=["forces", "virials"])

    gsd_writer = hoomd.write.GSD(
        filename=str(case.trajectory_gsd),
        trigger=hoomd.trigger.Periodic(case.trajectory_write_period),
        mode="wb",
        dynamic=["property", "particles/orientation"],
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    print(
        f"case={case.case_id} label={case.label} "
        f"R={case.radius:.12g} C={case.circumference:.12g} "
        f"Ntheta={case.n_theta} Nx={case.n_x} "
        f"Lx_target={case.lx_target:.12g} Lx={case.lx:.12g} "
        f"wall_R={case.wall_radius:.12g} N={case.n_particles} "
        f"steps={case.run_steps} write_period={case.trajectory_write_period} "
        f"output={case.trajectory_gsd}"
    )
    sim.run(case.run_steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_case(
        get_case(args.case),
        overwrite=args.overwrite,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
