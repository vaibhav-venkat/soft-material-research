from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import gsd.hoomd
import hoomd
import numpy as np

from hexatic.constants import cylinder

from .cases import GSD_DIR, UnwrappedCase, ensure_output_dirs, get_case


def _output_gsd(case: UnwrappedCase) -> Path:
    return GSD_DIR / f"trajectory_{case.case_id}_last_frame.gsd"


def _ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def random_uniform_quaternions(
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    u1 = rng.random(n_particles)
    u2 = rng.random(n_particles)
    u3 = rng.random(n_particles)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    quaternions = np.column_stack((qw, qx, qy, qz))
    quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]
    return quaternions


def _inner_lattice_positions(
    box: np.ndarray,
    cylinder_radius: float,
) -> np.ndarray:
    """Build a dense rectangular lattice strictly inside the cylinder."""
    analysis = cylinder.ANALYSIS
    lx = float(box[0])
    axial_extent = 0.5 * lx - analysis.last_frame_lattice_axial_gap
    corner_radius = cylinder_radius - analysis.last_frame_lattice_radial_gap
    transverse_extent = corner_radius / math.sqrt(2.0)
    if axial_extent <= 0.0 or transverse_extent <= 0.0:
        raise ValueError("Configured lattice gaps leave no interior lattice volume")

    target_spacing = analysis.last_frame_lattice_spacing
    n_axial = max(2, round(2.0 * axial_extent / target_spacing) + 1)
    n_transverse = max(2, round(2.0 * transverse_extent / target_spacing) + 1)
    x = np.linspace(-axial_extent, axial_extent, n_axial)
    transverse = np.linspace(-transverse_extent, transverse_extent, n_transverse)
    xv, yv, zv = np.meshgrid(x, transverse, transverse, indexing="ij")
    return np.column_stack((xv.ravel(), yv.ravel(), zv.ravel()))


def _write_refilled_initial_frame(case: UnwrappedCase, output_gsd: Path) -> tuple[int, int]:
    analysis = cylinder.ANALYSIS
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as trajectory:
        if len(trajectory) == 0:
            raise ValueError(f"Trajectory contains no frames: {case.trajectory_gsd}")
        source = trajectory[-1]

    source_positions = np.asarray(source.particles.position)
    radial_distance = np.linalg.norm(source_positions[:, 1:3], axis=1)
    inner_radius = case.radius - analysis.frozen_shell_delta
    shell_mask = radial_distance > inner_radius
    shell_indices = np.flatnonzero(shell_mask)
    n_shell = shell_indices.size
    n_deleted_inner = source_positions.shape[0] - n_shell
    if n_deleted_inner == 0:
        raise ValueError(f"No interior particles to replace for case {case.case_id}")

    box = np.asarray(source.configuration.box, dtype=np.float64)
    lattice_positions = _inner_lattice_positions(box, case.radius)
    n_inner = lattice_positions.shape[0]
    rng = np.random.default_rng(case.seed)

    frame = gsd.hoomd.Frame()
    frame.configuration.box = box
    frame.configuration.dimensions = source.configuration.dimensions
    frame.configuration.step = source.configuration.step
    frame.particles.N = n_shell + n_inner
    simulation = cylinder.SIMULATION
    frame.particles.types = [
        simulation.shell_particle_type,
        simulation.center_particle_type,
    ]
    typeids = np.full(
        frame.particles.N,
        simulation.shell_particle_type_id,
        dtype=np.uint32,
    )
    typeids[n_shell:] = simulation.center_particle_type_id
    frame.particles.typeid = typeids
    frame.particles.position = np.vstack(
        (source_positions[shell_indices], lattice_positions)
    )
    frame.particles.orientation = np.vstack(
        (
            np.asarray(source.particles.orientation)[shell_indices],
            random_uniform_quaternions(n_inner, rng),
        )
    )
    frame.particles.velocity = np.zeros((frame.particles.N, 3), dtype=np.float32)
    frame.particles.image = np.zeros((frame.particles.N, 3), dtype=np.int32)
    frame.particles.diameter = np.full(
        frame.particles.N,
        analysis.particle_diameter,
        dtype=np.float32,
    )
    frame.particles.moment_inertia = np.ones(
        (frame.particles.N, 3),
        dtype=np.float32,
    )

    with gsd.hoomd.open(name=str(output_gsd), mode="w") as target:
        target.append(frame)
    return n_shell, n_inner


def run_case(
    case: UnwrappedCase,
    run_steps: int = cylinder.SIMULATION.last_frame_run_steps,
    overwrite: bool = False,
    device_name: str = "gpu",
    gpu_id: int | None = None,
) -> None:
    if run_steps < 0:
        raise ValueError("run_steps must be non-negative")

    ensure_output_dirs()
    if not case.trajectory_gsd.exists():
        raise FileNotFoundError(f"Missing trajectory GSD: {case.trajectory_gsd}")

    output_gsd = _output_gsd(case)
    _ensure_can_write(output_gsd, overwrite)
    n_shell, n_inner = _write_refilled_initial_frame(case, output_gsd)

    analysis = cylinder.ANALYSIS
    simulation = cylinder.SIMULATION
    device = (
        hoomd.device.CPU()
        if device_name == "cpu"
        else hoomd.device.GPU(gpu_id=gpu_id)
    )
    sim = hoomd.Simulation(device=device, seed=case.seed)
    sim.create_state_from_gsd(filename=str(output_gsd), frame=0)

    center = hoomd.filter.Type([simulation.center_particle_type])
    integrator = hoomd.md.Integrator(dt=simulation.timestep)
    integrator.methods.append(hoomd.md.methods.OverdampedViscous(filter=center))
    sim.operations.integrator = integrator

    cell = hoomd.md.nlist.Cell(buffer=simulation.neighbor_list_buffer)
    lj = hoomd.md.pair.LJ(nlist=cell)
    particle_types = (
        simulation.shell_particle_type,
        simulation.center_particle_type,
    )
    pair_parameters = dict(
        epsilon=(
            simulation.interaction_epsilon_multiplier
            * simulation.gamma
            * simulation.u0
            * analysis.sigma
        ),
        sigma=analysis.sigma,
    )
    for type_pair in itertools.combinations_with_replacement(particle_types, 2):
        lj.params[type_pair] = pair_parameters
        lj.r_cut[type_pair] = analysis.wall_cutoff
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
    for particle_type in particle_types:
        lj_wall.params[particle_type] = {
            "sigma": analysis.sigma,
            "epsilon": (
                simulation.interaction_epsilon_multiplier
                * simulation.gamma
                * simulation.u0
                * analysis.sigma
            ),
            "r_cut": analysis.wall_cutoff,
        }
    integrator.forces.append(lj_wall)

    active = hoomd.md.force.Active(filter=center)
    active.use_orientation = True
    active.active_force[simulation.center_particle_type] = (
        simulation.gamma * simulation.u0,
        0.0,
        0.0,
    )
    active.active_torque[simulation.center_particle_type] = (0.0, 0.0, 0.0)
    integrator.forces.append(active)

    rot_diff = active.create_diffusion_updater(
        trigger=hoomd.trigger.Periodic(simulation.rotational_diffusion_period),
        rotational_diffusion=1 / simulation.tau_r,
    )
    sim.operations += rot_diff

    gsd_writer = hoomd.write.GSD(
        filename=str(output_gsd),
        trigger=hoomd.trigger.Periodic(case.trajectory_write_period),
        mode="ab",
        dynamic=["property", "particles/orientation"],
    )
    sim.operations.writers.append(gsd_writer)

    print(
        f"case={case.case_id} source={case.trajectory_gsd} "
        f"lattice={n_inner} shell={n_shell} total={sim.state.N_particles} "
        f"shell_delta={analysis.frozen_shell_delta:.12g} steps={run_steps} "
        f"device={device_name} output={output_gsd}"
    )
    sim.run(run_steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refill the interior of an unwrapped case with a Cartesian lattice."
    )
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        default=cylinder.SIMULATION.last_frame_run_steps,
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_case(
        get_case(args.case),
        run_steps=args.steps,
        overwrite=args.overwrite,
        device_name=args.device,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
