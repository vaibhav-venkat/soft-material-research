from __future__ import annotations

import argparse
from pathlib import Path

import gsd.hoomd
import hoomd
import numpy as np

from hexatic.constants import cylinder

from .cases import GSD_DIR, UnwrappedCase, ensure_output_dirs, get_case


RUN_STEPS = int(1e7)


def _output_gsd(case: UnwrappedCase) -> Path:
    return GSD_DIR / f"trajectory_{case.case_id}_last_frame.gsd"


def _ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def _inner_particle_tags(case: UnwrappedCase) -> np.ndarray:
    """Select movable tags once from the source trajectory's final frame."""
    with gsd.hoomd.open(name=str(case.trajectory_gsd), mode="r") as trajectory:
        if len(trajectory) == 0:
            raise ValueError(f"Trajectory contains no frames: {case.trajectory_gsd}")
        positions = np.asarray(trajectory[-1].particles.position)

    radial_distance = np.linalg.norm(positions[:, 1:3], axis=1)
    inner_radius = case.radius - cylinder.ANALYSIS.shell_delta
    return np.flatnonzero(radial_distance <= inner_radius).astype(np.uint32)


def run_case(
    case: UnwrappedCase,
    overwrite: bool = False,
    gpu_id: int | None = None,
) -> None:
    ensure_output_dirs()
    if not case.trajectory_gsd.exists():
        raise FileNotFoundError(f"Missing trajectory GSD: {case.trajectory_gsd}")

    output_gsd = _output_gsd(case)
    _ensure_can_write(output_gsd, overwrite)
    inner_tags = _inner_particle_tags(case)
    if inner_tags.size == 0:
        raise ValueError(
            f"No particles lie inside the shell cutoff for case {case.case_id}"
        )

    analysis = cylinder.ANALYSIS
    simulation = cylinder.SIMULATION
    device = hoomd.device.GPU(gpu_id=gpu_id)
    sim = hoomd.Simulation(device=device, seed=case.seed)
    sim.create_state_from_gsd(filename=str(case.trajectory_gsd), frame=-1)

    shell_tags = np.setdiff1d(
        np.arange(sim.state.N_particles, dtype=np.uint32),
        inner_tags,
        assume_unique=True,
    )
    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        snapshot.particles.velocity[shell_tags, 0] = 1.0
    sim.state.set_snapshot(snapshot)

    # Tags is intentionally static: particles remain integrated even if they later
    # enter the shell region, while particles initially in the shell stay frozen.
    inner = hoomd.filter.Tags(inner_tags)
    integrator = hoomd.md.Integrator(dt=simulation.timestep)
    integrator.methods.append(hoomd.md.methods.OverdampedViscous(filter=inner))
    sim.operations.integrator = integrator

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

    active = hoomd.md.force.Active(filter=inner)
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
        filename=str(output_gsd),
        trigger=hoomd.trigger.Periodic(case.trajectory_write_period),
        mode="wb",
        dynamic=["property", "particles/orientation"],
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    n_particles = sim.state.N_particles
    print(
        f"case={case.case_id} source={case.trajectory_gsd} "
        f"inner={inner_tags.size} shell={shell_tags.size} "
        f"shell_delta={analysis.shell_delta:.12g} steps={RUN_STEPS} "
        f"write_period={case.trajectory_write_period} output={output_gsd}"
    )
    sim.run(RUN_STEPS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue an unwrapped case with its outer shell frozen."
    )
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
