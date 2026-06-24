from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gsd.hoomd
import hoomd
import numpy as np

from hexatic.constants import cylinder

from .cases import RadiusCase, ensure_output_dirs, get_case


def _ensure_can_write(paths: tuple[Path, ...], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        text = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing file(s):\n{text}")


def cylinder_cross_section_points(radius: float, min_spacing: float) -> np.ndarray:
    analysis = cylinder.ANALYSIS
    simulation = cylinder.SIMULATION
    center_limit = max(
        0.0,
        radius
        - analysis.wall_cutoff
        - simulation.wall_clearance_epsilon * analysis.particle_diameter,
    )
    if center_limit == 0.0:
        return np.asarray([[0.0, 0.0]], dtype=np.float64)

    n_radial = max(1, int(math.ceil(center_limit / min_spacing)) + 1)
    candidates: list[tuple[float, float]] = []
    for ring_radius in np.linspace(center_limit, 0.0, n_radial):
        if ring_radius == 0.0:
            candidates.append((0.0, 0.0))
            continue
        n_angles = max(
            simulation.min_angular_candidates,
            int(math.ceil(2.0 * math.pi * ring_radius / min_spacing))
            * simulation.angular_candidate_multiplier,
        )
        for angle in np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False):
            candidates.append(
                (
                    ring_radius * math.cos(angle),
                    ring_radius * math.sin(angle),
                )
            )

    selected: list[tuple[float, float]] = []
    min_distance_squared = min_spacing**2
    for candidate in candidates:
        if all(
            (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2
            >= min_distance_squared
            for point in selected
        ):
            selected.append(candidate)

    if not selected:
        selected.append((0.0, 0.0))
    return np.asarray(selected, dtype=np.float64)


def generate_cylinder_lattice(
    n_particles: int,
    box_length_x: float,
    radius: float,
    spacing: float,
) -> np.ndarray:
    cross_section = cylinder_cross_section_points(radius, spacing)
    n_x = int(math.ceil(n_particles / len(cross_section)))
    dx = box_length_x / n_x
    x_values = -0.5 * box_length_x + (np.arange(n_x, dtype=np.float64) + 0.5) * dx
    positions = np.empty((n_x * len(cross_section), 3), dtype=np.float64)

    particle_idx = 0
    for x in x_values:
        for y, z in cross_section:
            positions[particle_idx] = (x, y, z)
            particle_idx += 1

    return positions[:n_particles]


def random_uniform_quaternions(n: int, rng=np.random) -> np.ndarray:
    u1 = rng.rand(n)
    u2 = rng.rand(n)
    u3 = rng.rand(n)

    qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    q = np.column_stack([qw, qx, qy, qz])
    q /= np.linalg.norm(q, axis=1)[:, None]
    return q


def _write_initial_state(case: RadiusCase) -> None:
    analysis = cylinder.ANALYSIS
    simulation = cylinder.SIMULATION

    np.random.seed(case.seed)
    position = generate_cylinder_lattice(
        case.n_particles,
        case.lx,
        case.radius,
        analysis.wall_cutoff,
    )
    n_particles = position.shape[0]

    transverse_margin = (
        simulation.transverse_wall_cutoff_margin * analysis.wall_cutoff
        + simulation.transverse_particle_diameter_margin * analysis.particle_diameter
    )
    transverse_length = 2.0 * (case.radius + transverse_margin)

    frame = gsd.hoomd.Frame()
    frame.particles.N = n_particles
    frame.particles.position = position
    frame.particles.diameter = [analysis.particle_diameter] * n_particles
    frame.particles.moment_inertia = [(1, 1, 1)] * n_particles
    frame.particles.orientation = random_uniform_quaternions(n_particles)
    frame.particles.image = np.zeros((n_particles, 3), dtype=np.int32)
    frame.particles.typeid = [0] * n_particles
    frame.configuration.box = [case.lx, transverse_length, transverse_length, 0, 0, 0]
    frame.configuration.dimensions = 3
    frame.particles.types = ["A"]

    case.initial_gsd.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=str(case.initial_gsd), mode="w") as target:
        target.append(frame)


def run_case(
    case: RadiusCase,
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
            radius=case.radius,
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
        f"R={case.radius:.12g} Lx={case.lx:.12g} N={case.n_particles} "
        f"steps={case.run_steps} write_period={case.trajectory_write_period} "
        f"output={case.trajectory_gsd}"
    )
    sim.run(case.run_steps)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--include-long-axis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_case(
        get_case(args.case, include_long_axis=args.include_long_axis),
        overwrite=args.overwrite,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()

