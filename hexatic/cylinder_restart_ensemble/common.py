from __future__ import annotations

import secrets
import sys
from pathlib import Path
from typing import Callable

import gsd.hoomd
import hoomd
import numpy as np

try:
    from hexatic.constants import cylinder
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from hexatic.constants import cylinder


paths = cylinder.PATHS
analysis = cylinder.ANALYSIS
simulation = cylinder.SIMULATION

SOURCE_GSD = paths.in_gsd
ENSEMBLE_OUTPUT_DIR = paths.in_gsd.parent / "restart_ensemble"
INITIAL_OUTPUT_DIR = ENSEMBLE_OUTPUT_DIR / "initial"
RUN_STEPS = int(5e7)
ORIGINAL_U0 = simulation.u0

FrameTransform = Callable[[gsd.hoomd.Frame, np.random.Generator], None]


def condition_paths(letter: str, replica: int | None = None) -> tuple[Path, Path]:
    suffix = letter if replica is None else f"{letter}_{replica}"
    initial_gsd = INITIAL_OUTPUT_DIR / f"initial_cylinder_{suffix}.gsd"
    trajectory_gsd = ENSEMBLE_OUTPUT_DIR / f"trajectory_cylinder_{suffix}.gsd"
    return initial_gsd, trajectory_gsd


def random_hoomd_seed() -> int:
    return secrets.randbelow(2**31 - 1) + 1


def ensure_paths_available(*target_paths: Path) -> None:
    existing = [path for path in target_paths if path.exists()]
    if existing:
        existing_text = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing file(s):\n{existing_text}")


def copy_particle_property(target, source, name: str) -> None:
    value = getattr(source, name, None)
    if value is None:
        return
    if name == "types":
        setattr(target, name, list(value))
    else:
        setattr(target, name, np.asarray(value).copy())


def restart_frame(
    transform: FrameTransform | None,
    rng: np.random.Generator,
    source_gsd: Path = SOURCE_GSD,
) -> gsd.hoomd.Frame:
    with gsd.hoomd.open(name=str(source_gsd), mode="r") as source:
        if len(source) == 0:
            raise ValueError(f"No frames found in {source_gsd}")
        num_before = -80
        source_frame = source[num_before]
        source_particles = source_frame.particles
        if source_particles.position is None:
            raise ValueError(f"Frame in {source_gsd} has no particle positions")
        if source_particles.orientation is None:
            raise ValueError(f"Frame in {source_gsd} has no particle orientations")

        frame = gsd.hoomd.Frame()
        frame.configuration.step = int(source_frame.configuration.step)
        frame.configuration.box = np.asarray(
            source_frame.configuration.box,
            dtype=np.float64,
        )
        dimensions = getattr(source_frame.configuration, "dimensions", None)
        if dimensions is not None:
            frame.configuration.dimensions = int(dimensions)
        frame.particles.N = int(source_particles.N)

        for property_name in (
            "typeid",
            "types",
            "body",
            "diameter",
            "mass",
            "charge",
            "moment_inertia",
            "position",
            "orientation",
            "velocity",
            "angmom",
            "image",
        ):
            copy_particle_property(frame.particles, source_particles, property_name)

    if transform is not None:
        transform(frame, rng)
    return frame


def write_initial_frame(frame: gsd.hoomd.Frame, initial_gsd: Path) -> None:
    ensure_paths_available(initial_gsd)
    initial_gsd.parent.mkdir(parents=True, exist_ok=True)
    with gsd.hoomd.open(name=str(initial_gsd), mode="w") as target:
        target.append(frame)


def run_hoomd_restart(
    initial_gsd: Path,
    trajectory_gsd: Path,
    active_u0: float,
    seed: int,
    run_steps: int = RUN_STEPS,
) -> None:
    ensure_paths_available(trajectory_gsd)
    trajectory_gsd.parent.mkdir(parents=True, exist_ok=True)

    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=int(seed))
    sim.create_state_from_gsd(filename=str(initial_gsd))

    integrator = hoomd.md.Integrator(dt=simulation.timestep)
    sim.operations.integrator = integrator
    filter_all = hoomd.filter.All()

    ov = hoomd.md.methods.OverdampedViscous(filter=filter_all)
    integrator.methods.append(ov)

    cell = hoomd.md.nlist.Cell(buffer=0.4)

    fixed_epsilon = 50 * simulation.gamma * ORIGINAL_U0 * analysis.sigma

    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[("A", "A")] = dict(
        epsilon=fixed_epsilon,
        sigma=analysis.sigma,
    )
    lj.r_cut[("A", "A")] = analysis.wall_cutoff
    integrator.forces.append(lj)

    walls = [
        hoomd.wall.Cylinder(
            radius=analysis.cylinder_radius,
            axis=(1, 0, 0),
            inside=True,
            open=True,
        )
    ]
    lj_wall = hoomd.md.external.wall.LJ(walls=walls)
    lj_wall.params["A"] = {
        "sigma": analysis.sigma,
        "epsilon": fixed_epsilon,
        "r_cut": analysis.wall_cutoff,
    }
    integrator.forces.append(lj_wall)

    active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
    active.use_orientation = True
    active.active_force["A"] = (simulation.gamma * active_u0, 0.0, 0.0)
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
        filename=str(trajectory_gsd),
        trigger=hoomd.trigger.Periodic(simulation.trajectory_write_period),
        mode="wb",
        dynamic=["property", "particles/orientation"],
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    print(
        "starting restart ensemble run "
        f"seed={seed}, active_u0={active_u0}, steps={run_steps}, "
        f"write_period={simulation.trajectory_write_period}, "
        f"output={trajectory_gsd}"
    )
    sim.run(run_steps)


def run_condition(
    letter: str,
    transform: FrameTransform | None = None,
    active_u0: float = ORIGINAL_U0,
    seed: int = simulation.seed,
    replica: int | None = None,
    run_steps: int = RUN_STEPS,
) -> None:
    initial_gsd, trajectory_gsd = condition_paths(letter, replica=replica)
    ensure_paths_available(initial_gsd, trajectory_gsd)

    rng = np.random.default_rng(int(seed))
    frame = restart_frame(transform=transform, rng=rng)
    write_initial_frame(frame, initial_gsd)
    run_hoomd_restart(
        initial_gsd=initial_gsd,
        trajectory_gsd=trajectory_gsd,
        active_u0=active_u0,
        seed=seed,
        run_steps=run_steps,
    )


def active_directions_from_orientations(orientations: np.ndarray) -> np.ndarray:
    q = np.asarray(orientations, dtype=np.float64).copy()
    q /= np.linalg.norm(q, axis=1)[:, np.newaxis]
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    return np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        )
    )


def quaternions_from_active_directions(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float64).copy()
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

    quaternions = np.zeros((directions.shape[0], 4), dtype=np.float64)
    near_negative_x = directions[:, 0] < -1.0 + 1e-12
    regular = ~near_negative_x

    quaternions[near_negative_x, 2] = 1.0
    quaternions[regular, 0] = 1.0 + directions[regular, 0]
    quaternions[regular, 2] = -directions[regular, 2]
    quaternions[regular, 3] = directions[regular, 1]
    quaternions[regular] /= np.linalg.norm(
        quaternions[regular],
        axis=1,
    )[:, np.newaxis]
    return quaternions


def active_directions_from_frame(frame: gsd.hoomd.Frame) -> np.ndarray:
    return active_directions_from_orientations(frame.particles.orientation)


def set_active_directions(frame: gsd.hoomd.Frame, directions: np.ndarray) -> None:
    frame.particles.orientation = quaternions_from_active_directions(directions)


def random_unit_vectors(n_vectors: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.uniform(-1.0, 1.0, size=n_vectors)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_vectors)
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack(
        (
            z,
            radius * np.cos(phi),
            radius * np.sin(phi),
        )
    )


def randomize_orientations(frame: gsd.hoomd.Frame, rng: np.random.Generator) -> None:
    set_active_directions(frame, random_unit_vectors(frame.particles.N, rng))


def shuffle_orientations(frame: gsd.hoomd.Frame, rng: np.random.Generator) -> None:
    orientations = np.asarray(frame.particles.orientation, dtype=np.float64)
    frame.particles.orientation = orientations[rng.permutation(frame.particles.N)].copy()


def zero_mean_px(frame: gsd.hoomd.Frame, rng: np.random.Generator) -> None:
    del rng
    directions = active_directions_from_frame(frame)
    transverse = directions[:, 1:3]
    transverse_norm = np.linalg.norm(transverse, axis=1)

    lo = np.min(directions[:, 0]) - 4.0
    hi = np.max(directions[:, 0]) + 4.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        shifted_x = directions[:, 0] - mid
        normalized_x = shifted_x / np.sqrt(shifted_x * shifted_x + transverse_norm**2)
        if np.mean(normalized_x) > 0.0:
            lo = mid
        else:
            hi = mid

    shifted_x = directions[:, 0] - 0.5 * (lo + hi)
    adjusted = np.column_stack((shifted_x, transverse))
    adjusted /= np.linalg.norm(adjusted, axis=1)[:, np.newaxis]
    set_active_directions(frame, adjusted)


def consistent_axial_mirror(frame: gsd.hoomd.Frame, rng: np.random.Generator) -> None:
    del rng
    positions = np.asarray(frame.particles.position, dtype=np.float64).copy()
    positions[:, 0] *= -1.0
    frame.particles.position = positions

    directions = active_directions_from_frame(frame)
    directions[:, 0] *= -1.0
    set_active_directions(frame, directions)


def orientation_axial_flip(frame: gsd.hoomd.Frame, rng: np.random.Generator) -> None:
    del rng
    directions = active_directions_from_frame(frame)
    directions[:, 0] *= -1.0
    set_active_directions(frame, directions)


def theta_mirror(frame: gsd.hoomd.Frame, rng: np.random.Generator) -> None:
    del rng
    positions = np.asarray(frame.particles.position, dtype=np.float64).copy()
    positions[:, 2] *= -1.0
    frame.particles.position = positions

    directions = active_directions_from_frame(frame)
    directions[:, 2] *= -1.0
    set_active_directions(frame, directions)
