from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import math
from pathlib import Path

import gsd.hoomd
import hoomd
import numpy as np

from hexatic.constants import cylinder
from hexatic.big_lx.analyze_case import analyze_case
from hexatic.big_lx.cases import CasePaths
from hexatic.big_lx.storage import write_json_atomic

from .cases import UnwrappedCase, get_case

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "last_frame_output"
RUN_STEPS = int(1e8)
TRAJECTORY_WRITE_PERIOD = int(1e5)


@dataclass(frozen=True)
class LastFrameCase:
    case_id: str
    label: str
    radius: float
    circumference: float
    lx: float
    n_particles: int
    run_steps: int
    trajectory_write_period: int
    seed: int

    @property
    def circumference_diameters(self) -> float:
        return self.circumference / cylinder.PARTICLE_DIAMETER

    @property
    def lx_multiplier(self) -> int:
        return 1

    @property
    def wall_radius(self) -> float:
        clearance = (
            cylinder.SIMULATION.wall_clearance_epsilon
            * cylinder.ANALYSIS.particle_diameter
        )
        return (
            self.radius
            + cylinder.ANALYSIS.wall_cutoff
            + clearance
        )

    def as_metadata(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "circumference_diameters": self.circumference_diameters,
            "circumference": self.circumference,
            "radius": self.radius,
            "wall_radius": self.wall_radius,
            "lx_multiplier": self.lx_multiplier,
            "base_lx": self.lx,
            "lx": self.lx,
            "base_n_particles": self.n_particles,
            "n_particles": self.n_particles,
            "particle_diameter": cylinder.PARTICLE_DIAMETER,
            "run_steps": self.run_steps,
            "trajectory_write_period": self.trajectory_write_period,
            "seed": self.seed,
            "variant": (
                "flipped" if self.case_id.endswith("_flipped") else "unflipped"
            ),
        }


def _variant_id(case: UnwrappedCase, flip_shell: bool) -> str:
    suffix = "_last_frame_flipped" if flip_shell else "_last_frame"
    return f"{case.case_id}{suffix}"


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


def _write_refilled_initial_frame(
    case: UnwrappedCase,
    input_gsd: Path,
    output_gsd: Path,
    flip_shell: bool,
) -> tuple[int, int]:
    analysis = cylinder.ANALYSIS
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as trajectory:
        if len(trajectory) == 0:
            raise ValueError(f"Trajectory contains no frames: {input_gsd}")
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
    shell_positions = source_positions[shell_indices]
    if flip_shell:
        shell_positions = -shell_positions
    frame.particles.position = np.vstack((shell_positions, lattice_positions))
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


def _refilled_geometry(
    case: UnwrappedCase,
    input_gsd: Path,
) -> tuple[float, int]:
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as trajectory:
        if len(trajectory) == 0:
            raise ValueError(f"Trajectory contains no frames: {input_gsd}")
        source = trajectory[-1]
    positions = np.asarray(source.particles.position)
    radial_distance = np.linalg.norm(positions[:, 1:3], axis=1)
    n_shell = int(
        np.count_nonzero(
            radial_distance
            > case.radius - cylinder.ANALYSIS.frozen_shell_delta
        )
    )
    box = np.asarray(source.configuration.box, dtype=np.float64)
    n_inner = len(_inner_lattice_positions(box, case.radius))
    return float(box[0]), n_shell + n_inner


def _run_variant(
    source_case: UnwrappedCase,
    analysis_case: LastFrameCase,
    source_gsd: Path,
    paths: CasePaths,
    flip_shell: bool,
    run_steps: int,
    write_period: int,
    device_name: str,
    gpu_id: int | None,
) -> None:
    n_shell, n_inner = _write_refilled_initial_frame(
        source_case,
        source_gsd,
        paths.initial_gsd,
        flip_shell,
    )

    analysis = cylinder.ANALYSIS
    simulation = cylinder.SIMULATION
    device = (
        hoomd.device.CPU()
        if device_name == "cpu"
        else hoomd.device.GPU(gpu_id=gpu_id)
    )
    sim = hoomd.Simulation(device=device, seed=analysis_case.seed)
    sim.create_state_from_gsd(filename=str(paths.initial_gsd), frame=0)

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
            radius=analysis_case.wall_radius,
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

    logger = hoomd.logging.Logger(categories=["particle"])
    logger.add(lj, quantities=["forces", "virials"])
    logger.add(lj_wall, quantities=["forces", "virials"])
    gsd_writer = hoomd.write.GSD(
        filename=str(paths.trajectory_gsd),
        trigger=hoomd.trigger.Periodic(write_period),
        mode="wb",
        dynamic=["property", "particles/orientation"],
        logger=logger,
    )
    sim.operations.writers.append(gsd_writer)

    print(
        f"case={analysis_case.case_id} source={source_gsd} source_frame=last "
        f"lattice={n_inner} shell={n_shell} total={sim.state.N_particles} "
        f"shell_delta={analysis.frozen_shell_delta:.12g} steps={run_steps} "
        f"write_period={write_period} flip_shell={flip_shell} "
        f"device={device_name} output={paths.trajectory_gsd}"
    )
    sim.run(run_steps)
    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as trajectory:
        frame_count = len(trajectory)
        final_step = (
            int(trajectory[-1].configuration.step) if frame_count else None
        )
    write_json_atomic(paths.metadata_json, analysis_case.as_metadata())
    write_json_atomic(
        paths.simulation_complete_json,
        {
            "case_id": analysis_case.case_id,
            "status": "complete",
            "frame_count": frame_count,
            "final_step": final_step,
            "seed": analysis_case.seed,
            "source_gsd": str(source_gsd),
            "source_frame": "last",
            "flip_shell": flip_shell,
            "trajectory_gsd": str(paths.trajectory_gsd),
        },
    )


def run_case(
    case: UnwrappedCase,
    input_gsd: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_steps: int = RUN_STEPS,
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD,
    overwrite: bool = False,
    device_name: str = "gpu",
    gpu_id: int | None = None,
    analysis_backend: str = "auto",
    require_analysis_gpu: bool = False,
    particle_block_size: int = 2048,
    target_shard_mib: int = 256,
) -> None:
    if run_steps < 0:
        raise ValueError("run_steps must be non-negative")
    if trajectory_write_period < 1:
        raise ValueError("trajectory_write_period must be positive")

    source_gsd = case.trajectory_gsd if input_gsd is None else input_gsd
    if not source_gsd.is_file():
        raise FileNotFoundError(f"Missing trajectory GSD: {source_gsd}")

    lx, n_particles = _refilled_geometry(case, source_gsd)
    variants = (
        (False, "unflipped"),
        (True, "flipped"),
    )
    cases_and_paths: list[tuple[bool, LastFrameCase, CasePaths]] = []
    for flip_shell, variant_name in variants:
        analysis_case = LastFrameCase(
            case_id=_variant_id(case, flip_shell),
            label=f"{case.label or case.case_id}, {variant_name} frozen shell",
            radius=case.radius,
            circumference=case.circumference,
            lx=lx,
            n_particles=n_particles,
            run_steps=run_steps,
            trajectory_write_period=trajectory_write_period,
            seed=case.seed,
        )
        paths = CasePaths(analysis_case, output_dir)
        paths.ensure_parent_dirs()
        if source_gsd.resolve() == paths.trajectory_gsd.resolve():
            raise ValueError(
                f"Input and output GSD paths must differ: {paths.trajectory_gsd}"
            )
        if not overwrite:
            existing = [
                path
                for path in (
                    paths.initial_gsd,
                    paths.trajectory_gsd,
                    paths.metadata_json,
                    paths.simulation_complete_json,
                    paths.analysis_dir,
                )
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite existing outputs: "
                    + ", ".join(str(path) for path in existing)
                )
        cases_and_paths.append((flip_shell, analysis_case, paths))

    for flip_shell, analysis_case, paths in cases_and_paths:
        _run_variant(
            case,
            analysis_case,
            source_gsd,
            paths,
            flip_shell,
            run_steps,
            trajectory_write_period,
            device_name,
            gpu_id,
        )

    for _flip_shell, analysis_case, _paths in cases_and_paths:
        print(
            f"[last-frame] analyzing case={analysis_case.case_id} "
            f"backend={analysis_backend}",
            flush=True,
        )
        analyze_case(
            analysis_case,
            output_root=output_dir,
            backend_name=analysis_backend,
            require_gpu=require_analysis_gpu,
            particle_block_size=particle_block_size,
            target_shard_mib=target_shard_mib,
            overwrite=overwrite,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refill the last input frame and sequentially simulate both its "
            "original and antipodally flipped frozen shells."
        )
    )
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--input-gsd",
        type=Path,
        help="Input trajectory; defaults to the selected case trajectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output root containing initial/, gsd/, metadata/, logs/, and "
            "safetensors_output/."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=RUN_STEPS,
    )
    parser.add_argument(
        "--trajectory-write-period",
        type=int,
        default=TRAJECTORY_WRITE_PERIOD,
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument(
        "--analysis-backend",
        choices=("auto", "jax", "numpy"),
        default="auto",
    )
    parser.add_argument("--require-analysis-gpu", action="store_true")
    parser.add_argument("--particle-block-size", type=int, default=2048)
    parser.add_argument("--target-shard-mib", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_case(
        get_case(args.case),
        input_gsd=args.input_gsd,
        output_dir=args.output_dir,
        run_steps=args.steps,
        trajectory_write_period=args.trajectory_write_period,
        overwrite=args.overwrite,
        device_name=args.device,
        gpu_id=args.gpu_id,
        analysis_backend=args.analysis_backend,
        require_analysis_gpu=args.require_analysis_gpu,
        particle_block_size=args.particle_block_size,
        target_shard_mib=args.target_shard_mib,
    )


if __name__ == "__main__":
    main()
