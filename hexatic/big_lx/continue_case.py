"""Continue the C=60.5D, Lx=8x case from the last frame of a GSD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gsd.hoomd
import hoomd

from hexatic.constants import cylinder

from .analyze_case import analyze_case
from .cases import CasePaths, get_case

CASE_ID = "circ_60_5D_lx_8x"
EXTRA_STEPS = 1_000_000_000
TRAJECTORY_WRITE_PERIOD = 1_000_000
EXPECTED_FRAME_COUNT = EXTRA_STEPS // TRAJECTORY_WRITE_PERIOD
ROTATIONAL_DIFFUSION_PERIOD = 10
TAU_R = 1


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _prepare_outputs(paths: CasePaths) -> None:
    paths.ensure_parent_dirs()
    outputs = (
        paths.trajectory_gsd,
        paths.metadata_json,
        paths.simulation_complete_json,
        paths.analysis_dir,
    )
    existing = tuple(path for path in outputs if path.exists())
    if existing:
        listed = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite continuation outputs. "
            "Choose an empty output directory:\n"
            f"{listed}"
        )


def continue_case(output_dir: Path, input_gsd: Path) -> None:
    """Continue the fixed big-Lx case and analyze the resulting trajectory."""
    output_dir = output_dir.resolve()
    input_gsd = input_gsd.resolve()
    if not input_gsd.is_file():
        raise FileNotFoundError(f"Missing input GSD: {input_gsd}")

    case = get_case(CASE_ID)
    paths = CasePaths(case, output_dir)
    _prepare_outputs(paths)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        if not len(source):
            raise ValueError(f"Input GSD contains no frames: {input_gsd}")
        last_frame = source[-1]
        input_frame_count = len(source)
        input_last_step = int(last_frame.configuration.step)
        input_particle_count = int(last_frame.particles.N)
        input_lx = float(last_frame.configuration.box[0])

    if input_particle_count != case.n_particles:
        raise ValueError(
            f"Expected {case.n_particles} particles for {case.label}, "
            f"found {input_particle_count}"
        )
    if abs(input_lx - case.lx) > 1.0e-5 * case.lx:
        raise ValueError(
            f"Expected Lx={case.lx:.12g} for {case.label}, found {input_lx:.12g}"
        )

    simulation = cylinder.SIMULATION
    if simulation.rotational_diffusion_period != ROTATIONAL_DIFFUSION_PERIOD:
        raise ValueError(
            "Cylinder rotational diffusion period changed: "
            f"expected {ROTATIONAL_DIFFUSION_PERIOD}, "
            f"found {simulation.rotational_diffusion_period}"
        )
    if simulation.tau_r != TAU_R:
        raise ValueError(
            f"Cylinder tau_r changed: expected {TAU_R}, found {simulation.tau_r}"
        )

    metadata = case.as_metadata()
    metadata.update(
        status="running",
        continuation=True,
        input_gsd=str(input_gsd),
        input_frame_count=input_frame_count,
        input_last_step=input_last_step,
        run_steps=EXTRA_STEPS,
        extra_steps=EXTRA_STEPS,
        trajectory_write_period=TRAJECTORY_WRITE_PERIOD,
        expected_frame_count=EXPECTED_FRAME_COUNT,
        rotational_diffusion_period=ROTATIONAL_DIFFUSION_PERIOD,
        tau_r=TAU_R,
        trajectory_gsd=str(paths.trajectory_gsd),
        device="gpu",
    )
    _write_json_atomic(paths.metadata_json, metadata)

    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=case.seed)
    sim.create_state_from_gsd(filename=str(input_gsd), frame=-1)

    integrator = hoomd.md.Integrator(dt=simulation.timestep)
    integrator.methods.append(
        hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All())
    )
    sim.operations.integrator = integrator

    neighbor_list = hoomd.md.nlist.Cell(buffer=simulation.neighbor_list_buffer)
    pair = hoomd.md.pair.LJ(nlist=neighbor_list)
    epsilon = (
        simulation.interaction_epsilon_multiplier
        * simulation.gamma
        * simulation.u0
        * cylinder.ANALYSIS.sigma
    )
    pair.params[("A", "A")] = {
        "epsilon": epsilon,
        "sigma": cylinder.ANALYSIS.sigma,
    }
    pair.r_cut[("A", "A")] = cylinder.ANALYSIS.wall_cutoff
    integrator.forces.append(pair)

    wall = hoomd.wall.Cylinder(
        radius=case.wall_radius,
        axis=(1, 0, 0),
        inside=True,
        open=True,
    )
    wall_pair = hoomd.md.external.wall.LJ(walls=[wall])
    wall_pair.params["A"] = {
        "sigma": cylinder.ANALYSIS.sigma,
        "epsilon": epsilon,
        "r_cut": cylinder.ANALYSIS.wall_cutoff,
    }
    integrator.forces.append(wall_pair)

    active = hoomd.md.force.Active(filter=hoomd.filter.Type(["A"]))
    active.use_orientation = True
    active.active_force["A"] = (simulation.gamma * simulation.u0, 0.0, 0.0)
    active.active_torque["A"] = (0.0, 0.0, 0.0)
    integrator.forces.append(active)
    sim.operations += active.create_diffusion_updater(
        trigger=hoomd.trigger.Periodic(ROTATIONAL_DIFFUSION_PERIOD),
        rotational_diffusion=1 / TAU_R,
    )

    logger = hoomd.logging.Logger(categories=["particle"])
    logger.add(pair, quantities=["forces", "virials"])
    logger.add(wall_pair, quantities=["forces", "virials"])
    sim.operations.writers.append(
        hoomd.write.GSD(
            filename=str(paths.trajectory_gsd),
            trigger=hoomd.trigger.Periodic(TRAJECTORY_WRITE_PERIOD),
            mode="wb",
            dynamic=["property", "particles/orientation"],
            logger=logger,
        )
    )

    print(
        f"[big_lx.continue] case={case.case_id} input_step={input_last_step} "
        f"extra_steps={EXTRA_STEPS} write_period={TRAJECTORY_WRITE_PERIOD} "
        f"tau_r={TAU_R} diffusion_period={ROTATIONAL_DIFFUSION_PERIOD}",
        flush=True,
    )
    sim.run(EXTRA_STEPS)

    with gsd.hoomd.open(name=str(paths.trajectory_gsd), mode="r") as trajectory:
        frame_count = len(trajectory)
        final_step = (
            int(trajectory[-1].configuration.step) if frame_count else input_last_step
        )
    if frame_count != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAME_COUNT} trajectory frames, found {frame_count}"
        )

    metadata.update(
        status="simulation_complete",
        frame_count=frame_count,
        final_step=final_step,
    )
    _write_json_atomic(paths.metadata_json, metadata)
    _write_json_atomic(
        paths.simulation_complete_json,
        {
            "case_id": case.case_id,
            "status": "complete",
            "continuation": True,
            "input_gsd": str(input_gsd),
            "input_last_step": input_last_step,
            "extra_steps": EXTRA_STEPS,
            "frame_count": frame_count,
            "final_step": final_step,
            "seed": case.seed,
            "trajectory_gsd": str(paths.trajectory_gsd),
        },
    )

    print("[big_lx.continue] simulation complete; starting analysis", flush=True)
    analyze_case(case, output_root=output_dir)
    metadata["status"] = "complete"
    _write_json_atomic(paths.metadata_json, metadata)
    print("[big_lx.continue] analysis complete", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the C=60.5D, Lx=8x case for 10^9 steps from the last "
            "frame of an input GSD, then run the standard big-Lx analysis."
        )
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("input_gsd", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    continue_case(args.output_dir, args.input_gsd)


if __name__ == "__main__":
    main()
