from __future__ import annotations

import argparse
from pathlib import Path

import gsd.fl
import matplotlib.pyplot as plt
import numpy as np

from hexatic.constants import cylinder

from .cases import ANALYSIS_DIR, UnwrappedCase, get_case
from .plot_last_frame_com import _box_vectors, _particle_images, _read_inherited_chunk


def _default_output(trajectory_gsd: Path) -> Path:
    return ANALYSIS_DIR / "output" / f"{trajectory_gsd.stem}_com.png"


def center_of_mass(trajectory_gsd: Path) -> tuple[np.ndarray, np.ndarray]:
    simulation = cylinder.SIMULATION
    with gsd.fl.open(name=str(trajectory_gsd), mode="r") as trajectory:
        if trajectory.nframes == 0:
            raise ValueError(f"Trajectory contains no frames: {trajectory_gsd}")

        steps = np.empty(trajectory.nframes, dtype=np.int64)
        center = np.empty((trajectory.nframes, 3), dtype=np.float64)
        for frame_index in range(trajectory.nframes):
            position = _read_inherited_chunk(
                trajectory,
                frame_index,
                "particles/position",
            )
            box = _read_inherited_chunk(
                trajectory,
                frame_index,
                "configuration/box",
            )
            image = _particle_images(trajectory, frame_index, position.shape)
            unwrapped_position = position + image @ _box_vectors(box)
            theta = np.arctan2(position[:, 1], position[:, 2])
            center[frame_index, 0] = np.mean(unwrapped_position[:, 0])
            center[frame_index, 1] = np.mean(
                np.linalg.norm(position[:, 1:3], axis=1)
            )
            center[frame_index, 2] = np.angle(np.mean(np.exp(1j * theta)))
            steps[frame_index] = _read_inherited_chunk(
                trajectory,
                frame_index,
                "configuration/step",
            ).reshape(-1)[0]

    elapsed_time = (steps - steps[0]) * simulation.timestep
    return elapsed_time, center


def plot_trajectory(trajectory_gsd: Path, output: Path | None = None) -> Path:
    elapsed_time, center = center_of_mass(trajectory_gsd)
    output = output or _default_output(trajectory_gsd)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = ("x COM", "mean r", "circular theta COM (rad)")
    figure, axes = plt.subplots(3, 1, sharex=True)
    for component, (axis, label) in enumerate(zip(axes, labels)):
        axis.plot(elapsed_time, center[:, component], marker=".")
        axis.set_ylabel(label)
    axes[-1].set_xlabel("elapsed simulation time")
    figure.suptitle("Particle center of mass")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def plot_case(case: UnwrappedCase, output: Path | None = None) -> Path:
    return plot_trajectory(case.trajectory_gsd, output=output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot center of mass from a regular unwrapped trajectory."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case")
    source.add_argument("--trajectory", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    trajectory_gsd = (
        get_case(args.case).trajectory_gsd
        if args.case is not None
        else args.trajectory
    )
    print(plot_trajectory(trajectory_gsd, output=args.output))


if __name__ == "__main__":
    main()
