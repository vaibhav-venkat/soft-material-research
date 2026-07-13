from __future__ import annotations

import argparse
from pathlib import Path

import gsd.fl
import matplotlib.pyplot as plt
import numpy as np

from hexatic.constants import cylinder

from .cases import ANALYSIS_DIR, GSD_DIR, UnwrappedCase, get_case


def _trajectory_gsd(case: UnwrappedCase) -> Path:
    return GSD_DIR / f"trajectory_{case.case_id}_last_frame.gsd"


def _default_output(case: UnwrappedCase) -> Path:
    return ANALYSIS_DIR / "output" / f"{case.case_id}_last_frame_inner_com.png"


def _read_inherited_chunk(
    trajectory: gsd.fl.GSDFile,
    frame: int,
    name: str,
) -> np.ndarray:
    for source_frame in range(frame, -1, -1):
        if trajectory.chunk_exists(frame=source_frame, name=name):
            return np.asarray(trajectory.read_chunk(frame=source_frame, name=name))
    raise KeyError(f"Missing required GSD chunk {name!r} at frame {frame}")


def _box_vectors(box: np.ndarray) -> np.ndarray:
    lx, ly, lz, xy, xz, yz = box
    return np.asarray(
        (
            (lx, 0.0, 0.0),
            (xy * ly, ly, 0.0),
            (xz * lz, yz * lz, lz),
        ),
        dtype=np.float64,
    )


def inner_center_of_mass(trajectory_gsd: Path) -> tuple[np.ndarray, np.ndarray]:
    simulation = cylinder.SIMULATION
    with gsd.fl.open(name=str(trajectory_gsd), mode="r") as trajectory:
        if trajectory.nframes == 0:
            raise ValueError(f"Trajectory contains no frames: {trajectory_gsd}")

        initial_velocity = _read_inherited_chunk(
            trajectory,
            0,
            "particles/velocity",
        )
        inner_tags = np.flatnonzero(
            initial_velocity[:, 0] != simulation.shell_velocity_x_marker
        )
        if inner_tags.size == 0:
            raise ValueError("The initial frame contains no marked inner particles")

        steps = np.empty(trajectory.nframes, dtype=np.int64)
        center_of_mass = np.empty((trajectory.nframes, 3), dtype=np.float64)
        for frame_index in range(trajectory.nframes):
            position = _read_inherited_chunk(
                trajectory,
                frame_index,
                "particles/position",
            )[inner_tags]
            box = _read_inherited_chunk(
                trajectory,
                frame_index,
                "configuration/box",
            )
            if trajectory.chunk_exists(frame=frame_index, name="particles/image"):
                image = np.asarray(
                    trajectory.read_chunk(frame=frame_index, name="particles/image")
                )[inner_tags]
            else:
                image = np.zeros_like(position, dtype=np.int32)
            unwrapped_position = position + image @ _box_vectors(box)
            center_of_mass[frame_index] = np.mean(unwrapped_position, axis=0)
            steps[frame_index] = _read_inherited_chunk(
                trajectory,
                frame_index,
                "configuration/step",
            ).reshape(-1)[0]

    elapsed_time = (steps - steps[0]) * simulation.timestep
    return elapsed_time, center_of_mass


def plot_case(
    case: UnwrappedCase,
    trajectory_gsd: Path | None = None,
    output: Path | None = None,
) -> Path:
    trajectory_gsd = trajectory_gsd or _trajectory_gsd(case)
    output = output or _default_output(case)
    elapsed_time, center_of_mass = inner_center_of_mass(trajectory_gsd)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    for component, label in enumerate(("x", "y", "z")):
        axis.plot(elapsed_time, center_of_mass[:, component], marker=".", label=label)
    axis.set_xlabel("elapsed simulation time")
    axis.set_ylabel("inner-particle center of mass")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot inner-particle center of mass from a last-frame simulation."
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = plot_case(
        get_case(args.case),
        trajectory_gsd=args.trajectory,
        output=args.output,
    )
    print(output)


if __name__ == "__main__":
    main()
