from __future__ import annotations

import argparse
import time
from pathlib import Path

import gsd.fl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from hexatic.constants import cylinder

from .cases import ANALYSIS_DIR, GSD_DIR, HEXATIC_OUTPUT_DIR, UnwrappedCase, get_case
from .plot_last_frame_com import _read_inherited_chunk


DEFAULT_LOG_EVERY = 1


def _trajectory_gsd(case: UnwrappedCase) -> Path:
    return GSD_DIR / f"trajectory_{case.case_id}_last_frame.gsd"


def _result_npz(case: UnwrappedCase) -> Path:
    return HEXATIC_OUTPUT_DIR / f"{case.case_id}_last_frame_hexatic_radial.npz"


def _plot_png(case: UnwrappedCase) -> Path:
    return ANALYSIS_DIR / "output" / f"{case.case_id}_last_frame_hexatic_vs_r.png"


def _nearest_neighbors_periodic_x(
    positions: np.ndarray,
    box_length_x: float,
    n_neighbors: int,
) -> np.ndarray:
    n_particles = positions.shape[0]
    if n_particles <= n_neighbors:
        raise ValueError(
            f"Need more than {n_neighbors} particles, found {n_particles}"
        )

    shifted = np.tile(positions, (3, 1))
    shifted[:n_particles, 0] -= box_length_x
    shifted[2 * n_particles :, 0] += box_length_x
    tree = cKDTree(shifted)
    _, augmented_neighbors = tree.query(positions, k=n_neighbors + 1)
    mapped_neighbors = augmented_neighbors % n_particles

    particle_ids = np.arange(n_particles)[:, np.newaxis]
    nonself = mapped_neighbors != particle_ids
    if not np.all(np.count_nonzero(nonself, axis=1) >= n_neighbors):
        raise RuntimeError("Periodic neighbor query returned too few distinct neighbors")
    return mapped_neighbors[nonself].reshape(n_particles, -1)[:, :n_neighbors]


def hexatic_order_frame(
    positions: np.ndarray,
    box_length_x: float,
    n_neighbors: int,
) -> np.ndarray:
    """Compute cylindrical tangent-plane psi6 using all particle types."""
    positions = np.asarray(positions, dtype=np.float64)
    neighbors = _nearest_neighbors_periodic_x(
        positions,
        box_length_x=box_length_x,
        n_neighbors=n_neighbors,
    )

    bonds = positions[neighbors] - positions[:, np.newaxis, :]
    bonds[:, :, 0] -= box_length_x * np.rint(bonds[:, :, 0] / box_length_x)

    yz = positions[:, 1:3]
    radii = np.linalg.norm(yz, axis=1)
    e_theta = np.zeros_like(positions)
    nonzero_radius = radii > np.finfo(np.float64).eps
    e_theta[nonzero_radius, 1] = positions[nonzero_radius, 2] / radii[nonzero_radius]
    e_theta[nonzero_radius, 2] = -positions[nonzero_radius, 1] / radii[nonzero_radius]
    e_theta[~nonzero_radius, 1] = 1.0

    bond_x = bonds[:, :, 0]
    bond_theta = np.einsum("nij,nj->ni", bonds, e_theta)
    bond_angles = np.arctan2(bond_theta, bond_x)
    return np.mean(np.exp(6j * bond_angles), axis=1)


def calculate_radial_hexatic(
    case: UnwrappedCase,
    trajectory_gsd: Path,
    frame_stride: int,
    log_every: int,
) -> dict[str, np.ndarray]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if log_every <= 0:
        raise ValueError("log_every must be positive")

    analysis = cylinder.ANALYSIS
    radial_edges = np.arange(
        0.0,
        case.wall_radius + analysis.particle_diameter,
        analysis.particle_diameter,
    )
    radial_bins = radial_edges.size - 1
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    started = time.monotonic()

    with gsd.fl.open(name=str(trajectory_gsd), mode="r") as trajectory:
        frame_indices = np.arange(0, trajectory.nframes, frame_stride, dtype=np.int64)
        if frame_indices.size == 0:
            raise ValueError(f"Trajectory contains no frames: {trajectory_gsd}")

        first_positions = _read_inherited_chunk(
            trajectory,
            int(frame_indices[0]),
            "particles/position",
        )
        n_particles = first_positions.shape[0]
        steps = np.empty(frame_indices.size, dtype=np.int64)
        radii = np.empty((frame_indices.size, n_particles), dtype=np.float32)
        psi6 = np.empty((frame_indices.size, n_particles), dtype=np.complex64)
        radial_sums = np.zeros((frame_indices.size, radial_bins), dtype=np.float64)
        radial_counts = np.zeros((frame_indices.size, radial_bins), dtype=np.int64)
        radial_volumes = np.empty((frame_indices.size, radial_bins), dtype=np.float64)
        shell_cross_section_areas = np.pi * (
            radial_edges[1:] ** 2 - radial_edges[:-1] ** 2
        )

        print(
            f"source={trajectory_gsd} frames={frame_indices.size} "
            f"particles={n_particles} neighbors={analysis.neighbors} "
            f"radial_bins={radial_bins}",
            flush=True,
        )
        for output_index, frame_index in enumerate(frame_indices):
            positions = _read_inherited_chunk(
                trajectory,
                int(frame_index),
                "particles/position",
            )
            if positions.shape[0] != n_particles:
                raise ValueError(
                    f"Particle count changed at frame {frame_index}: "
                    f"expected {n_particles}, found {positions.shape[0]}"
                )
            box = _read_inherited_chunk(
                trajectory,
                int(frame_index),
                "configuration/box",
            )
            frame_psi6 = hexatic_order_frame(
                positions,
                box_length_x=float(box[0]),
                n_neighbors=analysis.neighbors,
            )
            frame_radii = np.linalg.norm(positions[:, 1:3], axis=1)
            abs_psi6 = np.abs(frame_psi6)

            psi6[output_index] = frame_psi6
            radii[output_index] = frame_radii
            radial_sums[output_index] = np.histogram(
                frame_radii,
                bins=radial_edges,
                weights=abs_psi6,
            )[0]
            radial_counts[output_index] = np.histogram(
                frame_radii,
                bins=radial_edges,
            )[0]
            radial_volumes[output_index] = shell_cross_section_areas * float(box[0])
            steps[output_index] = _read_inherited_chunk(
                trajectory,
                int(frame_index),
                "configuration/step",
            ).reshape(-1)[0]

            completed = output_index + 1
            if completed % log_every == 0 or completed == frame_indices.size:
                elapsed = time.monotonic() - started
                eta = elapsed * (frame_indices.size - completed) / completed
                print(
                    f"frame={completed}/{frame_indices.size} "
                    f"source_index={frame_index} step={steps[output_index]} "
                    f"mean_abs_psi6={np.mean(abs_psi6):.6f} "
                    f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                    flush=True,
                )

    total_sums = np.sum(radial_sums, axis=0)
    total_counts = np.sum(radial_counts, axis=0)
    total_volumes = np.sum(radial_volumes, axis=0)
    radial_mean = np.divide(
        total_sums,
        total_counts,
        out=np.full(radial_bins, np.nan, dtype=np.float64),
        where=total_counts > 0,
    )
    radial_number_density = np.divide(
        total_counts,
        total_volumes,
        out=np.zeros(radial_bins, dtype=np.float64),
        where=total_volumes > 0.0,
    )
    return {
        "frame_indices": frame_indices,
        "steps": steps,
        "radii": radii,
        "psi6": psi6,
        "radial_edges": radial_edges,
        "radial_centers": radial_centers,
        "frame_radial_sums": radial_sums,
        "frame_radial_counts": radial_counts,
        "frame_radial_volumes": radial_volumes,
        "radial_mean_abs_psi6": radial_mean,
        "radial_counts": total_counts,
        "radial_volumes": total_volumes,
        "radial_number_density": radial_number_density,
    }


def plot_radial_hexatic(result: dict[str, np.ndarray], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    populated = result["radial_counts"] > 0
    hexatic_line = axis.plot(
        result["radial_centers"][populated],
        result["radial_mean_abs_psi6"][populated],
        marker=".",
        label="mean |psi6|",
    )[0]
    axis.set_xlabel("r")
    axis.set_ylabel("mean |psi6|")
    axis.set_ylim(0.0, 1.0)
    density_axis = axis.twinx()
    density_line = density_axis.plot(
        result["radial_centers"],
        result["radial_number_density"],
        marker=".",
        color="tab:orange",
        label="number density",
    )[0]
    density_axis.set_ylabel("number density")
    density_axis.set_ylim(bottom=0.0)
    axis.legend(handles=(hexatic_line, density_line))
    axis.set_title("Combined film and center hexatic order")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    return output


def run(
    case: UnwrappedCase,
    trajectory_gsd: Path | None = None,
    result_npz: Path | None = None,
    plot_png: Path | None = None,
    frame_stride: int = 1,
    log_every: int = DEFAULT_LOG_EVERY,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    trajectory_gsd = trajectory_gsd or _trajectory_gsd(case)
    result_npz = result_npz or _result_npz(case)
    plot_png = plot_png or _plot_png(case)
    if not trajectory_gsd.exists():
        raise FileNotFoundError(f"Missing trajectory GSD: {trajectory_gsd}")
    existing = [path for path in (result_npz, plot_png) if path.exists()]
    if existing and not overwrite:
        text = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing file(s):\n{text}")

    result = calculate_radial_hexatic(
        case,
        trajectory_gsd,
        frame_stride=frame_stride,
        log_every=log_every,
    )
    result_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(result_npz, **result)
    print(f"result={result_npz}", flush=True)
    plot_radial_hexatic(result, plot_png)
    print(f"plot={plot_png}", flush=True)
    return result_npz, plot_png


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and plot combined last-frame hexatic order versus r."
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--trajectory", type=Path, default=None)
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run(
        get_case(args.case),
        trajectory_gsd=args.trajectory,
        result_npz=args.result,
        plot_png=args.output,
        frame_stride=args.frame_stride,
        log_every=args.log_every,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
