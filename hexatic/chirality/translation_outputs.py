from copy import deepcopy
from pathlib import Path

import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np

try:
    from hexatic.constants import cylinder
except ImportError:
    from constants import cylinder

from .translation import compute_translation_chirality_frame


def _minimum_image_x(
    vectors: np.ndarray,
    box_length_x: float | None = None,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    if box_length_x is None:
        return vectors
    wrapped = vectors.copy()
    wrapped[..., 0] -= box_length_x * np.round(wrapped[..., 0] / box_length_x)
    return wrapped


def _shell_mask(
    positions: np.ndarray,
    cylinder_radius: float,
    shell_delta: float,
) -> np.ndarray:
    radii = np.linalg.norm(np.asarray(positions, dtype=np.float64)[:, 1:3], axis=1)
    return radii > cylinder_radius - shell_delta


def _shell_bond_translation_chirality_frame(
    positions: np.ndarray,
    neighborhood_radius: float,
    cylinder_radius: float,
    shell_delta: float,
    box_length_x: float | None = None,
) -> tuple[float, int]:
    positions = np.asarray(positions, dtype=np.float64)
    assert positions.ndim == 2 and positions.shape[1] == 3
    assert neighborhood_radius > 0.0

    shell = _shell_mask(positions, cylinder_radius, shell_delta)
    shell_positions = positions[shell]
    if len(shell_positions) < 2:
        return np.nan, 0

    bonds = shell_positions[np.newaxis, :, :] - shell_positions[:, np.newaxis, :]
    bonds = _minimum_image_x(bonds, box_length_x=box_length_x)
    bond_lengths = np.linalg.norm(bonds, axis=2)
    valid_bonds = (
        np.isfinite(bond_lengths)
        & (bond_lengths > 0.0)
        & (bond_lengths <= neighborhood_radius)
    )
    values = np.divide(
        bonds[:, :, 0],
        bond_lengths,
        out=np.zeros_like(bond_lengths, dtype=np.float64),
        where=valid_bonds,
    )
    bond_values = values[valid_bonds]
    if bond_values.size == 0:
        return np.nan, 0
    return float(np.mean(np.abs(bond_values))), int(bond_values.size)


def shell_bond_translation_chirality_series(
    input_gsd: str | Path,
    neighborhood_radius: float,
    cylinder_radius: float = cylinder.CYLINDER_RADIUS,
    shell_delta: float = cylinder.SHELL_DELTA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: list[int] = []
    mean_abs_values: list[float] = []
    counts: list[int] = []
    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        for frame in source:
            positions = np.asarray(frame.particles.position, dtype=np.float64)
            box_length_x = float(frame.configuration.box[0])
            mean_abs_value, count = _shell_bond_translation_chirality_frame(
                positions,
                neighborhood_radius=neighborhood_radius,
                cylinder_radius=cylinder_radius,
                shell_delta=shell_delta,
                box_length_x=box_length_x,
            )
            steps.append(int(frame.configuration.step))
            mean_abs_values.append(mean_abs_value)
            counts.append(count)
    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(mean_abs_values, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def plot_shell_bond_translation_chirality(
    input_gsd: str | Path,
    output_png: str | Path,
    neighborhood_radius: float,
    cylinder_radius: float = cylinder.CYLINDER_RADIUS,
    shell_delta: float = cylinder.SHELL_DELTA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, mean_abs_values, counts = shell_bond_translation_chirality_series(
        input_gsd,
        neighborhood_radius=neighborhood_radius,
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
    )
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    axis.plot(steps, mean_abs_values, lw=1.8, label="mean abs shell bond chirality")
    axis.set_xlabel("step")
    axis.set_ylabel("mean abs bond translation chirality")
    axis.set_title("Shell translation chirality")
    axis.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return steps, mean_abs_values, counts


def write_translation_chirality_measure_gsd(
    input_gsd: str | Path,
    output_gsd: str | Path,
    neighborhood_radius: float,
) -> None:
    output_path = Path(output_gsd)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        with gsd.hoomd.open(name=str(output_path), mode="w") as destination:
            for frame in source:
                positions = np.asarray(frame.particles.position, dtype=np.float64)
                n_particles = int(frame.particles.N)
                assert positions.shape == (n_particles, 3)
                chirality = compute_translation_chirality_frame(
                    positions,
                    neighborhood_radius=neighborhood_radius,
                    box_length_x=float(frame.configuration.box[0]),
                )
                new_frame = deepcopy(frame)
                velocity = np.zeros((n_particles, 3), dtype=np.float32)
                velocity[:, 0] = chirality.astype(np.float32)
                new_frame.particles.velocity = velocity
                destination.append(new_frame)


def write_translation_chirality_measure_outputs(
    input_gsd: str | Path,
    plot_png: str | Path,
    measure_gsd: str | Path,
    neighborhood_radius: float,
    series_npz: str | Path | None = None,
    cylinder_radius: float = cylinder.CYLINDER_RADIUS,
    shell_delta: float = cylinder.SHELL_DELTA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, mean_abs_values, counts = plot_shell_bond_translation_chirality(
        input_gsd,
        plot_png,
        neighborhood_radius=neighborhood_radius,
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
    )
    if series_npz is not None:
        series_path = Path(series_npz)
        series_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            series_path,
            steps=steps,
            mean_abs_bond_translation_chirality=mean_abs_values,
            bond_counts=counts,
            neighborhood_radius=neighborhood_radius,
        )
    write_translation_chirality_measure_gsd(
        input_gsd,
        measure_gsd,
        neighborhood_radius=neighborhood_radius,
    )
    return steps, mean_abs_values, counts
