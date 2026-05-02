from __future__ import annotations

import copy
from pathlib import Path
import gsd.hoomd
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
IntArray = npt.NDArray[np.int64]


def _validate_positions(positions: np.ndarray) -> FloatArray:
    arr = np.asarray(positions, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 3
    return arr


def _validate_center(center: np.ndarray | None) -> FloatArray:
    if center is None:
        return np.zeros(3, dtype=np.float64)

    arr = np.asarray(center, dtype=np.float64)
    assert arr.shape == (3,)
    return arr


def sphere_normals(
    positions: np.ndarray,
    center: np.ndarray | None = None,
) -> FloatArray:
    """Return a unit normal for each particle"""

    positions = _validate_positions(positions)
    center_arr = _validate_center(center)
    radial_vectors = positions - center_arr
    radii = np.linalg.norm(radial_vectors, axis=1)

    assert not np.any(radii <= 0.0)

    return radial_vectors / radii[:, np.newaxis]


def local_tangent_basis(normal: np.ndarray) -> tuple[FloatArray, FloatArray]:
    """Build a orthonormal tangent basis for the normal"""

    normal_arr = np.asarray(normal, dtype=np.float64)
    assert normal_arr.shape == (3,)

    normal_norm = np.linalg.norm(normal_arr)
    assert normal_norm >= 0.0

    unit_normal = normal_arr / normal_norm
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(unit_normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(reference, unit_normal)
    e1_norm = np.linalg.norm(e1)
    assert e1_norm >= 0.0
    e1 = e1 / e1_norm

    e2 = np.cross(unit_normal, e1)
    e2_norm = np.linalg.norm(e2)
    assert e2_norm >= 0.0
    e2 = e2 / e2_norm

    return e1, e2


def nearest_neighbors(
    positions: np.ndarray,
    n_neighbors: int = 6,
) -> IntArray:
    """Return indices of the n closest particles"""

    positions = _validate_positions(positions)

    n_particles = positions.shape[0]
    assert n_particles >= n_neighbors

    position_norms = np.sum(positions * positions, axis=1)
    distances_sq = (
        position_norms[:, np.newaxis]
        + position_norms[np.newaxis, :]
        - 2.0 * positions @ positions.T
    )
    np.maximum(distances_sq, 0.0, out=distances_sq)
    np.fill_diagonal(distances_sq, np.inf)

    neighbor_indices = np.argpartition(
        distances_sq, kth=n_neighbors - 1, axis=1
    )[:, :n_neighbors]
    neighbor_distances = np.take_along_axis(
        distances_sq, neighbor_indices, axis=1
    )
    distance_order = np.argsort(neighbor_distances, axis=1)
    neighbor_indices = np.take_along_axis(neighbor_indices, distance_order, axis=1)

    return neighbor_indices.astype(np.int64, copy=False)


def compute_hexatic_order_frame(
    positions: np.ndarray,
    n_neighbors: int = 6,
    center: np.ndarray | None = None,
) -> tuple[ComplexArray, IntArray]:
    """Compute hexatic order for every particle in a frame"""

    positions = _validate_positions(positions)
    neighbors = nearest_neighbors(positions, n_neighbors)
    normals = sphere_normals(positions, center)
    psi = np.empty(positions.shape[0], dtype=np.complex128)

    for particle_idx, neighbor_ids in enumerate(neighbors):
        normal = normals[particle_idx]
        e1, e2 = local_tangent_basis(normal)

        bonds = positions[neighbor_ids] - positions[particle_idx]
        normal_components = bonds @ normal
        # Tangent-plane projection of the 3D bond vector.
        tangent_bonds = bonds - normal_components[:, np.newaxis] * normal

        x_coords = tangent_bonds @ e1
        y_coords = tangent_bonds @ e2
        theta = np.arctan2(y_coords, x_coords)
        psi[particle_idx] = np.mean(np.exp(6j * theta))

    return psi, neighbors


def compute_hexatic_order_trajectory(
    filename: str | Path,
    n_neighbors: int = 6,
    center: np.ndarray | None = None,
) -> tuple[IntArray, ComplexArray]:
    """Compute hexatic order for every frame"""

    steps: list[int] = []
    psi_frames: list[ComplexArray] = []
    n_particles: int | None = None

    with gsd.hoomd.open(name=str(filename), mode="r") as trajectory:
        for frame_idx, frame in enumerate(trajectory):
            particles = frame.particles
            assert particles.position is not None

            positions = _validate_positions(particles.position)
            if n_particles is None:
                n_particles = positions.shape[0]
            assert positions.shape[0] == n_particles

            psi, _ = compute_hexatic_order_frame(
                positions,
                n_neighbors=n_neighbors,
                center=center,
            )
            psi_frames.append(psi)
            steps.append(int(frame.configuration.step))

    assert psi_frames is not None
    return np.asarray(steps, dtype=np.int64), np.vstack(psi_frames)


def save_hexatic_text(
    filename: str | Path,
    steps: np.ndarray,
    psi: np.ndarray,
) -> None:
    """Save hexatic order as a text table"""

    steps = np.asarray(steps, dtype=np.int64)
    psi = np.asarray(psi, dtype=np.complex128)
    assert psi.ndim == 2 and steps.shape == (psi.shape[0],)

    n_frames, n_particles = psi.shape
    frame_col = np.repeat(np.arange(n_frames, dtype=np.int64), n_particles)
    step_col = np.repeat(steps, n_particles)
    particle_col = np.tile(np.arange(n_particles, dtype=np.int64), n_frames)
    psi_flat = psi.reshape(-1)
    data = np.column_stack(
        (
            frame_col,
            step_col,
            particle_col,
            psi_flat.real,
            psi_flat.imag,
            np.abs(psi_flat),
        )
    )

    np.savetxt(
        filename,
        data,
        fmt=("%d", "%d", "%d", "%.18e", "%.18e", "%.18e"),
        header="frame step particle psi_real psi_imag psi_abs",
    )


def load_hexatic_text(filename: str | Path) -> FloatArray:
    """Load a text table"""

    data = np.loadtxt(filename, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    assert data.ndim == 2 and data.shape[1] >= 6
    return data


def hexatic_abs_matrix_from_table(
    table: np.ndarray,
    n_frames: int | None = None,
    n_particles: int | None = None,
) -> FloatArray:
    """Convert hexatic table to a matrix"""

    table = np.asarray(table, dtype=np.float64)
    assert table.ndim == 2 and table.shape[1] >= 6

    frame_indices = table[:, 0].astype(np.int64)
    particle_indices = table[:, 2].astype(np.int64)
    psi_abs = table[:, 5]

    assert not np.any(frame_indices < 0) and not np.any(particle_indices < 0)

    inferred_frames = int(frame_indices.max()) + 1
    inferred_particles = int(particle_indices.max()) + 1
    if n_frames is None:
        n_frames = inferred_frames
    if n_particles is None:
        n_particles = inferred_particles

    assert inferred_frames <= n_frames and inferred_particles <= n_particles

    flat_indices = frame_indices * n_particles + particle_indices
    assert np.unique(flat_indices).size == flat_indices.size

    matrix = np.full((n_frames, n_particles), np.nan, dtype=np.float64)
    matrix[frame_indices, particle_indices] = psi_abs
    assert not np.any(np.isnan(matrix))

    return matrix


def hexatic_probability_distribution(
    hexatic_abs: np.ndarray,
    frame_indices: np.ndarray,
    min_frame: int = 10,
    bins: int = 50,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Return the probability density of psi for frames after cutoff"""

    values = np.asarray(hexatic_abs, dtype=np.float64).reshape(-1)
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    assert values.shape == frames.shape

    selected = values[frames > min_frame]
    assert selected.size != 0

    selected = np.clip(selected, 0.0, 1.0)
    density, edges = np.histogram(
        selected,
        bins=bins,
        range=(0.0, 1.0),
        density=True,
    )
    counts, _ = np.histogram(selected, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return (
        centers.astype(np.float64, copy=False),
        density.astype(np.float64, copy=False),
        counts.astype(np.int64, copy=False),
    )


def save_distribution_text(
    filename: str | Path,
    bin_centers: np.ndarray,
    probability_density: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Save the probability distribution table"""

    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    probability_density = np.asarray(probability_density, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    assert (
        bin_centers.shape == probability_density.shape == counts.shape
        and bin_centers.ndim == 1
    )

    data = np.column_stack((bin_centers, probability_density, counts))
    np.savetxt(
        filename,
        data,
        fmt=("%.18e", "%.18e", "%d"),
        header="bin_center probability_density count",
    )


def write_hexatic_velocity_gsd(
    input_gsd: str | Path,
    output_gsd: str | Path,
    hexatic_txt: str | Path,
    component: int = 0,
) -> None:
    """Write .gsd file with psi as the velocity"""

    assert component in (0, 1, 2)

    input_path = Path(input_gsd)
    output_path = Path(output_gsd)
    assert input_path.resolve() != output_path.resolve()

    table = load_hexatic_text(hexatic_txt)

    with gsd.hoomd.open(name=str(input_path), mode="r") as source:
        n_frames = len(source)
        assert n_frames != 0

        first_frame = source[0]
        n_particles = int(first_frame.particles.N)
        hexatic_abs = hexatic_abs_matrix_from_table(
            table,
            n_frames=n_frames,
            n_particles=n_particles,
        )

        with gsd.hoomd.open(name=str(output_path), mode="w") as destination:
            for frame_idx, frame in enumerate(source):
                assert int(frame.particles.N) == n_particles

                new_frame = copy.deepcopy(frame)
                velocity = np.zeros((n_particles, 3), dtype=np.float32)
                velocity[:, component] = hexatic_abs[frame_idx].astype(np.float32)
                new_frame.particles.velocity = velocity
                destination.append(new_frame)
