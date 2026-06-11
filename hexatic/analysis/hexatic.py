from __future__ import annotations

from abc import ABC, abstractmethod
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


def _minimum_image_x(
    vectors: np.ndarray,
    box_length_x: float | None = None,
) -> FloatArray:
    """Apply minimum-image wrapping"""

    arr = np.asarray(vectors, dtype=np.float64)
    if box_length_x is None:
        return arr

    assert box_length_x > 0.0
    wrapped = arr.copy()
    wrapped[..., 0] -= box_length_x * np.round(wrapped[..., 0] / box_length_x)
    return wrapped


def _pairwise_distances_sq(
    positions: np.ndarray,
    box_length_x: float | None = None,
) -> FloatArray:
    """Return pairwise squared distances"""

    positions = _validate_positions(positions)
    dx = positions[:, 0, np.newaxis] - positions[np.newaxis, :, 0]
    if box_length_x is not None:
        assert box_length_x > 0.0
        dx -= box_length_x * np.round(dx / box_length_x)

    dy = positions[:, 1, np.newaxis] - positions[np.newaxis, :, 1]
    dz = positions[:, 2, np.newaxis] - positions[np.newaxis, :, 2]
    distances_sq = dx * dx + dy * dy + dz * dz
    np.maximum(distances_sq, 0.0, out=distances_sq)
    np.fill_diagonal(distances_sq, np.inf)
    return distances_sq


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


def cylinder_radial_distances(
    positions: np.ndarray,
    center: np.ndarray | None = None,
) -> FloatArray:
    """Return radial distance from the x-axis"""

    positions = _validate_positions(positions)
    center_arr = _validate_center(center)
    yz_vectors = positions[:, 1:3] - center_arr[1:3]
    return np.linalg.norm(yz_vectors, axis=1)


def cylinder_normals(
    positions: np.ndarray,
    center: np.ndarray | None = None,
) -> FloatArray:
    """Return outward cylinder normals orthogonal to the x axis."""

    positions = _validate_positions(positions)
    center_arr = _validate_center(center)
    yz_vectors = positions[:, 1:3] - center_arr[1:3]
    radii = np.linalg.norm(yz_vectors, axis=1)

    assert np.all(radii > 0.0)
    normals = np.zeros_like(positions, dtype=np.float64)
    normals[:, 1:3] = yz_vectors / radii[:, np.newaxis]
    return normals


def local_tangent_basis(normal: np.ndarray) -> tuple[FloatArray, FloatArray]:
    """Build an orthonormal tangent basis for the normal."""

    normal_arr = np.asarray(normal, dtype=np.float64)
    assert normal_arr.shape == (3,)

    normal_norm = np.linalg.norm(normal_arr)
    assert normal_norm > 0.0

    unit_normal = normal_arr / normal_norm
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(unit_normal, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(reference, unit_normal)
    e1_norm = np.linalg.norm(e1)
    assert e1_norm > 0.0
    e1 = e1 / e1_norm

    e2 = np.cross(unit_normal, e1)
    e2_norm = np.linalg.norm(e2)
    assert e2_norm > 0.0
    e2 = e2 / e2_norm

    return e1, e2


def local_cylinder_tangent_basis(normal: np.ndarray) -> tuple[FloatArray, FloatArray]:
    """Return tangent axes for a cylinder"""

    normal_arr = np.asarray(normal, dtype=np.float64)
    assert normal_arr.shape == (3,)

    normal_norm = np.linalg.norm(normal_arr)
    assert normal_norm > 0.0

    unit_normal = normal_arr / normal_norm
    axis = np.array([1.0, 0.0, 0.0])
    assert abs(float(np.dot(unit_normal, axis))) < 1e-12

    e1 = axis
    e2 = np.cross(unit_normal, e1)
    e2_norm = np.linalg.norm(e2)
    assert e2_norm > 0.0
    e2 = e2 / e2_norm

    return e1, e2


def _hexatic_from_neighbors(
    positions: np.ndarray,
    neighbors: np.ndarray,
    normals: np.ndarray,
    box_length_x: float | None = None,
    cylinder_basis: bool = False,
) -> ComplexArray:
    """Compute psi_6."""

    positions = _validate_positions(positions)
    neighbors = np.asarray(neighbors, dtype=np.int64)
    normals = _validate_positions(normals)
    assert neighbors.ndim == 2 and neighbors.shape[0] == positions.shape[0]
    assert normals.shape == positions.shape

    psi = np.empty(positions.shape[0], dtype=np.complex128)
    for particle_idx, neighbor_ids in enumerate(neighbors):
        normal = normals[particle_idx]
        if cylinder_basis:
            e1, e2 = local_cylinder_tangent_basis(normal)
        else:
            e1, e2 = local_tangent_basis(normal)

        bonds = positions[neighbor_ids] - positions[particle_idx]
        bonds = _minimum_image_x(bonds, box_length_x=box_length_x)
        normal_components = bonds @ normal
        tangent_bonds = bonds - normal_components[:, np.newaxis] * normal

        x_coords = tangent_bonds @ e1
        y_coords = tangent_bonds @ e2
        theta = np.arctan2(y_coords, x_coords)
        psi[particle_idx] = np.mean(np.exp(6j * theta))

    return psi


def nearest_neighbors(
    positions: np.ndarray,
    n_neighbors: int = 6,
    box_length_x: float | None = None,
) -> IntArray:
    """Return indices of the n closest particles."""

    positions = _validate_positions(positions)

    n_particles = positions.shape[0]
    assert n_particles > n_neighbors

    distances_sq = _pairwise_distances_sq(positions, box_length_x=box_length_x)

    neighbor_indices = np.argpartition(
        distances_sq, kth=n_neighbors - 1, axis=1
    )[:, :n_neighbors]
    neighbor_distances = np.take_along_axis(
        distances_sq, neighbor_indices, axis=1
    )
    distance_order = np.argsort(neighbor_distances, axis=1)
    neighbor_indices = np.take_along_axis(neighbor_indices, distance_order, axis=1)

    return neighbor_indices.astype(np.int64, copy=False)


def count_neighbors_within_radius(
    positions: np.ndarray,
    radius: float,
    box_length_x: float | None = None,
) -> IntArray:
    """Count neighbors within a 3D distance cutoff."""

    positions = _validate_positions(positions)
    assert radius > 0.0

    distances_sq = _pairwise_distances_sq(positions, box_length_x=box_length_x)

    counts = np.count_nonzero(distances_sq <= radius**2, axis=1)
    return counts.astype(np.int64, copy=False)


def compute_hexatic_order_frame(
    positions: np.ndarray,
    n_neighbors: int = 6,
    center: np.ndarray | None = None,
    cavity_radius: float | None = None,
    shell_delta: float | None = None,
) -> tuple[ComplexArray, IntArray]:
    """Compute hexatic order for every particle in a frame"""

    positions = _validate_positions(positions)
    if cavity_radius is not None or shell_delta is not None:
        assert cavity_radius is not None and shell_delta is not None
        return compute_hexatic_order_frame_near_cavity(
            positions,
            cavity_radius=cavity_radius,
            shell_delta=shell_delta,
            n_neighbors=n_neighbors,
            center=center,
        )

    neighbors = nearest_neighbors(positions, n_neighbors)
    normals = sphere_normals(positions, center)
    psi = _hexatic_from_neighbors(positions, neighbors, normals)

    return psi, neighbors


def cavity_shell_mask(
    positions: np.ndarray,
    cavity_radius: float,
    shell_delta: float,
    center: np.ndarray | None = None,
) -> npt.NDArray[np.bool_]:
    """Return particles w/ radius r_i > R - Delta"""

    positions = _validate_positions(positions)
    center_arr = _validate_center(center)
    assert cavity_radius > 0.0
    assert shell_delta > 0.0

    radii = np.linalg.norm(positions - center_arr, axis=1)
    return radii > cavity_radius - shell_delta


def cylinder_shell_mask(
    positions: np.ndarray,
    cylinder_radius: float,
    shell_delta: float,
    center: np.ndarray | None = None,
) -> npt.NDArray[np.bool_]:
    """Return particles with yz r_i > R - Delta."""

    positions = _validate_positions(positions)
    assert cylinder_radius > 0.0
    assert shell_delta > 0.0

    radii = cylinder_radial_distances(positions, center=center)
    return radii > cylinder_radius - shell_delta


class SurfaceHexaticCalculator(ABC):
    """Shell calculations"""

    cylinder_basis = False

    def __init__(
        self,
        surface_radius: float,
        shell_delta: float,
        n_neighbors: int = 6,
        center: np.ndarray | None = None,
    ) -> None:
        assert surface_radius > 0.0
        assert shell_delta > 0.0
        assert n_neighbors > 0

        self.surface_radius = float(surface_radius)
        self.shell_delta = float(shell_delta)
        self.n_neighbors = int(n_neighbors)
        self.center = _validate_center(center)

    @abstractmethod
    def shell_mask(self, positions: np.ndarray) -> npt.NDArray[np.bool_]:
        """Return the particles participating in the surface calculation."""

    @abstractmethod
    def surface_normals(self, positions: np.ndarray) -> FloatArray:
        """Return normals for the selected surface geometry."""

    def frame_box_length_x(self, frame) -> float | None:
        """Return the x periodic length for a trajectory frame, if needed."""

        return None

    def compute_hexatic_order_frame(
        self,
        positions: np.ndarray,
        box_length_x: float | None = None,
    ) -> tuple[ComplexArray, IntArray]:
        """Compute psi_6 for particles selected by this surface geometry."""

        positions = _validate_positions(positions)
        shell_indices = np.flatnonzero(self.shell_mask(positions))

        n_particles = positions.shape[0]
        psi = np.zeros(n_particles, dtype=np.complex128)
        neighbors = np.full((n_particles, self.n_neighbors), -1, dtype=np.int64)

        if len(shell_indices) <= self.n_neighbors:
            return psi, neighbors

        shell_positions = positions[shell_indices]
        shell_neighbors = nearest_neighbors(
            shell_positions,
            n_neighbors=self.n_neighbors,
            box_length_x=box_length_x,
        )
        shell_normals = self.surface_normals(shell_positions)
        shell_psi = _hexatic_from_neighbors(
            shell_positions,
            shell_neighbors,
            shell_normals,
            box_length_x=box_length_x,
            cylinder_basis=self.cylinder_basis,
        )

        psi[shell_indices] = shell_psi
        neighbors[shell_indices] = shell_indices[shell_neighbors]

        return psi, neighbors

    def compute_neighbor_counts_frame(
        self,
        positions: np.ndarray,
        neighbor_radius: float,
        box_length_x: float | None = None,
    ) -> IntArray:
        """Count fixed-radius neighbors for selected surface particles."""

        positions = _validate_positions(positions)
        shell_indices = np.flatnonzero(self.shell_mask(positions))

        counts = np.zeros(positions.shape[0], dtype=np.int64)
        if len(shell_indices) == 0:
            return counts

        shell_counts = count_neighbors_within_radius(
            positions[shell_indices],
            radius=neighbor_radius,
            box_length_x=box_length_x,
        )
        counts[shell_indices] = shell_counts

        return counts

    def compute_hexatic_order_trajectory(
        self,
        filename: str | Path,
    ) -> tuple[IntArray, ComplexArray]:
        """Compute surface hexatic order for every frame."""

        steps: list[int] = []
        psi_frames: list[ComplexArray] = []
        n_particles: int | None = None

        with gsd.hoomd.open(name=str(filename), mode="r") as trajectory:
            for frame in trajectory:
                particles = frame.particles
                assert particles.position is not None

                positions = _validate_positions(particles.position)
                if n_particles is None:
                    n_particles = positions.shape[0]
                assert positions.shape[0] == n_particles

                psi, _ = self.compute_hexatic_order_frame(
                    positions,
                    box_length_x=self.frame_box_length_x(frame),
                )
                psi_frames.append(psi)
                steps.append(int(frame.configuration.step))

        return np.asarray(steps, dtype=np.int64), np.vstack(psi_frames)

    def compute_neighbor_counts_trajectory(
        self,
        filename: str | Path,
        neighbor_radius: float,
    ) -> tuple[IntArray, IntArray]:
        """Count fixed-radius surface neighbors for every frame."""

        steps: list[int] = []
        count_frames: list[IntArray] = []
        n_particles: int | None = None

        with gsd.hoomd.open(name=str(filename), mode="r") as trajectory:
            for frame in trajectory:
                particles = frame.particles
                assert particles.position is not None

                positions = _validate_positions(particles.position)
                if n_particles is None:
                    n_particles = positions.shape[0]
                assert positions.shape[0] == n_particles

                counts = self.compute_neighbor_counts_frame(
                    positions,
                    neighbor_radius=neighbor_radius,
                    box_length_x=self.frame_box_length_x(frame),
                )
                count_frames.append(counts)
                steps.append(int(frame.configuration.step))

        return np.asarray(steps, dtype=np.int64), np.vstack(count_frames)


class SphereHexaticCalculator(SurfaceHexaticCalculator):
    """Spherical cavity calculations"""

    def __init__(
        self,
        cavity_radius: float,
        shell_delta: float,
        n_neighbors: int = 6,
        center: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            surface_radius=cavity_radius,
            shell_delta=shell_delta,
            n_neighbors=n_neighbors,
            center=center,
        )
        self.cavity_radius = self.surface_radius

    def shell_mask(self, positions: np.ndarray) -> npt.NDArray[np.bool_]:
        return cavity_shell_mask(
            positions,
            cavity_radius=self.cavity_radius,
            shell_delta=self.shell_delta,
            center=self.center,
        )

    def surface_normals(self, positions: np.ndarray) -> FloatArray:
        return sphere_normals(positions, center=self.center)


class CylinderHexaticCalculator(SurfaceHexaticCalculator):
    """Cylinder calculations"""

    cylinder_basis = True

    def __init__(
        self,
        cylinder_radius: float,
        shell_delta: float,
        n_neighbors: int = 6,
        center: np.ndarray | None = None,
    ) -> None:
        super().__init__(
            surface_radius=cylinder_radius,
            shell_delta=shell_delta,
            n_neighbors=n_neighbors,
            center=center,
        )
        self.cylinder_radius = self.surface_radius

    def shell_mask(self, positions: np.ndarray) -> npt.NDArray[np.bool_]:
        return cylinder_shell_mask(
            positions,
            cylinder_radius=self.cylinder_radius,
            shell_delta=self.shell_delta,
            center=self.center,
        )

    def surface_normals(self, positions: np.ndarray) -> FloatArray:
        return cylinder_normals(positions, center=self.center)

    def frame_box_length_x(self, frame) -> float | None:
        return float(frame.configuration.box[0])


def compute_hexatic_order_frame_near_cavity(
    positions: np.ndarray,
    cavity_radius: float,
    shell_delta: float,
    n_neighbors: int = 6,
    center: np.ndarray | None = None,
) -> tuple[ComplexArray, IntArray]:
    """Compute hexatic order only for particles close to the cavity wall"""

    calculator = SphereHexaticCalculator(
        cavity_radius=cavity_radius,
        shell_delta=shell_delta,
        n_neighbors=n_neighbors,
        center=center,
    )
    return calculator.compute_hexatic_order_frame(positions)


def compute_hexatic_order_frame_on_cylinder(
    positions: np.ndarray,
    cylinder_radius: float,
    shell_delta: float,
    n_neighbors: int = 6,
    center: np.ndarray | None = None,
    box_length_x: float | None = None,
) -> tuple[ComplexArray, IntArray]:
    """Compute hexatic order for particles on a cylindrical surface."""

    calculator = CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        n_neighbors=n_neighbors,
        center=center,
    )
    return calculator.compute_hexatic_order_frame(positions, box_length_x=box_length_x)


def compute_neighbor_counts_frame_near_cavity(
    positions: np.ndarray,
    cavity_radius: float,
    shell_delta: float,
    neighbor_radius: float,
    center: np.ndarray | None = None,
) -> IntArray:
    """Count fixed-radius neighbors for surface particles only."""

    calculator = SphereHexaticCalculator(
        cavity_radius=cavity_radius,
        shell_delta=shell_delta,
        center=center,
    )
    return calculator.compute_neighbor_counts_frame(
        positions,
        neighbor_radius=neighbor_radius,
    )


def compute_neighbor_counts_frame_on_cylinder(
    positions: np.ndarray,
    cylinder_radius: float,
    shell_delta: float,
    neighbor_radius: float,
    center: np.ndarray | None = None,
    box_length_x: float | None = None,
) -> IntArray:
    """Count fixed-radius neighbors for cylinder surface particles only."""

    calculator = CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        center=center,
    )
    return calculator.compute_neighbor_counts_frame(
        positions,
        neighbor_radius=neighbor_radius,
        box_length_x=box_length_x,
    )


def compute_hexatic_order_trajectory(
    filename: str | Path,
    n_neighbors: int = 6,
    center: np.ndarray | None = None,
    cavity_radius: float | None = None,
    shell_delta: float | None = None,
) -> tuple[IntArray, ComplexArray]:
    """Compute hexatic order for every frame"""

    if cavity_radius is not None or shell_delta is not None:
        assert cavity_radius is not None and shell_delta is not None
        calculator = SphereHexaticCalculator(
            cavity_radius=cavity_radius,
            shell_delta=shell_delta,
            n_neighbors=n_neighbors,
            center=center,
        )
        return calculator.compute_hexatic_order_trajectory(filename)

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
                cavity_radius=cavity_radius,
                shell_delta=shell_delta,
            )
            psi_frames.append(psi)
            steps.append(int(frame.configuration.step))

    assert psi_frames is not None
    return np.asarray(steps, dtype=np.int64), np.vstack(psi_frames)


def compute_hexatic_order_cylinder_trajectory(
    filename: str | Path,
    cylinder_radius: float,
    shell_delta: float,
    n_neighbors: int = 6,
    center: np.ndarray | None = None,
) -> tuple[IntArray, ComplexArray]:
    """Compute cylinder-surface hexatic order for every frame."""

    calculator = CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        n_neighbors=n_neighbors,
        center=center,
    )
    return calculator.compute_hexatic_order_trajectory(filename)


def compute_neighbor_counts_trajectory(
    filename: str | Path,
    neighbor_radius: float,
    cavity_radius: float,
    shell_delta: float,
    center: np.ndarray | None = None,
) -> tuple[IntArray, IntArray]:
    """Count surface neighbors for every frame"""

    calculator = SphereHexaticCalculator(
        cavity_radius=cavity_radius,
        shell_delta=shell_delta,
        center=center,
    )
    return calculator.compute_neighbor_counts_trajectory(
        filename,
        neighbor_radius=neighbor_radius,
    )


def compute_neighbor_counts_cylinder_trajectory(
    filename: str | Path,
    neighbor_radius: float,
    cylinder_radius: float,
    shell_delta: float,
    center: np.ndarray | None = None,
) -> tuple[IntArray, IntArray]:
    """Count cylinder-surface neighbors for every frame."""

    calculator = CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=shell_delta,
        center=center,
    )
    return calculator.compute_neighbor_counts_trajectory(
        filename,
        neighbor_radius=neighbor_radius,
    )


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

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        filename,
        data,
        fmt=("%d", "%d", "%d", "%.18e", "%.18e", "%.18e"),
        header="frame step particle psi_real psi_imag psi_abs",
    )


def save_neighbor_count_text(
    filename: str | Path,
    steps: np.ndarray,
    neighbor_counts: np.ndarray,
) -> None:
    """Save fixed-radius neighbor counts as a long-form text table."""

    steps = np.asarray(steps, dtype=np.int64)
    neighbor_counts = np.asarray(neighbor_counts, dtype=np.int64)
    assert neighbor_counts.ndim == 2 and steps.shape == (neighbor_counts.shape[0],)

    n_frames, n_particles = neighbor_counts.shape
    frame_col = np.repeat(np.arange(n_frames, dtype=np.int64), n_particles)
    step_col = np.repeat(steps, n_particles)
    particle_col = np.tile(np.arange(n_particles, dtype=np.int64), n_frames)
    data = np.column_stack(
        (
            frame_col,
            step_col,
            particle_col,
            neighbor_counts.reshape(-1),
        )
    )

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        filename,
        data,
        fmt=("%d", "%d", "%d", "%d"),
        header="frame step particle neighbor_count",
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
    exclude_zeros: bool = False,
) -> tuple[FloatArray, FloatArray, IntArray]:
    """Return the probability density of psi for frames after cutoff"""

    values = np.asarray(hexatic_abs, dtype=np.float64).reshape(-1)
    frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    assert values.shape == frames.shape

    selected = values[frames > min_frame]
    if exclude_zeros:
        selected = selected[selected > 0.0]
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
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
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
    neighbor_counts: np.ndarray | None = None,
    neighbor_component: int = 1,
    disclination_charges: np.ndarray | None = None,
    charge_component: int = 2,
) -> None:
    """Write .gsd file with psi as the velocity"""

    assert component in (0, 1, 2)
    assert neighbor_component in (0, 1, 2)
    assert charge_component in (0, 1, 2)
    assert component != neighbor_component or neighbor_counts is None
    assert component != charge_component or disclination_charges is None
    assert neighbor_component != charge_component or (
        neighbor_counts is None or disclination_charges is None
    )

    input_path = Path(input_gsd)
    output_path = Path(output_gsd)
    assert input_path.resolve() != output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        if neighbor_counts is not None:
            neighbor_counts = np.asarray(neighbor_counts, dtype=np.float64)
            assert neighbor_counts.shape == (n_frames, n_particles)
        if disclination_charges is not None:
            disclination_charges = np.asarray(disclination_charges, dtype=np.float64)
            assert disclination_charges.shape == (n_frames, n_particles)

        with gsd.hoomd.open(name=str(output_path), mode="w") as destination:
            for frame_idx, frame in enumerate(source):
                assert int(frame.particles.N) == n_particles

                new_frame = copy.deepcopy(frame)
                velocity = np.zeros((n_particles, 3), dtype=np.float32)
                velocity[:, component] = hexatic_abs[frame_idx].astype(np.float32)
                if neighbor_counts is not None:
                    velocity[:, neighbor_component] = neighbor_counts[frame_idx].astype(
                        np.float32
                    )
                if disclination_charges is not None:
                    velocity[:, charge_component] = disclination_charges[
                        frame_idx
                    ].astype(np.float32)
                new_frame.particles.velocity = velocity
                destination.append(new_frame)


_EXCLUDED_EXPORTS = {
    "ABC",
    "Path",
    "abstractmethod",
    "annotations",
    "copy",
    "gsd",
    "np",
    "npt",
}
__all__ = [
    name
    for name in globals()
    if not name.startswith("_") and name not in _EXCLUDED_EXPORTS
]
