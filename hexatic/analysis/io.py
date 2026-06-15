import copy as _copy
from pathlib import Path as _Path

import gsd.hoomd as _gsd_hoomd
import numpy as _np

from .types import FloatArray, HexaticVelocityFields, ProbabilityDistribution


def save_hexatic_text(
    filename: str | _Path,
    steps: _np.ndarray,
    psi: _np.ndarray,
) -> None:
    steps = _np.asarray(steps, dtype=_np.int64)
    psi = _np.asarray(psi, dtype=_np.complex128)
    assert psi.ndim == 2 and steps.shape == (psi.shape[0],)

    n_frames, n_particles = psi.shape
    data = _np.column_stack(
        (
            _np.repeat(_np.arange(n_frames, dtype=_np.int64), n_particles),
            _np.repeat(steps, n_particles),
            _np.tile(_np.arange(n_particles, dtype=_np.int64), n_frames),
            psi.reshape(-1).real,
            psi.reshape(-1).imag,
            _np.abs(psi.reshape(-1)),
        )
    )
    _Path(filename).parent.mkdir(parents=True, exist_ok=True)
    _np.savetxt(
        filename,
        data,
        fmt=("%d", "%d", "%d", "%.18e", "%.18e", "%.18e"),
        header="frame step particle psi_real psi_imag psi_abs",
    )


def save_neighbor_count_text(
    filename: str | _Path,
    steps: _np.ndarray,
    neighbor_counts: _np.ndarray,
) -> None:
    steps = _np.asarray(steps, dtype=_np.int64)
    neighbor_counts = _np.asarray(neighbor_counts, dtype=_np.int64)
    assert neighbor_counts.ndim == 2 and steps.shape == (neighbor_counts.shape[0],)

    n_frames, n_particles = neighbor_counts.shape
    data = _np.column_stack(
        (
            _np.repeat(_np.arange(n_frames, dtype=_np.int64), n_particles),
            _np.repeat(steps, n_particles),
            _np.tile(_np.arange(n_particles, dtype=_np.int64), n_frames),
            neighbor_counts.reshape(-1),
        )
    )
    _Path(filename).parent.mkdir(parents=True, exist_ok=True)
    _np.savetxt(
        filename,
        data,
        fmt=("%d", "%d", "%d", "%d"),
        header="frame step particle neighbor_count",
    )


def load_hexatic_text(filename: str | _Path) -> FloatArray:
    data = _np.loadtxt(filename, dtype=_np.float64)
    if data.ndim == 1:
        data = data[_np.newaxis, :]
    assert data.ndim == 2 and data.shape[1] >= 6
    return data


def hexatic_abs_matrix_from_table(
    table: _np.ndarray,
    n_frames: int | None = None,
    n_particles: int | None = None,
) -> FloatArray:
    table = _np.asarray(table, dtype=_np.float64)
    assert table.ndim == 2 and table.shape[1] >= 6
    frame_indices = table[:, 0].astype(_np.int64)
    particle_indices = table[:, 2].astype(_np.int64)
    assert not _np.any(frame_indices < 0) and not _np.any(particle_indices < 0)

    n_frames = int(frame_indices.max()) + 1 if n_frames is None else n_frames
    n_particles = int(particle_indices.max()) + 1 if n_particles is None else n_particles
    flat_indices = frame_indices * n_particles + particle_indices
    assert _np.unique(flat_indices).size == flat_indices.size

    matrix = _np.full((n_frames, n_particles), _np.nan, dtype=_np.float64)
    matrix[frame_indices, particle_indices] = table[:, 5]
    assert not _np.any(_np.isnan(matrix))
    return matrix


def hexatic_probability_distribution(
    hexatic_abs: _np.ndarray,
    frame_indices: _np.ndarray,
    min_frame: int = 10,
    bins: int = 50,
    exclude_zeros: bool = False,
) -> ProbabilityDistribution:
    values = _np.asarray(hexatic_abs, dtype=_np.float64).reshape(-1)
    frames = _np.asarray(frame_indices, dtype=_np.int64).reshape(-1)
    assert values.shape == frames.shape
    selected = values[frames > min_frame]
    if exclude_zeros:
        selected = selected[selected > 0.0]
    assert selected.size != 0

    density, edges = _np.histogram(
        _np.clip(selected, 0.0, 1.0),
        bins=bins,
        range=(0.0, 1.0),
        density=True,
    )
    counts, _ = _np.histogram(selected, bins=edges)
    return ProbabilityDistribution(
        bin_centers=(0.5 * (edges[:-1] + edges[1:])).astype(_np.float64, copy=False),
        probability_density=density.astype(_np.float64, copy=False),
        counts=counts.astype(_np.int64, copy=False),
    )


def save_distribution_text(
    filename: str | _Path,
    bin_centers: _np.ndarray,
    probability_density: _np.ndarray,
    counts: _np.ndarray,
) -> None:
    bin_centers = _np.asarray(bin_centers, dtype=_np.float64)
    probability_density = _np.asarray(probability_density, dtype=_np.float64)
    counts = _np.asarray(counts, dtype=_np.int64)
    assert bin_centers.shape == probability_density.shape == counts.shape
    data = _np.column_stack((bin_centers, probability_density, counts))
    _Path(filename).parent.mkdir(parents=True, exist_ok=True)
    _np.savetxt(
        filename,
        data,
        fmt=("%.18e", "%.18e", "%d"),
        header="bin_center probability_density count",
    )


def write_hexatic_velocity_gsd(
    input_gsd: str | _Path,
    output_gsd: str | _Path,
    hexatic_txt: str | _Path,
    component: int = 0,
    neighbor_counts: _np.ndarray | None = None,
    neighbor_component: int = 1,
    disclination_charges: _np.ndarray | None = None,
    charge_component: int = 2,
    dislocation_particles: _np.ndarray | None = None,
) -> None:
    fields = HexaticVelocityFields(
        component=component,
        neighbor_counts=neighbor_counts,
        neighbor_component=neighbor_component,
        disclination_charges=disclination_charges,
        charge_component=charge_component,
        dislocation_particles=dislocation_particles,
    )
    fields.validate_components()

    input_path = _Path(input_gsd)
    output_path = _Path(output_gsd)
    assert input_path.resolve() != output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = load_hexatic_text(hexatic_txt)

    with _gsd_hoomd.open(name=str(input_path), mode="r") as source:
        n_frames = len(source)
        n_particles = int(source[0].particles.N)
        hexatic_abs = hexatic_abs_matrix_from_table(table, n_frames, n_particles)
        neighbor_counts = _field_matrix(fields.neighbor_counts, n_frames, n_particles)
        disclination_charges = _field_matrix(
            fields.disclination_charges,
            n_frames,
            n_particles,
        )
        dislocation_particles = _field_matrix(
            fields.dislocation_particles,
            n_frames,
            n_particles,
        )

        with _gsd_hoomd.open(name=str(output_path), mode="w") as destination:
            for frame_idx, frame in enumerate(source):
                new_frame = _copy.deepcopy(frame)
                velocity = _np.zeros((n_particles, 3), dtype=_np.float32)
                velocity[:, fields.component] = hexatic_abs[frame_idx].astype(_np.float32)
                if neighbor_counts is not None:
                    velocity[:, fields.neighbor_component] = neighbor_counts[frame_idx]
                if disclination_charges is not None:
                    velocity[:, fields.charge_component] = disclination_charges[frame_idx]
                new_frame.particles.velocity = velocity

                if dislocation_particles is not None:
                    orientation = new_frame.particles.orientation
                    orientation = (
                        _np.zeros((n_particles, 4), dtype=_np.float32)
                        if orientation is None
                        else _np.asarray(orientation, dtype=_np.float32).copy()
                    )
                    assert orientation.shape == (n_particles, 4)
                    orientation[:, 0] = dislocation_particles[frame_idx]
                    new_frame.particles.orientation = orientation
                destination.append(new_frame)


def _field_matrix(
    values: _np.ndarray | None,
    n_frames: int,
    n_particles: int,
) -> _np.ndarray | None:
    if values is None:
        return None
    matrix = _np.asarray(values, dtype=_np.float32)
    assert matrix.shape == (n_frames, n_particles)
    return matrix
