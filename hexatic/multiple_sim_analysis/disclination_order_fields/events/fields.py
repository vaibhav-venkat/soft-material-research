from __future__ import annotations

import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from ...common import (
    FRAME_START,
    FRAME_STOP,
    active_fields_path,
    load_active_fields,
)
from .constants import DEFAULT_EVENT_CONSTANTS, EventAnalysisConstants
from .geometry import cylindrical_to_cartesian, cylinder_distances, minimum_image_x_delta
from .io import event_metric_npz_path, save_event_metric_npz


def angular_delta(theta: np.ndarray | float) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def physical_cylinder_delta(
    from_coords: np.ndarray,
    to_coords: np.ndarray,
    *,
    radius: float,
    box_length_x: float,
) -> np.ndarray:
    """Return local physical deltas ``(dx, R dtheta)`` from one coord array to another."""
    delta = np.asarray(to_coords, dtype=np.float64)[..., :2] - np.asarray(
        from_coords,
        dtype=np.float64,
    )[..., :2]
    delta[..., 0] = minimum_image_x_delta(delta[..., 0], box_length_x)
    delta[..., 1] = radius * angular_delta(delta[..., 1])
    return delta


def particle_velocity_from_coords(
    coords: np.ndarray,
    steps: np.ndarray,
    *,
    radius: float,
    box_length_x: float,
    timestep: float = float(cylinder.TIMESTEP),
) -> np.ndarray:
    """Frame-local particle velocity in physical coordinates ``(vx, R*dtheta/dt, dr/dt)``.

    Velocity for frame ``t`` compares frame ``t-1`` to ``t``; frame 0 is undefined.
    """
    coords = np.asarray(coords, dtype=np.float64)
    steps = np.asarray(steps, dtype=np.int64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (frames, particles, 3), got {coords.shape}")
    if steps.shape != (coords.shape[0],):
        raise ValueError("steps length must match coords frame count")

    velocities = np.full_like(coords, np.nan, dtype=np.float64)
    for frame_idx in range(1, coords.shape[0]):
        delta_t = (steps[frame_idx] - steps[frame_idx - 1]) * timestep
        if delta_t <= 0.0:
            continue
        physical_delta = physical_cylinder_delta(
            coords[frame_idx - 1],
            coords[frame_idx],
            radius=radius,
            box_length_x=box_length_x,
        )
        velocities[frame_idx, :, 0] = physical_delta[:, 0] / delta_t
        velocities[frame_idx, :, 1] = physical_delta[:, 1] / delta_t
        velocities[frame_idx, :, 2] = (
            coords[frame_idx, :, 2] - coords[frame_idx - 1, :, 2]
        ) / delta_t
    return velocities


def defect_annulus_masks(
    particle_coords: np.ndarray,
    target_coord: np.ndarray,
    charges: np.ndarray,
    box_length_x: float,
    *,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(core_mask, annulus_mask, distances)`` for one target defect.

    The annulus excludes all particles whose disclination charge is non-zero.
    """
    particle_cartesian = cylindrical_to_cartesian(np.asarray(particle_coords, dtype=np.float64))
    target_cartesian = cylindrical_to_cartesian(
        np.asarray(target_coord, dtype=np.float64)[np.newaxis, :]
    )
    distances = cylinder_distances(particle_cartesian, target_cartesian, box_length_x)[:, 0]
    core_mask = distances < float(constants.annulus_core_radius)
    annulus_mask = (
        (distances > float(constants.annulus_core_radius))
        & (distances < float(constants.annulus_outer_radius))
        & (np.asarray(charges) == 0)
    )
    return core_mask, annulus_mask, distances


def _nanmean_for_mask(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[mask]
    if selected.size == 0:
        return np.nan
    finite = np.isfinite(selected)
    if not np.any(finite):
        return np.nan
    return float(np.mean(selected[finite]))


def annulus_velocity_statistics(velocities: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = np.asarray(velocities, dtype=np.float64)[mask]
    if selected.size == 0:
        return {"u_mean": np.nan, "u_rms": np.nan, "u_fluct": np.nan}
    finite = np.all(np.isfinite(selected), axis=1)
    selected = selected[finite]
    if selected.size == 0:
        return {"u_mean": np.nan, "u_rms": np.nan, "u_fluct": np.nan}
    mean_velocity = np.mean(selected, axis=0)
    speeds = np.linalg.norm(selected, axis=1)
    fluctuations = selected - mean_velocity
    return {
        "u_mean": float(np.linalg.norm(mean_velocity)),
        "u_rms": float(np.mean(speeds)),
        "u_fluct": float(np.sqrt(np.mean(np.sum(fluctuations * fluctuations, axis=1)))),
    }


def sample_defect_annulus_fields(
    particle_coords: np.ndarray,
    target_coord: np.ndarray,
    charges: np.ndarray,
    box_length_x: float,
    *,
    scalar_fields: dict[str, np.ndarray] | None = None,
    vector_fields: dict[str, np.ndarray] | None = None,
    velocities: np.ndarray | None = None,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> dict[str, float]:
    """Sample particle-local arrays in a defect-excluded annulus.

    Empty annuli deliberately return ``NaN`` for means and expose ``annulus_count``.
    """
    _, annulus_mask, distances = defect_annulus_masks(
        particle_coords,
        target_coord,
        charges,
        box_length_x,
        constants=constants,
    )
    result: dict[str, float] = {
        "annulus_count": float(np.count_nonzero(annulus_mask)),
        "nearest_defect_distance": np.nan,
    }
    defect_distances = distances[np.asarray(charges) != 0]
    defect_distances = defect_distances[defect_distances > 0.0]
    if defect_distances.size:
        result["nearest_defect_distance"] = float(np.min(defect_distances))

    if scalar_fields:
        for name, values in scalar_fields.items():
            result[name] = _nanmean_for_mask(np.asarray(values), annulus_mask)

    if vector_fields:
        for name, values in vector_fields.items():
            selected = np.asarray(values, dtype=np.float64)[annulus_mask]
            if selected.size == 0:
                result[f"{name}_norm"] = np.nan
                continue
            finite = np.all(np.isfinite(selected), axis=1)
            result[f"{name}_norm"] = (
                float(np.mean(np.linalg.norm(selected[finite], axis=1)))
                if np.any(finite)
                else np.nan
            )

    if velocities is not None:
        result.update(annulus_velocity_statistics(np.asarray(velocities), annulus_mask))
    return result


def _neighbor_vectors_physical(
    coords: np.ndarray,
    center_idx: int,
    *,
    radius: float,
    box_length_x: float,
) -> np.ndarray:
    centers = np.repeat(coords[center_idx][np.newaxis, :], coords.shape[0], axis=0)
    return physical_cylinder_delta(
        centers,
        coords,
        radius=radius,
        box_length_x=box_length_x,
    )


def compute_d2min_frame(
    coords_t: np.ndarray,
    coords_next: np.ndarray,
    *,
    radius: float,
    box_length_x: float,
    neighbor_radius: float = DEFAULT_EVENT_CONSTANTS.neighbor_count_radius,
    normalization_length: float = DEFAULT_EVENT_CONSTANTS.neighbor_count_radius,
    min_neighbors: int = 3,
) -> np.ndarray:
    """Compute per-particle ``D^2_min / normalization_length^2`` for one frame pair."""
    coords_t = np.asarray(coords_t, dtype=np.float64)
    coords_next = np.asarray(coords_next, dtype=np.float64)
    if coords_t.shape != coords_next.shape or coords_t.ndim != 2 or coords_t.shape[1] != 3:
        raise ValueError("coords_t and coords_next must both have shape (particles, 3)")

    d2min = np.full(coords_t.shape[0], np.nan, dtype=np.float64)
    norm = float(normalization_length) ** 2
    if norm <= 0.0:
        raise ValueError("normalization_length must be positive")

    for particle_idx in range(coords_t.shape[0]):
        reference = _neighbor_vectors_physical(
            coords_t,
            particle_idx,
            radius=radius,
            box_length_x=box_length_x,
        )
        neighbor_mask = (
            (np.linalg.norm(reference, axis=1) <= neighbor_radius)
            & (np.arange(coords_t.shape[0]) != particle_idx)
        )
        if np.count_nonzero(neighbor_mask) < min_neighbors:
            continue
        deformed = _neighbor_vectors_physical(
            coords_next,
            particle_idx,
            radius=radius,
            box_length_x=box_length_x,
        )
        p = reference[neighbor_mask]
        q = deformed[neighbor_mask]
        try:
            affine, _, rank, _ = np.linalg.lstsq(p, q, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < 2:
            continue
        residual = q - p @ affine
        d2min[particle_idx] = float(np.mean(np.sum(residual * residual, axis=1)) / norm)
    return d2min


def compute_d2min_series(
    coords: np.ndarray,
    *,
    radius: float,
    box_length_x: float,
    neighbor_radius: float = DEFAULT_EVENT_CONSTANTS.neighbor_count_radius,
    normalization_length: float = DEFAULT_EVENT_CONSTANTS.neighbor_count_radius,
    min_neighbors: int = 3,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (frames, particles, 3), got {coords.shape}")
    d2min = np.full(coords.shape[:2], np.nan, dtype=np.float64)
    for frame_idx in range(coords.shape[0] - 1):
        d2min[frame_idx] = compute_d2min_frame(
            coords[frame_idx],
            coords[frame_idx + 1],
            radius=radius,
            box_length_x=box_length_x,
            neighbor_radius=neighbor_radius,
            normalization_length=normalization_length,
            min_neighbors=min_neighbors,
        )
    return d2min


def save_d2min_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    *,
    overwrite: bool = False,
    coords: np.ndarray | None = None,
    steps: np.ndarray | None = None,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> np.ndarray:
    output_path = event_metric_npz_path("D2min", case)
    if output_path.exists() and not overwrite:
        with np.load(output_path) as data:
            return np.asarray(data["D2min"], dtype=np.float64)

    if coords is None or steps is None:
        fields = load_active_fields(active_fields_path(case))
        coords = np.asarray(fields.coords, dtype=np.float64)
        steps = np.asarray(fields.steps, dtype=np.int64)
    d2min = compute_d2min_series(
        np.asarray(coords, dtype=np.float64),
        radius=float(case.radius),
        box_length_x=float(case.lx),
        neighbor_radius=float(constants.neighbor_count_radius),
        normalization_length=float(constants.neighbor_count_radius),
    )
    save_event_metric_npz(
        output_path,
        (case,),
        "D2min",
        {
            "steps": np.asarray(steps, dtype=np.int64),
            "D2min": d2min,
        },
        frame_start=frame_start,
        frame_stop=frame_stop,
    )
    return d2min


def compute_event_centered_fields(*args, **kwargs):
    """Plotting-facing placeholder; use sampler helpers above for local fields."""
    raise NotImplementedError("event-centered aggregation is reserved for plotting work")
