from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from ...common import (
    FRAME_START,
    FRAME_STOP,
    active_fields_path,
    frame_indices,
    load_active_fields,
    neighbor_counts_path,
)
from ...disclination import _load_neighbor_counts
from .constants import DEFAULT_EVENT_CONSTANTS, EventAnalysisConstants
from .geometry import (
    cylindrical_to_cartesian,
    cylinder_distances,
    minimum_image_x_delta,
)
from .io import event_metric_npz_path, save_event_metric_npz
from .validation import validate_frame_particle_shape, validate_step_alignment


@dataclass(frozen=True)
class DefectFrame:
    frame_index: int
    step: int
    particle_indices: np.ndarray
    charges: np.ndarray
    positions: np.ndarray
    cartesian_positions: np.ndarray


@dataclass(frozen=True)
class FrameMatchResult:
    prev_indices: np.ndarray
    cur_indices: np.ndarray
    distances: np.ndarray
    unmatched_prev_indices: np.ndarray
    unmatched_cur_indices: np.ndarray
    rejected_prev_indices: np.ndarray
    rejected_cur_indices: np.ndarray
    ambiguous_pair_count: int
    rejected_swap_count: int


@dataclass(frozen=True)
class DefectTrackTable:
    frame_axis: np.ndarray
    steps: np.ndarray
    track_id: np.ndarray
    charge: np.ndarray
    frame_start: np.ndarray
    frame_stop: np.ndarray
    particle_index: np.ndarray
    positions: np.ndarray
    matched_confident: np.ndarray
    velocity: np.ndarray
    birth_ambiguous: np.ndarray
    death_ambiguous: np.ndarray
    annihilation_flag: np.ndarray
    annihilation_partner_id: np.ndarray
    ambiguity_count_by_frame: np.ndarray
    rejected_swap_count_by_frame: np.ndarray

    @property
    def n_tracks(self) -> int:
        return int(self.track_id.size)

    @property
    def n_frames(self) -> int:
        return int(self.frame_axis.size)


def _load_neighbor_count_steps(path: str | Path) -> np.ndarray:
    table = np.loadtxt(path, dtype=np.int64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    if table.shape[1] < 2:
        raise ValueError(f"Neighbor count table lacks a step column: {path}")

    frame_indices_table = table[:, 0]
    step_values = table[:, 1]
    n_frames = int(frame_indices_table.max()) + 1
    steps = np.full(n_frames, -1, dtype=np.int64)
    for frame_idx in range(n_frames):
        frame_steps = np.unique(step_values[frame_indices_table == frame_idx])
        if frame_steps.size != 1:
            raise ValueError(
                f"Neighbor count table has inconsistent steps for frame {frame_idx}: {path}"
            )
        steps[frame_idx] = int(frame_steps[0])
    return steps


def validate_tracking_inputs(
    coords: np.ndarray,
    field_steps: np.ndarray,
    counts: np.ndarray,
    count_steps: np.ndarray,
) -> None:
    coords = np.asarray(coords, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    field_steps = np.asarray(field_steps, dtype=np.int64)
    count_steps = np.asarray(count_steps, dtype=np.int64)

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (frames, particles, 3), got {coords.shape}")
    if counts.ndim != 2:
        raise ValueError(f"neighbor counts must have shape (frames, particles), got {counts.shape}")

    if coords.shape[0] <= 0:
        raise ValueError("tracking inputs contain no common frames")
    if (
        coords.shape[0] != counts.shape[0]
        or field_steps.shape != count_steps.shape
        or field_steps.shape != (counts.shape[0],)
    ):
        raise ValueError(
            "active-field coords, active-field steps, and neighbor-count tables "
            "must have the same frame count"
        )
    validate_frame_particle_shape("coords", coords, counts.shape)
    validate_step_alignment("active fields", field_steps, count_steps)
    if np.any(np.diff(field_steps) <= 0):
        raise ValueError("active-field steps must be strictly increasing")


def build_defect_frames(
    coords: np.ndarray,
    charges: np.ndarray,
    steps: np.ndarray,
    frame_axis: np.ndarray,
) -> list[DefectFrame]:
    coords = np.asarray(coords, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.int64)
    steps = np.asarray(steps, dtype=np.int64)
    frame_axis = np.asarray(frame_axis, dtype=np.int64)
    frames: list[DefectFrame] = []
    for frame_idx in frame_axis:
        frame_charges = charges[frame_idx]
        particle_indices = np.flatnonzero(frame_charges != 0).astype(np.int64)
        positions = np.asarray(coords[frame_idx, particle_indices], dtype=np.float64)
        frames.append(
            DefectFrame(
                frame_index=int(frame_idx),
                step=int(steps[frame_idx]),
                particle_indices=particle_indices,
                charges=frame_charges[particle_indices].astype(np.int64),
                positions=positions,
                cartesian_positions=cylindrical_to_cartesian(positions),
            )
        )
    return frames


def _second_or_inf(sorted_distances: np.ndarray) -> float:
    if sorted_distances.size < 2:
        return np.inf
    return float(sorted_distances[1])


def _match_charge_group(
    prev_indices: np.ndarray,
    cur_indices: np.ndarray,
    distances: np.ndarray,
    match_tol: float,
    ambiguity_ratio: float,
    use_linear_assignment: bool,
) -> tuple[list[tuple[int, int, float]], set[int], set[int], int, int]:
    if prev_indices.size == 0 or cur_indices.size == 0:
        return [], set(), set(), 0, 0

    candidate_mask = distances <= match_tol
    ambiguous_pair_count = 0
    rejected_swap_count = 0
    confident_pairs: list[tuple[int, int, float]] = []

    for local_prev in range(prev_indices.size):
        cur_candidates = np.flatnonzero(candidate_mask[local_prev])
        if cur_candidates.size == 0:
            continue
        sorted_for_prev = np.sort(distances[local_prev, cur_candidates])
        prev_best = float(sorted_for_prev[0])
        prev_second = _second_or_inf(sorted_for_prev)
        for local_cur in cur_candidates:
            dist = float(distances[local_prev, local_cur])
            prev_clear = prev_second >= ambiguity_ratio * max(dist, np.finfo(float).eps)
            prev_is_nearest = dist <= prev_best + 1e-12

            prev_candidates = np.flatnonzero(candidate_mask[:, local_cur])
            sorted_for_cur = np.sort(distances[prev_candidates, local_cur])
            cur_best = float(sorted_for_cur[0])
            cur_second = _second_or_inf(sorted_for_cur)
            cur_clear = cur_second >= ambiguity_ratio * max(dist, np.finfo(float).eps)
            cur_is_nearest = dist <= cur_best + 1e-12

            if prev_is_nearest and cur_is_nearest and prev_clear and cur_clear:
                confident_pairs.append(
                    (int(prev_indices[local_prev]), int(cur_indices[local_cur]), dist)
                )
            else:
                ambiguous_pair_count += 1

    matches: list[tuple[int, int, float]] = []
    used_prev: set[int] = set()
    used_cur: set[int] = set()
    for prev_idx, cur_idx, dist in sorted(confident_pairs, key=lambda item: item[2]):
        if prev_idx in used_prev or cur_idx in used_cur:
            continue
        matches.append((prev_idx, cur_idx, dist))
        used_prev.add(prev_idx)
        used_cur.add(cur_idx)

    if use_linear_assignment:
        remaining_prev_local = [
            i for i, idx in enumerate(prev_indices) if int(idx) not in used_prev
        ]
        remaining_cur_local = [
            j for j, idx in enumerate(cur_indices) if int(idx) not in used_cur
        ]
        if remaining_prev_local and remaining_cur_local:
            try:
                from scipy.optimize import linear_sum_assignment
            except ImportError:
                linear_sum_assignment = None
            if linear_sum_assignment is not None:
                cost = distances[np.ix_(remaining_prev_local, remaining_cur_local)].copy()
                cost[cost > match_tol] = match_tol * 1e6
                row_ind, col_ind = linear_sum_assignment(cost)
                for row, col in zip(row_ind, col_ind):
                    local_prev = remaining_prev_local[int(row)]
                    local_cur = remaining_cur_local[int(col)]
                    if cost[int(row), int(col)] > match_tol:
                        continue
                    prev_idx = int(prev_indices[local_prev])
                    cur_idx = int(cur_indices[local_cur])
                    if prev_idx in used_prev or cur_idx in used_cur:
                        continue
                    sorted_for_prev = np.sort(distances[local_prev, candidate_mask[local_prev]])
                    sorted_for_cur = np.sort(distances[candidate_mask[:, local_cur], local_cur])
                    dist = float(distances[local_prev, local_cur])
                    if (
                        sorted_for_prev.size
                        and sorted_for_cur.size
                        and dist <= sorted_for_prev[0] + 1e-12
                        and dist <= sorted_for_cur[0] + 1e-12
                        and _second_or_inf(sorted_for_prev)
                        >= ambiguity_ratio * max(dist, np.finfo(float).eps)
                        and _second_or_inf(sorted_for_cur)
                        >= ambiguity_ratio * max(dist, np.finfo(float).eps)
                    ):
                        matches.append((prev_idx, cur_idx, dist))
                        used_prev.add(prev_idx)
                        used_cur.add(cur_idx)

    rejected_prev = {
        int(prev_indices[i])
        for i in range(prev_indices.size)
        if int(prev_indices[i]) not in used_prev and np.any(candidate_mask[i])
    }
    rejected_cur = {
        int(cur_indices[j])
        for j in range(cur_indices.size)
        if int(cur_indices[j]) not in used_cur and np.any(candidate_mask[:, j])
    }
    rejected_swap_count += len(rejected_prev) + len(rejected_cur)
    return matches, rejected_prev, rejected_cur, ambiguous_pair_count, rejected_swap_count


def _match_defects_frame(
    prev: DefectFrame,
    cur: DefectFrame,
    box_length_x: float,
    *,
    match_tol: float = DEFAULT_EVENT_CONSTANTS.match_tolerance,
    ambiguity_ratio: float = 1.5,
    use_linear_assignment: bool = True,
) -> FrameMatchResult:
    """Match same-charge defects between adjacent frames.

    Returned indices are local defect indices within ``prev`` and ``cur``. Matches are
    accepted only when nearest-neighbor identity is well separated from alternatives;
    close rejected alternatives are exposed so event classification can avoid
    treating ambiguous relabeling as physics.
    """
    if match_tol <= 0.0:
        raise ValueError("match_tol must be positive")
    if ambiguity_ratio <= 1.0:
        raise ValueError("ambiguity_ratio must be greater than 1")

    if prev.particle_indices.size == 0:
        return FrameMatchResult(
            prev_indices=np.asarray([], dtype=np.int64),
            cur_indices=np.asarray([], dtype=np.int64),
            distances=np.asarray([], dtype=np.float64),
            unmatched_prev_indices=np.asarray([], dtype=np.int64),
            unmatched_cur_indices=np.arange(cur.particle_indices.size, dtype=np.int64),
            rejected_prev_indices=np.asarray([], dtype=np.int64),
            rejected_cur_indices=np.asarray([], dtype=np.int64),
            ambiguous_pair_count=0,
            rejected_swap_count=0,
        )
    if cur.particle_indices.size == 0:
        return FrameMatchResult(
            prev_indices=np.asarray([], dtype=np.int64),
            cur_indices=np.asarray([], dtype=np.int64),
            distances=np.asarray([], dtype=np.float64),
            unmatched_prev_indices=np.arange(prev.particle_indices.size, dtype=np.int64),
            unmatched_cur_indices=np.asarray([], dtype=np.int64),
            rejected_prev_indices=np.asarray([], dtype=np.int64),
            rejected_cur_indices=np.asarray([], dtype=np.int64),
            ambiguous_pair_count=0,
            rejected_swap_count=0,
        )

    all_distances = cylinder_distances(
        cur.cartesian_positions,
        prev.cartesian_positions,
        box_length_x,
    ).T
    all_matches: list[tuple[int, int, float]] = []
    rejected_prev: set[int] = set()
    rejected_cur: set[int] = set()
    ambiguous_pair_count = 0
    rejected_swap_count = 0

    for charge in np.unique(np.concatenate((prev.charges, cur.charges))):
        prev_group = np.flatnonzero(prev.charges == charge)
        cur_group = np.flatnonzero(cur.charges == charge)
        group_matches, group_rejected_prev, group_rejected_cur, ambiguous, rejected = (
            _match_charge_group(
                prev_group,
                cur_group,
                all_distances[np.ix_(prev_group, cur_group)],
                match_tol,
                ambiguity_ratio,
                use_linear_assignment,
            )
        )
        all_matches.extend(group_matches)
        rejected_prev.update(group_rejected_prev)
        rejected_cur.update(group_rejected_cur)
        ambiguous_pair_count += ambiguous
        rejected_swap_count += rejected

    all_matches.sort(key=lambda item: (item[2], item[0], item[1]))
    matched_prev = np.asarray([item[0] for item in all_matches], dtype=np.int64)
    matched_cur = np.asarray([item[1] for item in all_matches], dtype=np.int64)
    distances = np.asarray([item[2] for item in all_matches], dtype=np.float64)
    unmatched_prev = np.setdiff1d(
        np.arange(prev.particle_indices.size, dtype=np.int64),
        matched_prev,
        assume_unique=False,
    )
    unmatched_cur = np.setdiff1d(
        np.arange(cur.particle_indices.size, dtype=np.int64),
        matched_cur,
        assume_unique=False,
    )
    return FrameMatchResult(
        prev_indices=matched_prev,
        cur_indices=matched_cur,
        distances=distances,
        unmatched_prev_indices=unmatched_prev,
        unmatched_cur_indices=unmatched_cur,
        rejected_prev_indices=np.asarray(sorted(rejected_prev), dtype=np.int64),
        rejected_cur_indices=np.asarray(sorted(rejected_cur), dtype=np.int64),
        ambiguous_pair_count=int(ambiguous_pair_count),
        rejected_swap_count=int(rejected_swap_count),
    )


def _empty_track_arrays(n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full(n_frames, -1, dtype=np.int64),
        np.full((n_frames, 3), np.nan, dtype=np.float64),
        np.zeros(n_frames, dtype=bool),
        np.full((n_frames, 3), np.nan, dtype=np.float64),
    )


def _coordinate_velocity(
    previous_position: np.ndarray,
    current_position: np.ndarray,
    delta_t: float,
    box_length_x: float,
) -> np.ndarray:
    velocity = np.full(3, np.nan, dtype=np.float64)
    if delta_t <= 0.0:
        return velocity
    delta = np.asarray(current_position, dtype=np.float64) - np.asarray(
        previous_position,
        dtype=np.float64,
    )
    delta[0] = minimum_image_x_delta(delta[0], box_length_x)
    delta[1] = (delta[1] + np.pi) % (2.0 * np.pi) - np.pi
    return delta / delta_t


def track_defect_frames(
    frames: list[DefectFrame],
    box_length_x: float,
    *,
    match_tol: float = DEFAULT_EVENT_CONSTANTS.match_tolerance,
    timestep: float = float(cylinder.TIMESTEP),
    use_linear_assignment: bool = True,
) -> DefectTrackTable:
    if not frames:
        raise ValueError("at least one defect frame is required")

    n_frames = len(frames)
    frame_axis = np.asarray([frame.frame_index for frame in frames], dtype=np.int64)
    steps = np.asarray([frame.step for frame in frames], dtype=np.int64)
    if np.any(np.diff(frame_axis) != 1):
        raise ValueError("tracking frames must be contiguous and sorted")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("tracking steps must be strictly increasing")

    track_particle: list[np.ndarray] = []
    track_positions: list[np.ndarray] = []
    track_matched: list[np.ndarray] = []
    track_velocity: list[np.ndarray] = []
    track_charge: list[int] = []
    birth_ambiguous: list[bool] = []
    death_ambiguous: list[bool] = []
    active_track_by_prev_local: dict[int, int] = {}
    ambiguity_count_by_frame = np.zeros(n_frames, dtype=np.int64)
    rejected_swap_count_by_frame = np.zeros(n_frames, dtype=np.int64)

    def new_track(charge: int, frame_offset: int, local_idx: int, ambiguous: bool) -> int:
        particle_index, positions, matched, velocity = _empty_track_arrays(n_frames)
        track_idx = len(track_particle)
        particle_index[frame_offset] = int(frames[frame_offset].particle_indices[local_idx])
        positions[frame_offset] = frames[frame_offset].positions[local_idx]
        track_particle.append(particle_index)
        track_positions.append(positions)
        track_matched.append(matched)
        track_velocity.append(velocity)
        track_charge.append(int(charge))
        birth_ambiguous.append(bool(ambiguous))
        death_ambiguous.append(False)
        return track_idx

    for local_idx, charge in enumerate(frames[0].charges):
        active_track_by_prev_local[local_idx] = new_track(int(charge), 0, local_idx, False)

    for frame_offset in range(1, n_frames):
        prev = frames[frame_offset - 1]
        cur = frames[frame_offset]
        match = _match_defects_frame(
            prev,
            cur,
            box_length_x,
            match_tol=match_tol,
            use_linear_assignment=use_linear_assignment,
        )
        ambiguity_count_by_frame[frame_offset] = match.ambiguous_pair_count
        rejected_swap_count_by_frame[frame_offset] = match.rejected_swap_count

        current_track_by_cur_local: dict[int, int] = {}
        rejected_prev = set(int(idx) for idx in match.rejected_prev_indices)
        rejected_cur = set(int(idx) for idx in match.rejected_cur_indices)
        for prev_local, cur_local in zip(match.prev_indices, match.cur_indices):
            track_idx = active_track_by_prev_local[int(prev_local)]
            current_track_by_cur_local[int(cur_local)] = track_idx
            track_particle[track_idx][frame_offset] = int(cur.particle_indices[int(cur_local)])
            track_positions[track_idx][frame_offset] = cur.positions[int(cur_local)]
            track_matched[track_idx][frame_offset] = True
            delta_t = (cur.step - prev.step) * timestep
            track_velocity[track_idx][frame_offset] = _coordinate_velocity(
                prev.positions[int(prev_local)],
                cur.positions[int(cur_local)],
                delta_t,
                box_length_x,
            )

        for prev_local in match.unmatched_prev_indices:
            track_idx = active_track_by_prev_local.get(int(prev_local))
            if track_idx is not None and int(prev_local) in rejected_prev:
                death_ambiguous[track_idx] = True

        for cur_local in match.unmatched_cur_indices:
            current_track_by_cur_local[int(cur_local)] = new_track(
                int(cur.charges[int(cur_local)]),
                frame_offset,
                int(cur_local),
                int(cur_local) in rejected_cur,
            )
        active_track_by_prev_local = current_track_by_cur_local

    if track_particle:
        particle_index = np.asarray(track_particle, dtype=np.int64)
        positions = np.asarray(track_positions, dtype=np.float64)
        matched_confident = np.asarray(track_matched, dtype=bool)
        velocity = np.asarray(track_velocity, dtype=np.float64)
    else:
        particle_index = np.full((0, n_frames), -1, dtype=np.int64)
        positions = np.full((0, n_frames, 3), np.nan, dtype=np.float64)
        matched_confident = np.zeros((0, n_frames), dtype=bool)
        velocity = np.full((0, n_frames, 3), np.nan, dtype=np.float64)
    charges = np.asarray(track_charge, dtype=np.int64)
    if particle_index.size == 0:
        frame_start = np.asarray([], dtype=np.int64)
        frame_stop = np.asarray([], dtype=np.int64)
    else:
        present = particle_index >= 0
        frame_start = frame_axis[np.argmax(present, axis=1)]
        reverse_stop = present.shape[1] - np.argmax(present[:, ::-1], axis=1)
        frame_stop = frame_axis[0] + reverse_stop

    track_id = np.arange(len(track_particle), dtype=np.int64)
    return DefectTrackTable(
        frame_axis=frame_axis,
        steps=steps,
        track_id=track_id,
        charge=charges,
        frame_start=frame_start.astype(np.int64),
        frame_stop=frame_stop.astype(np.int64),
        particle_index=particle_index,
        positions=positions,
        matched_confident=matched_confident,
        velocity=velocity,
        birth_ambiguous=np.asarray(birth_ambiguous, dtype=bool),
        death_ambiguous=np.asarray(death_ambiguous, dtype=bool),
        annihilation_flag=np.zeros(track_id.size, dtype=bool),
        annihilation_partner_id=np.full(track_id.size, -1, dtype=np.int64),
        ambiguity_count_by_frame=ambiguity_count_by_frame,
        rejected_swap_count_by_frame=rejected_swap_count_by_frame,
    )


def defect_track_values(table: DefectTrackTable) -> dict[str, np.ndarray]:
    return {
        "frame_axis": table.frame_axis,
        "steps": table.steps,
        "track_id": table.track_id,
        "charge": table.charge,
        "track_frame_start": table.frame_start,
        "track_frame_stop": table.frame_stop,
        "particle_index_by_track_frame": table.particle_index,
        "x": table.positions[..., 0],
        "theta": table.positions[..., 1],
        "r": table.positions[..., 2],
        "matched_confident": table.matched_confident,
        "velocity_x": table.velocity[..., 0],
        "velocity_theta": table.velocity[..., 1],
        "velocity_r": table.velocity[..., 2],
        "birth_ambiguous": table.birth_ambiguous,
        "death_ambiguous": table.death_ambiguous,
        "annihilation_flag": table.annihilation_flag,
        "annihilation_partner_id": table.annihilation_partner_id,
        "ambiguity_count_by_frame": table.ambiguity_count_by_frame,
        "rejected_swap_count_by_frame": table.rejected_swap_count_by_frame,
    }


def defect_track_table_from_values(values: dict[str, np.ndarray]) -> DefectTrackTable:
    positions = np.stack(
        (
            np.asarray(values["x"], dtype=np.float64),
            np.asarray(values["theta"], dtype=np.float64),
            np.asarray(values["r"], dtype=np.float64),
        ),
        axis=-1,
    )
    velocity = np.stack(
        (
            np.asarray(values["velocity_x"], dtype=np.float64),
            np.asarray(values["velocity_theta"], dtype=np.float64),
            np.asarray(values["velocity_r"], dtype=np.float64),
        ),
        axis=-1,
    )
    return DefectTrackTable(
        frame_axis=np.asarray(values["frame_axis"], dtype=np.int64),
        steps=np.asarray(values["steps"], dtype=np.int64),
        track_id=np.asarray(values["track_id"], dtype=np.int64),
        charge=np.asarray(values["charge"], dtype=np.int64),
        frame_start=np.asarray(values["track_frame_start"], dtype=np.int64),
        frame_stop=np.asarray(values["track_frame_stop"], dtype=np.int64),
        particle_index=np.asarray(values["particle_index_by_track_frame"], dtype=np.int64),
        positions=positions,
        matched_confident=np.asarray(values["matched_confident"], dtype=bool),
        velocity=velocity,
        birth_ambiguous=np.asarray(values["birth_ambiguous"], dtype=bool),
        death_ambiguous=np.asarray(values["death_ambiguous"], dtype=bool),
        annihilation_flag=np.asarray(values["annihilation_flag"], dtype=bool),
        annihilation_partner_id=np.asarray(values["annihilation_partner_id"], dtype=np.int64),
        ambiguity_count_by_frame=np.asarray(values["ambiguity_count_by_frame"], dtype=np.int64),
        rejected_swap_count_by_frame=np.asarray(
            values["rejected_swap_count_by_frame"],
            dtype=np.int64,
        ),
    )


def load_defect_tracks_npz(path: str | Path) -> DefectTrackTable:
    with np.load(path) as data:
        values = {name: np.asarray(data[name]) for name in data.files}
    return defect_track_table_from_values(values)


def track_persistent_defects(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    *,
    overwrite: bool = False,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> DefectTrackTable:
    output_path = event_metric_npz_path("defect_tracks", case)
    if output_path.exists() and not overwrite:
        return load_defect_tracks_npz(output_path)

    counts_path = neighbor_counts_path(case)
    counts = _load_neighbor_counts(counts_path)
    count_steps = _load_neighbor_count_steps(counts_path)
    fields = load_active_fields(active_fields_path(case))
    coords = np.asarray(fields.coords, dtype=np.float64)
    field_steps = np.asarray(fields.steps, dtype=np.int64)
    validate_tracking_inputs(coords, field_steps, counts, count_steps)

    n_frames = counts.shape[0]
    selected = frame_indices(n_frames, frame_start, frame_stop)
    if selected.size == 0:
        raise ValueError(
            f"tracking frame window [{frame_start}, {frame_stop}) selects no frames"
        )

    charges = cylinder.NEIGHBORS - counts[:n_frames]
    frames = build_defect_frames(coords[:n_frames], charges, field_steps[:n_frames], selected)
    table = track_defect_frames(
        frames,
        float(case.lx),
        match_tol=float(constants.match_tolerance),
        timestep=float(cylinder.TIMESTEP),
    )
    save_event_metric_npz(
        output_path,
        (case,),
        "defect_tracks",
        defect_track_values(table),
        frame_start=int(selected[0]),
        frame_stop=int(selected[-1]) + 1,
    )
    return table
