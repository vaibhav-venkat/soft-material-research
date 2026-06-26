from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from ..shared import CellList, build_cell_list
from ...common import FRAME_START, FRAME_STOP
from .constants import DEFAULT_EVENT_CONSTANTS, EventAnalysisConstants
from .geometry import cylindrical_to_cartesian, minimum_image_x_delta
from .io import event_metric_npz_path, save_event_metric_npz
from .tracking import (
    DefectTrackTable,
    defect_track_table_from_values,
    track_persistent_defects,
)


@dataclass(frozen=True)
class DefectEventTable:
    birth_track_id: np.ndarray
    birth_frames: np.ndarray
    birth_steps: np.ndarray
    birth_positions: np.ndarray
    birth_charge: np.ndarray
    birth_nearest_opposite_track_id: np.ndarray
    birth_nearest_opposite_distance: np.ndarray
    birth_nearest_same_track_id: np.ndarray
    birth_nearest_same_distance: np.ndarray
    death_track_id: np.ndarray
    death_frames: np.ndarray
    death_steps: np.ndarray
    death_positions: np.ndarray
    death_charge: np.ndarray
    death_nearest_opposite_track_id: np.ndarray
    death_nearest_opposite_distance: np.ndarray
    death_nearest_same_track_id: np.ndarray
    death_nearest_same_distance: np.ndarray
    death_annihilated: np.ndarray
    death_partner_track_id: np.ndarray
    track_annihilation_flag: np.ndarray
    track_annihilation_partner_id: np.ndarray
    ambiguous_birth_track_count: int
    ambiguous_death_track_count: int


@dataclass(frozen=True)
class FrameTrackLookup:
    frame: int
    track_indices: np.ndarray
    charges: np.ndarray
    cartesian_positions: np.ndarray
    cell_list: CellList
    max_query_radius: float


def _frame_offset(table: DefectTrackTable, frame: int) -> int:
    offset = int(frame) - int(table.frame_axis[0])
    if offset < 0 or offset >= table.frame_axis.size:
        raise ValueError(f"frame {frame} is outside tracked frame axis")
    return offset


def _track_lifetime_frames(table: DefectTrackTable, track_idx: int) -> int:
    return int(table.frame_stop[track_idx] - table.frame_start[track_idx])


def _present_track_indices(table: DefectTrackTable, frame: int) -> np.ndarray:
    offset = _frame_offset(table, frame)
    return np.flatnonzero(table.particle_index[:, offset] >= 0).astype(np.int64)


def _cartesian_distance(first: np.ndarray, second: np.ndarray, box_length_x: float) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    delta[0] = minimum_image_x_delta(delta[0], box_length_x)
    return float(np.linalg.norm(delta))


def _track_distance_at_frame(
    table: DefectTrackTable,
    first_track_idx: int,
    second_track_idx: int,
    frame: int,
    box_length_x: float,
) -> float:
    offset = _frame_offset(table, frame)
    first = cylindrical_to_cartesian(table.positions[first_track_idx, offset][np.newaxis, :])[0]
    second = cylindrical_to_cartesian(table.positions[second_track_idx, offset][np.newaxis, :])[0]
    return _cartesian_distance(first, second, box_length_x)


def _build_frame_track_lookups(
    table: DefectTrackTable,
    box_length_x: float,
    cell_size: float,
) -> dict[int, FrameTrackLookup]:
    lookups: dict[int, FrameTrackLookup] = {}
    cell_size = float(cell_size)
    for frame in table.frame_axis:
        frame_int = int(frame)
        offset = _frame_offset(table, frame_int)
        present = np.flatnonzero(table.particle_index[:, offset] >= 0).astype(np.int64)
        if present.size == 0:
            continue
        cartesian = cylindrical_to_cartesian(table.positions[present, offset])
        finite = cartesian[np.all(np.isfinite(cartesian), axis=1)]
        radial_extent = float(np.max(np.abs(finite[:, 1:]))) if finite.size else cell_size
        cell_list = build_cell_list(
            cartesian,
            cell_size=cell_size,
            box_length_x=box_length_x,
            y_min=-radial_extent - cell_size,
            y_max=radial_extent + cell_size,
            z_min=-radial_extent - cell_size,
            z_max=radial_extent + cell_size,
        )
        max_query_radius = float(
            np.sqrt((0.5 * max(float(box_length_x), 0.0)) ** 2 + (2.0 * radial_extent) ** 2)
        )
        lookups[frame_int] = FrameTrackLookup(
            frame=frame_int,
            track_indices=present,
            charges=table.charge[present],
            cartesian_positions=cartesian,
            cell_list=cell_list,
            max_query_radius=max(max_query_radius, cell_size),
        )
    return lookups


def _nearest_tracks(
    table: DefectTrackTable,
    frame: int,
    target_track_idx: int,
    box_length_x: float,
    *,
    lookups: dict[int, FrameTrackLookup] | None = None,
    initial_query_radius: float = DEFAULT_EVENT_CONSTANTS.match_tolerance,
) -> tuple[int, float, int, float]:
    lookup = None if lookups is None else lookups.get(int(frame))
    if lookup is None:
        present = _present_track_indices(table, frame)
        present = present[present != target_track_idx]
        if present.size == 0:
            return -1, np.nan, -1, np.nan

        offset = _frame_offset(table, frame)
        target = cylindrical_to_cartesian(table.positions[target_track_idx, offset][np.newaxis, :])
        candidates = cylindrical_to_cartesian(table.positions[present, offset])
        distances = np.asarray(
            [_cartesian_distance(candidate, target[0], box_length_x) for candidate in candidates],
            dtype=np.float64,
        )
        charges = table.charge[present]
        return _nearest_from_distances(table, present, charges, distances, target_track_idx)

    if lookup.track_indices.size <= 1:
        return -1, np.nan, -1, np.nan

    offset = _frame_offset(table, frame)
    target = cylindrical_to_cartesian(table.positions[target_track_idx, offset][np.newaxis, :])[0]
    best_by_local: dict[int, float] = {}
    radius = max(float(initial_query_radius), lookup.cell_list.cell_size)
    while radius <= lookup.max_query_radius * 2.0:
        for local_idx in lookup.cell_list.iter_neighbor_indices(target, radius):
            track_idx = int(lookup.track_indices[local_idx])
            if track_idx == target_track_idx:
                continue
            distance = _cartesian_distance(
                lookup.cartesian_positions[local_idx],
                target,
                box_length_x,
            )
            if distance <= radius:
                best_by_local[local_idx] = distance
        selected = np.asarray(sorted(best_by_local), dtype=np.int64)
        if selected.size:
            present = lookup.track_indices[selected]
            charges = lookup.charges[selected]
            distances = np.asarray([best_by_local[int(idx)] for idx in selected], dtype=np.float64)
            nearest = _nearest_from_distances(table, present, charges, distances, target_track_idx)
            if nearest[0] >= 0 and nearest[2] >= 0:
                return nearest
            if radius >= lookup.max_query_radius:
                return nearest
        radius *= 2.0
    return -1, np.nan, -1, np.nan


def _nearest_from_distances(
    table: DefectTrackTable,
    present: np.ndarray,
    charges: np.ndarray,
    distances: np.ndarray,
    target_track_idx: int,
) -> tuple[int, float, int, float]:
    target_charge = int(table.charge[target_track_idx])
    same_mask = charges == target_charge
    opposite_mask = charges == -target_charge

    same_track_id = -1
    same_distance = np.nan
    if np.any(same_mask):
        same_candidates = np.flatnonzero(same_mask)
        best = same_candidates[np.argmin(distances[same_mask])]
        same_track_id = int(table.track_id[present[best]])
        same_distance = float(distances[best])

    opposite_track_id = -1
    opposite_distance = np.nan
    if np.any(opposite_mask):
        opposite_candidates = np.flatnonzero(opposite_mask)
        best = opposite_candidates[np.argmin(distances[opposite_mask])]
        opposite_track_id = int(table.track_id[present[best]])
        opposite_distance = float(distances[best])

    return opposite_track_id, opposite_distance, same_track_id, same_distance


def _minimum_track_distance_near_death(
    table: DefectTrackTable,
    first_track_idx: int,
    second_track_idx: int,
    box_length_x: float,
    frame_window: int,
) -> tuple[float, int]:
    first_death = int(table.frame_stop[first_track_idx] - 1)
    second_death = int(table.frame_stop[second_track_idx] - 1)
    start = max(int(table.frame_axis[0]), min(first_death, second_death) - frame_window)
    stop = min(int(table.frame_axis[-1]), max(first_death, second_death)) + 1
    best_distance = np.inf
    best_frame = -1
    for frame in range(start, stop):
        offset = _frame_offset(table, frame)
        if (
            table.particle_index[first_track_idx, offset] < 0
            or table.particle_index[second_track_idx, offset] < 0
        ):
            continue
        distance = _track_distance_at_frame(
            table,
            first_track_idx,
            second_track_idx,
            frame,
            box_length_x,
        )
        if distance < best_distance:
            best_distance = distance
            best_frame = frame
    return best_distance, best_frame


def _annihilation_pairs(
    table: DefectTrackTable,
    death_track_indices: np.ndarray,
    box_length_x: float,
    *,
    distance_threshold: float,
    frame_window: int,
    lookups: dict[int, FrameTrackLookup] | None = None,
) -> dict[int, int]:
    candidates: list[tuple[float, int, int, int, int]] = []
    death_set = {int(idx) for idx in death_track_indices}
    considered: set[tuple[int, int]] = set()
    for left_idx in death_track_indices:
        left_idx = int(left_idx)
        left_death = int(table.frame_stop[left_idx] - 1)
        frame_start = max(int(table.frame_axis[0]), left_death - frame_window)
        frame_stop = min(int(table.frame_axis[-1]), left_death + frame_window) + 1
        local_opposite: set[int] = set()
        for frame in range(frame_start, frame_stop):
            offset = _frame_offset(table, frame)
            if table.particle_index[left_idx, offset] < 0:
                continue
            lookup = None if lookups is None else lookups.get(frame)
            if lookup is None:
                present = _present_track_indices(table, frame)
                for right_idx in present:
                    if int(right_idx) == left_idx:
                        continue
                    if int(right_idx) not in death_set:
                        continue
                    if int(table.charge[left_idx]) == int(table.charge[right_idx]):
                        continue
                    distance = _track_distance_at_frame(
                        table,
                        left_idx,
                        int(right_idx),
                        frame,
                        box_length_x,
                    )
                    if distance <= distance_threshold:
                        local_opposite.add(int(right_idx))
                continue

            target = cylindrical_to_cartesian(table.positions[left_idx, offset][np.newaxis, :])[0]
            for local_idx in lookup.cell_list.iter_neighbor_indices(target, distance_threshold):
                right_idx = int(lookup.track_indices[local_idx])
                if right_idx == left_idx or right_idx not in death_set:
                    continue
                if int(table.charge[left_idx]) == int(table.charge[right_idx]):
                    continue
                distance = _cartesian_distance(
                    lookup.cartesian_positions[local_idx],
                    target,
                    box_length_x,
                )
                if distance <= distance_threshold:
                    local_opposite.add(right_idx)

        for right_idx in local_opposite:
            pair = tuple(sorted((left_idx, int(right_idx))))
            if pair in considered:
                continue
            considered.add(pair)
            if int(table.charge[left_idx]) == int(table.charge[right_idx]):
                continue
            death_gap = abs(int(table.frame_stop[left_idx]) - int(table.frame_stop[right_idx]))
            if death_gap > frame_window:
                continue
            min_distance, min_frame = _minimum_track_distance_near_death(
                table,
                int(left_idx),
                int(right_idx),
                box_length_x,
                frame_window,
            )
            if min_distance <= distance_threshold:
                first_id = int(table.track_id[left_idx])
                second_id = int(table.track_id[right_idx])
                candidates.append(
                    (
                        float(min_distance),
                        int(death_gap),
                        int(min_frame),
                        min(first_id, second_id),
                        max(first_id, second_id),
                    )
                )

    partner_by_track_id: dict[int, int] = {}
    for _, _, _, first_id, second_id in sorted(candidates):
        if first_id in partner_by_track_id or second_id in partner_by_track_id:
            continue
        partner_by_track_id[first_id] = second_id
        partner_by_track_id[second_id] = first_id
    return partner_by_track_id


def classify_defect_events(
    table: DefectTrackTable,
    box_length_x: float,
    *,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
    annihilation_frame_window: int = 2,
) -> DefectEventTable:
    """Classify persistent track starts/stops as birth, death, and annihilation events."""
    if table.n_frames == 0:
        raise ValueError("cannot classify events for an empty frame axis")

    first_frame = int(table.frame_axis[0])
    final_frame = int(table.frame_axis[-1])
    persistence_frames = int(constants.persistence_frames)
    persistent = np.asarray(
        [
            _track_lifetime_frames(table, track_idx) >= persistence_frames
            for track_idx in range(table.n_tracks)
        ],
        dtype=bool,
    )

    birth_rows: list[tuple[int, int, int, np.ndarray, int, int, float, int, float]] = []
    death_track_indices: list[int] = []
    death_base_rows: list[tuple[int, int, int, np.ndarray, int, int, float, int, float]] = []
    lookups = _build_frame_track_lookups(
        table,
        box_length_x,
        cell_size=float(constants.match_tolerance),
    )

    for track_idx in range(table.n_tracks):
        if not persistent[track_idx]:
            continue
        birth_frame = int(table.frame_start[track_idx])
        if birth_frame > first_frame and not bool(table.birth_ambiguous[track_idx]):
            birth_offset = _frame_offset(table, birth_frame)
            opp_id, opp_distance, same_id, same_distance = _nearest_tracks(
                table,
                birth_frame,
                track_idx,
                box_length_x,
                lookups=lookups,
                initial_query_radius=float(constants.match_tolerance),
            )
            birth_rows.append(
                (
                    int(table.track_id[track_idx]),
                    birth_frame,
                    int(table.steps[birth_offset]),
                    table.positions[track_idx, birth_offset].copy(),
                    int(table.charge[track_idx]),
                    opp_id,
                    opp_distance,
                    same_id,
                    same_distance,
                )
            )

        death_frame = int(table.frame_stop[track_idx] - 1)
        if death_frame < final_frame and not bool(table.death_ambiguous[track_idx]):
            death_offset = _frame_offset(table, death_frame)
            opp_id, opp_distance, same_id, same_distance = _nearest_tracks(
                table,
                death_frame,
                track_idx,
                box_length_x,
                lookups=lookups,
                initial_query_radius=float(constants.match_tolerance),
            )
            death_track_indices.append(track_idx)
            death_base_rows.append(
                (
                    int(table.track_id[track_idx]),
                    death_frame,
                    int(table.steps[death_offset]),
                    table.positions[track_idx, death_offset].copy(),
                    int(table.charge[track_idx]),
                    opp_id,
                    opp_distance,
                    same_id,
                    same_distance,
                )
            )

    death_track_array = np.asarray(death_track_indices, dtype=np.int64)
    partner_by_track_id = _annihilation_pairs(
        table,
        death_track_array,
        box_length_x,
        distance_threshold=2.5 * float(constants.neighbor_count_radius),
        frame_window=annihilation_frame_window,
        lookups=lookups,
    )

    track_annihilation_flag = np.zeros(table.n_tracks, dtype=bool)
    track_annihilation_partner_id = np.full(table.n_tracks, -1, dtype=np.int64)
    track_id_to_idx = {int(track_id): idx for idx, track_id in enumerate(table.track_id)}
    for track_id, partner_id in partner_by_track_id.items():
        track_idx = track_id_to_idx[track_id]
        track_annihilation_flag[track_idx] = True
        track_annihilation_partner_id[track_idx] = int(partner_id)

    death_annihilated = []
    death_partner = []
    for row in death_base_rows:
        track_id = row[0]
        partner_id = partner_by_track_id.get(track_id, -1)
        death_annihilated.append(partner_id >= 0)
        death_partner.append(partner_id)

    return _event_table_from_rows(
        birth_rows,
        death_base_rows,
        np.asarray(death_annihilated, dtype=bool),
        np.asarray(death_partner, dtype=np.int64),
        track_annihilation_flag,
        track_annihilation_partner_id,
        ambiguous_birth_track_count=int(np.count_nonzero(table.birth_ambiguous)),
        ambiguous_death_track_count=int(np.count_nonzero(table.death_ambiguous)),
    )


def _event_table_from_rows(
    birth_rows: list[tuple[int, int, int, np.ndarray, int, int, float, int, float]],
    death_rows: list[tuple[int, int, int, np.ndarray, int, int, float, int, float]],
    death_annihilated: np.ndarray,
    death_partner: np.ndarray,
    track_annihilation_flag: np.ndarray,
    track_annihilation_partner_id: np.ndarray,
    *,
    ambiguous_birth_track_count: int,
    ambiguous_death_track_count: int,
) -> DefectEventTable:
    def unpack_int(rows, index: int) -> np.ndarray:
        return np.asarray([row[index] for row in rows], dtype=np.int64)

    def unpack_float(rows, index: int) -> np.ndarray:
        return np.asarray([row[index] for row in rows], dtype=np.float64)

    def unpack_positions(rows) -> np.ndarray:
        if not rows:
            return np.full((0, 3), np.nan, dtype=np.float64)
        return np.asarray([row[3] for row in rows], dtype=np.float64)

    return DefectEventTable(
        birth_track_id=unpack_int(birth_rows, 0),
        birth_frames=unpack_int(birth_rows, 1),
        birth_steps=unpack_int(birth_rows, 2),
        birth_positions=unpack_positions(birth_rows),
        birth_charge=unpack_int(birth_rows, 4),
        birth_nearest_opposite_track_id=unpack_int(birth_rows, 5),
        birth_nearest_opposite_distance=unpack_float(birth_rows, 6),
        birth_nearest_same_track_id=unpack_int(birth_rows, 7),
        birth_nearest_same_distance=unpack_float(birth_rows, 8),
        death_track_id=unpack_int(death_rows, 0),
        death_frames=unpack_int(death_rows, 1),
        death_steps=unpack_int(death_rows, 2),
        death_positions=unpack_positions(death_rows),
        death_charge=unpack_int(death_rows, 4),
        death_nearest_opposite_track_id=unpack_int(death_rows, 5),
        death_nearest_opposite_distance=unpack_float(death_rows, 6),
        death_nearest_same_track_id=unpack_int(death_rows, 7),
        death_nearest_same_distance=unpack_float(death_rows, 8),
        death_annihilated=np.asarray(death_annihilated, dtype=bool),
        death_partner_track_id=np.asarray(death_partner, dtype=np.int64),
        track_annihilation_flag=np.asarray(track_annihilation_flag, dtype=bool),
        track_annihilation_partner_id=np.asarray(
            track_annihilation_partner_id,
            dtype=np.int64,
        ),
        ambiguous_birth_track_count=int(ambiguous_birth_track_count),
        ambiguous_death_track_count=int(ambiguous_death_track_count),
    )


def defect_event_values(events: DefectEventTable) -> dict[str, np.ndarray]:
    return {
        "birth_track_id": events.birth_track_id,
        "birth_frames": events.birth_frames,
        "birth_steps": events.birth_steps,
        "birth_x": events.birth_positions[:, 0],
        "birth_theta": events.birth_positions[:, 1],
        "birth_r": events.birth_positions[:, 2],
        "birth_charge": events.birth_charge,
        "birth_nearest_opp_track_id": events.birth_nearest_opposite_track_id,
        "birth_nearest_opp_dist": events.birth_nearest_opposite_distance,
        "birth_nearest_same_track_id": events.birth_nearest_same_track_id,
        "birth_nearest_same_dist": events.birth_nearest_same_distance,
        "death_track_id": events.death_track_id,
        "death_frames": events.death_frames,
        "death_steps": events.death_steps,
        "death_x": events.death_positions[:, 0],
        "death_theta": events.death_positions[:, 1],
        "death_r": events.death_positions[:, 2],
        "death_charge": events.death_charge,
        "death_nearest_opp_track_id": events.death_nearest_opposite_track_id,
        "death_nearest_opp_dist": events.death_nearest_opposite_distance,
        "death_nearest_same_track_id": events.death_nearest_same_track_id,
        "death_nearest_same_dist": events.death_nearest_same_distance,
        "death_annihilated": events.death_annihilated,
        "death_partner_track_id": events.death_partner_track_id,
        "death_partner_idx": events.death_partner_track_id,
        "track_annihilation_flag": events.track_annihilation_flag,
        "track_annihilation_partner_id": events.track_annihilation_partner_id,
        "ambiguous_birth_track_count": np.asarray(events.ambiguous_birth_track_count),
        "ambiguous_death_track_count": np.asarray(events.ambiguous_death_track_count),
    }


def defect_event_table_from_values(values: dict[str, np.ndarray]) -> DefectEventTable:
    birth_positions = np.column_stack(
        (
            np.asarray(values["birth_x"], dtype=np.float64),
            np.asarray(values["birth_theta"], dtype=np.float64),
            np.asarray(values["birth_r"], dtype=np.float64),
        )
    )
    death_positions = np.column_stack(
        (
            np.asarray(values["death_x"], dtype=np.float64),
            np.asarray(values["death_theta"], dtype=np.float64),
            np.asarray(values["death_r"], dtype=np.float64),
        )
    )
    return DefectEventTable(
        birth_track_id=np.asarray(values["birth_track_id"], dtype=np.int64),
        birth_frames=np.asarray(values["birth_frames"], dtype=np.int64),
        birth_steps=np.asarray(values["birth_steps"], dtype=np.int64),
        birth_positions=birth_positions,
        birth_charge=np.asarray(values["birth_charge"], dtype=np.int64),
        birth_nearest_opposite_track_id=np.asarray(
            values["birth_nearest_opp_track_id"],
            dtype=np.int64,
        ),
        birth_nearest_opposite_distance=np.asarray(
            values["birth_nearest_opp_dist"],
            dtype=np.float64,
        ),
        birth_nearest_same_track_id=np.asarray(
            values["birth_nearest_same_track_id"],
            dtype=np.int64,
        ),
        birth_nearest_same_distance=np.asarray(
            values["birth_nearest_same_dist"],
            dtype=np.float64,
        ),
        death_track_id=np.asarray(values["death_track_id"], dtype=np.int64),
        death_frames=np.asarray(values["death_frames"], dtype=np.int64),
        death_steps=np.asarray(values["death_steps"], dtype=np.int64),
        death_positions=death_positions,
        death_charge=np.asarray(values["death_charge"], dtype=np.int64),
        death_nearest_opposite_track_id=np.asarray(
            values["death_nearest_opp_track_id"],
            dtype=np.int64,
        ),
        death_nearest_opposite_distance=np.asarray(
            values["death_nearest_opp_dist"],
            dtype=np.float64,
        ),
        death_nearest_same_track_id=np.asarray(
            values["death_nearest_same_track_id"],
            dtype=np.int64,
        ),
        death_nearest_same_distance=np.asarray(
            values["death_nearest_same_dist"],
            dtype=np.float64,
        ),
        death_annihilated=np.asarray(values["death_annihilated"], dtype=bool),
        death_partner_track_id=np.asarray(values["death_partner_track_id"], dtype=np.int64),
        track_annihilation_flag=np.asarray(values["track_annihilation_flag"], dtype=bool),
        track_annihilation_partner_id=np.asarray(
            values["track_annihilation_partner_id"],
            dtype=np.int64,
        ),
        ambiguous_birth_track_count=int(np.asarray(values["ambiguous_birth_track_count"])),
        ambiguous_death_track_count=int(np.asarray(values["ambiguous_death_track_count"])),
    )


def load_defect_events_npz(path: str | Path) -> DefectEventTable:
    with np.load(path) as data:
        values = {name: np.asarray(data[name]) for name in data.files}
    return defect_event_table_from_values(values)


def classify_defect_events_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    *,
    overwrite: bool = False,
    track_table: DefectTrackTable | None = None,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> DefectEventTable:
    output_path = event_metric_npz_path("defect_events", case)
    if output_path.exists() and not overwrite:
        return load_defect_events_npz(output_path)

    if track_table is None:
        track_path = event_metric_npz_path("defect_tracks", case)
        if track_path.exists() and not overwrite:
            with np.load(track_path) as data:
                track_values = {name: np.asarray(data[name]) for name in data.files}
            track_table = defect_track_table_from_values(track_values)
        else:
            track_table = track_persistent_defects(
                case,
                frame_start,
                frame_stop,
                overwrite=overwrite,
                constants=constants,
            )
    events = classify_defect_events(
        track_table,
        float(case.lx),
        constants=constants,
    )
    save_event_metric_npz(
        output_path,
        (case,),
        "defect_events",
        defect_event_values(events),
        frame_start=int(track_table.frame_axis[0]),
        frame_stop=int(track_table.frame_axis[-1]) + 1,
    )
    return events
