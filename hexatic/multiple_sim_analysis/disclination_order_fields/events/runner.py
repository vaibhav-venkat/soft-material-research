from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hexatic.radii_analysis.cases import RadiusCase

from ...common import FRAME_START, FRAME_STOP
from .constants import DEFAULT_EVENT_CONSTANTS, EventAnalysisConstants
from .events import (
    DefectEventTable,
    classify_defect_events_for_case,
    defect_event_values,
    load_defect_events_npz,
)
from .geometry import cylindrical_to_cartesian, cylinder_distances
from .io import event_metric_npz_path, save_event_metric_npz
from .tracking import DefectTrackTable, track_persistent_defects


SUMMARY_VALUE_NAMES = (
    "defect_track_count",
    "birth_event_count",
    "death_event_count",
    "annihilation_event_count",
)


@dataclass(frozen=True)
class ClusterFrameResult:
    labels: np.ndarray
    cluster_id: np.ndarray
    size: np.ndarray
    charge: np.ndarray
    total: np.ndarray
    com: np.ndarray
    velocity: np.ndarray


def _periodic_mean(values: np.ndarray, period: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    reference = float(values[0])
    deltas = values - reference
    deltas -= period * np.round(deltas / period)
    return float(reference + np.mean(deltas))


def _circular_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    angle = float(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))
    return angle % (2.0 * np.pi)


def _union_find_labels(n_items: int, edges: list[tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n_items, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left, right in edges:
        union(int(left), int(right))
    roots = np.asarray([find(item) for item in range(n_items)], dtype=np.int64)
    unique_roots = {root: idx for idx, root in enumerate(np.unique(roots))}
    return np.asarray([unique_roots[root] for root in roots], dtype=np.int64)


def cluster_defects_frame(
    positions: np.ndarray,
    charges: np.ndarray,
    box_length_x: float,
    *,
    velocities: np.ndarray | None = None,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> ClusterFrameResult:
    positions = np.asarray(positions, dtype=np.float64)
    charges = np.asarray(charges, dtype=np.int64)
    if positions.shape[0] != charges.shape[0]:
        raise ValueError("positions and charges must contain the same number of defects")
    if positions.size == 0:
        return ClusterFrameResult(
            labels=np.asarray([], dtype=np.int64),
            cluster_id=np.asarray([], dtype=np.int64),
            size=np.asarray([], dtype=np.int64),
            charge=np.asarray([], dtype=np.int64),
            total=np.asarray([], dtype=np.int64),
            com=np.full((0, 3), np.nan, dtype=np.float64),
            velocity=np.full((0, 3), np.nan, dtype=np.float64),
        )

    cartesian = cylindrical_to_cartesian(positions)
    distances = cylinder_distances(cartesian, cartesian, box_length_x)
    edges: list[tuple[int, int]] = []
    for left in range(positions.shape[0]):
        for right in range(left + 1, positions.shape[0]):
            if distances[left, right] < float(constants.cluster_bond_length):
                edges.append((left, right))
    labels = _union_find_labels(positions.shape[0], edges)
    cluster_ids = np.unique(labels)

    cluster_size = np.empty(cluster_ids.size, dtype=np.int64)
    cluster_charge = np.empty(cluster_ids.size, dtype=np.int64)
    cluster_total = np.empty(cluster_ids.size, dtype=np.int64)
    cluster_com = np.full((cluster_ids.size, 3), np.nan, dtype=np.float64)
    cluster_velocity = np.full((cluster_ids.size, 3), np.nan, dtype=np.float64)
    velocities_array = None if velocities is None else np.asarray(velocities, dtype=np.float64)

    for row, cluster_id in enumerate(cluster_ids):
        mask = labels == cluster_id
        cluster_size[row] = int(np.count_nonzero(mask))
        cluster_charge[row] = int(np.sum(charges[mask]))
        cluster_total[row] = cluster_size[row]
        cluster_com[row, 0] = _periodic_mean(positions[mask, 0], box_length_x)
        cluster_com[row, 1] = _circular_mean(positions[mask, 1])
        cluster_com[row, 2] = float(np.mean(positions[mask, 2]))
        if velocities_array is not None:
            selected = velocities_array[mask]
            finite = np.all(np.isfinite(selected), axis=1)
            if np.any(finite):
                cluster_velocity[row] = np.mean(selected[finite], axis=0)

    return ClusterFrameResult(
        labels=labels,
        cluster_id=cluster_ids.astype(np.int64),
        size=cluster_size,
        charge=cluster_charge,
        total=cluster_total,
        com=cluster_com,
        velocity=cluster_velocity,
    )


def _event_counts_by_frame(
    frame_axis: np.ndarray,
    event_frames: np.ndarray,
    event_charges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    plus = np.zeros(frame_axis.size, dtype=np.int64)
    minus = np.zeros(frame_axis.size, dtype=np.int64)
    frame_to_offset = {int(frame): idx for idx, frame in enumerate(frame_axis)}
    for frame, charge in zip(event_frames, event_charges):
        offset = frame_to_offset.get(int(frame))
        if offset is None:
            continue
        if int(charge) > 0:
            plus[offset] += 1
        elif int(charge) < 0:
            minus[offset] += 1
    return plus, minus


def cluster_and_frame_values(
    table: DefectTrackTable,
    events: DefectEventTable,
    box_length_x: float,
    *,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> dict[str, np.ndarray]:
    cluster_frames: list[int] = []
    cluster_ids: list[int] = []
    cluster_sizes: list[int] = []
    cluster_charges: list[int] = []
    cluster_totals: list[int] = []
    cluster_com: list[np.ndarray] = []
    cluster_velocity: list[np.ndarray] = []
    n_plus = np.zeros(table.n_frames, dtype=np.int64)
    n_minus = np.zeros(table.n_frames, dtype=np.int64)
    n_total = np.zeros(table.n_frames, dtype=np.int64)
    cluster_count = np.zeros(table.n_frames, dtype=np.int64)
    cluster_max_size = np.zeros(table.n_frames, dtype=np.int64)
    cluster_mean_size = np.full(table.n_frames, np.nan, dtype=np.float64)

    for offset, frame in enumerate(table.frame_axis):
        present = np.flatnonzero(table.particle_index[:, offset] >= 0)
        if present.size == 0:
            continue
        charges = table.charge[present]
        n_plus[offset] = int(np.count_nonzero(charges > 0))
        n_minus[offset] = int(np.count_nonzero(charges < 0))
        n_total[offset] = int(present.size)
        clusters = cluster_defects_frame(
            table.positions[present, offset],
            charges,
            box_length_x,
            velocities=table.velocity[present, offset],
            constants=constants,
        )
        cluster_count[offset] = int(clusters.cluster_id.size)
        if clusters.size.size:
            cluster_max_size[offset] = int(np.max(clusters.size))
            cluster_mean_size[offset] = float(np.mean(clusters.size))
        for local_cluster_id, size, charge, total, com, velocity in zip(
            clusters.cluster_id,
            clusters.size,
            clusters.charge,
            clusters.total,
            clusters.com,
            clusters.velocity,
        ):
            cluster_frames.append(int(frame))
            cluster_ids.append(int(local_cluster_id))
            cluster_sizes.append(int(size))
            cluster_charges.append(int(charge))
            cluster_totals.append(int(total))
            cluster_com.append(np.asarray(com, dtype=np.float64))
            cluster_velocity.append(np.asarray(velocity, dtype=np.float64))

    birth_plus, birth_minus = _event_counts_by_frame(
        table.frame_axis,
        events.birth_frames,
        events.birth_charge,
    )
    death_plus, death_minus = _event_counts_by_frame(
        table.frame_axis,
        events.death_frames,
        events.death_charge,
    )
    annihilation_plus, annihilation_minus = _event_counts_by_frame(
        table.frame_axis,
        events.death_frames[events.death_annihilated],
        events.death_charge[events.death_annihilated],
    )
    annihilation_total = annihilation_plus + annihilation_minus
    annihilation_top = 0.5 * annihilation_total.astype(np.float64)
    a_top = np.full(table.n_frames, np.nan, dtype=np.float64)
    nonzero = n_total > 0
    a_top[nonzero] = annihilation_top[nonzero] / n_total[nonzero]

    if cluster_com:
        cluster_com_array = np.asarray(cluster_com, dtype=np.float64)
        cluster_velocity_array = np.asarray(cluster_velocity, dtype=np.float64)
    else:
        cluster_com_array = np.full((0, 3), np.nan, dtype=np.float64)
        cluster_velocity_array = np.full((0, 3), np.nan, dtype=np.float64)

    return {
        "frame_axis": table.frame_axis,
        "steps": table.steps,
        "n_plus": n_plus,
        "n_minus": n_minus,
        "n_total": n_total,
        "birth_plus": birth_plus,
        "birth_minus": birth_minus,
        "death_plus": death_plus,
        "death_minus": death_minus,
        "annihilation_plus": annihilation_plus,
        "annihilation_minus": annihilation_minus,
        "annihilation_top": annihilation_top,
        "a_top": a_top,
        "cluster_count": cluster_count,
        "cluster_max_size": cluster_max_size,
        "cluster_mean_size": cluster_mean_size,
        "cluster_frame": np.asarray(cluster_frames, dtype=np.int64),
        "cluster_id": np.asarray(cluster_ids, dtype=np.int64),
        "cluster_size": np.asarray(cluster_sizes, dtype=np.int64),
        "cluster_charge": np.asarray(cluster_charges, dtype=np.int64),
        "cluster_total": np.asarray(cluster_totals, dtype=np.int64),
        "cluster_com_x": cluster_com_array[:, 0],
        "cluster_com_theta": cluster_com_array[:, 1],
        "cluster_com_r": cluster_com_array[:, 2],
        "cluster_velocity_x": cluster_velocity_array[:, 0],
        "cluster_velocity_theta": cluster_velocity_array[:, 1],
        "cluster_velocity_r": cluster_velocity_array[:, 2],
    }


def load_cluster_values_for_case(case: RadiusCase) -> dict[str, np.ndarray]:
    path = event_metric_npz_path("cluster_fields", case)
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def run_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    *,
    overwrite: bool = False,
    constants: EventAnalysisConstants = DEFAULT_EVENT_CONSTANTS,
) -> dict[str, float]:
    tracks = track_persistent_defects(
        case,
        frame_start,
        frame_stop,
        overwrite=overwrite,
        constants=constants,
    )
    events_path = event_metric_npz_path("defect_events", case)
    if events_path.exists() and not overwrite:
        events = load_defect_events_npz(events_path)
    else:
        events = classify_defect_events_for_case(
            case,
            frame_start,
            frame_stop,
            overwrite=overwrite,
            track_table=tracks,
            constants=constants,
        )

    cluster_path = event_metric_npz_path("cluster_fields", case)
    if cluster_path.exists() and not overwrite:
        cluster_values = load_cluster_values_for_case(case)
    else:
        cluster_values = cluster_and_frame_values(
            tracks,
            events,
            float(case.lx),
            constants=constants,
        )
        save_event_metric_npz(
            cluster_path,
            (case,),
            "cluster_fields",
            cluster_values,
            frame_start=int(tracks.frame_axis[0]),
            frame_stop=int(tracks.frame_axis[-1]) + 1,
        )

    event_values = defect_event_values(events)
    return {
        "defect_track_count": float(tracks.n_tracks),
        "birth_event_count": float(event_values["birth_frames"].size),
        "death_event_count": float(event_values["death_frames"].size),
        "annihilation_event_count": float(np.count_nonzero(event_values["death_annihilated"])),
        "mean_n_total": float(np.nanmean(cluster_values["n_total"]))
        if cluster_values["n_total"].size
        else np.nan,
    }


def run(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    values = {name: [] for name in SUMMARY_VALUE_NAMES}
    for case in cases:
        summary = run_case(
            case,
            frame_start,
            frame_stop,
            overwrite=overwrite,
        )
        for name in SUMMARY_VALUE_NAMES:
            values[name].append(summary[name])
    return {name: np.asarray(series, dtype=np.float64) for name, series in values.items()}
