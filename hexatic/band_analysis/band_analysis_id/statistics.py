"""Band-area and lifetime statistics for identification plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import EventCode, TrackedFrame


@dataclass(frozen=True)
class TrackHistory:
    areas: np.ndarray
    tau: np.ndarray
    born: bool
    terminal_tau: float | None
    terminal_event: EventCode | None

    @property
    def lifetime(self) -> float | None:
        if not self.born or self.terminal_tau is None:
            return None
        return self.terminal_tau - float(self.tau[0])


@dataclass(frozen=True)
class BandStatistics:
    areas: np.ndarray
    delta_start_areas: np.ndarray
    delta_areas: np.ndarray
    passage_areas: np.ndarray
    passage_times: np.ndarray
    split_passage_areas: np.ndarray
    split_passage_times: np.ndarray
    merge_passage_areas: np.ndarray
    merge_passage_times: np.ndarray
    initial_areas: np.ndarray
    lifetimes: np.ndarray
    split_lifetimes: np.ndarray
    merge_lifetimes: np.ndarray
    raw_splits: int
    raw_merges: int


@dataclass(frozen=True)
class TrackingSeries:
    frames: list[TrackedFrame]
    include_from_tau: float = 0.0
    initial_bands_are_born: bool = False


def track_histories(
    frames: list[TrackedFrame], *, initial_bands_are_born: bool = False
) -> list[TrackHistory]:
    if not frames:
        return []
    samples: dict[int, list[tuple[float, float]]] = {}
    born: set[int] = set()
    terminals: dict[int, tuple[float, EventCode]] = {}
    first_frame = frames[0].frame_index
    if initial_bands_are_born:
        born.update(band.track_id for band in frames[0].bands)
    for frame in frames:
        for band in frame.bands:
            samples.setdefault(band.track_id, []).append((frame.tau, band.area))
        retired = {
            track_id for event in frame.events for track_id in event.old_track_ids
        }
        if frame.edges is not None:
            row_degree = np.sum(frame.edges.edges, axis=1)
            column_degree = np.sum(frame.edges.edges, axis=0)
            for row, track_id in enumerate(frame.edges.old_track_ids):
                if track_id not in retired:
                    continue
                columns = np.flatnonzero(frame.edges.edges[row])
                if row_degree[row] > 1:
                    event_code = EventCode.SPLIT
                elif np.any(column_degree[columns] > 1):
                    event_code = EventCode.MERGE
                else:
                    event_code = EventCode.DISAPPEARANCE
                terminals[track_id] = (frame.tau, event_code)
        for event in frame.events:
            if frame.frame_index != first_frame:
                born.update(event.new_track_ids)
    return [
        TrackHistory(
            areas=np.asarray([area for _, area in values]),
            tau=np.asarray([tau for tau, _ in values]),
            born=track_id in born,
            terminal_tau=(terminals[track_id][0] if track_id in terminals else None),
            terminal_event=(terminals[track_id][1] if track_id in terminals else None),
        )
        for track_id, values in samples.items()
    ]


def calculate_statistics(
    series: list[TrackingSeries], particle_diameter: float, lag: float = 1.0
) -> BandStatistics:
    histories = [
        (history, item.include_from_tau)
        for item in series
        for history in track_histories(
            item.frames, initial_bands_are_born=item.initial_bands_are_born
        )
    ]
    event_edges = [
        frame.edges
        for item in series
        for frame in item.frames
        if frame.edges is not None and frame.tau >= item.include_from_tau
    ]
    raw_splits = sum(
        int(np.count_nonzero(np.sum(edges.edges, axis=1) > 1))
        for edges in event_edges
    )
    raw_merges = sum(
        int(np.count_nonzero(np.sum(edges.edges, axis=0) > 1))
        for edges in event_edges
    )
    area_scale = particle_diameter**2
    included_areas = [
        history.areas[history.tau >= include_from]
        for history, include_from in histories
    ]
    areas = np.concatenate(included_areas) / area_scale
    delta_start: list[np.ndarray] = []
    delta: list[np.ndarray] = []
    passage_area: list[np.ndarray] = []
    passage_time: list[np.ndarray] = []
    split_passage_area: list[np.ndarray] = []
    split_passage_time: list[np.ndarray] = []
    merge_passage_area: list[np.ndarray] = []
    merge_passage_time: list[np.ndarray] = []
    initial_area: list[float] = []
    lifetimes: list[float] = []
    split_lifetimes: list[float] = []
    merge_lifetimes: list[float] = []
    for history, include_from in histories:
        target = history.tau + lag
        valid = (history.tau >= include_from) & (target <= history.tau[-1])
        if np.any(valid):
            future = np.interp(target[valid], history.tau, history.areas)
            delta_start.append(history.areas[valid] / area_scale)
            delta.append((future - history.areas[valid]) / area_scale)
        terminal_included = (
            history.terminal_tau is not None
            and history.terminal_tau >= include_from
        )
        if terminal_included and history.terminal_event == EventCode.DISAPPEARANCE:
            observed = history.tau >= include_from
            passage_area.append(history.areas[observed] / area_scale)
            assert history.terminal_tau is not None
            passage_time.append(history.terminal_tau - history.tau[observed])
        elif terminal_included and history.terminal_event == EventCode.SPLIT:
            observed = history.tau >= include_from
            split_passage_area.append(history.areas[observed] / area_scale)
            assert history.terminal_tau is not None
            split_passage_time.append(history.terminal_tau - history.tau[observed])
        elif terminal_included and history.terminal_event == EventCode.MERGE:
            observed = history.tau >= include_from
            merge_passage_area.append(history.areas[observed] / area_scale)
            assert history.terminal_tau is not None
            merge_passage_time.append(history.terminal_tau - history.tau[observed])
        lifetime = history.lifetime
        if terminal_included and lifetime is not None and lifetime > 0.0:
            if history.terminal_event == EventCode.DISAPPEARANCE:
                initial_area.append(float(history.areas[0] / area_scale))
                lifetimes.append(lifetime)
            elif history.terminal_event == EventCode.SPLIT:
                split_lifetimes.append(lifetime)
            elif history.terminal_event == EventCode.MERGE:
                merge_lifetimes.append(lifetime)
    return BandStatistics(
        areas=areas,
        delta_start_areas=np.concatenate(delta_start) if delta_start else np.empty(0),
        delta_areas=np.concatenate(delta) if delta else np.empty(0),
        passage_areas=np.concatenate(passage_area) if passage_area else np.empty(0),
        passage_times=np.concatenate(passage_time) if passage_time else np.empty(0),
        split_passage_areas=(
            np.concatenate(split_passage_area) if split_passage_area else np.empty(0)
        ),
        split_passage_times=(
            np.concatenate(split_passage_time) if split_passage_time else np.empty(0)
        ),
        merge_passage_areas=(
            np.concatenate(merge_passage_area) if merge_passage_area else np.empty(0)
        ),
        merge_passage_times=(
            np.concatenate(merge_passage_time) if merge_passage_time else np.empty(0)
        ),
        initial_areas=np.asarray(initial_area),
        lifetimes=np.asarray(lifetimes),
        split_lifetimes=np.asarray(split_lifetimes),
        merge_lifetimes=np.asarray(merge_lifetimes),
        raw_splits=raw_splits,
        raw_merges=raw_merges,
    )


def binned_moment(
    x: np.ndarray, y: np.ndarray, *, bins: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.histogram_bin_edges(x, bins=bins)
    index = np.digitize(x, edges[1:-1])
    count = np.bincount(index, minlength=bins)
    total = np.bincount(index, weights=y, minlength=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = count > 0
    return centers[valid], total[valid] / count[valid]


def probability_points(
    values: np.ndarray, *, bins: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    density, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[density > 0.0], density[density > 0.0]


def survival_points(lifetimes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = np.concatenate(([0.0], np.unique(np.sort(lifetimes))))
    survival = np.asarray([np.mean(lifetimes > value) for value in tau])
    return tau, survival
