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
    initial_areas: np.ndarray
    lifetimes: np.ndarray
    split_lifetimes: np.ndarray
    merge_lifetimes: np.ndarray


def track_histories(frames: list[TrackedFrame]) -> list[TrackHistory]:
    if not frames:
        return []
    samples: dict[int, list[tuple[float, float]]] = {}
    born: set[int] = set()
    terminals: dict[int, tuple[float, EventCode]] = {}
    first_frame = frames[0].frame_index
    for frame in frames:
        for band in frame.bands:
            samples.setdefault(band.track_id, []).append((frame.tau, band.area))
        for event in frame.events:
            if event.code in {
                EventCode.DISAPPEARANCE,
                EventCode.SPLIT,
                EventCode.MERGE,
            }:
                terminals.update(
                    {
                        track_id: (frame.tau, event.code)
                        for track_id in event.old_track_ids
                    }
                )
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
    seed_frames: list[list[TrackedFrame]], particle_diameter: float, lag: float = 1.0
) -> BandStatistics:
    histories = [history for frames in seed_frames for history in track_histories(frames)]
    area_scale = particle_diameter**2
    areas = np.concatenate([history.areas for history in histories]) / area_scale
    delta_start: list[np.ndarray] = []
    delta: list[np.ndarray] = []
    passage_area: list[np.ndarray] = []
    passage_time: list[np.ndarray] = []
    initial_area: list[float] = []
    lifetimes: list[float] = []
    split_lifetimes: list[float] = []
    merge_lifetimes: list[float] = []
    for history in histories:
        target = history.tau + lag
        valid = target <= history.tau[-1]
        if np.any(valid):
            future = np.interp(target[valid], history.tau, history.areas)
            delta_start.append(history.areas[valid] / area_scale)
            delta.append((future - history.areas[valid]) / area_scale)
        if history.terminal_event == EventCode.DISAPPEARANCE:
            passage_area.append(history.areas / area_scale)
            assert history.terminal_tau is not None
            passage_time.append(history.terminal_tau - history.tau)
        lifetime = history.lifetime
        if lifetime is not None and lifetime > 0.0:
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
        initial_areas=np.asarray(initial_area),
        lifetimes=np.asarray(lifetimes),
        split_lifetimes=np.asarray(split_lifetimes),
        merge_lifetimes=np.asarray(merge_lifetimes),
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
