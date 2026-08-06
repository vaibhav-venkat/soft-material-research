"""Band-area and lifetime statistics for identification plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import TrackedFrame


@dataclass(frozen=True)
class TrackHistory:
    areas: np.ndarray
    tau: np.ndarray
    born: bool
    death_tau: float | None

    @property
    def lifetime(self) -> float | None:
        if not self.born or self.death_tau is None:
            return None
        return self.death_tau - float(self.tau[0])


@dataclass(frozen=True)
class BandStatistics:
    areas: np.ndarray
    delta_start_areas: np.ndarray
    delta_areas: np.ndarray
    passage_areas: np.ndarray
    passage_times: np.ndarray
    initial_areas: np.ndarray
    lifetimes: np.ndarray


def track_histories(frames: list[TrackedFrame]) -> list[TrackHistory]:
    if not frames:
        return []
    samples: dict[int, list[tuple[float, float]]] = {}
    born: set[int] = set()
    deaths: dict[int, float] = {}
    first_frame = frames[0].frame_index
    for frame in frames:
        for band in frame.bands:
            samples.setdefault(band.track_id, []).append((frame.tau, band.area))
        for event in frame.events:
            deaths.update({track_id: frame.tau for track_id in event.old_track_ids})
            if frame.frame_index != first_frame:
                born.update(event.new_track_ids)
    return [
        TrackHistory(
            areas=np.asarray([area for _, area in values]),
            tau=np.asarray([tau for tau, _ in values]),
            born=track_id in born,
            death_tau=deaths.get(track_id),
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
    for history in histories:
        target = history.tau + lag
        valid = target <= history.tau[-1]
        if np.any(valid):
            future = np.interp(target[valid], history.tau, history.areas)
            delta_start.append(history.areas[valid] / area_scale)
            delta.append((future - history.areas[valid]) / area_scale)
        if history.death_tau is not None:
            passage_area.append(history.areas / area_scale)
            passage_time.append(history.death_tau - history.tau)
        lifetime = history.lifetime
        if lifetime is not None and lifetime > 0.0:
            initial_area.append(float(history.areas[0] / area_scale))
            lifetimes.append(lifetime)
    return BandStatistics(
        areas=areas,
        delta_start_areas=np.concatenate(delta_start) if delta_start else np.empty(0),
        delta_areas=np.concatenate(delta) if delta else np.empty(0),
        passage_areas=np.concatenate(passage_area) if passage_area else np.empty(0),
        passage_times=np.concatenate(passage_time) if passage_time else np.empty(0),
        initial_areas=np.asarray(initial_area),
        lifetimes=np.asarray(lifetimes),
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
