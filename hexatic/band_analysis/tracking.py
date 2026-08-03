from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jaccard

from .characterization import (
    BandCharacterization,
    align_to_temporal_image,
    minimum_image,
)


class EventCode(IntEnum):
    NONE = 0
    BIRTH = 1
    DISAPPEARANCE = 2
    SPLIT = 3
    MERGE = 4
    UNCERTAIN_MATCH = 5


@dataclass(frozen=True)
class EventRecord:
    code: EventCode
    old_track_id: int = -1
    new_track_id: int = -1
    jaccard: float = float("nan")


@dataclass(frozen=True)
class TrackedBand:
    track_id: int
    segment_id: int
    characterization: BandCharacterization
    assignment_jaccard: float
    event_code: EventCode
    event_boundary: bool


@dataclass(frozen=True)
class TrackedFrame:
    frame_index: int
    step: int
    bands: tuple[TrackedBand, ...]
    events: tuple[EventRecord, ...] = ()


@dataclass
class _TrackState:
    unwrapped_position: float
    first_position: float
    segment_id: int
    mask: np.ndarray


class BandTracker:
    """Track periodic bands using mask overlap and axial-center tie breaking."""

    def __init__(self, lx: float, *, overlap_threshold: float = 0.05) -> None:
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must lie in [0, 1]")
        self.lx = lx
        self.overlap_threshold = overlap_threshold
        self._active: dict[int, _TrackState] = {}
        self._next_track_id = 0
        self._next_segment_id = 0

    def _new_segment(self) -> int:
        segment_id = self._next_segment_id
        self._next_segment_id += 1
        return segment_id

    def _new_track(
        self,
        band: BandCharacterization,
        mask: np.ndarray,
        *,
        event_code: EventCode,
    ) -> TrackedBand:
        track_id = self._next_track_id
        self._next_track_id += 1
        segment_id = self._new_segment()
        position = band.wrapped_axial_position
        self._active[track_id] = _TrackState(
            position, position, segment_id, mask.copy()
        )
        return TrackedBand(
            track_id,
            segment_id,
            align_to_temporal_image(band, position, position),
            float("nan"),
            event_code,
            True,
        )

    @staticmethod
    def _overlap(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            raise ValueError("tracked masks must have identical shapes")
        distance = float(jaccard(first.ravel(), second.ravel()))
        return 1.0 - distance

    def update(
        self,
        bands: list[BandCharacterization],
        masks: list[np.ndarray],
    ) -> tuple[list[TrackedBand], tuple[EventRecord, ...]]:
        if len(bands) != len(masks):
            raise ValueError("bands and masks must have equal length")
        masks = [np.asarray(mask, dtype=bool) for mask in masks]

        if not self._active:
            tracked = [
                self._new_track(band, mask, event_code=EventCode.BIRTH)
                for band, mask in zip(bands, masks, strict=True)
            ]
            events = tuple(
                EventRecord(EventCode.BIRTH, new_track_id=item.track_id)
                for item in tracked
            )
            return tracked, events

        old_ids = list(self._active)
        if not bands:
            events = tuple(
                EventRecord(EventCode.DISAPPEARANCE, old_track_id=track_id)
                for track_id in old_ids
            )
            self._active = {}
            return [], events

        overlaps = np.empty((len(old_ids), len(bands)), dtype=float)
        distances = np.empty_like(overlaps)
        for row, track_id in enumerate(old_ids):
            state = self._active[track_id]
            for column, (band, mask) in enumerate(
                zip(bands, masks, strict=True)
            ):
                overlaps[row, column] = self._overlap(state.mask, mask)
                distances[row, column] = abs(
                    minimum_image(
                        band.wrapped_axial_position - state.unwrapped_position,
                        self.lx,
                    )
                )

        # Jaccard is the primary cost. The small normalized distance term makes
        # equal-overlap and zero-overlap assignments deterministic.
        costs = 1.0 - overlaps + 1.0e-6 * distances / self.lx
        rows, columns = linear_sum_assignment(costs)
        assigned_by_new = {
            int(column): int(row) for row, column in zip(rows, columns, strict=True)
        }
        assigned_old = {int(row) for row in rows}
        event_edges = overlaps >= self.overlap_threshold
        old_degree = np.sum(event_edges, axis=1)
        new_degree = np.sum(event_edges, axis=0)

        previous = self._active
        self._active = {}
        tracked: list[TrackedBand] = []
        events: list[EventRecord] = []
        for column, (band, mask) in enumerate(zip(bands, masks, strict=True)):
            if column not in assigned_by_new:
                event_code = (
                    EventCode.SPLIT if new_degree[column] > 0 else EventCode.BIRTH
                )
                item = self._new_track(band, mask, event_code=event_code)
                tracked.append(item)
                events.append(
                    EventRecord(event_code, new_track_id=item.track_id)
                )
                continue

            row = assigned_by_new[column]
            track_id = old_ids[row]
            old_state = previous[track_id]
            overlap = float(overlaps[row, column])
            if new_degree[column] >= 2:
                event_code = EventCode.MERGE
            elif old_degree[row] >= 2:
                event_code = EventCode.SPLIT
            elif overlap < self.overlap_threshold:
                event_code = EventCode.UNCERTAIN_MATCH
            else:
                event_code = EventCode.NONE
            boundary = event_code != EventCode.NONE
            segment_id = self._new_segment() if boundary else old_state.segment_id
            position = old_state.unwrapped_position + minimum_image(
                band.wrapped_axial_position - old_state.unwrapped_position,
                self.lx,
            )
            self._active[track_id] = _TrackState(
                position,
                old_state.first_position,
                segment_id,
                mask.copy(),
            )
            tracked.append(
                TrackedBand(
                    track_id,
                    segment_id,
                    align_to_temporal_image(
                        band, position, old_state.first_position
                    ),
                    overlap,
                    event_code,
                    boundary,
                )
            )
            if boundary:
                events.append(
                    EventRecord(event_code, track_id, track_id, overlap)
                )

        for row, track_id in enumerate(old_ids):
            if row in assigned_old:
                continue
            event_code = (
                EventCode.MERGE if old_degree[row] > 0 else EventCode.DISAPPEARANCE
            )
            events.append(EventRecord(event_code, old_track_id=track_id))

        tracked.sort(key=lambda item: item.track_id)
        return tracked, tuple(events)
