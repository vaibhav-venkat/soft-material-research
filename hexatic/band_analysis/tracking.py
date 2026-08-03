from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .characterization import (
    BandCharacterization,
    align_to_temporal_image,
    minimum_image,
)


@dataclass(frozen=True)
class TrackedBand:
    track_id: int
    characterization: BandCharacterization


@dataclass(frozen=True)
class TrackedFrame:
    frame_index: int
    step: int
    bands: tuple[TrackedBand, ...]


@dataclass
class _TrackState:
    unwrapped_position: float
    first_position: float


class BandTracker:
    """Track bands by periodic axial center, resetting on merges or splits."""

    def __init__(self, lx: float) -> None:
        self.lx = lx
        self._active: dict[int, _TrackState] = {}
        self._next_track_id = 0

    def _start_tracks(
        self, bands: list[BandCharacterization]
    ) -> list[TrackedBand]:
        self._active = {}
        tracked = []
        for band in bands:
            track_id = self._next_track_id
            self._next_track_id += 1
            position = band.wrapped_axial_position
            self._active[track_id] = _TrackState(position, position)
            tracked.append(
                TrackedBand(
                    track_id,
                    align_to_temporal_image(band, position, position),
                )
            )
        return tracked

    def update(self, bands: list[BandCharacterization]) -> list[TrackedBand]:
        if len(bands) != len(self._active):
            return self._start_tracks(bands)
        if not bands:
            return []

        track_ids = list(self._active)
        costs = np.empty((len(track_ids), len(bands)))
        for row, track_id in enumerate(track_ids):
            previous = self._active[track_id].unwrapped_position
            for column, band in enumerate(bands):
                costs[row, column] = abs(
                    minimum_image(band.wrapped_axial_position - previous, self.lx)
                )
        rows, columns = linear_sum_assignment(costs)
        tracked = []
        for row, column in zip(rows, columns, strict=True):
            track_id = track_ids[int(row)]
            band = bands[int(column)]
            state = self._active[track_id]
            position = state.unwrapped_position + minimum_image(
                band.wrapped_axial_position - state.unwrapped_position,
                self.lx,
            )
            state.unwrapped_position = position
            tracked.append(
                TrackedBand(
                    track_id,
                    align_to_temporal_image(band, position, state.first_position),
                )
            )
        tracked.sort(key=lambda item: item.track_id)
        return tracked
