from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from .characterization import DetectedBand


class EventCode(IntEnum):
    NONE = 0
    BIRTH = 1
    DISAPPEARANCE = 2
    SPLIT = 3
    MERGE = 4
    UNCERTAIN = 5
    TRANSIENT_GAP = 6
    RESTORED = 7


@dataclass(frozen=True)
class DetectionFrame:
    frame_index: int
    step: int
    timestep: float
    bands: tuple[DetectedBand, ...]

    @property
    def tau(self) -> float:
        return float(self.step) * self.timestep


@dataclass(frozen=True)
class EventRecord:
    code: EventCode
    old_track_ids: tuple[int, ...] = ()
    new_track_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrackedBand:
    track_id: int
    area: float
    axial_center: float
    mask: np.ndarray
    event_code: EventCode = EventCode.NONE


@dataclass(frozen=True)
class TrackedFrame:
    frame_index: int
    step: int
    tau: float
    bands: tuple[TrackedBand, ...]
    events: tuple[EventRecord, ...] = ()
    clean: bool = True
    tracking_epoch: int = 0


@dataclass(frozen=True)
class _TrackState:
    band: DetectedBand


class BandTracker:
    """Resolve overlap identities after persistent topology is known."""

    def __init__(
        self,
        *,
        overlap_threshold: float = 0.05,
        persistence_frames: int = 5,
    ) -> None:
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must lie in [0, 1]")
        if persistence_frames < 1:
            raise ValueError("persistence_frames must be positive")
        self.overlap_threshold = overlap_threshold
        self.persistence_frames = persistence_frames
        self._next_track_id = 0

    @staticmethod
    def _overlap(first: np.ndarray, second: np.ndarray) -> float:
        if first.shape != second.shape:
            raise ValueError("tracked masks must have identical shapes")
        union = np.count_nonzero(first | second)
        return float(np.count_nonzero(first & second) / union) if union else 0.0

    def _edges(
        self,
        active: dict[int, _TrackState],
        bands: tuple[DetectedBand, ...],
    ) -> tuple[list[int], np.ndarray]:
        old_ids = sorted(active)
        edges = np.zeros((len(old_ids), len(bands)), dtype=bool)
        for row, track_id in enumerate(old_ids):
            for column, band in enumerate(bands):
                edges[row, column] = (
                    self._overlap(active[track_id].band.mask, band.mask)
                    >= self.overlap_threshold
                )
        return old_ids, edges

    @staticmethod
    def _unique_mapping(edges: np.ndarray) -> dict[int, int] | None:
        if edges.shape[0] != edges.shape[1]:
            return None
        if edges.size == 0:
            return {}
        if not np.all(np.sum(edges, axis=0) == 1):
            return None
        if not np.all(np.sum(edges, axis=1) == 1):
            return None
        return {
            int(row): int(np.flatnonzero(edges[row])[0])
            for row in range(len(edges))
        }

    def _new_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    @staticmethod
    def _tracked(
        track_id: int,
        band: DetectedBand,
        event_code: EventCode = EventCode.NONE,
    ) -> TrackedBand:
        return TrackedBand(
            track_id,
            band.area,
            band.axial_center,
            band.mask,
            event_code,
        )

    def track(self, frames: list[DetectionFrame]) -> list[TrackedFrame]:
        if not frames:
            return []
        self._next_track_id = 0
        epoch = 0
        active: dict[int, _TrackState] = {}
        initial_bands = []
        for band in frames[0].bands:
            track_id = self._new_id()
            active[track_id] = _TrackState(band)
            initial_bands.append(self._tracked(track_id, band, EventCode.BIRTH))
        output = [
            TrackedFrame(
                frames[0].frame_index,
                frames[0].step,
                frames[0].tau,
                tuple(initial_bands),
                (EventRecord(EventCode.BIRTH, new_track_ids=tuple(active)),),
                tracking_epoch=epoch,
            )
        ]

        index = 1
        while index < len(frames):
            frame = frames[index]
            old_ids, edges = self._edges(active, frame.bands)
            mapping = self._unique_mapping(edges)
            if mapping is not None:
                tracked = []
                next_active = {}
                for row, column in mapping.items():
                    track_id = old_ids[row]
                    band = frame.bands[column]
                    tracked.append(self._tracked(track_id, band))
                    next_active[track_id] = _TrackState(band)
                active = next_active
                output.append(
                    TrackedFrame(
                        frame.frame_index,
                        frame.step,
                        frame.tau,
                        tuple(sorted(tracked, key=lambda item: item.track_id)),
                        tracking_epoch=epoch,
                    )
                )
                index += 1
                continue

            restore_at = None
            restore_mapping = None
            confirmation_stop = index + self.persistence_frames
            for candidate_index in range(
                index + 1, min(confirmation_stop, len(frames))
            ):
                _, candidate_edges = self._edges(active, frames[candidate_index].bands)
                candidate_mapping = self._unique_mapping(candidate_edges)
                if candidate_mapping is not None:
                    restore_at = candidate_index
                    restore_mapping = candidate_mapping
                    break

            if restore_at is not None and restore_mapping is not None:
                epoch += 1
                for gap in frames[index:restore_at]:
                    output.append(
                        TrackedFrame(
                            gap.frame_index,
                            gap.step,
                            gap.tau,
                            (),
                            (EventRecord(EventCode.TRANSIENT_GAP, tuple(old_ids)),),
                            clean=False,
                            tracking_epoch=epoch,
                        )
                    )
                restored = frames[restore_at]
                tracked = []
                next_active = {}
                for row, column in restore_mapping.items():
                    track_id = old_ids[row]
                    band = restored.bands[column]
                    tracked.append(self._tracked(track_id, band, EventCode.RESTORED))
                    next_active[track_id] = _TrackState(band)
                active = next_active
                output.append(
                    TrackedFrame(
                        restored.frame_index,
                        restored.step,
                        restored.tau,
                        tuple(sorted(tracked, key=lambda item: item.track_id)),
                        (
                            EventRecord(
                                EventCode.RESTORED,
                                tuple(old_ids),
                                tuple(old_ids),
                            ),
                        ),
                        tracking_epoch=epoch,
                    )
                )
                index = restore_at + 1
                continue

            if confirmation_stop > len(frames):
                epoch += 1
                for gap in frames[index:]:
                    output.append(
                        TrackedFrame(
                            gap.frame_index,
                            gap.step,
                            gap.tau,
                            (),
                            (EventRecord(EventCode.TRANSIENT_GAP, tuple(old_ids)),),
                            clean=False,
                            tracking_epoch=epoch,
                        )
                    )
                break

            old_degree = np.sum(edges, axis=1)
            new_degree = np.sum(edges, axis=0)
            safe_pairs = {
                row: int(np.flatnonzero(edges[row])[0])
                for row in range(len(old_ids))
                if old_degree[row] == 1
                and new_degree[int(np.flatnonzero(edges[row])[0])] == 1
            }
            safe_columns = set(safe_pairs.values())
            retired = tuple(
                old_ids[row]
                for row in range(len(old_ids))
                if row not in safe_pairs
            )
            tracked = []
            next_active = {}
            for row, column in safe_pairs.items():
                track_id = old_ids[row]
                band = frame.bands[column]
                tracked.append(self._tracked(track_id, band))
                next_active[track_id] = _TrackState(band)

            created = []
            event_codes = []
            for column, band in enumerate(frame.bands):
                if column in safe_columns:
                    continue
                overlapping_rows = np.flatnonzero(edges[:, column])
                if len(overlapping_rows) > 1:
                    code = EventCode.MERGE
                elif (
                    len(overlapping_rows) == 1
                    and old_degree[overlapping_rows[0]] > 1
                ):
                    code = EventCode.SPLIT
                elif len(overlapping_rows) == 0 and len(old_ids) < len(frame.bands):
                    code = EventCode.BIRTH
                else:
                    code = EventCode.UNCERTAIN
                track_id = self._new_id()
                created.append(track_id)
                event_codes.append(code)
                tracked.append(self._tracked(track_id, band, code))
                next_active[track_id] = _TrackState(band)

            if not created:
                code = EventCode.DISAPPEARANCE
            elif EventCode.MERGE in event_codes:
                code = EventCode.MERGE
            elif EventCode.SPLIT in event_codes:
                code = EventCode.SPLIT
            elif EventCode.BIRTH in event_codes:
                code = EventCode.BIRTH
            else:
                code = EventCode.UNCERTAIN
            epoch += 1
            active = next_active
            output.append(
                TrackedFrame(
                    frame.frame_index,
                    frame.step,
                    frame.tau,
                    tuple(sorted(tracked, key=lambda item: item.track_id)),
                    (EventRecord(code, retired, tuple(created)),),
                    tracking_epoch=epoch,
                )
            )
            index += 1
        return output
