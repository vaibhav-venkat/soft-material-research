from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tracking import TrackedFrame


@dataclass(frozen=True)
class StableSegment:
    seed_id: str
    track_ids: tuple[int, ...]
    frame_indices: np.ndarray
    tau: np.ndarray
    areas: np.ndarray

    @property
    def n_bands(self) -> int:
        return len(self.track_ids)


def build_stable_segments(
    frames: list[TrackedFrame],
    *,
    seed_id: str,
    persistence_frames: int,
) -> list[StableSegment]:
    """Build maximal clean, fixed-identity segments in track-ID order."""
    segments: list[StableSegment] = []
    current: list[tuple[TrackedFrame, tuple[int, ...], tuple[float, ...]]] = []
    current_key: tuple[int, tuple[int, ...]] | None = None
    appearances: dict[int, int] = {}
    for frame in frames:
        if not frame.clean:
            continue
        for band in frame.bands:
            appearances[band.track_id] = appearances.get(band.track_id, 0) + 1
    stable_ids = {
        track_id
        for track_id, count in appearances.items()
        if count >= persistence_frames
    }

    def finish() -> None:
        if not current:
            return
        track_ids = current[0][1]
        segments.append(
            StableSegment(
                seed_id=seed_id,
                track_ids=track_ids,
                frame_indices=np.asarray(
                    [frame.frame_index for frame, _, _ in current],
                    dtype=np.int64,
                ),
                tau=np.asarray(
                    [frame.tau for frame, _, _ in current], dtype=np.float64
                ),
                areas=np.asarray([areas for _, _, areas in current], dtype=np.float64),
            )
        )

    for frame in frames:
        stable_bands = tuple(
            band for band in frame.bands if band.track_id in stable_ids
        )
        track_ids = tuple(band.track_id for band in stable_bands)
        key = (frame.tracking_epoch, track_ids)
        if not frame.clean or not track_ids:
            finish()
            current = []
            current_key = None
            continue
        if key != current_key:
            finish()
            current = []
            current_key = key
        current.append((frame, track_ids, tuple(band.area for band in stable_bands)))
    finish()
    return segments
