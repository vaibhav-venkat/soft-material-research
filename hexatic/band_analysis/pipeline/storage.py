"""Store compatible segment and inference artifacts atomically."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from ..detection.segments import StableSegment


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a sibling temporary path that replaces `path` on clean exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    yield temporary
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(value)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_arrays(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, str]
) -> None:
    with atomic_path(path) as temporary:
        save_file(
            {key: np.ascontiguousarray(value) for key, value in arrays.items()},
            temporary,
            metadata=metadata,
        )


def save_seed_segments(
    path: Path,
    segments: list[StableSegment],
    *,
    seed_id: str,
    source_fingerprint: str,
    settings: dict[str, Any],
) -> None:
    """Atomically store one seed's ragged stable segments."""
    lengths = np.asarray(
        [len(segment.tau) for segment in segments], dtype=np.int64
    )
    band_counts = np.asarray(
        [segment.n_bands for segment in segments], dtype=np.int64
    )
    tensors = {
        "segment_length": lengths,
        "band_count": band_counts,
        "time_offset": np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        "track_offset": np.concatenate(([0], np.cumsum(band_counts))).astype(
            np.int64
        ),
        "area_offset": np.concatenate(
            ([0], np.cumsum(lengths * band_counts))
        ).astype(np.int64),
        "tau": np.concatenate([segment.tau for segment in segments])
        if segments
        else np.empty(0, dtype=np.float64),
        "frame_index": np.concatenate(
            [segment.frame_indices for segment in segments]
        )
        if segments
        else np.empty(0, dtype=np.int64),
        "track_id": np.asarray(
            [track_id for segment in segments for track_id in segment.track_ids],
            dtype=np.int64,
        ),
        "area": np.concatenate([segment.areas.ravel() for segment in segments])
        if segments
        else np.empty(0, dtype=np.float64),
    }
    write_arrays(
        path,
        tensors,
        {
            "seed_id": seed_id,
            "source_fingerprint": source_fingerprint,
            "settings": json.dumps(settings, sort_keys=True),
            "settings_fingerprint": fingerprint(settings),
        },
    )


def load_seed_segments(
    path: Path,
    *,
    source_fingerprint: str | None = None,
    settings: dict[str, Any] | None = None,
) -> tuple[list[StableSegment], dict[str, Any]]:
    with safe_open(path, framework="numpy") as handle:
        metadata = dict(handle.metadata())
    if (
        source_fingerprint is not None
        and metadata.get("source_fingerprint") != source_fingerprint
    ):
        raise ValueError(f"input changed for segment cache {path}; rerun with --overwrite")
    if settings is not None and metadata.get("settings_fingerprint") != fingerprint(
        settings
    ):
        raise ValueError(f"settings changed for segment cache {path}; rerun with --overwrite")

    tensors = load_file(path)
    required = {
        "segment_length",
        "band_count",
        "time_offset",
        "track_offset",
        "area_offset",
        "tau",
        "frame_index",
        "track_id",
        "area",
    }
    if missing := sorted(required - tensors.keys()):
        raise ValueError(f"incomplete segment cache {path}: missing {missing}")
    seed_id = metadata.get("seed_id", "")
    segments = []
    for index, (length, n_bands) in enumerate(
        zip(tensors["segment_length"], tensors["band_count"], strict=True)
    ):
        time_start, time_stop = tensors["time_offset"][index : index + 2]
        track_start, track_stop = tensors["track_offset"][index : index + 2]
        area_start, area_stop = tensors["area_offset"][index : index + 2]
        segments.append(
            StableSegment(
                seed_id=seed_id,
                track_ids=tuple(
                    int(value) for value in tensors["track_id"][track_start:track_stop]
                ),
                frame_indices=tensors["frame_index"][time_start:time_stop].copy(),
                tau=tensors["tau"][time_start:time_stop].copy(),
                areas=tensors["area"][area_start:area_stop]
                .reshape(int(length), int(n_bands))
                .copy(),
            )
        )
    return segments, metadata
