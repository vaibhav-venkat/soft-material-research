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

from ..detection.chains import EventChain
from ..detection.events import EventStep
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


def save_seed_event_chains(
    path: Path,
    chains: list[EventChain],
    *,
    seed_id: str,
    source_fingerprint: str,
    settings: dict[str, Any],
) -> None:
    """Store locally padded chains; collection-wide padding happens after load."""
    n_max = chains[0].n_max if chains else 0
    lengths = np.asarray([chain.n_steps for chain in chains], dtype=np.int64)
    rows = int(lengths.sum())
    steps = int(np.maximum(lengths - 1, 0).sum())

    def row_array(name: str, *, dtype: np.dtype[Any]) -> np.ndarray:
        return (
            np.concatenate([getattr(chain, name) for chain in chains])
            if chains
            else np.empty((0, n_max), dtype=dtype)
        )

    def step_array(name: str, *, matrix: bool = False) -> np.ndarray:
        values = [getattr(step, name) for chain in chains for step in chain.events]
        shape = (steps, n_max, n_max) if matrix else (steps, n_max)
        return np.asarray(values) if values else np.empty(shape)

    tensors = {
        "chain_length": lengths,
        "row_offset": np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
        "event_offset": np.concatenate(
            ([0], np.cumsum(np.maximum(lengths - 1, 0)))
        ).astype(np.int64),
        "slot_id": row_array("slot_ids", dtype=np.dtype(np.int64)),
        "frame_index": np.concatenate([chain.frame_indices for chain in chains])
        if rows
        else np.empty(0, dtype=np.int64),
        "tau": np.concatenate([chain.tau for chain in chains])
        if rows
        else np.empty(0),
        "area": row_array("areas", dtype=np.dtype(np.float64)),
        "mask": row_array("mask", dtype=np.dtype(bool)),
        "event_H": step_array("H", matrix=True),
        "event_g": step_array("g"),
        "event_C": step_array("C", matrix=True),
        "event_o": step_array("o"),
        "event_sigma_e_rows": step_array("sigma_e_rows"),
        "event_mask_next": step_array("mask_next"),
        "event_y": step_array("y"),
    }
    write_arrays(
        path,
        tensors,
        {
            "seed_id": seed_id,
            "source_fingerprint": source_fingerprint,
            "settings_fingerprint": fingerprint(settings),
        },
    )


def load_seed_event_chains(
    path: Path,
    *,
    source_fingerprint: str,
    settings: dict[str, Any],
) -> list[EventChain]:
    with safe_open(path, framework="numpy") as handle:
        metadata = dict(handle.metadata())
    if metadata.get("source_fingerprint") != source_fingerprint:
        raise ValueError(f"input changed for event cache {path}; rerun with --overwrite")
    if metadata.get("settings_fingerprint") != fingerprint(settings):
        raise ValueError(f"settings changed for event cache {path}; rerun with --overwrite")
    tensors = load_file(path)
    chains = []
    seed_id = metadata.get("seed_id", "")
    for index, length_value in enumerate(tensors["chain_length"]):
        length = int(length_value)
        row_start, row_stop = tensors["row_offset"][index : index + 2]
        event_start, event_stop = tensors["event_offset"][index : index + 2]
        events = tuple(
            EventStep(
                H=tensors["event_H"][step].copy(),
                g=tensors["event_g"][step].copy(),
                C=tensors["event_C"][step].copy(),
                o=tensors["event_o"][step].astype(bool),
                sigma_e_rows=tensors["event_sigma_e_rows"][step].astype(bool),
                mask_next=tensors["event_mask_next"][step].astype(bool),
                y=tensors["event_y"][step].copy(),
            )
            for step in range(int(event_start), int(event_stop))
        )
        chains.append(
            EventChain(
                seed_id=seed_id,
                slot_ids=tensors["slot_id"][row_start:row_stop].astype(np.int64),
                frame_indices=tensors["frame_index"][row_start:row_stop].astype(
                    np.int64
                ),
                tau=tensors["tau"][row_start:row_stop].copy(),
                areas=tensors["area"][row_start:row_stop].copy(),
                mask=tensors["mask"][row_start:row_stop].astype(bool),
                events=events,
            )
        )
        if len(events) != length - 1:
            raise ValueError(f"incomplete event cache {path}")
    return chains
