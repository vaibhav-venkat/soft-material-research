from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from hexatic.confinement_comparison.storage import (
    save_safetensors_atomic,
    write_json_atomic,
)


class FrameShardWriter:
    def __init__(
        self,
        output_dir: Path,
        manifest: dict[str, Any],
        *,
        backend_name: str,
        frames_per_shard: int,
    ) -> None:
        if frames_per_shard < 1:
            raise ValueError("frames_per_shard must be positive")
        self.output_dir = output_dir
        self.manifest = manifest
        self.backend_name = backend_name
        self.frames_per_shard = frames_per_shard
        self.pending: list[dict[str, np.ndarray]] = []

    def add(self, frame: dict[str, np.ndarray]) -> None:
        self.pending.append(frame)
        if len(self.pending) >= self.frames_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        indices = [int(frame["frame_index"]) for frame in self.pending]
        start, stop = indices[0], indices[-1] + 1
        tensors = {
            name: np.stack([frame[name] for frame in self.pending], axis=0)
            for name in self.pending[0]
        }
        filename = f"frames_{start:06d}_{stop:06d}.safetensors"
        save_safetensors_atomic(
            self.output_dir / filename,
            tensors,
            backend_name=self.backend_name,
            metadata={
                "schema": "hexatic.new_sims.frames.v1",
                "frame_start": str(start),
                "frame_stop": str(stop),
            },
        )
        self.manifest["shards"].append(
            {
                "file": filename,
                "frame_start": start,
                "frame_stop": stop,
                "steps": tensors["step"].astype(np.int64).tolist(),
                "bytes": sum(array.nbytes for array in tensors.values()),
            }
        )
        write_json_atomic(self.output_dir / "manifest.json", self.manifest)
        self.pending.clear()

