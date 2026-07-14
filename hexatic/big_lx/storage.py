from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def save_safetensors_atomic(
    path: Path,
    tensors: dict[str, np.ndarray],
    *,
    backend_name: str,
    metadata: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    contiguous = {
        name: np.ascontiguousarray(value)
        for name, value in tensors.items()
    }
    if backend_name == "jax":
        import jax.numpy as jnp
        from safetensors.flax import save_file

        save_file(
            {name: jnp.asarray(value) for name, value in contiguous.items()},
            temporary,
            metadata=metadata,
        )
    else:
        from safetensors.numpy import save_file

        save_file(contiguous, temporary, metadata=metadata)
    temporary.replace(path)


def prepare_analysis_dir(
    output_dir: Path,
    overwrite: bool,
    resume: bool,
) -> tuple[bool, dict[str, Any] | None]:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") is True:
            print(f"skipping completed analysis {output_dir.name}")
            return False, manifest
        if resume:
            return True, manifest
        raise FileExistsError(
            f"Incomplete analysis output exists at {output_dir}; "
            "pass --resume to continue it or --overwrite to replace it"
        )
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Analysis output exists at {output_dir} without a usable manifest; "
                "pass --overwrite to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return True, None


class FrameShardWriter:
    def __init__(
        self,
        output_dir: Path,
        manifest: dict[str, Any],
        *,
        backend_name: str,
        target_bytes: int,
    ) -> None:
        self.output_dir = output_dir
        self.manifest = manifest
        self.backend_name = backend_name
        self.target_bytes = target_bytes
        self.pending: list[dict[str, np.ndarray]] = []
        self.pending_bytes = 0

    def add(self, frame: dict[str, np.ndarray]) -> None:
        frame_bytes = sum(value.nbytes for value in frame.values())
        if self.pending and self.pending_bytes + frame_bytes > self.target_bytes:
            self.flush()
        self.pending.append(frame)
        self.pending_bytes += frame_bytes

    def flush(self) -> None:
        if not self.pending:
            return
        frame_indices = [int(frame["frame_index"]) for frame in self.pending]
        start = frame_indices[0]
        stop = frame_indices[-1] + 1
        names = self.pending[0].keys()
        tensors = {
            name: np.stack([frame[name] for frame in self.pending], axis=0)
            for name in names
        }
        filename = f"frames_{start:06d}_{stop:06d}.safetensors"
        save_safetensors_atomic(
            self.output_dir / filename,
            tensors,
            backend_name=self.backend_name,
            metadata={
                "schema": "hexatic.big_lx.frames.v1",
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
                "bytes": sum(value.nbytes for value in tensors.values()),
            }
        )
        write_json_atomic(self.output_dir / "manifest.json", self.manifest)
        self.pending.clear()
        self.pending_bytes = 0
