"""Load simulation arrays and metadata used by band extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from safetensors import safe_open

from .storage import fingerprint


@dataclass(frozen=True)
class InputMetadata:
    input_dir: Path
    manifest: dict[str, Any]
    lx: float
    circumference: float
    radius: float
    particle_diameter: float
    run_steps: int
    trajectory_write_period: int

    @property
    def source_fingerprint(self) -> str:
        return fingerprint(self.manifest)

    @property
    def compatibility_fingerprint(self) -> str:
        case = {
            key: value
            for key, value in self.manifest["case"].items()
            if key not in {"case_id", "label", "seed"}
        }
        preprocessing_keys = (
            "pocket_radius",
            "gaussian_cutoff_multiplier",
            "gaussian_cutoff",
            "dtype",
        )
        preprocessing = {
            key: self.manifest[key]
            for key in preprocessing_keys
            if key in self.manifest
        }
        return fingerprint({"case": case, "preprocessing": preprocessing})


def load_metadata(input_dir: Path) -> InputMetadata:
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.big_lx.analysis.v1":
        raise ValueError(f"Unsupported manifest schema in {manifest_path}")
    case = manifest["case"]
    return InputMetadata(
        input_dir=input_dir,
        manifest=manifest,
        lx=float(case["lx"]),
        circumference=float(case["circumference"]),
        radius=float(case["radius"]),
        particle_diameter=float(case["particle_diameter"]),
        run_steps=int(case["run_steps"]),
        trajectory_write_period=int(case["trajectory_write_period"]),
    )


def require_compatible_seeds(metadata: list[InputMetadata]) -> str:
    if not metadata:
        raise ValueError("at least one input directory is required")
    expected = metadata[0].compatibility_fingerprint
    incompatible = [
        str(item.input_dir)
        for item in metadata[1:]
        if item.compatibility_fingerprint != expected
    ]
    if incompatible:
        raise ValueError(
            "seed manifests do not describe the same physical case and preprocessing: "
            + ", ".join(incompatible)
        )
    return expected


def frame_numbers(
    metadata: InputMetadata,
    start: int,
    stop: int | None,
    stride: int,
) -> list[int]:
    available_stop = max(int(shard["frame_stop"]) for shard in metadata.manifest["shards"])
    selected_stop = available_stop if stop is None else min(stop, available_stop)
    return list(range(start, selected_stop, stride))


def iter_frames(
    metadata: InputMetadata,
    selected: set[int],
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield frame index, simulation step, and saved cylindrical coordinates."""
    for shard in metadata.manifest["shards"]:
        shard_start = int(shard["frame_start"])
        shard_stop = int(shard["frame_stop"])
        wanted = sorted(frame for frame in selected if shard_start <= frame < shard_stop)
        if not wanted:
            continue
        path = metadata.input_dir / str(shard["file"])
        with safe_open(path, framework="numpy") as tensors:
            keys = tensors.keys()
            if "coords" not in keys or "step" not in keys:
                raise KeyError(f"{path} must contain coords and step tensors")
            coords = tensors.get_slice("coords")
            steps = tensors.get_slice("step")
            for frame in wanted:
                local = frame - shard_start
                yield frame, int(steps[local]), np.asarray(coords[local])
