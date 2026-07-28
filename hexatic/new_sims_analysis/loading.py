"""Manifest, safetensor, and case-metadata loading for the new-sims outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open


MANIFEST_SCHEMA = "hexatic.new_sims.analysis.v1"


PLANAR_COORDINATE_ORDER = ["x", "y"]


CARTESIAN_COORDINATE_ORDER = ["x", "y", "z"]


CYLINDRICAL_COORDINATE_ORDER = ["x", "theta", "r"]


def _load_manifest(
    shard_dir: Path,
    *,
    allow_2d: bool,
) -> dict[str, Any]:
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    schema = manifest.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported manifest schema {schema!r} in {manifest_path}; "
            f"expected {MANIFEST_SCHEMA!r}"
        )
    if manifest.get("complete") is not True:
        raise ValueError(f"analysis manifest is incomplete: {manifest_path}")
    case = manifest.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"missing case metadata in {manifest_path}")
    coordinate_order = manifest.get("coordinate_order")
    supported_orders = [
        CARTESIAN_COORDINATE_ORDER,
        CYLINDRICAL_COORDINATE_ORDER,
    ]
    if allow_2d:
        supported_orders.append(PLANAR_COORDINATE_ORDER)
    if coordinate_order not in supported_orders:
        expected = (
            f"{PLANAR_COORDINATE_ORDER}, "
            if allow_2d
            else ""
        )
        raise ValueError(
            f"expected coordinate_order {expected}"
            f"{CARTESIAN_COORDINATE_ORDER}, or "
            f"{CYLINDRICAL_COORDINATE_ORDER}; got {coordinate_order!r}"
        )
    dimensions = case.get("dimensions")
    if (
        coordinate_order == CARTESIAN_COORDINATE_ORDER
        and dimensions is not None
        and int(dimensions) != 3
    ):
        raise ValueError(f"expected a 3D case, got dimensions={dimensions!r}")
    return manifest


def _load_lx(static_path: Path) -> float:
    if not static_path.is_file():
        raise FileNotFoundError(f"missing static tensors: {static_path}")
    with safe_open(static_path, framework="np") as static_file:
        keys = set(static_file.keys())
        if "lx" in keys:
            value = np.asarray(static_file.get_tensor("lx"))
            if value.size != 1:
                raise ValueError(f"expected scalar lx in {static_path}")
            lx = float(value.reshape(-1)[0])
        elif "box" in keys:
            box = np.asarray(static_file.get_tensor("box"))
            if box.ndim != 1 or box.size < 1:
                raise ValueError(f"invalid box tensor in {static_path}")
            lx = float(box[0])
        else:
            raise KeyError(f"{static_path} has neither 'lx' nor 'box' tensor")
    if not np.isfinite(lx) or lx <= 0.0:
        raise ValueError(f"invalid axial box length lx={lx}")
    return lx


class _PeriodicComUnwrapper:
    def __init__(self, n_particles: int, period: float) -> None:
        self._n_particles = n_particles
        self._period = period
        self._previous = np.empty(n_particles, dtype=np.float64)
        self._unwrapped = np.empty(n_particles, dtype=np.float64)
        self._started = False

    def push(self, values: NDArray[np.float64], frame_index: int) -> float:
        if len(values) != self._n_particles:
            raise ValueError(f"particle count changes at frame {frame_index}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite coordinate at frame {frame_index}")
        if not self._started:
            self._unwrapped[:] = values
            self._started = True
        else:
            displacement = values - self._previous
            scaled = displacement / self._period
            nearest_integer = np.copysign(
                np.floor(np.abs(scaled) + 0.5), scaled
            )
            self._unwrapped += displacement - self._period * nearest_integer
        self._previous[:] = values
        return float(np.mean(self._unwrapped, dtype=np.float64))


def _cartesian_directions(
    shard: Any,
    shard_path: Path,
) -> NDArray[np.float64]:
    """Load the per-particle Cartesian orientation vectors from a shard.

    The stored polarization values are Cartesian (x, y, z) unit vectors.
    """
    keys = set(shard.keys())
    name = "polarization" if "polarization" in keys else None
    if name is None:
        raise KeyError(
            f"{shard_path} has no 'polarization' tensor"
        )
    directions = np.asarray(shard.get_tensor(name), dtype=np.float64)
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError(
            f"expected {name} shape (frames, particles, 3) in {shard_path}, "
            f"got {directions.shape}"
        )
    if not np.all(np.isfinite(directions)):
        raise ValueError(f"{name} contains non-finite values in {shard_path}")
    norms = np.linalg.norm(directions, axis=2)
    if not np.allclose(norms, 1.0, rtol=2e-4, atol=2e-4):
        raise ValueError(f"{name} in {shard_path} is not unit-normalized")
    return directions


def _load_frame_series(
    shard_dir: Path,
    manifest: dict[str, Any],
    frame_limit: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float32],
    NDArray[np.int64],
]:
    """Load the per-frame COM coordinates and per-particle orientations.

    Returns ``(com, q, steps)``. Both COM and orientations use Cartesian
    ``(x, y, z)`` components for bulk inputs. Cylindrical coordinates use
    ``(x, theta, r)`` and are unwrapped along x and theta before taking COM.

    Note that ``q`` holds every particle for every frame, so memory scales as
    frames x particles; ``--frames`` is the knob for keeping it bounded.
    """
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest contains no safetensor shards")

    com_frames: list[NDArray[np.float64] | tuple[float, float, float]] = []
    q_arrays: list[NDArray[np.float32]] = []
    step_frames: list[int] = []
    loaded_frames = 0
    expected_start = 0
    cylindrical = (
        manifest["coordinate_order"] == CYLINDRICAL_COORDINATE_ORDER
    )
    lx = _load_lx(shard_dir / "static.safetensors") if cylindrical else None
    axial_unwrapper: _PeriodicComUnwrapper | None = None
    azimuthal_unwrapper: _PeriodicComUnwrapper | None = None
    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError("invalid shard entry in manifest")
        start = entry.get("frame_start")
        stop = entry.get("frame_stop")
        filename = entry.get("file")
        if start != expected_start or not isinstance(stop, int) or stop <= start:
            raise ValueError("safetensor shards are not contiguous")
        if not isinstance(filename, str):
            raise ValueError("shard entry has no filename")

        shard_path = shard_dir / filename
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="np") as shard:
            keys = set(shard.keys())
            if "coords" not in keys:
                raise KeyError(f"{shard_path} has no 'coords' tensor")
            if "step" not in keys:
                raise KeyError(f"{shard_path} has no 'step' tensor")
            coords = np.asarray(shard.get_tensor("coords"), dtype=np.float64)
            steps = np.asarray(shard.get_tensor("step"), dtype=np.int64)
            directions = _cartesian_directions(shard, shard_path)

        coordinate_count = len(manifest["coordinate_order"])
        if (
            coords.ndim != 3
            or coords.shape[1] == 0
            or coords.shape[2] != coordinate_count
        ):
            raise ValueError(
                "expected coords shape "
                f"(frames, particles, {coordinate_count}), "
                f"got {coords.shape}"
            )
        n_frames_in_shard = coords.shape[0]
        n_particles = coords.shape[1]
        if directions.shape[:2] != coords.shape[:2]:
            raise ValueError(
                f"orientation frame/particle shape {directions.shape[:2]} "
                f"does not match coords shape {coords.shape[:2]} in {shard_path}"
            )
        if steps.ndim != 1 or len(steps) != n_frames_in_shard:
            raise ValueError(
                f"expected one simulation step per frame in {shard_path}"
            )
        if n_frames_in_shard != stop - start:
            raise ValueError(
                f"shard range [{start}, {stop}) does not match "
                f"{n_frames_in_shard} stored frames in {shard_path}"
            )
        if cylindrical and axial_unwrapper is None:
            assert lx is not None
            axial_unwrapper = _PeriodicComUnwrapper(n_particles, lx)
            azimuthal_unwrapper = _PeriodicComUnwrapper(
                n_particles, 2.0 * np.pi
            )
        take = frame_limit - loaded_frames
        coords = coords[:take]
        steps = steps[:take]
        directions = directions[:take]
        actual_take = coords.shape[0]

        q_arrays.append(directions.astype(np.float32))

        for local_frame in range(actual_take):
            if cylindrical:
                assert axial_unwrapper is not None
                assert azimuthal_unwrapper is not None
                frame_index = start + local_frame
                com_frames.append(
                    (
                        axial_unwrapper.push(
                            coords[local_frame, :, 0], frame_index
                        ),
                        azimuthal_unwrapper.push(
                            coords[local_frame, :, 1], frame_index
                        ),
                        float(np.mean(coords[local_frame, :, 2])),
                    )
                )
            else:
                com_frames.append(
                    np.mean(
                        coords[local_frame], axis=0, dtype=np.float64
                    )
                )
            step_frames.append(int(steps[local_frame]))

        loaded_frames += actual_take
        expected_start = stop
        if loaded_frames >= frame_limit:
            break

    if len(com_frames) < 3:
        raise ValueError(
            "at least three frames are required for velocity correlation"
        )
    return (
        np.array(com_frames, dtype=np.float64),
        np.concatenate(q_arrays, axis=0),
        np.array(step_frames, dtype=np.int64),
    )


def _elapsed_time(
    steps: NDArray[np.int64],
    timestep: float,
) -> NDArray[np.float64]:
    initial = steps[0]
    return (steps.astype(np.float64) - float(initial)) * timestep


def _case_identity(
    input_dir: Path,
    manifest: dict[str, Any],
) -> tuple[str, str]:
    """Return the same case label used by big-Lx and a safe output slug."""
    case = manifest["case"]
    case_id = case.get("case_id")
    label = case.get("label")
    if not isinstance(case_id, str) or not case_id:
        case_id = input_dir.resolve().name
    if not isinstance(label, str) or not label:
        label = case_id
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    if not slug:
        raise ValueError(f"cannot derive an output filename for {input_dir}")
    return label, slug
