from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from safetensors import safe_open
from safetensors.numpy import load_file, save_file

from .characterization import N_MODES, mode_wave_numbers
from .tracking import TrackedFrame


SCHEMA = "hexatic.band_characterization.v3"

SCALAR_FIELDS = (
    "area",
    "mean_width",
    "wrapped_axial_position",
    "unwrapped_axial_position",
    "displacement",
    "width_variance",
    "centerline_variance",
    "interface_length",
)

NET_FIELDS = (
    "area",
    "mean_width",
    "axial_displacement",
    "width_variance",
    "centerline_variance",
    "interface_length",
)


def build_characterization_tensors(
    frames: list[TrackedFrame],
    *,
    ns: int,
    circumference: float,
    timestep: float,
    run_steps: int,
    trajectory_write_period: int,
    persistence_frames: int = 5,
) -> dict[str, np.ndarray]:
    """Build padded time-by-track tensors; validity defines occupied slots."""
    if timestep <= 0.0:
        raise ValueError("timestep must be positive")
    if persistence_frames < 1:
        raise ValueError("persistence_frames must be positive")
    n_time = len(frames)
    n_tracks = 1 + max(
        (band.track_id for frame in frames for band in frame.bands),
        default=-1,
    )
    max_events = max((len(frame.events) for frame in frames), default=0)
    steps = np.asarray([frame.step for frame in frames], dtype=np.int64)
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("simulation steps must be strictly increasing")
    tensors: dict[str, np.ndarray] = {
        "frame_index": np.asarray(
            [frame.frame_index for frame in frames], dtype=np.int64
        ),
        "simulation_step": steps,
        "physical_time": steps.astype(np.float64) * timestep,
        "timestep": np.asarray(timestep, dtype=np.float64),
        "run_steps": np.asarray(run_steps, dtype=np.int64),
        "run_duration_tau": np.asarray(run_steps * timestep, dtype=np.float64),
        "trajectory_write_period": np.asarray(
            trajectory_write_period, dtype=np.int64
        ),
        "track_id": np.arange(n_tracks, dtype=np.int64),
        "valid": np.zeros((n_time, n_tracks), dtype=bool),
        "stable": np.zeros((n_time, n_tracks), dtype=bool),
        "segment_id": np.full((n_time, n_tracks), -1, dtype=np.int64),
        "assignment_jaccard": np.full((n_time, n_tracks), np.nan),
        "event_code": np.zeros((n_time, n_tracks), dtype=np.int8),
        "event_boundary": np.zeros((n_time, n_tracks), dtype=bool),
        "lower": np.full((n_time, n_tracks, ns), np.nan),
        "upper": np.full((n_time, n_tracks, ns), np.nan),
        "centerline": np.full((n_time, n_tracks, ns), np.nan),
        "local_width": np.full((n_time, n_tracks, ns), np.nan),
        "geometrically_complex": np.zeros((n_time, n_tracks), dtype=bool),
        "q_n": mode_wave_numbers(circumference),
        "frame_event_count": np.asarray(
            [len(frame.events) for frame in frames], dtype=np.int64
        ),
        "frame_event_code": np.zeros((n_time, max_events), dtype=np.int8),
        "frame_event_old_track_id": np.full(
            (n_time, max_events), -1, dtype=np.int64
        ),
        "frame_event_new_track_id": np.full(
            (n_time, max_events), -1, dtype=np.int64
        ),
        "frame_event_jaccard": np.full((n_time, max_events), np.nan),
    }
    for name in SCALAR_FIELDS:
        tensors[name] = np.full((n_time, n_tracks), np.nan)
    for prefix in ("centerline_mode", "width_mode"):
        for part in ("real", "imag", "amplitude"):
            tensors[f"{prefix}_{part}"] = np.full(
                (n_time, n_tracks, N_MODES), np.nan
            )

    for time_index, frame in enumerate(frames):
        for event_index, event in enumerate(frame.events):
            tensors["frame_event_code"][time_index, event_index] = int(event.code)
            tensors["frame_event_old_track_id"][time_index, event_index] = (
                event.old_track_id
            )
            tensors["frame_event_new_track_id"][time_index, event_index] = (
                event.new_track_id
            )
            tensors["frame_event_jaccard"][time_index, event_index] = event.jaccard
        for tracked in frame.bands:
            track_id = tracked.track_id
            band = tracked.characterization
            tensors["valid"][time_index, track_id] = True
            tensors["segment_id"][time_index, track_id] = tracked.segment_id
            tensors["assignment_jaccard"][time_index, track_id] = (
                tracked.assignment_jaccard
            )
            tensors["event_code"][time_index, track_id] = int(tracked.event_code)
            tensors["event_boundary"][time_index, track_id] = tracked.event_boundary
            tensors["lower"][time_index, track_id] = band.lower
            tensors["upper"][time_index, track_id] = band.upper
            tensors["centerline"][time_index, track_id] = band.centerline
            tensors["local_width"][time_index, track_id] = band.width
            tensors["geometrically_complex"][time_index, track_id] = (
                band.geometrically_complex
            )
            for name in SCALAR_FIELDS:
                tensors[name][time_index, track_id] = getattr(band, name)
            tensors["centerline_mode_real"][time_index, track_id] = (
                band.centerline_modes.real
            )
            tensors["centerline_mode_imag"][time_index, track_id] = (
                band.centerline_modes.imag
            )
            tensors["centerline_mode_amplitude"][time_index, track_id] = (
                band.centerline_amplitudes
            )
            tensors["width_mode_real"][time_index, track_id] = band.width_modes.real
            tensors["width_mode_imag"][time_index, track_id] = band.width_modes.imag
            tensors["width_mode_amplitude"][time_index, track_id] = (
                band.width_amplitudes
            )

    valid_segments = tensors["segment_id"][tensors["valid"]]
    if valid_segments.size:
        segment_ids, counts = np.unique(valid_segments, return_counts=True)
        stable_ids = segment_ids[counts >= persistence_frames]
        tensors["stable"] = tensors["valid"] & np.isin(
            tensors["segment_id"], stable_ids
        )

    valid = tensors["valid"]
    valid_device = jnp.asarray(valid)
    for name in NET_FIELDS:
        source = "displacement" if name == "axial_displacement" else name
        tensors[f"net_{name}"] = np.asarray(
            jax.device_get(
                jnp.sum(
                    jnp.where(valid_device, jnp.asarray(tensors[source]), 0.0),
                    axis=1,
                )
            )
        )
    tensors["net_centerline_mode_amplitude"] = np.asarray(
        jax.device_get(
            jnp.sum(
                jnp.where(
                    valid_device[..., jnp.newaxis],
                    jnp.asarray(tensors["centerline_mode_amplitude"]),
                    0.0,
                ),
                axis=1,
            )
        )
    )
    tensors["net_width_mode_amplitude"] = np.asarray(
        jax.device_get(
            jnp.sum(
                jnp.where(
                    valid_device[..., jnp.newaxis],
                    jnp.asarray(tensors["width_mode_amplitude"]),
                    0.0,
                ),
                axis=1,
            )
        )
    )
    return tensors


def save_characterization(
    path: Path,
    tensors: dict[str, np.ndarray],
    *,
    configuration: dict[str, Any],
    compute_provenance: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {name: np.ascontiguousarray(value) for name, value in tensors.items()},
        temporary,
        metadata={
            "schema": SCHEMA,
            "axial_component": "x",
            "circumferential_coordinate": "s",
            "configuration": json.dumps(configuration, sort_keys=True),
            "compute_provenance": json.dumps(compute_provenance, sort_keys=True),
        },
    )
    temporary.replace(path)


def load_characterization(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata()
    if metadata.get("schema") != SCHEMA:
        raise ValueError(
            f"incompatible characterization cache {path}; rerun with --overwrite"
        )
    try:
        configuration = json.loads(metadata["configuration"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(
            f"incomplete characterization cache {path}; rerun with --overwrite"
        ) from error
    tensors = load_file(path)
    required = {
        "physical_time",
        "stable",
        "dynamics_axial_position_drift",
        "position_msd",
        "neighbor_area_covariance",
        "area_diffusion_fit_alpha",
        "area_diffusion_fit_valid",
    }
    missing = sorted(required - tensors.keys())
    if missing:
        raise ValueError(
            f"incomplete characterization cache {path} (missing {missing}); "
            "rerun with --overwrite"
        )
    return tensors, configuration
