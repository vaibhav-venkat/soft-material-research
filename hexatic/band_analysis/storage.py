from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from .characterization import N_MODES, mode_wave_numbers
from .tracking import TrackedFrame


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
) -> dict[str, np.ndarray]:
    """Build padded time-by-track tensors; validity defines occupied slots."""
    n_time = len(frames)
    n_tracks = 1 + max(
        (band.track_id for frame in frames for band in frame.bands),
        default=-1,
    )
    tensors: dict[str, np.ndarray] = {
        "frame_index": np.asarray(
            [frame.frame_index for frame in frames], dtype=np.int64
        ),
        "simulation_step": np.asarray([frame.step for frame in frames], dtype=np.int64),
        "track_id": np.arange(n_tracks, dtype=np.int64),
        "valid": np.zeros((n_time, n_tracks), dtype=bool),
        "lower": np.full((n_time, n_tracks, ns), np.nan),
        "upper": np.full((n_time, n_tracks, ns), np.nan),
        "centerline": np.full((n_time, n_tracks, ns), np.nan),
        "local_width": np.full((n_time, n_tracks, ns), np.nan),
        "geometrically_complex": np.zeros((n_time, n_tracks), dtype=bool),
        "q_n": mode_wave_numbers(circumference),
    }
    for name in SCALAR_FIELDS:
        tensors[name] = np.full((n_time, n_tracks), np.nan)
    for prefix in ("centerline_mode", "width_mode"):
        for part in ("real", "imag", "amplitude"):
            tensors[f"{prefix}_{part}"] = np.full(
                (n_time, n_tracks, N_MODES), np.nan
            )

    for time_index, frame in enumerate(frames):
        for tracked in frame.bands:
            track_id = tracked.track_id
            band = tracked.characterization
            tensors["valid"][time_index, track_id] = True
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

    valid = tensors["valid"]
    for name in NET_FIELDS:
        source = "displacement" if name == "axial_displacement" else name
        tensors[f"net_{name}"] = np.sum(
            np.where(valid, tensors[source], 0.0), axis=1
        )
    tensors["net_centerline_mode_amplitude"] = np.sum(
        np.where(
            valid[..., np.newaxis],
            tensors["centerline_mode_amplitude"],
            0.0,
        ),
        axis=1,
    )
    tensors["net_width_mode_amplitude"] = np.sum(
        np.where(
            valid[..., np.newaxis],
            tensors["width_mode_amplitude"],
            0.0,
        ),
        axis=1,
    )
    return tensors


def save_characterization(path: Path, tensors: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {name: np.ascontiguousarray(value) for name, value in tensors.items()},
        temporary,
        metadata={
            "schema": "hexatic.band_characterization.v1",
            "axial_component": "x",
            "circumferential_coordinate": "s",
        },
    )
    temporary.replace(path)
