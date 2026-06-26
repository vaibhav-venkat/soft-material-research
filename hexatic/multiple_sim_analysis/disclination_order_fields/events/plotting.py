from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .io import event_plot_path


DEFAULT_EVENT_FIELD_NAMES = (
    "rho",
    "J_r",
    "u_rms",
    "u_fluct",
    "S",
    "psi6_abs",
    "chi",
    "D2min",
    "strain",
    "nearest_defect_distance",
)


def _as_float_array(values: np.ndarray | Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _resolve_output_path(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = event_plot_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _event_positions_from_values(
    values: Mapping[str, np.ndarray],
    event_kind: str,
) -> np.ndarray:
    positions_name = f"{event_kind}_positions"
    if positions_name in values:
        positions = _as_float_array(values[positions_name])
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"{positions_name} must have shape (events, 3)")
        return positions

    names = (f"{event_kind}_x", f"{event_kind}_theta", f"{event_kind}_r")
    if all(name in values for name in names):
        return np.column_stack([_as_float_array(values[name]) for name in names])
    raise KeyError(f"event values must include {positions_name} or {names}")


def _nearest_particle_index(
    particle_coords: np.ndarray,
    target_coord: np.ndarray,
    *,
    box_length_x: float | None,
) -> int:
    delta = particle_coords[:, :2] - target_coord[:2]
    if box_length_x is not None and box_length_x > 0.0:
        delta[:, 0] -= box_length_x * np.round(delta[:, 0] / box_length_x)
    delta[:, 1] = (delta[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
    if particle_coords.shape[1] > 2 and target_coord.shape[0] > 2:
        radial = particle_coords[:, 2] - target_coord[2]
    else:
        radial = 0.0
    distance2 = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + radial * radial
    return int(np.nanargmin(distance2))


def event_centered_samples(
    frame_axis: np.ndarray,
    event_frames: np.ndarray,
    event_positions: np.ndarray,
    field_values: Mapping[str, np.ndarray],
    *,
    tau: int = 2,
    coords: np.ndarray | None = None,
    box_length_x: float | None = None,
    field_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Collect event-centered windows from precomputed frame fields.

    Field arrays may be either frame-local scalars ``(frames,)`` or particle-local
    values ``(frames, particles)``. Particle-local fields require ``coords`` with
    shape ``(frames, particles, 3)`` so the event position can be sampled by nearest
    particle. Events too close to the frame boundary are truncated with ``NaN``.
    """
    frame_axis = np.asarray(frame_axis, dtype=np.int64)
    event_frames = np.asarray(event_frames, dtype=np.int64)
    event_positions = _as_float_array(event_positions)
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if event_positions.shape != (event_frames.size, 3):
        raise ValueError("event_positions must have shape (events, 3)")

    if field_names is None:
        field_names = tuple(field_values)
    relative_frames = np.arange(-tau, tau + 1, dtype=np.int64)
    frame_to_offset = {int(frame): idx for idx, frame in enumerate(frame_axis)}
    samples = {
        name: np.full((event_frames.size, relative_frames.size), np.nan, dtype=np.float64)
        for name in field_names
    }

    coords_array = None if coords is None else _as_float_array(coords)
    for event_idx, (event_frame, event_position) in enumerate(zip(event_frames, event_positions)):
        for rel_idx, rel_frame in enumerate(relative_frames):
            frame_offset = frame_to_offset.get(int(event_frame + rel_frame))
            if frame_offset is None:
                continue
            particle_idx = None
            if coords_array is not None:
                particle_idx = _nearest_particle_index(
                    coords_array[frame_offset],
                    event_position,
                    box_length_x=box_length_x,
                )
            for name in field_names:
                field = _as_float_array(field_values[name])
                if field.shape[0] != frame_axis.size:
                    raise ValueError(f"{name} first dimension must match frame_axis")
                if field.ndim == 1:
                    samples[name][event_idx, rel_idx] = field[frame_offset]
                elif field.ndim == 2:
                    if particle_idx is None:
                        raise ValueError(f"coords are required to sample particle-local field {name}")
                    samples[name][event_idx, rel_idx] = field[frame_offset, particle_idx]
                else:
                    raise ValueError(f"{name} must be a frame or frame/particle scalar array")
    samples["relative_frames"] = relative_frames.astype(np.float64)
    return samples


def event_centered_summary(samples: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return nanmean, standard error, and contributing count for each sampled field."""
    summary: dict[str, np.ndarray] = {
        "relative_frames": _as_float_array(samples["relative_frames"]),
    }
    for name, values in samples.items():
        if name == "relative_frames":
            continue
        array = _as_float_array(values)
        finite = np.isfinite(array)
        count = np.sum(finite, axis=0)
        mean = np.full(array.shape[1], np.nan, dtype=np.float64)
        stderr = np.full(array.shape[1], np.nan, dtype=np.float64)
        valid = count > 0
        if np.any(valid):
            mean[valid] = np.nanmean(array[:, valid], axis=0)
        if np.any(count > 1):
            stderr[count > 1] = np.nanstd(array[:, count > 1], axis=0) / np.sqrt(count[count > 1])
        summary[f"{name}_mean"] = mean
        summary[f"{name}_stderr"] = stderr
        summary[f"{name}_count"] = count.astype(np.float64)
    return summary


def plot_event_centered_summary(
    summary: Mapping[str, np.ndarray],
    output_dir: str | Path,
    *,
    prefix: str,
    field_names: Sequence[str] | None = None,
    title_prefix: str | None = None,
) -> list[Path]:
    """Write one line plot per event-centered field and return created paths."""
    import matplotlib.pyplot as plt

    output_path = _resolve_output_path(output_dir)
    relative_frames = _as_float_array(summary["relative_frames"])
    if field_names is None:
        field_names = tuple(
            key[: -len("_mean")]
            for key in summary
            if key.endswith("_mean") and f"{key[:-5]}_stderr" in summary
        )
    written: list[Path] = []
    for name in field_names:
        mean_key = f"{name}_mean"
        if mean_key not in summary:
            continue
        mean = _as_float_array(summary[mean_key])
        stderr = _as_float_array(summary.get(f"{name}_stderr", np.full_like(mean, np.nan)))
        fig, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
        axis.plot(relative_frames, mean, marker="o", color="#0072b2", linewidth=1.8)
        finite_error = np.isfinite(stderr)
        if np.any(finite_error):
            axis.fill_between(
                relative_frames,
                mean - np.nan_to_num(stderr, nan=0.0),
                mean + np.nan_to_num(stderr, nan=0.0),
                color="#0072b2",
                alpha=0.18,
                linewidth=0.0,
            )
        axis.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.set_xlabel("Frame offset from event")
        axis.set_ylabel(name)
        axis.set_title(f"{title_prefix or prefix}: {name}")
        axis.grid(True, linestyle="--", alpha=0.25)
        figure_path = output_path / f"{prefix}_{name}.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
        written.append(figure_path)
    return written


def periodic_central_gradient(
    values: np.ndarray,
    x_centers: np.ndarray,
    theta_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central differences for ``(frames, x, theta)`` grids periodic in x/theta."""
    grid = _as_float_array(values)
    if grid.ndim == 2:
        grid = grid[np.newaxis, ...]
    if grid.ndim != 3:
        raise ValueError("values must have shape (frames, x, theta) or (x, theta)")
    x_centers = _as_float_array(x_centers)
    theta_centers = _as_float_array(theta_centers)
    if grid.shape[1:] != (x_centers.size, theta_centers.size):
        raise ValueError("grid spatial dimensions must match centers")
    dx = float(np.nanmean(np.diff(np.sort(x_centers)))) if x_centers.size > 1 else 1.0
    dtheta = (
        float(np.nanmean(np.diff(np.sort(theta_centers)))) if theta_centers.size > 1 else 1.0
    )
    grad_x = (np.roll(grid, -1, axis=1) - np.roll(grid, 1, axis=1)) / (2.0 * dx)
    grad_theta = (np.roll(grid, -1, axis=2) - np.roll(grid, 1, axis=2)) / (2.0 * dtheta)
    magnitude = np.sqrt(grad_x * grad_x + grad_theta * grad_theta)
    return grad_x, grad_theta, magnitude


def bucket_event_probability(
    exposure_values: np.ndarray,
    event_values: np.ndarray,
    *,
    bins: int | Sequence[float] = 12,
) -> dict[str, np.ndarray]:
    """Bucket event counts by pre-event field values and normalize by exposure."""
    exposure = _as_float_array(exposure_values).ravel()
    events = _as_float_array(event_values).ravel()
    exposure = exposure[np.isfinite(exposure)]
    events = events[np.isfinite(events)]
    if isinstance(bins, int):
        if exposure.size == 0 and events.size == 0:
            edges = np.linspace(0.0, 1.0, bins + 1)
        else:
            source = exposure if exposure.size else events
            lo, hi = np.nanpercentile(source, [0.0, 100.0])
            if lo == hi:
                lo -= 0.5
                hi += 0.5
            edges = np.linspace(float(lo), float(hi), bins + 1)
    else:
        edges = _as_float_array(bins)
    exposure_count, _ = np.histogram(exposure, bins=edges)
    event_count, _ = np.histogram(events, bins=edges)
    probability = np.full(exposure_count.shape, np.nan, dtype=np.float64)
    valid = exposure_count > 0
    probability[valid] = event_count[valid] / exposure_count[valid]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "bin_edges": edges,
        "bin_centers": centers,
        "exposure_count": exposure_count.astype(np.float64),
        "event_count": event_count.astype(np.float64),
        "probability": probability,
    }


def probability_summaries(
    exposure_fields: Mapping[str, np.ndarray],
    event_fields: Mapping[str, Mapping[str, np.ndarray] | np.ndarray],
    *,
    bins: int | Sequence[float] = 12,
) -> dict[str, np.ndarray]:
    """Build probability tables for birth/death/annihilation field buckets."""
    output: dict[str, np.ndarray] = {}
    for field_name, exposure in exposure_fields.items():
        if field_name not in event_fields:
            continue
        event_source = event_fields[field_name]
        if isinstance(event_source, Mapping):
            items = event_source.items()
        else:
            items = (("event", event_source),)
        for event_name, event_values in items:
            table = bucket_event_probability(exposure, event_values, bins=bins)
            prefix = f"{event_name}_{field_name}"
            for key, values in table.items():
                output[f"{prefix}_{key}"] = values
    return output


def plot_probability_table(
    table: Mapping[str, np.ndarray],
    output_dir: str | Path,
    *,
    prefix: str = "probability",
) -> list[Path]:
    """Plot all probability series in a flat probability summary dictionary."""
    import matplotlib.pyplot as plt

    output_path = _resolve_output_path(output_dir)
    written: list[Path] = []
    for key, probability in table.items():
        if not key.endswith("_probability"):
            continue
        stem = key[: -len("_probability")]
        centers_key = f"{stem}_bin_centers"
        if centers_key not in table:
            continue
        centers = _as_float_array(table[centers_key])
        probability = _as_float_array(probability)
        fig, axis = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
        axis.plot(centers, probability, marker="o", color="#d55e00", linewidth=1.7)
        axis.set_xlabel(stem)
        axis.set_ylabel("Event probability per exposure")
        axis.set_title(stem.replace("_", " "))
        axis.grid(True, linestyle="--", alpha=0.25)
        figure_path = output_path / f"{prefix}_{stem}.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
        written.append(figure_path)
    return written


def event_centered_summaries_from_tables(
    event_values: Mapping[str, np.ndarray],
    frame_axis: np.ndarray,
    field_values: Mapping[str, np.ndarray],
    *,
    tau: int = 2,
    coords: np.ndarray | None = None,
    box_length_x: float | None = None,
    field_names: Sequence[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Convenience wrapper for birth, death, and death subpopulation summaries."""
    outputs: dict[str, dict[str, np.ndarray]] = {}
    if field_names is None:
        field_names = tuple(name for name in DEFAULT_EVENT_FIELD_NAMES if name in field_values)
    for event_kind in ("birth", "death"):
        frames_key = f"{event_kind}_frames"
        if frames_key not in event_values:
            continue
        positions = _event_positions_from_values(event_values, event_kind)
        samples = event_centered_samples(
            frame_axis,
            np.asarray(event_values[frames_key], dtype=np.int64),
            positions,
            field_values,
            tau=tau,
            coords=coords,
            box_length_x=box_length_x,
            field_names=field_names,
        )
        outputs[event_kind] = event_centered_summary(samples)

    if "death_annihilated" in event_values and "death_frames" in event_values:
        mask = np.asarray(event_values["death_annihilated"], dtype=bool)
        positions = _event_positions_from_values(event_values, "death")
        for name, submask in (
            ("death_annihilated", mask),
            ("death_non_annihilated", ~mask),
        ):
            samples = event_centered_samples(
                frame_axis,
                np.asarray(event_values["death_frames"], dtype=np.int64)[submask],
                positions[submask],
                field_values,
                tau=tau,
                coords=coords,
                box_length_x=box_length_x,
                field_names=field_names,
            )
            outputs[name] = event_centered_summary(samples)
    return outputs


def plot_event_summaries(
    event_values: Mapping[str, np.ndarray],
    frame_axis: np.ndarray,
    field_values: Mapping[str, np.ndarray],
    output_dir: str | Path = "events",
    *,
    tau: int = 2,
    coords: np.ndarray | None = None,
    box_length_x: float | None = None,
    field_names: Sequence[str] | None = None,
) -> dict[str, list[Path]]:
    """Compute and plot event-centered summaries from supplied arrays only."""
    summaries = event_centered_summaries_from_tables(
        event_values,
        frame_axis,
        field_values,
        tau=tau,
        coords=coords,
        box_length_x=box_length_x,
        field_names=field_names,
    )
    written: dict[str, list[Path]] = {}
    for name, summary in summaries.items():
        written[name] = plot_event_centered_summary(
            summary,
            Path(output_dir) / name,
            prefix=name,
            field_names=field_names,
            title_prefix=name.replace("_", " "),
        )
    return written
