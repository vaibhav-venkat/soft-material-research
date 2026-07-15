from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import chain
import json
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import numpy as np
from safetensors.numpy import load_file
from scipy.ndimage import label
from scipy.spatial import cKDTree

from hexatic.constants import cylinder

from .cases import DEFAULT_OUTPUT_ROOT, CasePaths, get_case


WINDING_NEGATIVE = -1
WINDING_POSITIVE_OUTWARD = 1
WINDING_POSITIVE_INWARD = 2
WINDING_COLORMAP = ListedColormap(("#2166ac", "#d73027", "#f28e2b"))
WINDING_NORM = BoundaryNorm((-1.5, -0.5, 1.5, 2.5), WINDING_COLORMAP.N)
WINDING_CATEGORIES = (
    (WINDING_NEGATIVE, "-1", "#2166ac"),
    (WINDING_POSITIVE_OUTWARD, "+1 outward", "#d73027"),
    (WINDING_POSITIVE_INWARD, "+1 inward", "#f28e2b"),
)


@dataclass(frozen=True)
class WindingDiagnostics:
    frames: np.ndarray
    steps: np.ndarray
    centers: np.ndarray
    unwrapped_centers: np.ndarray
    velocities: np.ndarray
    counts: np.ndarray
    total_charge: np.ndarray


def _frame_numbers(
    manifest: dict[str, object],
    start: int,
    stop: int | None,
    stride: int,
) -> list[int]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("The analysis manifest contains no frame shards")
    available_stop = max(int(shard["frame_stop"]) for shard in shards)
    requested_stop = available_stop if stop is None else min(stop, available_stop)
    if start < 0 or requested_stop <= start:
        raise ValueError(
            f"Empty frame range: start={start}, stop={requested_stop}, "
            f"available_stop={available_stop}"
        )
    return list(range(start, requested_stop, stride))


def _iter_frames(
    analysis_dir: Path,
    manifest: dict[str, object],
    selected_frames: set[int],
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    required = (
        "step",
        "coords",
        "rho",
        "polar_cylindrical",
        "active_shell_mask",
    )
    for shard in manifest["shards"]:
        shard_start = int(shard["frame_start"])
        shard_stop = int(shard["frame_stop"])
        wanted = sorted(
            frame for frame in selected_frames if shard_start <= frame < shard_stop
        )
        if not wanted:
            continue
        shard_path = analysis_dir / str(shard["file"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing frame shard: {shard_path}")
        tensors = load_file(shard_path)
        missing = [name for name in required if name not in tensors]
        if missing:
            raise KeyError(f"{shard_path} is missing tensors: {', '.join(missing)}")
        for frame in wanted:
            local = frame - shard_start
            yield frame, {name: tensors[name][local] for name in required}
        del tensors


def _particle_vectors(
    frame: dict[str, np.ndarray],
    *,
    radius: float,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(frame["coords"], dtype=np.float32)
    polar = np.asarray(frame["polar_cylindrical"], dtype=np.float32)
    rho = np.asarray(frame["rho"], dtype=np.float32)
    stored_film = np.asarray(frame["active_shell_mask"], dtype=np.bool_)
    if stored_film.shape != rho.shape:
        raise ValueError("active_shell_mask must have the same shape as rho")
    particle_diameter = float(cylinder.ANALYSIS.particle_diameter)
    radial_tolerance = max(
        0.01,
        float(cylinder.SIMULATION.wall_clearance_epsilon),
    ) * particle_diameter
    upper_boundary = (
        (coords[:, 2] > radius - particle_diameter)
        & (coords[:, 2] <= radius + radial_tolerance)
    )
    film = stored_film | upper_boundary
    valid = (
        film
        & np.all(np.isfinite(coords), axis=1)
        & np.all(np.isfinite(polar), axis=1)
        & np.isfinite(rho)
    )
    if quantity == "polarization":
        valid &= rho > np.finfo(np.float32).eps
        vectors = np.divide(
            polar[valid],
            rho[valid, None],
            out=np.zeros_like(polar[valid]),
            where=rho[valid, None] > np.finfo(np.float32).eps,
        )
    else:
        vectors = polar[valid]
    return coords[valid, 0], np.mod(coords[valid, 1], 2.0 * np.pi), vectors


def _periodic_grid(
    lx: float,
    circumference: float,
    target_spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    nx = max(1, int(np.ceil(lx / target_spacing)))
    ntheta = max(1, int(np.ceil(circumference / target_spacing)))
    dx = lx / nx
    ds = circumference / ntheta
    x_edges = np.linspace(-0.5 * lx, 0.5 * lx, nx + 1)
    s_edges = np.linspace(0.0, circumference, ntheta + 1)
    x = 0.5 * (x_edges[:-1] + x_edges[1:])
    s = 0.5 * (s_edges[:-1] + s_edges[1:])
    return x, s, x_edges, s_edges, dx, ds


def _circle_offsets(radius: float, dx: float, ds: float) -> tuple[tuple[int, int], ...]:
    sample_count = max(16, int(np.ceil(2.0 * np.pi * radius / min(dx, ds))))
    offsets: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for angle in np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False):
        offset = (
            int(np.rint(radius * np.sin(angle) / ds)),
            int(np.rint(radius * np.cos(angle) / dx)),
        )
        if offset not in seen:
            offsets.append(offset)
            seen.add(offset)
    if len(offsets) < 8:
        raise ValueError(
            f"Winding circle at radius {radius:g} has only {len(offsets)} grid samples"
        )
    return tuple(offsets)


def _interpolate_periodic_polarization(
    particle_x: np.ndarray,
    particle_s: np.ndarray,
    particle_vectors: np.ndarray,
    *,
    grid_x: np.ndarray,
    grid_s: np.ndarray,
    lx: float,
    circumference: float,
    support_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = (len(grid_s), len(grid_x))
    if len(particle_x) < 3:
        return np.zeros((*shape, 2), dtype=np.float64), np.zeros(shape, dtype=np.bool_)

    particle_points = np.column_stack(
        (
            np.mod(particle_x + 0.5 * lx, lx),
            np.mod(particle_s, circumference),
        )
    )
    grid_x_values, grid_s_values = np.meshgrid(
        np.mod(grid_x + 0.5 * lx, lx),
        grid_s,
    )
    grid_points = np.column_stack((grid_x_values.ravel(), grid_s_values.ravel()))
    tree = cKDTree(particle_points, boxsize=(lx, circumference))
    distances, indices = tree.query(grid_points, k=3, workers=-1)
    supported = distances[:, 2] <= support_radius

    distance_floor = np.finfo(np.float64).eps * max(lx, circumference)
    weights = np.reciprocal(np.maximum(distances, distance_floor) ** 2)
    interpolated = np.sum(
        weights[:, :, None] * particle_vectors[indices],
        axis=1,
    ) / np.sum(weights, axis=1)[:, None]
    finite = np.all(np.isfinite(interpolated), axis=1)
    nonzero = np.linalg.norm(interpolated, axis=1) > np.finfo(np.float32).eps
    valid = supported & finite & nonzero
    return interpolated.reshape((*shape, 2)), valid.reshape(shape)


def _winding_for_offsets(
    angles: np.ndarray,
    field: np.ndarray,
    valid: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    *,
    dx: float,
    ds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_angles = [
        np.roll(angles, shift=(-ds_index, -x_index), axis=(0, 1))
        for ds_index, x_index in offsets
    ]
    loop_valid = np.ones(valid.shape, dtype=np.bool_)
    radial_alignment = np.zeros(valid.shape, dtype=np.float64)
    for ds_index, x_index in offsets:
        loop_valid &= np.roll(
            valid,
            shift=(-ds_index, -x_index),
            axis=(0, 1),
        )
        sampled_field = np.roll(
            field,
            shift=(-ds_index, -x_index),
            axis=(0, 1),
        )
        sampled_norm = np.linalg.norm(sampled_field, axis=2)
        radial_x = x_index * dx
        radial_s = ds_index * ds
        radial_norm = np.hypot(radial_x, radial_s)
        radial_alignment += np.divide(
            sampled_field[:, :, 0] * radial_x
            + sampled_field[:, :, 1] * radial_s,
            sampled_norm * radial_norm,
            out=np.zeros(valid.shape, dtype=np.float64),
            where=(sampled_norm > np.finfo(np.float32).eps) & (radial_norm > 0.0),
        )
    radial_alignment /= len(offsets)

    total_turn = np.zeros(angles.shape, dtype=np.float64)
    for current, following in zip(
        sampled_angles,
        sampled_angles[1:] + sampled_angles[:1],
    ):
        difference = following - current
        total_turn += np.arctan2(np.sin(difference), np.cos(difference))
    charges = np.rint(total_turn / (2.0 * np.pi)).astype(np.int16)
    return charges, loop_valid, radial_alignment


def _stable_winding(
    particle_x: np.ndarray,
    particle_theta: np.ndarray,
    particle_vectors: np.ndarray,
    *,
    grid_x: np.ndarray,
    grid_s: np.ndarray,
    circle_offsets: tuple[tuple[tuple[int, int], ...], ...],
    lx: float,
    circumference: float,
    radius: float,
    support_radius: float,
    dx: float,
    ds: float,
    radial_threshold: float,
) -> np.ndarray:
    polarization = particle_vectors[:, (0, 2)]
    field, field_valid = _interpolate_periodic_polarization(
        particle_x,
        radius * particle_theta,
        polarization,
        grid_x=grid_x,
        grid_s=grid_s,
        lx=lx,
        circumference=circumference,
        support_radius=support_radius,
    )
    angles = np.arctan2(field[:, :, 1], field[:, :, 0])
    radius_results = [
        _winding_for_offsets(
            angles,
            field,
            field_valid,
            offsets,
            dx=dx,
            ds=ds,
        )
        for offsets in circle_offsets
    ]

    stable = np.zeros(field_valid.shape, dtype=np.int16)
    best_count = np.zeros(field_valid.shape, dtype=np.int16)
    ambiguous = np.zeros(field_valid.shape, dtype=np.bool_)
    charges = np.stack([result[0] for result in radius_results])
    valid = np.stack([result[1] for result in radius_results])
    radial_alignment = np.stack([result[2] for result in radius_results])
    candidates = np.unique(charges[valid & (charges != 0)])
    for candidate in candidates:
        count = np.count_nonzero(valid & (charges == candidate), axis=0)
        better = count > best_count
        tied = (count == best_count) & (count >= 2) & (stable != candidate)
        stable[better] = candidate
        best_count[better] = count[better]
        ambiguous[better] = False
        ambiguous[tied] = True
    stable[(best_count < 2) | ambiguous] = 0

    classified = np.zeros(field_valid.shape, dtype=np.int8)
    classified[stable == -1] = WINDING_NEGATIVE
    stable_positive = stable == 1
    outward_votes = np.count_nonzero(
        valid & (charges == 1) & (radial_alignment >= radial_threshold),
        axis=0,
    )
    inward_votes = np.count_nonzero(
        valid & (charges == 1) & (radial_alignment <= -radial_threshold),
        axis=0,
    )
    classified[
        stable_positive & (outward_votes >= 2) & (outward_votes > inward_votes)
    ] = WINDING_POSITIVE_OUTWARD
    classified[
        stable_positive & (inward_votes >= 2) & (inward_votes > outward_votes)
    ] = WINDING_POSITIVE_INWARD
    return classified


def _periodic_mean(values: np.ndarray, period: float, origin: float = 0.0) -> float:
    if not len(values):
        return float("nan")
    angles = 2.0 * np.pi * (np.asarray(values) - origin) / period
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    result = origin + np.mod(mean_angle, 2.0 * np.pi) * period / (2.0 * np.pi)
    result = origin + np.mod(result - origin, period)
    if np.isclose(result, origin + period, rtol=0.0, atol=1.0e-12 * period):
        result = origin
    return float(result)


def _periodic_component_centers(
    winding: np.ndarray,
    category: int,
    *,
    grid_x: np.ndarray,
    grid_s: np.ndarray,
    lx: float,
    circumference: float,
) -> np.ndarray:
    mask = winding == category
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)

    parent = np.arange(count + 1, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        if first == 0 or second == 0:
            return
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    n_s, n_x = mask.shape
    for s_index in range(n_s):
        for offset in (-1, 0, 1):
            union(
                int(labels[s_index, 0]),
                int(labels[(s_index + offset) % n_s, n_x - 1]),
            )
    for x_index in range(n_x):
        for offset in (-1, 0, 1):
            union(
                int(labels[0, x_index]),
                int(labels[n_s - 1, (x_index + offset) % n_x]),
            )

    groups: dict[int, list[tuple[int, int]]] = {}
    for s_index, x_index in np.argwhere(mask):
        root = find(int(labels[s_index, x_index]))
        groups.setdefault(root, []).append((int(s_index), int(x_index)))

    centers = np.empty((len(groups), 2), dtype=np.float64)
    for row, cells in enumerate(groups.values()):
        s_indices, x_indices = np.asarray(cells, dtype=np.int64).T
        centers[row, 0] = _periodic_mean(
            grid_x[x_indices],
            lx,
            origin=-0.5 * lx,
        )
        centers[row, 1] = _periodic_mean(grid_s[s_indices], circumference)
    return centers


def _unwrap_periodic_series(values: np.ndarray, period: float) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if index == 0 or not np.isfinite(result[index - 1]):
            result[index] = value
            continue
        previous_wrapped = np.mod(result[index - 1], period)
        delta = np.mod(value - previous_wrapped + 0.5 * period, period) - 0.5 * period
        result[index] = result[index - 1] + delta
    return result


def _build_winding_diagnostics(
    frames: list[int],
    steps: list[int],
    winding_by_frame: dict[int, np.ndarray],
    *,
    grid_x: np.ndarray,
    grid_s: np.ndarray,
    lx: float,
    circumference: float,
) -> WindingDiagnostics:
    frame_values = np.asarray(frames, dtype=np.int64)
    step_values = np.asarray(steps, dtype=np.int64)
    centers = np.full((len(frames), len(WINDING_CATEGORIES), 2), np.nan)
    counts = np.zeros((len(frames), len(WINDING_CATEGORIES)), dtype=np.int64)

    for frame_row, frame_index in enumerate(frames):
        winding = winding_by_frame[frame_index]
        for category_row, (category, _, _) in enumerate(WINDING_CATEGORIES):
            component_centers = _periodic_component_centers(
                winding,
                category,
                grid_x=grid_x,
                grid_s=grid_s,
                lx=lx,
                circumference=circumference,
            )
            counts[frame_row, category_row] = len(component_centers)
            if len(component_centers):
                centers[frame_row, category_row, 0] = _periodic_mean(
                    component_centers[:, 0],
                    lx,
                    origin=-0.5 * lx,
                )
                centers[frame_row, category_row, 1] = _periodic_mean(
                    component_centers[:, 1],
                    circumference,
                )

    unwrapped = centers.copy()
    for category_row in range(len(WINDING_CATEGORIES)):
        x_shifted = centers[:, category_row, 0] + 0.5 * lx
        unwrapped[:, category_row, 0] = (
            _unwrap_periodic_series(x_shifted, lx) - 0.5 * lx
        )
        unwrapped[:, category_row, 1] = _unwrap_periodic_series(
            centers[:, category_row, 1],
            circumference,
        )

    velocities = np.full(unwrapped.shape, np.nan, dtype=np.float64)
    elapsed = np.diff(step_values).astype(np.float64) * float(
        cylinder.SIMULATION.timestep
    )
    if np.any(elapsed <= 0.0):
        raise ValueError("Selected frame steps must be strictly increasing")
    for index in range(1, len(frames)):
        finite = np.all(np.isfinite(unwrapped[index - 1 : index + 1]), axis=0)
        velocities[index, finite] = (
            unwrapped[index, finite] - unwrapped[index - 1, finite]
        ) / elapsed[index - 1]

    total_charge = counts[:, 1] + counts[:, 2] - counts[:, 0]
    return WindingDiagnostics(
        frames=frame_values,
        steps=step_values,
        centers=centers,
        unwrapped_centers=unwrapped,
        velocities=velocities,
        counts=counts,
        total_charge=total_charge,
    )


def _component_output_path(
    output: Path | None,
    output_root: Path,
    case_id: str,
    quantity: str,
    component_name: str,
) -> Path:
    if output is None:
        base = output_root / "movies" / f"{case_id}_{quantity.replace('-', '_')}"
    elif output.suffix.lower() == ".gif":
        base = output.with_suffix("")
    elif output.suffix:
        raise ValueError("--output must end in .gif or have no filename suffix")
    else:
        base = output
    return base.parent / f"{base.name}_{component_name}.gif"


def _diagnostics_output_path(
    output: Path | None,
    diagnostics_output: Path | None,
    output_root: Path,
    case_id: str,
    quantity: str,
) -> Path:
    if diagnostics_output is not None:
        if diagnostics_output.suffix.lower() != ".png":
            raise ValueError("--diagnostics-output must end in .png")
        return diagnostics_output
    if output is None:
        return (
            output_root
            / "plots"
            / f"{case_id}_{quantity.replace('-', '_')}_winding_diagnostics.png"
        )
    base = output.with_suffix("") if output.suffix.lower() == ".gif" else output
    if base.suffix:
        raise ValueError("--output must end in .gif or have no filename suffix")
    return base.parent / f"{base.name}_winding_diagnostics.png"


def _plot_winding_diagnostics(
    diagnostics: WindingDiagnostics,
    output_path: Path,
    *,
    case_label: str,
    stride: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(16.0, 14.0), sharex=True)
    for category_index, (_, label_text, color) in enumerate(WINDING_CATEGORIES):
        axes[0, 0].plot(
            diagnostics.steps,
            diagnostics.unwrapped_centers[:, category_index, 0],
            color=color,
            label=label_text,
        )
        axes[0, 1].plot(
            diagnostics.steps,
            diagnostics.unwrapped_centers[:, category_index, 1],
            color=color,
            label=label_text,
        )
        axes[1, 0].plot(
            diagnostics.steps,
            diagnostics.velocities[:, category_index, 0],
            color=color,
            label=label_text,
        )
        axes[1, 1].plot(
            diagnostics.steps,
            diagnostics.velocities[:, category_index, 1],
            color=color,
            label=label_text,
        )
        axes[2, 0].plot(
            diagnostics.steps,
            diagnostics.counts[:, category_index],
            color=color,
            label=label_text,
        )

    axes[2, 1].plot(
        diagnostics.steps,
        diagnostics.total_charge,
        color="black",
        label=r"$N_{+1,\mathrm{out}}+N_{+1,\mathrm{in}}-N_{-1}$",
    )
    axes[2, 1].axhline(0.0, color="gray", linewidth=1.0, alpha=0.5)
    axes[0, 0].set_ylabel("unwrapped x COM")
    axes[0, 1].set_ylabel(r"unwrapped $R\theta$ COM")
    axes[1, 0].set_ylabel(r"$v_{x,\mathrm{COM}}$")
    axes[1, 1].set_ylabel(r"$v_{R\theta,\mathrm{COM}}$")
    axes[2, 0].set_ylabel("component count")
    axes[2, 1].set_ylabel("total winding charge")
    axes[2, 0].set_xlabel("simulation step")
    axes[2, 1].set_xlabel("simulation step")
    axes[0, 0].set_title("Periodic center of mass: x")
    axes[0, 1].set_title(r"Periodic center of mass: $R\theta$")
    axes[1, 0].set_title("Center-of-mass velocity: x")
    axes[1, 1].set_title(r"Center-of-mass velocity: $R\theta$")
    axes[2, 0].set_title("Detected winding components")
    axes[2, 1].set_title("Net total winding charge")
    for axis in axes.ravel():
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.legend(loc="best")
    figure.suptitle(
        f"{case_label}: winding diagnostics (frame stride {stride})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def write_polarization_movies(
    case_id: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output: Path | None = None,
    diagnostics_output: Path | None = None,
    quantity: str = "polarization",
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    fps: int = 20,
    dpi: int = 150,
    color_max: float | None = None,
    figure_width: float = 20.0,
    figure_height: float = 10.0,
    show_winding: bool = True,
    winding_grid_spacing_d: float = 0.5,
    winding_radii_d: tuple[float, ...] = (1.5, 2.0, 2.5),
    winding_support_radius_d: float = 1.5,
    winding_radial_threshold: float = 0.25,
    skip_gifs: bool = False,
) -> tuple[Path | None, Path | None, Path]:
    if stride < 1 or fps < 1 or dpi < 1:
        raise ValueError("stride, fps, and dpi must be positive")
    if figure_width <= 0.0 or figure_height <= 0.0:
        raise ValueError("figure-width and figure-height must be positive")
    if color_max is not None and color_max <= 0.0:
        raise ValueError("color-max must be positive")
    if winding_grid_spacing_d <= 0.0:
        raise ValueError("winding-grid-spacing-d must be positive")
    if len(winding_radii_d) < 2:
        raise ValueError("winding-radii-d must contain at least two radii")
    if any(radius <= 0.0 for radius in winding_radii_d):
        raise ValueError("winding-radii-d values must be positive")
    if any(
        following <= current
        for current, following in zip(winding_radii_d, winding_radii_d[1:])
    ):
        raise ValueError("winding-radii-d values must be strictly increasing")
    if winding_support_radius_d <= 0.0:
        raise ValueError("winding-support-radius-d must be positive")
    if not 0.0 <= winding_radial_threshold < 1.0:
        raise ValueError("winding-radial-threshold must be in [0, 1)")
    if quantity not in ("polarization", "polar-density"):
        raise ValueError("quantity must be 'polarization' or 'polar-density'")
    case = get_case(case_id)
    paths = CasePaths(case, output_root)
    manifest_path = paths.analysis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.big_lx.analysis.v1":
        raise ValueError(f"Unsupported analysis schema in {manifest_path}")

    frames = _frame_numbers(manifest, start, stop, stride)
    selected = set(frames)
    particle_diameter = float(cylinder.ANALYSIS.particle_diameter)
    film_upper_radius = case.radius + max(
        0.01,
        float(cylinder.SIMULATION.wall_clearance_epsilon),
    ) * particle_diameter
    (
        winding_grid_x,
        winding_grid_s,
        winding_x_edges,
        winding_s_edges,
        winding_dx,
        winding_ds,
    ) = _periodic_grid(
        case.lx,
        case.circumference,
        winding_grid_spacing_d * particle_diameter,
    )
    winding_circle_offsets = tuple(
        _circle_offsets(radius_d * particle_diameter, winding_dx, winding_ds)
        for radius_d in winding_radii_d
    )

    winding_by_frame: dict[int, np.ndarray] = {}
    step_by_frame: dict[int, int] = {}
    for rendered, (frame_index, frame) in enumerate(
        _iter_frames(paths.analysis_dir, manifest, selected),
        start=1,
    ):
        winding_x, winding_theta, winding_vectors = _particle_vectors(
            frame,
            radius=case.radius,
            quantity="polarization",
        )
        if not len(winding_x):
            radial = np.asarray(frame["coords"], dtype=np.float32)[:, 2]
            finite_radial = radial[np.isfinite(radial)]
            observed = (
                f"[{float(np.min(finite_radial)):.6g}, "
                f"{float(np.max(finite_radial)):.6g}]"
                if finite_radial.size
                else "no finite radii"
            )
            raise ValueError(
                f"No plottable film particles in frame {frame_index}; expected "
                f"{case.radius - particle_diameter:.6g} < r <= "
                f"{film_upper_radius:.6g}, observed r={observed}"
            )
        winding_by_frame[frame_index] = _stable_winding(
            winding_x,
            winding_theta,
            winding_vectors,
            grid_x=winding_grid_x,
            grid_s=winding_grid_s,
            circle_offsets=winding_circle_offsets,
            lx=case.lx,
            circumference=case.circumference,
            radius=case.radius,
            support_radius=winding_support_radius_d * particle_diameter,
            dx=winding_dx,
            ds=winding_ds,
            radial_threshold=winding_radial_threshold,
        )
        step_by_frame[frame_index] = int(frame["step"])
        print(
            f"[big_lx.winding] frame {frame_index} ({rendered}/{len(frames)})",
            flush=True,
        )
    missing_frames = [frame for frame in frames if frame not in winding_by_frame]
    if missing_frames:
        raise RuntimeError(f"Missing requested winding frames: {missing_frames}")

    diagnostics = _build_winding_diagnostics(
        frames,
        [step_by_frame[frame] for frame in frames],
        winding_by_frame,
        grid_x=winding_grid_x,
        grid_s=winding_grid_s,
        lx=case.lx,
        circumference=case.circumference,
    )
    diagnostics_path = _diagnostics_output_path(
        output,
        diagnostics_output,
        output_root,
        case_id,
        quantity,
    )
    _plot_winding_diagnostics(
        diagnostics,
        diagnostics_path,
        case_label=case.label,
        stride=stride,
        dpi=max(dpi, 150),
    )
    print(f"[big_lx.winding] wrote {diagnostics_path}", flush=True)
    if skip_gifs:
        return None, None, diagnostics_path

    # Saved cylindrical order is (x, radial, azimuthal), so Ptheta is index 2.
    movies = (
        ("px_ptheta", r"$(P_x,P_\theta)$", "in_film"),
        ("px", r"$P_x$", "x_only"),
    )
    outputs: list[Path] = []
    for component_name, component_label, vector_mode in movies:
        output_path = _component_output_path(
            output, output_root, case_id, quantity, component_name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)
        limit = color_max or (1.0 if quantity == "polarization" else None)
        norm = Normalize(vmin=0.0, vmax=limit)
        fig, axis = plt.subplots(figsize=(figure_width, figure_height))
        rendered = 0
        try:
            frame_iterator = _iter_frames(paths.analysis_dir, manifest, selected)
            try:
                first_frame = next(frame_iterator)
            except StopIteration as error:
                raise RuntimeError("No requested frames were found in the shards") from error
            first_index, first_values = first_frame
            first_x, _, _ = _particle_vectors(
                first_values,
                radius=case.radius,
                quantity=quantity,
            )
            if not first_x.size:
                radial = np.asarray(first_values["coords"], dtype=np.float32)[:, 2]
                finite_radial = radial[np.isfinite(radial)]
                observed = (
                    f"[{float(np.min(finite_radial)):.6g}, "
                    f"{float(np.max(finite_radial)):.6g}]"
                    if finite_radial.size
                    else "no finite radii"
                )
                raise ValueError(
                    f"No plottable film particles in frame {first_index}; expected "
                    f"{case.radius - particle_diameter:.6g} < r <= "
                    f"{film_upper_radius:.6g}, observed r={observed}"
                )
            with writer.saving(fig, str(output_path), dpi=dpi):
                for frame_index, frame in chain((first_frame,), frame_iterator):
                    particle_x, particle_theta, vectors = _particle_vectors(
                        frame,
                        radius=case.radius,
                        quantity=quantity,
                    )
                    if not particle_x.size:
                        radial = np.asarray(frame["coords"], dtype=np.float32)[:, 2]
                        finite_radial = radial[np.isfinite(radial)]
                        observed = (
                            f"[{float(np.min(finite_radial)):.6g}, "
                            f"{float(np.max(finite_radial)):.6g}]"
                            if finite_radial.size
                            else "no finite radii"
                        )
                        raise ValueError(
                            f"No plottable film particles in frame {frame_index}; "
                            f"expected {case.radius - particle_diameter:.6g} < r <= "
                            f"{film_upper_radius:.6g}, "
                            f"observed r={observed}"
                        )
                    px_values = vectors[:, 0]
                    if vector_mode == "in_film":
                        py_values = vectors[:, 2]
                        colors = np.hypot(px_values, py_values)
                    else:
                        py_values = np.zeros_like(px_values)
                        colors = np.abs(px_values)
                    if norm.vmax is None:
                        finite = colors[np.isfinite(colors)]
                        limit = (
                            float(np.percentile(finite, 99.5))
                            if finite.size
                            else 1.0
                        )
                        if limit <= 0.0:
                            limit = 1.0
                        norm.vmax = limit

                    axis.clear()
                    winding_mesh = None
                    if show_winding and vector_mode == "in_film":
                        winding = winding_by_frame[frame_index]
                        winding_mesh = axis.pcolormesh(
                            winding_x_edges,
                            winding_s_edges,
                            np.ma.masked_equal(winding, 0),
                            cmap=WINDING_COLORMAP,
                            norm=WINDING_NORM,
                            shading="flat",
                            alpha=0.45,
                            zorder=0,
                        )
                    sampled_norm = np.hypot(px_values, py_values)
                    arrow_valid = sampled_norm > np.finfo(np.float32).eps
                    arrow_u = np.divide(
                        px_values,
                        sampled_norm,
                        out=np.zeros_like(px_values),
                        where=arrow_valid,
                    )
                    arrow_v = np.divide(
                        py_values,
                        sampled_norm,
                        out=np.zeros_like(py_values),
                        where=arrow_valid,
                    )
                    quiver = axis.quiver(
                        particle_x[arrow_valid],
                        case.radius * particle_theta[arrow_valid],
                        arrow_u[arrow_valid],
                        arrow_v[arrow_valid],
                        colors[arrow_valid],
                        cmap="viridis",
                        norm=norm,
                        alpha=0.95,
                        angles="uv",
                        scale_units="inches",
                        scale=6.5,
                        width=0.0012,
                        headwidth=4.0,
                        headlength=5.0,
                        headaxislength=4.5,
                        pivot="middle",
                        zorder=2,
                    )
                    axis.set_xlim(-0.5 * case.lx, 0.5 * case.lx)
                    axis.set_ylim(0.0, 2.0 * np.pi * case.radius)
                    axis.set_xlabel("x")
                    axis.set_ylabel(r"$R_{\mathrm{case}}\theta$")
                    axis.set_title(
                        f"{case.label}: {component_label} magnitude and direction "
                        f"(frame {frame_index}, step {int(frame['step'])})"
                    )
                    if rendered == 0:
                        colorbar = fig.colorbar(quiver, ax=axis, pad=0.01)
                        suffix = "/rho" if quantity == "polarization" else ""
                        if vector_mode == "in_film":
                            colorbar.set_label(f"sqrt(Px^2 + Ptheta^2) {suffix}")
                        else:
                            colorbar.set_label(f"abs(Px) {suffix}")
                        if winding_mesh is not None:
                            winding_colorbar = fig.colorbar(
                                winding_mesh,
                                ax=axis,
                                pad=0.06,
                            )
                            winding_colorbar.set_label("winding classification")
                            winding_colorbar.set_ticks((-1, 1, 2))
                            winding_colorbar.set_ticklabels(
                                ("-1", "+1 outward", "+1 inward")
                            )
                    fig.tight_layout()
                    writer.grab_frame()
                    rendered += 1
                    print(
                        f"[big_lx.movie] {component_name}: frame {frame_index} "
                        f"({rendered}/{len(frames)})",
                        flush=True,
                    )
        finally:
            plt.close(fig)
        if rendered != len(frames):
            raise RuntimeError(
                f"Rendered {rendered} of {len(frames)} requested frames"
            )
        outputs.append(output_path)
        print(f"[big_lx.movie] wrote {output_path}", flush=True)
    return outputs[0], outputs[1], diagnostics_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render polarization GIFs and winding COM, velocity, count, and total "
            "charge diagnostics on the unwrapped film."
        )
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output filename prefix; a trailing .gif is removed before component names.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Optional .png path for the winding diagnostics figure.",
    )
    parser.add_argument(
        "--quantity",
        choices=("polarization", "polar-density"),
        default="polarization",
        help="Plot P/rho (default) or the stored polar density P.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--color-max", type=float)
    parser.add_argument("--figure-width", type=float, default=20.0)
    parser.add_argument("--figure-height", type=float, default=10.0)
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Write only the winding diagnostics PNG and skip both GIFs.",
    )
    parser.add_argument(
        "--no-winding",
        action="store_true",
        help="Disable the signed winding-number overlay on the Px/Ptheta movie.",
    )
    parser.add_argument("--winding-grid-spacing-d", type=float, default=0.5)
    parser.add_argument(
        "--winding-radii-d",
        type=float,
        nargs="+",
        default=(1.5, 2.0, 2.5),
        metavar="R_OVER_D",
    )
    parser.add_argument("--winding-support-radius-d", type=float, default=1.5)
    parser.add_argument(
        "--winding-radial-threshold",
        type=float,
        default=0.25,
        help=(
            "Minimum absolute mean radial alignment for classifying a +1 winding "
            "as outward or inward."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_polarization_movies(
        args.case,
        output_root=args.output_root,
        output=args.output,
        diagnostics_output=args.diagnostics_output,
        quantity=args.quantity,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        fps=args.fps,
        dpi=args.dpi,
        color_max=args.color_max,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        show_winding=not args.no_winding,
        winding_grid_spacing_d=args.winding_grid_spacing_d,
        winding_radii_d=tuple(args.winding_radii_d),
        winding_support_radius_d=args.winding_support_radius_d,
        winding_radial_threshold=args.winding_radial_threshold,
        skip_gifs=args.skip_gifs,
    )


if __name__ == "__main__":
    main()
