"""Plot particle-centered freud local-density distributions for big-Lx cases.

``freud.density.LocalDensity`` is evaluated at every particle center in every
selected frame.  Query points are therefore never empty grid locations.
Particles marked by the stored ``hexatic_shell_mask`` receive a factor of two
to correct the missing outer hemisphere.  A radial shell-mask fallback uses
the repository shell cutoff if the stored mask is absent.

Because query points are sampled at particles, the empirical histogram is
converted to an approximate volume distribution with inverse-density weights
``w_i = 1/rho_i``.  Histogram fractions are reported for comparison, while
the primary phase fractions still come from global conservation.

The two modes of the empirical ``P(rho)`` define ``rho_beta`` (dilute) and
``rho_alpha`` (dense).  Fractions are calculated only from conservation,

``x_alpha = (rho_total - rho_beta) / (rho_alpha - rho_beta)``.

Cases run in parallel CPU processes.  The sole output is one SVG per case.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path

import freud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from hexatic.constants import cylinder

from .run_dynamics import Replicate, _collect_manifests, _load_replicate, _style
from .surface_area import _load_circumference, _load_static_lx


@dataclass(frozen=True)
class ParticleDensitySamples:
    values: NDArray[np.float64]
    particle_count_sum: float
    n_frames: int
    lx: float
    center_radius: float


@dataclass(frozen=True)
class DensityModes:
    rho_beta: float
    rho_alpha: float
    boundary: float


@dataclass(frozen=True)
class CaseOptions:
    output_dir: Path
    overwrite: bool
    particle_diameter: float
    neighborhood_radius: float
    wall_offset: float
    shell_delta: float
    density_bins: int
    mode_smoothing_window: int
    mode_prominence_fraction: float
    frame_start: int
    frame_stride: int


def _safe_case_name(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in case_id)


def _circumference_prefix(value: str) -> str:
    token = value.strip().upper()
    if token.endswith("D"):
        token = token[:-1]
    try:
        circumference = float(token)
    except ValueError as error:
        raise ValueError('--circ must look like "60.5D"') from error
    return f"circ_{format(circumference, '.15g').replace('.', '_')}D_"


def _load_radius(static_file: Path, circumference: float) -> float:
    with safe_open(static_file, framework="np") as tensors:
        radius = (
            float(tensors.get_tensor("radius").item())
            if "radius" in tensors.keys()
            else circumference / (2.0 * np.pi)
        )
    if not np.isclose(circumference, 2.0 * np.pi * radius, rtol=1e-6, atol=1e-8):
        raise ValueError(
            f"Manifest circumference and static radius disagree for {static_file}"
        )
    return radius


def _select_replicates(
    manifests: list[Path],
    input_dirs: list[Path],
    cases: list[str],
    circumference: str | None,
) -> list[Replicate]:
    paths = _collect_manifests(manifests, input_dirs)
    replicates = [_load_replicate(path) for path in paths]
    if cases:
        selected = set(cases)
        replicates = [r for r in replicates if r.case_id in selected]
        missing = selected - {r.case_id for r in replicates}
        if missing:
            raise ValueError(f"Requested cases not found: {sorted(missing)}")
    if circumference:
        prefix = _circumference_prefix(circumference)
        replicates = [r for r in replicates if r.case_id.startswith(prefix)]
    if not replicates:
        raise ValueError("No matching replicates found")
    return replicates


def _frame_shell_mask(
    tensors: object,
    local_frame: int,
    radial: NDArray[np.float32],
    cylinder_radius: float,
    shell_delta: float,
) -> NDArray[np.bool_]:
    keys = tensors.keys()  # type: ignore[attr-defined]
    if "hexatic_shell_mask" in keys:
        return np.asarray(
            tensors.get_tensor("hexatic_shell_mask")[local_frame],  # type: ignore[attr-defined]
            dtype=np.bool_,
        )
    return np.asarray(radial > cylinder_radius - shell_delta, dtype=np.bool_)


def _load_particle_density_samples(
    replicate: Replicate, options: CaseOptions
) -> ParticleDensitySamples:
    lx = _load_static_lx(replicate.static_file)
    circumference = _load_circumference(replicate.manifest)
    radius = _load_radius(replicate.static_file, circumference)
    center_radius = radius - 0.5 * options.particle_diameter - options.wall_offset
    if center_radius <= 0.0:
        raise ValueError(f"No accessible particle-center volume for {replicate.case_id}")

    densities: list[NDArray[np.float64]] = []
    particle_count_sum = 0.0
    n_frames = 0
    global_frame = 0
    local_density = freud.density.LocalDensity(
        r_max=options.neighborhood_radius,
        diameter=options.particle_diameter,
    )
    self_density = 1.0 / (
        (4.0 / 3.0) * np.pi * options.neighborhood_radius**3
    )
    transverse_box_length = 2.0 * (
        radius + options.neighborhood_radius + options.particle_diameter
    )
    box = freud.box.Box(
        Lx=lx,
        Ly=transverse_box_length,
        Lz=transverse_box_length,
    )

    for path in replicate.shard_files:
        with safe_open(path, framework="np") as tensors:
            coords = tensors.get_tensor("coords")
            for local_frame in range(coords.shape[0]):
                selected = (
                    global_frame >= options.frame_start
                    and (global_frame - options.frame_start)
                    % options.frame_stride
                    == 0
                )
                global_frame += 1
                if not selected:
                    continue
                frame = coords[local_frame]
                finite = (
                    np.isfinite(frame[:, 0])
                    & np.isfinite(frame[:, 1])
                    & np.isfinite(frame[:, 2])
                    & (frame[:, 2] >= 0.0)
                )
                frame = frame[finite]
                radial = np.asarray(frame[:, 2], dtype=np.float32)
                x = np.mod(frame[:, 0] + 0.5 * lx, lx) - 0.5 * lx
                theta = frame[:, 1]
                points = np.column_stack(
                    (x, radial * np.cos(theta), radial * np.sin(theta))
                ).astype(np.float32)
                result = local_density.compute(system=(box, points))
                frame_density = np.asarray(result.density, dtype=np.float64).copy()
                # freud excludes a query point from its own neighbor list.
                # Add that particle explicitly so particle-centered queries
                # cannot produce the empty-neighborhood rho=0 artifact.
                frame_density += self_density
                full_shell_mask = _frame_shell_mask(
                    tensors,
                    local_frame,
                    np.asarray(coords[local_frame, :, 2]),
                    radius,
                    options.shell_delta,
                )
                shell_mask = full_shell_mask[finite]
                frame_density[shell_mask] *= 2.0
                densities.append(frame_density)
                particle_count_sum += float(points.shape[0])
                n_frames += 1
    if not densities:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    return ParticleDensitySamples(
        values=np.concatenate(densities),
        particle_count_sum=particle_count_sum,
        n_frames=n_frames,
        lx=lx,
        center_radius=center_radius,
    )


def _combine_samples(
    case_id: str, samples: list[ParticleDensitySamples]
) -> ParticleDensitySamples:
    reference = samples[0]
    for sample in samples[1:]:
        if not np.isclose(sample.lx, reference.lx) or not np.isclose(
            sample.center_radius, reference.center_radius
        ):
            raise ValueError(f"Replicates for {case_id} have different geometry")
    return ParticleDensitySamples(
        values=np.concatenate([sample.values for sample in samples]),
        particle_count_sum=sum(sample.particle_count_sum for sample in samples),
        n_frames=sum(sample.n_frames for sample in samples),
        lx=reference.lx,
        center_radius=reference.center_radius,
    )


def _empirical_distribution_and_modes(
    values: NDArray[np.float64],
    volume_weights: NDArray[np.float64],
    density_bins: int,
    smoothing_window: int,
    prominence_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], DensityModes]:
    probability, edges = np.histogram(
        values,
        bins=density_bins,
        weights=volume_weights,
        density=True,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = uniform_filter1d(probability, smoothing_window, mode="nearest")
    baseline = float(smoothed.min())
    peaks, properties = find_peaks(
        np.concatenate(([baseline], smoothed, [baseline])),
        prominence=prominence_fraction * float(smoothed.max()),
        distance=max(2, density_bins // 10),
    )
    peaks = peaks - 1
    valid = (peaks >= 0) & (peaks < density_bins)
    peaks = peaks[valid]
    prominences = properties["prominences"][valid]
    if peaks.size < 2:
        raise ValueError(
            "Fewer than two empirical density modes were found; adjust "
            "--density-bins, --mode-smoothing-window, or "
            "--mode-prominence-fraction"
        )
    selected = peaks[np.argsort(prominences)[-2:]]
    selected.sort()
    valley_slice = slice(selected[0], selected[1] + 1)
    valley_index = selected[0] + int(np.argmin(smoothed[valley_slice]))
    modes = DensityModes(
        rho_beta=float(centers[selected[0]]),
        rho_alpha=float(centers[selected[1]]),
        boundary=float(centers[valley_index]),
    )
    return edges, probability, smoothed, modes


def _plot_distribution(
    case_label: str,
    edges: NDArray[np.float64],
    probability: NDArray[np.float64],
    smoothed_probability: NDArray[np.float64],
    rho_total: float,
    modes: DensityModes,
    x_alpha: float,
    x_beta: float,
    x_alpha_histogram: float,
    x_beta_histogram: float,
    output: Path,
) -> None:
    sns = _style()
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    axis.stairs(
        probability,
        edges,
        fill=True,
        alpha=0.3,
        color="0.45",
        label=r"Inverse-density weighted $P_V(\rho)$",
    )
    axis.plot(
        centers,
        smoothed_probability,
        color="black",
        linewidth=1.8,
        label="Moving-average empirical PDF (mode finding)",
    )
    colors = sns.color_palette("colorblind", n_colors=2)
    axis.axvline(
        modes.rho_beta,
        color=colors[0],
        linestyle="--",
        linewidth=1.3,
        label=(
            rf"$\rho_\beta={modes.rho_beta:.4g}$, "
            rf"$x_\beta^{{\rm cons}}={x_beta:.3f}$, "
            rf"$x_\beta^{{\rm hist}}={x_beta_histogram:.3f}$"
        ),
    )
    axis.axvline(
        modes.rho_alpha,
        color=colors[1],
        linestyle="--",
        linewidth=1.3,
        label=(
            rf"$\rho_\alpha={modes.rho_alpha:.4g}$, "
            rf"$x_\alpha^{{\rm cons}}={x_alpha:.3f}$, "
            rf"$x_\alpha^{{\rm hist}}={x_alpha_histogram:.3f}$"
        ),
    )
    axis.axvline(
        modes.boundary,
        color="0.5",
        linestyle="-.",
        linewidth=1.0,
        label=rf"population boundary $={modes.boundary:.4g}$",
    )
    axis.axvline(
        rho_total,
        color="0.25",
        linestyle=":",
        linewidth=1.1,
        label=rf"$\rho_{{\rm total}}={rho_total:.4g}$",
    )
    axis.set(
        title=case_label,
        xlabel=r"Particle-centered local density $\rho$ [$N/L^3$]",
        ylabel=r"Volume-weighted probability density $P_V(\rho)$",
        xlim=(edges[0], edges[-1]),
    )
    axis.grid(axis="y", color="0.9", linewidth=0.7)
    axis.legend(frameon=False)
    sns.despine(ax=axis)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.big_lx_analysis.density_distribution"},
    )
    plt.close(figure)


def _analyze_case(task: tuple[str, list[Replicate], CaseOptions]) -> tuple[str, Path]:
    case_id, case_replicates, options = task
    samples = _combine_samples(
        case_id,
        [_load_particle_density_samples(r, options) for r in case_replicates],
    )
    if np.any(samples.values <= 0.0):
        raise ValueError(
            f"Particle-centered densities must be positive for {case_id}"
        )
    volume_weights = 1.0 / samples.values
    volume = np.pi * samples.center_radius**2 * samples.lx
    rho_total = samples.particle_count_sum / (samples.n_frames * volume)
    edges, probability, smoothed, modes = _empirical_distribution_and_modes(
        samples.values,
        volume_weights,
        options.density_bins,
        options.mode_smoothing_window,
        options.mode_prominence_fraction,
    )
    difference = modes.rho_alpha - modes.rho_beta
    x_alpha = (rho_total - modes.rho_beta) / difference
    x_beta = 1.0 - x_alpha
    dense_samples = samples.values > modes.boundary
    total_weight = float(volume_weights.sum())
    x_alpha_histogram = float(volume_weights[dense_samples].sum() / total_weight)
    x_beta_histogram = 1.0 - x_alpha_histogram
    if not 0.0 <= x_alpha <= 1.0:
        raise ValueError(
            f"Detected modes do not bracket rho_total for {case_id}: "
            f"rho_beta={modes.rho_beta}, rho_total={rho_total}, "
            f"rho_alpha={modes.rho_alpha}"
        )
    output = options.output_dir / f"{_safe_case_name(case_id)}_density_distribution.svg"
    if output.exists() and not options.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace")
    _plot_distribution(
        case_replicates[0].label,
        edges,
        probability,
        smoothed,
        rho_total,
        modes,
        x_alpha,
        x_beta,
        x_alpha_histogram,
        x_beta_histogram,
        output,
    )
    return (
        f"{case_id}: rho_total={rho_total:.6g}, rho_beta={modes.rho_beta:.6g}, "
        f"rho_alpha={modes.rho_alpha:.6g}, "
        f"x_beta_cons={x_beta:.6g}, x_alpha_cons={x_alpha:.6g}, "
        f"x_beta_hist={x_beta_histogram:.6g}, "
        f"x_alpha_hist={x_alpha_histogram:.6g}",
        output,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("density_distribution_plots")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--particle-diameter",
        type=float,
        default=float(cylinder.PARTICLE_DIAMETER),
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=float,
        default=None,
        help="freud LocalDensity r_max (default: 2.5D)",
    )
    parser.add_argument(
        "--wall-offset",
        type=float,
        default=1.0e-3,
        help="Small inward offset in the particle-center volume (default: 1e-3)",
    )
    parser.add_argument(
        "--shell-delta",
        type=float,
        default=float(cylinder.SHELL_DELTA),
        help="Radial shell fallback when hexatic_shell_mask is absent",
    )
    parser.add_argument("--density-bins", type=int, default=80)
    parser.add_argument("--mode-smoothing-window", type=int, default=5)
    parser.add_argument("--mode-prominence-fraction", type=float, default=0.02)
    parser.add_argument("--frame-start", type=int, default=700)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--circ")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frame_start < 0 or args.frame_stride < 1:
        raise ValueError("Frame start must be non-negative and stride positive")
    if args.density_bins < 3 or args.mode_smoothing_window < 1:
        raise ValueError("Density bins and smoothing window must be positive")
    if args.workers < 0 or not 0.0 <= args.mode_prominence_fraction < 1.0:
        raise ValueError("Workers/prominence options are invalid")
    diameter = args.particle_diameter
    neighborhood_radius = (
        2.5 * diameter
        if args.neighborhood_radius is None
        else args.neighborhood_radius
    )
    if min(diameter, neighborhood_radius, args.shell_delta) <= 0 or args.wall_offset < 0:
        raise ValueError("Particle, neighborhood, and shell lengths must be valid")

    replicates = _select_replicates(
        args.manifest, args.input_dir, args.case, args.circ
    )
    grouped: dict[str, list[Replicate]] = defaultdict(list)
    for replicate in replicates:
        grouped[replicate.case_id].append(replicate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    options = CaseOptions(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        particle_diameter=diameter,
        neighborhood_radius=neighborhood_radius,
        wall_offset=args.wall_offset,
        shell_delta=args.shell_delta,
        density_bins=args.density_bins,
        mode_smoothing_window=args.mode_smoothing_window,
        mode_prominence_fraction=args.mode_prominence_fraction,
        frame_start=args.frame_start,
        frame_stride=args.frame_stride,
    )
    tasks = [
        (case_id, case_replicates, options)
        for case_id, case_replicates in sorted(grouped.items())
    ]
    worker_count = (
        min(len(tasks), os.cpu_count() or 1)
        if args.workers == 0
        else min(len(tasks), args.workers)
    )
    print(f"analyzing {len(tasks)} cases with {worker_count} workers", flush=True)
    if worker_count == 1:
        results = map(_analyze_case, tasks)
        for report, output in results:
            print(report, flush=True)
            print(f"wrote {output}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for report, output in executor.map(_analyze_case, tasks):
                print(report, flush=True)
                print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
