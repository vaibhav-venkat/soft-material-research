"""Plot volumetric-density distributions for big-Lx cylinder cases.

Selected frames are divided into equal-volume ``(x, theta, r)`` voxels.  Each
voxel's local density is time-averaged before constructing the distribution,
so transient empty-bin counting noise does not create an artificial delta peak
at zero.  Dilute and dense densities are the two modes of the empirical
probability density, with no parametric mixture fit.  Their volume fractions
are calculated directly from the conservation lever rule

``rho_total = x_dense * rho_dense + (1 - x_dense) * rho_dilute``.

The sole generated artifact is one SVG probability-density plot per case.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from hexatic.constants import cylinder

from .run_dynamics import Replicate, _collect_manifests, _load_replicate, _style
from .surface_area import _load_circumference, _load_static_lx


@dataclass(frozen=True)
class VolumeDensitySamples:
    values: NDArray[np.float64]
    n_frames: int
    n_particles: int
    nx: int
    ntheta: int
    nr: int
    dx: float
    dtheta: float
    radial_edges: NDArray[np.float64]
    voxel_volume: float
    lx: float
    radius: float


@dataclass(frozen=True)
class DensityModes:
    rho_dilute: float
    rho_dense: float


@dataclass(frozen=True)
class CaseOptions:
    output_dir: Path
    overwrite: bool
    requested_dx: float
    requested_arc_width: float
    radial_bins: int
    density_bins: int
    mode_smoothing_sigma: float
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


def _volume_grid(
    lx: float,
    circumference: float,
    radius: float,
    requested_dx: float,
    requested_arc_width: float,
    radial_bins: int,
) -> tuple[int, int, float, float, NDArray[np.float64], float]:
    if requested_dx <= 0.0 or requested_arc_width <= 0.0:
        raise ValueError("Spatial bin widths must be positive")
    if radial_bins < 1:
        raise ValueError("radial_bins must be positive")
    nx = max(1, int(np.rint(lx / requested_dx)))
    ntheta = max(1, int(np.rint(circumference / requested_arc_width)))
    dx = lx / nx
    dtheta = 2.0 * np.pi / ntheta

    # Uniform spacing in r^2 makes every radial cylindrical sector have the
    # same volume, despite the factor of r in the cylindrical volume element.
    radial_edges = radius * np.sqrt(
        np.linspace(0.0, 1.0, radial_bins + 1, dtype=np.float64)
    )
    radial_sector_volume = 0.5 * dx * dtheta * (
        radial_edges[1] ** 2 - radial_edges[0] ** 2
    )
    return nx, ntheta, dx, dtheta, radial_edges, radial_sector_volume


def _load_volume_density_samples(
    replicate: Replicate,
    *,
    requested_dx: float,
    requested_arc_width: float,
    radial_bins: int,
    frame_start: int,
    frame_stride: int,
) -> VolumeDensitySamples:
    lx = _load_static_lx(replicate.static_file)
    circumference = _load_circumference(replicate.manifest)
    radius = _load_radius(replicate.static_file, circumference)
    nx, ntheta, dx, dtheta, radial_edges, voxel_volume = _volume_grid(
        lx,
        circumference,
        radius,
        requested_dx,
        requested_arc_width,
        radial_bins,
    )
    accumulated_counts = np.zeros(
        nx * ntheta * radial_bins, dtype=np.float64
    )
    n_frames = 0
    n_particles: int | None = None
    global_frame = 0
    for path in replicate.shard_files:
        with safe_open(path, framework="np") as tensors:
            coords = tensors.get_tensor("coords")
            if n_particles is None:
                n_particles = int(coords.shape[1])
            elif coords.shape[1] != n_particles:
                raise ValueError(
                    f"Particle count changes between shards for {replicate.case_id}"
                )
            for local_frame in range(coords.shape[0]):
                selected = (
                    global_frame >= frame_start
                    and (global_frame - frame_start) % frame_stride == 0
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
                    & (frame[:, 2] <= radius)
                )
                x = np.mod(frame[finite, 0] + 0.5 * lx, lx)
                theta = np.mod(frame[finite, 1], 2.0 * np.pi)
                radial = frame[finite, 2]
                x_index = np.minimum((x / dx).astype(np.intp), nx - 1)
                theta_index = np.minimum(
                    (theta / dtheta).astype(np.intp), ntheta - 1
                )
                radial_index = np.searchsorted(
                    radial_edges, radial, side="right"
                ) - 1
                radial_index = np.minimum(radial_index, radial_bins - 1)
                flat_index = (
                    (x_index * ntheta + theta_index) * radial_bins
                    + radial_index
                )
                counts = np.bincount(
                    flat_index,
                    minlength=nx * ntheta * radial_bins,
                )
                accumulated_counts += counts
                n_frames += 1
    if n_frames == 0 or n_particles is None:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    return VolumeDensitySamples(
        values=accumulated_counts / (n_frames * voxel_volume),
        n_frames=n_frames,
        n_particles=n_particles,
        nx=nx,
        ntheta=ntheta,
        nr=radial_bins,
        dx=dx,
        dtheta=dtheta,
        radial_edges=radial_edges,
        voxel_volume=voxel_volume,
        lx=lx,
        radius=radius,
    )


def _combine_samples(
    case_id: str, samples: list[VolumeDensitySamples]
) -> VolumeDensitySamples:
    reference = samples[0]
    for sample in samples[1:]:
        if (
            sample.nx != reference.nx
            or sample.ntheta != reference.ntheta
            or sample.nr != reference.nr
            or not np.isclose(sample.dx, reference.dx)
            or not np.isclose(sample.dtheta, reference.dtheta)
            or not np.allclose(sample.radial_edges, reference.radial_edges)
            or not np.isclose(sample.voxel_volume, reference.voxel_volume)
            or not np.isclose(sample.lx, reference.lx)
            or not np.isclose(sample.radius, reference.radius)
        ):
            raise ValueError(f"Replicates for {case_id} have different grids")
        if sample.n_particles != reference.n_particles:
            raise ValueError(
                f"Replicates for {case_id} have different particle counts"
            )
    return VolumeDensitySamples(
        values=np.concatenate([sample.values for sample in samples]),
        n_frames=sum(sample.n_frames for sample in samples),
        n_particles=reference.n_particles,
        nx=reference.nx,
        ntheta=reference.ntheta,
        nr=reference.nr,
        dx=reference.dx,
        dtheta=reference.dtheta,
        radial_edges=reference.radial_edges,
        voxel_volume=reference.voxel_volume,
        lx=reference.lx,
        radius=reference.radius,
    )


def _empirical_distribution_and_modes(
    values: NDArray[np.float64],
    density_bins: int,
    smoothing_sigma: float,
    prominence_fraction: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    DensityModes,
]:
    """Return the empirical PDF, its smoothed copy, and its two modes."""
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        raise ValueError("At least four finite density samples are required")
    probability, edges = np.histogram(finite, bins=density_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(
        probability.astype(np.float64),
        sigma=smoothing_sigma,
        mode="nearest",
    )
    minimum_prominence = prominence_fraction * float(smoothed.max())
    baseline = float(smoothed.min())
    padded = np.concatenate(([baseline], smoothed, [baseline]))
    padded_peaks, properties = find_peaks(
        padded,
        prominence=minimum_prominence,
        distance=max(2, density_bins // 10),
    )
    peaks = padded_peaks - 1
    valid = (peaks >= 0) & (peaks < density_bins)
    peaks = peaks[valid]
    prominences = properties["prominences"][valid]
    if peaks.size < 2:
        raise ValueError(
            "Fewer than two empirical density modes were found; adjust "
            "--density-bins, --mode-smoothing-sigma, or "
            "--mode-prominence-fraction"
        )
    selected = peaks[np.argsort(prominences)[-2:]]
    selected.sort()
    modes = DensityModes(
        rho_dilute=float(centers[selected[0]]),
        rho_dense=float(centers[selected[1]]),
    )
    return edges, probability, smoothed, modes


def _plot_distribution(
    case_label: str,
    edges: NDArray[np.float64],
    probability: NDArray[np.float64],
    smoothed_probability: NDArray[np.float64],
    rho_total: float,
    modes: DensityModes,
    x_dense: float,
    x_dilute: float,
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
        label=r"Empirical $P(\rho)$",
    )
    axis.plot(
        centers,
        smoothed_probability,
        color="black",
        linewidth=1.8,
        label="Smoothed empirical PDF (mode finding)",
    )
    colors = sns.color_palette("colorblind", n_colors=2)
    axis.axvline(
        modes.rho_dilute,
        color=colors[0],
        linestyle="--",
        linewidth=1.3,
        label=(
            rf"$\rho_{{\rm dilute}}={modes.rho_dilute:.4g}$, "
            rf"$x_{{\rm dilute}}={x_dilute:.3f}$"
        ),
    )
    axis.axvline(
        modes.rho_dense,
        color=colors[1],
        linestyle="--",
        linewidth=1.3,
        label=(
            rf"$\rho_{{\rm dense}}={modes.rho_dense:.4g}$, "
            rf"$x_{{\rm dense}}={x_dense:.3f}$"
        ),
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
        xlabel=r"Local volume density $\rho$ [$N/L^3$]",
        ylabel=r"Probability density $P(\rho)$ (log scale)",
        xlim=(edges[0], edges[-1]),
    )
    positive = np.concatenate(
        (
            probability[probability > 0.0],
            smoothed_probability[smoothed_probability > 0.0],
        )
    )
    if positive.size:
        axis.set_yscale("log")
        axis.set_ylim(
            max(float(positive.min()) * 0.5, float(positive.max()) * 1.0e-7),
            float(positive.max()) * 1.5,
        )
    axis.grid(axis="y", which="both", color="0.9", linewidth=0.7)
    axis.legend(frameon=False)
    sns.despine(ax=axis)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.big_lx_analysis.density_distribution"},
    )
    plt.close(figure)


def _analyze_case(
    task: tuple[str, list[Replicate], CaseOptions],
) -> tuple[str, Path]:
    """Analyze one case in an isolated worker process."""
    case_id, case_replicates, options = task
    samples = _combine_samples(
        case_id,
        [
            _load_volume_density_samples(
                replicate,
                requested_dx=options.requested_dx,
                requested_arc_width=options.requested_arc_width,
                radial_bins=options.radial_bins,
                frame_start=options.frame_start,
                frame_stride=options.frame_stride,
            )
            for replicate in case_replicates
        ],
    )
    rho_total = float(samples.values.mean())
    edges, probability, smoothed_probability, modes = (
        _empirical_distribution_and_modes(
            samples.values,
            options.density_bins,
            options.mode_smoothing_sigma,
            options.mode_prominence_fraction,
        )
    )
    density_difference = modes.rho_dense - modes.rho_dilute
    if density_difference <= 0.0:
        raise ValueError(f"Dense and dilute modes coincide for {case_id}")
    x_dense = (rho_total - modes.rho_dilute) / density_difference
    x_dilute = 1.0 - x_dense
    if not 0.0 <= x_dense <= 1.0:
        raise ValueError(
            f"Fitted modes do not bracket rho_total for {case_id}: "
            f"rho_dilute={modes.rho_dilute}, rho_total={rho_total}, "
            f"rho_dense={modes.rho_dense}"
        )

    output = (
        options.output_dir
        / f"{_safe_case_name(case_id)}_density_distribution.svg"
    )
    if output.exists() and not options.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace")
    _plot_distribution(
        case_replicates[0].label,
        edges,
        probability,
        smoothed_probability,
        rho_total,
        modes,
        x_dense,
        x_dilute,
        output,
    )
    report = (
        f"{case_id}: rho_total={rho_total:.6g}, "
        f"rho_dilute={modes.rho_dilute:.6g}, "
        f"rho_dense={modes.rho_dense:.6g}, "
        f"x_dilute={x_dilute:.6g}, x_dense={x_dense:.6g}"
    )
    return report, output


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
    parser.add_argument("--dx", type=float, default=None)
    parser.add_argument("--arc-bin-width", type=float, default=None)
    parser.add_argument("--radial-bins", type=int, default=20)
    parser.add_argument("--density-bins", type=int, default=80)
    parser.add_argument(
        "--mode-smoothing-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing in histogram-bin units used only to find modes",
    )
    parser.add_argument(
        "--mode-prominence-fraction",
        type=float,
        default=0.02,
        help="Minimum mode prominence as a fraction of the PDF maximum",
    )
    parser.add_argument("--frame-start", type=int, default=700)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Case worker processes; 0 uses the available CPUs (default: 0)",
    )
    parser.add_argument("--circ")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    if args.radial_bins < 1 or args.density_bins < 1:
        raise ValueError("--radial-bins and --density-bins must be positive")
    if not np.isfinite(args.mode_smoothing_sigma) or args.mode_smoothing_sigma <= 0:
        raise ValueError("--mode-smoothing-sigma must be finite and positive")
    if not 0.0 <= args.mode_prominence_fraction < 1.0:
        raise ValueError("--mode-prominence-fraction must lie in [0, 1)")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if not np.isfinite(args.particle_diameter) or args.particle_diameter <= 0:
        raise ValueError("--particle-diameter must be finite and positive")
    requested_dx = (
        5.0 * args.particle_diameter if args.dx is None else args.dx
    )
    requested_arc_width = (
        5.0 * args.particle_diameter
        if args.arc_bin_width is None
        else args.arc_bin_width
    )
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
        requested_dx=requested_dx,
        requested_arc_width=requested_arc_width,
        radial_bins=args.radial_bins,
        density_bins=args.density_bins,
        mode_smoothing_sigma=args.mode_smoothing_sigma,
        mode_prominence_fraction=args.mode_prominence_fraction,
        frame_start=args.frame_start,
        frame_stride=args.frame_stride,
    )
    tasks = [
        (case_id, case_replicates, options)
        for case_id, case_replicates in sorted(grouped.items())
    ]
    available_cpus = os.cpu_count() or 1
    worker_count = (
        min(len(tasks), available_cpus)
        if args.workers == 0
        else min(len(tasks), args.workers)
    )
    print(
        f"analyzing {len(tasks)} cases with {worker_count} worker processes",
        flush=True,
    )
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
