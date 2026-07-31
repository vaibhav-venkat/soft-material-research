"""Plot Gaussian-neighborhood density distributions for big-Lx cylinders.

Particle-center density is evaluated at a Cartesian grid of neighborhood
centers inside ``r <= R - D/2 - wall_offset``.  Each center uses the same
fixed-radius spherical Gaussian kernel.  Normalized convolution with the
accessible-domain mask corrects neighborhoods truncated by the cylindrical
wall; at the outer limit this reduces to the requested hemisphere correction.

The two modes of the empirical ``P(rho)`` define ``rho_dilute`` and
``rho_dense``.  Fractions come only from the conservation lever rule,

``x_dense = (rho_total - rho_dilute) / (rho_dense - rho_dilute)``.

Cases run in parallel CPU processes.  The sole output is one SVG per case.
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
from scipy.signal import fftconvolve, find_peaks

from hexatic.constants import cylinder

from .run_dynamics import Replicate, _collect_manifests, _load_replicate, _style
from .surface_area import _load_circumference, _load_static_lx


@dataclass(frozen=True)
class NeighborhoodSamples:
    values: NDArray[np.float64]
    particle_count_sum: float
    n_frames: int
    lx: float
    accessible_radius: float
    dx: float
    dy: float
    dz: float


@dataclass(frozen=True)
class DensityModes:
    rho_dilute: float
    rho_dense: float


@dataclass(frozen=True)
class CaseOptions:
    output_dir: Path
    overwrite: bool
    particle_diameter: float
    grid_spacing: float
    neighborhood_radius: float
    gaussian_bandwidth: float
    wall_offset: float
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


def _spherical_gaussian_kernel(
    dx: float,
    dy: float,
    dz: float,
    radius: float,
    bandwidth: float,
) -> NDArray[np.float64]:
    nx = int(np.ceil(radius / dx))
    ny = int(np.ceil(radius / dy))
    nz = int(np.ceil(radius / dz))
    x = np.arange(-nx, nx + 1, dtype=np.float64) * dx
    y = np.arange(-ny, ny + 1, dtype=np.float64) * dy
    z = np.arange(-nz, nz + 1, dtype=np.float64) * dz
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    distance_sq = xx**2 + yy**2 + zz**2
    kernel = np.exp(-0.5 * distance_sq / bandwidth**2)
    kernel[distance_sq > radius**2] = 0.0
    total = float(kernel.sum())
    if total <= 0.0:
        raise ValueError("The neighborhood kernel contains no grid points")
    return kernel / total


def _normalized_cylindrical_convolution(
    density: NDArray[np.float64],
    mask: NDArray[np.bool_],
    kernel: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convolve periodically in x and normalize truncated wall kernels."""
    pad_x = kernel.shape[0] // 2
    pad_y = kernel.shape[1] // 2
    pad_z = kernel.shape[2] // 2

    def padded(field: NDArray[np.float64]) -> NDArray[np.float64]:
        periodic = np.pad(field, ((pad_x, pad_x), (0, 0), (0, 0)), mode="wrap")
        return np.pad(
            periodic,
            ((0, 0), (pad_y, pad_y), (pad_z, pad_z)),
            mode="constant",
        )

    numerator = fftconvolve(padded(density), kernel, mode="same")
    denominator = fftconvolve(
        padded(mask.astype(np.float64)), kernel, mode="same"
    )
    crop = (
        slice(pad_x, -pad_x if pad_x else None),
        slice(pad_y, -pad_y if pad_y else None),
        slice(pad_z, -pad_z if pad_z else None),
    )
    numerator = numerator[crop]
    denominator = denominator[crop]
    result = np.full_like(numerator, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 1.0e-12)
    return result


def _load_neighborhood_samples(
    replicate: Replicate, options: CaseOptions
) -> NeighborhoodSamples:
    lx = _load_static_lx(replicate.static_file)
    circumference = _load_circumference(replicate.manifest)
    radius = _load_radius(replicate.static_file, circumference)
    accessible_radius = (
        radius - 0.5 * options.particle_diameter - options.wall_offset
    )
    if accessible_radius <= 0.0:
        raise ValueError(f"No accessible particle-center radius for {replicate.case_id}")

    nx = max(1, int(np.rint(lx / options.grid_spacing)))
    ny = max(3, int(np.ceil(2.0 * accessible_radius / options.grid_spacing)))
    nz = ny
    x_edges = np.linspace(-0.5 * lx, 0.5 * lx, nx + 1)
    y_edges = np.linspace(-accessible_radius, accessible_radius, ny + 1)
    z_edges = np.linspace(-accessible_radius, accessible_radius, nz + 1)
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    dz = float(z_edges[1] - z_edges[0])
    cell_volume = dx * dy * dz
    accumulated_counts = np.zeros((nx, ny, nz), dtype=np.float64)
    particle_count_sum = 0.0
    n_frames = 0
    global_frame = 0

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
                included = (
                    np.isfinite(frame[:, 0])
                    & np.isfinite(frame[:, 1])
                    & np.isfinite(frame[:, 2])
                    & (frame[:, 2] >= 0.0)
                    & (frame[:, 2] <= accessible_radius)
                )
                x = np.mod(frame[included, 0] + 0.5 * lx, lx) - 0.5 * lx
                theta = frame[included, 1]
                radial = frame[included, 2]
                y = radial * np.cos(theta)
                z = radial * np.sin(theta)
                points = np.column_stack((x, y, z))
                accumulated_counts += np.histogramdd(
                    points, bins=(x_edges, y_edges, z_edges)
                )[0]
                particle_count_sum += float(np.count_nonzero(included))
                n_frames += 1
    if n_frames == 0:
        raise ValueError(f"No frames selected for {replicate.case_id}")

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    del x_centers  # x is periodic and every axial center is accessible.
    yy, zz = np.meshgrid(y_centers, z_centers, indexing="ij")
    radial_mask = yy**2 + zz**2 <= accessible_radius**2
    mask = np.broadcast_to(radial_mask, (nx, ny, nz))
    raw_density = accumulated_counts / (n_frames * cell_volume)
    kernel = _spherical_gaussian_kernel(
        dx,
        dy,
        dz,
        options.neighborhood_radius,
        options.gaussian_bandwidth,
    )
    smoothed_density = _normalized_cylindrical_convolution(
        raw_density, mask, kernel
    )
    values = smoothed_density[mask]
    return NeighborhoodSamples(
        values=values[np.isfinite(values)],
        particle_count_sum=particle_count_sum,
        n_frames=n_frames,
        lx=lx,
        accessible_radius=accessible_radius,
        dx=dx,
        dy=dy,
        dz=dz,
    )


def _combine_samples(
    case_id: str, samples: list[NeighborhoodSamples]
) -> NeighborhoodSamples:
    reference = samples[0]
    for sample in samples[1:]:
        if (
            not np.isclose(sample.lx, reference.lx)
            or not np.isclose(sample.accessible_radius, reference.accessible_radius)
            or not np.isclose(sample.dx, reference.dx)
            or not np.isclose(sample.dy, reference.dy)
            or not np.isclose(sample.dz, reference.dz)
        ):
            raise ValueError(f"Replicates for {case_id} have different grids")
    return NeighborhoodSamples(
        values=np.concatenate([sample.values for sample in samples]),
        particle_count_sum=sum(sample.particle_count_sum for sample in samples),
        n_frames=sum(sample.n_frames for sample in samples),
        lx=reference.lx,
        accessible_radius=reference.accessible_radius,
        dx=reference.dx,
        dy=reference.dy,
        dz=reference.dz,
    )


def _empirical_distribution_and_modes(
    values: NDArray[np.float64],
    density_bins: int,
    smoothing_sigma: float,
    prominence_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], DensityModes]:
    probability, edges = np.histogram(values, bins=density_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = gaussian_filter1d(probability, smoothing_sigma, mode="nearest")
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
            "--density-bins, --mode-smoothing-sigma, or "
            "--mode-prominence-fraction"
        )
    selected = peaks[np.argsort(prominences)[-2:]]
    selected.sort()
    return (
        edges,
        probability,
        smoothed,
        DensityModes(float(centers[selected[0]]), float(centers[selected[1]])),
    )


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
            rf"$\rho_\beta={modes.rho_dilute:.4g}$, "
            rf"$x_\beta={x_dilute:.3f}$"
        ),
    )
    axis.axvline(
        modes.rho_dense,
        color=colors[1],
        linestyle="--",
        linewidth=1.3,
        label=(
            rf"$\rho_\alpha={modes.rho_dense:.4g}$, "
            rf"$x_\alpha={x_dense:.3f}$"
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
        xlabel=r"Gaussian-neighborhood density $\rho$ [$N/L^3$]",
        ylabel=r"Probability density $P(\rho)$ (log scale)",
        xlim=(edges[0], edges[-1]),
    )
    positive = np.concatenate(
        (probability[probability > 0], smoothed_probability[smoothed_probability > 0])
    )
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


def _analyze_case(task: tuple[str, list[Replicate], CaseOptions]) -> tuple[str, Path]:
    case_id, case_replicates, options = task
    samples = _combine_samples(
        case_id,
        [_load_neighborhood_samples(replicate, options) for replicate in case_replicates],
    )
    cylinder_volume = np.pi * samples.accessible_radius**2 * samples.lx
    rho_total = samples.particle_count_sum / (samples.n_frames * cylinder_volume)
    edges, probability, smoothed, modes = _empirical_distribution_and_modes(
        samples.values,
        options.density_bins,
        options.mode_smoothing_sigma,
        options.mode_prominence_fraction,
    )
    difference = modes.rho_dense - modes.rho_dilute
    x_dense = (rho_total - modes.rho_dilute) / difference
    x_dilute = 1.0 - x_dense
    if not 0.0 <= x_dense <= 1.0:
        raise ValueError(
            f"Detected modes do not bracket rho_total for {case_id}: "
            f"rho_dilute={modes.rho_dilute}, rho_total={rho_total}, "
            f"rho_dense={modes.rho_dense}"
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
        x_dense,
        x_dilute,
        output,
    )
    report = (
        f"{case_id}: rho_total={rho_total:.6g}, "
        f"rho_beta={modes.rho_dilute:.6g}, rho_alpha={modes.rho_dense:.6g}, "
        f"x_beta={x_dilute:.6g}, x_alpha={x_dense:.6g}"
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
    parser.add_argument("--grid-spacing", type=float, default=None)
    parser.add_argument("--neighborhood-radius", type=float, default=None)
    parser.add_argument("--gaussian-bandwidth", type=float, default=None)
    parser.add_argument(
        "--wall-offset",
        type=float,
        default=1.0e-3,
        help="Small inward offset from R-D/2 (default: 1e-3)",
    )
    parser.add_argument("--density-bins", type=int, default=80)
    parser.add_argument("--mode-smoothing-sigma", type=float, default=1.5)
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
    if args.density_bins < 3 or args.workers < 0:
        raise ValueError("Density bins must be at least 3 and workers non-negative")
    if not 0.0 <= args.mode_prominence_fraction < 1.0:
        raise ValueError("--mode-prominence-fraction must lie in [0, 1)")
    if args.mode_smoothing_sigma <= 0.0 or args.wall_offset < 0.0:
        raise ValueError("Smoothing must be positive and wall offset non-negative")
    diameter = args.particle_diameter
    grid_spacing = diameter if args.grid_spacing is None else args.grid_spacing
    neighborhood_radius = (
        2.5 * diameter
        if args.neighborhood_radius is None
        else args.neighborhood_radius
    )
    gaussian_bandwidth = (
        0.5 * neighborhood_radius
        if args.gaussian_bandwidth is None
        else args.gaussian_bandwidth
    )
    if min(diameter, grid_spacing, neighborhood_radius, gaussian_bandwidth) <= 0:
        raise ValueError("Particle and neighborhood length scales must be positive")

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
        grid_spacing=grid_spacing,
        neighborhood_radius=neighborhood_radius,
        gaussian_bandwidth=gaussian_bandwidth,
        wall_offset=args.wall_offset,
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
