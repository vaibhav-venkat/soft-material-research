"""Plot volumetric-density distributions for big-Lx cylinder cases.

Selected frames are divided into equal-volume ``(x, theta, r)`` voxels.  The
local volume density is the particle count divided by the exact cylindrical
sector volume.  A two-component Gaussian mixture supplies dilute and dense
density modes.  Their volume fractions are not taken from the mixture weights;
they are calculated from the conservation lever rule

``rho_total = x_dense * rho_dense + (1 - x_dense) * rho_dilute``.

The sole generated artifact is one SVG probability-density plot per case.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open
from sklearn.mixture import GaussianMixture

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
    sigma_dilute: float
    rho_dense: float
    sigma_dense: float


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
    frame_densities: list[NDArray[np.float64]] = []
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
                frame_densities.append(
                    counts.astype(np.float64) / voxel_volume
                )
    if not frame_densities or n_particles is None:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    return VolumeDensitySamples(
        values=np.concatenate(frame_densities),
        n_frames=len(frame_densities),
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


def _fit_density_modes(
    values: NDArray[np.float64], max_fit_samples: int
) -> DensityModes:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        raise ValueError("At least four finite density samples are required")
    rng = np.random.default_rng(0)
    fit_values = (
        rng.choice(finite, size=max_fit_samples, replace=False)
        if finite.size > max_fit_samples
        else finite
    )
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=0,
        reg_covar=1.0e-8,
    )
    model.fit(fit_values.reshape(-1, 1))
    order = np.argsort(model.means_[:, 0])
    dilute, dense = int(order[0]), int(order[1])
    return DensityModes(
        rho_dilute=float(model.means_[dilute, 0]),
        sigma_dilute=float(np.sqrt(model.covariances_[dilute, 0, 0])),
        rho_dense=float(model.means_[dense, 0]),
        sigma_dense=float(np.sqrt(model.covariances_[dense, 0, 0])),
    )


def _normal_pdf(
    x: NDArray[np.float64], mean: float, standard_deviation: float
) -> NDArray[np.float64]:
    z = (x - mean) / standard_deviation
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * standard_deviation)


def _plot_distribution(
    case_label: str,
    values: NDArray[np.float64],
    rho_total: float,
    modes: DensityModes,
    x_dense: float,
    x_dilute: float,
    density_bins: int,
    output: Path,
) -> None:
    sns = _style()
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    sns.histplot(
        values,
        bins=density_bins,
        stat="density",
        element="step",
        fill=True,
        alpha=0.3,
        color="0.45",
        ax=axis,
    )
    lower, upper = np.quantile(values, [0.001, 0.999])
    padding = 0.08 * max(upper - lower, np.finfo(np.float64).eps)
    x = np.linspace(lower - padding, upper + padding, 600)
    dilute_pdf = x_dilute * _normal_pdf(
        x, modes.rho_dilute, modes.sigma_dilute
    )
    dense_pdf = x_dense * _normal_pdf(
        x, modes.rho_dense, modes.sigma_dense
    )
    colors = sns.color_palette("colorblind", n_colors=2)
    axis.plot(
        x,
        dilute_pdf,
        color=colors[0],
        linestyle="--",
        linewidth=1.4,
        label=(
            rf"$\rho_{{\rm dilute}}={modes.rho_dilute:.4g}$, "
            rf"$x_{{\rm dilute}}={x_dilute:.3f}$"
        ),
    )
    axis.plot(
        x,
        dense_pdf,
        color=colors[1],
        linestyle="--",
        linewidth=1.4,
        label=(
            rf"$\rho_{{\rm dense}}={modes.rho_dense:.4g}$, "
            rf"$x_{{\rm dense}}={x_dense:.3f}$"
        ),
    )
    axis.plot(
        x,
        dilute_pdf + dense_pdf,
        color="black",
        linewidth=1.8,
        label="two-population reconstruction",
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
        ylabel=r"Probability density $P(\rho)$",
        xlim=(x[0], x[-1]),
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
    parser.add_argument("--mixture-max-fit-samples", type=int, default=100_000)
    parser.add_argument("--frame-start", type=int, default=700)
    parser.add_argument("--frame-stride", type=int, default=1)
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
    if args.mixture_max_fit_samples < 4:
        raise ValueError("--mixture-max-fit-samples must be at least 4")
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
    for case_id, case_replicates in sorted(grouped.items()):
        samples = _combine_samples(
            case_id,
            [
                _load_volume_density_samples(
                    replicate,
                    requested_dx=requested_dx,
                    requested_arc_width=requested_arc_width,
                    radial_bins=args.radial_bins,
                    frame_start=args.frame_start,
                    frame_stride=args.frame_stride,
                )
                for replicate in case_replicates
            ],
        )
        rho_total = float(samples.values.mean())
        modes = _fit_density_modes(
            samples.values, args.mixture_max_fit_samples
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
            args.output_dir
            / f"{_safe_case_name(case_id)}_density_distribution.svg"
        )
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace")
        _plot_distribution(
            case_replicates[0].label,
            samples.values,
            rho_total,
            modes,
            x_dense,
            x_dilute,
            args.density_bins,
            output,
        )
        print(
            f"{case_id}: rho_total={rho_total:.6g}, "
            f"rho_dilute={modes.rho_dilute:.6g}, "
            f"rho_dense={modes.rho_dense:.6g}, "
            f"x_dilute={x_dilute:.6g}, x_dense={x_dense:.6g}",
            flush=True,
        )
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
