"""Plot film-density distributions for each big-Lx cylinder case.

Example
-------
pixi run python -m hexatic.big_lx_analysis.density_distribution \
    --input-dir /mnt/drive3/vaibhav_data/big_lx_production_run3 \
    --input-dir /mnt/drive3/vaibhav_data/big_lx_production_run4 \
    --input-dir /mnt/drive3/vaibhav_data/big_lx_production_run5 \
    --circ "60.5D" \
    --overwrite

The cylindrical film is unwrapped onto ``(x, R theta)``.  Particles in the
interior are excluded using ``hexatic_shell_mask`` (or the same radial
fallback as :mod:`surface_area`).  The reported local density is the particle
area fraction in each surface bin,

``phi = N_bin * (pi D**2 / 4) / (R dx dtheta)``.
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from hexatic.constants import cylinder

from .run_dynamics import Replicate, _collect_manifests, _load_replicate, _style
from .surface_area import (
    _load_circumference,
    _load_static_lx,
    _recompute_shell_mask_from_coords,
)

@dataclass(frozen=True)
class FilmDensitySamples:
    """Local film area fractions and the grid used to obtain them."""

    values: NDArray[np.float64]
    bulk_counts: NDArray[np.float64]
    n_particles: int
    nx: int
    ntheta: int
    dx: float
    arc_width: float


@dataclass(frozen=True)
class PhaseClassification:
    """Density modes and the equal-area-bin phase classification."""

    rho_beta: float
    rho_alpha: float
    mean_beta: float
    mean_alpha: float
    threshold: float
    x_alpha: float


def _surface_grid(
    lx: float,
    circumference: float,
    requested_dx: float,
    requested_arc_width: float,
) -> tuple[int, int, float, float]:
    """Return an integer periodic grid with widths close to those requested."""
    if lx <= 0.0 or circumference <= 0.0:
        raise ValueError("Cylinder dimensions must be positive")
    if requested_dx <= 0.0 or requested_arc_width <= 0.0:
        raise ValueError("Surface-bin widths must be positive")
    nx = max(1, int(np.rint(lx / requested_dx)))
    ntheta = max(1, int(np.rint(circumference / requested_arc_width)))
    return nx, ntheta, lx / nx, circumference / ntheta


def _load_film_density_samples(
    replicate: Replicate,
    *,
    lx: float,
    circumference: float,
    cylinder_radius: float | None,
    shell_delta: float,
    particle_diameter: float,
    requested_dx: float,
    requested_arc_width: float,
    frame_start: int,
    frame_stride: int,
) -> FilmDensitySamples:
    """Compute local surface packing fractions for one replicate."""
    nx, ntheta, dx, arc_width = _surface_grid(
        lx, circumference, requested_dx, requested_arc_width
    )
    particle_area = np.pi * (particle_diameter / 2.0) ** 2
    bin_area = dx * arc_width
    samples: list[NDArray[np.float64]] = []
    bulk_counts: list[float] = []
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
            keys = tensors.keys()
            stored_mask = (
                tensors.get_tensor("hexatic_shell_mask")
                if "hexatic_shell_mask" in keys
                else None
            )
            if stored_mask is None and cylinder_radius is None:
                raise KeyError(
                    "hexatic_shell_mask not found and static radius is absent; "
                    "cannot exclude interior particles"
            )

            for local_frame in range(coords.shape[0]):
                use_frame = (
                    global_frame >= frame_start
                    and (global_frame - frame_start) % frame_stride == 0
                )
                global_frame += 1
                if not use_frame:
                    continue
                frame_coords = coords[local_frame]
                mask = (
                    np.asarray(stored_mask[local_frame], dtype=np.bool_)
                    if stored_mask is not None
                    else _recompute_shell_mask_from_coords(
                        frame_coords,
                        _require_radius(cylinder_radius),
                        shell_delta,
                    )
                )
                finite = (
                    mask
                    & np.isfinite(frame_coords[:, 0])
                    & np.isfinite(frame_coords[:, 1])
                )
                bulk_counts.append(float(np.count_nonzero(~mask)))

                # Modulo makes both periodic seams independent of the stored
                # coordinate convention (centered or [0, period)).
                x = np.mod(frame_coords[finite, 0] + 0.5 * lx, lx)
                theta = np.mod(frame_coords[finite, 1], 2.0 * np.pi)
                x_index = np.minimum((x / dx).astype(np.intp), nx - 1)
                theta_index = np.minimum(
                    (theta * ntheta / (2.0 * np.pi)).astype(np.intp),
                    ntheta - 1,
                )
                counts = np.bincount(
                    x_index * ntheta + theta_index,
                    minlength=nx * ntheta,
                )
                samples.append(counts.astype(np.float64) * particle_area / bin_area)

    if not samples:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    assert n_particles is not None
    return FilmDensitySamples(
        values=np.concatenate(samples),
        bulk_counts=np.asarray(bulk_counts, dtype=np.float64),
        n_particles=n_particles,
        nx=nx,
        ntheta=ntheta,
        dx=dx,
        arc_width=arc_width,
    )


def _require_radius(radius: float | None) -> float:
    """Narrow an optional radius after the missing-mask validation."""
    if radius is None:
        raise ValueError("Cylinder radius is required to reconstruct the shell mask")
    return radius


def _safe_case_name(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in case_id)


def _classify_phases(
    values: NDArray[np.float64],
    density_bins: int,
) -> PhaseClassification:
    """Find the two modes and classify bins at the valley between them."""
    mode_bins = max(128, 2 * density_bins)
    density, edges = np.histogram(values, bins=mode_bins, density=True)
    smoothed = gaussian_filter1d(density, sigma=3.0)
    minimum_separation = max(1, mode_bins // 4)
    interior_peaks, _ = find_peaks(
        smoothed,
        distance=minimum_separation,
        prominence=0.02 * float(smoothed.max()),
    )
    candidates = [int(index) for index in interior_peaks]
    if smoothed[0] >= smoothed[1]:
        candidates.append(0)
    if smoothed[-1] >= smoothed[-2]:
        candidates.append(mode_bins - 1)

    pairs = [
        (left, right)
        for left in candidates
        for right in candidates
        if right - left >= minimum_separation
    ]
    if pairs:
        beta_index, alpha_index = max(
            pairs,
            key=lambda pair: (
                smoothed[pair[0]]
                * smoothed[pair[1]]
                * (pair[1] - pair[0])
            ),
        )
    else:
        # Boundary modes can be monotone after smoothing and therefore fail
        # the local-maximum test. Pair the global mode with the strongest mode
        # at least one quarter of the histogram away.
        primary = int(np.argmax(smoothed))
        separated = np.flatnonzero(
            np.abs(np.arange(mode_bins) - primary) >= minimum_separation
        )
        if separated.size == 0:
            raise ValueError("Density range is too narrow to classify two phases")
        secondary = int(separated[np.argmax(smoothed[separated])])
        beta_index, alpha_index = sorted((primary, secondary))

    valley_index = int(beta_index) + int(
        np.argmin(smoothed[int(beta_index) : int(alpha_index) + 1])
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    threshold = float(centers[valley_index])
    alpha_mask = values > threshold
    if not np.any(alpha_mask) or np.all(alpha_mask):
        raise ValueError("Phase boundary does not separate alpha and beta bins")
    return PhaseClassification(
        rho_beta=float(centers[beta_index]),
        rho_alpha=float(centers[alpha_index]),
        mean_beta=float(values[~alpha_mask].mean()),
        mean_alpha=float(values[alpha_mask].mean()),
        threshold=threshold,
        x_alpha=float(np.count_nonzero(alpha_mask) / values.size),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("density_distribution_plots"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--particle-diameter",
        type=float,
        default=float(cylinder.PARTICLE_DIAMETER),
    )
    parser.add_argument(
        "--shell-delta",
        type=float,
        default=float(cylinder.SHELL_DELTA),
        help="Fallback shell thickness when a stored shell mask is unavailable",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=None,
        help="Requested axial bin width (default: 5 particle diameters)",
    )
    parser.add_argument(
        "--arc-bin-width",
        type=float,
        default=None,
        help="Requested R*dtheta bin width (default: 5 particle diameters)",
    )
    parser.add_argument(
        "--density-bins",
        type=int,
        default=80,
        help="Number of bins in the probability-density histogram",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=700,
        help="First zero-based frame index to include (default: 700)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--circ",
        help='Only include cases with a given circumference, e.g. "60.5D"',
    )
    args = parser.parse_args()

    if args.density_bins < 1:
        raise ValueError("--density-bins must be positive")
    if args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    if args.particle_diameter <= 0.0:
        raise ValueError("--particle-diameter must be positive")
    requested_dx = (
        5.0 * args.particle_diameter if args.dx is None else args.dx
    )
    requested_arc_width = (
        5.0 * args.particle_diameter
        if args.arc_bin_width is None
        else args.arc_bin_width
    )

    manifests = _collect_manifests(args.manifest, args.input_dir)
    replicates = [_load_replicate(path) for path in manifests]
    if args.case:
        selected = set(args.case)
        replicates = [r for r in replicates if r.case_id in selected]
        missing = selected - {r.case_id for r in replicates}
        if missing:
            raise ValueError(f"Requested cases not found: {sorted(missing)}")
    if args.circ:
        token = args.circ.strip().upper()
        if token.endswith("D"):
            token = token[:-1]
        try:
            circumference_value = float(token)
        except ValueError as error:
            raise ValueError('--circ must look like "60.5D"') from error
        prefix = (
            f"circ_{format(circumference_value, '.15g').replace('.', '_')}D_"
        )
        replicates = [r for r in replicates if r.case_id.startswith(prefix)]
    if not replicates:
        raise ValueError("No matching replicates found")

    grouped: dict[str, list[Replicate]] = defaultdict(list)
    for replicate in replicates:
        grouped[replicate.case_id].append(replicate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns = _style()
    for case_id, case_replicates in sorted(grouped.items()):
        all_values: list[NDArray[np.float64]] = []
        all_bulk_counts: list[NDArray[np.float64]] = []
        grid: FilmDensitySamples | None = None
        case_n_particles: int | None = None
        case_lx: float | None = None
        case_circumference: float | None = None
        for replicate in case_replicates:
            lx = _load_static_lx(replicate.static_file)
            circumference = _load_circumference(replicate.manifest)
            radius: float | None = None
            with safe_open(replicate.static_file, framework="np") as tensors:
                if "radius" in tensors.keys():
                    radius = float(tensors.get_tensor("radius").item())
            if radius is not None and not np.isclose(
                circumference, 2.0 * np.pi * radius, rtol=1e-6, atol=1e-8
            ):
                raise ValueError(
                    f"Manifest circumference and static radius disagree for {case_id}"
                )
            result = _load_film_density_samples(
                replicate,
                lx=lx,
                circumference=circumference,
                cylinder_radius=radius,
                shell_delta=args.shell_delta,
                particle_diameter=args.particle_diameter,
                requested_dx=requested_dx,
                requested_arc_width=requested_arc_width,
                frame_start=args.frame_start,
                frame_stride=args.frame_stride,
            )
            if grid is not None and (
                result.nx != grid.nx
                or result.ntheta != grid.ntheta
                or not np.isclose(result.dx, grid.dx)
                or not np.isclose(result.arc_width, grid.arc_width)
            ):
                raise ValueError(f"Replicates for {case_id} have different grids")
            if case_n_particles is not None and result.n_particles != case_n_particles:
                raise ValueError(
                    f"Replicates for {case_id} have different particle counts"
                )
            grid = result
            case_n_particles = result.n_particles
            case_lx = lx
            case_circumference = circumference
            all_values.append(result.values)
            all_bulk_counts.append(result.bulk_counts)

        assert grid is not None
        assert case_n_particles is not None
        assert case_lx is not None
        assert case_circumference is not None
        values = np.concatenate(all_values)
        phases = _classify_phases(values, args.density_bins)
        particle_area = np.pi * (args.particle_diameter / 2.0) ** 2
        rho_2d_alpha = phases.mean_alpha / particle_area
        rho_2d_beta = phases.mean_beta / particle_area
        surface_area = case_circumference * case_lx
        cylinder_radius = case_circumference / (2.0 * np.pi)
        bulk_radius = cylinder_radius - args.shell_delta
        if bulk_radius <= 0.0:
            raise ValueError(
                f"Shell cutoff leaves no interior volume for {case_id}"
            )
        bulk_volume = np.pi * bulk_radius**2 * case_lx
        inferred_film_count = surface_area * (
            rho_2d_alpha * phases.x_alpha
            + rho_2d_beta * (1.0 - phases.x_alpha)
        )
        rho_3d_gamma_conservation = (
            case_n_particles - inferred_film_count
        ) / bulk_volume
        rho_3d_gamma_measured = (
            float(np.concatenate(all_bulk_counts).mean()) / bulk_volume
        )
        relative_gamma_error = (
            (rho_3d_gamma_conservation - rho_3d_gamma_measured)
            / rho_3d_gamma_measured
            if rho_3d_gamma_measured != 0.0
            else np.nan
        )
        diameter_cubed = args.particle_diameter**3
        output = args.output_dir / f"{_safe_case_name(case_id)}_density_distribution.svg"
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace")

        fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        sns.histplot(
            values,
            bins=args.density_bins,
            stat="density",
            element="step",
            fill=True,
            alpha=0.35,
            ax=ax,
        )
        ax.axvline(
            phases.rho_beta,
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
            label=rf"$\phi_\beta={phases.rho_beta:.3g}$ (dilute)",
        )
        ax.axvline(
            phases.rho_alpha,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=rf"$\phi_\alpha={phases.rho_alpha:.3g}$ (dense)",
        )
        ax.axvline(
            phases.threshold,
            color="0.25",
            linestyle=":",
            linewidth=1.2,
            label=rf"classification boundary $={phases.threshold:.3g}$",
        )
        replicate_suffix = (
            f", {len(case_replicates)} replicates"
            if len(case_replicates) > 1
            else ""
        )
        ax.set(
            title=f"{case_replicates[0].label}{replicate_suffix}",
            xlabel=r"Local film area fraction $\phi$",
            ylabel=r"Probability density $p(\phi)$",
        )
        ax.text(
            0.98,
            0.96,
            rf"$\Delta x={grid.dx / args.particle_diameter:.3g}D$, "
            rf"$R\Delta\theta={grid.arc_width / args.particle_diameter:.3g}D$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize="small",
        )
        ax.text(
            0.98,
            0.87,
            rf"$x_\alpha={phases.x_alpha:.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize="small",
        )
        ax.text(
            0.98,
            0.79,
            rf"$\rho^{{3D}}_{{\gamma,\mathrm{{cons}}}}D^3="
            rf"{rho_3d_gamma_conservation * diameter_cubed:.3g}$, "
            rf"$\rho^{{3D}}_{{\gamma,\mathrm{{count}}}}D^3="
            rf"{rho_3d_gamma_measured * diameter_cubed:.3g}$"
            "\n"
            rf"relative error $={100.0 * relative_gamma_error:.2f}\%$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize="small",
        )
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.legend(frameon=False, loc="upper left")
        fig.savefig(
            output,
            format="svg",
            bbox_inches="tight",
            metadata={"Creator": "hexatic.big_lx_analysis.density_distribution"},
        )
        plt.close(fig)
        print(
            f"{case_id}: phi_alpha={phases.rho_alpha:.6g}, "
            f"phi_beta={phases.rho_beta:.6g}, "
            f"mean_phi_alpha={phases.mean_alpha:.6g}, "
            f"mean_phi_beta={phases.mean_beta:.6g}, "
            f"rho_2d_alpha_D2={rho_2d_alpha * args.particle_diameter**2:.6g}, "
            f"rho_2d_beta_D2={rho_2d_beta * args.particle_diameter**2:.6g}, "
            f"classification_threshold={phases.threshold:.6g}, "
            f"x_alpha={phases.x_alpha:.6g}, "
            f"rho_3d_gamma_conservation_D3="
            f"{rho_3d_gamma_conservation * diameter_cubed:.6g}, "
            f"rho_3d_gamma_measured_D3="
            f"{rho_3d_gamma_measured * diameter_cubed:.6g}, "
            f"rho_3d_gamma_relative_error={relative_gamma_error:.6g}",
            flush=True,
        )
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
