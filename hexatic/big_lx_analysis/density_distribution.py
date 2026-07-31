"""Compute surface-excess density and particle-number closure for big-Lx cases.

This command first calls :mod:`rho_gamma_finding` to ensure that every case
has a cached ``rho_gamma`` and film cutoff.  It then implements PLAN.md steps
2 and 3 on equal-area ``(x, theta)`` bins:

``rho_s,i = (<N_i> - rho_gamma * V_i) / A_i``.

For each case, the only plot compares the measured particle number with
``V * rho_gamma + A_s * rho_surface`` and reports the closure error.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open

from hexatic.constants import cylinder

from .rho_gamma_finding import (
    RhoGammaOptions,
    load_radius,
    safe_case_name,
    select_replicates,
    write_rho_gamma_summaries,
)
from .run_dynamics import Replicate, _style
from .surface_area import _load_circumference, _load_static_lx


@dataclass(frozen=True)
class SurfaceGrid:
    counts: NDArray[np.float64]
    n_frames: int
    n_particles: int
    nx: int
    ntheta: int
    dx: float
    dtheta: float
    lx: float
    radius: float


def _surface_grid(
    lx: float,
    circumference: float,
    requested_dx: float,
    requested_arc_width: float,
) -> tuple[int, int, float, float]:
    if lx <= 0.0 or circumference <= 0.0:
        raise ValueError("Cylinder dimensions must be positive")
    if requested_dx <= 0.0 or requested_arc_width <= 0.0:
        raise ValueError("Surface-bin widths must be positive")
    nx = max(1, int(np.rint(lx / requested_dx)))
    ntheta = max(1, int(np.rint(circumference / requested_arc_width)))
    return nx, ntheta, lx / nx, 2.0 * np.pi / ntheta


def _load_surface_grid(
    replicate: Replicate,
    *,
    radial_cutoff: float,
    requested_dx: float,
    requested_arc_width: float,
    frame_start: int,
    frame_stride: int,
) -> SurfaceGrid:
    lx = _load_static_lx(replicate.static_file)
    circumference = _load_circumference(replicate.manifest)
    radius = load_radius(replicate.static_file, circumference)
    if not 0.0 < radial_cutoff < radius:
        raise ValueError(
            f"rho-gamma cutoff must lie inside the cylinder for "
            f"{replicate.case_id}: cutoff={radial_cutoff}, R={radius}"
        )
    nx, ntheta, dx, dtheta = _surface_grid(
        lx, circumference, requested_dx, requested_arc_width
    )
    counts = np.zeros(nx * ntheta, dtype=np.float64)
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
                included = (
                    np.isfinite(frame[:, 0])
                    & np.isfinite(frame[:, 1])
                    & np.isfinite(frame[:, 2])
                    & (frame[:, 2] >= radial_cutoff)
                )
                x = np.mod(frame[included, 0] + 0.5 * lx, lx)
                theta = np.mod(frame[included, 1], 2.0 * np.pi)
                x_index = np.minimum((x / dx).astype(np.intp), nx - 1)
                theta_index = np.minimum(
                    (theta / dtheta).astype(np.intp), ntheta - 1
                )
                counts += np.bincount(
                    x_index * ntheta + theta_index,
                    minlength=nx * ntheta,
                )
                n_frames += 1
    if n_frames == 0 or n_particles is None:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    return SurfaceGrid(
        counts.reshape(nx, ntheta),
        n_frames,
        n_particles,
        nx,
        ntheta,
        dx,
        dtheta,
        lx,
        radius,
    )


def _combine_surface_grids(
    case_id: str, grids: list[SurfaceGrid]
) -> SurfaceGrid:
    reference = grids[0]
    for grid in grids[1:]:
        if (
            grid.nx != reference.nx
            or grid.ntheta != reference.ntheta
            or not np.isclose(grid.dx, reference.dx)
            or not np.isclose(grid.dtheta, reference.dtheta)
            or not np.isclose(grid.lx, reference.lx)
            or not np.isclose(grid.radius, reference.radius)
        ):
            raise ValueError(f"Replicates for {case_id} have different grids")
        if grid.n_particles != reference.n_particles:
            raise ValueError(
                f"Replicates for {case_id} have different particle counts"
            )
    return SurfaceGrid(
        counts=np.sum([grid.counts for grid in grids], axis=0),
        n_frames=sum(grid.n_frames for grid in grids),
        n_particles=reference.n_particles,
        nx=reference.nx,
        ntheta=reference.ntheta,
        dx=reference.dx,
        dtheta=reference.dtheta,
        lx=reference.lx,
        radius=reference.radius,
    )


def _plot_number_closure(
    case_label: str,
    actual: float,
    reconstructed: float,
    relative_error: float,
    output: Path,
) -> None:
    sns = _style()
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    colors = sns.color_palette("colorblind", n_colors=2)
    bars = axis.bar(
        ["Measured $N$", r"$V\rho_\gamma+A_s\rho_{\rm surface}$"],
        [actual, reconstructed],
        color=colors,
        width=0.62,
    )
    axis.bar_label(bars, fmt="%.6g", padding=4)
    axis.set(
        title=case_label,
        ylabel="Particle number",
    )
    axis.text(
        0.98,
        0.95,
        f"signed relative error = {100.0 * relative_error:.4g}%",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize="small",
    )
    axis.grid(axis="y", color="0.9", linewidth=0.7)
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
    parser.add_argument("--radial-bins", type=int, default=100)
    parser.add_argument("--frame-start", type=int, default=700)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--rho-gamma-min-r", type=float, default=8.0)
    parser.add_argument("--rho-gamma-smoothing-window", type=int, default=5)
    parser.add_argument(
        "--rho-gamma-prominence-fraction", type=float, default=0.10
    )
    parser.add_argument("--circ")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
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
    if requested_dx <= 0.0 or requested_arc_width <= 0.0:
        raise ValueError("Surface-bin widths must be positive")

    replicates = select_replicates(
        args.manifest, args.input_dir, args.case, args.circ
    )
    rho_options = RhoGammaOptions(
        radial_bins=args.radial_bins,
        particle_diameter=args.particle_diameter,
        frame_start=args.frame_start,
        frame_stride=args.frame_stride,
        minimum_r=args.rho_gamma_min_r,
        smoothing_window=args.rho_gamma_smoothing_window,
        minimum_prominence_fraction=args.rho_gamma_prominence_fraction,
    )
    rho_paths = write_rho_gamma_summaries(
        replicates,
        args.output_dir,
        rho_options,
        overwrite=args.overwrite,
    )

    grouped: dict[str, list[Replicate]] = defaultdict(list)
    for replicate in replicates:
        grouped[replicate.case_id].append(replicate)
    for case_id, case_replicates in sorted(grouped.items()):
        rho_payload = json.loads(rho_paths[case_id].read_text())
        rho_gamma = float(rho_payload["rho_gamma"])
        radial_cutoff = float(rho_payload["rho_gamma_cutoff_r"])
        grids = [
            _load_surface_grid(
                replicate,
                radial_cutoff=radial_cutoff,
                requested_dx=requested_dx,
                requested_arc_width=requested_arc_width,
                frame_start=args.frame_start,
                frame_stride=args.frame_stride,
            )
            for replicate in case_replicates
        ]
        grid = _combine_surface_grids(case_id, grids)

        mean_counts = grid.counts / grid.n_frames
        bin_area = grid.radius * grid.dx * grid.dtheta
        bin_volume = (
            0.5
            * grid.dx
            * grid.dtheta
            * (grid.radius**2 - radial_cutoff**2)
        )
        rho_surface_field = (
            mean_counts - rho_gamma * bin_volume
        ) / bin_area
        rho_surface = float(rho_surface_field.mean())

        surface_area = 2.0 * np.pi * grid.radius * grid.lx
        total_volume = np.pi * grid.radius**2 * grid.lx
        actual_n = float(grid.n_particles)
        reconstructed_n = (
            total_volume * rho_gamma + surface_area * rho_surface
        )
        absolute_error = reconstructed_n - actual_n
        relative_error = absolute_error / actual_n

        stem = safe_case_name(case_id)
        plot_output = args.output_dir / f"{stem}_number_closure.svg"
        data_output = args.output_dir / f"{stem}_surface_excess_summary.json"
        existing = [
            path for path in (plot_output, data_output) if path.exists()
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{existing[0]} exists; pass --overwrite to replace"
            )
        _plot_number_closure(
            case_replicates[0].label,
            actual_n,
            reconstructed_n,
            relative_error,
            plot_output,
        )
        payload = {
            "case_id": case_id,
            "rho_gamma": rho_gamma,
            "rho_gamma_units": "N/L^3",
            "radial_cutoff": radial_cutoff,
            "rho_surface": rho_surface,
            "rho_surface_units": "N/L^2",
            "surface_bin_area": bin_area,
            "surface_bin_volume": bin_volume,
            "surface_bin_volume_formula": (
                "0.5 * dx * dtheta * (R^2 - radial_cutoff^2)"
            ),
            "nx": grid.nx,
            "ntheta": grid.ntheta,
            "n_selected_frames": grid.n_frames,
            "n_replicates": len(grids),
            "total_volume": total_volume,
            "surface_area": surface_area,
            "actual_n": actual_n,
            "reconstructed_n": reconstructed_n,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
        }
        data_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"{case_id}: N={actual_n:.6g}, "
            f"V*rho_gamma+As*rho_surface={reconstructed_n:.6g}, "
            f"absolute_error={absolute_error:.6g}, "
            f"relative_error={relative_error:.6g}",
            flush=True,
        )
        print(f"wrote {plot_output}", flush=True)
        print(f"wrote {data_output}", flush=True)


if __name__ == "__main__":
    main()
