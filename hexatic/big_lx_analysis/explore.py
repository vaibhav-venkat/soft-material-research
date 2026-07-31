"""Explore the mean radial number-density profile of big-Lx cases.

Example
-------
pixi run python -m hexatic.big_lx_analysis.explore \
    --input-dir /mnt/drive3/vaibhav_data/big_lx_production_run3 \
    --input-dir /mnt/drive3/vaibhav_data/big_lx_production_run4 \
    --input-dir /mnt/drive3/vaibhav_data/big_lx_production_run5 \
    --circ "60.5D" \
    --overwrite

Particles are accumulated over all selected frames and replicates before the
counts are divided by both the number of selected frames and the exact
cylindrical-annulus volume.  The histogram has a hard upper cutoff at
``R - D``; particles beyond that radius do not contribute.  The radial
derivative uses three-point finite differences: centered differences in the
interior and second-order one-sided differences at the two endpoints.
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

from .density_distribution import _safe_case_name
from .run_dynamics import Replicate, _collect_manifests, _load_replicate, _style
from .surface_area import _load_circumference, _load_static_lx


@dataclass(frozen=True)
class RadialProfile:
    """Accumulated radial counts and geometry for one replicate."""

    bin_edges: NDArray[np.float64]
    counts: NDArray[np.float64]
    n_frames: int
    lx: float
    radius: float


@dataclass(frozen=True)
class Plateau:
    """Detected low-slope radial interval."""

    start: float
    end: float


def _load_radius(static_file: Path, circumference: float) -> float:
    """Load and validate the physical cylinder radius."""
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


def _load_radial_profile(
    replicate: Replicate,
    *,
    radial_bins: int,
    particle_diameter: float,
    frame_start: int,
    frame_stride: int,
) -> RadialProfile:
    """Accumulate radial counts from all selected particles and frames."""
    lx = _load_static_lx(replicate.static_file)
    circumference = _load_circumference(replicate.manifest)
    radius = _load_radius(replicate.static_file, circumference)
    radial_cutoff = radius - particle_diameter
    if radial_cutoff <= 0.0:
        raise ValueError(
            f"R - D must be positive for {replicate.case_id}: "
            f"R={radius}, D={particle_diameter}"
        )
    edges = np.linspace(0.0, radial_cutoff, radial_bins + 1, dtype=np.float64)
    counts = np.zeros(radial_bins, dtype=np.float64)
    n_frames = 0
    global_frame = 0

    for path in replicate.shard_files:
        with safe_open(path, framework="np") as tensors:
            coords = tensors.get_tensor("coords")
            for local_frame in range(coords.shape[0]):
                selected = (
                    global_frame >= frame_start
                    and (global_frame - frame_start) % frame_stride == 0
                )
                global_frame += 1
                if not selected:
                    continue
                radial_positions = np.asarray(coords[local_frame, :, 2])
                included = radial_positions[
                    np.isfinite(radial_positions)
                    & (radial_positions >= 0.0)
                    & (radial_positions <= radial_cutoff)
                ]
                counts += np.histogram(included, bins=edges)[0]
                n_frames += 1

    if n_frames == 0:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    return RadialProfile(edges, counts, n_frames, lx, radius)


def _three_point_derivative(
    values: NDArray[np.float64], spacing: float
) -> NDArray[np.float64]:
    """Return the second-order three-point derivative on a uniform grid."""
    if values.size < 3:
        raise ValueError("At least three radial bins are required")
    derivative = np.empty_like(values)
    derivative[1:-1] = (values[2:] - values[:-2]) / (2.0 * spacing)
    derivative[0] = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (
        2.0 * spacing
    )
    derivative[-1] = (
        3.0 * values[-1] - 4.0 * values[-2] + values[-3]
    ) / (2.0 * spacing)
    return derivative


def _smooth(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Smooth a one-dimensional profile without changing its length."""
    if window == 1:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, padding, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _find_plateaus(
    density: NDArray[np.float64],
    edges: NDArray[np.float64],
    *,
    minimum_r: float,
    slope_quantile: float,
    minimum_bins: int,
    smoothing_window: int,
) -> tuple[list[Plateau], float]:
    """Find contiguous low-slope regions beyond a requested radius.

    The slope cutoff is the requested quantile of ``abs(d rho / dr)`` among
    bins whose centers lie beyond ``minimum_r``.  This is an exploratory,
    data-relative plateau criterion rather than a fitted phase boundary.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth(density, smoothing_window)
    slope = np.abs(_three_point_derivative(smoothed, float(edges[1] - edges[0])))
    eligible = centers > minimum_r
    if np.count_nonzero(eligible) < minimum_bins:
        return [], np.nan
    cutoff = float(np.quantile(slope[eligible], slope_quantile))
    low_slope = eligible & (slope <= cutoff)

    plateaus: list[Plateau] = []
    start: int | None = None
    for index, selected in enumerate(low_slope):
        if selected and start is None:
            start = index
        if start is not None and (not selected or index == low_slope.size - 1):
            stop = index + 1 if selected else index
            if stop - start >= minimum_bins:
                plateaus.append(Plateau(float(edges[start]), float(edges[stop])))
            start = None
    return plateaus, cutoff


def _circumference_prefix(value: str) -> str:
    token = value.strip().upper()
    if token.endswith("D"):
        token = token[:-1]
    try:
        circumference = float(token)
    except ValueError as error:
        raise ValueError('--circ must look like "60.5D"') from error
    return f"circ_{format(circumference, '.15g').replace('.', '_')}D_"


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
        "--radial-bins",
        type=int,
        default=100,
        help="Number of equal-width radial annuli (default: 100)",
    )
    parser.add_argument("--frame-start", type=int, default=700)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--plateau-min-r",
        type=float,
        default=8.0,
        help="Only detect plateaus beyond this radius (default: 8)",
    )
    parser.add_argument(
        "--plateau-slope-quantile",
        type=float,
        default=0.35,
        help=(
            "Low-slope cutoff quantile beyond --plateau-min-r "
            "(default: 0.35)"
        ),
    )
    parser.add_argument(
        "--plateau-min-bins",
        type=int,
        default=3,
        help="Minimum contiguous bins in a reported plateau (default: 3)",
    )
    parser.add_argument(
        "--plateau-smoothing-window",
        type=int,
        default=5,
        help="Odd moving-average window used only for detection (default: 5)",
    )
    parser.add_argument(
        "--circ", help='Only include cases with a circumference such as "60.5D"'
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.radial_bins < 3:
        raise ValueError("--radial-bins must be at least 3")
    if args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    if not np.isfinite(args.particle_diameter) or args.particle_diameter <= 0.0:
        raise ValueError("--particle-diameter must be finite and positive")
    if not np.isfinite(args.plateau_min_r) or args.plateau_min_r < 0.0:
        raise ValueError("--plateau-min-r must be finite and non-negative")
    if not 0.0 < args.plateau_slope_quantile < 1.0:
        raise ValueError("--plateau-slope-quantile must lie between 0 and 1")
    if args.plateau_min_bins < 1:
        raise ValueError("--plateau-min-bins must be positive")
    if (
        args.plateau_smoothing_window < 1
        or args.plateau_smoothing_window % 2 == 0
    ):
        raise ValueError(
            "--plateau-smoothing-window must be a positive odd integer"
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
        prefix = _circumference_prefix(args.circ)
        replicates = [r for r in replicates if r.case_id.startswith(prefix)]
    if not replicates:
        raise ValueError("No matching replicates found")

    grouped: dict[str, list[Replicate]] = defaultdict(list)
    for replicate in replicates:
        grouped[replicate.case_id].append(replicate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns = _style()
    for case_id, case_replicates in sorted(grouped.items()):
        profiles = [
            _load_radial_profile(
                replicate,
                radial_bins=args.radial_bins,
                particle_diameter=args.particle_diameter,
                frame_start=args.frame_start,
                frame_stride=args.frame_stride,
            )
            for replicate in case_replicates
        ]
        reference = profiles[0]
        for profile in profiles[1:]:
            if (
                not np.allclose(profile.bin_edges, reference.bin_edges)
                or not np.isclose(profile.lx, reference.lx)
                or not np.isclose(profile.radius, reference.radius)
            ):
                raise ValueError(f"Replicates for {case_id} have different geometry")

        counts = np.sum([profile.counts for profile in profiles], axis=0)
        total_frames = sum(profile.n_frames for profile in profiles)
        edges = reference.bin_edges
        radial_centers = 0.5 * (edges[:-1] + edges[1:])
        annulus_volumes = np.pi * reference.lx * (edges[1:] ** 2 - edges[:-1] ** 2)
        density = counts / (total_frames * annulus_volumes)
        derivative = _three_point_derivative(density, float(edges[1] - edges[0]))
        plateaus, plateau_slope_cutoff = _find_plateaus(
            density,
            edges,
            minimum_r=args.plateau_min_r,
            slope_quantile=args.plateau_slope_quantile,
            minimum_bins=args.plateau_min_bins,
            smoothing_window=args.plateau_smoothing_window,
        )
        if len(plateaus) != 1:
            raise ValueError(
                f"Expected exactly one plateau beyond r={args.plateau_min_r:g} "
                f"for {case_id}, detected {len(plateaus)}; adjust the plateau "
                "detection options"
            )
        plateau = plateaus[0]
        plateau_mask = (
            (radial_centers >= plateau.start)
            & (radial_centers <= plateau.end)
        )
        rho_gamma = float(density[plateau_mask].mean())

        output = args.output_dir / f"{_safe_case_name(case_id)}_radial_density.svg"
        data_output = (
            args.output_dir
            / f"{_safe_case_name(case_id)}_radial_density_summary.json"
        )
        existing = [path for path in (output, data_output) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{existing[0]} exists; pass --overwrite to replace"
            )

        fig, density_axis = plt.subplots(
            figsize=(7.4, 4.8), constrained_layout=True
        )
        derivative_axis = density_axis.twinx()
        density_line = density_axis.plot(
            radial_centers,
            density,
            color="tab:blue",
            linewidth=1.8,
            label=r"$\langle\rho(r)\rangle_t$",
        )[0]
        derivative_line = derivative_axis.plot(
            radial_centers,
            derivative,
            color="tab:orange",
            linewidth=1.3,
            label=r"$\langle\rho(r)\rangle_t'$ (three-point finite difference)",
        )[0]
        wall_line = density_axis.axvline(
            reference.radius - args.particle_diameter,
            color="0.25",
            linestyle="--",
            linewidth=1.1,
            label=r"$R-D$",
        )
        for index, plateau in enumerate(plateaus, start=1):
            density_axis.axvspan(
                plateau.start,
                plateau.end,
                color="tab:green",
                alpha=0.12,
                linewidth=0.0,
            )
            density_axis.axvline(
                plateau.start, color="tab:green", linestyle=":", linewidth=0.9
            )
            density_axis.axvline(
                plateau.end, color="tab:green", linestyle=":", linewidth=0.9
            )
            density_axis.text(
                0.5 * (plateau.start + plateau.end),
                0.97,
                f"plateau {index}: {plateau.start:.3f}–{plateau.end:.3f}"
                f"\n$\\rho_\\gamma={rho_gamma:.6g}$",
                transform=density_axis.get_xaxis_transform(),
                ha="center",
                va="top",
                color="tab:green",
                fontsize="small",
            )
        density_axis.set(
            title=case_replicates[0].label,
            xlabel=r"Radial distance $r$",
            ylabel=r"Mean number density $\langle\rho(r)\rangle_t$",
            xlim=(0.0, reference.radius),
        )
        derivative_axis.set_ylabel(
            r"Radial derivative $\langle\rho(r)\rangle_t'$"
        )
        density_axis.grid(color="0.9", linewidth=0.7)
        density_axis.legend(
            handles=[density_line, derivative_line, wall_line],
            frameon=False,
            loc="upper left",
        )
        sns.despine(ax=density_axis, right=False)
        fig.savefig(
            output,
            format="svg",
            bbox_inches="tight",
            metadata={"Creator": "hexatic.big_lx_analysis.explore"},
        )
        plt.close(fig)
        summary = {
            "case_id": case_id,
            "rho_gamma": rho_gamma,
            "rho_definition": "time-averaged radial number density",
            "plateau_start": plateau.start,
            "plateau_end": plateau.end,
            "plateau_min_r": float(args.plateau_min_r),
            "absolute_slope_cutoff": plateau_slope_cutoff,
            "plateau_slope_quantile": float(args.plateau_slope_quantile),
            "plateau_min_bins": int(args.plateau_min_bins),
            "plateau_smoothing_window": int(args.plateau_smoothing_window),
            "radial_bins": int(args.radial_bins),
            "frame_start": int(args.frame_start),
            "frame_stride": int(args.frame_stride),
            "n_selected_frames": total_frames,
            "n_replicates": len(profiles),
        }
        data_output.write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"{case_id}: plateau=[{plateau.start:.6g}, "
            f"{plateau.end:.6g}], rho_gamma={rho_gamma:.6g}, "
            f"absolute_slope_cutoff={plateau_slope_cutoff:.6g}",
            flush=True,
        )
        print(f"wrote {output}", flush=True)
        print(f"wrote {data_output}", flush=True)


if __name__ == "__main__":
    main()
