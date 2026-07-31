"""Find and cache the bulk density for big-Lx cylinder cases.

The time-averaged radial number density uses exact cylindrical-shell volumes,

``V_i = pi * Lx * (r_outer**2 - r_inner**2)``.

Only radii up to ``R - D`` enter the radial histogram.  Beyond a configurable
minimum radius, the most prominent persistent local minimum of a smoothed copy
of the profile supplies ``rho_gamma`` and the film's inner radial cutoff.  One
JSON summary is written per case; this module intentionally produces no plots.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open

from hexatic.constants import cylinder

from .run_dynamics import Replicate, _collect_manifests, _load_replicate
from .surface_area import _load_circumference, _load_static_lx


@dataclass(frozen=True)
class RhoGammaOptions:
    radial_bins: int = 100
    particle_diameter: float = float(cylinder.PARTICLE_DIAMETER)
    frame_start: int = 700
    frame_stride: int = 1
    minimum_r: float = 8.0
    smoothing_window: int = 5
    minimum_prominence_fraction: float = 0.10


@dataclass(frozen=True)
class RadialProfile:
    bin_edges: NDArray[np.float64]
    counts: NDArray[np.float64]
    n_frames: int
    lx: float
    radius: float


@dataclass(frozen=True)
class PersistentMinimum:
    index: int
    radius: float
    density: float
    prominence: float


def safe_case_name(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in case_id)


def circumference_prefix(value: str) -> str:
    token = value.strip().upper()
    if token.endswith("D"):
        token = token[:-1]
    try:
        circumference = float(token)
    except ValueError as error:
        raise ValueError('--circ must look like "60.5D"') from error
    return f"circ_{format(circumference, '.15g').replace('.', '_')}D_"


def load_radius(static_file: Path, circumference: float) -> float:
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


def select_replicates(
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
        prefix = circumference_prefix(circumference)
        replicates = [r for r in replicates if r.case_id.startswith(prefix)]
    if not replicates:
        raise ValueError("No matching replicates found")
    return replicates


def _validate_options(options: RhoGammaOptions) -> None:
    if options.radial_bins < 3:
        raise ValueError("radial_bins must be at least 3")
    if options.frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if options.frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if not np.isfinite(options.particle_diameter) or options.particle_diameter <= 0:
        raise ValueError("particle_diameter must be finite and positive")
    if not np.isfinite(options.minimum_r) or options.minimum_r < 0:
        raise ValueError("minimum_r must be finite and non-negative")
    if not 0.0 <= options.minimum_prominence_fraction < 1.0:
        raise ValueError("minimum_prominence_fraction must lie in [0, 1)")
    if options.smoothing_window < 1 or options.smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")


def _load_radial_profile(
    replicate: Replicate, options: RhoGammaOptions
) -> RadialProfile:
    lx = _load_static_lx(replicate.static_file)
    circumference = _load_circumference(replicate.manifest)
    radius = load_radius(replicate.static_file, circumference)
    radial_limit = radius - options.particle_diameter
    if radial_limit <= 0.0:
        raise ValueError(f"R - D is not positive for {replicate.case_id}")
    edges = np.linspace(0.0, radial_limit, options.radial_bins + 1)
    counts = np.zeros(options.radial_bins, dtype=np.float64)
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
                radial = np.asarray(coords[local_frame, :, 2])
                included = radial[
                    np.isfinite(radial)
                    & (radial >= 0.0)
                    & (radial <= radial_limit)
                ]
                counts += np.histogram(included, bins=edges)[0]
                n_frames += 1
    if n_frames == 0:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    return RadialProfile(edges, counts, n_frames, lx, radius)


def _smooth(values: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    if window == 1:
        return values.copy()
    padding = window // 2
    padded = np.pad(values, padding, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _find_persistent_minimum(
    density: NDArray[np.float64],
    edges: NDArray[np.float64],
    options: RhoGammaOptions,
) -> PersistentMinimum:
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth(density, options.smoothing_window)
    eligible = np.flatnonzero(centers > options.minimum_r)
    if eligible.size < 3:
        raise ValueError(
            f"Fewer than three radial bins lie beyond r={options.minimum_r:g}"
        )
    first = int(eligible[0])
    last = int(eligible[-1])
    required = options.minimum_prominence_fraction * float(
        np.ptp(smoothed[eligible])
    )
    candidates: list[PersistentMinimum] = []
    for index in range(first + 1, last):
        if not (
            smoothed[index] <= smoothed[index - 1]
            and smoothed[index] < smoothed[index + 1]
        ):
            continue
        left_peak = float(np.max(smoothed[first:index]))
        right_peak = float(np.max(smoothed[index + 1 : last + 1]))
        prominence = min(
            left_peak - float(smoothed[index]),
            right_peak - float(smoothed[index]),
        )
        if prominence >= required:
            candidates.append(
                PersistentMinimum(
                    index,
                    float(centers[index]),
                    float(smoothed[index]),
                    prominence,
                )
            )
    if not candidates:
        raise ValueError(
            f"No persistent density minimum found beyond r={options.minimum_r:g}; "
            "adjust the smoothing window or prominence fraction"
        )
    return max(candidates, key=lambda candidate: candidate.prominence)


def summary_path(output_dir: Path, case_id: str) -> Path:
    return output_dir / f"{safe_case_name(case_id)}_rho_gamma.json"


def write_rho_gamma_summaries(
    replicates: list[Replicate],
    output_dir: Path,
    options: RhoGammaOptions,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """Compute missing rho-gamma summaries and return their paths."""
    _validate_options(options)
    grouped: dict[str, list[Replicate]] = defaultdict(list)
    for replicate in replicates:
        grouped[replicate.case_id].append(replicate)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for case_id, case_replicates in sorted(grouped.items()):
        output = summary_path(output_dir, case_id)
        outputs[case_id] = output
        if output.exists() and not overwrite:
            continue
        profiles = [_load_radial_profile(r, options) for r in case_replicates]
        reference = profiles[0]
        for profile in profiles[1:]:
            if (
                not np.allclose(profile.bin_edges, reference.bin_edges)
                or not np.isclose(profile.lx, reference.lx)
                or not np.isclose(profile.radius, reference.radius)
            ):
                raise ValueError(f"Replicates for {case_id} have different geometry")
        counts = np.sum([profile.counts for profile in profiles], axis=0)
        n_frames = sum(profile.n_frames for profile in profiles)
        edges = reference.bin_edges
        volumes = np.pi * reference.lx * (
            edges[1:] ** 2 - edges[:-1] ** 2
        )
        density = counts / (n_frames * volumes)
        minimum = _find_persistent_minimum(density, edges, options)
        payload = {
            "case_id": case_id,
            "rho_gamma": minimum.density,
            "rho_units": "N/L^3",
            "rho_gamma_cutoff_r": minimum.radius,
            "rho_gamma_raw_bin_value": float(density[minimum.index]),
            "rho_gamma_minimum_prominence": minimum.prominence,
            "radial_bin_volume_formula": "pi * Lx * (r_outer^2 - r_inner^2)",
            "radial_bins": options.radial_bins,
            "particle_diameter": options.particle_diameter,
            "frame_start": options.frame_start,
            "frame_stride": options.frame_stride,
            "n_selected_frames": n_frames,
            "n_replicates": len(profiles),
            "rho_gamma_min_r": options.minimum_r,
            "rho_gamma_smoothing_window": options.smoothing_window,
            "rho_gamma_prominence_fraction": (
                options.minimum_prominence_fraction
            ),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"{case_id}: rho_gamma={minimum.density:.6g}, "
            f"cutoff_r={minimum.radius:.6g}",
            flush=True,
        )
        print(f"wrote {output}", flush=True)
    return outputs


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
    replicates = select_replicates(
        args.manifest, args.input_dir, args.case, args.circ
    )
    options = RhoGammaOptions(
        radial_bins=args.radial_bins,
        particle_diameter=args.particle_diameter,
        frame_start=args.frame_start,
        frame_stride=args.frame_stride,
        minimum_r=args.rho_gamma_min_r,
        smoothing_window=args.rho_gamma_smoothing_window,
        minimum_prominence_fraction=args.rho_gamma_prominence_fraction,
    )
    write_rho_gamma_summaries(
        replicates, args.output_dir, options, overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
