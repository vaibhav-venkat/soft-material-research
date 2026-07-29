"""Compare velocity-correlation Laplace transforms across analysis families.

Edit ``INPUT_DIRS`` below to point at the analysis directories to include.
Directories with the same family and case ID are averaged as random seeds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .all_cases_loading import CaseSeries, case_key, load_case_series
from .correlation import (
    _finite_difference_vector,
    _lag_times,
    _normalized_autocorrelation,
)
from .laplace import _laplace_transform
from .laplacian_grid import LaplaceEntry, plot_grid, validate_shared_grid
from .loading import _elapsed_time
from .plot_correlation import _nonnegative_int
from .spectral_entropy import (
    EntropyPoint,
    geometry_group,
    plot_entropy,
    spectral_entropy,
)


INPUT_DIRS: tuple[str, ...] = (
    # Add absolute paths, for example:
    # "/mnt/drive3/vaibhav_data/.../safetensors_output/ideal_abp_cylinder",
    # "/mnt/drive3/vaibhav_data/.../safetensors_output/circ_60_5D_lx_16x",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot_all_output"),
        help="Output directory (default: plot_all_output/).",
    )
    parser.add_argument(
        "--frames",
        type=_nonnegative_int,
        default=1000,
        help="Maximum initial frames per case (default: 1000).",
    )
    parser.add_argument(
        "--max-lag",
        type=_nonnegative_int,
        default=998,
        help="Maximum lag in frames (default: 998).",
    )
    parser.add_argument(
        "--laplace-r-points",
        type=int,
        default=161,
        help="Number of Laplace decay-coordinate samples (default: 161).",
    )
    parser.add_argument(
        "--laplace-omega-points",
        type=int,
        default=241,
        help="Number of Laplace frequency samples (default: 241).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class PreparedCase:
    """One case's seeds reduced to everything both variants need.

    The COM velocity and the lag grid do not depend on the variant, and the
    finite difference is a Python-level loop per axis, so computing this once and
    slicing per variant halves the expensive part of the run.
    """

    key: str
    case_id: str
    family: str
    geometry_group: str
    interaction_class: str
    lx_multiplier: int
    coordinate_kind: str
    seed_count: int
    velocities: list[NDArray[np.float64]]
    mean_radius: float
    lag_time: NDArray[np.float64]
    max_lag: int


def _validate_seed_group(key: str, seeds: list[CaseSeries]) -> None:
    reference = seeds[0]
    for seed in seeds[1:]:
        if not np.array_equal(seed.steps, reference.steps):
            raise ValueError(f"seed frame steps do not match for {key}")
        fields = ("geometry_kind", "lx_multiplier", "coordinate_kind", "interaction_class")
        for field in fields:
            if getattr(seed, field) != getattr(reference, field):
                raise ValueError(f"seed {field} does not match for {key}")


def _prepare_case(key: str, seeds: list[CaseSeries], max_lag: int) -> PreparedCase:
    _validate_seed_group(key, seeds)
    reference = seeds[0]
    elapsed = _elapsed_time(reference.steps, reference.timestep)
    effective_max_lag = min(max_lag, len(reference.steps) - 2)
    if effective_max_lag != max_lag:
        print(
            f"[plot_all] {key}: capping --max-lag {max_lag} -> "
            f"{effective_max_lag}",
            flush=True,
        )
    mean_radius = 1.0
    if reference.coordinate_kind == "cylindrical":
        mean_radius = float(np.mean(reference.com[:, 2]))
        if not np.isfinite(mean_radius) or mean_radius <= 0.0:
            raise ValueError(f"invalid mean cylinder radius {mean_radius!r} for {key}")
    return PreparedCase(
        key=key,
        case_id=reference.case_id,
        family=reference.family,
        geometry_group=geometry_group(
            reference.geometry_kind, tracer=reference.is_tracer
        ),
        interaction_class=reference.interaction_class,
        lx_multiplier=reference.lx_multiplier,
        coordinate_kind=reference.coordinate_kind,
        seed_count=len(seeds),
        velocities=[
            _finite_difference_vector(seed.com, elapsed) for seed in seeds
        ],
        mean_radius=mean_radius,
        lag_time=_lag_times(elapsed, effective_max_lag),
        max_lag=effective_max_lag,
    )


def _variant_signal(case: PreparedCase, velocity: NDArray[np.float64], variant: str):
    """Select the velocity components that feed the variant's correlation.

    ``axial`` is always the logical axial component. ``full`` takes every
    component, except on a cylinder, where the raw COM columns are
    ``(x, theta, r)`` and mix a length with radians. There the transverse
    velocity is the surface arc rate ``r_mean * theta_dot``, rather than the
    derivative of the product ``r * theta``, which would add a spurious
    ``r_dot * theta`` term -- ``theta`` is unwrapped and so grows without bound.
    """
    if variant == "axial":
        return velocity[:, :1]
    if case.coordinate_kind != "cylindrical":
        return velocity
    return np.stack(
        [velocity[:, 0], case.mean_radius * velocity[:, 1]], axis=1
    )


def _laplace_entry(
    case: PreparedCase,
    variant: str,
    r_points: int,
    omega_points: int,
) -> LaplaceEntry:
    correlations = [
        _normalized_autocorrelation(
            _variant_signal(case, velocity, variant), case.max_lag, normalize=False
        )
        for velocity in case.velocities
    ]
    correlation = np.mean(np.asarray(correlations, dtype=np.float64), axis=0)
    if not np.isfinite(correlation[0]) or correlation[0] == 0.0:
        raise ValueError(
            f"cannot normalize {case.key} with lag-0 value {correlation[0]!r}"
        )
    correlation /= correlation[0]
    r, omega, values = _laplace_transform(
        case.lag_time, correlation, r_points, omega_points
    )
    return LaplaceEntry(
        key=case.key,
        case_id=case.case_id,
        seed_count=case.seed_count,
        r=r,
        omega=omega,
        values=values,
    )


def _load_prepared_cases(frames: int, max_lag: int) -> list[PreparedCase]:
    grouped: dict[str, list[CaseSeries]] = defaultdict(list)
    for index, raw_directory in enumerate(INPUT_DIRS, start=1):
        directory = Path(raw_directory)
        if not directory.is_absolute():
            raise ValueError(f"INPUT_DIRS entries must be absolute: {raw_directory!r}")
        print(
            f"[plot_all] loading {index}/{len(INPUT_DIRS)} dir={directory}",
            flush=True,
        )
        series = load_case_series(directory, frames)
        key = case_key(series.family, series.case_id)
        grouped[key].append(series)
        print(f"[plot_all] loaded {len(series.com)} frames for {key}", flush=True)
    print(
        f"[plot_all] grouped {len(INPUT_DIRS)} directories into "
        f"{len(grouped)} cases",
        flush=True,
    )
    return [_prepare_case(key, seeds, max_lag) for key, seeds in grouped.items()]


def _plot_variant(
    cases: list[PreparedCase],
    variant: str,
    output_dir: Path,
    r_points: int,
    omega_points: int,
) -> None:
    print(f"[plot_all] computing {variant} transforms", flush=True)
    entries = [
        _laplace_entry(case, variant, r_points, omega_points) for case in cases
    ]
    # One check for both outputs: the shared color scale and the entropy bins
    # are each only comparable across cases on an identical grid.
    validate_shared_grid(entries)
    grid_path = output_dir / f"laplacian_grid_{variant}.svg"
    plot_grid(entries, variant, grid_path)
    print(f"[plot_all] wrote {grid_path} and {grid_path.with_suffix('.png')}", flush=True)

    entropy_points = [
        EntropyPoint(
            key=case.key,
            case_id=case.case_id,
            family=case.family,
            geometry_group=case.geometry_group,
            interaction_class=case.interaction_class,
            lx_multiplier=case.lx_multiplier,
            entropy=spectral_entropy(entry.values[:, -1]),
        )
        for case, entry in zip(cases, entries, strict=True)
    ]
    entropy_path = output_dir / f"spectral_entropy_{variant}.svg"
    plot_entropy(entropy_points, variant, entropy_path)
    print(
        f"[plot_all] wrote {entropy_path} and {entropy_path.with_suffix('.png')}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    if not INPUT_DIRS:
        raise ValueError("INPUT_DIRS is empty; edit plot_all.py and add absolute paths")
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")
    if args.laplace_r_points < 2 or args.laplace_omega_points < 2:
        raise ValueError("Laplace grid dimensions must each be at least 2")

    cases = _load_prepared_cases(args.frames, args.max_lag)
    for variant in ("full", "axial"):
        _plot_variant(
            cases,
            variant,
            args.output_dir,
            args.laplace_r_points,
            args.laplace_omega_points,
        )
    print("[plot_all] done", flush=True)


if __name__ == "__main__":
    main()
