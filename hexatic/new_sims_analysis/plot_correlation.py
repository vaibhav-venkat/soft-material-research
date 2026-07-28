"""Plot velocity/orientation correlations, their Laplace transform, and the MSD.

Each seed directory is reduced on its own; only the resulting curves are
averaged across seeds, never the raw positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hexatic.constants import cylinder

from .correlation import _correlation_curves, _lag_times, _plot_correlation
from .laplace import _laplace_transform, _plot_laplace
from .loading import (
    CYLINDRICAL_COORDINATE_ORDER,
    PLANAR_COORDINATE_ORDER,
    _case_identity,
    _elapsed_time,
    _load_frame_series,
    _load_manifest,
)
from .msd import _fit_msd_power_law, _msd_from_com, _msd_variants, _plot_msd


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        required=True,
        help=(
            "Safetensor output directory for one random seed (repeatable). "
            "All directories are treated as one simulation ensemble."
        ),
    )
    parser.add_argument(
        "--2d",
        dest="two_d",
        action="store_true",
        help=(
            "Enable 2D Cartesian inputs with coordinate order ['x', 'y']; "
            "velocity and orientation correlations use both planar components."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("correlation_output"),
        help="Directory for the output SVG and PNG (default: correlation_output/).",
    )
    parser.add_argument(
        "--frames",
        type=_nonnegative_int,
        default=1000,
        help="Maximum number of initial frames to use (default: 1000).",
    )
    parser.add_argument(
        "--max-lag",
        type=_nonnegative_int,
        default=998,
        help="Maximum lag in frames (default: 998).",
    )
    parser.add_argument(
        "--timestep",
        type=_positive_float,
        default=None,
        help=(
            "Override the simulation timestep (default: case metadata, then "
            "the cylinder constant)."
        ),
    )
    parser.add_argument(
        "--unnormalize",
        action="store_true",
        help="Plot unbiased autocorrelations without dividing by lag zero.",
    )
    parser.add_argument(
        "--msd-fit-start",
        type=_positive_float,
        default=50.0,
        help=(
            "Simulation time tau after which the MSD is fitted to A t^alpha "
            "with t = tau - msd_fit_start (default: 50.0)."
        ),
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
        help="Number of Laplace angular-frequency samples (default: 241).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")
    if args.laplace_r_points < 2 or args.laplace_omega_points < 2:
        raise ValueError("Laplace grid dimensions must each be at least 2")

    seed_curves: list[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ] = []
    seed_msd: dict[str, list[NDArray[np.float64]]] = {}
    msd_titles: dict[str, str] = {}
    reference_steps: NDArray[np.int64] | None = None
    reference_timestep: float | None = None
    reference_u0: float | None = None
    reference_particles: int | None = None
    reference_coordinate_order: list[str] | None = None
    label: str | None = None
    slug: str | None = None

    for index, input_dir in enumerate(args.input_dir, start=1):
        print(
            f"[correlation] loading seed {index}/{len(args.input_dir)} "
            f"dir={input_dir}",
            flush=True,
        )
        manifest = _load_manifest(input_dir, allow_2d=args.two_d)
        coordinate_order = manifest["coordinate_order"]
        if args.two_d and coordinate_order != PLANAR_COORDINATE_ORDER:
            raise ValueError(
                f"--2d requires coordinate_order {PLANAR_COORDINATE_ORDER}, "
                f"got {coordinate_order!r} for {input_dir}"
            )
        if reference_coordinate_order is None:
            reference_coordinate_order = coordinate_order
        elif coordinate_order != reference_coordinate_order:
            raise ValueError(
                "all --input-dir values must use the same geometry; cannot "
                "mix 2D Cartesian, 3D Cartesian, and cylindrical seeds"
            )
        seed_label, seed_slug = _case_identity(input_dir, manifest)
        if label is None:
            label, slug = seed_label, seed_slug
        elif seed_slug != slug:
            raise ValueError(
                "all --input-dir manifests must describe the same case: "
                f"expected {slug!r}, got {seed_slug!r} for {input_dir}"
            )

        case_meta = manifest.get("case", {})
        stored_timestep = case_meta.get("timestep")
        if args.timestep is not None:
            timestep = args.timestep
        elif stored_timestep is not None:
            timestep = float(stored_timestep)
            if timestep <= 0.0 or not np.isfinite(timestep):
                raise ValueError(
                    f"invalid timestep in manifest for {input_dir}: "
                    f"{stored_timestep!r}"
                )
        else:
            timestep = float(cylinder.SIMULATION.timestep)
        u0 = float(case_meta.get("u0", cylinder.SIMULATION.u0))
        if not np.isfinite(u0) or u0 <= 0.0:
            raise ValueError(f"invalid U0 in manifest for {input_dir}: {u0!r}")

        com, q, steps = _load_frame_series(input_dir, manifest, args.frames)
        print(
            f"[correlation] loaded {len(com)} frames, "
            f"{q.shape[1]} particles",
            flush=True,
        )
        if reference_steps is None:
            reference_steps = steps
            reference_timestep = timestep
            reference_u0 = u0
            reference_particles = q.shape[1]
        else:
            if not np.array_equal(steps, reference_steps):
                raise ValueError(
                    f"seed frame steps do not match in {input_dir}"
                )
            if timestep != reference_timestep or u0 != reference_u0:
                raise ValueError(
                    f"seed timestep or U0 does not match in {input_dir}"
                )
            if q.shape[1] != reference_particles:
                raise ValueError(
                    f"seed particle count does not match in {input_dir}"
                )
        elapsed = _elapsed_time(steps, timestep)
        effective_max_lag = min(args.max_lag, len(steps) - 2)
        for suffix, title, coords in _msd_variants(
            com, coordinate_order == CYLINDRICAL_COORDINATE_ORDER
        ):
            seed_msd.setdefault(suffix, []).append(_msd_from_com(coords))
            msd_titles[suffix] = title
        seed_curves.append(
            _correlation_curves(
                com,
                q,
                elapsed,
                effective_max_lag,
                u0,
                coordinate_order == CYLINDRICAL_COORDINATE_ORDER,
                coordinate_order == PLANAR_COORDINATE_ORDER,
            )
        )

    assert reference_steps is not None
    assert reference_timestep is not None
    assert reference_u0 is not None
    assert reference_particles is not None
    assert reference_coordinate_order is not None
    assert label is not None and slug is not None
    cylindrical = reference_coordinate_order == CYLINDRICAL_COORDINATE_ORDER
    planar = reference_coordinate_order == PLANAR_COORDINATE_ORDER

    elapsed = _elapsed_time(reference_steps, reference_timestep)
    effective_max_lag = min(args.max_lag, len(reference_steps) - 2)
    if effective_max_lag != args.max_lag:
        print(
            f"[correlation] capping --max-lag {args.max_lag} -> "
            f"{effective_max_lag}",
            flush=True,
        )

    velocity_correlation, self_correlation, distinct_correlation = np.mean(
        np.asarray(seed_curves, dtype=np.float64), axis=0
    )
    normalize = not args.unnormalize
    if normalize:
        for name, correlation in (
            ("C_V", velocity_correlation),
            ("C_q_self", self_correlation),
            ("C_q_distinct", distinct_correlation),
        ):
            if correlation[0] == 0.0:
                raise ValueError(f"cannot normalize {name} with zero lag-0 value")
            correlation /= correlation[0]

    lag_time = _lag_times(elapsed, effective_max_lag)
    output_path = args.output_dir / f"velocity_correlation_{slug}.svg"
    _plot_correlation(
        lag_time,
        velocity_correlation,
        self_correlation,
        distinct_correlation,
        normalize,
        cylindrical,
        planar,
        f"{label} ({len(seed_curves)} seeds)",
        output_path,
    )
    print(f"[correlation] wrote {output_path}", flush=True)
    print(f"[correlation] wrote {output_path.with_suffix('.png')}", flush=True)
    for suffix, curves in seed_msd.items():
        msd = np.mean(np.asarray(curves, dtype=np.float64), axis=0)
        alpha, amplitude, tau_fit, msd_fit = _fit_msd_power_law(
            elapsed, msd, args.msd_fit_start
        )
        msd_path = args.output_dir / f"msd{suffix}_{slug}.svg"
        _plot_msd(
            elapsed,
            msd,
            tau_fit,
            msd_fit,
            alpha,
            len(curves),
            f"{label}{msd_titles[suffix]}",
            msd_path,
        )
        print(
            f"[correlation] MSD{suffix} fit for tau > {args.msd_fit_start}: "
            f"alpha={alpha:.4f}, A={amplitude:.4g}",
            flush=True,
        )
        print(f"[correlation] wrote {msd_path}", flush=True)
        print(f"[correlation] wrote {msd_path.with_suffix('.png')}", flush=True)

    laplace_correlation = velocity_correlation.copy()
    if not normalize:
        if laplace_correlation[0] == 0.0:
            raise ValueError(
                "cannot normalize the velocity correlation for its Laplace transform"
            )
        laplace_correlation /= laplace_correlation[0]
    r, omega, laplace_values = _laplace_transform(
        lag_time,
        laplace_correlation,
        args.laplace_r_points,
        args.laplace_omega_points,
    )
    laplace_path = args.output_dir / f"laplace_transform_{slug}.svg"
    _plot_laplace(r, omega, laplace_values, laplace_path)
    print(f"[correlation] wrote {laplace_path}", flush=True)
    print(f"[correlation] wrote {laplace_path.with_suffix('.png')}", flush=True)
    print(
        f"[correlation] done — averaged {len(seed_curves)} seed(s) into one "
        "correlation plot, one MSD plot, and one Laplace heatmap"
    )


if __name__ == "__main__":
    main()
