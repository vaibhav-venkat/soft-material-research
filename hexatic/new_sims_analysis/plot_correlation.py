"""Plot velocity/orientation correlations, their Laplace transform, and the MSD.

Each seed directory is reduced on its own; only the resulting curves are
averaged across seeds, never the raw positions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hexatic.constants import cylinder

from .correlation import _correlation_curves, _lag_times, _plot_correlation
from .isf import (
    ISF_Q_VALUES,
    IsfResult,
    average_seed_results,
    component_seed,
    isotropic_3d_seed,
    plot_heatmap,
    plot_summary,
    select_particle_ids,
    validate_results,
)
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


_BULK_GEOMETRIES = {
    "periodic_3d_bulk",
    "ideal_3d_bulk",
    "ideal_3d_bulk_period_10",
    "ideal_3d_bulk_period_1",
}
_TRACER_GEOMETRIES = {
    "single_active_tracer_cylinder",
    "inversion_active_pair_cylinder",
}


def _isf_geometry_route(
    case_meta: dict[str, Any],
    coordinate_order: list[str],
    *,
    disabled: bool,
) -> str | None:
    """Return ``bulk``/``cylinder`` or explain why this case is skipped."""
    if disabled:
        return None
    geometry = str(case_meta.get("geometry_kind", ""))
    n_particles = int(case_meta.get("n_particles", 0))
    active_count = int(case_meta.get("active_count", n_particles))
    if geometry in _TRACER_GEOMETRIES or (n_particles > 0 and active_count < n_particles):
        raise ValueError(
            "ISF analysis is not defined for tracer cases; pass --no-isf "
            "to retain correlation/MSD/Laplace analysis"
        )
    if coordinate_order == CYLINDRICAL_COORDINATE_ORDER:
        return "cylinder"
    if coordinate_order == PLANAR_COORDINATE_ORDER:
        return None
    if geometry in _BULK_GEOMETRIES:
        return "bulk"
    return None


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
        "--tau-min",
        type=_positive_float,
        nargs="+",
        default=[50.0],
        help="Lower lag bound(s) of the MSD fit windows (default: 50).",
    )
    parser.add_argument(
        "--tau-max",
        type=_positive_float,
        nargs="+",
        default=None,
        help="Upper lag bound(s), one per --tau-min (default: longest lag).",
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
    parser.add_argument(
        "--no-isf",
        action="store_true",
        help="Disable ISF trajectory retention and plots.",
    )
    parser.add_argument(
        "--isf-particles",
        type=_positive_int,
        default=256,
        help="Maximum deterministic particle sample for ISFs (default: 256).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")
    if args.laplace_r_points < 2 or args.laplace_omega_points < 2:
        raise ValueError("Laplace grid dimensions must each be at least 2")
    if args.tau_max is not None and len(args.tau_max) != len(args.tau_min):
        raise ValueError("--tau-min and --tau-max must have the same length")
    seed_curves: list[
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ]
    ] = []
    seed_msd: dict[str, list[NDArray[np.float64]]] = {}
    msd_titles: dict[str, str] = {}
    msd_dimensions: dict[str, int] = {}
    reference_steps: NDArray[np.int64] | None = None
    reference_timestep: float | None = None
    reference_u0: float | None = None
    reference_tau_r: float | None = None
    reference_radius: float | None = None
    reference_geometry: str | None = None
    reference_particles: int | None = None
    reference_coordinate_order: list[str] | None = None
    isf_route: str | None = None
    sampled_particle_ids: NDArray[np.int64] | None = None
    isf_inputs: list[
        tuple[NDArray[np.float64], NDArray[np.float64]]
    ] = []
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
        seed_route = _isf_geometry_route(
            case_meta, coordinate_order, disabled=args.no_isf
        )
        if index == 1:
            isf_route = seed_route
            if not args.no_isf and seed_route is None:
                geometry_name = case_meta.get("geometry_kind", "unknown")
                print(
                    f"[isf] skipping unsupported 2D/prism geometry "
                    f"{geometry_name!r}; existing plots remain enabled",
                    flush=True,
                )
        elif seed_route != isf_route:
            raise ValueError("seed ISF geometry routing does not match")
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
        tau_r = float(case_meta.get("tau_r", cylinder.SIMULATION.tau_r))
        if not np.isfinite(tau_r) or tau_r <= 0.0:
            raise ValueError(
                f"invalid tau_r in manifest for {input_dir}: {tau_r!r}"
            )
        geometry = str(case_meta.get("geometry_kind", ""))
        radius: float | None = None
        if coordinate_order == CYLINDRICAL_COORDINATE_ORDER:
            stored_radius = case_meta.get("radius")
            if stored_radius is None:
                raise ValueError(f"missing cylinder radius in {input_dir}")
            radius = float(stored_radius)
            if not np.isfinite(radius) or radius <= 0.0:
                raise ValueError(
                    f"invalid cylinder radius in {input_dir}: {stored_radius!r}"
                )

        if isf_route is not None and sampled_particle_ids is None:
            stored_particles = case_meta.get("n_particles")
            if stored_particles is None:
                raise ValueError(
                    f"ISF analysis requires n_particles metadata in {input_dir}"
                )
            sampled_particle_ids = select_particle_ids(
                int(stored_particles), args.isf_particles
            )

        if sampled_particle_ids is None:
            com, q, steps = _load_frame_series(
                input_dir, manifest, args.frames
            )
            sampled_tracks = None
        else:
            com, q, steps, sampled_tracks = _load_frame_series(
                input_dir, manifest, args.frames, sampled_particle_ids
            )
        print(
            f"[correlation] loaded {len(com)} frames, "
            f"{q.shape[1]} particles",
            flush=True,
        )
        if reference_steps is None:
            reference_steps = steps
            reference_timestep = timestep
            reference_u0 = u0
            reference_tau_r = tau_r
            reference_radius = radius
            reference_geometry = geometry
            reference_particles = q.shape[1]
        else:
            if not np.array_equal(steps, reference_steps):
                raise ValueError(
                    f"seed frame steps do not match in {input_dir}"
                )
            if (
                timestep != reference_timestep
                or u0 != reference_u0
                or tau_r != reference_tau_r
            ):
                raise ValueError(
                    f"seed timestep, U0, or tau_r does not match in {input_dir}"
                )
            if geometry != reference_geometry or radius != reference_radius:
                raise ValueError(
                    f"seed geometry or cylinder radius does not match in {input_dir}"
                )
            if q.shape[1] != reference_particles:
                raise ValueError(
                    f"seed particle count does not match in {input_dir}"
                )
        stored_particles = case_meta.get("n_particles")
        if stored_particles is not None and int(stored_particles) != q.shape[1]:
            raise ValueError(
                f"manifest particle count does not match tensors in {input_dir}"
            )
        elapsed = _elapsed_time(steps, timestep)
        effective_max_lag = min(args.max_lag, len(steps) - 2)
        for suffix, title, coords, transport_dimensions in _msd_variants(
            com, coordinate_order == CYLINDRICAL_COORDINATE_ORDER
        ):
            seed_msd.setdefault(suffix, []).append(
                _msd_from_com(coords, effective_max_lag)
            )
            msd_titles[suffix] = title
            msd_dimensions[suffix] = transport_dimensions
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
        if sampled_tracks is not None:
            isf_inputs.append((com, sampled_tracks))

    assert reference_steps is not None
    assert reference_timestep is not None
    assert reference_u0 is not None
    assert reference_tau_r is not None
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

    isf_results: list[IsfResult] = []
    if isf_route is not None:
        assert sampled_particle_ids is not None
        persistence_length = reference_u0 * reference_tau_r
        k_values = ISF_Q_VALUES / persistence_length
        seed_estimators: dict[str, list] = {}
        for com, sampled_tracks in isf_inputs:
            if isf_route == "bulk":
                seed_estimators.setdefault("isotropic", []).append(
                    isotropic_3d_seed(
                        com, sampled_tracks, k_values,
                        reference_particles, effective_max_lag,
                    )
                )
                seed_estimators.setdefault("x", []).append(
                    component_seed(
                        com[:, 0], sampled_tracks[:, :, 0], k_values,
                        reference_particles, effective_max_lag,
                    )
                )
            else:
                assert reference_radius is not None
                seed_estimators.setdefault("x", []).append(
                    component_seed(
                        com[:, 0], sampled_tracks[:, :, 0], k_values,
                        reference_particles, effective_max_lag,
                    )
                )
                seed_estimators.setdefault("theta", []).append(
                    component_seed(
                        reference_radius * com[:, 1], sampled_tracks[:, :, 1],
                        k_values, reference_particles, effective_max_lag,
                    )
                )
        if isf_route == "bulk":
            result_specs = (
                ("isotropic", r"isotropic 3D $F(k,\tau)$", 3),
                ("x", r"axial $F_x(k,\tau)$", 1),
            )
        else:
            result_specs = (
                ("x", r"axial $F_x(k,\tau)$", 1),
                ("theta", r"azimuthal $F_\theta(k,\tau)$", 1),
            )
        isf_results = [
            average_seed_results(
                key, title, dimensions, seed_estimators[key], k_values
            )
            for key, title, dimensions in result_specs
        ]
        diagnostics = validate_results(isf_results, k_values)
        scaled_lag = lag_time / reference_tau_r
        summary_path = args.output_dir / f"isf_summary_{slug}.svg"
        heatmap_path = args.output_dir / f"isf_heatmap_{slug}.svg"
        plot_summary(
            scaled_lag, ISF_Q_VALUES, isf_results, label,
            len(seed_curves), summary_path,
        )
        plot_heatmap(
            scaled_lag, ISF_Q_VALUES, isf_results, label, heatmap_path
        )
        small_k_error = diagnostics.maximum_small_k_msd_error
        small_k_summary = (
            "n/a (no eligible lag)"
            if small_k_error is None
            else f"{small_k_error:.3%}"
        )
        print(
            f"[isf] N={reference_particles}, selected="
            f"{len(sampled_particle_ids)}, U0={reference_u0:g}, "
            f"tau_r={reference_tau_r:g}, l_p={persistence_length:g}, "
            f"k={np.array2string(k_values, precision=6)}, "
            f"max |Im F|={diagnostics.maximum_imaginary_leakage:.4g}, "
            f"max small-k MSD error={small_k_summary}",
            flush=True,
        )
        for path in (summary_path, heatmap_path):
            print(f"[isf] wrote {path}", flush=True)
            print(f"[isf] wrote {path.with_suffix('.png')}", flush=True)
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
    longest_lag = float(lag_time[-1])
    tau_max_values = (
        args.tau_max
        if args.tau_max is not None
        else [longest_lag] * len(args.tau_min)
    )
    fit_windows = list(zip(args.tau_min, tau_max_values, strict=True))
    for tau_min, _ in fit_windows:
        if tau_min >= longest_lag:
            raise ValueError(
                f"--tau-min {tau_min:g} is at or beyond the longest available "
                f"lag {longest_lag:g}; raise --frames or --max-lag"
            )

    for suffix, curves in seed_msd.items():
        # The COM averages out N independent trajectories, so its MSD is the
        # single-particle MSD over N. Scaling back up recovers the equivalent
        # per-particle MSD; exact only because these particles do not interact.
        msd = (
            np.mean(np.asarray(curves, dtype=np.float64), axis=0)
            * reference_particles
        )
        msd_path = args.output_dir / f"msd{suffix}_{slug}.svg"
        fits = [
            _fit_msd_power_law(
                lag_time, msd, tau_min, tau_max, msd_dimensions[suffix]
            )
            for tau_min, tau_max in fit_windows
        ]
        _plot_msd(
            lag_time,
            msd,
            fits,
            len(curves),
            reference_particles,
            f"{label}{msd_titles[suffix]}",
            msd_path,
        )
        for fit in fits:
            print(
                f"[correlation] MSD{suffix} fit over "
                f"[{fit.tau_min:g}, {fit.tau_max:g}]: "
                f"alpha={fit.alpha:.4f}, A={fit.amplitude:.4g}, "
                f"D={fit.diffusion:.4g} (d_eff={fit.dimensions})",
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
        + (", plus ISF summary and heatmap pairs" if isf_results else "")
    )


if __name__ == "__main__":
    main()
