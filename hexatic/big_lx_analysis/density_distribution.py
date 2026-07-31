"""Compute and fit surface-excess density distributions for big-Lx cases.

This command first calls :mod:`rho_gamma_finding` to ensure that every case
has a cached ``rho_gamma`` and film cutoff.  It then implements PLAN.md steps
2 and 3 on equal-area ``(x, theta)`` bins:

``rho_s,i = (<N_i> - rho_gamma * V_i) / A_i``.

For each case, the plot shows the area-weighted probability distribution
``P(rho_s)`` and a probabilistic two- or three-component Gaussian mixture.
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
from sklearn.mixture import GaussianMixture

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


@dataclass(frozen=True)
class MixtureComponent:
    name: str
    mean: float
    standard_deviation: float
    weight: float


@dataclass(frozen=True)
class MixtureFit:
    n_components: int
    bic: float
    components: tuple[MixtureComponent, ...]


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
    frame_counts: list[NDArray[np.float64]] = []
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
                frame_counts.append(
                    np.bincount(
                        x_index * ntheta + theta_index,
                        minlength=nx * ntheta,
                    ).reshape(nx, ntheta)
                )
                n_frames += 1
    if n_frames == 0 or n_particles is None:
        raise ValueError(f"No frames selected for {replicate.case_id}")
    return SurfaceGrid(
        np.stack(frame_counts).astype(np.float64),
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
        counts=np.concatenate([grid.counts for grid in grids], axis=0),
        n_frames=sum(grid.n_frames for grid in grids),
        n_particles=reference.n_particles,
        nx=reference.nx,
        ntheta=reference.ntheta,
        dx=reference.dx,
        dtheta=reference.dtheta,
        lx=reference.lx,
        radius=reference.radius,
    )


def _fit_phase_mixture(
    values: NDArray[np.float64],
    *,
    requested_components: str,
    max_fit_samples: int,
) -> MixtureFit:
    """Fit a two- or three-component Gaussian mixture to equal-area samples."""
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        raise ValueError("At least four finite surface-density samples are required")
    rng = np.random.default_rng(0)
    fit_values = (
        rng.choice(finite, size=max_fit_samples, replace=False)
        if finite.size > max_fit_samples
        else finite
    )
    fit_column = fit_values.reshape(-1, 1)
    candidates = (
        [2, 3] if requested_components == "auto" else [int(requested_components)]
    )
    fitted: list[tuple[float, GaussianMixture]] = []
    for n_components in candidates:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            n_init=10,
            random_state=0,
            reg_covar=1.0e-8,
        )
        model.fit(fit_column)
        fitted.append((float(model.bic(fit_column)), model))
    bic, model = min(fitted, key=lambda item: item[0])
    means = model.means_[:, 0]
    order = np.argsort(means)
    names = (
        ("beta", "alpha")
        if model.n_components == 2
        else ("beta", "interface", "alpha")
    )
    components = tuple(
        MixtureComponent(
            name=name,
            mean=float(means[index]),
            standard_deviation=float(np.sqrt(model.covariances_[index, 0, 0])),
            weight=float(model.weights_[index]),
        )
        for name, index in zip(names, order, strict=True)
    )
    return MixtureFit(model.n_components, bic, components)


def _normal_pdf(
    x: NDArray[np.float64], mean: float, standard_deviation: float
) -> NDArray[np.float64]:
    z = (x - mean) / standard_deviation
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * standard_deviation)


def _plot_density_distribution(
    case_label: str,
    values: NDArray[np.float64],
    mixture: MixtureFit,
    number_residual: float,
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
    colors = sns.color_palette("colorblind", n_colors=mixture.n_components)
    total = np.zeros_like(x)
    for color, component in zip(colors, mixture.components, strict=True):
        component_pdf = component.weight * _normal_pdf(
            x, component.mean, component.standard_deviation
        )
        total += component_pdf
        symbol = "I" if component.name == "interface" else component.name
        axis.plot(
            x,
            component_pdf,
            color=color,
            linestyle="--",
            linewidth=1.4,
            label=(
                rf"$\rho_{{{symbol}}}={component.mean:.4g}$, "
                rf"$x_{{{symbol}}}={component.weight:.3f}$"
            ),
        )
        axis.axvline(component.mean, color=color, linestyle=":", linewidth=0.9)
    axis.plot(x, total, color="black", linewidth=1.8, label="mixture total")
    axis.set(
        title=case_label,
        xlabel=r"Local surface-excess density $\rho_s$ [$N/L^2$]",
        ylabel=r"Probability density $P(\rho_s)$",
        xlim=(x[0], x[-1]),
    )
    axis.text(
        0.98,
        0.96,
        rf"$\epsilon_N={number_residual:.6g}$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize="small",
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
    parser.add_argument("--density-bins", type=int, default=80)
    parser.add_argument(
        "--mixture-components",
        choices=("auto", "2", "3"),
        default="auto",
        help="Fit two, three, or the BIC-selected number of components",
    )
    parser.add_argument(
        "--mixture-max-fit-samples",
        type=int,
        default=100_000,
        help="Maximum deterministic subsample used to fit the mixture",
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
    if args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be positive")
    if args.density_bins < 1:
        raise ValueError("--density-bins must be positive")
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

        bin_area = grid.radius * grid.dx * grid.dtheta
        bin_volume = (
            0.5
            * grid.dx
            * grid.dtheta
            * (grid.radius**2 - radial_cutoff**2)
        )
        rho_surface_field = (
            grid.counts - rho_gamma * bin_volume
        ) / bin_area
        values = rho_surface_field.reshape(-1)
        # Every (x, theta) bin has area A_i = R dx dtheta.  Repeating those
        # equal weights for every selected frame implements
        # rho_surface = sum_i A_i rho_s,i / sum_i A_i explicitly.
        area_weights = np.full(values.shape, bin_area, dtype=np.float64)
        rho_surface = float(
            np.average(values, weights=area_weights)
        )
        mixture = _fit_phase_mixture(
            values,
            requested_components=args.mixture_components,
            max_fit_samples=args.mixture_max_fit_samples,
        )

        stem = safe_case_name(case_id)
        plot_output = args.output_dir / f"{stem}_density_distribution.svg"
        data_output = args.output_dir / f"{stem}_surface_excess_summary.json"
        existing = [
            path for path in (plot_output, data_output) if path.exists()
        ]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"{existing[0]} exists; pass --overwrite to replace"
            )
        beta = mixture.components[0]
        alpha = mixture.components[-1]
        total_volume = np.pi * grid.radius**2 * grid.lx
        surface_area = 2.0 * np.pi * grid.radius * grid.lx
        phase_surface_density = (
            alpha.weight * alpha.mean + beta.weight * beta.mean
        )
        reconstructed_n = (
            total_volume * rho_gamma
            + surface_area * phase_surface_density
        )
        actual_n = float(grid.n_particles)
        number_residual = (actual_n - reconstructed_n) / actual_n

        _plot_density_distribution(
            case_replicates[0].label,
            values,
            mixture,
            number_residual,
            args.density_bins,
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
            "rho_surface_averaging": (
                "area-weighted mean of per-frame equal-area local bins"
            ),
            "nx": grid.nx,
            "ntheta": grid.ntheta,
            "n_selected_frames": grid.n_frames,
            "n_replicates": len(grids),
            "n_distribution_samples": int(values.size),
            "mixture_components": mixture.n_components,
            "mixture_bic": mixture.bic,
            "rho_beta": beta.mean,
            "rho_alpha": alpha.mean,
            "rho_phase_units": "N/L^2",
            "x_beta": beta.weight,
            "x_alpha": alpha.weight,
            "total_volume": total_volume,
            "surface_area": surface_area,
            "actual_n": actual_n,
            "alpha_beta_surface_density": phase_surface_density,
            "alpha_beta_reconstructed_n": reconstructed_n,
            "epsilon_n": number_residual,
            "epsilon_n_formula": (
                "(N - (V*rho_gamma + A_s*(x_alpha*rho_alpha + "
                "x_beta*rho_beta))) / N"
            ),
            "components": [
                {
                    "name": component.name,
                    "mean": component.mean,
                    "standard_deviation": component.standard_deviation,
                    "weight": component.weight,
                }
                for component in mixture.components
            ],
        }
        data_output.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            f"{case_id}: rho_surface={rho_surface:.6g}, "
            f"rho_beta={beta.mean:.6g}, rho_alpha={alpha.mean:.6g}, "
            f"x_beta={beta.weight:.6g}, x_alpha={alpha.weight:.6g}, "
            f"epsilon_N={number_residual:.6g}, "
            f"components={mixture.n_components}, BIC={mixture.bic:.6g}",
            flush=True,
        )
        print(f"wrote {plot_output}", flush=True)
        print(f"wrote {data_output}", flush=True)


if __name__ == "__main__":
    main()
