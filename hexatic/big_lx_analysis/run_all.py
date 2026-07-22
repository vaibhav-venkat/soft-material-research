"""Run COM, Laplacian, and structural-cluster analyses and write SVG summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hexatic.constants import cylinder

from .clusters import ClusterOptions, ClusterRatioMode, analyze_clusters
from .dynamics import DynamicsResult
from .laplacian import LaplacianOptions, LaplacianResult, analyze_laplacian
from .run_dynamics import (
    _collect_manifests,
    _load_replicate,
    _plot_com,
    _plot_correlation,
    _require_common_grid,
    _style,
    _summarize,
)


def _save(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output, format="svg", bbox_inches="tight",
                   metadata={"Creator": "hexatic.big_lx_analysis"})
    plt.close(figure)


def _plot_laplace(results: dict[str, tuple[str, list[LaplacianResult]]], output: Path) -> None:
    sns = _style()
    rows = len(results)
    figure, axes = plt.subplots(rows, 3, figsize=(12.0, 3.4 * rows),
                                squeeze=False, constrained_layout=True)
    column_titles = (r"$\log_{10}|L(r,\omega)|$", r"$\Re L(r,\omega)$", r"$\Im L(r,\omega)$")
    for row, (case_id, (label, values)) in enumerate(results.items()):
        reference = values[0]
        for value in values[1:]:
            _require_common_grid(reference.r, value.r, case_id, "r")
            _require_common_grid(reference.omega, value.omega, case_id, "omega")
        mean = np.mean(np.stack([value.values for value in values]), axis=0)
        fields = (np.log10(np.maximum(np.abs(mean), np.finfo(float).tiny)), mean.real, mean.imag)
        for column, field in enumerate(fields):
            axis = axes[row, column]
            image = axis.pcolormesh(reference.r, reference.omega, field, shading="auto", cmap="viridis")
            figure.colorbar(image, ax=axis, pad=0.02)
            axis.set_xlabel(r"decay coordinate $r$")
            if column == 0:
                axis.set_ylabel(f"{label}\nangular frequency $\\omega$")
            if row == 0:
                axis.set_title(column_titles[column])
            sns.despine(ax=axis)
    _save(figure, output)


def _plot_preferred(results: dict[str, tuple[str, list[LaplacianResult]]], output: Path) -> None:
    sns = _style()
    labels = [label for label, _ in results.values()]
    r_values = [np.array([value.preferred_r.coordinate for value in values]) for _, values in results.values()]
    omega_values = [np.array([value.preferred_omega.coordinate for value in values]) for _, values in results.values()]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    for axis, samples, title, ylabel in (
        (axes[0], r_values, "Preferred decay coordinate", r"preferred $r$"),
        (axes[1], omega_values, "Preferred angular frequency", r"preferred $\omega$"),
    ):
        means = np.array([sample.mean() for sample in samples])
        errors = np.array([sample.std(ddof=1) if sample.size > 1 else 0.0 for sample in samples])
        axis.errorbar(np.arange(len(labels)), means, yerr=errors, fmt="o", capsize=4, lw=1.5)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="0.9")
        sns.despine(ax=axis)
    _save(figure, output)


def _plot_fits(
    dynamics: dict[str, tuple[str, list[DynamicsResult]]],
    laplace: dict[str, tuple[str, list[LaplacianResult]]],
    output: Path,
) -> None:
    sns = _style()
    colors = sns.color_palette("colorblind", n_colors=len(laplace))
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for color, case_id in zip(colors, laplace, strict=True):
        label, fits = laplace[case_id]
        correlations = dynamics[case_id][1]
        lag = correlations[0].lag_time
        for value in correlations[1:]:
            _require_common_grid(lag, value.lag_time, case_id, "lag-time")
        observed = np.mean(np.stack([value.pearson for value in correlations]), axis=0)
        predicted = np.mean(np.stack([value.fit.prediction for value in fits]), axis=0)
        if predicted.shape != lag.shape:
            raise ValueError(f"Fit prediction length differs from lag grid for {case_id}")
        axis.plot(lag, observed, color=color, lw=2.0, label=f"{label} observed")
        axis.plot(lag, predicted, color=color, lw=1.6, ls="--", label=f"{label} fit")
    axis.axhline(0.0, color="0.7", lw=0.8)
    axis.set(title="Damped-cosine fits", xlabel="Lag time", ylabel="Pearson correlation")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", color="0.9")
    sns.despine(ax=axis)
    _save(figure, output)


def _plot_clusters(samples: dict[str, tuple[str, list[np.ndarray]]], output: Path) -> None:
    sns = _style()
    colors = sns.color_palette("colorblind", n_colors=len(samples))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    pooled = [np.concatenate(replicates) for _, replicates in samples.values() if replicates]
    pooled = [values for values in pooled if values.size]
    if not pooled:
        for axis in axes:
            axis.text(0.5, 0.5, "No qualifying structural clusters",
                      ha="center", va="center", transform=axis.transAxes)
            sns.despine(ax=axis)
        _save(figure, output)
        return
    all_area = np.concatenate(pooled)
    area_bins = np.histogram_bin_edges(all_area, bins="auto")
    sqrt_bins = np.histogram_bin_edges(np.sqrt(all_area), bins="auto")
    for color, (label, replicates) in zip(colors, samples.values(), strict=True):
        area = np.concatenate(replicates) if replicates else np.empty(0)
        if area.size == 0:
            continue
        # A cluster contributes in proportion to the surface area it contains,
        # rather than every tiny and large cluster receiving equal mass.
        sns.histplot(area, weights=area, stat="probability", element="step", fill=False,
                     common_norm=False, bins=area_bins, color=color, label=label, ax=axes[0])
        sns.histplot(np.sqrt(area), weights=area, stat="probability", element="step", fill=False,
                     common_norm=False, bins=sqrt_bins, color=color, label=label, ax=axes[1])
    axes[0].set(
        title="Area-weighted structural-cluster fractions",
        xlabel=r"$A/SA$",
        ylabel="Area-weighted probability",
    )
    axes[1].set(
        title="Area-weighted structural-cluster circumference ratios",
        xlabel=r"$\sqrt{A/SA}$",
        ylabel="Area-weighted probability",
    )
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(axis="y", color="0.9")
        sns.despine(ax=axis)
    _save(figure, output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("big_lx_analysis_output"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timestep", type=float, default=float(cylinder.SIMULATION.timestep))
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--max-lag", type=int)
    parser.add_argument("--device-ordinal", type=int, default=0)
    parser.add_argument("--r-points", type=int, default=161)
    parser.add_argument("--omega-points", type=int, default=241)
    parser.add_argument("--psi6-minimum", type=float, default=0.7)
    parser.add_argument("--misorientation-degrees", type=float, default=5.0)
    parser.add_argument("--neighbor-radius-diameters", type=float, default=1.7272)
    parser.add_argument("--minimum-particles", type=int, default=2)
    parser.add_argument("--particle-diameter", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists; pass --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    replicates = [_load_replicate(path) for path in _collect_manifests(args.manifest, args.input_dir)]
    if args.case:
        selected = set(args.case)
        replicates = [value for value in replicates if value.case_id in selected]
    if not replicates:
        raise ValueError("No matching replicates")

    laplacian_options = LaplacianOptions(
        frame_start=args.frame_start, frame_stop=args.frame_stop, timestep=args.timestep,
        max_lag=args.max_lag, device_ordinal=args.device_ordinal,
        r_points=args.r_points, omega_points=args.omega_points,
        preferred_r_points=args.r_points, preferred_omega_points=args.omega_points,
    )
    cluster_options = ClusterOptions(
        frame_start=args.frame_start, frame_stop=args.frame_stop,
        psi6_minimum=args.psi6_minimum,
        misorientation_degrees=args.misorientation_degrees,
        neighbor_radius_diameters=args.neighbor_radius_diameters,
        minimum_particles=args.minimum_particles, particle_diameter=args.particle_diameter,
        ratio_mode=ClusterRatioMode.AREA_FRACTION,
    )
    dynamics: dict[str, tuple[str, list[DynamicsResult]]] = {}
    laplace: dict[str, tuple[str, list[LaplacianResult]]] = {}
    cluster_samples: dict[str, tuple[str, list[np.ndarray]]] = {}
    for index, replicate in enumerate(replicates, 1):
        print(f"[analysis] {index}/{len(replicates)} case={replicate.case_id}", flush=True)
        l = analyze_laplacian(replicate.static_file, replicate.shard_files, laplacian_options)
        d = l.dynamics
        c = analyze_clusters(replicate.static_file, replicate.shard_files, cluster_options)
        dynamics.setdefault(replicate.case_id, (replicate.label, []))[1].append(d)
        laplace.setdefault(replicate.case_id, (replicate.label, []))[1].append(l)
        cluster_samples.setdefault(replicate.case_id, (replicate.label, []))[1].append(c.ratios)

    summaries = {case_id: _summarize(case_id, label, values)
                 for case_id, (label, values) in sorted(dynamics.items())}
    dynamics = dict(sorted(dynamics.items()))
    laplace = dict(sorted(laplace.items()))
    cluster_samples = dict(sorted(cluster_samples.items()))
    _plot_com(summaries, args.output_dir / "unwrapped_com_and_velocity.svg")
    _plot_correlation(summaries, args.output_dir / "lagged_velocity_correlation.svg")
    _plot_laplace(laplace, args.output_dir / "laplace_transform.svg")
    _plot_preferred(laplace, args.output_dir / "preferred_coordinates.svg")
    _plot_fits(dynamics, laplace, args.output_dir / "damped_cosine_fit.svg")
    _plot_clusters(cluster_samples, args.output_dir / "cluster_ratio_distributions.svg")
    print(f"[analysis] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
