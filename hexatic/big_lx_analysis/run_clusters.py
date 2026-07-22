"""Analyze and plot structural clusters from big-Lx safetensor outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .clusters import ClusterOptions, ClusterRatioMode, analyze_clusters
from .run_dynamics import _collect_manifests, _load_replicate, _style


@dataclass(frozen=True)
class CaseClusters:
    case_id: str
    label: str
    lx: float
    lx_multiplier: int
    circumference_diameters: float
    surface_area: float
    replicate_count: int
    area_fraction: NDArray[np.float64]

    @property
    def absolute_area(self) -> NDArray[np.float64]:
        """Absolute cluster area reconstructed from the stored A/SA ratio."""
        return self.area_fraction * self.surface_area


@dataclass(frozen=True)
class ClusterSummary:
    lx_multiplier: int
    area_weighted_mean: float
    area_weighted_mode: float
    sqrt_area_weighted_mean: float
    sqrt_area_weighted_mode: float
    absolute_area_weighted_mean: float
    absolute_area_weighted_mode: float


def _case_metadata(manifest: Path) -> tuple[float, int, float, float, str]:
    payload: Any = json.loads(manifest.read_text())
    case = payload.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"Manifest has no case metadata: {manifest}")
    lx = case.get("lx")
    if not isinstance(lx, (int, float)) or not np.isfinite(lx) or lx <= 0.0:
        raise ValueError(f"Manifest has an invalid Lx: {manifest}")
    lx_multiplier = case.get("lx_multiplier")
    if not isinstance(lx_multiplier, int) or lx_multiplier <= 0:
        raise ValueError(f"Manifest has an invalid Lx multiplier: {manifest}")
    circumference_diameters = case.get("circumference_diameters")
    if (
        not isinstance(circumference_diameters, (int, float))
        or not np.isfinite(circumference_diameters)
        or circumference_diameters <= 0.0
    ):
        raise ValueError(f"Manifest has invalid circumference diameters: {manifest}")
    circumference = case.get("circumference")
    if (
        not isinstance(circumference, (int, float))
        or not np.isfinite(circumference)
        or circumference <= 0.0
    ):
        raise ValueError(f"Manifest has an invalid circumference: {manifest}")
    label = case.get("label")
    return (
        float(lx),
        lx_multiplier,
        float(circumference_diameters),
        float(lx) * float(circumference),
        label if isinstance(label, str) and label else "",
    )


def _load_cases(manifests: tuple[Path, ...], options: ClusterOptions) -> list[CaseClusters]:
    grouped: dict[
        str,
        tuple[str, float, int, float, float, list[NDArray[np.float64]]],
    ] = {}
    for index, manifest in enumerate(manifests, 1):
        replicate = _load_replicate(manifest)
        (
            lx,
            lx_multiplier,
            circumference_diameters,
            surface_area,
            metadata_label,
        ) = _case_metadata(manifest)
        label = metadata_label or replicate.label
        print(
            f"[clusters] {index}/{len(manifests)} case={replicate.case_id} "
            f"manifest={manifest}",
            flush=True,
        )
        result = analyze_clusters(replicate.static_file, replicate.shard_files, options)
        ratios = result.ratios[np.isfinite(result.ratios) & (result.ratios > 0.0)]
        if replicate.case_id not in grouped:
            grouped[replicate.case_id] = (
                label,
                lx,
                lx_multiplier,
                circumference_diameters,
                surface_area,
                [ratios],
            )
            continue
        (
            previous_label,
            previous_lx,
            previous_multiplier,
            previous_circumference_diameters,
            previous_surface_area,
            samples,
        ) = grouped[replicate.case_id]
        if not np.isclose(previous_lx, lx, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(f"Replicates for {replicate.case_id} have different Lx values")
        if previous_label != label:
            raise ValueError(f"Replicates for {replicate.case_id} have different labels")
        if previous_multiplier != lx_multiplier or not np.isclose(
            previous_circumference_diameters,
            circumference_diameters,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Replicates for {replicate.case_id} have different case geometry"
            )
        if not np.isclose(
            previous_surface_area,
            surface_area,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Replicates for {replicate.case_id} have different surface areas"
            )
        samples.append(ratios)

    cases = [
        CaseClusters(
            case_id=case_id,
            label=label,
            lx=lx,
            lx_multiplier=lx_multiplier,
            circumference_diameters=circumference_diameters,
            surface_area=surface_area,
            replicate_count=len(samples),
            area_fraction=np.concatenate(samples) if samples else np.empty(0),
        )
        for case_id, (
            label,
            lx,
            lx_multiplier,
            circumference_diameters,
            surface_area,
            samples,
        ) in grouped.items()
    ]
    return sorted(
        cases,
        key=lambda value: (value.circumference_diameters, value.lx_multiplier),
    )


def _common_bins(cases: list[CaseClusters], transform: Any = None) -> list[float]:
    nonempty = [case.area_fraction for case in cases if case.area_fraction.size]
    if not nonempty:
        raise ValueError("No qualifying structural clusters were found")
    values = np.concatenate(nonempty)
    if transform is not None:
        values = transform(values)
    # Lists avoid Seaborn 0.13's ambiguous ndarray/string comparison with NumPy 2.
    return np.histogram_bin_edges(values, bins="auto").tolist()


def _plot_distributions(cases: list[CaseClusters], output: Path) -> None:
    sns = _style()
    colors = sns.color_palette("colorblind", n_colors=len(cases))
    area_bins = _common_bins(cases)
    sqrt_bins = _common_bins(cases, np.sqrt)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for color, case in zip(colors, cases, strict=True):
        area = case.area_fraction
        if area.size == 0:
            continue
        suffix = f" (n={case.replicate_count})" if case.replicate_count > 1 else ""
        label = case.label + suffix
        sns.histplot(
            x=area,
            weights=area,
            stat="probability",
            element="step",
            fill=False,
            common_norm=False,
            bins=area_bins,
            color=color,
            label=label,
            ax=axes[0],
        )
        sns.histplot(
            x=np.sqrt(area),
            weights=area,
            stat="probability",
            element="step",
            fill=False,
            common_norm=False,
            bins=sqrt_bins,
            color=color,
            label=label,
            ax=axes[1],
        )
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
        axis.legend(frameon=False, fontsize="small")
        axis.grid(axis="y", color="0.9", lw=0.7)
        sns.despine(ax=axis)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.big_lx_analysis.run_clusters"},
    )
    plt.close(figure)


def _weighted_mode(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    bins: list[float],
) -> float:
    bin_weights, edges = np.histogram(values, bins=bins, weights=weights)
    mode_bin = int(np.argmax(bin_weights))
    return float((edges[mode_bin] + edges[mode_bin + 1]) / 2.0)


def _summarize(
    case: CaseClusters,
    area_bins: list[float],
    sqrt_bins: list[float],
    absolute_bins: list[float],
) -> ClusterSummary:
    area = case.area_fraction
    if area.size == 0:
        return ClusterSummary(
            case.lx_multiplier,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    sqrt_area = np.sqrt(area)
    absolute_area = case.absolute_area
    return ClusterSummary(
        lx_multiplier=case.lx_multiplier,
        area_weighted_mean=float(np.average(area, weights=area)),
        area_weighted_mode=_weighted_mode(area, area, area_bins),
        sqrt_area_weighted_mean=float(np.average(sqrt_area, weights=area)),
        sqrt_area_weighted_mode=_weighted_mode(sqrt_area, area, sqrt_bins),
        absolute_area_weighted_mean=float(
            np.average(absolute_area, weights=absolute_area)
        ),
        absolute_area_weighted_mode=_weighted_mode(
            absolute_area,
            absolute_area,
            absolute_bins,
        ),
    )


def _plot_mean_mode(cases: list[CaseClusters], output: Path) -> None:
    sns = _style()
    area_bins = _common_bins(cases)
    sqrt_bins = _common_bins(cases, np.sqrt)
    absolute_values = np.concatenate(
        [case.absolute_area for case in cases if case.absolute_area.size]
    )
    absolute_bins = np.histogram_bin_edges(absolute_values, bins="auto").tolist()
    summaries = [
        _summarize(case, area_bins, sqrt_bins, absolute_bins) for case in cases
    ]
    area_means = np.asarray([summary.area_weighted_mean for summary in summaries])
    area_modes = np.asarray([summary.area_weighted_mode for summary in summaries])
    sqrt_means = np.asarray([summary.sqrt_area_weighted_mean for summary in summaries])
    sqrt_modes = np.asarray([summary.sqrt_area_weighted_mode for summary in summaries])
    absolute_means = np.asarray(
        [summary.absolute_area_weighted_mean for summary in summaries]
    )
    absolute_modes = np.asarray(
        [summary.absolute_area_weighted_mode for summary in summaries]
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)
    for axis, means, modes, title, ylabel in (
        (
            axes[0],
            area_means,
            area_modes,
            "Area-weighted cluster fractions versus axial-length multiplier",
            r"Cluster area fraction $A/SA$",
        ),
        (
            axes[1],
            sqrt_means,
            sqrt_modes,
            "Area-weighted circumference ratios versus axial-length multiplier",
            r"Cluster circumference ratio $\sqrt{A/SA}$",
        ),
        (
            axes[2],
            absolute_means,
            absolute_modes,
            "Absolute cluster area versus axial-length multiplier",
            r"Absolute cluster area $A$",
        ),
    ):
        circumferences = sorted({case.circumference_diameters for case in cases})
        colors = sns.color_palette("colorblind", n_colors=len(circumferences))
        for color, circumference in zip(colors, circumferences, strict=True):
            indices = [
                index
                for index, case in enumerate(cases)
                if np.isclose(case.circumference_diameters, circumference)
            ]
            multipliers = np.asarray([cases[index].lx_multiplier for index in indices])
            axis.plot(
                multipliers,
                means[indices],
                color=color,
                marker="o",
                lw=2.0,
                label=rf"Mean, $C={circumference:g}D$",
            )
            axis.plot(
                multipliers,
                modes[indices],
                color=color,
                marker="s",
                ls="--",
                lw=2.0,
                label=rf"Mode, $C={circumference:g}D$",
            )
        axis.set(
            title=title,
            xlabel=r"Axial-length multiplier $L_x/L_{x,1}$",
            ylabel=ylabel,
        )
        axis.set_xticks((1, 2, 4, 8, 16), ("1", "2", "4", "8", "16"))
        axis.grid(color="0.9", lw=0.7)
        axis.legend(frameon=False)
        sns.despine(ax=axis)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.big_lx_analysis.run_clusters"},
    )
    plt.close(figure)
    for case, summary in zip(cases, summaries, strict=True):
        print(
            f"case={case.case_id} Lx={case.lx:.12g} replicates={case.replicate_count} "
            f"area_weighted_mean={summary.area_weighted_mean:.12g} "
            f"area_weighted_mode={summary.area_weighted_mode:.12g} "
            f"sqrt_area_weighted_mean={summary.sqrt_area_weighted_mean:.12g} "
            f"sqrt_area_weighted_mode={summary.sqrt_area_weighted_mode:.12g} "
            f"absolute_area_weighted_mean={summary.absolute_area_weighted_mean:.12g} "
            f"absolute_area_weighted_mode={summary.absolute_area_weighted_mode:.12g}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("big_lx_cluster_plots"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int)
    parser.add_argument("--psi6-minimum", type=float, default=0.7)
    parser.add_argument("--misorientation-degrees", type=float, default=5.0)
    parser.add_argument("--neighbor-radius-diameters", type=float, default=1.7272)
    parser.add_argument("--minimum-particles", type=int, default=2)
    parser.add_argument("--particle-diameter", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifests = _collect_manifests(args.manifest, args.input_dir)
    if args.case:
        selected = set(args.case)
        manifests = tuple(
            manifest
            for manifest in manifests
            if _load_replicate(manifest).case_id in selected
        )
        missing = selected.difference(_load_replicate(path).case_id for path in manifests)
        if missing:
            raise ValueError(f"No manifests found for cases: {', '.join(sorted(missing))}")
    if not manifests:
        raise ValueError("No matching big-Lx analysis manifests were selected")

    options = ClusterOptions(
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        psi6_minimum=args.psi6_minimum,
        misorientation_degrees=args.misorientation_degrees,
        neighbor_radius_diameters=args.neighbor_radius_diameters,
        minimum_particles=args.minimum_particles,
        particle_diameter=args.particle_diameter,
        ratio_mode=ClusterRatioMode.AREA_FRACTION,
    )
    cases = _load_cases(manifests, options)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_output = args.output_dir / "cluster_ratio_distributions.svg"
    summary_output = args.output_dir / "cluster_mean_mode_vs_lx.svg"
    existing = [
        path
        for path in (
            distribution_output,
            summary_output,
        )
        if path.exists()
    ]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to replace {names}; pass --overwrite")
    _plot_distributions(cases, distribution_output)
    _plot_mean_mode(cases, summary_output)
    print(f"wrote {distribution_output}")
    print(f"wrote {summary_output}")


if __name__ == "__main__":
    main()
