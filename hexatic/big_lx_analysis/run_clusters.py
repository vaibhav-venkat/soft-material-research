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
    replicate_count: int
    area_fraction: NDArray[np.float64]


@dataclass(frozen=True)
class ClusterSummary:
    lx: float
    weighted_mean: float
    weighted_mode: float


def _case_metadata(manifest: Path) -> tuple[float, str]:
    payload: Any = json.loads(manifest.read_text())
    case = payload.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"Manifest has no case metadata: {manifest}")
    lx = case.get("lx")
    if not isinstance(lx, (int, float)) or not np.isfinite(lx) or lx <= 0.0:
        raise ValueError(f"Manifest has an invalid Lx: {manifest}")
    label = case.get("label")
    return float(lx), label if isinstance(label, str) and label else ""


def _load_cases(manifests: tuple[Path, ...], options: ClusterOptions) -> list[CaseClusters]:
    grouped: dict[str, tuple[str, float, list[NDArray[np.float64]]]] = {}
    for index, manifest in enumerate(manifests, 1):
        replicate = _load_replicate(manifest)
        lx, metadata_label = _case_metadata(manifest)
        label = metadata_label or replicate.label
        print(
            f"[clusters] {index}/{len(manifests)} case={replicate.case_id} "
            f"manifest={manifest}",
            flush=True,
        )
        result = analyze_clusters(replicate.static_file, replicate.shard_files, options)
        if replicate.case_id not in grouped:
            grouped[replicate.case_id] = (label, lx, [result.ratios])
            continue
        previous_label, previous_lx, samples = grouped[replicate.case_id]
        if not np.isclose(previous_lx, lx, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError(f"Replicates for {replicate.case_id} have different Lx values")
        if previous_label != label:
            raise ValueError(f"Replicates for {replicate.case_id} have different labels")
        samples.append(result.ratios)

    cases = [
        CaseClusters(
            case_id=case_id,
            label=label,
            lx=lx,
            replicate_count=len(samples),
            area_fraction=np.concatenate(samples) if samples else np.empty(0),
        )
        for case_id, (label, lx, samples) in grouped.items()
    ]
    return sorted(cases, key=lambda value: value.lx)


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


def _summarize(case: CaseClusters, bins: list[float]) -> ClusterSummary:
    area = case.area_fraction
    if area.size == 0:
        return ClusterSummary(case.lx, np.nan, np.nan)
    weighted_mean = float(np.average(area, weights=area))
    bin_weights, edges = np.histogram(area, bins=bins, weights=area)
    mode_bin = int(np.argmax(bin_weights))
    weighted_mode = float((edges[mode_bin] + edges[mode_bin + 1]) / 2.0)
    return ClusterSummary(case.lx, weighted_mean, weighted_mode)


def _plot_mean_mode(cases: list[CaseClusters], output: Path) -> None:
    sns = _style()
    bins = _common_bins(cases)
    summaries = [_summarize(case, bins) for case in cases]
    lx = np.asarray([summary.lx for summary in summaries])
    means = np.asarray([summary.weighted_mean for summary in summaries])
    modes = np.asarray([summary.weighted_mode for summary in summaries])
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(lx, means, marker="o", lw=2.0, label="Mean")
    axis.plot(lx, modes, marker="s", lw=2.0, label="Mode")
    axis.set(
        title="Area-weighted cluster statistics versus axial length",
        xlabel=r"Axial box length $L_x$",
        ylabel=r"Cluster area fraction $A/SA$",
    )
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
            f"weighted_mean={summary.weighted_mean:.12g} "
            f"weighted_mode={summary.weighted_mode:.12g}"
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
    existing = [path for path in (distribution_output, summary_output) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to replace {names}; pass --overwrite")
    _plot_distributions(cases, distribution_output)
    _plot_mean_mode(cases, summary_output)
    print(f"wrote {distribution_output}")
    print(f"wrote {summary_output}")


if __name__ == "__main__":
    main()
