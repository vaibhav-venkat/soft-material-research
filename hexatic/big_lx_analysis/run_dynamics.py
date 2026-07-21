"""Run native big-Lx dynamics analysis and plot case/replicate summaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from hexatic.constants import cylinder

from .dynamics import DynamicsOptions, DynamicsResult, analyze_dynamics


@dataclass(frozen=True)
class Replicate:
    case_id: str
    label: str
    manifest: Path
    static_file: Path
    shard_files: tuple[Path, ...]


@dataclass(frozen=True)
class CaseSummary:
    label: str
    replicate_count: int
    elapsed_time: NDArray[np.float64]
    center_mean: NDArray[np.float64]
    center_std: NDArray[np.float64]
    velocity_mean: NDArray[np.float64]
    velocity_std: NDArray[np.float64]
    lag_time: NDArray[np.float64]
    pearson_mean: NDArray[np.float64]
    pearson_std: NDArray[np.float64]


def _load_replicate(path: Path) -> Replicate:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "hexatic.big_lx.analysis.v1":
        raise ValueError(f"Unsupported manifest schema: {path}")
    if payload.get("complete") is not True:
        raise ValueError(f"Analysis manifest is incomplete: {path}")
    case = payload.get("case")
    if not isinstance(case, dict):
        raise ValueError(f"Manifest has no case metadata: {path}")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"Manifest has an invalid case ID: {path}")
    label = case.get("label")
    if not isinstance(label, str) or not label:
        label = case_id

    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"Manifest has no frame shards: {path}")
    expected_start = 0
    shard_files: list[Path] = []
    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid shard entry in {path}")
        filename = entry.get("file")
        start = entry.get("frame_start")
        stop = entry.get("frame_stop")
        if not isinstance(filename, str) or start != expected_start:
            raise ValueError(f"Non-contiguous shard entry in {path}")
        if not isinstance(stop, int) or stop <= expected_start:
            raise ValueError(f"Invalid shard range in {path}")
        shard_file = path.parent / filename
        if not shard_file.is_file():
            raise FileNotFoundError(shard_file)
        shard_files.append(shard_file)
        expected_start = stop

    static_file = path.parent / "static.safetensors"
    if not static_file.is_file():
        raise FileNotFoundError(static_file)
    return Replicate(
        case_id=case_id,
        label=label,
        manifest=path,
        static_file=static_file,
        shard_files=tuple(shard_files),
    )


def _collect_manifests(explicit: list[Path], roots: list[Path]) -> tuple[Path, ...]:
    candidates = list(explicit)
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
        for path in root.rglob("manifest.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("schema") == "hexatic.big_lx.analysis.v1":
                candidates.append(path)
    unique = {path.resolve(): path.resolve() for path in candidates}
    if not unique:
        raise ValueError("No big-Lx analysis manifests were selected")
    return tuple(sorted(unique.values()))


def _require_common_grid(
    reference: NDArray[np.float64],
    candidate: NDArray[np.float64],
    case_id: str,
    name: str,
) -> None:
    if reference.shape != candidate.shape or not np.allclose(reference, candidate, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Replicates for {case_id} have different {name} grids")


def _summarize(case_id: str, label: str, results: list[DynamicsResult]) -> CaseSummary:
    reference = results[0]
    for value in results[1:]:
        _require_common_grid(reference.elapsed_time, value.elapsed_time, case_id, "time")
        _require_common_grid(reference.lag_time, value.lag_time, case_id, "lag-time")
    centers = np.stack([value.center for value in results])
    velocities = np.stack([value.velocity for value in results])
    pearson = np.stack([value.pearson for value in results])
    ddof = 1 if len(results) > 1 else 0
    return CaseSummary(
        label=label,
        replicate_count=len(results),
        elapsed_time=reference.elapsed_time,
        center_mean=centers.mean(axis=0),
        center_std=centers.std(axis=0, ddof=ddof),
        velocity_mean=velocities.mean(axis=0),
        velocity_std=velocities.std(axis=0, ddof=ddof),
        lag_time=reference.lag_time,
        pearson_mean=pearson.mean(axis=0),
        pearson_std=pearson.std(axis=0, ddof=ddof),
    )


def _style() -> Any:
    sns = import_module("seaborn")
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    return sns


def _plot_com(summaries: dict[str, CaseSummary], output: Path) -> None:
    sns = _style()
    colors = sns.color_palette("colorblind", n_colors=len(summaries))
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, constrained_layout=True)
    for color, summary in zip(colors, summaries.values(), strict=True):
        suffix = f" (n={summary.replicate_count})" if summary.replicate_count > 1 else ""
        plot_label = summary.label + suffix
        axes[0].plot(summary.elapsed_time, summary.center_mean, color=color, lw=2.0, label=plot_label)
        axes[1].plot(summary.elapsed_time, summary.velocity_mean, color=color, lw=2.0, label=plot_label)
        if summary.replicate_count > 1:
            axes[0].fill_between(
                summary.elapsed_time,
                summary.center_mean - summary.center_std,
                summary.center_mean + summary.center_std,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
            axes[1].fill_between(
                summary.elapsed_time,
                summary.velocity_mean - summary.velocity_std,
                summary.velocity_mean + summary.velocity_std,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    axes[0].set_title("Unwrapped axial center of mass")
    axes[0].set_ylabel(r"$x_{\mathrm{COM}}$")
    axes[1].set_title("Axial center-of-mass velocity")
    axes[1].set_ylabel(r"$v_{\mathrm{COM}}$")
    axes[1].set_xlabel("Elapsed simulation time")
    for axis in axes:
        axis.axhline(0.0, color="0.75", lw=0.8, zorder=0)
        axis.grid(axis="y", color="0.9", lw=0.7)
        axis.legend(frameon=False, ncol=2)
        sns.despine(ax=axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight", metadata={"Creator": "hexatic.big_lx_analysis"})
    plt.close(figure)


def _plot_correlation(summaries: dict[str, CaseSummary], output: Path) -> None:
    sns = _style()
    colors = sns.color_palette("colorblind", n_colors=len(summaries))
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for color, summary in zip(colors, summaries.values(), strict=True):
        suffix = f" (n={summary.replicate_count})" if summary.replicate_count > 1 else ""
        axis.plot(summary.lag_time, summary.pearson_mean, color=color, lw=2.0, label=summary.label + suffix)
        if summary.replicate_count > 1:
            axis.fill_between(
                summary.lag_time,
                summary.pearson_mean - summary.pearson_std,
                summary.pearson_mean + summary.pearson_std,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    axis.axhline(0.0, color="0.65", lw=0.9)
    axis.set_title("Lagged axial COM-velocity Pearson correlation")
    axis.set_xlabel("Lag time")
    axis.set_ylabel("Pearson correlation")
    axis.set_ylim(-1.05, 1.05)
    axis.grid(axis="y", color="0.9", lw=0.7)
    axis.legend(frameon=False, ncol=2)
    sns.despine(ax=axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", bbox_inches="tight", metadata={"Creator": "hexatic.big_lx_analysis"})
    plt.close(figure)


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
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--case", action="append", default=[], help="Case ID to retain; repeat as needed")
    parser.add_argument("--output-dir", type=Path, default=Path("dynamics_analysis_output"))
    parser.add_argument(
        "--timestep",
        type=_positive_float,
        default=float(cylinder.SIMULATION.timestep),
    )
    parser.add_argument("--frame-start", type=_nonnegative_int, default=0)
    parser.add_argument("--frame-stop", type=_nonnegative_int)
    parser.add_argument("--max-lag", type=_nonnegative_int)
    parser.add_argument("--device-ordinal", type=_nonnegative_int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frame_stop is not None and args.frame_stop <= args.frame_start:
        raise ValueError("--frame-stop must be greater than --frame-start")
    manifests = _collect_manifests(args.manifest, args.input_dir)
    replicates = [_load_replicate(path) for path in manifests]
    if args.case:
        selected = set(args.case)
        replicates = [replicate for replicate in replicates if replicate.case_id in selected]
        missing = selected.difference(replicate.case_id for replicate in replicates)
        if missing:
            raise ValueError(f"Requested cases were not found: {sorted(missing)}")

    options = DynamicsOptions(
        frame_start=args.frame_start,
        frame_stop=args.frame_stop,
        timestep=args.timestep,
        max_lag=args.max_lag,
        device_ordinal=args.device_ordinal,
    )
    grouped: dict[str, tuple[str, list[DynamicsResult]]] = {}
    for index, replicate in enumerate(replicates, start=1):
        print(
            f"[dynamics] {index}/{len(replicates)} case={replicate.case_id} "
            f"manifest={replicate.manifest}",
            flush=True,
        )
        result = analyze_dynamics(replicate.static_file, replicate.shard_files, options)
        if replicate.case_id not in grouped:
            grouped[replicate.case_id] = (replicate.label, [])
        grouped[replicate.case_id][1].append(result)
    summaries = {
        case_id: _summarize(case_id, label, results)
        for case_id, (label, results) in sorted(grouped.items())
    }
    _plot_com(summaries, args.output_dir / "unwrapped_com_and_velocity.svg")
    _plot_correlation(summaries, args.output_dir / "lagged_velocity_correlation.svg")
    print(f"[dynamics] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
