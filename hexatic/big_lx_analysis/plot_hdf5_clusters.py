"""Plot size-weighted cluster statistics from simulation-analysis HDF5 output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns


STATIC_SCHEMA = "simulation_analysis.static.v1"
FRAME_SCHEMA = "simulation_analysis.frames.v3"


@dataclass(frozen=True)
class ClusterSamples:
    """Normalized cluster-area samples for one simulation-analysis directory."""

    source: Path
    lx: float
    area_fraction: NDArray[np.float64]


@dataclass(frozen=True)
class ClusterSummary:
    lx: float
    weighted_mean: float
    weighted_mode: float


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _analysis_directories(inputs: list[Path]) -> tuple[Path, ...]:
    directories: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.name == "static.h5" or path.name.startswith("frames_"):
                directories.add(path.parent)
                continue
            raise ValueError(f"Expected an HDF5 analysis file, got {path}")
        if (path / "static.h5").is_file():
            directories.add(path)
        else:
            directories.update(candidate.parent for candidate in path.rglob("static.h5"))
    if not directories:
        raise ValueError("No directories containing static.h5 were found")
    return tuple(sorted(directories))


def _read_static(path: Path) -> tuple[float, float, int]:
    with h5py.File(path, "r") as handle:
        schema = _attribute_text(handle.attrs.get("schema", ""))
        if schema != STATIC_SCHEMA:
            raise ValueError(f"Unsupported static schema {schema!r} in {path}")
        radius = float(handle.attrs["cylinder_radius"])
        particle_diameter = float(handle.attrs["particle_diameter"])
        particle_count = int(handle.attrs["particle_count"])
    if radius <= 0.0 or particle_diameter <= 0.0 or particle_count <= 0:
        raise ValueError(f"Invalid geometry metadata in {path}")
    return radius, particle_diameter, particle_count


def load_cluster_samples(directory: Path) -> ClusterSamples:
    """Load raw cluster counts and convert them to A/SA samples."""
    radius, particle_diameter, particle_count = _read_static(directory / "static.h5")
    shard_paths = tuple(sorted(directory.glob("frames_*.h5")))
    if not shard_paths:
        raise ValueError(f"No frames_*.h5 shards found in {directory}")

    cluster_chunks: list[NDArray[np.float64]] = []
    lx_values: list[NDArray[np.float64]] = []
    for path in shard_paths:
        with h5py.File(path, "r") as handle:
            schema = _attribute_text(handle.attrs.get("schema", ""))
            if schema != FRAME_SCHEMA:
                raise ValueError(f"Unsupported frame schema {schema!r} in {path}")
            if "cluster_sizes" not in handle or "box" not in handle:
                raise ValueError(f"Required cluster_sizes or box dataset is missing from {path}")
            sizes = np.asarray(handle["cluster_sizes"], dtype=np.float64)
            boxes = np.asarray(handle["box"], dtype=np.float64)
        if sizes.ndim != 1 or np.any(sizes <= 0.0) or np.any(sizes > particle_count):
            raise ValueError(f"Invalid cluster_sizes dataset in {path}")
        if boxes.ndim != 2 or boxes.shape[1] != 6 or boxes.shape[0] == 0:
            raise ValueError(f"Invalid box dataset in {path}; expected shape (frames, 6)")
        cluster_chunks.append(sizes)
        lx_values.append(boxes[:, 0])

    all_lx = np.concatenate(lx_values)
    lx = float(all_lx[0])
    if lx <= 0.0 or not np.allclose(all_lx, lx, rtol=1.0e-7, atol=1.0e-10):
        raise ValueError(f"Lx is non-positive or changes between frames in {directory}")
    cluster_sizes = np.concatenate(cluster_chunks)
    surface_area = 2.0 * np.pi * radius * lx
    particle_area = np.pi * particle_diameter**2 / 4.0
    area_fraction = cluster_sizes * particle_area / surface_area
    return ClusterSamples(directory, lx, area_fraction)


def _bin_edges(samples: list[ClusterSamples], bins: str | int) -> NDArray[np.float64]:
    values = np.concatenate([sample.area_fraction for sample in samples])
    return np.asarray(np.histogram_bin_edges(values, bins=bins), dtype=np.float64)


def _pool_equal_lx(samples: list[ClusterSamples]) -> list[ClusterSamples]:
    """Pool replicates whose float32 box lengths describe the same Lx."""
    pooled: list[ClusterSamples] = []
    for sample in sorted(samples, key=lambda value: value.lx):
        if pooled and np.isclose(sample.lx, pooled[-1].lx, rtol=1.0e-7, atol=1.0e-10):
            previous = pooled[-1]
            pooled[-1] = ClusterSamples(
                previous.source,
                previous.lx,
                np.concatenate((previous.area_fraction, sample.area_fraction)),
            )
        else:
            pooled.append(sample)
    return pooled


def summarize(
    sample: ClusterSamples,
    bin_edges: NDArray[np.float64],
) -> ClusterSummary:
    """Calculate moments of the size-weighted empirical distribution."""
    values = sample.area_fraction
    if values.size == 0:
        return ClusterSummary(sample.lx, np.nan, np.nan)
    weighted_mean = float(np.average(values, weights=values))
    bin_weights, _ = np.histogram(values, bins=bin_edges, weights=values)
    mode_bin = int(np.argmax(bin_weights))
    weighted_mode = float((bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2.0)
    return ClusterSummary(sample.lx, weighted_mean, weighted_mode)


def _configure_style() -> None:
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)


def plot_distributions(
    samples: list[ClusterSamples],
    bin_edges: NDArray[np.float64],
    output: Path,
) -> None:
    _configure_style()
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    colors = sns.color_palette("colorblind", n_colors=len(samples))
    for color, sample in zip(colors, sorted(samples, key=lambda value: value.lx), strict=True):
        if sample.area_fraction.size == 0:
            continue
        sns.histplot(
            x=sample.area_fraction,
            weights=sample.area_fraction,
            bins=bin_edges.tolist(),
            stat="probability",
            element="step",
            fill=False,
            common_norm=False,
            color=color,
            label=rf"$L_x={sample.lx:g}$",
            ax=axis,
        )
    axis.set(
        title="Size-weighted structural-cluster distribution",
        xlabel=r"Cluster area fraction $A/SA$",
        ylabel="Size-weighted probability",
    )
    axis.grid(axis="y", color="0.9", lw=0.7)
    axis.legend(frameon=False)
    sns.despine(ax=axis)
    figure.savefig(output, format=output.suffix.lstrip("."), bbox_inches="tight")
    plt.close(figure)


def plot_summary(summaries: list[ClusterSummary], output: Path) -> None:
    _configure_style()
    ordered = sorted(summaries, key=lambda value: value.lx)
    lx = np.asarray([value.lx for value in ordered])
    means = np.asarray([value.weighted_mean for value in ordered])
    modes = np.asarray([value.weighted_mode for value in ordered])
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.plot(lx, means, marker="o", lw=2.0, label="Mean")
    axis.plot(lx, modes, marker="s", lw=2.0, label="Mode")
    axis.set(
        title="Size-weighted cluster statistics versus axial length",
        xlabel=r"Axial box length $L_x$",
        ylabel=r"Cluster area fraction $A/SA$",
    )
    axis.grid(color="0.9", lw=0.7)
    axis.legend(frameon=False)
    sns.despine(ax=axis)
    figure.savefig(output, format=output.suffix.lstrip("."), bbox_inches="tight")
    plt.close(figure)


def _parse_bins(value: str) -> str | int:
    if value in {"auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"}:
        return value
    try:
        bins = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("bins must be a positive integer or NumPy bin rule") from error
    if bins <= 0:
        raise argparse.ArgumentTypeError("bins must be positive")
    return bins


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Analysis directories, HDF5 files, or roots to search recursively.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("cluster_analysis_plots"))
    parser.add_argument("--bins", type=_parse_bins, default="auto")
    parser.add_argument("--format", choices=("svg", "png", "pdf"), default="svg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    directories = _analysis_directories(args.input)
    samples = _pool_equal_lx([load_cluster_samples(directory) for directory in directories])
    nonempty = [sample for sample in samples if sample.area_fraction.size]
    if not nonempty:
        raise ValueError("No qualifying cluster samples were found")

    bin_edges = _bin_edges(nonempty, args.bins)
    summaries = [summarize(sample, bin_edges) for sample in nonempty]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution_output = args.output_dir / f"cluster_size_weighted_distributions.{args.format}"
    summary_output = args.output_dir / f"cluster_mean_mode_vs_lx.{args.format}"
    existing = [path for path in (distribution_output, summary_output) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to replace {names}; pass --overwrite")

    plot_distributions(nonempty, bin_edges, distribution_output)
    plot_summary(summaries, summary_output)
    for summary in sorted(summaries, key=lambda value: value.lx):
        print(
            f"Lx={summary.lx:.12g} weighted_mean={summary.weighted_mean:.12g} "
            f"weighted_mode={summary.weighted_mode:.12g}"
        )
    print(f"wrote {distribution_output}")
    print(f"wrote {summary_output}")


if __name__ == "__main__":
    main()
