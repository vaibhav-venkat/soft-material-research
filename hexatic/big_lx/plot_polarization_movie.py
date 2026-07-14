from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import Normalize
import numpy as np
from safetensors.numpy import load_file

from .cases import DEFAULT_OUTPUT_ROOT, CasePaths, get_case


def _frame_numbers(
    manifest: dict[str, object],
    start: int,
    stop: int | None,
    stride: int,
) -> list[int]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("The analysis manifest contains no frame shards")
    available_stop = max(int(shard["frame_stop"]) for shard in shards)
    requested_stop = available_stop if stop is None else min(stop, available_stop)
    if start < 0 or requested_stop <= start:
        raise ValueError(
            f"Empty frame range: start={start}, stop={requested_stop}, "
            f"available_stop={available_stop}"
        )
    return list(range(start, requested_stop, stride))


def _iter_frames(
    analysis_dir: Path,
    manifest: dict[str, object],
    selected_frames: set[int],
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    required = (
        "step",
        "coords",
        "rho",
        "polar_cylindrical",
    )
    for shard in manifest["shards"]:
        shard_start = int(shard["frame_start"])
        shard_stop = int(shard["frame_stop"])
        wanted = sorted(
            frame for frame in selected_frames if shard_start <= frame < shard_stop
        )
        if not wanted:
            continue
        shard_path = analysis_dir / str(shard["file"])
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing frame shard: {shard_path}")
        tensors = load_file(shard_path)
        missing = [name for name in required if name not in tensors]
        if missing:
            raise KeyError(f"{shard_path} is missing tensors: {', '.join(missing)}")
        for frame in wanted:
            local = frame - shard_start
            yield frame, {name: tensors[name][local] for name in required}
        del tensors


def _subsample(indices: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or indices.size <= maximum:
        return indices
    positions = np.linspace(0, indices.size - 1, maximum, dtype=np.int64)
    return indices[positions]


def _plot_values(
    frame: dict[str, np.ndarray],
    *,
    quantity: str,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(frame["coords"], dtype=np.float32)
    polar = np.asarray(frame["polar_cylindrical"], dtype=np.float32)
    rho = np.asarray(frame["rho"], dtype=np.float32)
    valid = np.all(np.isfinite(coords), axis=1) & np.all(np.isfinite(polar), axis=1)

    if quantity == "polarization":
        valid &= np.isfinite(rho) & (rho > np.finfo(np.float32).eps)
        vectors = np.divide(
            polar,
            rho[:, None],
            out=np.zeros_like(polar),
            where=rho[:, None] > np.finfo(np.float32).eps,
        )
    else:
        vectors = polar

    valid &= np.all(np.isfinite(vectors), axis=1)
    indices = np.flatnonzero(valid)
    point_ids = _subsample(indices, max_points)
    x = coords[:, 0]
    r_theta = coords[:, 2] * coords[:, 1]
    return x[point_ids], r_theta[point_ids], vectors[point_ids]


def _component_output_path(
    output: Path | None,
    output_root: Path,
    case_id: str,
    quantity: str,
    component_name: str,
) -> Path:
    if output is None:
        base = output_root / "movies" / f"{case_id}_{quantity.replace('-', '_')}"
    elif output.suffix.lower() == ".gif":
        base = output.with_suffix("")
    elif output.suffix:
        raise ValueError("--output must end in .gif or have no filename suffix")
    else:
        base = output
    return base.parent / f"{base.name}_{component_name}.gif"


def write_polarization_movies(
    case_id: str,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    output: Path | None = None,
    quantity: str = "polarization",
    start: int = 0,
    stop: int | None = None,
    stride: int = 1,
    fps: int = 20,
    dpi: int = 150,
    color_max: float | None = None,
    max_points: int = 0,
) -> tuple[Path, Path, Path]:
    if stride < 1 or fps < 1 or dpi < 1:
        raise ValueError("stride, fps, and dpi must be positive")
    if max_points < 0:
        raise ValueError("max-points cannot be negative")
    if color_max is not None and color_max <= 0.0:
        raise ValueError("color-max must be positive")
    if quantity not in ("polarization", "polar-density"):
        raise ValueError("quantity must be 'polarization' or 'polar-density'")
    case = get_case(case_id)
    paths = CasePaths(case, output_root)
    manifest_path = paths.analysis_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing analysis manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hexatic.big_lx.analysis.v1":
        raise ValueError(f"Unsupported analysis schema in {manifest_path}")

    frames = _frame_numbers(manifest, start, stop, stride)
    selected = set(frames)
    # Saved cylindrical order is (x, radial, azimuthal). Emit the requested
    # display order Px, Ptheta, Pr while retaining the correct array indices.
    components = (
        ("px", 0, r"P_x"),
        ("ptheta", 2, r"P_\theta"),
        ("pr", 1, r"P_r"),
    )
    outputs: list[Path] = []
    for component_name, component_index, component_label in components:
        output_path = _component_output_path(
            output, output_root, case_id, quantity, component_name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)
        limit = color_max or (1.0 if quantity == "polarization" else None)
        norm = Normalize(vmin=-limit if limit is not None else None, vmax=limit)
        fig, axis = plt.subplots(figsize=(16, 6))
        rendered = 0
        try:
            with writer.saving(fig, str(output_path), dpi=dpi):
                for frame_index, frame in _iter_frames(
                    paths.analysis_dir, manifest, selected
                ):
                    px, py, vectors = _plot_values(
                        frame,
                        quantity=quantity,
                        max_points=max_points,
                    )
                    if not px.size:
                        raise ValueError(
                            f"No plottable particles in frame {frame_index}"
                        )
                    colors = vectors[:, component_index]
                    if norm.vmax is None:
                        finite = np.abs(colors[np.isfinite(colors)])
                        limit = (
                            float(np.percentile(finite, 99.5))
                            if finite.size
                            else 1.0
                        )
                        if limit <= 0.0:
                            limit = 1.0
                        norm.vmin = -limit
                        norm.vmax = limit

                    axis.clear()
                    scatter = axis.scatter(
                        px,
                        py,
                        c=colors,
                        s=3.0,
                        cmap="coolwarm",
                        norm=norm,
                        linewidths=0,
                    )
                    axis.set_xlim(-0.5 * case.lx, 0.5 * case.lx)
                    axis.set_ylim(0.0, case.circumference)
                    axis.set_xlabel("x")
                    axis.set_ylabel(r"$r\theta$")
                    axis.set_title(
                        f"{case.label}: ${component_label}$ "
                        f"(frame {frame_index}, step {int(frame['step'])})"
                    )
                    axis.grid(alpha=0.15)
                    if rendered == 0:
                        colorbar = fig.colorbar(scatter, ax=axis, pad=0.01)
                        suffix = r"/\rho" if quantity == "polarization" else ""
                        colorbar.set_label(f"${component_label}{suffix}$")
                    fig.tight_layout()
                    writer.grab_frame()
                    rendered += 1
                    print(
                        f"[big_lx.movie] {component_name}: frame {frame_index} "
                        f"({rendered}/{len(frames)})",
                        flush=True,
                    )
        finally:
            plt.close(fig)
        if rendered != len(frames):
            raise RuntimeError(
                f"Rendered {rendered} of {len(frames)} requested frames"
            )
        outputs.append(output_path)
        print(f"[big_lx.movie] wrote {output_path}", flush=True)
    return outputs[0], outputs[1], outputs[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Px, Ptheta, and Pr GIFs on x versus r*theta from a "
            "big-Lx safetensors polarization field."
        )
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output filename prefix; a trailing .gif is removed before component names.",
    )
    parser.add_argument(
        "--quantity",
        choices=("polarization", "polar-density"),
        default="polarization",
        help="Plot P/rho (default) or the stored polar density P.",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--color-max", type=float)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum colored points per frame; 0 keeps all points.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_polarization_movies(
        args.case,
        output_root=args.output_root,
        output=args.output,
        quantity=args.quantity,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
        fps=args.fps,
        dpi=args.dpi,
        color_max=args.color_max,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
