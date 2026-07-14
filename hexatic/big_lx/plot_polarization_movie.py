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


def _particle_vectors(
    frame: dict[str, np.ndarray],
    *,
    quantity: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(frame["coords"], dtype=np.float32)
    polar = np.asarray(frame["polar_cylindrical"], dtype=np.float32)
    rho = np.asarray(frame["rho"], dtype=np.float32)
    valid = (
        np.all(np.isfinite(coords), axis=1)
        & np.all(np.isfinite(polar), axis=1)
        & np.isfinite(rho)
    )
    if quantity == "polarization":
        valid &= rho > np.finfo(np.float32).eps
        vectors = np.divide(
            polar[valid],
            rho[valid, None],
            out=np.zeros_like(polar[valid]),
            where=rho[valid, None] > np.finfo(np.float32).eps,
        )
    else:
        vectors = polar[valid]
    return coords[valid, 0], np.mod(coords[valid, 1], 2.0 * np.pi), vectors


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
    figure_width: float = 20.0,
    figure_height: float = 10.0,
) -> tuple[Path, Path]:
    if stride < 1 or fps < 1 or dpi < 1:
        raise ValueError("stride, fps, and dpi must be positive")
    if figure_width <= 0.0 or figure_height <= 0.0:
        raise ValueError("figure-width and figure-height must be positive")
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

    # Saved cylindrical order is (x, radial, azimuthal), so Ptheta is index 2.
    movies = (
        ("px_ptheta", r"$(P_x,P_\theta)$", "in_film"),
        ("px", r"$P_x$", "x_only"),
    )
    outputs: list[Path] = []
    for component_name, component_label, vector_mode in movies:
        output_path = _component_output_path(
            output, output_root, case_id, quantity, component_name
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)
        limit = color_max or (1.0 if quantity == "polarization" else None)
        norm = Normalize(vmin=0.0, vmax=limit)
        fig, axis = plt.subplots(figsize=(figure_width, figure_height))
        rendered = 0
        try:
            with writer.saving(fig, str(output_path), dpi=dpi):
                for frame_index, frame in _iter_frames(
                    paths.analysis_dir, manifest, selected
                ):
                    particle_x, particle_theta, vectors = _particle_vectors(
                        frame,
                        quantity=quantity,
                    )
                    if not particle_x.size:
                        raise ValueError(
                            f"No plottable particles in frame {frame_index}"
                        )
                    px_values = vectors[:, 0]
                    if vector_mode == "in_film":
                        py_values = vectors[:, 2]
                        colors = np.hypot(px_values, py_values)
                    else:
                        py_values = np.zeros_like(px_values)
                        colors = np.abs(px_values)
                    if norm.vmax is None:
                        finite = colors[np.isfinite(colors)]
                        limit = (
                            float(np.percentile(finite, 99.5))
                            if finite.size
                            else 1.0
                        )
                        if limit <= 0.0:
                            limit = 1.0
                        norm.vmax = limit

                    axis.clear()
                    sampled_norm = np.hypot(px_values, py_values)
                    arrow_valid = sampled_norm > np.finfo(np.float32).eps
                    arrow_u = np.divide(
                        px_values,
                        sampled_norm,
                        out=np.zeros_like(px_values),
                        where=arrow_valid,
                    )
                    arrow_v = np.divide(
                        py_values,
                        sampled_norm,
                        out=np.zeros_like(py_values),
                        where=arrow_valid,
                    )
                    quiver = axis.quiver(
                        particle_x[arrow_valid],
                        case.radius * particle_theta[arrow_valid],
                        arrow_u[arrow_valid],
                        arrow_v[arrow_valid],
                        colors[arrow_valid],
                        cmap="viridis",
                        norm=norm,
                        alpha=0.95,
                        angles="uv",
                        scale_units="inches",
                        scale=6.5,
                        width=0.0012,
                        headwidth=4.0,
                        headlength=5.0,
                        headaxislength=4.5,
                        pivot="middle",
                    )
                    axis.set_xlim(-0.5 * case.lx, 0.5 * case.lx)
                    axis.set_ylim(0.0, 2.0 * np.pi * case.radius)
                    axis.set_xlabel("x")
                    axis.set_ylabel(r"$R_{\mathrm{case}}\theta$")
                    axis.set_title(
                        f"{case.label}: {component_label} magnitude and direction "
                        f"(frame {frame_index}, step {int(frame['step'])})"
                    )
                    if rendered == 0:
                        colorbar = fig.colorbar(quiver, ax=axis, pad=0.01)
                        suffix = "/rho" if quantity == "polarization" else ""
                        if vector_mode == "in_film":
                            colorbar.set_label(f"sqrt(Px^2 + Ptheta^2) {suffix}")
                        else:
                            colorbar.set_label(f"abs(Px) {suffix}")
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
    return outputs[0], outputs[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render per-particle (Px, Ptheta) and Px direction GIFs on the unwrapped "
            "film coordinate x versus R_case*theta."
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
    parser.add_argument("--figure-width", type=float, default=20.0)
    parser.add_argument("--figure-height", type=float, default=10.0)
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
        figure_width=args.figure_width,
        figure_height=args.figure_height,
    )


if __name__ == "__main__":
    main()
