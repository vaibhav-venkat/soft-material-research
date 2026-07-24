
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from safetensors import safe_open

from hexatic.constants import cylinder

from .run_dynamics import _collect_manifests, _load_replicate, _style


def _load_static_lx(static_file: Path) -> float:
    """Read the axial box length from a static.safetensors file."""
    with safe_open(static_file, framework="np") as f:
        return float(f.get_tensor("lx").item())


def _recompute_shell_mask_from_coords(
    coords: NDArray[np.float32],
    cylinder_radius: float,
    shell_delta: float,
) -> NDArray[np.bool_]:
    """Compute per-particle shell mask from cylindrical coordinates.

    A particle is on the shell when its radial coordinate exceeds
    ``cylinder_radius - shell_delta``.
    """
    r = coords[..., 2]  # last component is radial distance
    return r > (cylinder_radius - shell_delta)


def _load_shell_density(
    shard_files: tuple[Path, ...],
    *,
    timestep: float,
    particle_diameter: float,
    lx: float,
    cylinder_radius: float | None,
    shell_delta: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load shell mask from frame shards and compute surface-area density.

    Parameters
    ----------
    shard_files:
        Ordered safetensor shards for one replicate.
    timestep:
        Simulation timestep (time units per step).
    particle_diameter:
        Diameter of a single particle.
    lx:
        Axial box length from static.safetensors.
    cylinder_radius:
        Cylinder radius from static.safetensors; required for the fallback
        recomputation when *hexatic_shell_mask* is absent.
    shell_delta:
        Radial threshold for shell membership (``cylinder.SHELL_DELTA``).

    Returns
    -------
    elapsed_time: (T,) float64
        Simulation time of each frame (relative to the first frame).
    density: (T,) float64
        Shell surface-area density ``N_shell * pi(D/2)^2 / L_x``.
    """
    particle_area = np.pi * (particle_diameter / 2.0) ** 2

    loaded_steps: list[NDArray[np.int64]] = []
    shell_counts: list[NDArray[np.int64]] = []

    for path in shard_files:
        with safe_open(path, framework="np") as f:
            keys = f.keys()
            step = f.get_tensor("step")  # (F,) int64
            loaded_steps.append(step)

            if "hexatic_shell_mask" in keys:
                mask = f.get_tensor("hexatic_shell_mask")  # (F, P) bool
                shell_counts.append(np.asarray(mask.sum(axis=1), dtype=np.int64))
            else:
                # Fallback: recompute from cylindrical coordinates.
                if cylinder_radius is None:
                    raise KeyError(
                        "hexatic_shell_mask not found and cylinder_radius was "
                        "not provided; cannot recompute shell mask"
                    )
                coords = f.get_tensor("coords")  # (F, P, 3) float32
                mask = _recompute_shell_mask_from_coords(
                    coords, cylinder_radius, shell_delta
                )
                shell_counts.append(np.asarray(mask.sum(axis=1), dtype=np.int64))

    steps_all = np.concatenate(loaded_steps, axis=0)   # (T,)
    n_shell = np.concatenate(shell_counts, axis=0).astype(np.float64)  # (T,)
    density = n_shell * particle_area / lx

    initial_step = int(steps_all[0])
    elapsed = np.asarray(steps_all - initial_step, dtype=np.float64) * timestep
    return elapsed, density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Path to a manifest.json file (repeatable)",
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        type=Path,
        default=[],
        help="Root directory containing a big-Lx analysis output (repeatable)",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID to retain (repeatable, default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("surface_area_plots"),
        help="Output directory for the SVG plot (default: surface_area_plots)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files without error",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=float(cylinder.SIMULATION.timestep),
        help="Simulation timestep (default from cylinder constants)",
    )
    parser.add_argument(
        "--particle-diameter",
        type=float,
        default=float(cylinder.PARTICLE_DIAMETER),
        help="Particle diameter (default from cylinder constants)",
    )
    parser.add_argument(
        "--shell-delta",
        type=float,
        default=float(cylinder.SHELL_DELTA),
        help=(
            "Radial threshold from the outer surface for shell membership; "
            "used only when hexatic_shell_mask is absent from the safetensors "
            "(default: cylinder.SHELL_DELTA)"
        ),
    )
    args = parser.parse_args()

    manifests = _collect_manifests(args.manifest, args.input_dir)
    replicates = [_load_replicate(path) for path in manifests]
    if args.case:
        selected = set(args.case)
        replicates = [r for r in replicates if r.case_id in selected]
        missing = selected - {r.case_id for r in replicates}
        if missing:
            raise ValueError(f"Requested cases not found: {sorted(missing)}")
    if not replicates:
        raise ValueError("No matching replicates found")

    # Build a long-form DataFrame for seaborn.
    rows: list[dict[str, float | str]] = []
    for r in replicates:
        lx = _load_static_lx(r.static_file)
        cylinder_radius: float | None = None
        with safe_open(r.static_file, framework="np") as f:
            if "radius" in f.keys():
                cylinder_radius = float(f.get_tensor("radius").item())

        elapsed, density = _load_shell_density(
            r.shard_files,
            timestep=args.timestep,
            particle_diameter=args.particle_diameter,
            lx=lx,
            cylinder_radius=cylinder_radius,
            shell_delta=args.shell_delta,
        )
        for t, d in zip(elapsed, density):
            rows.append({
                "elapsed_time": t,
                "density": d,
                "case_label": r.label,
            })

    df = pd.DataFrame(rows)

    sns = _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    sns.lineplot(
        data=df,
        x="elapsed_time",
        y="density",
        hue="case_label",
        errorbar="sd",
        ax=ax,
    )
    ax.set(
        title="Shell surface-area density",
        xlabel="Simulation time",
        ylabel=r"$N_{\mathrm{shell}}\,\pi\,(D/2)^2\,/\,L_x$",
    )
    ax.grid(axis="y", color="0.9", lw=0.7)
    sns.move_legend(ax, frameon=False, title=None)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "shell_surface_area_density.svg"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists; pass --overwrite to replace"
        )
    fig.savefig(
        output_path,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.big_lx_analysis.surface_area"},
    )
    plt.close(fig)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
