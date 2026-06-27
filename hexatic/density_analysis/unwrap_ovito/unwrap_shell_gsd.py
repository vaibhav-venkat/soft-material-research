from __future__ import annotations

import argparse
from pathlib import Path

import gsd.hoomd
import numpy as np


DENSITY_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_GSD = (
    DENSITY_ANALYSIS_DIR / "hexatic_output" / "radius_15D_hexatic_velocity.gsd"
)
DEFAULT_ACTIVE_FIELDS_NPZ = (
    DENSITY_ANALYSIS_DIR / "npz" / "radius_15D_active_matter_fields.npz"
)
DEFAULT_OUTPUT_GSD = (
    DENSITY_ANALYSIS_DIR
    / "output"
    / "unwrap_ovito"
    / "radius_15D_shell_xtheta_velocity.gsd"
)


def write_shell_xtheta_gsd(
    input_gsd: str | Path = DEFAULT_INPUT_GSD,
    output_gsd: str | Path = DEFAULT_OUTPUT_GSD,
    active_fields_npz: str | Path = DEFAULT_ACTIVE_FIELDS_NPZ,
    z_value: float = 0.0,
) -> None:
    input_path = Path(input_gsd)
    output_path = Path(output_gsd)
    active_path = Path(active_fields_npz)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_gsd and output_gsd must be different files.")

    coords, shell_mask, steps, x_edges, theta_edges = _load_cached_shell_geometry(
        active_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(input_path), mode="r") as source:
        _validate_cached_geometry(source, coords, shell_mask, steps, active_path)
        with gsd.hoomd.open(name=str(output_path), mode="w") as destination:
            for frame_idx, frame in enumerate(source):
                mask = shell_mask[frame_idx]
                new_frame = gsd.hoomd.Frame()
                new_frame.configuration.step = frame.configuration.step
                new_frame.configuration.box = _xtheta_box(x_edges, theta_edges)

                shell_coords = coords[frame_idx, mask]
                positions = np.zeros((shell_coords.shape[0], 3), dtype=np.float32)
                positions[:, 0] = shell_coords[:, 0].astype(np.float32)
                positions[:, 1] = 3 * shell_coords[:, 1].astype(np.float32)
                positions[:, 2] = np.float32(z_value)

                new_frame.particles.N = positions.shape[0]
                new_frame.particles.position = positions
                _copy_particle_metadata(frame.particles, new_frame.particles, mask)
                destination.append(new_frame)


def _load_cached_shell_geometry(
    active_fields_npz: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(active_fields_npz, allow_pickle=False) as data:
        required = ("coords", "shell_mask", "steps", "x_edges", "theta_edges")
        missing = tuple(name for name in required if name not in data)
        if missing:
            raise KeyError(
                f"{active_fields_npz} is missing required arrays: {', '.join(missing)}"
            )
        coords = np.asarray(data["coords"], dtype=np.float64)
        shell_mask = np.asarray(data["shell_mask"], dtype=bool)
        steps = np.asarray(data["steps"], dtype=np.int64)
        x_edges = np.asarray(data["x_edges"], dtype=np.float64)
        theta_edges = np.asarray(data["theta_edges"], dtype=np.float64)

    if coords.ndim != 3 or coords.shape[2] < 2:
        raise ValueError("coords must have shape (frame, particle, >=2).")
    if shell_mask.shape != coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes.")
    return coords, shell_mask, steps, x_edges, theta_edges


def _validate_cached_geometry(
    source,
    coords: np.ndarray,
    shell_mask: np.ndarray,
    steps: np.ndarray,
    active_fields_npz: Path,
) -> None:
    if len(source) != coords.shape[0]:
        raise ValueError(
            f"{active_fields_npz} has {coords.shape[0]} frames, but GSD has "
            f"{len(source)} frames."
        )
    for frame_idx, frame in enumerate(source):
        if int(frame.particles.N) != coords.shape[1]:
            raise ValueError(
                f"frame {frame_idx} has {frame.particles.N} particles, but cached "
                f"coords have {coords.shape[1]}."
            )
        if steps.shape[0] > frame_idx and int(steps[frame_idx]) != frame.configuration.step:
            raise ValueError(
                f"frame {frame_idx} step mismatch: cached step {steps[frame_idx]}, "
                f"GSD step {frame.configuration.step}."
            )
    if shell_mask.shape != coords.shape[:2]:
        raise ValueError("shell_mask must match coords frame/particle axes.")


def _xtheta_box(x_edges: np.ndarray, theta_edges: np.ndarray) -> np.ndarray:
    lx = float(x_edges[-1] - x_edges[0])
    ltheta = float(theta_edges[-1] - theta_edges[0])
    return np.asarray([lx, ltheta, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _copy_particle_metadata(source, destination, mask: np.ndarray) -> None:
    destination.types = ["A"] if source.types is None else list(source.types)
    if source.typeid is None:
        destination.typeid = np.zeros(np.count_nonzero(mask), dtype=np.uint32)
    else:
        destination.typeid = np.asarray(source.typeid)[mask].copy()

    for name in (
        "velocity",
        "mass",
        "charge",
        "diameter",
        "body",
        "image",
        "orientation",
        "moment_inertia",
        "angular_momentum",
    ):
        _copy_masked_particle_property(source, destination, name, mask)


def _copy_masked_particle_property(source, destination, name: str, mask: np.ndarray) -> None:
    values = getattr(source, name, None)
    if values is None:
        return
    values = np.asarray(values)
    if values.shape[:1] != mask.shape:
        return
    setattr(destination, name, values[mask].copy())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a shell-only OVITO GSD with positions unwrapped to x,theta."
    )
    parser.add_argument("--input-gsd", type=Path, default=DEFAULT_INPUT_GSD)
    parser.add_argument("--active-fields", type=Path, default=DEFAULT_ACTIVE_FIELDS_NPZ)
    parser.add_argument("--output-gsd", type=Path, default=DEFAULT_OUTPUT_GSD)
    parser.add_argument("--z", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_shell_xtheta_gsd(
        input_gsd=args.input_gsd,
        output_gsd=args.output_gsd,
        active_fields_npz=args.active_fields,
        z_value=args.z,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
