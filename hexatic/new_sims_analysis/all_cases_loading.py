"""Load center-of-mass series from all supported analysis families."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from safetensors import safe_open

from hexatic.confinement_comparison.cases import (
    GeometryKind,
    get_case as get_confinement_case,
)
from hexatic.constants import cylinder
from hexatic.new_sims.cases import get_case as get_new_sims_case

from .loading import (
    CARTESIAN_COORDINATE_ORDER,
    CYLINDRICAL_COORDINATE_ORDER,
    PLANAR_COORDINATE_ORDER,
    _load_lx,
)


# The three families never agreed on a manifest contract: only new_sims records
# `coordinate_order`, none records `timestep`, the axial box length is stored as
# `lx`, `box[0]`, or `logical_lx` depending on family, and confinement even
# switches position tensor by geometry. Everything per-family below is a shim
# over that drift, kept because the outputs already exist remotely. The real fix
# is to normalise those keys in the three writers (and backfill the existing
# manifests, which are small JSON/static tensors) -- do that before adding a
# fourth family rather than adding a fourth branch to each helper here.
SUPPORTED_SCHEMAS: dict[str, str] = {
    "hexatic.new_sims.analysis.v1": "new_sims",
    "hexatic.big_lx.analysis.v1": "big_lx",
    "hexatic.confinement_comparison.analysis.v1": "confinement",
}
TARGET_CIRCUMFERENCE_DIAMETERS = 60.5

CYLINDRICAL_CONFINEMENT_KINDS = frozenset(
    {GeometryKind.CYLINDER_RATTLE, GeometryKind.CYLINDER_RATTLE_TANGENT}
)
CARTESIAN_CONFINEMENT_KINDS = frozenset(
    {
        GeometryKind.PRISM_VOLUME,
        GeometryKind.PRISM_SURFACE_AREA,
        GeometryKind.SANDWICH_VOLUME,
        GeometryKind.SANDWICH_SURFACE_AREA,
    }
)


@dataclass(frozen=True)
class CaseSeries:
    family: str
    case_id: str
    geometry_kind: str
    lx_multiplier: int
    coordinate_kind: str
    is_tracer: bool
    interaction_class: str
    com: NDArray[np.float64]
    steps: NDArray[np.int64]
    timestep: float


def load_manifest(directory: Path) -> tuple[dict[str, Any], str]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be an object: {manifest_path}")
    schema = manifest.get("schema")
    family = SUPPORTED_SCHEMAS.get(schema)
    if family is None:
        raise ValueError(f"unsupported manifest schema {schema!r} in {directory}")
    if manifest.get("complete") is not True:
        raise ValueError(f"analysis manifest is incomplete: {manifest_path}")
    if not isinstance(manifest.get("case"), dict):
        raise ValueError(f"missing case metadata in {manifest_path}")
    return manifest, family


def _validate_circumference(case: dict[str, Any]) -> None:
    stored = case.get("circumference_diameters")
    if stored is not None:
        value = float(stored)
    else:
        try:
            value = float(case["circumference"]) / float(case["particle_diameter"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise ValueError("case metadata cannot determine circumference in diameters") from error
    if not np.isfinite(value) or not np.isclose(
        value, TARGET_CIRCUMFERENCE_DIAMETERS, rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(
            f"expected circumference {TARGET_CIRCUMFERENCE_DIAMETERS}D, got {value!r}D"
        )


def _lx_multiplier(case: dict[str, Any]) -> int:
    stored = case.get("lx_multiplier")
    if stored is not None:
        multiplier = int(stored)
    else:
        base_case_id = case.get("base_case_id")
        match = re.search(r"_lx_(\d+)x$", base_case_id) if isinstance(base_case_id, str) else None
        if match is None:
            raise ValueError("case metadata cannot determine the Lx multiplier")
        multiplier = int(match.group(1))
    if multiplier < 1:
        raise ValueError(f"invalid Lx multiplier {multiplier}")
    return multiplier


def _coordinate_spec(
    manifest: dict[str, Any], case: dict[str, Any], family: str
) -> tuple[str, str]:
    if family == "new_sims":
        order = manifest.get("coordinate_order")
        kinds = {
            tuple(PLANAR_COORDINATE_ORDER): "planar",
            tuple(CARTESIAN_COORDINATE_ORDER): "cartesian",
            tuple(CYLINDRICAL_COORDINATE_ORDER): "cylindrical",
        }
        kind = kinds.get(tuple(order) if isinstance(order, list) else ())
        if kind is None:
            raise ValueError(f"unsupported coordinate_order {order!r}")
        return "coords", kind
    if family == "big_lx":
        return "coords", "cylindrical"

    geometry_kind = case.get("geometry_kind")
    if geometry_kind in CYLINDRICAL_CONFINEMENT_KINDS:
        return "coords", "cylindrical"
    if geometry_kind == GeometryKind.TWO_DIMENSION:
        return "coords", "planar"
    if geometry_kind in CARTESIAN_CONFINEMENT_KINDS:
        return "position_cartesian", "cartesian"
    raise ValueError(f"unsupported confinement geometry_kind {geometry_kind!r}")


def _static_scalar(path: Path, name: str) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"missing static tensors: {path}")
    with safe_open(path, framework="np") as tensors:
        if name not in tensors.keys():
            raise KeyError(f"{path} has no {name!r} tensor")
        value = np.asarray(tensors.get_tensor(name))
    if value.size != 1:
        raise ValueError(f"expected scalar {name} in {path}")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"invalid {name}={result!r} in {path}")
    return result


def _axial_period(directory: Path, family: str) -> float:
    static_path = directory / "static.safetensors"
    if family == "confinement":
        return _static_scalar(static_path, "logical_lx")
    return _load_lx(static_path)


def _component_periods(
    directory: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    family: str,
    coordinate_kind: str,
) -> list[float | None]:
    if coordinate_kind == "cylindrical":
        return [_axial_period(directory, family), 2.0 * np.pi, None]
    if family == "new_sims":
        storage = manifest.get("coordinate_storage")
        if storage != "cartesian_unwrapped_on_periodic_axes":
            raise ValueError(
                "new_sims non-cylindrical coordinates must be declared "
                "cartesian_unwrapped_on_periodic_axes"
            )
        return [None] * (2 if coordinate_kind == "planar" else 3)

    # Which axes are periodic, and their transverse box widths, are pure
    # functions of the case definition -- not of the run -- so take them from the
    # case registry rather than the manifest. Production metadata was written
    # before `periodic_axes` and `stored_box` were added to `as_metadata()`, so
    # the already-computed outputs simply do not carry those keys.
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("confinement case metadata has no case_id")
    try:
        comparison = get_confinement_case(case_id)
    except KeyError as error:
        raise ValueError(
            f"confinement case {case_id!r} is not in the case registry, so its "
            "periodic axes cannot be determined"
        ) from error
    if comparison.logical_to_stored_axes != (0, 1, 2):
        raise ValueError(
            "non-cylinder confinement coordinates require "
            f"logical_to_stored_axes == (0, 1, 2), got "
            f"{comparison.logical_to_stored_axes} for {case_id}"
        )
    stored_box = comparison.stored_box
    periodic_axes = comparison.periodic_axes
    axis_names = ("x", "y") if coordinate_kind == "planar" else ("x", "y", "z")
    periods: list[float | None] = []
    for index, name in enumerate(axis_names):
        if name not in periodic_axes:
            periods.append(None)
            continue
        period = _axial_period(directory, family) if name == "x" else float(stored_box[index])
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError(f"invalid period for confinement axis {name}: {period!r}")
        periods.append(period)
    return periods


def _axis_com(
    values: NDArray[np.float64],
    period: float | None,
    carry: tuple[NDArray[np.float64], NDArray[np.float64]] | None,
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]] | None]:
    """Reduce one axis of ``(frames, particles)`` coordinates to a COM series.

    The vectorised counterpart of ``loading._PeriodicComUnwrapper``: that class
    walks frame by frame, which costs a Python call and a full particle-width
    reduction per frame per axis. The same recurrence is a cumulative sum, so one
    ``np.cumsum`` over the shard replaces the loop. ``carry`` threads the last raw
    and unwrapped frame across shard boundaries; ``None`` period means the axis is
    wall-bounded and needs no unwrapping at all.

    Summation order differs from the sequential form, so results agree to
    floating-point rounding rather than bit-exactly.
    """
    if period is None:
        return values.mean(axis=1), None
    if carry is None:
        displacement = np.diff(values, axis=0)
    else:
        previous, _ = carry
        displacement = np.diff(np.concatenate([previous[None, :], values]), axis=0)
    scaled = displacement / period
    jumps = displacement - period * np.copysign(
        np.floor(np.abs(scaled) + 0.5), scaled
    )
    if carry is None:
        unwrapped = np.concatenate(
            [values[:1], values[:1] + np.cumsum(jumps, axis=0)]
        )
    else:
        unwrapped = carry[1] + np.cumsum(jumps, axis=0)
    return unwrapped.mean(axis=1), (values[-1].copy(), unwrapped[-1].copy())


def load_com_series(
    directory: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    family: str,
    tensor_name: str,
    coordinate_kind: str,
    frame_limit: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Load only particle coordinates and reduce them to an unwrapped COM."""
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest contains no safetensor shards")
    component_count = 2 if coordinate_kind == "planar" else 3
    periods = _component_periods(
        directory, manifest, case, family, coordinate_kind
    )
    carries: list[tuple[NDArray[np.float64], NDArray[np.float64]] | None]
    carries = [None] * component_count
    com_chunks: list[NDArray[np.float64]] = []
    step_chunks: list[NDArray[np.int64]] = []
    loaded_frames = 0
    expected_start = 0
    particle_count: int | None = None

    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError("invalid shard entry in manifest")
        start, stop, filename = (
            entry.get("frame_start"),
            entry.get("frame_stop"),
            entry.get("file"),
        )
        if start != expected_start or not isinstance(stop, int) or stop <= start:
            raise ValueError("safetensor shards are not contiguous")
        if not isinstance(filename, str):
            raise ValueError("shard entry has no filename")
        shard_path = directory / filename
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with safe_open(shard_path, framework="np") as shard:
            keys = set(shard.keys())
            if tensor_name not in keys or "step" not in keys:
                raise KeyError(f"{shard_path} must contain {tensor_name!r} and 'step'")
            coords = np.asarray(shard.get_tensor(tensor_name), dtype=np.float64)
            steps = np.asarray(shard.get_tensor("step"), dtype=np.int64)
        if coords.ndim != 3 or coords.shape[1] == 0 or coords.shape[2] != component_count:
            raise ValueError(
                f"expected {tensor_name} shape (frames, particles, {component_count}), "
                f"got {coords.shape} in {shard_path}"
            )
        if steps.ndim != 1 or len(steps) != len(coords):
            raise ValueError(f"expected one simulation step per frame in {shard_path}")
        if len(coords) != stop - start:
            raise ValueError(
                f"shard range [{start}, {stop}) does not match {len(coords)} stored frames"
            )
        if particle_count is None:
            particle_count = coords.shape[1]
        elif coords.shape[1] != particle_count:
            raise ValueError(f"particle count changes in {shard_path}")
        # Trim before validating and unwrapping: a shard past the frame limit
        # would otherwise be finite-checked over every particle and discarded.
        take = min(frame_limit - loaded_frames, len(coords))
        coords = coords[:take]
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"coordinates contain non-finite values in {shard_path}")
        axis_series = []
        for axis in range(component_count):
            series, carries[axis] = _axis_com(
                coords[:, :, axis], periods[axis], carries[axis]
            )
            axis_series.append(series)
        com_chunks.append(np.stack(axis_series, axis=1))
        step_chunks.append(steps[:take])
        loaded_frames += take
        expected_start = stop
        if loaded_frames >= frame_limit:
            break

    # A case that is short of the requested frames would silently get its own
    # shorter lag grid, and the only symptom downstream is an opaque Laplace
    # r-grid mismatch. Name the directory here instead.
    if loaded_frames < frame_limit:
        raise ValueError(
            f"{directory} has only {loaded_frames} analysed frames, but "
            f"{frame_limit} were requested; every case must supply the same "
            "frame count so the Laplace grids agree"
        )
    if loaded_frames < 3:
        raise ValueError("at least three frames are required for velocity correlation")
    return (
        np.concatenate(com_chunks, axis=0),
        np.concatenate(step_chunks, axis=0),
    )


def case_key(family: str, case_id: str) -> str:
    return f"{family}:{case_id}"


def _classify(family: str, case_id: str, case: dict[str, Any]) -> tuple[bool, str]:
    """Return ``(is_tracer, interaction_class)`` for the plot's style channels.

    Prefer the case registries: whether a case has pair interactions or is a
    tracer setup is part of its definition, and the manifest fields are optional
    (older production metadata omits ``pair_interaction``, which would silently
    mis-colour a non-interacting case as interacting rather than failing). Fall
    back to the metadata heuristic only for a case_id no registry knows.
    """
    if family == "new_sims":
        try:
            registered = get_new_sims_case(case_id)
        except KeyError:
            pass
        else:
            if registered.is_tracer:
                return True, "tracer"
            return False, (
                "interacting" if registered.has_pair_interaction else "non-interacting"
            )
    elif family in {"big_lx", "confinement"}:
        # Every big_lx and confinement case is a fully interacting film.
        return False, "interacting"

    active_count = case.get("active_count")
    particle_count = case.get("n_particles")
    tracer = (
        active_count is not None
        and particle_count is not None
        and int(active_count) < int(particle_count)
    )
    if tracer:
        return True, "tracer"
    return False, (
        "interacting" if bool(case.get("pair_interaction", True)) else "non-interacting"
    )


def load_case_series(directory: Path, frame_limit: int) -> CaseSeries:
    manifest, family = load_manifest(directory)
    manifest_path = directory / "manifest.json"
    case = manifest["case"]
    _validate_circumference(case)
    tensor_name, coordinate_kind = _coordinate_spec(manifest, case, family)
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"missing case_id in {manifest_path}")
    geometry_kind = case.get("geometry_kind", "cylinder" if family == "big_lx" else None)
    if not isinstance(geometry_kind, str) or not geometry_kind:
        raise ValueError(f"missing geometry_kind in {manifest_path}")
    # Only new_sims records a timestep; big_lx and confinement_comparison omit
    # it entirely. All three simulate with the same constant, so read it from
    # the constant rather than letting one family fall back and silently shift
    # its lag grid relative to the others.
    timestep = float(cylinder.SIMULATION.timestep)
    if not np.isfinite(timestep) or timestep <= 0.0:
        raise ValueError(f"invalid simulation timestep constant {timestep!r}")
    com, steps = load_com_series(
        directory,
        manifest,
        case,
        family,
        tensor_name,
        coordinate_kind,
        frame_limit,
    )
    tracer, interaction_class = _classify(family, case_id, case)
    return CaseSeries(
        family=family,
        case_id=case_id,
        geometry_kind=geometry_kind,
        lx_multiplier=_lx_multiplier(case),
        coordinate_kind=coordinate_kind,
        is_tracer=tracer,
        interaction_class=interaction_class,
        com=com,
        steps=steps,
        timestep=timestep,
    )
