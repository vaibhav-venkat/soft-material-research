from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_npz_arrays(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_cache(path: str | Path) -> dict[str, Any]:
    return load_npz_arrays(path)


def write_cache(
    path: str | Path,
    *,
    overwrite: bool = False,
    **arrays: Any,
) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} already exists; pass overwrite=True to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def flatten_array_dict(
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {f"{prefix}__names": np.asarray(tuple(arrays))}
    for name, values in arrays.items():
        flattened[f"{prefix}__{name}"] = values
    return flattened


def reconstruct_array_dict(
    arrays: dict[str, Any],
    prefix: str,
) -> dict[str, np.ndarray]:
    names_key = f"{prefix}__names"
    if names_key not in arrays:
        return {}
    names = tuple(str(name) for name in np.asarray(arrays[names_key]))
    return {
        name: np.asarray(arrays[f"{prefix}__{name}"])
        for name in names
        if f"{prefix}__{name}" in arrays
    }


def flatten_float_dict(values: dict[str, float], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}__names": np.asarray(tuple(values)),
        f"{prefix}__values": np.asarray([values[name] for name in values], dtype=float),
    }


def reconstruct_float_dict(arrays: dict[str, Any], prefix: str) -> dict[str, float]:
    names_key = f"{prefix}__names"
    values_key = f"{prefix}__values"
    if names_key not in arrays or values_key not in arrays:
        return {}
    names = tuple(str(name) for name in np.asarray(arrays[names_key]))
    values = np.asarray(arrays[values_key], dtype=float)
    return {name: float(value) for name, value in zip(names, values, strict=True)}
