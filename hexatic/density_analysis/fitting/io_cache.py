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
