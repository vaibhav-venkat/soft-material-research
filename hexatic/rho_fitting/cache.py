"""Cache helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def write_npz_atomic(path: Path, overwrite: bool = False, **arrays: np.ndarray) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, schema_version=SCHEMA_VERSION, **arrays)
    tmp.replace(path)
