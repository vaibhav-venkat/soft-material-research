"""Cache helpers."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np


SCHEMA_VERSION = 2


def write_npz_atomic(
    path: Path,
    overwrite: bool = False,
    metadata: dict[str, object] | None = None,
    **arrays: np.ndarray,
) -> None:
    """Write a compressed NPZ cache through a temporary file and atomic replace.

    Parameters:
        path: Final cache path.
        overwrite: Whether an existing final path may be replaced.
        metadata: JSON-serializable metadata stored under ``metadata_json``.
        arrays: Named arrays to store alongside the schema version.

    Edge cases:
        A stale ``.tmp`` file from a failed write may be overwritten, but an existing final
        cache is preserved unless ``overwrite`` is true.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = dict(arrays)
    payload["schema_version"] = np.asarray(SCHEMA_VERSION)
    payload["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True))
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp.replace(path)
