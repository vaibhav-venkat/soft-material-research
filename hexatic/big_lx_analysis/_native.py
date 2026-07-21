"""Shared ctypes helpers for the Zig analysis libraries."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class F64Buffer(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_double)), ("len", ctypes.c_size_t)]


class UsizeBuffer(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_size_t)), ("len", ctypes.c_size_t)]


def load_library(name: str) -> ctypes.CDLL:
    """Load a package-local Zig shared library, with an environment override."""
    override = os.environ.get(f"HEXATIC_{name.upper()}_LIBRARY")
    candidates = [Path(override)] if override else []
    suffix = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    prefix = "" if sys.platform == "win32" else "lib"
    repository_root = Path(__file__).resolve().parents[2]
    candidates.append(
        repository_root / "packages" / name / "zig-out" / "lib" / f"{prefix}{name}{suffix}"
    )
    for candidate in candidates:
        if candidate.is_file():
            return ctypes.CDLL(str(candidate))
    raise RuntimeError(
        f"{name} Zig library is not built; run `cd packages/{name} && zig build` "
        f"or set HEXATIC_{name.upper()}_LIBRARY"
    )


def encode_paths(
    files: str | Path | Sequence[str | Path],
) -> tuple[ctypes.Array[ctypes.c_char_p], list[bytes]]:
    """Create a C string array while retaining the encoded-string owners."""
    values = [files] if isinstance(files, (str, Path)) else list(files)
    if not values:
        raise ValueError("at least one safetensor file is required")
    encoded = [os.fsencode(Path(value)) for value in values]
    array_type = ctypes.c_char_p * len(encoded)
    return array_type(*encoded), encoded


def copy_f64(buffer: F64Buffer) -> NDArray[np.float64]:
    if buffer.len == 0:
        return np.empty(0, dtype=np.float64)
    return np.ctypeslib.as_array(buffer.ptr, shape=(buffer.len,)).copy()


def copy_usize(buffer: UsizeBuffer) -> NDArray[np.intp]:
    if buffer.len == 0:
        return np.empty(0, dtype=np.intp)
    return np.ctypeslib.as_array(buffer.ptr, shape=(buffer.len,)).astype(np.intp, copy=True)


def raise_for_status(status: int, package: str) -> None:
    messages = {
        1: "invalid paths or options",
        2: "native allocation failed",
        3: "a safetensor file could not be opened or validated",
    }
    if status:
        raise RuntimeError(f"{package}: {messages.get(status, f'native error {status}')}")
