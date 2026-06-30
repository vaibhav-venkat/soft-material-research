"""Report writing for rho fitting."""

from __future__ import annotations

from pathlib import Path


def write_report(path: Path, lines: list[str], overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
