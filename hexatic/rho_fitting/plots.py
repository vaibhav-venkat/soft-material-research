"""Plotting hooks for rho fitting."""

from __future__ import annotations

from pathlib import Path


def write_all_plots(output_dir: Path, case_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = case_id
