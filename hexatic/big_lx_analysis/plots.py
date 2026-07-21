"""Seaborn plotting entry points for big-Lx analysis results."""

from importlib import import_module


def configure_style() -> None:
    """Apply the shared plotting defaults for this analysis package."""
    sns = import_module("seaborn")
    sns.set_theme(context="paper", style="ticks")
