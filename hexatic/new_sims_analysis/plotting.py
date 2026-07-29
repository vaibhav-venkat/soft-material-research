"""Shared figure output for the new-sims plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def _save_figure(figure: Figure, output: Path) -> None:
    """Write ``output`` as SVG plus a 300-dpi PNG sibling, then close the figure.

    Every plot in this package emits the same pair, so the convention lives here
    rather than being repeated at each call site.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Creator": "hexatic.new_sims_analysis"},
    )
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
