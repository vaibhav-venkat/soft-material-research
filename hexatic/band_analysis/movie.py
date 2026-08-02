from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from .density import SurfaceGrid


def write_band_movie(
    frames: Iterable[tuple[int, int, np.ndarray]],
    *,
    grid: SurfaceGrid,
    output: Path,
    fps: int,
    dpi: int,
) -> None:
    frames = list(frames)
    maximum_label = max((int(np.max(labels)) for _, _, labels in frames), default=0)
    band_colors = matplotlib.colormaps["turbo"]
    colors = ["black"] + [
        band_colors(index / max(1, maximum_label - 1))
        for index in range(maximum_label)
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)
    figure, axis = plt.subplots(figsize=(10.0, 5.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    try:
        with writer.saving(figure, str(output), dpi=dpi):
            for frame_index, step, labels in frames:
                axis.clear()
                axis.imshow(
                    labels.T,
                    origin="lower",
                    interpolation="nearest",
                    aspect="auto",
                    extent=(-0.5 * grid.lx, 0.5 * grid.lx, 0.0, grid.circumference),
                    cmap=cmap,
                    norm=norm,
                )
                axis.set_xlabel("x")
                axis.set_ylabel(r"$s=R_s\theta$")
                axis.set_title(f"dilute wrapped bands: frame {frame_index}, step {step}")
                figure.tight_layout()
                writer.grab_frame()
    finally:
        plt.close(figure)
