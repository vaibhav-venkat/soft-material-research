from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from ..config import ACTIVE_MOVIE_FPS, ActiveMatterFields


def _write_movie(
    fields: ActiveMatterFields,
    filename: str | Path,
    draw_frame,
    fps: int = ACTIVE_MOVIE_FPS,
) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 5))
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, str(output_path), dpi=160):
        for frame_idx in range(len(fields.steps)):
            fig.clear()
            axis = fig.add_subplot(111)
            draw_frame(fig, axis, frame_idx)
            fig.tight_layout()
            writer.grab_frame()
    plt.close(fig)
