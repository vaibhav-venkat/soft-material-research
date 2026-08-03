"""Public facade and orchestration for band-analysis plots."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np

matplotlib.use("Agg")

from .dynamics import DYNAMIC_PROPERTIES
from .plotting.conditional import _plot_conditional_property
from .plotting.diagnostics import _plot_area_diagnostics
from .plotting.fits import (
    _plot_area_diffusion_fits,
    _plot_area_geometry_fits,
    _plot_area_increment_conservation,
)
from .plotting.neighbors import (
    _plot_neighbor_relative_area_change_drift,
    _plot_neighbor_relative_area_drift,
    _plot_normalized_neighbor_relative_area_drift,
)
from .plotting.overview import (
    _plot_area_coupling,
    _plot_area_cv_squared_drift,
    _plot_area_heterogeneity_events,
    _plot_total_area_drift,
)


_LEGACY_NET_PLOTS = (
    "net_band_area.png",
    "net_mean_width.png",
    "net_axial_displacement.png",
    "net_width_variance.png",
    "net_centerline_variance.png",
    "net_interface_length.png",
    *(f"net_centerline_mode_{mode}.png" for mode in range(1, 7)),
    *(f"net_width_mode_{mode}.png" for mode in range(1, 7)),
)


def plot_characterization(
    tensors: dict[str, np.ndarray],
    output_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """Render every stochastic and morphology diagnostic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in _LEGACY_NET_PLOTS:
        (output_dir / filename).unlink(missing_ok=True)

    outputs: dict[str, str] = {}
    for plot_index, prop in enumerate(DYNAMIC_PROPERTIES, start=1):
        filename = f"stochastic_{prop.slug}.png"
        _plot_conditional_property(
            tensors,
            slug=prop.slug,
            label=prop.label,
            output=output_dir / filename,
            include_msd=prop.slug == "axial_position",
        )
        outputs[prop.slug] = filename
        if progress is not None:
            progress(
                f"stage=plots stochastic={plot_index}/{len(DYNAMIC_PROPERTIES)} "
                f"property={prop.slug}"
            )
    outputs.update(_plot_area_coupling(tensors, output_dir))
    outputs.update(_plot_area_heterogeneity_events(tensors, output_dir))
    outputs.update(_plot_area_cv_squared_drift(tensors, output_dir))
    outputs.update(_plot_total_area_drift(tensors, output_dir))
    outputs.update(_plot_area_diffusion_fits(tensors, output_dir))
    outputs.update(_plot_area_geometry_fits(tensors, output_dir))
    outputs.update(_plot_area_increment_conservation(tensors, output_dir))
    outputs.update(_plot_neighbor_relative_area_drift(tensors, output_dir))
    outputs.update(
        _plot_neighbor_relative_area_change_drift(tensors, output_dir)
    )
    outputs.update(
        _plot_normalized_neighbor_relative_area_drift(tensors, output_dir)
    )
    outputs.update(_plot_area_diagnostics(tensors, output_dir))
    if progress is not None:
        progress("stage=plots coupling=2/2 complete")
        progress("stage=plots area_diagnostics=13/13 complete")
    return outputs


__all__ = ["plot_characterization"]
