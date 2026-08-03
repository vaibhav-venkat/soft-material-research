from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .characterization import N_MODES


_SCALAR_PLOTS = {
    "net_area": ("net_band_area.png", r"net band area"),
    "net_mean_width": ("net_mean_width.png", r"net mean width"),
    "net_axial_displacement": (
        "net_axial_displacement.png",
        r"net axial displacement",
    ),
    "net_width_variance": ("net_width_variance.png", r"net width variance"),
    "net_centerline_variance": (
        "net_centerline_variance.png",
        r"net centerline variance",
    ),
    "net_interface_length": (
        "net_interface_length.png",
        r"net two-interface length",
    ),
}


def _plot(steps: np.ndarray, values: np.ndarray, ylabel: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(steps, values, color="black", linewidth=1.6)
    axis.set_xlabel("simulation step")
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_characterization(
    tensors: dict[str, np.ndarray], output_dir: Path
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = tensors["simulation_step"]
    outputs: dict[str, str] = {}
    for field, (filename, ylabel) in _SCALAR_PLOTS.items():
        _plot(steps, tensors[field], ylabel, output_dir / filename)
        outputs[field] = filename
    for mode in range(N_MODES):
        number = mode + 1
        for kind, field, ylabel in (
            (
                "centerline",
                "net_centerline_mode_amplitude",
                rf"net centerline mode {number} amplitude",
            ),
            (
                "width",
                "net_width_mode_amplitude",
                rf"net width mode {number} amplitude",
            ),
        ):
            key = f"net_{kind}_mode_{number}"
            filename = f"{key}.png"
            _plot(steps, tensors[field][:, mode], ylabel, output_dir / filename)
            outputs[key] = filename
    return outputs
