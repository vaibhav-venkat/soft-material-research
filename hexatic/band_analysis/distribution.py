from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class DistributionFeatures:
    peaks: np.ndarray
    valleys: np.ndarray


def find_features(values: np.ndarray, bins: int = 160) -> tuple[np.ndarray, np.ndarray, DistributionFeatures]:
    counts_device, edges_device = jnp.histogram(
        jnp.asarray(values), bins=bins, density=True
    )
    counts, edges = jax.device_get((counts_device, edges_device))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(counts, sigma=1.5)
    prominence = 0.02 * float(np.max(smooth))
    peaks, _ = find_peaks(smooth, prominence=prominence)
    valleys, _ = find_peaks(-smooth, prominence=prominence)
    return centers, smooth, DistributionFeatures(centers[peaks], centers[valleys])


def plot_distribution(
    area_fractions: np.ndarray,
    *,
    threshold: float,
    output: Path,
) -> DistributionFeatures:
    centers, density, features = find_features(area_fractions)
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(centers, density, color="black", linewidth=1.8)
    for index, peak in enumerate(features.peaks):
        axis.axvline(peak, color="tab:blue", alpha=0.55, linestyle=":", label="peak" if index == 0 else None)
    for index, valley in enumerate(features.valleys):
        axis.axvline(valley, color="tab:orange", alpha=0.7, linestyle="--", label="valley" if index == 0 else None)
    axis.axvline(threshold, color="tab:red", linewidth=1.5, label=r"fixed $\phi_c$")
    axis.set_xlabel(r"local projected area fraction $\phi=\rho\pi D^2/4$")
    axis.set_ylabel("probability density")
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return features
