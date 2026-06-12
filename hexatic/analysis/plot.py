from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

__all__ = ["DistributionPlotStyle", "plot_hexatic_distribution"]


@dataclass(frozen=True)
class DistributionPlotStyle:
    figsize: tuple[float, float] = (8.0, 5.0)
    fallback_width: float = 0.02
    width_fraction: float = 0.9
    x_min: float = 0.0
    x_max: float = 1.0
    dpi: int = 200


def plot_hexatic_distribution(
    bin_centers: npt.NDArray[np.float64],
    probability_density: npt.NDArray[np.float64],
    title: str = "Hexatic Order Distribution",
    filename: str | Path | None = None,
    style: DistributionPlotStyle | None = None,
) -> None:
    style = DistributionPlotStyle() if style is None else style
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    probability_density = np.asarray(probability_density, dtype=np.float64)
    assert bin_centers.ndim == 1 and probability_density.ndim == 1
    assert bin_centers.shape == probability_density.shape
    if len(bin_centers) > 1:
        width = float(np.mean(np.diff(bin_centers)))
    else:
        width = style.fallback_width

    plt.figure(figsize=style.figsize)
    plt.bar(
        bin_centers,
        probability_density,
        width=style.width_fraction * width,
        align="center",
        edgecolor="black",
        linewidth=0.4,
    )
    plt.xlabel("|psi_6|")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.xlim(style.x_min, style.x_max)
    plt.grid(True, axis="y", ls="--", alpha=0.5)
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=style.dpi)
        plt.close()
