from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_hexatic_distribution(
    bin_centers: npt.NDArray[np.float64],
    probability_density: npt.NDArray[np.float64],
    title: str = "Hexatic Order Distribution",
    filename: str | Path  | None = None,
) -> None:
    bin_centers = np.asarray(bin_centers, dtype=np.float64)
    probability_density = np.asarray(probability_density, dtype=np.float64)
    assert bin_centers.ndim == 1 and probability_density.ndim == 1
    assert bin_centers.shape == probability_density.shape
    if len(bin_centers) > 1:
        width = float(np.mean(np.diff(bin_centers)))
    else:
        width = 0.02

    plt.figure(figsize=(8, 5))
    plt.bar(
        bin_centers,
        probability_density,
        width=0.9 * width,
        align="center",
        edgecolor="black",
        linewidth=0.4,
    )
    plt.xlabel("|psi_6|")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.grid(True, axis="y", ls="--", alpha=0.5)
    plt.tight_layout()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=200)
        plt.close()
