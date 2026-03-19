import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot(
    avg_structure_factor: npt.NDArray[np.float64],
    k: np.ndarray,
    label: str,
    title: str = "Averaged Structure Factor",
    filename: str | None = None,
):
    positive_k_indices = np.where(k > 0)
    k_pos = k[positive_k_indices]
    structure_positive = avg_structure_factor[positive_k_indices]

    sort_k_indices = np.argsort(k_pos)
    k_plot = k_pos[sort_k_indices]
    s_plot = structure_positive[sort_k_indices]

    plt.plot(k_plot, np.log(s_plot), label=label)
    plt.xscale("log")
    plt.xlabel("Wavenumber, k")
    plt.ylabel("log(|h(k)|^2)")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def plot_two(
    avg_structure_factor1: npt.NDArray[np.float64],
    k1: np.ndarray,
    label1: str,
    avg_structure_factor2: npt.NDArray[np.float64],
    k2: np.ndarray,
    label2: str,
    title: str = "Averaged Structure Factor",
    filename: str | None = None,
):
    plt.figure(figsize=(10, 6))

    positive_k_indices1 = np.where(k1 > 0)
    k_pos1 = k1[positive_k_indices1]
    structure_positive1 = avg_structure_factor1[positive_k_indices1]
    sort_k_indices1 = np.argsort(k_pos1)
    k_plot1 = k_pos1[sort_k_indices1]
    s_plot1 = structure_positive1[sort_k_indices1]
    plt.plot(k_plot1, np.log(s_plot1), label=label1)

    positive_k_indices2 = np.where(k2 > 0)
    k_pos2 = k2[positive_k_indices2]
    structure_positive2 = avg_structure_factor2[positive_k_indices2]
    sort_k_indices2 = np.argsort(k_pos2)
    k_plot2 = k_pos2[sort_k_indices2]
    s_plot2 = structure_positive2[sort_k_indices2]
    plt.plot(k_plot2, np.log(s_plot2), label=label2)

    plt.xscale("log")
    plt.xlabel("Wavenumber, k")
    plt.ylabel("log(|h(k)|^2)")
    plt.title(title)
    plt.grid(True, which="both", ls="--")
    plt.legend()

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
