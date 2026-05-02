from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from active_passive_calc.analysis import (
        ANALYZERS,
        Membrane,
        ModeAnalyzer,
        Vesicle,
        calculate,
        extract_scaling_factor,
        get_analyzer,
        interpolate_periodic,
        plot,
        plot_two,
    )
else:
    from analysis import (
        ANALYZERS,
        Membrane,
        ModeAnalyzer,
        Vesicle,
        calculate,
        extract_scaling_factor,
        get_analyzer,
        interpolate_periodic,
        plot,
        plot_two,
    )

PROJECT_DIR = Path(__file__).resolve().parent
FN_ACTIVE_VESICLE = PROJECT_DIR / "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-active.gsd"
FN_PASSIVE_VESICLE = PROJECT_DIR / "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-passive.gsd"
FN_ACTIVE_MEMBRANE = PROJECT_DIR / "perimeter-conserved-2D-membrane_CPU_harmonic_bonds-active.gsd"
FN_PASSIVE_MEMBRANE = PROJECT_DIR / "perimeter-conserved-2D-membrane_CPU_harmonic_bonds-passive.gsd"
EQUILIBRIUM_FRAMES = 10
VERTEX_TYPE = 0




def plot_N_abp(
    n: int,
    sigma_vertex: float = 0.05,
    equilibrium_frames: int = EQUILIBRIUM_FRAMES,
    vertex_type: int = VERTEX_TYPE,
    title: str = "Averaged Structure Factor of Active Membrane by N_abp",
    filename: str | None = None,
) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")

    plt.figure(figsize=(10, 6))
    plotted_count = 0

    for idx in range(1, n + 1):
        file_name = PROJECT_DIR / (
            f"perimeter-conserved-2D-membrane_CPU_harmonic_bonds-active-{idx}.gsd"
        )
        avg_structure_factor, k = calculate(
            file_name,
            sigma_vertex,
            equilibrium_frames,
            vertex_type,
            mode="membrane",
        )

        if avg_structure_factor is None or k is None:
            print(f"Skipping {file_name}")
            continue

        positive_k_indices = np.where(k > 0)
        k_pos = k[positive_k_indices]
        structure_positive = avg_structure_factor[positive_k_indices]

        if len(k_pos) == 0:
            print(f"Skipping {file_name}")
            continue

        sort_k_indices = np.argsort(k_pos)
        k_plot = k_pos[sort_k_indices]
        s_plot = structure_positive[sort_k_indices]
        plt.plot(k_plot, np.log(s_plot), label=f"N_abp={10 ** (idx - 1)}", linewidth=1.2)
        plotted_count += 1

    if plotted_count == 0:
        plt.close()
        raise RuntimeError("No valid active membrane files were plotted.")

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
        plt.close()


__all__ = [
    "ModeAnalyzer",
    "Vesicle",
    "Membrane",
    "ANALYZERS",
    "get_analyzer",
    "interpolate_periodic",
    "calculate",
    "extract_scaling_factor",
    "plot",
    "plot_two",
    "FN_ACTIVE_VESICLE",
    "FN_PASSIVE_VESICLE",
    "FN_ACTIVE_MEMBRANE",
    "FN_PASSIVE_MEMBRANE",
    "EQUILIBRIUM_FRAMES",
    "VERTEX_TYPE",
    "plot_N_abp",
]

if __name__ == "__main__":
    MODE = "membrane"

    plot_N_abp(4, filename=PROJECT_DIR / "images" / "N_abp_plot.png")
    # if MODE == "vesicle":
    #     fn_active = FN_ACTIVE_VESICLE
    #     fn_passive = FN_PASSIVE_VESICLE
    #     fn_save = "images/comparison_vesicle.png"
    #     sigma_vertex = 0.01
    #     ttle = "Averaged Structure Factor of Vesicle"
    # elif MODE == "membrane":
    #     fn_active = FN_ACTIVE_MEMBRANE
    #     fn_passive = FN_PASSIVE_MEMBRANE
    #     fn_save = "images/comparison_membrane.png"
    #     sigma_vertex = 0.05
    #     ttle = "Averaged Structure Factor of Membrane"
    # else:
    #     raise ValueError(f"Unknown mode: {MODE}")

    # s1, k1 = calculate(fn_active, sigma_vertex, EQUILIBRIUM_FRAMES, VERTEX_TYPE, MODE)
    # s2, k2 = calculate(fn_passive, sigma_vertex, EQUILIBRIUM_FRAMES, VERTEX_TYPE, MODE)

    # if s1 is not None and k1 is not None and s2 is not None and k2 is not None:
    #     plot_two(s1, k1, "Active", s2, k2, "Passive", title=ttle, filename=fn_save)

    #     scaling_factor_s1 = extract_scaling_factor(s1, k1, MODE)
    #     print(f"Scaling factor for s1 (Active): {scaling_factor_s1}")
    #     scaling_factor_s2 = extract_scaling_factor(s2, k2, MODE)
    #     print(f"Scaling factor for s2 (Passive): {scaling_factor_s2}")
    # else:
    #     print("Error during calculation")
