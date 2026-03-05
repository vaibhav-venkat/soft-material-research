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

FN_ACTIVE_VESICLE = "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-active.gsd"
FN_PASSIVE_VESICLE = "perimeter-conserved-2D-vesicle_CPU_harmonic_bonds-passive.gsd"
FN_ACTIVE_MEMBRANE = "perimeter-conserved-2D-membrane_CPU_harmonic_bonds-active.gsd"
FN_PASSIVE_MEMBRANE = "perimeter-conserved-2D-membrane_CPU_harmonic_bonds-passive.gsd"
EQUILIBRIUM_FRAMES = 10
VERTEX_TYPE = 0

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
]

if __name__ == "__main__":
    MODE = "vesicle"

    if MODE == "vesicle":
        fn_active = FN_ACTIVE_VESICLE
        fn_passive = FN_PASSIVE_VESICLE
        fn_save = "images/comparison_vesicle.png"
        sigma_vertex = 0.01
        ttle = "Averaged Structure Factor of Vesicle"
    elif MODE == "membrane":
        fn_active = FN_ACTIVE_MEMBRANE
        fn_passive = FN_PASSIVE_MEMBRANE
        fn_save = "images/comparison_membrane.png"
        sigma_vertex = 0.05
        ttle = "Averaged Structure Factor of Membrane"
    else:
        raise ValueError(f"Unknown mode: {MODE}")

    s1, k1 = calculate(fn_active, sigma_vertex, EQUILIBRIUM_FRAMES, VERTEX_TYPE, MODE)
    s2, k2 = calculate(fn_passive, sigma_vertex, EQUILIBRIUM_FRAMES, VERTEX_TYPE, MODE)

    if s1 is not None and k1 is not None and s2 is not None and k2 is not None:
        plot_two(s1, k1, "Active", s2, k2, "Passive", title=ttle, filename=fn_save)

        scaling_factor_s1 = extract_scaling_factor(s1, k1, MODE)
        print(f"Scaling factor for s1 (Active): {scaling_factor_s1}")
        scaling_factor_s2 = extract_scaling_factor(s2, k2, MODE)
        print(f"Scaling factor for s2 (Passive): {scaling_factor_s2}")
    else:
        print("Error during calculation")
