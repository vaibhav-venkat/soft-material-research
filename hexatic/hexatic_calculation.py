from pathlib import Path

import numpy as np

if __package__:
    from hexatic import analysis as hx
else:
    import analysis as hx

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
SPHERE_OUTPUT_DIR = OUTPUT_DIR / "sphere"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
IN_GSD = SPHERE_OUTPUT_DIR / "trajectory.gsd"
HEXATIC_TXT = SPHERE_OUTPUT_DIR / "hexatic_order.txt"
NEIGHBOR_COUNT_TXT = SPHERE_OUTPUT_DIR / "surface_neighbor_counts.txt"
DISTRIBUTION_TXT = SPHERE_OUTPUT_DIR / "hexatic_order_distribution.txt"
FIGURE_FILE = IMAGE_OUTPUT_DIR / "hexatic_order_distribution.png"
OUT_GSD = SPHERE_OUTPUT_DIR / "trajectory_hexatic_velocity.gsd"
EQUILIBRIUM_FRAME = 10
NEIGHBORS = 6
DISTRIBUTION_BINS = 50
VELOCITY_COMPONENT = 0
NEIGHBOR_COUNT_COMPONENT = 1
N_PARTICLES = 1000
RHO = 0.2
SIGMA = 1.0
VOLUME = N_PARTICLES / RHO
CAVITY_RADIUS = 1.4 * (VOLUME * 3.0 / 4.0 / np.pi) ** (1.0 / 3.0)
CUTOFF = 2.0 ** (1.0 / 6.0) * SIGMA
# NEIGHBOR_COUNT_RADIUS = 2.0 ** (7.0 / 6.0) * SIGMA
NEIGHBOR_COUNT_RADIUS = 2 ** (4.0/6.0)  * SIGMA
SHELL_THICKNESS = 0.05 * SIGMA
SHELL_DELTA = CUTOFF + SHELL_THICKNESS


def main() -> None:
    calculator = hx.SphereHexaticCalculator(
        cavity_radius=CAVITY_RADIUS,
        shell_delta=SHELL_DELTA,
        n_neighbors=NEIGHBORS,
    )
    steps, psi = calculator.compute_hexatic_order_trajectory(IN_GSD)
    hx.save_hexatic_text(HEXATIC_TXT, steps, psi)

    count_steps, neighbor_counts = calculator.compute_neighbor_counts_trajectory(
        IN_GSD,
        neighbor_radius=NEIGHBOR_COUNT_RADIUS,
    )
    assert np.array_equal(count_steps, steps)
    hx.save_neighbor_count_text(NEIGHBOR_COUNT_TXT, steps, neighbor_counts)

    hexatic_table = hx.load_hexatic_text(HEXATIC_TXT)
    frame_indices = hexatic_table[:, 0].astype(int)
    psi_abs = hexatic_table[:, 5]

    bin_centers, probability_density, counts = hx.hexatic_probability_distribution(
        psi_abs,
        frame_indices,
        min_frame=EQUILIBRIUM_FRAME,
        bins=DISTRIBUTION_BINS,
        exclude_zeros=True,
    )
    hx.save_distribution_text(
        DISTRIBUTION_TXT,
        bin_centers,
        probability_density,
        counts,
    )
    hx.plot_hexatic_distribution(
        bin_centers,
        probability_density,
        title=f"hexatic distribution, frames > {EQUILIBRIUM_FRAME}",
        filename=FIGURE_FILE,
    )

    hx.write_hexatic_velocity_gsd(
        IN_GSD,
        OUT_GSD,
        HEXATIC_TXT,
        component=VELOCITY_COMPONENT,
        neighbor_counts=neighbor_counts,
        neighbor_component=NEIGHBOR_COUNT_COMPONENT,
    )

    selected_psi_abs = psi_abs[frame_indices > EQUILIBRIUM_FRAME]
    nonzero_selected_psi_abs = np.count_nonzero(selected_psi_abs)
    distribution_psi_abs = selected_psi_abs[selected_psi_abs > 0.0]
    selected_neighbor_counts = neighbor_counts[EQUILIBRIUM_FRAME + 1 :].reshape(-1)
    shell_neighbor_counts = selected_neighbor_counts[selected_psi_abs > 0.0]
    print(f"Loaded {psi.shape[0]} frames and {psi.shape[1]} particles.")
    print(f"Used cavity radius R={CAVITY_RADIUS:.6f}.")
    print(f"Used wall repulsion cutoff={CUTOFF:.6f}.")
    print(f"Used neighbor-count radius={NEIGHBOR_COUNT_RADIUS:.6f}.")
    print(f"Used shell thickness={SHELL_THICKNESS:.6f}.")
    print(f"Used radial cutoff Delta={SHELL_DELTA:.6f}.")
    print(f"Wrote hexatic order to {HEXATIC_TXT}.")
    print(f"Wrote neighbor counts to {NEIGHBOR_COUNT_TXT}.")
    print(f"Wrote distribution to {DISTRIBUTION_TXT}.")
    print(f"Wrote plot to {FIGURE_FILE}.")
    print(f"Wrote OVITO file to {OUT_GSD}.")
    print(f"OVITO velocity.x stores |psi_6|.")
    print(f"OVITO velocity.y stores surface neighbor count.")
    print(
        "Distribution frames on shell"
        f"min={distribution_psi_abs.min():.6f}, "
        f"mean={distribution_psi_abs.mean():.6f}, "
        f"max={distribution_psi_abs.max():.6f}, "
        f"nonzero={nonzero_selected_psi_abs}"
    )
    print(
        "Surface neighbor counts on shell"
        f"min={shell_neighbor_counts.min()}, "
        f"mean={shell_neighbor_counts.mean():.6f}, "
        f"max={shell_neighbor_counts.max()}, "
        f"particles={len(shell_neighbor_counts)}"
    )


if __name__ == "__main__":
    main()
