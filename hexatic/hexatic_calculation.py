from pathlib import Path

import numpy as np

if __package__:
    from hexatic.analysis import (
        compute_hexatic_order_trajectory,
        hexatic_probability_distribution,
        load_hexatic_text,
        plot_hexatic_distribution,
        save_distribution_text,
        save_hexatic_text,
        write_hexatic_velocity_gsd,
    )
else:
    from analysis import (
        compute_hexatic_order_trajectory,
        hexatic_probability_distribution,
        load_hexatic_text,
        plot_hexatic_distribution,
        save_distribution_text,
        save_hexatic_text,
        write_hexatic_velocity_gsd,
    )

PROJECT_DIR = Path(__file__).resolve().parent
IN_GSD = PROJECT_DIR / "trajectory.gsd"
HEXATIC_TXT = PROJECT_DIR / "hexatic_order.txt"
DISTRIBUTION_TXT = PROJECT_DIR / "hexatic_order_distribution.txt"
FIGURE_FILE = PROJECT_DIR / "images" / "hexatic_order_distribution.png"
OUT_GSD = PROJECT_DIR / "trajectory_hexatic_velocity.gsd"
EQUILIBRIUM_FRAME = 10
NEIGHBORS = 6
DISTRIBUTION_BINS = 50
VELOCITY_COMPONENT = 0
N_PARTICLES = 1000
RHO = 0.2
SIGMA = 1.0
VOLUME = N_PARTICLES / RHO
CAVITY_RADIUS = 1.4 * (VOLUME * 3.0 / 4.0 / np.pi) ** (1.0 / 3.0)
CUTOFF = 2.0 ** (1.0 / 6.0) * SIGMA
SHELL_THICKNESS = 0.05 * SIGMA
SHELL_DELTA = CUTOFF + SHELL_THICKNESS


def main() -> None:
    steps, psi = compute_hexatic_order_trajectory(
        IN_GSD,
        n_neighbors=NEIGHBORS,
        cavity_radius=CAVITY_RADIUS,
        shell_delta=SHELL_DELTA,
    )
    save_hexatic_text(HEXATIC_TXT, steps, psi)

    hexatic_table = load_hexatic_text(HEXATIC_TXT)
    frame_indices = hexatic_table[:, 0].astype(int)
    psi_abs = hexatic_table[:, 5]

    bin_centers, probability_density, counts = hexatic_probability_distribution(
        psi_abs,
        frame_indices,
        min_frame=EQUILIBRIUM_FRAME,
        bins=DISTRIBUTION_BINS,
        exclude_zeros=True,
    )
    save_distribution_text(
        DISTRIBUTION_TXT,
        bin_centers,
        probability_density,
        counts,
    )
    Path(FIGURE_FILE).parent.mkdir(parents=True, exist_ok=True)
    plot_hexatic_distribution(
        bin_centers,
        probability_density,
        title=f"Hexatic order distribution, shell particles, frames > {EQUILIBRIUM_FRAME}",
        filename=FIGURE_FILE,
    )

    write_hexatic_velocity_gsd(
        IN_GSD,
        OUT_GSD,
        HEXATIC_TXT,
        component=VELOCITY_COMPONENT,
    )

    selected_psi_abs = psi_abs[frame_indices > EQUILIBRIUM_FRAME]
    nonzero_selected_psi_abs = np.count_nonzero(selected_psi_abs)
    distribution_psi_abs = selected_psi_abs[selected_psi_abs > 0.0]
    print(f"Loaded {psi.shape[0]} frames and {psi.shape[1]} particles.")
    print(f"Used cavity radius R={CAVITY_RADIUS:.6f}.")
    print(f"Used wall repulsion cutoff={CUTOFF:.6f}.")
    print(f"Used shell thickness={SHELL_THICKNESS:.6f}.")
    print(f"Used radial cutoff Delta={SHELL_DELTA:.6f}.")
    print(f"Wrote hexatic order to {HEXATIC_TXT}.")
    print(f"Wrote distribution to {DISTRIBUTION_TXT}.")
    print(f"Wrote plot to {FIGURE_FILE}.")
    print(f"Wrote OVITO file to {OUT_GSD}.")
    print(
        "Distribution frames shell-only "
        f"min={distribution_psi_abs.min():.6f}, "
        f"mean={distribution_psi_abs.mean():.6f}, "
        f"max={distribution_psi_abs.max():.6f}, "
        f"nonzero={nonzero_selected_psi_abs}"
    )


if __name__ == "__main__":
    main()
