from pathlib import Path

import numpy as np

if __package__:
    from hexatic.analysis import (
        compute_hexatic_order_cylinder_trajectory,
        compute_neighbor_counts_cylinder_trajectory,
        hexatic_probability_distribution,
        load_hexatic_text,
        plot_hexatic_distribution,
        save_distribution_text,
        save_hexatic_text,
        save_neighbor_count_text,
        write_hexatic_velocity_gsd,
    )
else:
    from analysis import (
        compute_hexatic_order_cylinder_trajectory,
        compute_neighbor_counts_cylinder_trajectory,
        hexatic_probability_distribution,
        load_hexatic_text,
        plot_hexatic_distribution,
        save_distribution_text,
        save_hexatic_text,
        save_neighbor_count_text,
        write_hexatic_velocity_gsd,
    )

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
CYLINDER_OUTPUT_DIR = OUTPUT_DIR / "cylinder"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
IN_GSD = CYLINDER_OUTPUT_DIR / "trajectory_cylinder.gsd"
HEXATIC_TXT = CYLINDER_OUTPUT_DIR / "cylinder_hexatic_order.txt"
NEIGHBOR_COUNT_TXT = CYLINDER_OUTPUT_DIR / "cylinder_surface_neighbor_counts.txt"
DISTRIBUTION_TXT = CYLINDER_OUTPUT_DIR / "cylinder_hexatic_order_distribution.txt"
FIGURE_FILE = IMAGE_OUTPUT_DIR / "cylinder_hexatic_order_distribution.png"
OUT_GSD = CYLINDER_OUTPUT_DIR / "trajectory_cylinder_hexatic_velocity.gsd"

EQUILIBRIUM_FRAME = 10
NEIGHBORS = 6
DISTRIBUTION_BINS = 50

HEXATIC_COMPONENT = 0
NEIGHBOR_COUNT_COMPONENT = 1
DISCLINATION_CHARGE_COMPONENT = 2

SIGMA = 1.0
CYLINDER_RADIUS = 10.0 * 2.0 ** (1.0 / 6.0)
WALL_CUTOFF = 2.0 ** (1.0 / 6.0) * SIGMA
MIN_NEIGHBOR_COUNT_RADIUS = WALL_CUTOFF
MAX_NEIGHBOR_COUNT_RADIUS = 2.0 ** (7.0 / 6.0) * SIGMA
SHELL_DELTA = WALL_CUTOFF
NEIGHBOR_COUNT_RADIUS = 1.7 * SIGMA


def disclination_charges_from_counts(
    neighbor_counts: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    neighbor_counts = np.asarray(neighbor_counts, dtype=np.int64)
    psi = np.asarray(psi, dtype=np.complex128)
    assert neighbor_counts.shape == psi.shape

    charges = np.zeros_like(neighbor_counts, dtype=np.int64)
    surface_mask = np.abs(psi) > 0.0
    charges[surface_mask] = NEIGHBORS - neighbor_counts[surface_mask]
    return charges


def main() -> None:

    print(f"Used cylinder radius R={CYLINDER_RADIUS:.6f}.")
    print(f"Used wall repulsion cutoff={WALL_CUTOFF:.6f}.")
    print(f"Used radial shell cutoff Delta={SHELL_DELTA:.6f}.")
    print(
        "Allowed neighbor-count radius range="
        f"({MIN_NEIGHBOR_COUNT_RADIUS:.6f}, {MAX_NEIGHBOR_COUNT_RADIUS:.6f})."
    )
    print(f"Used neighbor-count radius={NEIGHBOR_COUNT_RADIUS:.6f}.")
    steps, psi = compute_hexatic_order_cylinder_trajectory(
        IN_GSD,
        cylinder_radius=CYLINDER_RADIUS,
        shell_delta=SHELL_DELTA,
        n_neighbors=NEIGHBORS,
    )
    print(f"Loaded {psi.shape[0]} frames and {psi.shape[1]} particles.")

    save_hexatic_text(HEXATIC_TXT, steps, psi)

    count_steps, neighbor_counts = compute_neighbor_counts_cylinder_trajectory(
        IN_GSD,
        neighbor_radius=NEIGHBOR_COUNT_RADIUS,
        cylinder_radius=CYLINDER_RADIUS,
        shell_delta=SHELL_DELTA,
    )
    assert np.array_equal(count_steps, steps)
    save_neighbor_count_text(NEIGHBOR_COUNT_TXT, steps, neighbor_counts)

    disclination_charges = disclination_charges_from_counts(neighbor_counts, psi)

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
        title=f"cylinder hexatic distribution, frames > {EQUILIBRIUM_FRAME}",
        filename=FIGURE_FILE,
    )

    write_hexatic_velocity_gsd(
        IN_GSD,
        OUT_GSD,
        HEXATIC_TXT,
        component=HEXATIC_COMPONENT,
        neighbor_counts=neighbor_counts,
        neighbor_component=NEIGHBOR_COUNT_COMPONENT,
        disclination_charges=disclination_charges,
        charge_component=DISCLINATION_CHARGE_COMPONENT,
    )

    selected_psi_abs = psi_abs[frame_indices > EQUILIBRIUM_FRAME]
    distribution_psi_abs = selected_psi_abs[selected_psi_abs > 0.0]
    surface_mask = np.abs(psi[EQUILIBRIUM_FRAME + 1 :]) > 0.0
    surface_neighbor_counts = neighbor_counts[EQUILIBRIUM_FRAME + 1 :][surface_mask]
    surface_charges = disclination_charges[EQUILIBRIUM_FRAME + 1 :][surface_mask]

    print(f"Wrote hexatic order to {HEXATIC_TXT}.")
    print(f"Wrote neighbor counts to {NEIGHBOR_COUNT_TXT}.")
    print(f"Wrote distribution to {DISTRIBUTION_TXT}.")
    print(f"Wrote plot to {FIGURE_FILE}.")
    print(f"Wrote OVITO file to {OUT_GSD}.")
    print("OVITO velocity.x stores |psi_6|.")
    print("OVITO velocity.y stores surface neighbor count.")
    print("OVITO velocity.z stores disclination charge q = 6 - neighbor_count.")
    print(
        "Distribution frames on shell "
        f"min={distribution_psi_abs.min():.6f}, "
        f"mean={distribution_psi_abs.mean():.6f}, "
        f"max={distribution_psi_abs.max():.6f}, "
        f"nonzero={len(distribution_psi_abs)}"
    )
    print(
        "Surface neighbor counts on shell "
        f"min={surface_neighbor_counts.min()}, "
        f"mean={surface_neighbor_counts.mean():.6f}, "
        f"max={surface_neighbor_counts.max()}, "
        f"particles={len(surface_neighbor_counts)}"
    )
    print(
        "Disclination charges on shell "
        f"q=+1 count={np.count_nonzero(surface_charges == 1)}, "
        f"q=0 count={np.count_nonzero(surface_charges == 0)}, "
        f"q=-1 count={np.count_nonzero(surface_charges == -1)}"
    )


if __name__ == "__main__":
    main()
