import numpy as np

if __package__:
    from hexatic import analysis as hx
    from hexatic.constants import cylinder
else:
    import analysis as hx
    from constants import cylinder


def disclination_charges_from_counts(
    neighbor_counts: np.ndarray,
    psi: np.ndarray,
) -> np.ndarray:
    neighbor_counts = np.asarray(neighbor_counts, dtype=np.int64)
    psi = np.asarray(psi, dtype=np.complex128)
    assert neighbor_counts.shape == psi.shape

    charges = np.zeros_like(neighbor_counts, dtype=np.int64)
    surface_mask = np.abs(psi) > 0.0
    charges[surface_mask] = cylinder.NEIGHBORS - neighbor_counts[surface_mask]
    return charges


def main() -> None:

    calculator = hx.CylinderHexaticCalculator(
        cylinder_radius=cylinder.CYLINDER_RADIUS,
        shell_delta=cylinder.SHELL_DELTA,
        n_neighbors=cylinder.NEIGHBORS,
    )

    print(f"Used cylinder radius R={cylinder.CYLINDER_RADIUS:.6f}.")
    print(f"Used wall repulsion cutoff={cylinder.WALL_CUTOFF:.6f}.")
    print(f"Used radial shell cutoff Delta={cylinder.SHELL_DELTA:.6f}.")
    print(
        "Allowed neighbor-count radius range="
        f"({cylinder.MIN_NEIGHBOR_COUNT_RADIUS:.6f}, "
        f"{cylinder.MAX_NEIGHBOR_COUNT_RADIUS:.6f})."
    )
    print(f"Used neighbor-count radius={cylinder.NEIGHBOR_COUNT_RADIUS:.6f}.")
    steps, psi = calculator.compute_hexatic_order_trajectory(cylinder.IN_GSD)
    print(f"Loaded {psi.shape[0]} frames and {psi.shape[1]} particles.")

    hx.save_hexatic_text(cylinder.HEXATIC_TXT, steps, psi)

    count_steps, neighbor_counts = calculator.compute_neighbor_counts_trajectory(
        cylinder.IN_GSD,
        neighbor_radius=cylinder.NEIGHBOR_COUNT_RADIUS,
    )
    assert np.array_equal(count_steps, steps)
    hx.save_neighbor_count_text(cylinder.NEIGHBOR_COUNT_TXT, steps, neighbor_counts)

    disclination_charges = disclination_charges_from_counts(neighbor_counts, psi)

    hexatic_table = hx.load_hexatic_text(cylinder.HEXATIC_TXT)
    frame_indices = hexatic_table[:, 0].astype(int)
    psi_abs = hexatic_table[:, 5]

    bin_centers, probability_density, counts = hx.hexatic_probability_distribution(
        psi_abs,
        frame_indices,
        min_frame=cylinder.EQUILIBRIUM_FRAME,
        bins=cylinder.DISTRIBUTION_BINS,
        exclude_zeros=True,
    )
    hx.save_distribution_text(
        cylinder.DISTRIBUTION_TXT,
        bin_centers,
        probability_density,
        counts,
    )
    hx.plot_hexatic_distribution(
        bin_centers,
        probability_density,
        title=f"cylinder hexatic distribution, frames > {cylinder.EQUILIBRIUM_FRAME}",
        filename=cylinder.FIGURE_FILE,
    )

    hx.write_hexatic_velocity_gsd(
        cylinder.IN_GSD,
        cylinder.OUT_GSD,
        cylinder.HEXATIC_TXT,
        component=cylinder.HEXATIC_COMPONENT,
        neighbor_counts=neighbor_counts,
        neighbor_component=cylinder.NEIGHBOR_COUNT_COMPONENT,
        disclination_charges=disclination_charges,
        charge_component=cylinder.DISCLINATION_CHARGE_COMPONENT,
    )

    selected_psi_abs = psi_abs[frame_indices > cylinder.EQUILIBRIUM_FRAME]
    distribution_psi_abs = selected_psi_abs[selected_psi_abs > 0.0]
    surface_mask = np.abs(psi[cylinder.EQUILIBRIUM_FRAME + 1 :]) > 0.0
    surface_neighbor_counts = neighbor_counts[
        cylinder.EQUILIBRIUM_FRAME + 1 :
    ][surface_mask]
    surface_charges = disclination_charges[
        cylinder.EQUILIBRIUM_FRAME + 1 :
    ][surface_mask]

    print(f"Wrote hexatic order to {cylinder.HEXATIC_TXT}.")
    print(f"Wrote neighbor counts to {cylinder.NEIGHBOR_COUNT_TXT}.")
    print(f"Wrote distribution to {cylinder.DISTRIBUTION_TXT}.")
    print(f"Wrote plot to {cylinder.FIGURE_FILE}.")
    print(f"Wrote OVITO file to {cylinder.OUT_GSD}.")
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
