import numpy as np

if __package__:
    from hexatic import analysis as hx
    from hexatic.constants import sphere
else:
    import analysis as hx
    from constants import sphere


def main() -> None:
    calculator = hx.SphereHexaticCalculator(
        cavity_radius=sphere.CAVITY_RADIUS,
        shell_delta=sphere.SHELL_DELTA,
        n_neighbors=sphere.NEIGHBORS,
    )
    steps, psi = calculator.compute_hexatic_order_trajectory(sphere.IN_GSD)
    hx.save_hexatic_text(sphere.HEXATIC_TXT, steps, psi)

    count_steps, neighbor_counts = calculator.compute_neighbor_counts_trajectory(
        sphere.IN_GSD,
        neighbor_radius=sphere.NEIGHBOR_COUNT_RADIUS,
    )
    assert np.array_equal(count_steps, steps)
    hx.save_neighbor_count_text(sphere.NEIGHBOR_COUNT_TXT, steps, neighbor_counts)

    hexatic_table = hx.load_hexatic_text(sphere.HEXATIC_TXT)
    frame_indices = hexatic_table[:, 0].astype(int)
    psi_abs = hexatic_table[:, 5]

    bin_centers, probability_density, counts = hx.hexatic_probability_distribution(
        psi_abs,
        frame_indices,
        min_frame=sphere.EQUILIBRIUM_FRAME,
        bins=sphere.DISTRIBUTION_BINS,
        exclude_zeros=True,
    )
    hx.save_distribution_text(
        sphere.DISTRIBUTION_TXT,
        bin_centers,
        probability_density,
        counts,
    )
    hx.plot_hexatic_distribution(
        bin_centers,
        probability_density,
        title=f"hexatic distribution, frames > {sphere.EQUILIBRIUM_FRAME}",
        filename=sphere.FIGURE_FILE,
    )

    hx.write_hexatic_velocity_gsd(
        sphere.IN_GSD,
        sphere.OUT_GSD,
        sphere.HEXATIC_TXT,
        component=sphere.VELOCITY_COMPONENT,
        neighbor_counts=neighbor_counts,
        neighbor_component=sphere.NEIGHBOR_COUNT_COMPONENT,
    )

    selected_psi_abs = psi_abs[frame_indices > sphere.EQUILIBRIUM_FRAME]
    nonzero_selected_psi_abs = np.count_nonzero(selected_psi_abs)
    distribution_psi_abs = selected_psi_abs[selected_psi_abs > 0.0]
    selected_neighbor_counts = neighbor_counts[
        sphere.EQUILIBRIUM_FRAME + 1 :
    ].reshape(-1)
    shell_neighbor_counts = selected_neighbor_counts[selected_psi_abs > 0.0]
    print(f"Loaded {psi.shape[0]} frames and {psi.shape[1]} particles.")
    print(f"Used cavity radius R={sphere.CAVITY_RADIUS:.6f}.")
    print(f"Used wall repulsion cutoff={sphere.CUTOFF:.6f}.")
    print(f"Used neighbor-count radius={sphere.NEIGHBOR_COUNT_RADIUS:.6f}.")
    print(f"Used shell thickness={sphere.SHELL_THICKNESS:.6f}.")
    print(f"Used radial cutoff Delta={sphere.SHELL_DELTA:.6f}.")
    print(f"Wrote hexatic order to {sphere.HEXATIC_TXT}.")
    print(f"Wrote neighbor counts to {sphere.NEIGHBOR_COUNT_TXT}.")
    print(f"Wrote distribution to {sphere.DISTRIBUTION_TXT}.")
    print(f"Wrote plot to {sphere.FIGURE_FILE}.")
    print(f"Wrote OVITO file to {sphere.OUT_GSD}.")
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
