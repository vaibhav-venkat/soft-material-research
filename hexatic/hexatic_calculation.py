from dataclasses import dataclass

import numpy as np

if __package__:
    from hexatic import analysis as hx
    from hexatic.constants import sphere
else:
    import analysis as hx
    from constants import sphere


@dataclass(frozen=True)
class SphereCalculationData:
    steps: np.ndarray
    psi: np.ndarray
    neighbor_counts: np.ndarray


def main() -> None:
    paths = sphere.PATHS
    analysis = sphere.ANALYSIS

    calculator = hx.SphereHexaticCalculator(
        cavity_radius=analysis.cavity_radius,
        shell_delta=analysis.shell_delta,
        n_neighbors=analysis.neighbors,
    )
    hexatic = calculator.compute_hexatic_order_trajectory(paths.in_gsd)
    hx.save_hexatic_text(paths.hexatic_txt, hexatic.steps, hexatic.psi)

    neighbors = calculator.compute_neighbor_counts_trajectory(
        paths.in_gsd,
        neighbor_radius=analysis.neighbor_count_radius,
    )
    assert np.array_equal(neighbors.steps, hexatic.steps)
    hx.save_neighbor_count_text(
        paths.neighbor_count_txt,
        hexatic.steps,
        neighbors.counts,
    )
    data = SphereCalculationData(
        steps=hexatic.steps,
        psi=hexatic.psi,
        neighbor_counts=neighbors.counts,
    )

    hexatic_table = hx.load_hexatic_text(paths.hexatic_txt)
    frame_indices = hexatic_table[:, 0].astype(int)
    psi_abs = hexatic_table[:, 5]

    distribution = hx.hexatic_probability_distribution(
        psi_abs,
        frame_indices,
        min_frame=analysis.equilibrium_frame,
        bins=analysis.distribution_bins,
        exclude_zeros=True,
    )
    hx.save_distribution_text(
        paths.distribution_txt,
        distribution.bin_centers,
        distribution.probability_density,
        distribution.counts,
    )
    hx.plot_hexatic_distribution(
        distribution.bin_centers,
        distribution.probability_density,
        title=f"hexatic distribution, frames > {analysis.equilibrium_frame}",
        filename=paths.figure_file,
    )

    hx.write_hexatic_velocity_gsd(
        paths.in_gsd,
        paths.out_gsd,
        paths.hexatic_txt,
        component=analysis.velocity_component,
        neighbor_counts=data.neighbor_counts,
        neighbor_component=analysis.neighbor_count_component,
    )

    selected_psi_abs = psi_abs[frame_indices > analysis.equilibrium_frame]
    nonzero_selected_psi_abs = np.count_nonzero(selected_psi_abs)
    distribution_psi_abs = selected_psi_abs[selected_psi_abs > 0.0]
    selected_neighbor_counts = data.neighbor_counts[
        analysis.equilibrium_frame + 1 :
    ].reshape(-1)
    shell_neighbor_counts = selected_neighbor_counts[selected_psi_abs > 0.0]
    print(f"Loaded {data.psi.shape[0]} frames and {data.psi.shape[1]} particles.")
    print(f"Used cavity radius R={analysis.cavity_radius:.6f}.")
    print(f"Used wall repulsion cutoff={analysis.cutoff:.6f}.")
    print(f"Used neighbor-count radius={analysis.neighbor_count_radius:.6f}.")
    print(f"Used shell thickness={analysis.shell_thickness:.6f}.")
    print(f"Used radial cutoff Delta={analysis.shell_delta:.6f}.")
    print(f"Wrote hexatic order to {paths.hexatic_txt}.")
    print(f"Wrote neighbor counts to {paths.neighbor_count_txt}.")
    print(f"Wrote distribution to {paths.distribution_txt}.")
    print(f"Wrote plot to {paths.figure_file}.")
    print(f"Wrote OVITO file to {paths.out_gsd}.")
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
