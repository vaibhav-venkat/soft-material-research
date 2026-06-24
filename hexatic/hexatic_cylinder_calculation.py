from dataclasses import dataclass

import numpy as np

if __package__:
    from hexatic import analysis as hx
    from hexatic.constants import cylinder
else:
    import analysis as hx
    from constants import cylinder


@dataclass(frozen=True)
class CylinderCalculationData:
    steps: np.ndarray
    psi: np.ndarray
    neighbor_counts: np.ndarray
    disclination_charges: np.ndarray
    dislocation_particles: np.ndarray


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
    paths = cylinder.PATHS
    analysis = cylinder.ANALYSIS
    analysis_cylinder_radius = analysis.cylinder_radius
    # analysis_cylinder_radius = (
    #     60.0 * analysis.particle_diameter / (2.0 * np.pi)
    # )

    calculator = hx.CylinderHexaticCalculator(
        # cylinder_radius=analysis.cylinder_radius,
        cylinder_radius=analysis_cylinder_radius,
        shell_delta=analysis.shell_delta,
        n_neighbors=analysis.neighbors,
    )

    # print(f"Used cylinder radius R={analysis.cylinder_radius:.6f}.")
    print(f"Used cylinder radius R={analysis_cylinder_radius:.6f}.")
    print(f"Used wall repulsion cutoff={analysis.wall_cutoff:.6f}.")
    print(f"Used radial shell cutoff Delta={analysis.shell_delta:.6f}.")
    print(
        "Allowed neighbor-count radius range="
        f"({analysis.min_neighbor_count_radius:.6f}, "
        f"{analysis.max_neighbor_count_radius:.6f})."
    )
    print(f"Used neighbor-count radius={analysis.neighbor_count_radius:.6f}.")
    hexatic = calculator.compute_hexatic_order_trajectory(paths.in_gsd)
    print(f"Loaded {hexatic.psi.shape[0]} frames and {hexatic.psi.shape[1]} particles.")

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

    disclination_charges = disclination_charges_from_counts(
        neighbors.counts,
        hexatic.psi,
    )
    dislocation_particles = hx.identify_dislocation_particles_trajectory(
        paths.in_gsd,
        disclination_charges,
        pair_distance=analysis.dislocation_pair_distance,
    )
    data = CylinderCalculationData(
        steps=hexatic.steps,
        psi=hexatic.psi,
        neighbor_counts=neighbors.counts,
        disclination_charges=disclination_charges,
        dislocation_particles=dislocation_particles,
    )

    hexatic_table = hx.load_hexatic_text(paths.hexatic_txt)
    frame_indices = hexatic_table[:, 0].astype(int)
    psi_abs = hexatic_table[:, 5]

    # distribution = hx.hexatic_probability_distribution(
    #     psi_abs,
    #     frame_indices,
    #     min_frame=analysis.equilibrium_frame,
    #     bins=analysis.distribution_bins,
    #     exclude_zeros=True,
    # )
    # hx.save_distribution_text(
    #     paths.distribution_txt,
    #     distribution.bin_centers,
    #     distribution.probability_density,
    #     distribution.counts,
    # )
    # hx.plot_hexatic_distribution(
    #     distribution.bin_centers,
    #     distribution.probability_density,
    #     title=f"cylinder hexatic distribution, frames > {analysis.equilibrium_frame}",
    #     filename=paths.figure_file,
    # )

    hx.write_hexatic_velocity_gsd(
        paths.in_gsd,
        paths.out_gsd,
        paths.hexatic_txt,
        component=analysis.hexatic_component,
        neighbor_counts=data.neighbor_counts,
        neighbor_component=analysis.neighbor_count_component,
        disclination_charges=data.disclination_charges,
        charge_component=analysis.disclination_charge_component,
        dislocation_particles=data.dislocation_particles,
    )

    selected_psi_abs = psi_abs[frame_indices > analysis.equilibrium_frame]
    frame_idx = 90
    distribution_psi_abs = selected_psi_abs[selected_psi_abs > 0.0]
    surface_mask = np.abs(data.psi[frame_idx]) > 0.0
    print(data.neighbor_counts)
    surface_neighbor_counts = data.neighbor_counts[frame_idx][surface_mask]
    surface_charges = data.disclination_charges[
        frame_idx
    ][surface_mask]
    surface_dislocations = data.dislocation_particles[
        frame_idx
    ][surface_mask]

    print(f"Wrote hexatic order to {paths.hexatic_txt}.")
    print(f"Wrote neighbor counts to {paths.neighbor_count_txt}.")
    print(f"Wrote distribution to {paths.distribution_txt}.")
    print(f"Wrote plot to {paths.figure_file}.")
    print(f"Wrote OVITO file to {paths.out_gsd}.")
    print("OVITO velocity.x stores |psi_6|.")
    print("OVITO velocity.y stores surface neighbor count.")
    print("OVITO velocity.z stores disclination charge q = 6 - neighbor_count.")
    print("OVITO orientation.w stores dislocation flag.")
    print(
        "Used dislocation pair distance="
        f"{analysis.dislocation_pair_distance:.6f}."
    )
    # print(
    #     "Distribution frames on shell "
    #     f"min={distribution_psi_abs.min():.6f}, "
    #     f"mean={distribution_psi_abs.mean():.6f}, "
    #     f"max={distribution_psi_abs.max():.6f}, "
    #     f"nonzero={len(distribution_psi_abs)}"
    # )
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
    print(
        "Dislocation-paired shell "
        f"count={np.count_nonzero(surface_dislocations)}"
    )


if __name__ == "__main__":
    main()
