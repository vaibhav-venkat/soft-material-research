import argparse
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class CylinderCalculationOutputs:
    hexatic_txt: Path
    neighbor_count_txt: Path
    out_gsd: Path | None


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


def calculate_cylinder_hexatic(
    input_gsd: str | Path,
    cylinder_radius: float,
    output_dir: str | Path | None = None,
    case_id: str = "",
    write_ovito_gsd: bool = True,
) -> CylinderCalculationOutputs:
    paths = cylinder.PATHS
    analysis = cylinder.ANALYSIS
    input_gsd = Path(input_gsd)
    prefix = f"{case_id}_" if case_id else ""
    if output_dir is None:
        hexatic_txt = Path(paths.hexatic_txt)
        neighbor_count_txt = Path(paths.neighbor_count_txt)
        out_gsd = Path(paths.out_gsd) if write_ovito_gsd else None
    else:
        output_path = Path(output_dir)
        hexatic_txt = output_path / f"{prefix}hexatic_order.txt"
        neighbor_count_txt = output_path / f"{prefix}neighbor_counts.txt"
        out_gsd = (
            output_path / f"{prefix}hexatic_velocity.gsd"
            if write_ovito_gsd
            else None
        )

    calculator = hx.CylinderHexaticCalculator(
        cylinder_radius=cylinder_radius,
        shell_delta=analysis.shell_delta,
        n_neighbors=analysis.neighbors,
    )

    print(f"Reading trajectory from {input_gsd}.")
    print(f"Used cylinder radius R={cylinder_radius:.6f}.")
    print(f"Used wall repulsion cutoff={analysis.wall_cutoff:.6f}.")
    print(f"Used radial shell cutoff Delta={analysis.shell_delta:.6f}.")
    print(
        "Allowed neighbor-count radius range="
        f"({analysis.min_neighbor_count_radius:.6f}, "
        f"{analysis.max_neighbor_count_radius:.6f})."
    )
    print(f"Used neighbor-count radius={analysis.neighbor_count_radius:.6f}.")
    hexatic = calculator.compute_hexatic_order_trajectory(input_gsd)
    print(f"Loaded {hexatic.psi.shape[0]} frames and {hexatic.psi.shape[1]} particles.")

    hx.save_hexatic_text(hexatic_txt, hexatic.steps, hexatic.psi)

    neighbors = calculator.compute_neighbor_counts_trajectory(
        input_gsd,
        neighbor_radius=analysis.neighbor_count_radius,
    )
    assert np.array_equal(neighbors.steps, hexatic.steps)
    hx.save_neighbor_count_text(
        neighbor_count_txt,
        hexatic.steps,
        neighbors.counts,
    )

    disclination_charges = disclination_charges_from_counts(
        neighbors.counts,
        hexatic.psi,
    )
    dislocation_particles = hx.identify_dislocation_particles_trajectory(
        input_gsd,
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

    if out_gsd is not None:
        hx.write_hexatic_velocity_gsd(
            input_gsd,
            out_gsd,
            hexatic_txt,
            component=analysis.hexatic_component,
            neighbor_counts=data.neighbor_counts,
            neighbor_component=analysis.neighbor_count_component,
            disclination_charges=data.disclination_charges,
            charge_component=analysis.disclination_charge_component,
            dislocation_particles=data.dislocation_particles,
        )

    frame_idx = min(90, data.neighbor_counts.shape[0] - 1)
    surface_mask = np.abs(data.psi[frame_idx]) > 0.0
    surface_neighbor_counts = data.neighbor_counts[frame_idx][surface_mask]
    surface_charges = data.disclination_charges[
        frame_idx
    ][surface_mask]
    surface_dislocations = data.dislocation_particles[
        frame_idx
    ][surface_mask]

    print(f"Wrote hexatic order to {hexatic_txt}.")
    print(f"Wrote neighbor counts to {neighbor_count_txt}.")
    if out_gsd is not None:
        print(f"Wrote OVITO file to {out_gsd}.")
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
    if len(surface_neighbor_counts):
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
    else:
        print(f"No shell particles found in frame {frame_idx}.")
    return CylinderCalculationOutputs(
        hexatic_txt=hexatic_txt,
        neighbor_count_txt=neighbor_count_txt,
        out_gsd=out_gsd,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-gsd", default=str(cylinder.PATHS.in_gsd))
    parser.add_argument(
        "--cylinder-radius",
        type=float,
        default=cylinder.ANALYSIS.cylinder_radius,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--case-id", default="")
    parser.add_argument("--no-ovito-gsd", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    calculate_cylinder_hexatic(
        input_gsd=args.input_gsd,
        cylinder_radius=args.cylinder_radius,
        output_dir=args.output_dir,
        case_id=args.case_id,
        write_ovito_gsd=not args.no_ovito_gsd,
    )


if __name__ == "__main__":
    main()
