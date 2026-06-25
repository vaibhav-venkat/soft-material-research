import argparse

import numpy as np

from hexatic.active_matter_cylinder import (
    ACTIVE_DATA_DIR,
    ACTIVE_IMAGE_DIR,
    compute_shear_flux_decomposition_series,
    save_shear_flux_decomposition_series,
    write_active_matter_field_outputs,
)
from hexatic.chirality import (
    CHIRALITY_DATA_DIR,
    CHIRALITY_IMAGE_DIR,
    ChiralityConfig,
    compute_translation_chirality_trajectory,
    write_translation_chirality_measure_outputs,
    write_chirality_outputs,
)

from .common import CYLINDER, CYLINDER_PATHS
from .gsd_io import write_dynamic_values_gsd
from hexatic.lagged_prediction import (
    LAGGED_PREDICTION_DATA,
    LAGGED_PREDICTION_IMAGE_DIR,
    write_lagged_predictive_decomposition_outputs,
)
from .plotting import (
    animate_outer_shell_xtheta_theta_velocity,
    animate_outer_shell_xtheta_x_velocity,
    plot_center_of_mass_series,
    plot_disclination_center_of_mass_series,
    plot_disclination_count_series,
    plot_dislocation_center_of_mass_series,
    plot_dislocation_count_series,
    plot_initial_cylinder_spatial_maps,
    plot_initial_local_balance_maps,
    plot_net_disclination_charge_series,
    plot_orientation_autocorrelation_diagnostics,
    plot_polar_source_residual_series,
    plot_polar_tangent_population_series,
    plot_restart_comparison_velocity_series,
    plot_shell_polar_component_series,
    plot_shell_px_change_cumsum,
    plot_shell_px_change_decomposition,
    plot_theta_center_of_mass_velocity_series,
    plot_velocity_series,
    plot_x_residual_diagnostics,
)
from .velocity_polar_force_density import (
    plot_velocity_polar_force_density_series,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz-only",
        action="store_true",
        help="Write active-property and translation-chirality NPZ files only.",
    )
    parser.add_argument(
        "--input-gsd",
        default=str(CYLINDER_PATHS.in_gsd),
        help="Input trajectory for --npz-only.",
    )
    parser.add_argument(
        "--cylinder-radius",
        type=float,
        default=CYLINDER.cylinder_radius,
        help="Cylinder radius for --npz-only.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(CYLINDER_PATHS.in_gsd.parent),
        help="Output directory for --npz-only NPZ files.",
    )
    parser.add_argument(
        "--case-id",
        default="",
        help="Optional filename prefix for --npz-only outputs.",
    )
    parser.add_argument(
        "--shear-series-stride",
        type=int,
        default=1,
        help="Frame stride for --npz-only shear flux decomposition series.",
    )
    parser.add_argument(
        "--skip-shear-series",
        action="store_true",
        help="Skip --npz-only shear flux decomposition series output.",
    )
    parser.add_argument(
        "--x-velocity-fit",
        choices=("auto", "single", "double"),
        default="auto",
        help=(
            "Relaxation model for the x center-of-mass velocity plot. "
            "'single' fits V_inf + A exp(-t / tau); 'double' fits two "
            "exponential stages; 'auto' tries both and selects by AICc."
        ),
    )
    parser.add_argument(
        "--skip-lagged-prediction",
        action="store_true",
        help="Skip lagged predictive decomposition outputs.",
    )
    parser.add_argument(
        "--restart-comparison-gsd",
        default=str(
            CYLINDER_PATHS.in_gsd.parent
            / "restart_ensemble"
            / "trajectory_cylinder_C.gsd"
        ),
        help="Restart ensemble trajectory to compare against the regular trajectory.",
    )
    parser.add_argument(
        "--restart-comparison-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_x_com_velocity_restart_comparison.png"
            )
        ),
        help="Output image for the restart-aligned velocity comparison plot.",
    )
    parser.add_argument(
        "--restart-initial-gsd",
        default=None,
        help=(
            "Optional initial restart GSD used to determine the alignment step. "
            "When omitted, the matching restart_ensemble/initial file is used if present."
        ),
    )
    parser.add_argument(
        "--shell-px-change-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_shell_px_change_decomposition.png"
            )
        ),
        help="Output image for the shell P_x change decomposition plot.",
    )
    parser.add_argument(
        "--shell-px-change-cumsum-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_shell_px_change_cumsum.png"
            )
        ),
        help="Output image for the cumulative shell P_x change decomposition plot.",
    )
    parser.add_argument(
        "--x-residual-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_x_residual_diagnostics.png"
            )
        ),
        help="Output image for R_x = V_x - U0 P_x residual diagnostics.",
    )
    parser.add_argument(
        "--shell-polar-component-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_shell_polar_component_series.png"
            )
        ),
        help="Output image for shell P_x, P_theta, and P_r over time.",
    )
    parser.add_argument(
        "--polar-tangent-population-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_polar_tangent_population_series.png"
            )
        ),
        help="Output image for tangent polarization magnitude and angle over time.",
    )
    parser.add_argument(
        "--polar-source-residual-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_polar_source_residual_series.png"
            )
        ),
        help="Output image for tangent polarization source residual over time.",
    )
    parser.add_argument(
        "--orientation-autocorrelation-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_orientation_autocorrelation_tau.png"
            )
        ),
        help="Output image for orientation autocorrelation and tau comparison.",
    )
    parser.add_argument(
        "--initial-spatial-maps-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_initial_spatial_maps.png"
            )
        ),
        help="Output image for initial x-theta spatial maps.",
    )
    parser.add_argument(
        "--initial-spatial-frame",
        type=int,
        default=0,
        help="Active-matter frame index to use for initial x-theta spatial maps.",
    )
    parser.add_argument(
        "--initial-spatial-chirality-npz",
        default=str(CYLINDER_PATHS.in_gsd.parent / "chirality_fields.npz"),
        help="Saved chirality fields used for the initial chirality x-theta map.",
    )
    parser.add_argument(
        "--initial-spatial-stress-npz",
        default=str(
            CYLINDER_PATHS.in_gsd.parent / "shear_flux_decomposition_series.npz"
        ),
        help="Saved shear/stress fields used for the initial stress-divergence map.",
    )
    parser.add_argument(
        "--initial-local-balance-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_initial_local_balance_maps.png"
            )
        ),
        help="Output image for initial local active-stress balance maps.",
    )
    parser.add_argument(
        "--velocity-polar-force-density-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_velocity_polar_force_density_series.png"
            )
        ),
        help=(
            "Output image for stacked x/theta velocity, U0P, and direct "
            "wall-plus-particle force density series."
        ),
    )
    parser.add_argument(
        "--translation-chirality-radius",
        type=float,
        default=CYLINDER.neighbor_count_radius,
        help="Neighborhood radius for translation chirality.",
    )
    parser.add_argument(
        "--translation-chirality-npz",
        default=str(CYLINDER_PATHS.in_gsd.parent / "translation_chirality_fields.npz"),
        help="Output NPZ for translation chirality fields.",
    )
    parser.add_argument(
        "--translation-chirality-plot",
        default=str(CHIRALITY_IMAGE_DIR / "translation_chirality_bond_mean.png"),
        help="Output plot for mean shell bond translation chirality.",
    )
    parser.add_argument(
        "--translation-chirality-measure-gsd",
        default=str(CYLINDER_PATHS.in_gsd.parent / "chirality_measure.gsd"),
        help="Output GSD storing per-particle translation chirality in velocity.x.",
    )
    return parser.parse_args()


def write_npz_only_outputs(
    input_gsd,
    cylinder_radius: float,
    output_dir,
    case_id: str = "",
    shear_series_stride: int = 1,
    skip_shear_series: bool = False,
) -> None:
    from pathlib import Path

    import gsd.hoomd

    from hexatic.active_matter_cylinder import (
        ACTIVE_FIELD_THETA_BINS,
        ACTIVE_FIELD_X_BINS,
        ACTIVE_FLUX_PLOT_THETA_BINS,
        ACTIVE_GRID_DX,
        ACTIVE_GRID_DY,
        ACTIVE_GRID_DZ,
        LOCAL_POCKET_RADIUS,
        active_matter_field_series,
        compute_cartesian_flux_comparison,
        compute_shear_flux_decomposition,
        compute_shear_flux_decomposition_series,
        save_active_matter_fields,
        save_cartesian_flux_comparison,
        save_shear_flux_decomposition,
        save_shear_flux_decomposition_series,
    )
    from hexatic.chirality import shell_bond_translation_chirality_series

    input_path = Path(input_gsd)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"{case_id}_" if case_id else ""

    fields = active_matter_field_series(
        input_path,
        pocket_radius=LOCAL_POCKET_RADIUS,
        n_x_bins=ACTIVE_FIELD_X_BINS,
        n_theta_bins=ACTIVE_FIELD_THETA_BINS,
        cylinder_radius=cylinder_radius,
    )
    save_active_matter_fields(
        fields,
        output_path / f"{prefix}active_matter_fields.npz",
        pocket_radius=LOCAL_POCKET_RADIUS,
    )

    comparison = compute_cartesian_flux_comparison(
        input_path,
        pocket_radius=LOCAL_POCKET_RADIUS,
        dx=ACTIVE_GRID_DX,
        dy=ACTIVE_GRID_DY,
        dz=ACTIVE_GRID_DZ,
        frame_index=-2,
        cylinder_radius=cylinder_radius,
    )
    save_cartesian_flux_comparison(
        comparison,
        output_path / f"{prefix}cartesian_flux_comparison.npz",
    )

    shear = compute_shear_flux_decomposition(
        input_path,
        pocket_radius=LOCAL_POCKET_RADIUS,
        dx=ACTIVE_GRID_DX,
        dr=ACTIVE_GRID_DY,
        n_theta_bins=ACTIVE_FLUX_PLOT_THETA_BINS,
        frame_index=-2,
        cylinder_radius=cylinder_radius,
    )
    save_shear_flux_decomposition(
        shear,
        output_path / f"{prefix}shear_flux_decomposition.npz",
    )

    if not skip_shear_series:
        if shear_series_stride <= 0:
            raise ValueError("shear_series_stride must be positive.")
        with gsd.hoomd.open(name=str(input_path), mode="r") as source:
            frame_indices = range(0, len(source), shear_series_stride)
        shear_series = compute_shear_flux_decomposition_series(
            input_path,
            frame_indices=frame_indices,
            pocket_radius=LOCAL_POCKET_RADIUS,
            dx=ACTIVE_GRID_DX,
            dr=ACTIVE_GRID_DY,
            n_theta_bins=ACTIVE_FLUX_PLOT_THETA_BINS,
            cylinder_radius=cylinder_radius,
        )
        save_shear_flux_decomposition_series(
            shear_series,
            output_path / f"{prefix}shear_flux_decomposition_series.npz",
        )

    translation_chirality = compute_translation_chirality_trajectory(
        input_path,
        neighborhood_radius=CYLINDER.neighbor_count_radius,
    )
    np.savez_compressed(
        output_path / f"{prefix}translation_chirality_fields.npz",
        steps=translation_chirality.steps,
        chirality=translation_chirality.chirality,
        neighborhood_radius=CYLINDER.neighbor_count_radius,
    )

    shell_steps, shell_values, shell_counts = shell_bond_translation_chirality_series(
        input_path,
        neighborhood_radius=CYLINDER.neighbor_count_radius,
        cylinder_radius=cylinder_radius,
        shell_delta=CYLINDER.shell_delta,
    )
    np.savez_compressed(
        output_path / f"{prefix}shell_bond_translation_chirality.npz",
        steps=shell_steps,
        mean_abs_bond_translation_chirality=shell_values,
        bond_counts=shell_counts,
        neighborhood_radius=CYLINDER.neighbor_count_radius,
        cylinder_radius=cylinder_radius,
    )


def main() -> None:
    args = _parse_args()
    if args.npz_only:
        write_npz_only_outputs(
            input_gsd=args.input_gsd,
            cylinder_radius=args.cylinder_radius,
            output_dir=args.output_dir,
            case_id=args.case_id,
            shear_series_stride=args.shear_series_stride,
            skip_shear_series=args.skip_shear_series,
        )
        return

    # write_active_matter_field_outputs(
    #     CYLINDER_PATHS.in_gsd,
    #     coordinate_system="xrtheta"
    # )
    # print(
    #     "Wrote active matter fields to "
    #     f"{ACTIVE_DATA_DIR / 'active_matter_fields.npz'}."
    # )
    # print(
    #     "Wrote active matter flux density plots to "
    #     f"{ACTIVE_IMAGE_DIR / 'flux'}."
    # )
    # write_dynamic_values_gsd(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.dynamic_values_gsd)
    # print(f"Wrote OVITO dynamic values file to {CYLINDER_PATHS.dynamic_values_gsd}.")
    # plot_center_of_mass_series(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.com_plot)
    # print(f"Wrote center-of-mass plot to {CYLINDER_PATHS.com_plot}.")
    # plot_velocity_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.x_com_velocity_plot,
    #     shell_only=True,
    #     relaxation_fit_mode="single",
    #     align_with_px=True,
    # )
    # print(
    #     "Wrote x center-of-mass velocity plot to "
    #     f"{CYLINDER_PATHS.x_com_velocity_plot}."
    # )
    # translation_chirality = compute_translation_chirality_trajectory(
    #     CYLINDER_PATHS.in_gsd,
    #     neighborhood_radius=args.translation_chirality_radius,
    # )
    # np.savez_compressed(
    #     args.translation_chirality_npz,
    #     steps=translation_chirality.steps,
    #     chirality=translation_chirality.chirality,
    #     neighborhood_radius=args.translation_chirality_radius,
    # )
    # print(f"Wrote translation chirality fields to {args.translation_chirality_npz}.")
    write_translation_chirality_measure_outputs(
        CYLINDER_PATHS.in_gsd,
        plot_png=args.translation_chirality_plot,
        measure_gsd=args.translation_chirality_measure_gsd,
        neighborhood_radius=args.translation_chirality_radius,
    )
    print(
        "Wrote translation chirality bond mean plot to "
        f"{args.translation_chirality_plot}."
    )
    print(
        "Wrote translation chirality measure GSD to "
        f"{args.translation_chirality_measure_gsd}."
    )
    # plot_shell_px_change_decomposition(filename=args.shell_px_change_plot)
    # print(
    #     "Wrote shell P_x change decomposition plot to "
    #     f"{args.shell_px_change_plot}."
    # )
    # plot_shell_px_change_cumsum(filename=args.shell_px_change_cumsum_plot)
    # print(
    #     "Wrote cumulative shell P_x change decomposition plot to "
    #     f"{args.shell_px_change_cumsum_plot}."
    # # )
    # plot_x_residual_diagnostics(
    #     CYLINDER_PATHS.in_gsd,
    #     filename=args.x_residual_plot,
    #     shell_only=True,
    # )
    # print(f"Wrote x residual diagnostics plot to {args.x_residual_plot}.")
    # plot_shell_polar_component_series(filename=args.shell_polar_component_plot)
    # print(
    #     "Wrote shell polar component plot to "
    #     f"{args.shell_polar_component_plot}."
    # )
    # plot_velocity_polar_force_density_series(
    #     filename=args.velocity_polar_force_density_plot,
    #     shell_only=True,
    # )
    # print(
    #     "Wrote velocity, U0P, and direct force-density plot to "
    #     f"{args.velocity_polar_force_density_plot}."
    # )
    # plot_polar_tangent_population_series(
    #     filename=args.polar_tangent_population_plot
    # )
    # print(
    #     "Wrote tangent polarization population plot to "
    #     f"{args.polar_tangent_population_plot}."
    # )
    # plot_polar_source_residual_series(filename=args.polar_source_residual_plot)
    # print(
    #     "Wrote tangent polarization source residual plot to "
    #     f"{args.polar_source_residual_plot}."
    # )
    # plot_orientation_autocorrelation_diagnostics(
    #     filename=args.orientation_autocorrelation_plot,
    #     relaxation_fit_mode="single",
    # )
    # print(
    #     "Wrote orientation autocorrelation tau comparison plot to "
    #     f"{args.orientation_autocorrelation_plot}."
    # )
    # plot_initial_cylinder_spatial_maps(
    #     filename=args.initial_spatial_maps_plot,
    #     frame_idx=args.initial_spatial_frame,
    #     chirality_npz=args.initial_spatial_chirality_npz,
    #     stress_npz=args.initial_spatial_stress_npz,
    # )
    # print(f"Wrote initial cylinder spatial maps to {args.initial_spatial_maps_plot}.")
    # plot_initial_local_balance_maps(
    #     filename=args.initial_local_balance_plot,
    #     frame_idx=args.initial_spatial_frame,
    #     chirality_npz=args.initial_spatial_chirality_npz,
    #     stress_npz=args.initial_spatial_stress_npz,
    # )
    # print(
    #     "Wrote initial local active-stress balance maps to "
    #     f"{args.initial_local_balance_plot}."
    # )
    # plot_restart_comparison_velocity_series(
    #     CYLINDER_PATHS.in_gsd,
    #     args.restart_comparison_gsd,
    #     args.restart_comparison_plot,
    #     restart_initial_gsd=args.restart_initial_gsd,
    #     shell_only=True,
    # )
    # print(
    #     "Wrote restart-aligned x center-of-mass velocity comparison plot to "
    #     f"{args.restart_comparison_plot}."
    # )
    # if not args.skip_lagged_prediction:
    #     # shear_series_file = ACTIVE_DATA_DIR / "shear_flux_decomposition_series.npz"
    #     # shear_series = compute_shear_flux_decomposition_series(CYLINDER_PATHS.in_gsd)
    #     # save_shear_flux_decomposition_series(shear_series, shear_series_file)
    #     # print(f"Wrote shear flux decomposition time series to {shear_series_file}.")

    #     write_lagged_predictive_decomposition_outputs(CYLINDER_PATHS.in_gsd)
    #     print(
    #         "Wrote lagged predictive decomposition to "
    #         f"{LAGGED_PREDICTION_DATA}."
    #     )
    #     print(
    #         "Wrote lagged predictive decomposition plots to "
    #         f"{LAGGED_PREDICTION_IMAGE_DIR}."
    #     )
    # plot_theta_center_of_mass_velocity_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.theta_com_velocity_plot,
    #     shell_only=True,
    # )
    # print(
    #     "Wrote theta center-of-mass velocity plot to "
    #     f"{CYLINDER_PATHS.theta_com_velocity_plot}."
    # )
    # animate_outer_shell_xtheta_x_velocity(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.shell_xtheta_x_velocity_movie,
    #     start_frame=0,
    # )
    # print(
    #     "Wrote outer-shell x-theta x velocity movie to "
    #     f"{CYLINDER_PATHS.shell_xtheta_x_velocity_movie}."
    # )
    # animate_outer_shell_xtheta_theta_velocity(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.shell_xtheta_theta_velocity_movie,
    #     start_frame=0,
    # )
    # print(
    #     "Wrote outer-shell x-theta theta velocity movie to "
    #     f"{CYLINDER_PATHS.shell_xtheta_theta_velocity_movie}."
    # )
    # plot_disclination_center_of_mass_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.disclination_com_plot,
    # )
    # print(
    #     "Wrote disclination center-of-mass plot to "
    #     f"{CYLINDER_PATHS.disclination_com_plot}."
    # )
    # plot_dislocation_center_of_mass_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.dislocation_com_plot,
    # )
    # print(
    #     "Wrote dislocation center-of-mass plot to "
    #     f"{CYLINDER_PATHS.dislocation_com_plot}."
    # )
    plot_dislocation_count_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.dislocation_count_plot,
    )
    print(f"Wrote dislocation count plot to {CYLINDER_PATHS.dislocation_count_plot}.")
    plot_disclination_count_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.disclination_count_plot,
    )
    print(f"Wrote disclination count plot to {CYLINDER_PATHS.disclination_count_plot}.")
    plot_net_disclination_charge_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.neighbor_count_txt,
        CYLINDER_PATHS.net_charge_plot,
    )
    print(f"Wrote net disclination charge plot to {CYLINDER_PATHS.net_charge_plot}.")
    # write_chirality_outputs(
    #     CYLINDER_PATHS.in_gsd,
    #     config = ChiralityConfig(limit_disclination=True)
    # )
    # print(
    #     "Wrote disclination chirality fields to "
    #     f"{CHIRALITY_DATA_DIR / 'chirality_disclinations'}."
    # )
    # print("OVITO position.x stores x")
    # print("OVITO position.y stores theta")
    # print("OVITO position.z stores r")
    # print("OVITO velocity duplicates position")
