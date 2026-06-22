import argparse

from hexatic.active_matter_cylinder import (
    ACTIVE_DATA_DIR,
    ACTIVE_IMAGE_DIR,
    compute_shear_flux_decomposition_series,
    save_shear_flux_decomposition_series,
    write_active_matter_field_outputs,
)
from hexatic.chirality import CHIRALITY_DATA_DIR, write_chirality_outputs, ChiralityConfig

from .common import CYLINDER_PATHS
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
    plot_net_disclination_charge_series,
    plot_orientation_autocorrelation_diagnostics,
    plot_restart_comparison_velocity_series,
    plot_shell_px_change_cumsum,
    plot_shell_px_change_decomposition,
    plot_theta_center_of_mass_velocity_series,
    plot_velocity_series,
    plot_x_residual_diagnostics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--orientation-autocorrelation-plot",
        default=str(
            CYLINDER_PATHS.x_com_velocity_plot.with_name(
                "cylinder_orientation_autocorrelation_tau.png"
            )
        ),
        help="Output image for orientation autocorrelation and tau comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

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
    plot_velocity_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.x_com_velocity_plot,
        shell_only=True,
        relaxation_fit_mode="single",
        align_with_px=True,
    )
    print(
        "Wrote x center-of-mass velocity plot to "
        f"{CYLINDER_PATHS.x_com_velocity_plot}."
    )
    plot_shell_px_change_decomposition(filename=args.shell_px_change_plot)
    print(
        "Wrote shell P_x change decomposition plot to "
        f"{args.shell_px_change_plot}."
    )
    plot_shell_px_change_cumsum(filename=args.shell_px_change_cumsum_plot)
    print(
        "Wrote cumulative shell P_x change decomposition plot to "
        f"{args.shell_px_change_cumsum_plot}."
    )
    plot_x_residual_diagnostics(
        CYLINDER_PATHS.in_gsd,
        filename=args.x_residual_plot,
        shell_only=True,
    )
    print(f"Wrote x residual diagnostics plot to {args.x_residual_plot}.")
    plot_orientation_autocorrelation_diagnostics(
        filename=args.orientation_autocorrelation_plot,
        relaxation_fit_mode="single",
    )
    print(
        "Wrote orientation autocorrelation tau comparison plot to "
        f"{args.orientation_autocorrelation_plot}."
    )
    plot_restart_comparison_velocity_series(
        CYLINDER_PATHS.in_gsd,
        args.restart_comparison_gsd,
        args.restart_comparison_plot,
        restart_initial_gsd=args.restart_initial_gsd,
        shell_only=True,
    )
    print(
        "Wrote restart-aligned x center-of-mass velocity comparison plot to "
        f"{args.restart_comparison_plot}."
    )
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
    # plot_dislocation_count_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.dislocation_count_plot,
    # )
    # print(f"Wrote dislocation count plot to {CYLINDER_PATHS.dislocation_count_plot}.")
    # plot_disclination_count_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.disclination_count_plot,
    # )
    # print(f"Wrote disclination count plot to {CYLINDER_PATHS.disclination_count_plot}.")
    # plot_net_disclination_charge_series(
    #     CYLINDER_PATHS.in_gsd,
    #     CYLINDER_PATHS.neighbor_count_txt,
    #     CYLINDER_PATHS.net_charge_plot,
    # )
    # print(f"Wrote net disclination charge plot to {CYLINDER_PATHS.net_charge_plot}.")
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
