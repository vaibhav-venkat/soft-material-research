from hexatic.active_matter_cylinder import (
    ACTIVE_IMAGE_DIR,
    write_active_matter_field_outputs,
)
from hexatic.chirality import CHIRALITY_DATA_DIR, write_chirality_outputs, ChiralityConfig

from .common import CYLINDER_PATHS
from .gsd_io import write_dynamic_values_gsd
from .plotting import (
    animate_outer_shell_xtheta_theta_velocity,
    animate_outer_shell_xtheta_x_velocity,
    plot_center_of_mass_series,
    plot_disclination_center_of_mass_series,
    plot_disclination_count_series,
    plot_dislocation_center_of_mass_series,
    plot_dislocation_count_series,
    plot_net_disclination_charge_series,
    plot_x_center_of_mass_velocity_series,
)

def main() -> None:
    write_dynamic_values_gsd(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.dynamic_values_gsd)
    print(f"Wrote OVITO dynamic values file to {CYLINDER_PATHS.dynamic_values_gsd}.")
    plot_center_of_mass_series(CYLINDER_PATHS.in_gsd, CYLINDER_PATHS.com_plot)
    print(f"Wrote center-of-mass plot to {CYLINDER_PATHS.com_plot}.")
    plot_x_center_of_mass_velocity_series(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.x_com_velocity_plot,
        shell_only=True,
    )
    print(
        "Wrote x center-of-mass velocity plot to "
        f"{CYLINDER_PATHS.x_com_velocity_plot}."
    )
    animate_outer_shell_xtheta_x_velocity(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.shell_xtheta_x_velocity_movie,
        start_frame=100,
    )
    print(
        "Wrote outer-shell x-theta x velocity movie to "
        f"{CYLINDER_PATHS.shell_xtheta_x_velocity_movie}."
    )
    animate_outer_shell_xtheta_theta_velocity(
        CYLINDER_PATHS.in_gsd,
        CYLINDER_PATHS.shell_xtheta_theta_velocity_movie,
        start_frame=100,
    )
    print(
        "Wrote outer-shell x-theta theta velocity movie to "
        f"{CYLINDER_PATHS.shell_xtheta_theta_velocity_movie}."
    )
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
    write_active_matter_field_outputs(
        CYLINDER_PATHS.in_gsd,
        coordinate_system="xrtheta"
    )
    print(
        "Wrote active matter flux density plots to "
        f"{ACTIVE_IMAGE_DIR / 'flux' / 'cartesian'}."
    )
    # write_chirality_outputs(
    #     CYLINDER_PATHS.in_gsd,
    #     config = ChiralityConfig(limit_disclination=True)
    # )
    # print(
    #     "Wrote disclination chirality fields to "
    #     f"{CHIRALITY_DATA_DIR / 'chirality_disclinations'}."
    # )
    print("OVITO position.x stores x")
    print("OVITO position.y stores theta")
    print("OVITO position.z stores r")
    print("OVITO velocity duplicates position")
