from dataclasses import dataclass
import numpy as np
try:
    from .project import CYLINDER_OUTPUT_DIR, IMAGE_OUTPUT_DIR
except ImportError:
    from constants.project import CYLINDER_OUTPUT_DIR, IMAGE_OUTPUT_DIR

SIGMA = 1.0
PARTICLE_DIAMETER = SIGMA * 2.0 ** (1.0 / 6.0)

RHO = 0.2
X_RATIO = 4
BASELINE_RADIUS_DIAMETERS = 10.0
BASELINE_CYLINDER_RADIUS = BASELINE_RADIUS_DIAMETERS * PARTICLE_DIAMETER


def lx_for_radius(cylinder_radius: float, x_ratio: float = X_RATIO) -> float:
    return x_ratio * 2.0 * cylinder_radius


def n_particles_for_radius(
    cylinder_radius: float,
    lx: float | None = None,
    rho: float = RHO,
) -> int:
    if lx is None:
        lx = lx_for_radius(cylinder_radius)
    return int(round(rho * np.pi * cylinder_radius**2 * lx))


LX = lx_for_radius(BASELINE_CYLINDER_RADIUS)
REQUESTED_N_PARTICLES = n_particles_for_radius(BASELINE_CYLINDER_RADIUS)
SEED = 1
KT = 1
GAMMA = 1
U0 = 100
TAU_R = 1
TIMESTEP = 1e-6
# TIMESTEP = 1e-7
ROTATIONAL_DIFFUSION_PERIOD = 10
TRAJECTORY_WRITE_PERIOD = int(1e6)

RUN_STEPS = int(1e8)
# RUN_STEPS = int(1e7)
LATTICE_SPACING = 2.0 ** (1.0 / 6.0) + 0.5
TRANSVERSE_WALL_CUTOFF_MARGIN = 8.0
TRANSVERSE_PARTICLE_DIAMETER_MARGIN = 2.0
WALL_CLEARANCE_EPSILON = 1.0e-3
FROZEN_SHELL_DELTA = 0.1 * PARTICLE_DIAMETER
LAST_FRAME_LATTICE_AXIAL_GAP = PARTICLE_DIAMETER
LAST_FRAME_LATTICE_RADIAL_GAP = 2 * PARTICLE_DIAMETER
LAST_FRAME_LATTICE_SPACING = PARTICLE_DIAMETER
LAST_FRAME_RUN_STEPS = int(1e7)
NEIGHBOR_LIST_BUFFER = 0.4
INTERACTION_EPSILON_MULTIPLIER = 50.0
SHELL_VELOCITY_X_MARKER = 1.0
MIN_ANGULAR_CANDIDATES = 12
ANGULAR_CANDIDATE_MULTIPLIER = 4


@dataclass(frozen=True)
class CylinderPaths:
    disc_image_dir = IMAGE_OUTPUT_DIR / "disc"
    # in_gsd = CYLINDER_OUTPUT_DIR / "trajectory_cylinder.gsd"
    in_gsd = CYLINDER_OUTPUT_DIR / "trajectory_cylinder_new_rad.gsd"
    # initial_gsd = CYLINDER_OUTPUT_DIR / "initial_mesh.gsd"
    initial_gsd = CYLINDER_OUTPUT_DIR / "initial_mesh_cylinder_new_rad.gsd"
    flipped_initial_gsd = CYLINDER_OUTPUT_DIR / "initial_mesh_cylinder_flipped.gsd"
    flipped_gsd = CYLINDER_OUTPUT_DIR / "trajectory_cylinder_flipped.gsd"
    hexatic_txt = CYLINDER_OUTPUT_DIR / "cylinder_hexatic_order.txt"
    neighbor_count_txt = CYLINDER_OUTPUT_DIR / "cylinder_surface_neighbor_counts.txt"
    distribution_txt = CYLINDER_OUTPUT_DIR / "cylinder_hexatic_order_distribution.txt"
    figure_file = IMAGE_OUTPUT_DIR / "cylinder_hexatic_order_distribution.png"
    out_gsd = CYLINDER_OUTPUT_DIR / "trajectory_cylinder_hexatic_velocity.gsd"
    dynamic_values_gsd = CYLINDER_OUTPUT_DIR / "trajectory_cylinder_dynamic_values.gsd"
    com_plot = IMAGE_OUTPUT_DIR / "cylinder_center_of_mass_x_theta.png"
    x_com_velocity_plot = IMAGE_OUTPUT_DIR / "cylinder_x_com_velocity.png"
    theta_com_velocity_plot = IMAGE_OUTPUT_DIR / "cylinder_theta_com_velocity.png"
    shell_xtheta_x_velocity_movie = IMAGE_OUTPUT_DIR / "cylinder_shell_xtheta_x_velocity.mp4"
    shell_xtheta_theta_velocity_movie = (
        IMAGE_OUTPUT_DIR / "cylinder_shell_xtheta_theta_velocity.mp4"
    )
    disclination_com_plot = IMAGE_OUTPUT_DIR / "cylinder_disclination_com_x_theta.png"
    dislocation_com_plot = disc_image_dir / "dislocation_center_of_mass_x_theta.png"
    dislocation_count_plot = disc_image_dir / "dislocation_count.png"
    disclination_count_plot = disc_image_dir / "disclination_count.png"
    net_charge_plot = disc_image_dir / "net_disclination_charge.png"


@dataclass(frozen=True)
class CylinderAnalysisConfig:
    equilibrium_frame: int = 10
    neighbors: int = 6
    distribution_bins: int = 50
    hexatic_component: int = 0
    neighbor_count_component: int = 1
    disclination_charge_component: int = 2
    sigma: float = SIGMA
    particle_diameter: float = PARTICLE_DIAMETER
    cylinder_radius: float = BASELINE_CYLINDER_RADIUS
    # cylinder_radius: float = 10.0 * PARTICLE_DIAMETER
    wall_cutoff: float = 2.0 ** (1.0 / 6.0) * SIGMA
    min_neighbor_count_radius: float = wall_cutoff
    max_neighbor_count_radius: float = 2.0 ** (7.0 / 6.0) * SIGMA
    shell_delta: float = wall_cutoff
    frozen_shell_delta: float = FROZEN_SHELL_DELTA
    last_frame_lattice_axial_gap: float = LAST_FRAME_LATTICE_AXIAL_GAP
    last_frame_lattice_radial_gap: float = LAST_FRAME_LATTICE_RADIAL_GAP
    last_frame_lattice_spacing: float = LAST_FRAME_LATTICE_SPACING
    # neighbor_count_radius: float = 1.85 * SIGMA
    neighbor_count_radius: float = 1.7272 * PARTICLE_DIAMETER
    dislocation_pair_distance: float = 1.7272 * PARTICLE_DIAMETER
    # dislocation_pair_distance: float = 1.85 * PARTICLE_DIAMETER


@dataclass(frozen=True)
class CylinderSimulationConfig:
    requested_n_particles: int = REQUESTED_N_PARTICLES
    rho: float = RHO
    lx: float = LX
    x_ratio: int = X_RATIO
    seed: int = SEED
    kt: int = KT
    gamma: int = GAMMA
    u0: int = U0
    tau_r: int = TAU_R
    timestep: float = TIMESTEP
    rotational_diffusion_period: int = ROTATIONAL_DIFFUSION_PERIOD
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    run_steps: int = RUN_STEPS
    lattice_spacing: float = LATTICE_SPACING
    transverse_wall_cutoff_margin: float = TRANSVERSE_WALL_CUTOFF_MARGIN
    transverse_particle_diameter_margin: float = TRANSVERSE_PARTICLE_DIAMETER_MARGIN
    wall_clearance_epsilon: float = WALL_CLEARANCE_EPSILON
    last_frame_run_steps: int = LAST_FRAME_RUN_STEPS
    neighbor_list_buffer: float = NEIGHBOR_LIST_BUFFER
    interaction_epsilon_multiplier: float = INTERACTION_EPSILON_MULTIPLIER
    shell_velocity_x_marker: float = SHELL_VELOCITY_X_MARKER
    min_angular_candidates: int = MIN_ANGULAR_CANDIDATES
    angular_candidate_multiplier: int = ANGULAR_CANDIDATE_MULTIPLIER

    def lx_for_radius(self, cylinder_radius: float) -> float:
        return lx_for_radius(cylinder_radius, self.x_ratio)

    def n_particles_for_radius(self, cylinder_radius: float) -> int:
        return n_particles_for_radius(
            cylinder_radius,
            self.lx_for_radius(cylinder_radius),
            self.rho,
        )


PATHS = CylinderPaths()
ANALYSIS = CylinderAnalysisConfig()
SIMULATION = CylinderSimulationConfig()

DISC_IMAGE_DIR = PATHS.disc_image_dir
IN_GSD = PATHS.in_gsd
INITIAL_GSD = PATHS.initial_gsd
FLIPPED_INITIAL_GSD = PATHS.flipped_initial_gsd
FLIPPED_GSD = PATHS.flipped_gsd
HEXATIC_TXT = PATHS.hexatic_txt
NEIGHBOR_COUNT_TXT = PATHS.neighbor_count_txt
DISTRIBUTION_TXT = PATHS.distribution_txt
FIGURE_FILE = PATHS.figure_file
OUT_GSD = PATHS.out_gsd
DYNAMIC_VALUES_GSD = PATHS.dynamic_values_gsd
COM_PLOT = PATHS.com_plot
X_COM_VELOCITY_PLOT = PATHS.x_com_velocity_plot
THETA_COM_VELOCITY_PLOT = PATHS.theta_com_velocity_plot
SHELL_XTHETA_X_VELOCITY_MOVIE = PATHS.shell_xtheta_x_velocity_movie
SHELL_XTHETA_THETA_VELOCITY_MOVIE = PATHS.shell_xtheta_theta_velocity_movie
DISCLINATION_COM_PLOT = PATHS.disclination_com_plot
DISLOCATION_COM_PLOT = PATHS.dislocation_com_plot
DISLOCATION_COUNT_PLOT = PATHS.dislocation_count_plot
DISCLINATION_COUNT_PLOT = PATHS.disclination_count_plot
NET_CHARGE_PLOT = PATHS.net_charge_plot

EQUILIBRIUM_FRAME = ANALYSIS.equilibrium_frame
NEIGHBORS = ANALYSIS.neighbors
DISTRIBUTION_BINS = ANALYSIS.distribution_bins
HEXATIC_COMPONENT = ANALYSIS.hexatic_component
NEIGHBOR_COUNT_COMPONENT = ANALYSIS.neighbor_count_component
DISCLINATION_CHARGE_COMPONENT = ANALYSIS.disclination_charge_component
CYLINDER_RADIUS = ANALYSIS.cylinder_radius
WALL_CUTOFF = ANALYSIS.wall_cutoff
MIN_NEIGHBOR_COUNT_RADIUS = ANALYSIS.min_neighbor_count_radius
MAX_NEIGHBOR_COUNT_RADIUS = ANALYSIS.max_neighbor_count_radius
SHELL_DELTA = ANALYSIS.shell_delta
NEIGHBOR_COUNT_RADIUS = ANALYSIS.neighbor_count_radius
DISLOCATION_PAIR_DISTANCE = ANALYSIS.dislocation_pair_distance

LX = SIMULATION.lx
TRANSVERSE_WALL_CUTOFF_MARGIN = SIMULATION.transverse_wall_cutoff_margin
TRANSVERSE_PARTICLE_DIAMETER_MARGIN = (
    SIMULATION.transverse_particle_diameter_margin
)
WALL_CLEARANCE_EPSILON = SIMULATION.wall_clearance_epsilon
MIN_ANGULAR_CANDIDATES = SIMULATION.min_angular_candidates
ANGULAR_CANDIDATE_MULTIPLIER = SIMULATION.angular_candidate_multiplier
