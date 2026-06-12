from dataclasses import dataclass
import math
from pathlib import Path

try:
    from .project import IMAGE_OUTPUT_DIR, SPHERE_OUTPUT_DIR
except ImportError:
    from constants.project import IMAGE_OUTPUT_DIR, SPHERE_OUTPUT_DIR

N_PARTICLES = 1000
RHO = 0.2
SIGMA = 1.0
VOLUME = N_PARTICLES / RHO
PARTICLE_DIAMETER = SIGMA * 2.0 ** (1.0 / 6.0)
SEED = 1
KT = 1
GAMMA = 1
U0 = 100
TAU_R = 1
TIMESTEP = 1e-6
ROTATIONAL_DIFFUSION_PERIOD = 10
TRAJECTORY_WRITE_PERIOD = int(1e5)
RUN_STEPS = int(1e7)


@dataclass(frozen=True)
class SpherePaths:
    in_gsd: Path = SPHERE_OUTPUT_DIR / "trajectory.gsd"
    initial_gsd: Path = SPHERE_OUTPUT_DIR / "initial_mesh.gsd"
    hexatic_txt: Path = SPHERE_OUTPUT_DIR / "hexatic_order.txt"
    neighbor_count_txt: Path = SPHERE_OUTPUT_DIR / "surface_neighbor_counts.txt"
    distribution_txt: Path = SPHERE_OUTPUT_DIR / "hexatic_order_distribution.txt"
    figure_file: Path = IMAGE_OUTPUT_DIR / "hexatic_order_distribution.png"
    out_gsd: Path = SPHERE_OUTPUT_DIR / "trajectory_hexatic_velocity.gsd"


@dataclass(frozen=True)
class SphereAnalysisConfig:
    equilibrium_frame: int = 10
    neighbors: int = 6
    distribution_bins: int = 50
    velocity_component: int = 0
    neighbor_count_component: int = 1
    sigma: float = SIGMA
    particle_diameter: float = PARTICLE_DIAMETER
    volume: float = VOLUME
    cavity_radius: float = 1.4 * (VOLUME * 3.0 / 4.0 / math.pi) ** (1.0 / 3.0)
    cutoff: float = 2.0 ** (1.0 / 6.0) * SIGMA
    neighbor_count_radius: float = 2.0 ** (4.0 / 6.0) * SIGMA
    shell_thickness: float = 0.05 * SIGMA
    shell_delta: float = cutoff + shell_thickness


@dataclass(frozen=True)
class SphereSimulationConfig:
    n_particles: int = N_PARTICLES
    rho: float = RHO
    seed: int = SEED
    kt: int = KT
    gamma: int = GAMMA
    u0: int = U0
    tau_r: int = TAU_R
    timestep: float = TIMESTEP
    rotational_diffusion_period: int = ROTATIONAL_DIFFUSION_PERIOD
    trajectory_write_period: int = TRAJECTORY_WRITE_PERIOD
    run_steps: int = RUN_STEPS


PATHS = SpherePaths()
ANALYSIS = SphereAnalysisConfig()
SIMULATION = SphereSimulationConfig()

IN_GSD = PATHS.in_gsd
INITIAL_GSD = PATHS.initial_gsd
HEXATIC_TXT = PATHS.hexatic_txt
NEIGHBOR_COUNT_TXT = PATHS.neighbor_count_txt
DISTRIBUTION_TXT = PATHS.distribution_txt
FIGURE_FILE = PATHS.figure_file
OUT_GSD = PATHS.out_gsd

EQUILIBRIUM_FRAME = ANALYSIS.equilibrium_frame
NEIGHBORS = ANALYSIS.neighbors
DISTRIBUTION_BINS = ANALYSIS.distribution_bins
VELOCITY_COMPONENT = ANALYSIS.velocity_component
NEIGHBOR_COUNT_COMPONENT = ANALYSIS.neighbor_count_component
CAVITY_RADIUS = ANALYSIS.cavity_radius
CUTOFF = ANALYSIS.cutoff
NEIGHBOR_COUNT_RADIUS = ANALYSIS.neighbor_count_radius
SHELL_THICKNESS = ANALYSIS.shell_thickness
SHELL_DELTA = ANALYSIS.shell_delta
