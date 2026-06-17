from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from hexatic.constants import cylinder
except ImportError:
    from constants import cylinder


CYLINDER = cylinder.ANALYSIS
CYLINDER_PATHS = cylinder.PATHS
CYLINDER_SIM = cylinder.SIMULATION
LOCAL_POCKET_RADIUS = 2.0 * CYLINDER.particle_diameter
ACTIVE_FIELD_X_BINS = 100
ACTIVE_FIELD_THETA_BINS = 72
ACTIVE_FLUX_PLOT_X_BINS = 32
ACTIVE_FLUX_PLOT_THETA_BINS = 18
ACTIVE_RADIAL_BIN_WIDTH = CYLINDER.particle_diameter
ACTIVE_RADIAL_MIN_MEAN_COUNT = 1
ACTIVE_MOVIE_FPS = 8
ACTIVE_DATA_DIR = Path(CYLINDER_PATHS.in_gsd).parent
ACTIVE_IMAGE_DIR = Path(CYLINDER_PATHS.com_plot).parent / "active"


@dataclass(frozen=True)
class ActiveMatterFields:
    steps: np.ndarray
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    coords: np.ndarray
    shell_mask: np.ndarray
    rho: np.ndarray
    active_direction: np.ndarray
    direction_cylindrical: np.ndarray
    polar_mean: np.ndarray
    polar_cylindrical: np.ndarray
    flux_cylindrical: np.ndarray
    force_density: np.ndarray
    force_density_cylindrical: np.ndarray
