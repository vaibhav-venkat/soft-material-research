from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from hexatic.active_matter_cylinder import ACTIVE_MOVIE_FPS
    from hexatic.constants import cylinder
except ImportError:
    from active_matter_cylinder import ACTIVE_MOVIE_FPS
    from constants import cylinder


CYLINDER = cylinder.ANALYSIS
CYLINDER_PATHS = cylinder.PATHS
CYLINDER_SIM = cylinder.SIMULATION
CHIRALITY_DATA_DIR = Path(CYLINDER_PATHS.in_gsd).parent
CHIRALITY_IMAGE_DIR = Path(CYLINDER_PATHS.com_plot).parent / "chirality"
DISCLINATION_CHIRALITY_IMAGE_DIR = CHIRALITY_IMAGE_DIR / "disclinations"


@dataclass(frozen=True)
class ChiralityConfig:
    radial_bin_width: float = CYLINDER.particle_diameter
    n_x_bins: int = 100
    n_theta_bins: int = 72
    lag_frames: tuple[int, ...] = (5,)
    min_count: int = 4
    xtheta_min_count: int = 1
    screw_min_screw_rate: float = 0
    radius_epsilon: float = 1e-12
    movie_fps: int = ACTIVE_MOVIE_FPS
    limit_disclination: bool = False


@dataclass(frozen=True)
class ChiralityFields:
    steps: np.ndarray
    metric_names: tuple[str, ...]
    metric_labels: tuple[str, ...]
    lag_frames: tuple[int, ...]
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    radial_edges: np.ndarray
    radial_centers: np.ndarray
    global_values: np.ndarray
    radial_values: np.ndarray
    radial_counts: np.ndarray
    xtheta_values: np.ndarray
    xtheta_counts: np.ndarray


@dataclass(frozen=True)
class NeighborCountMatrix:
    steps: np.ndarray
    counts: np.ndarray
