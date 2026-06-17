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
GEOMETRIC_CHIRALITY_DATA_DIR = Path(CYLINDER_PATHS.in_gsd).parent
GEOMETRIC_CHIRALITY_IMAGE_DIR = (
    Path(CYLINDER_PATHS.com_plot).parent / "chirality" / "geometric"
)
GEOMETRIC_XTHETA_X_BINS = 32
GEOMETRIC_XTHETA_THETA_BINS = 18

GEOMETRIC_METRIC_NAMES = ("ccm", "chi_strand", "chi_trajectory")
GEOMETRIC_METRIC_LABELS = ("CCM", "strand chi", "trajectory chi")


@dataclass(frozen=True)
class GeometricChiralityConfig:
    radial_bin_width: float = CYLINDER.particle_diameter
    n_x_bins: int = GEOMETRIC_XTHETA_X_BINS
    n_theta_bins: int = GEOMETRIC_XTHETA_THETA_BINS
    min_count: int = 3
    chi_min_ordered_points: int = 5
    n_strand_theta_sectors: int = 24
    trajectory_lag_frames: int = 3
    radius_epsilon: float = 1e-12
    denominator_epsilon: float = 1e-12
    movie_fps: int = ACTIVE_MOVIE_FPS


@dataclass(frozen=True)
class GeometricChiralityFields:
    steps: np.ndarray
    metric_names: tuple[str, ...]
    metric_labels: tuple[str, ...]
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    radial_edges: np.ndarray
    radial_centers: np.ndarray
    global_values: np.ndarray
    global_counts: np.ndarray
    global_numerators: np.ndarray
    global_denominators: np.ndarray
    radial_values: np.ndarray
    radial_counts: np.ndarray
    radial_numerators: np.ndarray
    radial_denominators: np.ndarray
    xtheta_values: np.ndarray
    xtheta_counts: np.ndarray
    xtheta_numerators: np.ndarray
    xtheta_denominators: np.ndarray


@dataclass(frozen=True)
class _TrajectoryData:
    steps: np.ndarray
    positions: np.ndarray
    unwrapped_positions: np.ndarray
    coords: np.ndarray
    masses: np.ndarray
    box_lengths_x: np.ndarray
    x_edges: np.ndarray
    x_centers: np.ndarray
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    radial_edges: np.ndarray
    radial_centers: np.ndarray
