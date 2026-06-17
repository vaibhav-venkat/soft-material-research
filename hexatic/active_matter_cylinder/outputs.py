from pathlib import Path

from .config import (
    ACTIVE_GRID_DX,
    ACTIVE_GRID_DY,
    ACTIVE_GRID_DZ,
    ACTIVE_DATA_DIR,
    ACTIVE_FIELD_THETA_BINS,
    ACTIVE_FIELD_X_BINS,
    ACTIVE_IMAGE_DIR,
    ACTIVE_MOVIE_FPS,
    CYLINDER_PATHS,
    LOCAL_POCKET_RADIUS,
    ActiveMatterFields,
    CartesianFluxComparison,
)
from .cartesian_flux import (
    compute_cartesian_flux_comparison,
    plot_cartesian_flux_comparison,
)
from .fields import active_matter_field_series, save_active_matter_fields
from .movies import plot_active_matter_movies
from .radial_px import plot_radial_px_fields
from .vector_plots import plot_active_component_series, plot_active_x_balance_series


def plot_active_matter_fields(
    fields: ActiveMatterFields,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
) -> None:
    plot_active_component_series(fields, image_dir=image_dir)
    plot_active_x_balance_series(fields, image_dir=image_dir)
    plot_radial_px_fields(fields, image_dir=image_dir)


def write_active_matter_field_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    data_dir: str | Path = ACTIVE_DATA_DIR,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    pocket_radius: float = LOCAL_POCKET_RADIUS,
    n_x_bins: int = ACTIVE_FIELD_X_BINS,
    n_theta_bins: int = ACTIVE_FIELD_THETA_BINS,
    frame_index: int = -1,
    write_movies: bool = True,
    movie_fps: int = ACTIVE_MOVIE_FPS,
    dx: float = ACTIVE_GRID_DX,
    dy: float = ACTIVE_GRID_DY,
    dz: float = ACTIVE_GRID_DZ,
) -> CartesianFluxComparison:
    comparison = compute_cartesian_flux_comparison(
        input_gsd,
        pocket_radius=pocket_radius,
        dx=dx,
        dy=dy,
        dz=dz,
        frame_index=frame_index if frame_index != -1 else -2,
    )
    plot_cartesian_flux_comparison(comparison, image_dir=image_dir)
    return comparison
