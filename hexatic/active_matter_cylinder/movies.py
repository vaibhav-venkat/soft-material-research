from pathlib import Path

from .config import ACTIVE_IMAGE_DIR, ACTIVE_MOVIE_FPS, ActiveMatterFields
from .movie_utils import _write_movie
from .radial_px import _write_radial_px_movie, plot_radial_px_fields
from .scalar_plots import (
    _draw_polar_radial_integral,
    _draw_polar_shell,
    _draw_rho_radial_integral,
    _draw_rho_shell,
)
from .vector_plots import (
    _draw_vector_density,
    plot_active_component_series,
    plot_active_x_balance_series,
)


def plot_active_matter_movies(
    fields: ActiveMatterFields,
    image_dir: str | Path = ACTIVE_IMAGE_DIR,
    fps: int = ACTIVE_MOVIE_FPS,
) -> None:
    image_path = Path(image_dir)
    plot_radial_px_fields(fields, image_dir=image_dir)
    _write_movie(
        fields,
        image_path / "rho" / "active_rho_shell.mp4",
        lambda fig, axis, frame_idx: _draw_rho_shell(fields, fig, axis, frame_idx),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "rho" / "active_rho_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_rho_radial_integral(
            fields,
            fig,
            axis,
            frame_idx,
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "polar" / "active_polar_shell.mp4",
        lambda fig, axis, frame_idx: _draw_polar_shell(fields, fig, axis, frame_idx),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "polar" / "active_polar_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_polar_radial_integral(
            fields,
            fig,
            axis,
            frame_idx,
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "flux" / "active_flux_shell.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.flux_cylindrical,
            fig,
            axis,
            frame_idx,
            "Outer-shell flux mean",
            r"$|\langle\dot{\mathbf{r}}\rangle|$",
            shell_only=True,
            cmap="plasma",
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "flux" / "active_flux_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.flux_cylindrical,
            fig,
            axis,
            frame_idx,
            "Radially averaged flux mean",
            r"$|R^{-1}\int dr\,\langle\dot{\mathbf{r}}\rangle_r|$",
            shell_only=False,
            cmap="plasma",
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "force_density" / "active_force_density_shell.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.force_density_cylindrical,
            fig,
            axis,
            frame_idx,
            "Outer-shell force mean",
            r"$|\langle\gamma^{-1}\mathbf{F}\rangle|$",
            shell_only=True,
            cmap="cividis",
        ),
        fps=fps,
    )
    _write_movie(
        fields,
        image_path / "force_density" / "active_force_density_radial_integral.mp4",
        lambda fig, axis, frame_idx: _draw_vector_density(
            fields,
            fields.force_density_cylindrical,
            fig,
            axis,
            frame_idx,
            "Radially averaged force mean",
            r"$|R^{-1}\int dr\,\langle\gamma^{-1}\mathbf{F}\rangle_r|$",
            shell_only=False,
            cmap="cividis",
        ),
        fps=fps,
    )
    _write_radial_px_movie(
        fields,
        image_path / "px_radius" / "active_px_radius_mean.mp4",
        statistic="mean",
        fps=fps,
    )
    _write_radial_px_movie(
        fields,
        image_path / "px_radius" / "active_px_radius_mean_abs.mp4",
        statistic="mean_abs",
        fps=fps,
    )
    _write_radial_px_movie(
        fields,
        image_path / "px_radius" / "active_px_radius_sum.mp4",
        statistic="sum",
        fps=fps,
    )
    plot_active_component_series(fields, image_dir=image_dir)
    plot_active_x_balance_series(fields, image_dir=image_dir)
