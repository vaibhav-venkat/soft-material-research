from pathlib import Path

import gsd.hoomd
import numpy as np

from .common import _cylinder_coords, _cylinder_shell_mask, _load_neighbor_count_matrix
from .compute import compute_chirality_fields
from .config import (
    CHIRALITY_DATA_DIR,
    CHIRALITY_IMAGE_DIR,
    CYLINDER,
    CYLINDER_PATHS,
    DISCLINATION_CHIRALITY_IMAGE_DIR,
    ChiralityConfig,
    ChiralityFields,
)
from .geometric_config import GeometricChiralityConfig, GeometricChiralityFields
from .geometric_plotting import write_geometric_chirality_outputs
from .plotting import (
    plot_chirality_global,
    plot_chirality_radial_heatmaps,
    save_chirality_fields,
    write_chirality_xtheta_movies,
)


def _geometric_config_from_chirality_config(
    config: ChiralityConfig,
) -> GeometricChiralityConfig:
    return GeometricChiralityConfig(
        radial_bin_width=config.radial_bin_width,
        movie_fps=config.movie_fps,
    )


def _disclination_charge_label(charge: int) -> str:
    assert charge in (-1, 1)
    return "plus_1" if charge == 1 else "minus_1"


def _disclination_particle_masks(
    input_gsd: str | Path,
    neighbor_count_txt: str | Path,
    charge: int,
) -> np.ndarray:
    assert charge in (-1, 1)
    neighbor_data = _load_neighbor_count_matrix(neighbor_count_txt)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        assert len(source) == neighbor_data.counts.shape[0]
        n_frames = len(source)
        n_particles = int(source[0].particles.N)
        assert neighbor_data.counts.shape[1] == n_particles
        masks = np.zeros((n_frames, n_particles), dtype=bool)

        for frame_idx, frame in enumerate(source):
            particles = frame.particles
            assert particles.position is not None
            assert int(particles.N) == n_particles
            assert int(frame.configuration.step) == int(neighbor_data.steps[frame_idx])

            coords = _cylinder_coords(np.asarray(particles.position, dtype=np.float64))
            charges = CYLINDER.neighbors - neighbor_data.counts[frame_idx]
            masks[frame_idx] = _cylinder_shell_mask(coords[:, 2]) & (charges == charge)

    return masks


def write_disclination_chirality_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    data_dir: str | Path = CHIRALITY_DATA_DIR,
    image_dir: str | Path = DISCLINATION_CHIRALITY_IMAGE_DIR,
    config: ChiralityConfig = ChiralityConfig(limit_disclination=True),
    write_movies: bool = True,
) -> dict[int, ChiralityFields]:
    fields_by_charge: dict[int, ChiralityFields] = {}
    data_path = Path(data_dir) / "chirality_disclinations"
    image_path = Path(image_dir)

    for charge in (1, -1):
        label = _disclination_charge_label(charge)
        particle_masks = _disclination_particle_masks(
            input_gsd,
            neighbor_count_txt,
            charge,
        )
        fields = compute_chirality_fields(
            input_gsd,
            config=config,
            particle_masks=particle_masks,
        )
        save_chirality_fields(
            fields,
            data_path / f"{label}_chirality_fields.npz",
            config=config,
        )

        charge_image_dir = image_path / label
        plot_chirality_global(
            fields,
            image_dir=charge_image_dir,
            title=f"Cylinder {charge:+d} disclination chirality diagnostics",
        )
        plot_chirality_radial_heatmaps(
            fields,
            image_dir=charge_image_dir,
            min_count=config.min_count,
        )
        if write_movies:
            write_chirality_xtheta_movies(
                fields,
                image_dir=charge_image_dir,
                min_count=config.xtheta_min_count,
                fps=config.movie_fps,
            )
        fields_by_charge[charge] = fields

    return fields_by_charge


def write_disclination_geometric_chirality_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    neighbor_count_txt: str | Path = CYLINDER_PATHS.neighbor_count_txt,
    data_dir: str | Path = CHIRALITY_DATA_DIR,
    image_dir: str | Path = CHIRALITY_IMAGE_DIR,
    config: GeometricChiralityConfig = GeometricChiralityConfig(),
    write_movies: bool = True,
) -> dict[int, GeometricChiralityFields]:
    fields_by_charge: dict[int, GeometricChiralityFields] = {}
    data_path = Path(data_dir) / "chirality_disclinations" / "geometric"
    image_path = Path(image_dir) / "disclinations" / "geometric"

    for charge in (1, -1):
        label = _disclination_charge_label(charge)
        particle_masks = _disclination_particle_masks(
            input_gsd,
            neighbor_count_txt,
            charge,
        )
        fields_by_charge[charge] = write_geometric_chirality_outputs(
            input_gsd,
            data_dir=data_path,
            image_dir=image_path / label,
            config=config,
            write_movies=write_movies,
            particle_masks=particle_masks,
            data_filename=f"{label}_geometric_chirality_fields.npz",
            plot_title=f"Cylinder {charge:+d} disclination geometric chirality diagnostics",
            radial_title=f"Cylinder {charge:+d} disclination geometric chirality by radius",
        )

    return fields_by_charge


def write_chirality_outputs(
    input_gsd: str | Path = CYLINDER_PATHS.in_gsd,
    data_dir: str | Path = CHIRALITY_DATA_DIR,
    image_dir: str | Path = CHIRALITY_IMAGE_DIR,
    config: ChiralityConfig = ChiralityConfig(),
    write_movies: bool = True,
) -> ChiralityFields | dict[int, ChiralityFields]:
    if config.limit_disclination:
        fields = write_disclination_chirality_outputs(
            input_gsd=input_gsd,
            neighbor_count_txt=CYLINDER_PATHS.neighbor_count_txt,
            data_dir=data_dir,
            image_dir=Path(image_dir) / "disclinations",
            config=config,
            write_movies=write_movies,
        )
        geometric_config = _geometric_config_from_chirality_config(config)
        write_geometric_chirality_outputs(
            input_gsd,
            data_dir=data_dir,
            image_dir=Path(image_dir) / "geometric",
            config=geometric_config,
            write_movies=write_movies,
        )
        write_disclination_geometric_chirality_outputs(
            input_gsd=input_gsd,
            neighbor_count_txt=CYLINDER_PATHS.neighbor_count_txt,
            data_dir=data_dir,
            image_dir=image_dir,
            config=geometric_config,
            write_movies=write_movies,
        )
        return fields

    fields = compute_chirality_fields(input_gsd, config=config)
    save_chirality_fields(
        fields,
        Path(data_dir) / "chirality_fields.npz",
        config=config,
    )
    plot_chirality_global(fields, image_dir=image_dir)
    plot_chirality_radial_heatmaps(
        fields,
        image_dir=image_dir,
        min_count=config.min_count,
    )
    if write_movies:
        write_chirality_xtheta_movies(
            fields,
            image_dir=image_dir,
            min_count=config.xtheta_min_count,
            fps=config.movie_fps,
        )
    write_geometric_chirality_outputs(
        input_gsd,
        data_dir=data_dir,
        image_dir=Path(image_dir) / "geometric",
        config=_geometric_config_from_chirality_config(config),
        write_movies=write_movies,
    )
    write_disclination_geometric_chirality_outputs(
        input_gsd=input_gsd,
        neighbor_count_txt=CYLINDER_PATHS.neighbor_count_txt,
        data_dir=data_dir,
        image_dir=image_dir,
        config=_geometric_config_from_chirality_config(config),
        write_movies=write_movies,
    )
    return fields
