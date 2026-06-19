from pathlib import Path

import gsd.hoomd
import numpy as np

from hexatic import analysis as hx

from .common import CYLINDER


def _copy_masked_particle_property(source, destination, name: str, mask: np.ndarray) -> None:
    values = getattr(source, name, None)
    if values is None:
        return

    values = np.asarray(values)
    if values.shape[:1] != mask.shape:
        return

    setattr(destination, name, values[mask].copy())


def write_dynamic_values_gsd(
    input_gsd: str | Path,
    output_gsd: str | Path,
    cylinder_radius: float = CYLINDER.cylinder_radius,
    wall_cutoff: float = CYLINDER.wall_cutoff,
) -> None:
    output_path = Path(output_gsd)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gsd.hoomd.open(name=str(input_gsd), mode="r") as source:
        with gsd.hoomd.open(name=str(output_path), mode="w") as destination:
            for frame in source:
                particles = frame.particles
                assert particles.position is not None

                dynamic_values = hx.get_dynamic_values(
                    particles.position,
                    contain_all=True,
                    cylinder_radius=cylinder_radius,
                    cutoff=wall_cutoff,
                )
                new_frame = gsd.hoomd.Frame()
                new_frame.configuration.step = frame.configuration.step
                new_frame.configuration.box = frame.configuration.box
                new_frame.particles.N = dynamic_values.coords.shape[0]
                new_frame.particles.position = dynamic_values.coords.astype(np.float32)
                new_frame.particles.velocity = dynamic_values.coords.astype(np.float32)
                if particles.types is None:
                    new_frame.particles.types = ["A"]
                else:
                    new_frame.particles.types = list(particles.types)

                if particles.typeid is None:
                    new_frame.particles.typeid = np.zeros(
                        dynamic_values.coords.shape[0],
                        dtype=np.uint32,
                    )
                else:
                    new_frame.particles.typeid = np.asarray(particles.typeid)[
                        dynamic_values.shell_mask
                    ].copy()

                for name in (
                    "diameter",
                    "mass",
                    "charge",
                    "body",
                    "image",
                    "orientation",
                    "moment_inertia",
                    "angular_momentum",
                ):
                    _copy_masked_particle_property(
                        particles,
                        new_frame.particles,
                        name,
                        dynamic_values.shell_mask,
                    )

                destination.append(new_frame)
