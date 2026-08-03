from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .components import BandComponent
from .density import SurfaceGrid
from .interfaces import BandInterfaces


N_MODES = 6


@dataclass(frozen=True)
class BandCharacterization:
    label: int
    lower: np.ndarray
    upper: np.ndarray
    centerline: np.ndarray
    width: np.ndarray
    area: float
    mean_width: float
    wrapped_axial_position: float
    unwrapped_axial_position: float
    displacement: float
    width_variance: float
    centerline_variance: float
    interface_length: float
    geometrically_complex: bool
    centerline_modes: np.ndarray
    centerline_amplitudes: np.ndarray
    width_modes: np.ndarray
    width_amplitudes: np.ndarray


def wrap_axial(value: float, lx: float) -> float:
    """Return the canonical axial image in ``[0, lx)``."""
    return float(value % lx)


def minimum_image(delta: float, period: float) -> float:
    return float((delta + 0.5 * period) % period - 0.5 * period)


@jax.jit
def _morphology_reductions(
    lower: jax.Array,
    upper: jax.Array,
    centerline: jax.Array,
    width: jax.Array,
    mean_width: float,
    ds: float,
    lx: float,
) -> tuple[jax.Array, ...]:
    def curve_length(values: jax.Array) -> jax.Array:
        differences = jnp.roll(values, -1) - values
        differences = (differences + 0.5 * lx) % lx - 0.5 * lx
        return jnp.sum(jnp.hypot(ds, differences))

    position = jnp.mean(centerline)
    centerline_modes = (
        jnp.fft.fft(centerline - position)[: N_MODES + 1][1:]
        / centerline.size
    )
    width_modes = (
        jnp.fft.fft(width - jnp.mean(width))[: N_MODES + 1][1:]
        / width.size
    )
    return (
        position,
        jnp.mean((width - mean_width) ** 2),
        jnp.mean((centerline - position) ** 2),
        curve_length(lower) + curve_length(upper),
        centerline_modes,
        width_modes,
    )


def characterize_band(
    component: BandComponent,
    interfaces: BandInterfaces,
    grid: SurfaceGrid,
) -> BandCharacterization:
    """Calculate morphology for one circumferential, axially localized band."""
    if component.winds_x or not component.winds_s:
        raise ValueError(
            "only components that wind around s and not around x can be characterized"
        )
    if grid.ns < 2 * N_MODES + 1:
        raise ValueError(
            f"at least {2 * N_MODES + 1} circumferential bins are required "
            f"for modes 1..{N_MODES}; got {grid.ns}"
        )
    arrays = (
        interfaces.lower,
        interfaces.upper,
        interfaces.centerline,
        interfaces.width,
    )
    if any(array.shape != (grid.ns,) for array in arrays):
        raise ValueError("interface arrays must have shape (grid.ns,)")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("a circumferential band must have an interface in every s column")

    area = component.area_cells * grid.dx * grid.ds
    mean_width = area / grid.circumference
    reductions = jax.device_get(
        _morphology_reductions(
            jnp.asarray(interfaces.lower),
            jnp.asarray(interfaces.upper),
            jnp.asarray(interfaces.centerline),
            jnp.asarray(interfaces.width),
            mean_width,
            grid.ds,
            grid.lx,
        )
    )
    position = float(reductions[0])
    width_variance = float(reductions[1])
    centerline_variance = float(reductions[2])
    interface_length = float(reductions[3])
    centerline_modes = np.asarray(reductions[4])
    width_modes = np.asarray(reductions[5])
    return BandCharacterization(
        label=component.label,
        lower=interfaces.lower.copy(),
        upper=interfaces.upper.copy(),
        centerline=interfaces.centerline.copy(),
        width=interfaces.width.copy(),
        area=float(area),
        mean_width=float(mean_width),
        wrapped_axial_position=wrap_axial(position, grid.lx),
        unwrapped_axial_position=position,
        displacement=0.0,
        width_variance=width_variance,
        centerline_variance=centerline_variance,
        interface_length=interface_length,
        geometrically_complex=interfaces.geometrically_complex,
        centerline_modes=centerline_modes,
        centerline_amplitudes=2.0 * np.abs(centerline_modes),
        width_modes=width_modes,
        width_amplitudes=np.abs(width_modes),
    )


def align_to_temporal_image(
    band: BandCharacterization,
    position: float,
    first_position: float,
) -> BandCharacterization:
    """Shift all axial geometry onto a track's selected temporal image."""
    shift = position - band.unwrapped_axial_position
    return replace(
        band,
        lower=band.lower + shift,
        upper=band.upper + shift,
        centerline=band.centerline + shift,
        unwrapped_axial_position=position,
        displacement=position - first_position,
    )


def mode_wave_numbers(circumference: float) -> np.ndarray:
    return 2.0 * np.pi * np.arange(1, N_MODES + 1) / circumference
