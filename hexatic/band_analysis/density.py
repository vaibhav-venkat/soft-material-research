from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any, Callable


@dataclass(frozen=True)
class SurfaceGrid:
    lx: float
    circumference: float
    nx: int
    ns: int

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def ds(self) -> float:
        return self.circumference / self.ns


def validate_gpu() -> None:
    jax = import_module("jax")

    devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not devices:
        raise RuntimeError("band analysis requires a JAX GPU/CUDA device")


def _gaussian_kernel(sigma: float) -> tuple[Any, Any]:
    jnp = import_module("jax.numpy")

    radius = max(1, int(math.ceil(4.0 * sigma)))
    offsets = jnp.arange(-radius, radius + 1)
    values = jnp.exp(-0.5 * (offsets / sigma) ** 2)
    return offsets, values / jnp.sum(values)


def make_density_kernel(
    grid: SurfaceGrid,
    *,
    radius: float,
    particle_diameter: float,
    shell_epsilon: float,
    smoothing_sigma: float,
) -> Callable[[Any], tuple[Any, Any, Any]]:
    """Build a JIT-compiled shell histogram and periodic Gaussian smoother."""
    jax = import_module("jax")
    jnp = import_module("jax.numpy")

    _, weights = _gaussian_kernel(smoothing_sigma)
    offset_values = range(-(weights.shape[0] // 2), weights.shape[0] // 2 + 1)
    lower_radius = radius - particle_diameter - shell_epsilon
    upper_radius = radius + shell_epsilon

    @jax.jit
    def kernel(coords: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        radial = coords[:, 2]
        outside = jnp.any(radial > upper_radius)
        shell = (radial > lower_radius) & (radial <= upper_radius)
        x = jnp.mod(coords[:, 0] + 0.5 * grid.lx, grid.lx)
        s = jnp.mod(radius * coords[:, 1], grid.circumference)
        ix = jnp.minimum((x / grid.dx).astype(jnp.int32), grid.nx - 1)
        js = jnp.minimum((s / grid.ds).astype(jnp.int32), grid.ns - 1)
        counts = jnp.zeros((grid.nx, grid.ns), dtype=jnp.float32)
        counts = counts.at[ix, js].add(shell.astype(jnp.float32))
        smoothed = sum(
            weight * jnp.roll(counts, int(offset), axis=0)
            for offset, weight in zip(offset_values, weights)
        )
        smoothed = sum(
            weight * jnp.roll(smoothed, int(offset), axis=1)
            for offset, weight in zip(offset_values, weights)
        )
        return smoothed / (grid.dx * grid.ds), jnp.sum(shell), outside

    return kernel


def make_density_batch_kernel(
    grid: SurfaceGrid,
    *,
    radius: float,
    particle_diameter: float,
    shell_epsilon: float,
    smoothing_sigma: float,
) -> Callable[[Any], tuple[Any, Any, Any]]:
    """Build a batched density kernel over equally sized trajectory frames."""
    jax = import_module("jax")

    single_frame = make_density_kernel(
        grid,
        radius=radius,
        particle_diameter=particle_diameter,
        shell_epsilon=shell_epsilon,
        smoothing_sigma=smoothing_sigma,
    )
    return jax.jit(jax.vmap(single_frame))
