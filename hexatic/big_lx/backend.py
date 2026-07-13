from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np


def _active_directions(xp: Any, orientation: Any) -> Any:
    norms = xp.linalg.norm(orientation, axis=1, keepdims=True)
    quat = orientation / norms
    w, x, y, z = (quat[:, index] for index in range(4))
    return xp.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        axis=1,
    )


def _coords(xp: Any, positions: Any) -> Any:
    radii = xp.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    theta = xp.mod(xp.arctan2(positions[:, 1], positions[:, 2]), 2.0 * xp.pi)
    return xp.stack((positions[:, 0], theta, radii), axis=1)


def _cylindrical(xp: Any, vectors: Any, theta: Any) -> Any:
    sin_theta = xp.sin(theta)
    cos_theta = xp.cos(theta)
    radial = vectors[:, 1] * sin_theta + vectors[:, 2] * cos_theta
    azimuthal = vectors[:, 1] * cos_theta - vectors[:, 2] * sin_theta
    return xp.stack((vectors[:, 0], radial, azimuthal), axis=1)


def _weighted_fields(
    xp: Any,
    distances_sq: Any,
    valid: Any,
    neighbor_directions: Any,
    neighbor_force_velocity: Any,
    neighbor_velocity: Any,
    pocket_radius: float,
) -> tuple[Any, Any, Any, Any]:
    normalization = (2.0 * math.pi) ** 1.5 * pocket_radius**3
    weights = (
        xp.exp(-0.5 * distances_sq / (pocket_radius * pocket_radius))
        / normalization
    ) * valid
    rho = xp.sum(weights, axis=1)
    polar = xp.sum(weights[..., None] * neighbor_directions, axis=1)
    force_density = xp.sum(weights[..., None] * neighbor_force_velocity, axis=1)
    flux = xp.sum(weights[..., None] * neighbor_velocity, axis=1)
    return rho, polar, force_density, flux


def _hexatic(xp: Any, bonds: Any, positions: Any) -> tuple[Any, Any]:
    radii = xp.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    normal_y = positions[:, 1] / radii
    normal_z = positions[:, 2] / radii
    e2_dot = bonds[:, :, 1] * normal_z[:, None] - bonds[:, :, 2] * normal_y[:, None]
    angles = xp.arctan2(e2_dot, bonds[:, :, 0])
    return xp.mean(xp.cos(6.0 * angles), axis=1), xp.mean(
        xp.sin(6.0 * angles), axis=1
    )


@dataclass
class ArrayBackend:
    name: str
    xp: Any
    device_description: str
    _device_get: Callable[[Any], Any]
    _jit: Callable[[Callable[..., Any]], Callable[..., Any]]
    _weighted_kernels: dict[float, Callable[..., Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        xp = self.xp
        self._directions_kernel = self._jit(lambda values: _active_directions(xp, values))
        self._coords_kernel = self._jit(lambda values: _coords(xp, values))
        self._cylindrical_kernel = self._jit(
            lambda vectors, theta: _cylindrical(xp, vectors, theta)
        )
        self._hexatic_kernel = self._jit(
            lambda bonds, positions: _hexatic(xp, bonds, positions)
        )

    @property
    def is_jax(self) -> bool:
        return self.name == "jax"

    def directions(self, orientation: np.ndarray) -> np.ndarray:
        return self.to_numpy(
            self._directions_kernel(self.xp.asarray(orientation, dtype=self.xp.float32))
        )

    def coordinates(self, positions: np.ndarray) -> np.ndarray:
        return self.to_numpy(
            self._coords_kernel(self.xp.asarray(positions, dtype=self.xp.float32))
        )

    def cylindrical(self, vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return self.to_numpy(
            self._cylindrical_kernel(
                self.xp.asarray(vectors, dtype=self.xp.float32),
                self.xp.asarray(theta, dtype=self.xp.float32),
            )
        )

    def hexatic(
        self,
        bonds: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        real, imaginary = self._hexatic_kernel(
            self.xp.asarray(bonds, dtype=self.xp.float32),
            self.xp.asarray(positions, dtype=self.xp.float32),
        )
        return self.to_numpy(real), self.to_numpy(imaginary)

    def weighted_fields(
        self,
        distances_sq: np.ndarray,
        valid: np.ndarray,
        neighbor_directions: np.ndarray,
        neighbor_force_velocity: np.ndarray,
        neighbor_velocity: np.ndarray,
        pocket_radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xp = self.xp

        def kernel(
            distance_values: Any,
            mask_values: Any,
            direction_values: Any,
            force_values: Any,
            velocity_values: Any,
        ) -> tuple[Any, Any, Any, Any]:
            return _weighted_fields(
                xp,
                distance_values,
                mask_values,
                direction_values,
                force_values,
                velocity_values,
                pocket_radius,
            )

        compiled = self._weighted_kernels.get(pocket_radius)
        if compiled is None:
            compiled = self._jit(kernel)
            self._weighted_kernels[pocket_radius] = compiled
        outputs = compiled(
            xp.asarray(distances_sq, dtype=xp.float32),
            xp.asarray(valid, dtype=xp.float32),
            xp.asarray(neighbor_directions, dtype=xp.float32),
            xp.asarray(neighbor_force_velocity, dtype=xp.float32),
            xp.asarray(neighbor_velocity, dtype=xp.float32),
        )
        return tuple(self.to_numpy(output) for output in outputs)  # type: ignore[return-value]

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(self._device_get(value))


def select_backend(name: str = "auto", require_gpu: bool = False) -> ArrayBackend:
    if name not in {"auto", "jax", "numpy"}:
        raise ValueError("backend must be auto, jax, or numpy")
    if name != "numpy":
        try:
            import jax
            import jax.numpy as jnp

            gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
            if gpu_devices:
                return ArrayBackend(
                    name="jax",
                    xp=jnp,
                    device_description=str(gpu_devices[0]),
                    _device_get=jax.device_get,
                    _jit=jax.jit,
                )
            if name == "jax" and not require_gpu:
                return ArrayBackend(
                    name="jax",
                    xp=jnp,
                    device_description=str(jax.devices()[0]),
                    _device_get=jax.device_get,
                    _jit=jax.jit,
                )
        except ImportError:
            if name == "jax":
                raise
    if require_gpu:
        raise RuntimeError("A JAX GPU backend was required but no GPU device was found")
    return ArrayBackend(
        name="numpy",
        xp=np,
        device_description="cpu",
        _device_get=lambda value: value,
        _jit=lambda function: function,
    )
