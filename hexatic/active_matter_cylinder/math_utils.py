import numpy as np


def _active_direction_from_quaternion(orientation: np.ndarray) -> np.ndarray:
    orientation = np.asarray(orientation, dtype=np.float64)
    assert orientation.ndim == 2 and orientation.shape[1] == 4

    norms = np.linalg.norm(orientation, axis=1)
    assert np.all(norms > 0.0)
    quat = orientation / norms[:, np.newaxis]
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    return np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        )
    )


def _logged_particle_array(frame, quantity: str, n_particles: int) -> np.ndarray:
    log = getattr(frame, "log", None)
    if not log:
        raise ValueError(f"GSD frame has no logger data; expected {quantity}.")

    quantity_lower = quantity.lower()
    candidates: list[np.ndarray] = []
    for key, value in log.items():
        key_lower = str(key).lower()
        array = np.asarray(value)
        if quantity_lower in key_lower and array.shape[:1] == (n_particles,):
            candidates.append(array)

    if not candidates:
        available = ", ".join(str(key) for key in log)
        raise ValueError(
            f"Could not find logged per-particle {quantity}. "
            f"Available logger keys: {available}"
        )

    total = np.zeros_like(np.asarray(candidates[0], dtype=np.float64))
    for array in candidates:
        total += np.asarray(array, dtype=np.float64)
    return total


def _minimum_image_delta(values: np.ndarray, period: float) -> np.ndarray:
    assert period > 0.0
    return values - period * np.round(values / period)


def _axis_edges_and_centers(low: float, high: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    assert high > low
    assert spacing > 0.0
    n_bins = int(np.ceil((high - low) / spacing))
    edges = low + spacing * np.arange(n_bins + 1)
    edges[-1] = high
    return edges, 0.5 * (edges[:-1] + edges[1:])


def _gaussian_kernel_volume(radius: float) -> float:
    assert radius > 0.0
    return (2.0 * np.pi) ** 1.5 * radius**3


def _gaussian_delta_weights(delta_sq: np.ndarray, radius: float) -> np.ndarray:
    return np.exp(-0.5 * delta_sq / (radius * radius)) / _gaussian_kernel_volume(radius)


def _density_sum(
    grid_points: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
    box_length_x: float,
    radius: float,
    block_size: int = 512,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output_shape = values.shape[1:]
    density = np.zeros((len(grid_points),) + output_shape, dtype=np.float64)
    flat_values = values.reshape(values.shape[0], -1)
    flat_density = density.reshape(len(grid_points), -1)

    for start in range(0, len(grid_points), block_size):
        stop = min(start + block_size, len(grid_points))
        deltas = grid_points[start:stop, np.newaxis, :] - positions[np.newaxis, :, :]
        deltas[..., 0] = _minimum_image_delta(deltas[..., 0], box_length_x)
        weights = _gaussian_delta_weights(np.sum(deltas * deltas, axis=2), radius)
        flat_density[start:stop] = weights @ flat_values

    return density


def _pocket_fields(
    positions: np.ndarray,
    directions: np.ndarray,
    box_length_x: float,
    pocket_radius: float,
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = len(positions)
    rho = np.zeros(n_particles, dtype=np.float64)
    polar_sum = np.zeros((n_particles, 3), dtype=np.float64)

    for start in range(0, n_particles, block_size):
        stop = min(start + block_size, n_particles)
        deltas = positions[start:stop, np.newaxis, :] - positions[np.newaxis, :, :]
        deltas[..., 0] = _minimum_image_delta(deltas[..., 0], box_length_x)
        weights = _gaussian_delta_weights(np.sum(deltas * deltas, axis=2), pocket_radius)
        rho[start:stop] = np.sum(weights, axis=1)
        polar_sum[start:stop] = weights @ directions

    return rho, polar_sum, polar_sum


def _pocket_vector_density(
    positions: np.ndarray,
    vectors: np.ndarray,
    box_length_x: float,
    pocket_radius: float,
    block_size: int = 256,
) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    assert vectors.ndim == 2 and vectors.shape[0] == len(positions)
    vector_density = np.zeros_like(vectors, dtype=np.float64)

    for start in range(0, len(positions), block_size):
        stop = min(start + block_size, len(positions))
        deltas = positions[start:stop, np.newaxis, :] - positions[np.newaxis, :, :]
        deltas[..., 0] = _minimum_image_delta(deltas[..., 0], box_length_x)
        weights = _gaussian_delta_weights(np.sum(deltas * deltas, axis=2), pocket_radius)
        vector_density[start:stop] = weights @ vectors

    return vector_density


def _cylindrical_components(vectors: np.ndarray, theta: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    assert vectors.ndim == 2 and vectors.shape[1] == 3
    assert theta.shape == (vectors.shape[0],)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    radial = vectors[:, 1] * sin_theta + vectors[:, 2] * cos_theta
    azimuthal = vectors[:, 1] * cos_theta - vectors[:, 2] * sin_theta
    return np.column_stack((vectors[:, 0], radial, azimuthal))


def _cylindrical_basis(points: np.ndarray) -> np.ndarray:
    theta = np.mod(np.arctan2(points[:, 1], points[:, 2]), 2.0 * np.pi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    basis = np.zeros((len(points), 3, 3), dtype=np.float64)
    basis[:, 0, 0] = 1.0
    basis[:, 1, 1] = sin_theta
    basis[:, 1, 2] = cos_theta
    basis[:, 2, 1] = cos_theta
    basis[:, 2, 2] = -sin_theta
    return basis


def _cartesian_tensor_to_cylindrical(points: np.ndarray, tensors: np.ndarray) -> np.ndarray:
    tensors = np.asarray(tensors, dtype=np.float64)
    assert tensors.shape == (len(points), 3, 3)
    basis = _cylindrical_basis(points)
    return np.einsum("iac,icd,ibd->iab", basis, tensors, basis)


def _cartesian_vector_to_cylindrical_components(points: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    assert vectors.shape == (len(points), 3)
    basis = _cylindrical_basis(points)
    return np.einsum("iac,ic->ia", basis, vectors)


def _cylindrical_plot_points(points: np.ndarray) -> np.ndarray:
    radii = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
    theta = np.mod(np.arctan2(points[:, 1], points[:, 2]), 2.0 * np.pi)
    return np.column_stack((points[:, 0], radii, theta))


def _cylindrical_plot_vectors(points: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    radii = np.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2)
    theta = np.mod(np.arctan2(points[:, 1], points[:, 2]), 2.0 * np.pi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    radial = vectors[:, 1] * sin_theta + vectors[:, 2] * cos_theta
    azimuthal = vectors[:, 1] * cos_theta - vectors[:, 2] * sin_theta
    angular = np.divide(
        azimuthal,
        radii,
        out=np.zeros_like(azimuthal, dtype=np.float64),
        where=radii > 0.0,
    )
    return np.column_stack((vectors[:, 0], radial, angular))
