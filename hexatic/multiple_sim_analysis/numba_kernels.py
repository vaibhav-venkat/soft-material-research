from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def mean_by_population(
    values: np.ndarray,
    shell_mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
) -> tuple[float, float]:
    start = min(max(frame_start, 0), values.shape[0])
    stop = min(max(frame_stop, start), values.shape[0])
    all_sum = 0.0
    all_count = 0
    shell_sum = 0.0
    shell_count = 0

    for frame_idx in range(start, stop):
        for particle_idx in range(values.shape[1]):
            value = values[frame_idx, particle_idx]
            if math.isfinite(value):
                all_sum += value
                all_count += 1
                if shell_mask[frame_idx, particle_idx]:
                    shell_sum += value
                    shell_count += 1

    all_mean = all_sum / all_count if all_count else np.nan
    shell_mean = shell_sum / shell_count if shell_count else np.nan
    return all_mean, shell_mean


@njit(cache=True)
def shell_core_density_means(
    shell_mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
    shell_volume: float,
    core_volume: float,
) -> tuple[float, float, float]:
    start = min(max(frame_start, 0), shell_mask.shape[0])
    stop = min(max(frame_stop, start), shell_mask.shape[0])
    if start >= stop or shell_volume <= 0.0 or core_volume <= 0.0:
        return np.nan, np.nan, np.nan

    shell_sum = 0.0
    core_sum = 0.0
    diff_sum = 0.0
    frame_count = 0
    n_particles = shell_mask.shape[1]

    for frame_idx in range(start, stop):
        n_shell = 0
        for particle_idx in range(n_particles):
            if shell_mask[frame_idx, particle_idx]:
                n_shell += 1
        rho_shell = n_shell / shell_volume
        rho_core = (n_particles - n_shell) / core_volume
        shell_sum += rho_shell
        core_sum += rho_core
        diff_sum += rho_shell - rho_core
        frame_count += 1

    return (
        shell_sum / frame_count,
        core_sum / frame_count,
        diff_sum / frame_count,
    )


@njit(cache=True)
def x_velocity_means_from_coords(
    x_positions: np.ndarray,
    shell_mask: np.ndarray,
    steps: np.ndarray,
    box_length_x: float,
    timestep: float,
    frame_start: int,
    frame_stop: int,
) -> tuple[float, float]:
    start = min(max(frame_start, 1), x_positions.shape[0])
    stop = min(max(frame_stop, start), x_positions.shape[0])
    all_sum = 0.0
    all_count = 0
    shell_sum = 0.0
    shell_count = 0

    for frame_idx in range(start, stop):
        delta_t = (steps[frame_idx] - steps[frame_idx - 1]) * timestep
        if delta_t <= 0.0:
            continue

        frame_all_sum = 0.0
        frame_all_count = 0
        frame_shell_sum = 0.0
        frame_shell_count = 0
        for particle_idx in range(x_positions.shape[1]):
            dx = x_positions[frame_idx, particle_idx] - x_positions[frame_idx - 1, particle_idx]
            dx -= box_length_x * round(dx / box_length_x)
            value = dx / delta_t
            if math.isfinite(value):
                frame_all_sum += value
                frame_all_count += 1
                if shell_mask[frame_idx, particle_idx] and shell_mask[frame_idx - 1, particle_idx]:
                    frame_shell_sum += value
                    frame_shell_count += 1

        if frame_all_count:
            all_sum += frame_all_sum / frame_all_count
            all_count += 1
        if frame_shell_count:
            shell_sum += frame_shell_sum / frame_shell_count
            shell_count += 1

    all_mean = all_sum / all_count if all_count else np.nan
    shell_mean = shell_sum / shell_count if shell_count else np.nan
    return all_mean, shell_mean


@njit(cache=True)
def mean_square_frame_mean(
    values: np.ndarray,
    frame_start: int,
    frame_stop: int,
) -> float:
    start = min(max(frame_start, 0), values.shape[0])
    stop = min(max(frame_stop, start), values.shape[0])
    total = 0.0
    frame_count = 0

    for frame_idx in range(start, stop):
        frame_sum = 0.0
        value_count = 0
        for particle_idx in range(values.shape[1]):
            value = values[frame_idx, particle_idx]
            if math.isfinite(value):
                frame_sum += value * value
                value_count += 1
        if value_count:
            total += frame_sum / value_count
            frame_count += 1

    return total / frame_count if frame_count else np.nan


@njit(cache=True)
def defect_x_com_velocity(
    x_positions: np.ndarray,
    charges: np.ndarray,
    steps: np.ndarray,
    box_length_x: float,
    timestep: float,
    charge: int,
    frame_start: int,
    frame_stop: int,
) -> float:
    centers = np.empty(steps.shape[0], dtype=np.float64)
    for frame_idx in range(steps.shape[0]):
        sin_sum = 0.0
        cos_sum = 0.0
        count = 0
        for particle_idx in range(x_positions.shape[1]):
            if charges[frame_idx, particle_idx] == charge:
                angle = 2.0 * math.pi * x_positions[frame_idx, particle_idx] / box_length_x
                sin_sum += math.sin(angle)
                cos_sum += math.cos(angle)
                count += 1
        if count:
            centers[frame_idx] = math.atan2(sin_sum / count, cos_sum / count) * box_length_x / (2.0 * math.pi)
        else:
            centers[frame_idx] = np.nan

    start = min(max(frame_start, 1), steps.shape[0])
    stop = min(max(frame_stop, start), steps.shape[0])
    velocity_sum = 0.0
    velocity_count = 0
    for frame_idx in range(start, stop):
        delta_t = (steps[frame_idx] - steps[frame_idx - 1]) * timestep
        if (
            delta_t <= 0.0
            or not math.isfinite(centers[frame_idx])
            or not math.isfinite(centers[frame_idx - 1])
        ):
            continue
        dx = centers[frame_idx] - centers[frame_idx - 1]
        dx -= box_length_x * round(dx / box_length_x)
        velocity_sum += dx / delta_t
        velocity_count += 1

    return velocity_sum / velocity_count if velocity_count else np.nan
