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
def tangent_nematic_means(
    direction_cylindrical: np.ndarray,
    shell_mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
) -> tuple[float, float, float, float]:
    start = min(max(frame_start, 0), direction_cylindrical.shape[0])
    stop = min(max(frame_stop, start), direction_cylindrical.shape[0])
    s_shell_sum = 0.0
    s_shell_count = 0
    s_core_sum = 0.0
    s_core_count = 0
    q_xx_shell_sum = 0.0
    q_xtheta_shell_sum = 0.0
    q_shell_count = 0

    for frame_idx in range(start, stop):
        shell_q_xx_sum = 0.0
        shell_q_xtheta_sum = 0.0
        shell_count = 0
        core_q_xx_sum = 0.0
        core_q_xtheta_sum = 0.0
        core_count = 0

        for particle_idx in range(direction_cylindrical.shape[1]):
            p_x = direction_cylindrical[frame_idx, particle_idx, 0]
            p_theta = direction_cylindrical[frame_idx, particle_idx, 2]
            norm = math.sqrt(p_x * p_x + p_theta * p_theta)
            if not math.isfinite(norm) or norm <= 0.0:
                continue

            u_x = p_x / norm
            u_theta = p_theta / norm
            q_xx = 2.0 * u_x * u_x - 1.0
            q_xtheta = 2.0 * u_x * u_theta

            if shell_mask[frame_idx, particle_idx]:
                shell_q_xx_sum += q_xx
                shell_q_xtheta_sum += q_xtheta
                shell_count += 1
            else:
                core_q_xx_sum += q_xx
                core_q_xtheta_sum += q_xtheta
                core_count += 1

        if shell_count:
            shell_q_xx = shell_q_xx_sum / shell_count
            shell_q_xtheta = shell_q_xtheta_sum / shell_count
            s_shell_sum += math.sqrt(
                shell_q_xx * shell_q_xx + shell_q_xtheta * shell_q_xtheta
            )
            s_shell_count += 1
            q_xx_shell_sum += shell_q_xx
            q_xtheta_shell_sum += shell_q_xtheta
            q_shell_count += 1

        if core_count:
            core_q_xx = core_q_xx_sum / core_count
            core_q_xtheta = core_q_xtheta_sum / core_count
            s_core_sum += math.sqrt(core_q_xx * core_q_xx + core_q_xtheta * core_q_xtheta)
            s_core_count += 1

    return (
        s_shell_sum / s_shell_count if s_shell_count else np.nan,
        s_core_sum / s_core_count if s_core_count else np.nan,
        q_xx_shell_sum / q_shell_count if q_shell_count else np.nan,
        q_xtheta_shell_sum / q_shell_count if q_shell_count else np.nan,
    )


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
