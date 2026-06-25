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
            s_core_sum += math.sqrt(
                core_q_xx * core_q_xx + core_q_xtheta * core_q_xtheta
            )
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
def _cell_index(
    x: float,
    y: float,
    z: float,
    x_min: float,
    y_min: float,
    z_min: float,
    cell_size: float,
    n_x: int,
    n_y: int,
    n_z: int,
    box_length_x: float,
) -> int:
    x_periodic = x - x_min
    if box_length_x > 0.0:
        x_periodic -= box_length_x * math.floor(x_periodic / box_length_x)
    i_x = int(math.floor(x_periodic / cell_size))
    if i_x < 0:
        i_x = 0
    elif i_x >= n_x:
        i_x = n_x - 1

    i_y = int(math.floor((y - y_min) / cell_size))
    if i_y < 0:
        i_y = 0
    elif i_y >= n_y:
        i_y = n_y - 1

    i_z = int(math.floor((z - z_min) / cell_size))
    if i_z < 0:
        i_z = 0
    elif i_z >= n_z:
        i_z = n_z - 1

    return (i_x * n_y + i_y) * n_z + i_z


@njit(cache=True)
def _nematic_order_from_q(q_xx_sum: float, q_xtheta_sum: float, count: int) -> float:
    if count <= 0:
        return np.nan
    q_xx = q_xx_sum / count
    q_xtheta = q_xtheta_sum / count
    return math.sqrt(q_xx * q_xx + q_xtheta * q_xtheta)


@njit(cache=True)
def local_disclination_field_contrasts(
    coords: np.ndarray,
    direction_cylindrical: np.ndarray,
    hexatic_abs: np.ndarray,
    disclination_mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
    core_radius: float,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    chirality_radius: float,
    box_length_x: float,
    x_min: float,
    cylinder_radius: float,
) -> tuple[float, float, float]:
    start = min(max(frame_start, 0), coords.shape[0])
    stop = min(max(frame_stop, start), coords.shape[0])
    cell_size = max(core_radius, chirality_radius)
    if cell_size <= 0.0:
        return np.nan, np.nan, np.nan

    n_particles = coords.shape[1]
    y_min = -cylinder_radius
    z_min = -cylinder_radius
    transverse_width = 2.0 * cylinder_radius
    n_x = max(1, int(math.ceil(box_length_x / cell_size)))
    n_y = max(1, int(math.ceil(transverse_width / cell_size)))
    n_z = max(1, int(math.ceil(transverse_width / cell_size)))
    n_cells = n_x * n_y * n_z
    search_cells = int(math.ceil(annulus_outer_radius / cell_size))

    core_sq = core_radius * core_radius
    annulus_inner_sq = annulus_inner_radius * annulus_inner_radius
    annulus_outer_sq = annulus_outer_radius * annulus_outer_radius
    chirality_sq_radius = chirality_radius * chirality_radius

    s_delta_sum = 0.0
    s_delta_count = 0
    hexatic_delta_sum = 0.0
    hexatic_delta_count = 0
    chirality_delta_sum = 0.0
    chirality_delta_count = 0

    for frame_idx in range(start, stop):
        x_values = np.empty(n_particles, dtype=np.float64)
        y_values = np.empty(n_particles, dtype=np.float64)
        z_values = np.empty(n_particles, dtype=np.float64)
        finite_position = np.zeros(n_particles, dtype=np.bool_)
        head = np.full(n_cells, -1, dtype=np.int64)
        next_index = np.full(n_particles, -1, dtype=np.int64)

        for particle_idx in range(n_particles):
            x_i = coords[frame_idx, particle_idx, 0]
            theta_i = coords[frame_idx, particle_idx, 1]
            radius_i = coords[frame_idx, particle_idx, 2]
            if (
                not math.isfinite(x_i)
                or not math.isfinite(theta_i)
                or not math.isfinite(radius_i)
            ):
                continue

            y_i = radius_i * math.sin(theta_i)
            z_i = radius_i * math.cos(theta_i)
            x_values[particle_idx] = x_i
            y_values[particle_idx] = y_i
            z_values[particle_idx] = z_i
            finite_position[particle_idx] = True

            cell_idx = _cell_index(
                x_i,
                y_i,
                z_i,
                x_min,
                y_min,
                z_min,
                cell_size,
                n_x,
                n_y,
                n_z,
                box_length_x,
            )
            next_index[particle_idx] = head[cell_idx]
            head[cell_idx] = particle_idx

        chirality_squared = np.full(n_particles, np.nan, dtype=np.float64)
        for particle_idx in range(n_particles):
            if not finite_position[particle_idx]:
                continue

            x_i = x_values[particle_idx]
            y_i = y_values[particle_idx]
            z_i = z_values[particle_idx]
            center_cell = _cell_index(
                x_i,
                y_i,
                z_i,
                x_min,
                y_min,
                z_min,
                cell_size,
                n_x,
                n_y,
                n_z,
                box_length_x,
            )
            center_x = center_cell // (n_y * n_z)
            center_y = (center_cell // n_z) % n_y
            center_z = center_cell % n_z
            chirality = 0.0

            for delta_x in range(-1, 2):
                cell_x = (center_x + delta_x) % n_x
                for delta_y in range(-1, 2):
                    cell_y = center_y + delta_y
                    if cell_y < 0 or cell_y >= n_y:
                        continue
                    for delta_z in range(-1, 2):
                        cell_z = center_z + delta_z
                        if cell_z < 0 or cell_z >= n_z:
                            continue
                        cell_idx = (cell_x * n_y + cell_y) * n_z + cell_z
                        neighbor_idx = head[cell_idx]
                        while neighbor_idx != -1:
                            if neighbor_idx != particle_idx:
                                dx = x_values[neighbor_idx] - x_i
                                if box_length_x > 0.0:
                                    dx -= box_length_x * round(dx / box_length_x)
                                dy = y_values[neighbor_idx] - y_i
                                dz = z_values[neighbor_idx] - z_i
                                distance_sq = dx * dx + dy * dy + dz * dz
                                if (
                                    distance_sq > 0.0
                                    and distance_sq <= chirality_sq_radius
                                ):
                                    chirality += dx / math.sqrt(distance_sq)
                            neighbor_idx = next_index[neighbor_idx]

            chirality_squared[particle_idx] = chirality * chirality

        for defect_idx in range(n_particles):
            if (
                not finite_position[defect_idx]
                or not disclination_mask[frame_idx, defect_idx]
            ):
                continue

            x_i = x_values[defect_idx]
            y_i = y_values[defect_idx]
            z_i = z_values[defect_idx]
            center_cell = _cell_index(
                x_i,
                y_i,
                z_i,
                x_min,
                y_min,
                z_min,
                cell_size,
                n_x,
                n_y,
                n_z,
                box_length_x,
            )
            center_x = center_cell // (n_y * n_z)
            center_y = (center_cell // n_z) % n_y
            center_z = center_cell % n_z

            core_s_q_xx_sum = 0.0
            core_s_q_xtheta_sum = 0.0
            core_s_count = 0
            annulus_s_q_xx_sum = 0.0
            annulus_s_q_xtheta_sum = 0.0
            annulus_s_count = 0
            core_hexatic_sum = 0.0
            core_hexatic_count = 0
            annulus_hexatic_sum = 0.0
            annulus_hexatic_count = 0
            core_chirality_sum = 0.0
            core_chirality_count = 0
            annulus_chirality_sum = 0.0
            annulus_chirality_count = 0

            for delta_x in range(-search_cells, search_cells + 1):
                cell_x = (center_x + delta_x) % n_x
                for delta_y in range(-search_cells, search_cells + 1):
                    cell_y = center_y + delta_y
                    if cell_y < 0 or cell_y >= n_y:
                        continue
                    for delta_z in range(-search_cells, search_cells + 1):
                        cell_z = center_z + delta_z
                        if cell_z < 0 or cell_z >= n_z:
                            continue
                        cell_idx = (cell_x * n_y + cell_y) * n_z + cell_z
                        particle_idx = head[cell_idx]
                        while particle_idx != -1:
                            dx = x_values[particle_idx] - x_i
                            if box_length_x > 0.0:
                                dx -= box_length_x * round(dx / box_length_x)
                            dy = y_values[particle_idx] - y_i
                            dz = z_values[particle_idx] - z_i
                            distance_sq = dx * dx + dy * dy + dz * dz
                            in_core = distance_sq < core_sq
                            in_annulus = (
                                distance_sq > annulus_inner_sq
                                and distance_sq < annulus_outer_sq
                            )
                            if not in_core and not in_annulus:
                                particle_idx = next_index[particle_idx]
                                continue

                            p_x = direction_cylindrical[frame_idx, particle_idx, 0]
                            p_theta = direction_cylindrical[frame_idx, particle_idx, 2]
                            norm = math.sqrt(p_x * p_x + p_theta * p_theta)
                            if math.isfinite(norm) and norm > 0.0:
                                u_x = p_x / norm
                                u_theta = p_theta / norm
                                q_xx = 2.0 * u_x * u_x - 1.0
                                q_xtheta = 2.0 * u_x * u_theta
                                if in_core:
                                    core_s_q_xx_sum += q_xx
                                    core_s_q_xtheta_sum += q_xtheta
                                    core_s_count += 1
                                else:
                                    annulus_s_q_xx_sum += q_xx
                                    annulus_s_q_xtheta_sum += q_xtheta
                                    annulus_s_count += 1

                            hexatic_value = hexatic_abs[frame_idx, particle_idx]
                            if math.isfinite(hexatic_value):
                                if in_core:
                                    core_hexatic_sum += hexatic_value
                                    core_hexatic_count += 1
                                else:
                                    annulus_hexatic_sum += hexatic_value
                                    annulus_hexatic_count += 1

                            chirality_value = chirality_squared[particle_idx]
                            if math.isfinite(chirality_value):
                                if in_core:
                                    core_chirality_sum += chirality_value
                                    core_chirality_count += 1
                                else:
                                    annulus_chirality_sum += chirality_value
                                    annulus_chirality_count += 1

                            particle_idx = next_index[particle_idx]

            if core_s_count > 0 and annulus_s_count > 0:
                core_s = _nematic_order_from_q(
                    core_s_q_xx_sum,
                    core_s_q_xtheta_sum,
                    core_s_count,
                )
                annulus_s = _nematic_order_from_q(
                    annulus_s_q_xx_sum,
                    annulus_s_q_xtheta_sum,
                    annulus_s_count,
                )
                if math.isfinite(core_s) and math.isfinite(annulus_s):
                    s_delta_sum += core_s - annulus_s
                    s_delta_count += 1

            if core_hexatic_count > 0 and annulus_hexatic_count > 0:
                hexatic_delta_sum += (
                    core_hexatic_sum / core_hexatic_count
                    - annulus_hexatic_sum / annulus_hexatic_count
                )
                hexatic_delta_count += 1

            if core_chirality_count > 0 and annulus_chirality_count > 0:
                chirality_delta_sum += (
                    core_chirality_sum / core_chirality_count
                    - annulus_chirality_sum / annulus_chirality_count
                )
                chirality_delta_count += 1

    s_delta = s_delta_sum / s_delta_count if s_delta_count else np.nan
    hexatic_delta = (
        hexatic_delta_sum / hexatic_delta_count if hexatic_delta_count else np.nan
    )
    chirality_delta = (
        chirality_delta_sum / chirality_delta_count
        if chirality_delta_count
        else np.nan
    )
    return s_delta, hexatic_delta, chirality_delta


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
