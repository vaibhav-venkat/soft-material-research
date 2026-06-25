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
def local_disclination_field_profiles(
    coords: np.ndarray,
    direction_cylindrical: np.ndarray,
    hexatic_abs: np.ndarray,
    disclination_mask: np.ndarray,
    frame_start: int,
    frame_stop: int,
    bin_edges: np.ndarray,
    chirality_radius: float,
    box_length_x: float,
    x_min: float,
    cylinder_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = min(max(frame_start, 0), coords.shape[0])
    stop = min(max(frame_stop, start), coords.shape[0])
    n_bins = bin_edges.shape[0] - 1
    s_sums = np.zeros(n_bins, dtype=np.float64)
    s_counts = np.zeros(n_bins, dtype=np.int64)
    hexatic_sums = np.zeros(n_bins, dtype=np.float64)
    hexatic_counts = np.zeros(n_bins, dtype=np.int64)
    chirality_sums = np.zeros(n_bins, dtype=np.float64)
    chirality_counts = np.zeros(n_bins, dtype=np.int64)
    annulus_chirality_sum = 0.0
    annulus_chirality_count = 0

    if n_bins <= 0 or chirality_radius <= 0.0:
        empty = np.full(0, np.nan, dtype=np.float64)
        return empty, empty, empty, np.nan

    max_radius = bin_edges[-1]
    if max_radius <= 0.0:
        empty = np.full(n_bins, np.nan, dtype=np.float64)
        return empty, empty, empty, np.nan

    cell_size = max(chirality_radius, bin_edges[1] - bin_edges[0])
    n_particles = coords.shape[1]
    y_min = -cylinder_radius
    z_min = -cylinder_radius
    transverse_width = 2.0 * cylinder_radius
    n_x = max(1, int(math.ceil(box_length_x / cell_size)))
    n_y = max(1, int(math.ceil(transverse_width / cell_size)))
    n_z = max(1, int(math.ceil(transverse_width / cell_size)))
    n_cells = n_x * n_y * n_z
    search_cells = int(math.ceil(max_radius / cell_size))
    chirality_sq_radius = chirality_radius * chirality_radius
    bin_edges_sq = bin_edges * bin_edges

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

            bin_q_xx_sums = np.zeros(n_bins, dtype=np.float64)
            bin_q_xtheta_sums = np.zeros(n_bins, dtype=np.float64)
            bin_s_counts = np.zeros(n_bins, dtype=np.int64)
            bin_hexatic_sums = np.zeros(n_bins, dtype=np.float64)
            bin_hexatic_counts = np.zeros(n_bins, dtype=np.int64)
            bin_chirality_sums = np.zeros(n_bins, dtype=np.float64)
            bin_chirality_counts = np.zeros(n_bins, dtype=np.int64)

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
                            if distance_sq >= bin_edges_sq[-1]:
                                particle_idx = next_index[particle_idx]
                                continue

                            bin_idx = -1
                            for candidate_bin in range(n_bins):
                                if (
                                    distance_sq >= bin_edges_sq[candidate_bin]
                                    and distance_sq < bin_edges_sq[candidate_bin + 1]
                                ):
                                    bin_idx = candidate_bin
                                    break
                            if bin_idx < 0:
                                particle_idx = next_index[particle_idx]
                                continue

                            p_x = direction_cylindrical[frame_idx, particle_idx, 0]
                            p_theta = direction_cylindrical[frame_idx, particle_idx, 2]
                            norm = math.sqrt(p_x * p_x + p_theta * p_theta)
                            if math.isfinite(norm) and norm > 0.0:
                                u_x = p_x / norm
                                u_theta = p_theta / norm
                                bin_q_xx_sums[bin_idx] += 2.0 * u_x * u_x - 1.0
                                bin_q_xtheta_sums[bin_idx] += 2.0 * u_x * u_theta
                                bin_s_counts[bin_idx] += 1

                            hexatic_value = hexatic_abs[frame_idx, particle_idx]
                            if math.isfinite(hexatic_value):
                                bin_hexatic_sums[bin_idx] += hexatic_value
                                bin_hexatic_counts[bin_idx] += 1

                            chirality_value = chirality_squared[particle_idx]
                            if math.isfinite(chirality_value):
                                bin_chirality_sums[bin_idx] += chirality_value
                                bin_chirality_counts[bin_idx] += 1

                            particle_idx = next_index[particle_idx]

            for bin_idx in range(n_bins):
                if bin_s_counts[bin_idx] > 0:
                    s_sums[bin_idx] += _nematic_order_from_q(
                        bin_q_xx_sums[bin_idx],
                        bin_q_xtheta_sums[bin_idx],
                        bin_s_counts[bin_idx],
                    )
                    s_counts[bin_idx] += 1
                if bin_hexatic_counts[bin_idx] > 0:
                    hexatic_sums[bin_idx] += (
                        bin_hexatic_sums[bin_idx] / bin_hexatic_counts[bin_idx]
                    )
                    hexatic_counts[bin_idx] += 1
                if bin_chirality_counts[bin_idx] > 0:
                    chirality_sums[bin_idx] += (
                        bin_chirality_sums[bin_idx] / bin_chirality_counts[bin_idx]
                    )
                    chirality_counts[bin_idx] += 1
            if n_bins >= 3:
                annulus_count = bin_chirality_counts[1] + bin_chirality_counts[2]
                if annulus_count > 0:
                    annulus_chirality_sum += (
                        bin_chirality_sums[1] + bin_chirality_sums[2]
                    ) / annulus_count
                    annulus_chirality_count += 1

    s_profile = np.empty(n_bins, dtype=np.float64)
    hexatic_profile = np.empty(n_bins, dtype=np.float64)
    chirality_profile = np.empty(n_bins, dtype=np.float64)
    for bin_idx in range(n_bins):
        s_profile[bin_idx] = (
            s_sums[bin_idx] / s_counts[bin_idx]
            if s_counts[bin_idx] > 0
            else np.nan
        )
        hexatic_profile[bin_idx] = (
            hexatic_sums[bin_idx] / hexatic_counts[bin_idx]
            if hexatic_counts[bin_idx] > 0
            else np.nan
        )
        chirality_profile[bin_idx] = (
            chirality_sums[bin_idx] / chirality_counts[bin_idx]
            if chirality_counts[bin_idx] > 0
            else np.nan
        )

    chirality_annulus = (
        annulus_chirality_sum / annulus_chirality_count
        if annulus_chirality_count > 0
        else np.nan
    )
    return s_profile, hexatic_profile, chirality_profile, chirality_annulus


@njit(cache=True)
def moving_defect_frontback_chirality(
    coords: np.ndarray,
    disclination_mask: np.ndarray,
    hexatic_abs: np.ndarray,
    steps: np.ndarray,
    frame_start: int,
    frame_stop: int,
    core_radius: float,
    annulus_outer_radius: float,
    chirality_radius: float,
    box_length_x: float,
    x_min: float,
    cylinder_radius: float,
    timestep: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    start = min(max(frame_start, 1), coords.shape[0])
    stop = min(max(frame_stop, start), coords.shape[0])
    cell_size = max(chirality_radius, core_radius)
    if cell_size <= 0.0:
        empty = np.empty(0, dtype=np.float64)
        return (
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        )

    n_particles = coords.shape[1]
    max_samples = max(0, stop - start) * n_particles
    speeds = np.empty(max_samples, dtype=np.float64)
    local_speeds = np.empty(max_samples, dtype=np.float64)
    defect_local_cosines = np.empty(max_samples, dtype=np.float64)
    residual_speeds = np.empty(max_samples, dtype=np.float64)
    residual_vx_values = np.empty(max_samples, dtype=np.float64)
    residual_vy_values = np.empty(max_samples, dtype=np.float64)
    residual_vz_values = np.empty(max_samples, dtype=np.float64)
    abs_delta_chirality = np.empty(max_samples, dtype=np.float64)
    velocity_direction = np.empty(max_samples, dtype=np.float64)
    delta_chirality_sign = np.empty(max_samples, dtype=np.float64)
    d_chi_x_values = np.empty(max_samples, dtype=np.float64)
    d_chi_y_values = np.empty(max_samples, dtype=np.float64)
    d_chi_z_values = np.empty(max_samples, dtype=np.float64)
    d_psi6_x_values = np.empty(max_samples, dtype=np.float64)
    d_psi6_y_values = np.empty(max_samples, dtype=np.float64)
    d_psi6_z_values = np.empty(max_samples, dtype=np.float64)
    residual_dot_d_chi_hat = np.empty(max_samples, dtype=np.float64)
    residual_dot_d_psi6_hat = np.empty(max_samples, dtype=np.float64)
    residual_cos_d_chi = np.empty(max_samples, dtype=np.float64)
    residual_cos_d_psi6 = np.empty(max_samples, dtype=np.float64)
    sample_count = 0

    y_min = -cylinder_radius
    z_min = -cylinder_radius
    transverse_width = 2.0 * cylinder_radius
    n_x = max(1, int(math.ceil(box_length_x / cell_size)))
    n_y = max(1, int(math.ceil(transverse_width / cell_size)))
    n_z = max(1, int(math.ceil(transverse_width / cell_size)))
    n_cells = n_x * n_y * n_z
    search_cells = int(math.ceil(annulus_outer_radius / cell_size))

    core_sq = core_radius * core_radius
    outer_sq = annulus_outer_radius * annulus_outer_radius
    chirality_sq_radius = chirality_radius * chirality_radius

    for frame_idx in range(start, stop):
        delta_t = (steps[frame_idx] - steps[frame_idx - 1]) * timestep
        if delta_t <= 0.0:
            continue

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

        chirality = np.full(n_particles, np.nan, dtype=np.float64)
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
            value = 0.0

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
                                    value += dx / math.sqrt(distance_sq)
                            neighbor_idx = next_index[neighbor_idx]

            chirality[particle_idx] = value

        for defect_idx in range(n_particles):
            if (
                not finite_position[defect_idx]
                or not disclination_mask[frame_idx, defect_idx]
            ):
                continue

            prev_x = coords[frame_idx - 1, defect_idx, 0]
            prev_theta = coords[frame_idx - 1, defect_idx, 1]
            prev_radius = coords[frame_idx - 1, defect_idx, 2]
            if (
                not math.isfinite(prev_x)
                or not math.isfinite(prev_theta)
                or not math.isfinite(prev_radius)
            ):
                continue

            x_i = x_values[defect_idx]
            y_i = y_values[defect_idx]
            z_i = z_values[defect_idx]
            prev_y = prev_radius * math.sin(prev_theta)
            prev_z = prev_radius * math.cos(prev_theta)
            vx = x_i - prev_x
            if box_length_x > 0.0:
                vx -= box_length_x * round(vx / box_length_x)
            vy = y_i - prev_y
            vz = z_i - prev_z
            displacement = math.sqrt(vx * vx + vy * vy + vz * vz)
            if not math.isfinite(displacement) or displacement <= 0.0:
                continue

            inv_displacement = 1.0 / displacement
            e_x = vx * inv_displacement
            e_y = vy * inv_displacement
            e_z = vz * inv_displacement

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

            front_sum = 0.0
            front_count = 0
            back_sum = 0.0
            back_count = 0
            local_vx_sum = 0.0
            local_vy_sum = 0.0
            local_vz_sum = 0.0
            local_velocity_count = 0
            annulus_chirality_sum = 0.0
            annulus_chirality_count = 0
            annulus_psi6_sum = 0.0
            annulus_psi6_count = 0

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
                            if distance_sq <= core_sq or distance_sq >= outer_sq:
                                particle_idx = next_index[particle_idx]
                                continue

                            prev_particle_x = coords[frame_idx - 1, particle_idx, 0]
                            prev_particle_theta = coords[
                                frame_idx - 1, particle_idx, 1
                            ]
                            prev_particle_radius = coords[
                                frame_idx - 1, particle_idx, 2
                            ]
                            if (
                                math.isfinite(prev_particle_x)
                                and math.isfinite(prev_particle_theta)
                                and math.isfinite(prev_particle_radius)
                            ):
                                particle_vx = x_values[particle_idx] - prev_particle_x
                                if box_length_x > 0.0:
                                    particle_vx -= box_length_x * round(
                                        particle_vx / box_length_x
                                    )
                                prev_particle_y = prev_particle_radius * math.sin(
                                    prev_particle_theta
                                )
                                prev_particle_z = prev_particle_radius * math.cos(
                                    prev_particle_theta
                                )
                                local_vx_sum += particle_vx
                                local_vy_sum += (
                                    y_values[particle_idx] - prev_particle_y
                                )
                                local_vz_sum += (
                                    z_values[particle_idx] - prev_particle_z
                                )
                                local_velocity_count += 1

                            chirality_value = chirality[particle_idx]
                            if math.isfinite(chirality_value):
                                annulus_chirality_sum += chirality_value
                                annulus_chirality_count += 1
                                projection = dx * e_x + dy * e_y + dz * e_z
                                if projection > 0.0:
                                    front_sum += chirality_value
                                    front_count += 1
                                elif projection < 0.0:
                                    back_sum += chirality_value
                                    back_count += 1

                            psi6_value = hexatic_abs[frame_idx, particle_idx]
                            if math.isfinite(psi6_value):
                                annulus_psi6_sum += psi6_value
                                annulus_psi6_count += 1

                            particle_idx = next_index[particle_idx]

            if local_velocity_count > 0:
                speeds[sample_count] = displacement / delta_t
                local_vx = local_vx_sum / local_velocity_count
                local_vy = local_vy_sum / local_velocity_count
                local_vz = local_vz_sum / local_velocity_count
                local_displacement = math.sqrt(
                    local_vx * local_vx + local_vy * local_vy + local_vz * local_vz
                )
                local_speeds[sample_count] = local_displacement / delta_t
                if local_displacement > 0.0:
                    defect_local_cosines[sample_count] = (
                        vx * local_vx + vy * local_vy + vz * local_vz
                    ) / (displacement * local_displacement)
                else:
                    defect_local_cosines[sample_count] = np.nan
                residual_vx = vx - local_vx
                residual_vy = vy - local_vy
                residual_vz = vz - local_vz
                residual_speed = (
                    math.sqrt(
                        residual_vx * residual_vx
                        + residual_vy * residual_vy
                        + residual_vz * residual_vz
                    )
                    / delta_t
                )
                residual_speeds[sample_count] = residual_speed
                residual_vx_values[sample_count] = residual_vx / delta_t
                residual_vy_values[sample_count] = residual_vy / delta_t
                residual_vz_values[sample_count] = residual_vz / delta_t

                if front_count > 0 and back_count > 0:
                    delta_chirality = front_sum / front_count - back_sum / back_count
                    abs_delta_chirality[sample_count] = abs(delta_chirality)
                    if delta_chirality > 0.0:
                        delta_chirality_sign[sample_count] = 1.0
                    elif delta_chirality < 0.0:
                        delta_chirality_sign[sample_count] = -1.0
                    else:
                        delta_chirality_sign[sample_count] = 0.0
                else:
                    abs_delta_chirality[sample_count] = np.nan
                    delta_chirality_sign[sample_count] = np.nan

                if vx > 0.0:
                    velocity_direction[sample_count] = 1.0
                elif vx < 0.0:
                    velocity_direction[sample_count] = -1.0
                else:
                    velocity_direction[sample_count] = 0.0

                d_chi_x = 0.0
                d_chi_y = 0.0
                d_chi_z = 0.0
                d_chi_count = 0
                d_psi6_x = 0.0
                d_psi6_y = 0.0
                d_psi6_z = 0.0
                d_psi6_count = 0
                mean_chirality = (
                    annulus_chirality_sum / annulus_chirality_count
                    if annulus_chirality_count > 0
                    else np.nan
                )
                mean_psi6 = (
                    annulus_psi6_sum / annulus_psi6_count
                    if annulus_psi6_count > 0
                    else np.nan
                )

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
                                if distance_sq <= core_sq or distance_sq >= outer_sq:
                                    particle_idx = next_index[particle_idx]
                                    continue

                                distance = math.sqrt(distance_sq)
                                if distance > 0.0:
                                    r_hat_x = dx / distance
                                    r_hat_y = dy / distance
                                    r_hat_z = dz / distance
                                    chirality_value = chirality[particle_idx]
                                    if (
                                        math.isfinite(chirality_value)
                                        and math.isfinite(mean_chirality)
                                    ):
                                        delta_value = chirality_value - mean_chirality
                                        d_chi_x += delta_value * r_hat_x
                                        d_chi_y += delta_value * r_hat_y
                                        d_chi_z += delta_value * r_hat_z
                                        d_chi_count += 1

                                    psi6_value = hexatic_abs[frame_idx, particle_idx]
                                    if (
                                        math.isfinite(psi6_value)
                                        and math.isfinite(mean_psi6)
                                    ):
                                        delta_value = psi6_value - mean_psi6
                                        d_psi6_x += delta_value * r_hat_x
                                        d_psi6_y += delta_value * r_hat_y
                                        d_psi6_z += delta_value * r_hat_z
                                        d_psi6_count += 1

                                particle_idx = next_index[particle_idx]

                if d_chi_count > 0:
                    d_chi_x /= d_chi_count
                    d_chi_y /= d_chi_count
                    d_chi_z /= d_chi_count
                    d_chi_norm = math.sqrt(
                        d_chi_x * d_chi_x + d_chi_y * d_chi_y + d_chi_z * d_chi_z
                    )
                    if d_chi_norm > 0.0 and residual_speed > 0.0:
                        residual_dot_d_chi_hat[sample_count] = (
                            residual_vx_values[sample_count] * d_chi_x
                            + residual_vy_values[sample_count] * d_chi_y
                            + residual_vz_values[sample_count] * d_chi_z
                        ) / d_chi_norm
                        residual_cos_d_chi[sample_count] = (
                            residual_dot_d_chi_hat[sample_count] / residual_speed
                        )
                    else:
                        residual_dot_d_chi_hat[sample_count] = np.nan
                        residual_cos_d_chi[sample_count] = np.nan
                else:
                    d_chi_x = np.nan
                    d_chi_y = np.nan
                    d_chi_z = np.nan
                    residual_dot_d_chi_hat[sample_count] = np.nan
                    residual_cos_d_chi[sample_count] = np.nan

                if d_psi6_count > 0:
                    d_psi6_x /= d_psi6_count
                    d_psi6_y /= d_psi6_count
                    d_psi6_z /= d_psi6_count
                    d_psi6_norm = math.sqrt(
                        d_psi6_x * d_psi6_x
                        + d_psi6_y * d_psi6_y
                        + d_psi6_z * d_psi6_z
                    )
                    if d_psi6_norm > 0.0 and residual_speed > 0.0:
                        residual_dot_d_psi6_hat[sample_count] = (
                            residual_vx_values[sample_count] * d_psi6_x
                            + residual_vy_values[sample_count] * d_psi6_y
                            + residual_vz_values[sample_count] * d_psi6_z
                        ) / d_psi6_norm
                        residual_cos_d_psi6[sample_count] = (
                            residual_dot_d_psi6_hat[sample_count] / residual_speed
                        )
                    else:
                        residual_dot_d_psi6_hat[sample_count] = np.nan
                        residual_cos_d_psi6[sample_count] = np.nan
                else:
                    d_psi6_x = np.nan
                    d_psi6_y = np.nan
                    d_psi6_z = np.nan
                    residual_dot_d_psi6_hat[sample_count] = np.nan
                    residual_cos_d_psi6[sample_count] = np.nan

                d_chi_x_values[sample_count] = d_chi_x
                d_chi_y_values[sample_count] = d_chi_y
                d_chi_z_values[sample_count] = d_chi_z
                d_psi6_x_values[sample_count] = d_psi6_x
                d_psi6_y_values[sample_count] = d_psi6_y
                d_psi6_z_values[sample_count] = d_psi6_z
                sample_count += 1

    return (
        speeds[:sample_count],
        local_speeds[:sample_count],
        defect_local_cosines[:sample_count],
        residual_speeds[:sample_count],
        residual_vx_values[:sample_count],
        residual_vy_values[:sample_count],
        residual_vz_values[:sample_count],
        abs_delta_chirality[:sample_count],
        velocity_direction[:sample_count],
        delta_chirality_sign[:sample_count],
        d_chi_x_values[:sample_count],
        d_chi_y_values[:sample_count],
        d_chi_z_values[:sample_count],
        d_psi6_x_values[:sample_count],
        d_psi6_y_values[:sample_count],
        d_psi6_z_values[:sample_count],
        residual_dot_d_chi_hat[:sample_count],
        residual_dot_d_psi6_hat[:sample_count],
        residual_cos_d_chi[:sample_count],
        residual_cos_d_psi6[:sample_count],
    )


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
