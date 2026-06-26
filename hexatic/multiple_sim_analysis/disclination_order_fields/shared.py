from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from numba import njit

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import HEXATIC_OUTPUT_DIR, RadiusCase

CIRCUMFERENCE_REFERENCE_CASE_ID = "circ_60_0D"
LOCAL_CONTRAST_LENGTH = cylinder.ANALYSIS.neighbor_count_radius
LOCAL_PROFILE_BIN_EDGES = LOCAL_CONTRAST_LENGTH * np.arange(6, dtype=np.float64)
LOCAL_PROFILE_BIN_LABELS = np.asarray(
    ("< a", "a-2a", "2a-3a", "3a-4a", "4a-5a")
)
LOCAL_PROFILE_COLORS = ("#111111", "#0072b2", "#d55e00", "#009e73", "#cc0000")


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
                                if distance_sq > 0.0 and distance_sq <= chirality_sq_radius:
                                    chirality += dx / math.sqrt(distance_sq)
                            neighbor_idx = next_index[neighbor_idx]

            chirality_squared[particle_idx] = chirality * chirality

        for defect_idx in range(n_particles):
            if not finite_position[defect_idx] or not disclination_mask[frame_idx, defect_idx]:
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
                core_s = _nematic_order_from_q(core_s_q_xx_sum, core_s_q_xtheta_sum, core_s_count)
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
    hexatic_delta = hexatic_delta_sum / hexatic_delta_count if hexatic_delta_count else np.nan
    chirality_delta = chirality_delta_sum / chirality_delta_count if chirality_delta_count else np.nan
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
                                if distance_sq > 0.0 and distance_sq <= chirality_sq_radius:
                                    chirality += dx / math.sqrt(distance_sq)
                            neighbor_idx = next_index[neighbor_idx]

            chirality_squared[particle_idx] = chirality * chirality

        for defect_idx in range(n_particles):
            if not finite_position[defect_idx] or not disclination_mask[frame_idx, defect_idx]:
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
                    hexatic_sums[bin_idx] += bin_hexatic_sums[bin_idx] / bin_hexatic_counts[bin_idx]
                    hexatic_counts[bin_idx] += 1
                if bin_chirality_counts[bin_idx] > 0:
                    chirality_sums[bin_idx] += bin_chirality_sums[bin_idx] / bin_chirality_counts[bin_idx]
                    chirality_counts[bin_idx] += 1
            if n_bins >= 3:
                annulus_count = bin_chirality_counts[1] + bin_chirality_counts[2]
                if annulus_count > 0:
                    annulus_chirality_sum += (bin_chirality_sums[1] + bin_chirality_sums[2]) / annulus_count
                    annulus_chirality_count += 1

    s_profile = np.empty(n_bins, dtype=np.float64)
    hexatic_profile = np.empty(n_bins, dtype=np.float64)
    chirality_profile = np.empty(n_bins, dtype=np.float64)
    for bin_idx in range(n_bins):
        s_profile[bin_idx] = s_sums[bin_idx] / s_counts[bin_idx] if s_counts[bin_idx] > 0 else np.nan
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


def hexatic_order_path(case: RadiusCase) -> Path:
    return HEXATIC_OUTPUT_DIR / f"{case.case_id}_hexatic_order.txt"


def _load_hexatic_abs(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim == 1:
        table = table[np.newaxis, :]
    if table.shape[1] < 6:
        raise ValueError(f"Hexatic order table is missing columns: {path}")

    frame_indices_table = table[:, 0].astype(np.int64)
    particle_indices = table[:, 2].astype(np.int64)
    if np.any(frame_indices_table >= shape[0]) or np.any(particle_indices >= shape[1]):
        raise ValueError(f"Hexatic order table does not match expected shape: {path}")

    psi_abs = np.full(shape, np.nan, dtype=np.float64)
    psi_abs[frame_indices_table, particle_indices] = table[:, 5]
    return psi_abs


def _disclination_mask(neighbor_counts: np.ndarray) -> np.ndarray:
    charges = cylinder.NEIGHBORS - neighbor_counts
    return np.abs(charges) == 1


def _validate_particle_frame_shape(
    name: str,
    values: np.ndarray,
    expected_shape: tuple[int, int],
) -> None:
    if values.shape[:2] != expected_shape:
        raise ValueError(
            f"{name} shape {values.shape[:2]} does not match "
            f"neighbor-count shape {expected_shape}."
        )
