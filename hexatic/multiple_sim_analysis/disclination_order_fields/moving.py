from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import float64, njit
from numba.experimental import jitclass

from hexatic.constants import cylinder
from hexatic.radii_analysis.cases import RadiusCase

from ..common import (
    FRAME_START,
    FRAME_STOP,
    NPZ_OUTPUT_DIR,
    PLOT_OUTPUT_DIR,
    active_fields_path,
    case_ids_for_cases,
    group_names_for_cases,
    labels_for_cases,
    load_active_fields,
    neighbor_counts_path,
    radii_for_cases,
)
from ..disclination import _load_neighbor_counts
from .shared import (
    LOCAL_CONTRAST_LENGTH,
    _cell_index,
    _disclination_mask,
    _load_hexatic_abs,
    _validate_particle_frame_shape,
    hexatic_order_path,
)


MOVING_VALUE_NAMES = (
    "speed",
    "abs_v_local",
    "cos_v_defect_v_local",
    "abs_v_residual",
    "v_residual_x",
    "v_residual_y",
    "v_residual_z",
    "abs_delta_chirality",
    "velocity_direction",
    "delta_chirality_sign",
    "d_chi_x",
    "d_chi_y",
    "d_chi_z",
    "d_psi6_x",
    "d_psi6_y",
    "d_psi6_z",
    "v_residual_dot_d_chi_hat",
    "v_residual_dot_d_psi6_hat",
    "cos_v_residual_d_chi",
    "cos_v_residual_d_psi6",
)


_MovingDefectFrontBackValuesSpec = [
    ("speed", float64[:]),
    ("abs_v_local", float64[:]),
    ("cos_v_defect_v_local", float64[:]),
    ("abs_v_residual", float64[:]),
    ("v_residual_x", float64[:]),
    ("v_residual_y", float64[:]),
    ("v_residual_z", float64[:]),
    ("abs_delta_chirality", float64[:]),
    ("velocity_direction", float64[:]),
    ("delta_chirality_sign", float64[:]),
    ("d_chi_x", float64[:]),
    ("d_chi_y", float64[:]),
    ("d_chi_z", float64[:]),
    ("d_psi6_x", float64[:]),
    ("d_psi6_y", float64[:]),
    ("d_psi6_z", float64[:]),
    ("v_residual_dot_d_chi_hat", float64[:]),
    ("v_residual_dot_d_psi6_hat", float64[:]),
    ("cos_v_residual_d_chi", float64[:]),
    ("cos_v_residual_d_psi6", float64[:]),
]


@jitclass(_MovingDefectFrontBackValuesSpec)
class MovingDefectFrontBackValues:
    def __init__(
        self,
        speed,
        abs_v_local,
        cos_v_defect_v_local,
        abs_v_residual,
        v_residual_x,
        v_residual_y,
        v_residual_z,
        abs_delta_chirality,
        velocity_direction,
        delta_chirality_sign,
        d_chi_x,
        d_chi_y,
        d_chi_z,
        d_psi6_x,
        d_psi6_y,
        d_psi6_z,
        v_residual_dot_d_chi_hat,
        v_residual_dot_d_psi6_hat,
        cos_v_residual_d_chi,
        cos_v_residual_d_psi6,
    ):
        self.speed = speed
        self.abs_v_local = abs_v_local
        self.cos_v_defect_v_local = cos_v_defect_v_local
        self.abs_v_residual = abs_v_residual
        self.v_residual_x = v_residual_x
        self.v_residual_y = v_residual_y
        self.v_residual_z = v_residual_z
        self.abs_delta_chirality = abs_delta_chirality
        self.velocity_direction = velocity_direction
        self.delta_chirality_sign = delta_chirality_sign
        self.d_chi_x = d_chi_x
        self.d_chi_y = d_chi_y
        self.d_chi_z = d_chi_z
        self.d_psi6_x = d_psi6_x
        self.d_psi6_y = d_psi6_y
        self.d_psi6_z = d_psi6_z
        self.v_residual_dot_d_chi_hat = v_residual_dot_d_chi_hat
        self.v_residual_dot_d_psi6_hat = v_residual_dot_d_psi6_hat
        self.cos_v_residual_d_chi = cos_v_residual_d_chi
        self.cos_v_residual_d_psi6 = cos_v_residual_d_psi6


def _moving_values_as_arrays(values: MovingDefectFrontBackValues) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(values, name), dtype=np.float64)
        for name in MOVING_VALUE_NAMES
    }


def _moving_values_from_arrays(arrays: dict[str, np.ndarray]) -> MovingDefectFrontBackValues:
    return MovingDefectFrontBackValues(*(np.asarray(arrays[name], dtype=np.float64) for name in MOVING_VALUE_NAMES))


@njit(cache=True)
def _moving_defect_frontback_chirality(
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
) -> MovingDefectFrontBackValues:
    start = min(max(frame_start, 1), coords.shape[0])
    stop = min(max(frame_stop, start), coords.shape[0])
    cell_size = max(chirality_radius, core_radius)
    if cell_size <= 0.0:
        empty = np.empty(0, dtype=np.float64)
        return MovingDefectFrontBackValues(
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
                                if distance_sq > 0.0 and distance_sq <= chirality_sq_radius:
                                    value += dx / math.sqrt(distance_sq)
                            neighbor_idx = next_index[neighbor_idx]

            chirality[particle_idx] = value

        for defect_idx in range(n_particles):
            if not finite_position[defect_idx] or not disclination_mask[frame_idx, defect_idx]:
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
                            prev_particle_theta = coords[frame_idx - 1, particle_idx, 1]
                            prev_particle_radius = coords[frame_idx - 1, particle_idx, 2]
                            if (
                                math.isfinite(prev_particle_x)
                                and math.isfinite(prev_particle_theta)
                                and math.isfinite(prev_particle_radius)
                            ):
                                particle_vx = x_values[particle_idx] - prev_particle_x
                                if box_length_x > 0.0:
                                    particle_vx -= box_length_x * round(particle_vx / box_length_x)
                                prev_particle_y = prev_particle_radius * math.sin(prev_particle_theta)
                                prev_particle_z = prev_particle_radius * math.cos(prev_particle_theta)
                                local_vx_sum += particle_vx
                                local_vy_sum += y_values[particle_idx] - prev_particle_y
                                local_vz_sum += z_values[particle_idx] - prev_particle_z
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
                local_displacement = math.sqrt(local_vx * local_vx + local_vy * local_vy + local_vz * local_vz)
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
                residual_speed = math.sqrt(
                    residual_vx * residual_vx
                    + residual_vy * residual_vy
                    + residual_vz * residual_vz
                ) / delta_t
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
                mean_psi6 = annulus_psi6_sum / annulus_psi6_count if annulus_psi6_count > 0 else np.nan

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
                                    if math.isfinite(chirality_value) and math.isfinite(mean_chirality):
                                        delta_value = chirality_value - mean_chirality
                                        d_chi_x += delta_value * r_hat_x
                                        d_chi_y += delta_value * r_hat_y
                                        d_chi_z += delta_value * r_hat_z
                                        d_chi_count += 1

                                    psi6_value = hexatic_abs[frame_idx, particle_idx]
                                    if math.isfinite(psi6_value) and math.isfinite(mean_psi6):
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
                    d_chi_norm = math.sqrt(d_chi_x * d_chi_x + d_chi_y * d_chi_y + d_chi_z * d_chi_z)
                    if d_chi_norm > 0.0 and residual_speed > 0.0:
                        residual_dot_d_chi_hat[sample_count] = (
                            residual_vx_values[sample_count] * d_chi_x
                            + residual_vy_values[sample_count] * d_chi_y
                            + residual_vz_values[sample_count] * d_chi_z
                        ) / d_chi_norm
                        residual_cos_d_chi[sample_count] = residual_dot_d_chi_hat[sample_count] / residual_speed
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
                        residual_cos_d_psi6[sample_count] = residual_dot_d_psi6_hat[sample_count] / residual_speed
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

    return MovingDefectFrontBackValues(
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


def moving_defect_frontback_values_for_case(
    case: RadiusCase,
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
) -> MovingDefectFrontBackValues:
    neighbor_counts = _load_neighbor_counts(neighbor_counts_path(case))
    expected_shape = neighbor_counts.shape
    disclinations = _disclination_mask(neighbor_counts)

    fields = load_active_fields(active_fields_path(case))
    coords = np.asarray(fields.coords, dtype=np.float64)
    _validate_particle_frame_shape("coords", coords, expected_shape)
    hexatic_abs = _load_hexatic_abs(hexatic_order_path(case), expected_shape)
    steps = np.asarray(fields.steps, dtype=np.int64)
    n_frames = min(
        coords.shape[0],
        disclinations.shape[0],
        hexatic_abs.shape[0],
        steps.shape[0],
    )
    coords = coords[:n_frames]
    disclinations = disclinations[:n_frames]
    hexatic_abs = hexatic_abs[:n_frames]
    steps = steps[:n_frames]

    x_edges = np.asarray(fields.x_edges, dtype=np.float64)
    box_length_x = float(x_edges[-1] - x_edges[0])
    return _moving_defect_frontback_chirality(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(disclinations, dtype=np.bool_),
        np.ascontiguousarray(hexatic_abs, dtype=np.float64),
        np.ascontiguousarray(steps, dtype=np.int64),
        frame_start,
        frame_stop,
        LOCAL_CONTRAST_LENGTH,
        3.0 * LOCAL_CONTRAST_LENGTH,
        cylinder.ANALYSIS.neighbor_count_radius,
        box_length_x,
        float(x_edges[0]),
        float(case.radius),
        float(cylinder.TIMESTEP),
    )


def _frontback_cache_matches(
    output_npz: Path,
    cases: tuple[RadiusCase, ...],
    frame_start: int,
    frame_stop: int,
) -> bool:
    with np.load(output_npz) as data:
        required = MOVING_VALUE_NAMES + (
            "sample_case_ids",
            "sample_radii",
            "case_ids",
            "frame_start",
            "frame_stop",
        )
        if not all(name in data for name in required):
            return False
        return (
            int(np.asarray(data["frame_start"]).item()) == int(frame_start)
            and int(np.asarray(data["frame_stop"]).item()) == int(frame_stop)
            and np.array_equal(np.asarray(data["case_ids"]), case_ids_for_cases(cases))
        )


def _load_frontback_cache(
    output_npz: Path,
) -> tuple[MovingDefectFrontBackValues, np.ndarray]:
    with np.load(output_npz) as data:
        arrays = {
            name: np.asarray(data[name], dtype=np.float64)
            for name in MOVING_VALUE_NAMES
        }
        return (
            _moving_values_from_arrays(arrays),
            np.asarray(data["sample_radii"], dtype=np.float64),
        )


def _save_frontback_cache(
    output_npz: Path,
    cases: tuple[RadiusCase, ...],
    arrays: dict[str, np.ndarray],
    frame_start: int,
    frame_stop: int,
) -> None:
    NPZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        metric_name="moving_defect_frontback_chirality",
        case_ids=case_ids_for_cases(cases),
        labels=labels_for_cases(cases),
        group_names=group_names_for_cases(cases),
        radii=radii_for_cases(cases),
        frame_start=int(frame_start),
        frame_stop=int(frame_stop),
        core_radius=np.asarray(LOCAL_CONTRAST_LENGTH, dtype=np.float64),
        annulus_inner_radius=np.asarray(LOCAL_CONTRAST_LENGTH, dtype=np.float64),
        annulus_outer_radius=np.asarray(3.0 * LOCAL_CONTRAST_LENGTH, dtype=np.float64),
        chirality_radius=np.asarray(
            cylinder.ANALYSIS.neighbor_count_radius,
            dtype=np.float64,
        ),
        **arrays,
    )


def _median_by_radius(
    sample_radii: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_radii = np.isfinite(sample_radii)
    radius_values = np.unique(sample_radii[finite_radii])
    medians = np.full(radius_values.shape, np.nan, dtype=np.float64)
    for radius_idx, radius in enumerate(radius_values):
        mask = (sample_radii == radius) & np.isfinite(values)
        if np.any(mask):
            medians[radius_idx] = float(np.median(values[mask]))
    return radius_values, medians


def _plot_local_vs_defect_speed(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    speed = np.asarray(arrays["speed"], dtype=np.float64)
    local_speed = np.asarray(arrays["abs_v_local"], dtype=np.float64)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)
    finite = (
        np.isfinite(local_speed)
        & np.isfinite(speed)
        & np.isfinite(sample_radii)
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    scatter = ax.scatter(
        local_speed[finite],
        speed[finite],
        c=sample_radii[finite],
        cmap="viridis",
        s=20,
        alpha=0.75,
        edgecolors="none",
    )
    ax.set_xlabel("|v_local|")
    ax.set_ylabel("|v_defect|")
    ax.set_title("Defect speed vs local annulus speed")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="R")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def _plot_defect_local_cosine_by_radius(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)
    cosines = np.asarray(arrays["cos_v_defect_v_local"], dtype=np.float64)
    radii, median_cosine = _median_by_radius(sample_radii, cosines)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(
        radii,
        median_cosine,
        "o-",
        color="#111111",
        label="median cos(v_defect, v_local)",
    )
    ax.axhline(0.0, color="#777777", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("R")
    ax.set_ylabel("median cos(v_defect, v_local)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Defect-local velocity alignment vs radius")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def _plot_residual_speed_ratio_by_radius(
    arrays: dict[str, np.ndarray],
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    speed = np.asarray(arrays["speed"], dtype=np.float64)
    residual_speed = np.asarray(arrays["abs_v_residual"], dtype=np.float64)
    sample_radii = np.asarray(arrays["sample_radii"], dtype=np.float64)
    ratio = np.divide(
        residual_speed,
        speed,
        out=np.full(residual_speed.shape, np.nan, dtype=np.float64),
        where=np.isfinite(speed) & (speed > 0.0),
    )
    radii, median_ratio = _median_by_radius(sample_radii, ratio)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(
        radii,
        median_ratio,
        "o-",
        color="#111111",
        label="median |v_residual| / |v_defect|",
    )
    ax.set_xlabel("R")
    ax.set_ylabel("median |v_residual| / |v_defect|")
    ax.set_title("Residual velocity ratio vs radius")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def run_moving_frontback_chirality(
    cases: tuple[RadiusCase, ...],
    frame_start: int = FRAME_START,
    frame_stop: int = FRAME_STOP,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    output_npz = NPZ_OUTPUT_DIR / "moving_defect_frontback_chirality.npz"
    ratio_output_png = PLOT_OUTPUT_DIR / "moving_defect_residual_speed_ratio.png"
    speed_scatter_output_png = PLOT_OUTPUT_DIR / "moving_defect_local_speed_scatter.png"
    cosine_output_png = PLOT_OUTPUT_DIR / "moving_defect_local_velocity_cosine.png"
    if (
        not overwrite
        and output_npz.exists()
        and _frontback_cache_matches(output_npz, cases, frame_start, frame_stop)
    ):
        cached_values, sample_radii = _load_frontback_cache(output_npz)
        arrays = _moving_values_as_arrays(cached_values)
        arrays["sample_radii"] = sample_radii
        if not ratio_output_png.exists():
            _plot_residual_speed_ratio_by_radius(arrays, ratio_output_png)
        if not speed_scatter_output_png.exists():
            _plot_local_vs_defect_speed(arrays, speed_scatter_output_png)
        if not cosine_output_png.exists():
            _plot_defect_local_cosine_by_radius(arrays, cosine_output_png)
        print(f"using cached moving-defect front/back chirality from {output_npz}")
        return arrays

    chunks: dict[str, list[np.ndarray]] = {name: [] for name in MOVING_VALUE_NAMES}
    chunks["sample_radii"] = []
    sample_case_ids: list[np.ndarray] = []
    for case in cases:
        case_values = moving_defect_frontback_values_for_case(
            case,
            frame_start,
            frame_stop,
        )
        case_arrays = _moving_values_as_arrays(case_values)
        n_samples = case_arrays["speed"].shape[0]
        for name in MOVING_VALUE_NAMES:
            chunks[name].append(case_arrays[name])
        chunks["sample_radii"].append(
            np.full(n_samples, float(case.radius), dtype=np.float64)
        )
        sample_case_ids.append(np.full(n_samples, case.case_id))

    arrays = {
        name: (
            np.concatenate(series)
            if series
            else np.asarray([], dtype=np.float64)
        )
        for name, series in chunks.items()
    }
    arrays["sample_case_ids"] = (
        np.concatenate(sample_case_ids) if sample_case_ids else np.asarray([])
    )
    _save_frontback_cache(output_npz, cases, arrays, frame_start, frame_stop)
    _plot_residual_speed_ratio_by_radius(arrays, ratio_output_png)
    _plot_local_vs_defect_speed(arrays, speed_scatter_output_png)
    _plot_defect_local_cosine_by_radius(arrays, cosine_output_png)
    return arrays
