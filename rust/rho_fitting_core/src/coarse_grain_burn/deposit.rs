use super::frame::FrameFields;
use super::grid::{flat_index, Grid3};
use super::EPS;

#[allow(clippy::too_many_arguments)]
pub(super) fn deposit_surface_shell_frame(
    x_centers: &[f32],
    theta_centers: &[f32],
    r_centers: &[f32],
    particle_x: &[f32],
    particle_theta: &[f32],
    particle_r: &[f32],
    valid: &[f32],
    components: &[Vec<f32>; 2],
    lx: f32,
    theta_period: f32,
    dx: f32,
    dtheta: f32,
    radial_width: f32,
    mass: &mut [f32],
    component_num: &mut [Vec<f32>; 2],
) -> usize {
    let mut deposited = 0;
    for particle in 0..particle_x.len() {
        if valid[particle] <= 0.0
            || !particle_x[particle].is_finite()
            || !particle_theta[particle].is_finite()
            || !particle_r[particle].is_finite()
        {
            continue;
        }
        let mut stencil = Vec::with_capacity(27);
        let mut norm = 0.0f32;
        let x_indices = periodic_stencil_indices(particle_x[particle], x_centers, dx, lx);
        let theta_indices = periodic_stencil_indices(
            particle_theta[particle],
            theta_centers,
            dtheta,
            theta_period,
        );
        let radial_indices = radial_stencil_indices(particle_r[particle], r_centers, radial_width);
        for ix in x_indices {
            let wx = tsc_weight_scalar(
                minimum_image_value(x_centers[ix] - particle_x[particle], lx) / dx,
            );
            if wx <= 0.0 {
                continue;
            }
            for itheta in theta_indices {
                let wt = tsc_weight_scalar(
                    minimum_image_value(
                        theta_centers[itheta] - particle_theta[particle],
                        theta_period,
                    ) / dtheta,
                );
                if wt <= 0.0 {
                    continue;
                }
                for ir in radial_indices.iter().flatten().copied() {
                    let wr =
                        tsc_weight_scalar((r_centers[ir] - particle_r[particle]) / radial_width);
                    let weight = wx * wt * wr;
                    if weight > 0.0 {
                        stencil.push((ix, itheta, ir, weight));
                        norm += weight;
                    }
                }
            }
        }
        if norm <= EPS {
            continue;
        }
        deposited += 1;
        for (ix, itheta, ir, weight) in stencil {
            let factor = weight / norm;
            let flat = flat_index(ix, itheta, ir, theta_centers.len(), r_centers.len());
            mass[flat] += factor;
            component_num[0][flat] += factor * components[0][particle];
            component_num[1][flat] += factor * components[1][particle];
        }
    }
    deposited
}

pub(super) fn collapse_surface_shell(
    mass: &[f32],
    component_num: &[Vec<f32>; 2],
    nx: usize,
    ntheta: usize,
    nr: usize,
) -> (Vec<f32>, [Vec<f32>; 2]) {
    let mut out_mass = vec![0.0; nx * ntheta];
    let mut out_components = [vec![0.0; nx * ntheta], vec![0.0; nx * ntheta]];
    for ix in 0..nx {
        for itheta in 0..ntheta {
            let out_flat = ix * ntheta + itheta;
            for ir in 0..nr {
                let flat = flat_index(ix, itheta, ir, ntheta, nr);
                out_mass[out_flat] += mass[flat];
                out_components[0][out_flat] += component_num[0][flat];
                out_components[1][out_flat] += component_num[1][flat];
            }
        }
    }
    (out_mass, out_components)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn deposit_mechanical_components(
    particle_x: &[f32],
    particle_theta: &[f32],
    particle_r: &[f32],
    valid: &[f32],
    dir: &[Vec<f32>; 3],
    vel: &[Vec<f32>; 3],
    psi6: &[f32],
    q_particle: &[Vec<f32>; 9],
    grid: &Grid3,
    fields: &mut FrameFields,
) -> usize {
    let mut deposited = 0;
    for particle in 0..particle_x.len() {
        if valid[particle] <= 0.0 {
            continue;
        }
        let x_indices =
            periodic_stencil_indices(particle_x[particle], &grid.x_centers, grid.dx, grid.lx);
        let theta_indices = periodic_stencil_indices(
            particle_theta[particle],
            &grid.theta_centers,
            grid.dtheta,
            grid.theta_period,
        );
        let radial_indices = radial_stencil_indices(particle_r[particle], &grid.r_centers, grid.dr);
        let mut stencil = Vec::with_capacity(27);
        let mut norm = 0.0f32;
        for ix in x_indices {
            let wx = tsc_weight_scalar(
                minimum_image_value(grid.x_centers[ix] - particle_x[particle], grid.lx) / grid.dx,
            );
            if wx <= 0.0 {
                continue;
            }
            for itheta in theta_indices {
                let wt = tsc_weight_scalar(
                    minimum_image_value(
                        grid.theta_centers[itheta] - particle_theta[particle],
                        grid.theta_period,
                    ) / grid.dtheta,
                );
                if wt <= 0.0 {
                    continue;
                }
                for ir in radial_indices.iter().flatten().copied() {
                    let wr =
                        tsc_weight_scalar((grid.r_centers[ir] - particle_r[particle]) / grid.dr);
                    let weight = wx * wt * wr;
                    if weight > 0.0 {
                        stencil.push((ix, itheta, ir, weight));
                        norm += weight;
                    }
                }
            }
        }
        if norm <= EPS {
            continue;
        }
        deposited += 1;
        for (ix, itheta, ir, weight) in stencil {
            let factor = weight / norm;
            let flat = flat_index(ix, itheta, ir, grid.ntheta, grid.nr);
            fields.mass[flat] += factor;
            fields.psi6_num[flat] += factor * psi6[particle];
            for component in 0..3 {
                fields.p_num[component][flat] += factor * dir[component][particle];
                fields.j_rho_num[component][flat] += factor * vel[component][particle];
            }
            for flux in 0..3 {
                for component in 0..3 {
                    fields.j_p_num[flux * 3 + component][flat] +=
                        factor * vel[flux][particle] * dir[component][particle];
                }
            }
            for row in 0..3 {
                for col in 0..3 {
                    let q_index = row * 3 + col;
                    let q_value = q_particle[q_index][particle];
                    fields.q_num[q_index][flat] += factor * q_value;
                    for flux in 0..3 {
                        fields.j_q_num[flux * 9 + q_index][flat] +=
                            factor * vel[flux][particle] * q_value;
                    }
                }
            }
        }
    }
    deposited
}

fn tsc_weight_scalar(s: f32) -> f32 {
    let abs_s = s.abs();
    if abs_s < 0.5 {
        0.75 - abs_s * abs_s
    } else if abs_s < 1.5 {
        let delta = 1.5 - abs_s;
        0.5 * delta * delta
    } else {
        0.0
    }
}

fn periodic_stencil_indices(value: f32, centers: &[f32], spacing: f32, period: f32) -> [usize; 3] {
    let nearest = (minimum_image_value(value - centers[0], period) / spacing).round() as isize;
    [
        wrap_index(nearest - 1, centers.len()),
        wrap_index(nearest, centers.len()),
        wrap_index(nearest + 1, centers.len()),
    ]
}

fn radial_stencil_indices(value: f32, centers: &[f32], spacing: f32) -> [Option<usize>; 3] {
    let nearest = ((value - centers[0]) / spacing).round() as isize;
    [
        checked_index(nearest - 1, centers.len()),
        checked_index(nearest, centers.len()),
        checked_index(nearest + 1, centers.len()),
    ]
}

fn wrap_index(index: isize, len: usize) -> usize {
    index.rem_euclid(len as isize) as usize
}

fn checked_index(index: isize, len: usize) -> Option<usize> {
    if index >= 0 && index < len as isize {
        Some(index as usize)
    } else {
        None
    }
}

fn minimum_image_value(delta: f32, period: f32) -> f32 {
    delta - (delta / period).round() * period
}
