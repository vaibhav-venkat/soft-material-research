mod libraries;
mod math;
mod operators;
mod sampling;
mod validation;

use ndarray::{
    Array3, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD, IxDyn,
};

use crate::geometry::{gaussian_2d, minimum_image};
use crate::{CoreError, CoreResult};

pub use libraries::build_mechanical_libraries;
pub use sampling::sample_component_rows;

use math::delta;
use validation::validate_particle_fields;

pub struct MechanicalFields {
    pub rho: Array3<f64>,
    pub p: Array4<f64>,
    pub q: Array5<f64>,
    pub a: Array5<f64>,
    pub psi6_sq: Array3<f64>,
    pub j_rho: Array4<f64>,
    pub j_p: Array5<f64>,
    pub j_q: Array6<f64>,
}

pub struct MechanicalLibraries {
    pub y_rho_names: Vec<String>,
    pub y_p_names: Vec<String>,
    pub y_q_names: Vec<String>,
    pub y_rho: ArrayD<f64>,
    pub y_p: ArrayD<f64>,
    pub y_q: ArrayD<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn build_mechanical_fields(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    psi6_abs: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
    gamma: f64,
    u0: f64,
) -> CoreResult<MechanicalFields> {
    validate_particle_fields(
        coords, directions, velocities, psi6_abs, mask, x_centers, y_centers, lx, ly, radius,
        sigma, gamma, u0,
    )?;
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ny = y_centers.len();
    let mut rho = Array3::<f64>::zeros((frames, nx, ny));
    let mut psi6 = Array3::<f64>::zeros((frames, nx, ny));
    let mut p = Array4::<f64>::zeros((frames, nx, ny, 3));
    let mut q = Array5::<f64>::zeros((frames, nx, ny, 3, 3));
    let mut j_rho = Array4::<f64>::zeros((frames, nx, ny, 2));
    let mut j_p = Array5::<f64>::zeros((frames, nx, ny, 2, 3));
    let mut j_q = Array6::<f64>::zeros((frames, nx, ny, 2, 3, 3));
    let cutoff = 4.0 * sigma;
    let cutoff2 = cutoff * cutoff;

    for t in 0..frames {
        println!(
            "[rho_fitting] mechanical coarse-grain frame {}/{}",
            t + 1,
            frames
        );
        for particle in 0..particles {
            if !mask[[t, particle]] {
                continue;
            }
            let particle_x = coords[[t, particle, 0]];
            let particle_y = radius * coords[[t, particle, 1]];
            let px = directions[[t, particle, 0]];
            let py = directions[[t, particle, 1]];
            let pz = directions[[t, particle, 2]];
            let vx = velocities[[t, particle, 0]];
            let vy = velocities[[t, particle, 1]];
            let psi6_value = psi6_abs[[t, particle]];
            if !(particle_x.is_finite()
                && particle_y.is_finite()
                && px.is_finite()
                && py.is_finite()
                && pz.is_finite()
                && vx.is_finite()
                && vy.is_finite()
                && psi6_value.is_finite())
            {
                continue;
            }
            let dir = [px, py, pz];
            let vel = [vx, vy];
            let mut q_particle = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    q_particle[i][j] = dir[i] * dir[j] - delta(i, j) / 3.0;
                }
            }
            for ix in 0..nx {
                let dx = minimum_image(x_centers[ix] - particle_x, lx);
                if dx.abs() > cutoff {
                    continue;
                }
                for iy in 0..ny {
                    let dy = minimum_image(y_centers[iy] - particle_y, ly);
                    if dy.abs() > cutoff || dx * dx + dy * dy > cutoff2 {
                        continue;
                    }
                    let weight = gaussian_2d(dx, dy, sigma);
                    rho[[t, ix, iy]] += weight;
                    psi6[[t, ix, iy]] += weight * psi6_value;
                    for component in 0..3 {
                        p[[t, ix, iy, component]] += weight * dir[component];
                    }
                    for component in 0..2 {
                        j_rho[[t, ix, iy, component]] += weight * vel[component];
                    }
                    for k in 0..2 {
                        for i in 0..3 {
                            j_p[[t, ix, iy, k, i]] += weight * vel[k] * dir[i];
                        }
                    }
                    for i in 0..3 {
                        for j in 0..3 {
                            let q_value = q_particle[i][j];
                            q[[t, ix, iy, i, j]] += weight * q_value;
                            for k in 0..2 {
                                j_q[[t, ix, iy, k, i, j]] += weight * vel[k] * q_value;
                            }
                        }
                    }
                }
            }
        }
    }

    let mut a = q.clone();
    let mut psi6_sq = Array3::<f64>::from_elem((frames, nx, ny), f64::NAN);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                if rho[[t, ix, iy]].is_finite() && rho[[t, ix, iy]] > 0.0 {
                    let value = psi6[[t, ix, iy]] / rho[[t, ix, iy]];
                    psi6_sq[[t, ix, iy]] = value * value;
                }
                for component in 0..3 {
                    a[[t, ix, iy, component, component]] += rho[[t, ix, iy]] / 3.0;
                }
            }
        }
    }

    Ok(MechanicalFields {
        rho,
        p,
        q,
        a,
        psi6_sq,
        j_rho,
        j_p,
        j_q,
    })
}

pub fn build_targets(
    p: ArrayViewD<'_, f64>,
    j_rho: ArrayViewD<'_, f64>,
    j_p: ArrayViewD<'_, f64>,
    j_q: ArrayViewD<'_, f64>,
    gamma: f64,
    u0: f64,
) -> CoreResult<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>)> {
    if p.ndim() != 4 || p.shape()[3] != 3 {
        return Err(CoreError::Shape(
            "P must have shape (T,Nx,Ny,3)".to_string(),
        ));
    }
    let expected_j_rho = [p.shape()[0], p.shape()[1], p.shape()[2], 2];
    if j_rho.shape() != expected_j_rho {
        return Err(CoreError::Shape(
            "J_rho must have shape (T,Nx,Ny,2)".to_string(),
        ));
    }
    let expected_j_p = [p.shape()[0], p.shape()[1], p.shape()[2], 2, 3];
    let expected_j_q = [p.shape()[0], p.shape()[1], p.shape()[2], 2, 3, 3];
    if j_p.shape() != expected_j_p {
        return Err(CoreError::Shape(
            "J_P must have shape (T,Nx,Ny,2,3)".to_string(),
        ));
    }
    if j_q.shape() != expected_j_q {
        return Err(CoreError::Shape(
            "J_Q must have shape (T,Nx,Ny,2,3,3)".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }
    let mut p_surface = ArrayD::<f64>::zeros(IxDyn(&expected_j_rho));
    for t in 0..p.shape()[0] {
        for ix in 0..p.shape()[1] {
            for iy in 0..p.shape()[2] {
                for k in 0..2 {
                    p_surface[IxDyn(&[t, ix, iy, k])] = p[IxDyn(&[t, ix, iy, k])];
                }
            }
        }
    }
    let y_rho = (&j_rho - &(p_surface * u0)) * gamma;
    let y_p = j_p.to_owned() / u0;
    let y_q = j_q.to_owned();
    Ok((y_rho, y_p, y_q))
}
