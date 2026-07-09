mod math;
mod sampling;
mod validation;

use ndarray::{
    Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD, IxDyn,
};

use crate::geometry::{gaussian_3d, minimum_image};
use crate::{CoreError, CoreResult};

pub use sampling::sample_component_rows;

use math::delta;
use validation::{validate_particle_fields, ParticleFieldInputs};

const ORIENTATION_DIMS: usize = 3;

#[non_exhaustive]
pub struct MechanicalFields {
    /// Coarse-grained scalar density and mechanical moments/currents on the 3D cylindrical grid.
    pub rho: Array4<f64>,
    pub p: Array5<f64>,
    pub q: Array6<f64>,
    pub a: Array6<f64>,
    pub psi6_sq: Array4<f64>,
    pub j_rho: Array5<f64>,
    pub j_p: Array6<f64>,
    pub j_q: ArrayD<f64>,
}

/// Coarse-grain particle positions, orientations, velocities, and hexatic order into fields.
///
/// Particle arrays use `(T, N, component)` axes, grid centers define the
/// periodic cylindrical volume, and the returned fields use 3D moment
/// components with 3D flux directions.
///
/// Example: `build_mechanical_fields(coords, dirs, vels, psi6, mask, xs, thetas, rs, lx, 2*pi, sigma, gamma, u0)`.
///
/// Edge cases: particles with non-finite position, direction, velocity, or
/// hexatic value are skipped; `psi6_sq` is left `NaN` where coarse-grained
/// density is zero.
#[allow(clippy::too_many_arguments)]
pub fn build_mechanical_fields(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    psi6_abs: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    theta_centers: ArrayView1<'_, f64>,
    r_centers: ArrayView1<'_, f64>,
    lx: f64,
    theta_period: f64,
    sigma: f64,
    gamma: f64,
    u0: f64,
) -> CoreResult<MechanicalFields> {
    validate_particle_fields(ParticleFieldInputs {
        coords,
        directions,
        velocities,
        psi6_abs,
        mask,
        x_centers,
        theta_centers,
        r_centers,
        lx,
        theta_period,
        sigma,
        gamma,
        u0,
    })?;
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ntheta = theta_centers.len();
    let nr = r_centers.len();
    let mut rho = Array4::<f64>::zeros((frames, nx, ntheta, nr));
    let mut psi6 = Array4::<f64>::zeros((frames, nx, ntheta, nr));
    let mut p = Array5::<f64>::zeros((frames, nx, ntheta, nr, 3));
    let mut q = Array6::<f64>::zeros((frames, nx, ntheta, nr, 3, 3));
    let mut j_rho = Array5::<f64>::zeros((frames, nx, ntheta, nr, 3));
    let mut j_p = Array6::<f64>::zeros((frames, nx, ntheta, nr, 3, 3));
    let mut j_q = ArrayD::<f64>::zeros(IxDyn(&[frames, nx, ntheta, nr, 3, 3, 3]));
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
            let particle_theta = coords[[t, particle, 1]];
            let particle_r = coords[[t, particle, 2]];
            let px = directions[[t, particle, 0]];
            let py = directions[[t, particle, 1]];
            let pz = directions[[t, particle, 2]];
            let vx = velocities[[t, particle, 0]];
            let vy = velocities[[t, particle, 1]];
            let vz = velocities[[t, particle, 2]];
            let psi6_value = psi6_abs[[t, particle]];
            if !(particle_x.is_finite()
                && particle_theta.is_finite()
                && particle_r.is_finite()
                && px.is_finite()
                && py.is_finite()
                && pz.is_finite()
                && vx.is_finite()
                && vy.is_finite()
                && vz.is_finite()
                && psi6_value.is_finite())
            {
                continue;
            }
            let dir = [px, py, pz];
            let vel = [vx, vy, vz];
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
                for itheta in 0..ntheta {
                    let dtheta =
                        minimum_image(theta_centers[itheta] - particle_theta, theta_period);
                    for ir in 0..nr {
                        let r_value = r_centers[ir];
                        let dy = r_value * dtheta;
                        let dr = r_value - particle_r;
                        if dy.abs() > cutoff
                            || dr.abs() > cutoff
                            || dx * dx + dy * dy + dr * dr > cutoff2
                        {
                            continue;
                        }
                        let weight = gaussian_3d(dx, dy, dr, sigma);
                        rho[[t, ix, itheta, ir]] += weight;
                        psi6[[t, ix, itheta, ir]] += weight * psi6_value;
                        for component in 0..3 {
                            p[[t, ix, itheta, ir, component]] += weight * dir[component];
                        }
                        for component in 0..3 {
                            j_rho[[t, ix, itheta, ir, component]] += weight * vel[component];
                        }
                        for k in 0..3 {
                            for i in 0..3 {
                                j_p[[t, ix, itheta, ir, k, i]] += weight * vel[k] * dir[i];
                            }
                        }
                        for i in 0..3 {
                            for j in 0..3 {
                                let q_value = q_particle[i][j];
                                q[[t, ix, itheta, ir, i, j]] += weight * q_value;
                                for k in 0..3 {
                                    j_q[IxDyn(&[t, ix, itheta, ir, k, i, j])] +=
                                        weight * vel[k] * q_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut a = q.clone();
    let mut psi6_sq = Array4::<f64>::from_elem((frames, nx, ntheta, nr), f64::NAN);
    for t in 0..frames {
        for ix in 0..nx {
            for itheta in 0..ntheta {
                for ir in 0..nr {
                    if rho[[t, ix, itheta, ir]].is_finite() && rho[[t, ix, itheta, ir]] > 0.0 {
                        let value = psi6[[t, ix, itheta, ir]] / rho[[t, ix, itheta, ir]];
                        psi6_sq[[t, ix, itheta, ir]] = value * value;
                    }
                    for component in 0..3 {
                        a[[t, ix, itheta, ir, component, component]] +=
                            rho[[t, ix, itheta, ir]] / 3.0;
                    }
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

/// Convert measured current tensors into mechanical closure targets.
///
/// `P` must be `(T,Nx,Ntheta,Nr,3)`, `J_rho` `(T,Nx,Ntheta,Nr,3)`, `J_P`
/// `(T,Nx,Ntheta,Nr,3,3)`, and `J_Q` `(T,Nx,Ntheta,Nr,3,3,3)`. Returns `(Y_rho,
/// Y_P, Y_Q)` with the same target shapes.
///
/// Edge cases: `u0` must be nonzero because `Y_P` divides by propulsion
/// speed; `gamma` is allowed to be negative only if finite.
pub fn build_targets(
    p: ArrayViewD<'_, f64>,
    j_rho: ArrayViewD<'_, f64>,
    j_p: ArrayViewD<'_, f64>,
    j_q: ArrayViewD<'_, f64>,
    gamma: f64,
    u0: f64,
) -> CoreResult<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>)> {
    if p.ndim() != 5 || p.shape()[4] != 3 {
        return Err(CoreError::Shape(
            "P must have shape (T,Nx,Ntheta,Nr,3)".to_string(),
        ));
    }
    let expected_j_rho = [
        p.shape()[0],
        p.shape()[1],
        p.shape()[2],
        p.shape()[3],
        ORIENTATION_DIMS,
    ];
    if j_rho.shape() != expected_j_rho {
        return Err(CoreError::Shape(
            "J_rho must have shape (T,Nx,Ntheta,Nr,3)".to_string(),
        ));
    }
    let expected_j_p = [
        p.shape()[0],
        p.shape()[1],
        p.shape()[2],
        p.shape()[3],
        ORIENTATION_DIMS,
        ORIENTATION_DIMS,
    ];
    let expected_j_q = [
        p.shape()[0],
        p.shape()[1],
        p.shape()[2],
        p.shape()[3],
        ORIENTATION_DIMS,
        ORIENTATION_DIMS,
        ORIENTATION_DIMS,
    ];
    if j_p.shape() != expected_j_p {
        return Err(CoreError::Shape(
            "J_P must have shape (T,Nx,Ntheta,Nr,3,3)".to_string(),
        ));
    }
    if j_q.shape() != expected_j_q {
        return Err(CoreError::Shape(
            "J_Q must have shape (T,Nx,Ntheta,Nr,3,3,3)".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }
    let mut p_full = ArrayD::<f64>::zeros(IxDyn(&expected_j_rho));
    for t in 0..p.shape()[0] {
        for ix in 0..p.shape()[1] {
            for itheta in 0..p.shape()[2] {
                for ir in 0..p.shape()[3] {
                    for k in 0..ORIENTATION_DIMS {
                        p_full[IxDyn(&[t, ix, itheta, ir, k])] = p[IxDyn(&[t, ix, itheta, ir, k])];
                    }
                }
            }
        }
    }
    let y_rho = (&j_rho - &(p_full * u0)) * gamma;
    let y_p = j_p.to_owned() / u0;
    let y_q = j_q.to_owned();
    Ok((y_rho, y_p, y_q))
}
