use ndarray::{
    Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayView1, ArrayView2, ArrayView3, ArrayViewD,
    IxDyn,
};

use crate::fft_ops;
use crate::geometry::{gaussian_2d, minimum_image};
use crate::{CoreError, CoreResult};

pub struct MechanicalFields {
    pub rho: Array3<f64>,
    pub p: Array4<f64>,
    pub q: Array5<f64>,
    pub a: Array5<f64>,
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
        coords, directions, velocities, mask, x_centers, y_centers, lx, ly, radius, sigma, gamma,
        u0,
    )?;
    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ny = y_centers.len();
    let mut rho = Array3::<f64>::zeros((frames, nx, ny));
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
            if !(particle_x.is_finite()
                && particle_y.is_finite()
                && px.is_finite()
                && py.is_finite()
                && pz.is_finite()
                && vx.is_finite()
                && vy.is_finite())
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
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
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

pub fn build_mechanical_libraries(
    rho: ArrayView3<'_, f64>,
    p: ArrayViewD<'_, f64>,
    a: ArrayViewD<'_, f64>,
    f_rho: ArrayViewD<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<MechanicalLibraries> {
    validate_grid(rho, lx, ly)?;
    let p4 = p
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| CoreError::Shape("P must be (T,Nx,Ny,3)".to_string()))?;
    let a5 = a
        .into_dimensionality::<ndarray::Ix5>()
        .map_err(|_| CoreError::Shape("A must be (T,Nx,Ny,3,3)".to_string()))?;
    let f4 = f_rho
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| CoreError::Shape("F_rho must be (T,Nx,Ny,2)".to_string()))?;
    if p4.dim() != (rho.dim().0, rho.dim().1, rho.dim().2, 3)
        || f4.dim() != (rho.dim().0, rho.dim().1, rho.dim().2, 2)
        || a5.dim() != (rho.dim().0, rho.dim().1, rho.dim().2, 3, 3)
    {
        return Err(CoreError::Shape(
            "mechanical library field shapes do not align".to_string(),
        ));
    }

    let y_rho_names = vec![
        "grad_rho".to_string(),
        "rho_grad_rho".to_string(),
        "rho2_grad_rho".to_string(),
        "grad_lap_rho".to_string(),
        "lap_rho_grad_rho".to_string(),
        "grad_rho_cubed".to_string(),
        "grad_grad_rho_norm2".to_string(),
    ];
    let y_p_names = vec![
        "A".to_string(),
        "rho_A".to_string(),
        "rho2_A".to_string(),
        "rho3_A".to_string(),
    ];
    let y_q_names = vec![
        "P_dot_alpha".to_string(),
        "rho_P_dot_alpha".to_string(),
        "rho2_P_dot_alpha".to_string(),
        "P_dot_II".to_string(),
        "F_rho_I".to_string(),
    ];

    let y_rho = stack_vector_terms(&build_y_rho_terms(rho, lx, ly)?);
    let y_p = stack_rank2_terms(&build_y_p_terms(rho, a5));
    let y_q = stack_rank3_terms(&build_y_q_terms(rho, p4, f4));
    Ok(MechanicalLibraries {
        y_rho_names,
        y_p_names,
        y_q_names,
        y_rho,
        y_p,
        y_q,
    })
}

pub fn sample_component_rows(
    target: ArrayViewD<'_, f64>,
    library: ArrayViewD<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
) -> CoreResult<(Array2<f64>, Array2<i64>)> {
    if target.ndim() < 4 {
        return Err(CoreError::Shape(
            "target must have grid axes plus component axes".to_string(),
        ));
    }
    if library.ndim() != target.ndim() + 1 {
        return Err(CoreError::Shape(
            "library must have a leading term axis plus target axes".to_string(),
        ));
    }
    let terms = library.shape()[0];
    if library.shape()[1..] != *target.shape() {
        return Err(CoreError::Shape(
            "library term fields must match target shape".to_string(),
        ));
    }
    if sample_indices.dim().1 != 3 {
        return Err(CoreError::Shape("sample_indices must be (N,3)".to_string()));
    }

    let component_shape = &target.shape()[3..];
    let component_count = component_shape.iter().product::<usize>();
    let mut x = Array2::<f64>::zeros((sample_indices.dim().0 * component_count, terms + 1));
    let mut meta = Array2::<i64>::zeros((
        sample_indices.dim().0 * component_count,
        3 + component_shape.len(),
    ));

    for row in 0..sample_indices.dim().0 {
        let t = checked_index(sample_indices[[row, 0]], target.shape()[0], "frame")?;
        let ix = checked_index(sample_indices[[row, 1]], target.shape()[1], "x")?;
        let iy = checked_index(sample_indices[[row, 2]], target.shape()[2], "y")?;
        for component_flat in 0..component_count {
            let out_row = row * component_count + component_flat;
            let components = unravel_component(component_flat, component_shape);
            let mut target_index = vec![t, ix, iy];
            target_index.extend(components.iter().copied());
            x[[out_row, 0]] = target[IxDyn(&target_index)];
            for term in 0..terms {
                let mut library_index = vec![term, t, ix, iy];
                library_index.extend(components.iter().copied());
                x[[out_row, term + 1]] = library[IxDyn(&library_index)];
            }
            meta[[out_row, 0]] = t as i64;
            meta[[out_row, 1]] = ix as i64;
            meta[[out_row, 2]] = iy as i64;
            for (component_axis, component) in components.iter().enumerate() {
                meta[[out_row, 3 + component_axis]] = *component as i64;
            }
        }
    }

    Ok((x, meta))
}

fn build_y_rho_terms(rho: ArrayView3<'_, f64>, lx: f64, ly: f64) -> CoreResult<Vec<Array4<f64>>> {
    let grad = fft_ops::gradient_scalar(rho, lx, ly)?;
    let lap = fft_ops::laplacian_scalar(rho, lx, ly)?;
    let grad_lap = fft_ops::gradient_scalar(lap.view(), lx, ly)?;
    let lap_grad = scalar_times_vector(lap.view(), grad.view());
    let grad_cubed = cubic_gradient_flux(grad.view());
    let norm2 = gradient_norm2(grad.view());
    let grad_norm2 = fft_ops::gradient_scalar(norm2.view(), lx, ly)?;
    Ok(vec![
        grad.clone(),
        rho_times_vector(rho, grad.view(), 1),
        rho_times_vector(rho, grad.view(), 2),
        grad_lap,
        lap_grad,
        grad_cubed,
        grad_norm2,
    ])
}

fn build_y_p_terms(rho: ArrayView3<'_, f64>, a: ndarray::ArrayView5<'_, f64>) -> Vec<Array5<f64>> {
    let a_surface = surface_rows_rank2(a);
    vec![
        a_surface.clone(),
        rho_times_rank2(rho, a_surface.view(), 1),
        rho_times_rank2(rho, a_surface.view(), 2),
        rho_times_rank2(rho, a_surface.view(), 3),
    ]
}

fn build_y_q_terms(
    rho: ArrayView3<'_, f64>,
    p: ndarray::ArrayView4<'_, f64>,
    f_rho: ndarray::ArrayView4<'_, f64>,
) -> Vec<Array6<f64>> {
    let p_alpha = p_dot_alpha(p);
    vec![
        p_alpha.clone(),
        rho_times_rank3(rho, p_alpha.view(), 1),
        rho_times_rank3(rho, p_alpha.view(), 2),
        p_dot_ii(p),
        vector_identity(f_rho),
    ]
}

fn stack_vector_terms(terms: &[Array4<f64>]) -> ArrayD<f64> {
    let shape = terms[0].dim();
    let mut out = ArrayD::<f64>::zeros(IxDyn(&[terms.len(), shape.0, shape.1, shape.2, shape.3]));
    for (term, values) in terms.iter().enumerate() {
        for t in 0..shape.0 {
            for ix in 0..shape.1 {
                for iy in 0..shape.2 {
                    for component in 0..shape.3 {
                        out[IxDyn(&[term, t, ix, iy, component])] = values[[t, ix, iy, component]];
                    }
                }
            }
        }
    }
    out
}

fn stack_rank2_terms(terms: &[Array5<f64>]) -> ArrayD<f64> {
    let shape = terms[0].dim();
    let mut out = ArrayD::<f64>::zeros(IxDyn(&[
        terms.len(),
        shape.0,
        shape.1,
        shape.2,
        shape.3,
        shape.4,
    ]));
    for (term, values) in terms.iter().enumerate() {
        for t in 0..shape.0 {
            for ix in 0..shape.1 {
                for iy in 0..shape.2 {
                    for i in 0..shape.3 {
                        for j in 0..shape.4 {
                            out[IxDyn(&[term, t, ix, iy, i, j])] = values[[t, ix, iy, i, j]];
                        }
                    }
                }
            }
        }
    }
    out
}

fn stack_rank3_terms(terms: &[Array6<f64>]) -> ArrayD<f64> {
    let shape = terms[0].dim();
    let mut out = ArrayD::<f64>::zeros(IxDyn(&[
        terms.len(),
        shape.0,
        shape.1,
        shape.2,
        shape.3,
        shape.4,
        shape.5,
    ]));
    for (term, values) in terms.iter().enumerate() {
        for t in 0..shape.0 {
            for ix in 0..shape.1 {
                for iy in 0..shape.2 {
                    for k in 0..shape.3 {
                        for i in 0..shape.4 {
                            for j in 0..shape.5 {
                                out[IxDyn(&[term, t, ix, iy, k, i, j])] =
                                    values[[t, ix, iy, k, i, j]];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn scalar_times_vector(
    scalar: ArrayView3<'_, f64>,
    vector: ndarray::ArrayView4<'_, f64>,
) -> Array4<f64> {
    let (frames, nx, ny) = scalar.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for component in 0..2 {
                    out[[t, ix, iy, component]] =
                        scalar[[t, ix, iy]] * vector[[t, ix, iy, component]];
                }
            }
        }
    }
    out
}

fn surface_rows_rank2(values: ndarray::ArrayView5<'_, f64>) -> Array5<f64> {
    let (frames, nx, ny, _, orientation_components) = values.dim();
    let mut out = Array5::<f64>::zeros((frames, nx, ny, 2, orientation_components));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for component in 0..orientation_components {
                        out[[t, ix, iy, k, component]] = values[[t, ix, iy, k, component]];
                    }
                }
            }
        }
    }
    out
}

fn rho_times_vector(
    rho: ArrayView3<'_, f64>,
    vector: ndarray::ArrayView4<'_, f64>,
    power: i32,
) -> Array4<f64> {
    let (frames, nx, ny) = rho.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let scale = rho[[t, ix, iy]].powi(power);
                for component in 0..2 {
                    out[[t, ix, iy, component]] = scale * vector[[t, ix, iy, component]];
                }
            }
        }
    }
    out
}

fn rho_times_rank2(
    rho: ArrayView3<'_, f64>,
    values: ndarray::ArrayView5<'_, f64>,
    power: i32,
) -> Array5<f64> {
    let (frames, nx, ny) = rho.dim();
    let shape = values.dim();
    let mut out = Array5::<f64>::zeros(shape);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let scale = rho[[t, ix, iy]].powi(power);
                for i in 0..shape.3 {
                    for j in 0..shape.4 {
                        out[[t, ix, iy, i, j]] = scale * values[[t, ix, iy, i, j]];
                    }
                }
            }
        }
    }
    out
}

fn rho_times_rank3(
    rho: ArrayView3<'_, f64>,
    values: ndarray::ArrayView6<'_, f64>,
    power: i32,
) -> Array6<f64> {
    let (frames, nx, ny) = rho.dim();
    let shape = values.dim();
    let mut out = Array6::<f64>::zeros(shape);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let scale = rho[[t, ix, iy]].powi(power);
                for k in 0..shape.3 {
                    for i in 0..shape.4 {
                        for j in 0..shape.5 {
                            out[[t, ix, iy, k, i, j]] = scale * values[[t, ix, iy, k, i, j]];
                        }
                    }
                }
            }
        }
    }
    out
}

fn cubic_gradient_flux(grad: ndarray::ArrayView4<'_, f64>) -> Array4<f64> {
    let (frames, nx, ny, _) = grad.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let norm2 = grad[[t, ix, iy, 0]].powi(2) + grad[[t, ix, iy, 1]].powi(2);
                out[[t, ix, iy, 0]] = norm2 * grad[[t, ix, iy, 0]];
                out[[t, ix, iy, 1]] = norm2 * grad[[t, ix, iy, 1]];
            }
        }
    }
    out
}

fn gradient_norm2(grad: ndarray::ArrayView4<'_, f64>) -> Array3<f64> {
    let (frames, nx, ny, _) = grad.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = grad[[t, ix, iy, 0]].powi(2) + grad[[t, ix, iy, 1]].powi(2);
            }
        }
    }
    out
}

fn p_dot_alpha(p: ndarray::ArrayView4<'_, f64>) -> Array6<f64> {
    let (frames, nx, ny, _) = p.dim();
    let mut out = Array6::<f64>::zeros((frames, nx, ny, 2, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for i in 0..3 {
                        for j in 0..3 {
                            out[[t, ix, iy, k, i, j]] = p[[t, ix, iy, k]] * delta(i, j)
                                + p[[t, ix, iy, i]] * delta(k, j)
                                + p[[t, ix, iy, j]] * delta(k, i);
                        }
                    }
                }
            }
        }
    }
    out
}

fn p_dot_ii(p: ndarray::ArrayView4<'_, f64>) -> Array6<f64> {
    let (frames, nx, ny, _) = p.dim();
    let mut out = Array6::<f64>::zeros((frames, nx, ny, 2, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for i in 0..3 {
                        out[[t, ix, iy, k, i, i]] = p[[t, ix, iy, k]];
                    }
                }
            }
        }
    }
    out
}

fn vector_identity(vector: ndarray::ArrayView4<'_, f64>) -> Array6<f64> {
    let (frames, nx, ny, _) = vector.dim();
    let mut out = Array6::<f64>::zeros((frames, nx, ny, 2, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for i in 0..3 {
                        out[[t, ix, iy, k, i, i]] = vector[[t, ix, iy, k]];
                    }
                }
            }
        }
    }
    out
}

fn delta(a: usize, b: usize) -> f64 {
    if a == b {
        1.0
    } else {
        0.0
    }
}

fn unravel_component(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        out[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    out
}

fn checked_index(value: i64, upper: usize, name: &str) -> CoreResult<usize> {
    if value < 0 || value as usize >= upper {
        return Err(CoreError::InvalidInput(format!(
            "{name} sample index out of bounds"
        )));
    }
    Ok(value as usize)
}

fn validate_grid(rho: ArrayView3<'_, f64>, lx: f64, ly: f64) -> CoreResult<()> {
    let (frames, nx, ny) = rho.dim();
    if frames == 0 || nx == 0 || ny == 0 {
        return Err(CoreError::InvalidInput(
            "rho grid axes must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite() && lx > 0.0 && ly.is_finite() && ly > 0.0) {
        return Err(CoreError::InvalidInput(
            "domain lengths must be positive".to_string(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_particle_fields(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
    gamma: f64,
    u0: f64,
) -> CoreResult<()> {
    let (frames, particles, coord_components) = coords.dim();
    if coord_components != 3 {
        return Err(CoreError::Shape(
            "coords must have shape (T,N,3)".to_string(),
        ));
    }
    if directions.dim() != (frames, particles, 3) || velocities.dim() != (frames, particles, 2) {
        return Err(CoreError::Shape(
            "directions must have shape (T,N,3) and velocities must have shape (T,N,2)"
                .to_string(),
        ));
    }
    if mask.dim() != (frames, particles) {
        return Err(CoreError::Shape("mask must have shape (T,N)".to_string()));
    }
    if x_centers.is_empty() || y_centers.is_empty() {
        return Err(CoreError::InvalidInput(
            "grid centers must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite()
        && lx > 0.0
        && ly.is_finite()
        && ly > 0.0
        && radius.is_finite()
        && radius > 0.0
        && sigma.is_finite()
        && sigma > 0.0)
    {
        return Err(CoreError::InvalidInput(
            "geometry values must be positive".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }
    Ok(())
}
