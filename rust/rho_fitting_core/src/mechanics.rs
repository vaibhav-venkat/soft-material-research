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

pub fn build_mechanical_libraries(
    rho: ArrayView3<'_, f64>,
    p: ArrayViewD<'_, f64>,
    q: ArrayViewD<'_, f64>,
    a: ArrayViewD<'_, f64>,
    psi6_sq: ArrayView3<'_, f64>,
    y_p: ArrayViewD<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<MechanicalLibraries> {
    validate_grid(rho, lx, ly)?;
    let p4 = p
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| CoreError::Shape("P must be (T,Nx,Ny,3)".to_string()))?;
    let q5 = q
        .into_dimensionality::<ndarray::Ix5>()
        .map_err(|_| CoreError::Shape("Q must be (T,Nx,Ny,3,3)".to_string()))?;
    let a5 = a
        .into_dimensionality::<ndarray::Ix5>()
        .map_err(|_| CoreError::Shape("A must be (T,Nx,Ny,3,3)".to_string()))?;
    let y_p5 = y_p
        .into_dimensionality::<ndarray::Ix5>()
        .map_err(|_| CoreError::Shape("Y_P must be (T,Nx,Ny,2,3)".to_string()))?;
    if p4.dim() != (rho.dim().0, rho.dim().1, rho.dim().2, 3)
        || q5.dim() != (rho.dim().0, rho.dim().1, rho.dim().2, 3, 3)
        || a5.dim() != (rho.dim().0, rho.dim().1, rho.dim().2, 3, 3)
        || y_p5.dim() != (rho.dim().0, rho.dim().1, rho.dim().2, 2, 3)
        || psi6_sq.dim() != rho.dim()
    {
        return Err(CoreError::Shape(
            "mechanical library field shapes do not align".to_string(),
        ));
    }

    let y_rho_names = vec![
        "grad_rho".to_string(),
        "grad_lap_rho".to_string(),
        "Q_dot_grad_rho".to_string(),
        "Q2_dot_grad_rho".to_string(),
        "trQ2_grad_rho".to_string(),
        "rho_Q_dot_grad_rho".to_string(),
        "div_Q".to_string(),
        "rho_div_Q".to_string(),
        "div_rho_Q".to_string(),
        "Q_dot_grad_lap_rho".to_string(),
        "P".to_string(),
        "rho_P".to_string(),
        "P_dot_grad_rho_P".to_string(),
        "P2_grad_rho".to_string(),
        "div_P_P".to_string(),
        "P_dot_grad_P".to_string(),
        "grad_P2".to_string(),
        "div_A".to_string(),
        "grad_trA".to_string(),
        "A_dot_grad_rho".to_string(),
        "trA_grad_rho".to_string(),
        "Q_dot_div_A".to_string(),
    ];
    let y_p_names = vec![
        "A".to_string(),
        "rho_A".to_string(),
        "delta_A".to_string(),
        "psi6_A".to_string(),
        "psi6sq_A".to_string(),
        "delta_psi6_A".to_string(),
        "delta_psi6sq_A".to_string(),
        "rho_psi6sq_A".to_string(),
        "trQ2_A".to_string(),
        "P2_A".to_string(),
        "P".to_string(),
        "rho_P".to_string(),
        "P2_P".to_string(),
        "trQ2_P".to_string(),
        "psi6sq_P".to_string(),
        "lap_P".to_string(),
        "grad_div_P".to_string(),
        "P_dot_grad_P".to_string(),
        "Q_dot_P".to_string(),
        "Q2_dot_P".to_string(),
        "A_dot_Q".to_string(),
        "Q_dot_A_dot_Q".to_string(),
        "Q_colon_A_P".to_string(),
    ];
    let y_q_names = vec![
        "P_dot_alpha_traceless".to_string(),
        "Ubar_P_dot_alpha_traceless".to_string(),
        "rho_Ubar_P_dot_alpha_traceless".to_string(),
        "delta_Ubar_P_dot_alpha_traceless".to_string(),
        "psi6sq_Ubar_P_dot_alpha_traceless".to_string(),
        "trQ2_Ubar_P_dot_alpha_traceless".to_string(),
        "P2_Ubar_P_dot_alpha_traceless".to_string(),
        "Q".to_string(),
        "rho_Q".to_string(),
        "delta_Q".to_string(),
        "trQ2_Q".to_string(),
        "psi6sq_Q".to_string(),
        "P2_Q".to_string(),
        "lap_Q".to_string(),
        "bilap_Q".to_string(),
        "PP_traceless_2d".to_string(),
        "rho_PP_traceless_2d".to_string(),
        "AA_traceless_2d".to_string(),
        "QA_plus_AQ".to_string(),
        "Q2_traceless_2d".to_string(),
        "sym_grad_P_traceless_2d".to_string(),
        "hess_rho_traceless_2d".to_string(),
    ];

    let ubar = estimate_ubar(y_p5, a5);
    let p_alpha = vector_dot_alpha_traceless(p4);
    let y_rho = stack_vector_terms(&build_y_rho_terms(rho, p4, q5, a5, lx, ly)?);
    let y_p = stack_rank2_terms(&build_y_p_terms(rho, p4, q5, a5, psi6_sq, lx, ly)?);
    let y_q = stack_rank3_terms(&build_y_q_terms(
        rho,
        p4,
        q5,
        a5,
        psi6_sq,
        ubar.view(),
        p_alpha.view(),
        lx,
        ly,
    )?);
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

fn build_y_rho_terms(
    rho: ArrayView3<'_, f64>,
    p: ndarray::ArrayView4<'_, f64>,
    q: ndarray::ArrayView5<'_, f64>,
    a: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<Array4<f64>>> {
    let lap_rho = fft_ops::laplacian_scalar(rho, lx, ly)?;
    let grad_rho = fft_ops::gradient_scalar(rho, lx, ly)?;
    let grad_lap_rho = fft_ops::gradient_scalar(lap_rho.view(), lx, ly)?;
    let p_surface = surface_vector(p);
    let p2 = vector_norm2_3(p);
    let tr_q2 = trace_square(q);
    let tr_a = trace_rank2(a);
    let div_p = fft_ops::divergence_vector(p_surface.view(), lx, ly)?;
    let div_q = divergence_rank2_surface(q, lx, ly)?;
    let div_a = divergence_rank2_surface(a, lx, ly)?;
    let rho_q = scalar_times_full_rank2(rho, q);
    let q_grad_rho = q_dot_grad_rho(q, grad_rho.view());
    Ok(vec![
        grad_rho.clone(),
        grad_lap_rho.clone(),
        q_grad_rho.clone(),
        q2_dot_vector(q, grad_rho.view()),
        scalar_times_vector(tr_q2.view(), grad_rho.view()),
        scalar_times_vector(rho, q_grad_rho.view()),
        div_q.clone(),
        scalar_times_vector(rho, div_q.view()),
        divergence_rank2_surface(rho_q.view(), lx, ly)?,
        q_dot_grad_rho(q, grad_lap_rho.view()),
        p_surface.clone(),
        scalar_times_vector(rho, p_surface.view()),
        scalar_times_vector(
            dot_vectors(p_surface.view(), grad_rho.view()).view(),
            p_surface.view(),
        ),
        scalar_times_vector(p2.view(), grad_rho.view()),
        scalar_times_vector(div_p.view(), p_surface.view()),
        advective_derivative_vector(p_surface.view(), p_surface.view(), lx, ly)?,
        fft_ops::gradient_scalar(p2.view(), lx, ly)?,
        div_a.clone(),
        fft_ops::gradient_scalar(tr_a.view(), lx, ly)?,
        a_dot_vector(a, grad_rho.view()),
        scalar_times_vector(tr_a.view(), grad_rho.view()),
        q_dot_vector(q, div_a.view()),
    ])
}

fn build_y_p_terms(
    rho: ArrayView3<'_, f64>,
    p: ndarray::ArrayView4<'_, f64>,
    q: ndarray::ArrayView5<'_, f64>,
    a: ndarray::ArrayView5<'_, f64>,
    psi6_sq: ArrayView3<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<Array5<f64>>> {
    let a_surface = surface_rows_rank2(a);
    let delta_rho = centered_scalar(rho);
    let psi6 = sqrt_nonnegative_scalar(psi6_sq);
    let delta_psi6 = centered_scalar(psi6.view());
    let delta_psi6_sq = centered_scalar(psi6_sq);
    let tr_q2 = trace_square(q);
    let p2 = vector_norm2_3(p);
    let p_rank2 = vector_as_rank2(p);
    let lap_p = laplacian_rank2(p_rank2.view(), lx, ly)?;
    let grad_div_p = grad_div_rank2(p, lx, ly)?;
    let p_surface = surface_vector(p);
    let p_dot_grad_p = advective_derivative_vector(p_surface.view(), p_surface.view(), lx, ly)?;
    Ok(vec![
        a_surface.clone(),
        scalar_times_rank2(rho, a_surface.view()),
        scalar_times_rank2(delta_rho.view(), a_surface.view()),
        scalar_times_rank2(psi6.view(), a_surface.view()),
        scalar_times_rank2(psi6_sq, a_surface.view()),
        scalar_times_rank2(delta_psi6.view(), a_surface.view()),
        scalar_times_rank2(delta_psi6_sq.view(), a_surface.view()),
        scalar_times_rank2(rho, scalar_times_rank2(psi6_sq, a_surface.view()).view()),
        scalar_times_rank2(tr_q2.view(), a_surface.view()),
        scalar_times_rank2(p2.view(), a_surface.view()),
        p_rank2.clone(),
        scalar_times_rank2(rho, p_rank2.view()),
        scalar_times_rank2(p2.view(), p_rank2.view()),
        scalar_times_rank2(tr_q2.view(), p_rank2.view()),
        scalar_times_rank2(psi6_sq, p_rank2.view()),
        lap_p,
        grad_div_p,
        vector_as_rank2_from_surface(p_dot_grad_p.view()),
        vector_as_rank2(q_dot_p(q, p).view()),
        vector_as_rank2(q2_dot_p(q, p).view()),
        surface_rows_rank2(matrix_product(a, q).view()),
        surface_rows_rank2(q_a_q(q, a).view()),
        scalar_times_rank2(q_colon_a(q, a).view(), p_rank2.view()),
    ])
}

#[allow(clippy::too_many_arguments)]
fn build_y_q_terms(
    rho: ArrayView3<'_, f64>,
    p: ndarray::ArrayView4<'_, f64>,
    q: ndarray::ArrayView5<'_, f64>,
    a: ndarray::ArrayView5<'_, f64>,
    psi6_sq: ArrayView3<'_, f64>,
    ubar: ArrayView3<'_, f64>,
    p_alpha: ndarray::ArrayView6<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<Array6<f64>>> {
    let ubar_alpha = scalar_times_rank3(ubar, p_alpha);
    let delta_rho = centered_scalar(rho);
    let delta_ubar = centered_scalar(ubar);
    let tr_q2 = trace_square(q);
    let p2 = vector_norm2_3(p);
    let q_rank3 = broadcast_rank2_to_rank3(q);
    let pp = pp_traceless_2d(p);
    let pp_rank3 = broadcast_rank2_to_rank3(pp.view());
    Ok(vec![
        p_alpha.to_owned(),
        ubar_alpha.clone(),
        scalar_times_rank3(rho, ubar_alpha.view()),
        scalar_times_rank3(delta_ubar.view(), p_alpha),
        scalar_times_rank3(psi6_sq, ubar_alpha.view()),
        scalar_times_rank3(tr_q2.view(), ubar_alpha.view()),
        scalar_times_rank3(p2.view(), ubar_alpha.view()),
        q_rank3.clone(),
        scalar_times_rank3(rho, q_rank3.view()),
        scalar_times_rank3(delta_rho.view(), q_rank3.view()),
        scalar_times_rank3(tr_q2.view(), q_rank3.view()),
        scalar_times_rank3(psi6_sq, q_rank3.view()),
        scalar_times_rank3(p2.view(), q_rank3.view()),
        laplacian_rank3(q, lx, ly)?,
        bilaplacian_rank3(q, lx, ly)?,
        pp_rank3.clone(),
        scalar_times_rank3(rho, pp_rank3.view()),
        broadcast_rank2_to_rank3(aa_traceless_2d(a).view()),
        broadcast_rank2_to_rank3(qa_plus_aq(q, a).view()),
        broadcast_rank2_to_rank3(q2_traceless_2d(q).view()),
        broadcast_rank2_to_rank3(sym_grad_p_traceless_2d(p, lx, ly)?.view()),
        broadcast_rank2_to_rank3(hess_rho_traceless_2d(rho, lx, ly)?.view()),
    ])
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

fn surface_vector(values: ndarray::ArrayView4<'_, f64>) -> Array4<f64> {
    let (frames, nx, ny, _) = values.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    out[[t, ix, iy, k]] = values[[t, ix, iy, k]];
                }
            }
        }
    }
    out
}

fn vector_as_rank2(values: ndarray::ArrayView4<'_, f64>) -> Array5<f64> {
    let (frames, nx, ny, components) = values.dim();
    let mut out = Array5::<f64>::zeros((frames, nx, ny, 2, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for component in 0..3 {
                        out[[t, ix, iy, k, component]] =
                            vector_component(values, t, ix, iy, component, components);
                    }
                }
            }
        }
    }
    out
}

fn vector_as_rank2_from_surface(values: ndarray::ArrayView4<'_, f64>) -> Array5<f64> {
    let (frames, nx, ny, _) = values.dim();
    let mut out = Array5::<f64>::zeros((frames, nx, ny, 2, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for component in 0..2 {
                        out[[t, ix, iy, k, component]] = values[[t, ix, iy, component]];
                    }
                }
            }
        }
    }
    out
}

fn dot_vectors(a: ndarray::ArrayView4<'_, f64>, b: ndarray::ArrayView4<'_, f64>) -> Array3<f64> {
    let (frames, nx, ny, _) = a.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] =
                    a[[t, ix, iy, 0]] * b[[t, ix, iy, 0]] + a[[t, ix, iy, 1]] * b[[t, ix, iy, 1]];
            }
        }
    }
    out
}

fn q_dot_grad_rho(
    q: ndarray::ArrayView5<'_, f64>,
    grad: ndarray::ArrayView4<'_, f64>,
) -> Array4<f64> {
    q_dot_vector(q, grad)
}

fn q_dot_vector(
    q: ndarray::ArrayView5<'_, f64>,
    vector: ndarray::ArrayView4<'_, f64>,
) -> Array4<f64> {
    let (frames, nx, ny, _, _) = q.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    out[[t, ix, iy, k]] = q[[t, ix, iy, k, 0]] * vector[[t, ix, iy, 0]]
                        + q[[t, ix, iy, k, 1]] * vector[[t, ix, iy, 1]];
                }
            }
        }
    }
    out
}

fn a_dot_vector(
    a: ndarray::ArrayView5<'_, f64>,
    vector: ndarray::ArrayView4<'_, f64>,
) -> Array4<f64> {
    let (frames, nx, ny, _, _) = a.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    out[[t, ix, iy, k]] = a[[t, ix, iy, k, 0]] * vector[[t, ix, iy, 0]]
                        + a[[t, ix, iy, k, 1]] * vector[[t, ix, iy, 1]];
                }
            }
        }
    }
    out
}

fn q2_dot_vector(
    q: ndarray::ArrayView5<'_, f64>,
    vector: ndarray::ArrayView4<'_, f64>,
) -> Array4<f64> {
    q_dot_vector(matrix_product(q, q).view(), vector)
}

fn q_dot_p(q: ndarray::ArrayView5<'_, f64>, p: ndarray::ArrayView4<'_, f64>) -> Array4<f64> {
    let (frames, nx, ny, _, _) = q.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..3 {
                    for j in 0..3 {
                        out[[t, ix, iy, i]] += q[[t, ix, iy, i, j]] * p[[t, ix, iy, j]];
                    }
                }
            }
        }
    }
    out
}

fn q2_dot_p(q: ndarray::ArrayView5<'_, f64>, p: ndarray::ArrayView4<'_, f64>) -> Array4<f64> {
    q_dot_p(matrix_product(q, q).view(), p)
}

fn scalar_times_vector(
    scalar: ArrayView3<'_, f64>,
    vector: ndarray::ArrayView4<'_, f64>,
) -> Array4<f64> {
    let (frames, nx, ny) = scalar.dim();
    let shape = vector.dim();
    let mut out = Array4::<f64>::zeros(shape);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for component in 0..shape.3 {
                    out[[t, ix, iy, component]] =
                        scalar[[t, ix, iy]] * vector[[t, ix, iy, component]];
                }
            }
        }
    }
    out
}

fn centered_scalar(values: ArrayView3<'_, f64>) -> Array3<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().copied() {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    values.mapv(|value| value - mean)
}

fn estimate_ubar(
    y_p: ndarray::ArrayView5<'_, f64>,
    a: ndarray::ArrayView5<'_, f64>,
) -> Array3<f64> {
    let (frames, nx, ny, _, _) = y_p.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for k in 0..2 {
                    for component in 0..3 {
                        let a_value = a[[t, ix, iy, k, component]];
                        numerator += y_p[[t, ix, iy, k, component]] * a_value;
                        denominator += a_value * a_value;
                    }
                }
                if denominator > 0.0 {
                    out[[t, ix, iy]] = numerator / denominator;
                }
            }
        }
    }
    out
}

fn scalar_times_rank2(
    scalar: ArrayView3<'_, f64>,
    values: ndarray::ArrayView5<'_, f64>,
) -> Array5<f64> {
    let (frames, nx, ny) = scalar.dim();
    let shape = values.dim();
    let mut out = Array5::<f64>::zeros(shape);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..shape.3 {
                    for j in 0..shape.4 {
                        out[[t, ix, iy, i, j]] = scalar[[t, ix, iy]] * values[[t, ix, iy, i, j]];
                    }
                }
            }
        }
    }
    out
}

fn scalar_times_rank3(
    scalar: ArrayView3<'_, f64>,
    values: ndarray::ArrayView6<'_, f64>,
) -> Array6<f64> {
    let (frames, nx, ny) = scalar.dim();
    let shape = values.dim();
    let mut out = Array6::<f64>::zeros(shape);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..shape.3 {
                    for i in 0..shape.4 {
                        for j in 0..shape.5 {
                            out[[t, ix, iy, k, i, j]] =
                                scalar[[t, ix, iy]] * values[[t, ix, iy, k, i, j]];
                        }
                    }
                }
            }
        }
    }
    out
}

fn scalar_times_full_rank2(
    scalar: ArrayView3<'_, f64>,
    values: ndarray::ArrayView5<'_, f64>,
) -> Array5<f64> {
    let (frames, nx, ny) = scalar.dim();
    let shape = values.dim();
    let mut out = Array5::<f64>::zeros(shape);
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..shape.3 {
                    for j in 0..shape.4 {
                        out[[t, ix, iy, i, j]] = scalar[[t, ix, iy]] * values[[t, ix, iy, i, j]];
                    }
                }
            }
        }
    }
    out
}

fn sqrt_nonnegative_scalar(values: ArrayView3<'_, f64>) -> Array3<f64> {
    values.mapv(|value| {
        if value.is_finite() && value > 0.0 {
            value.sqrt()
        } else {
            0.0
        }
    })
}

fn vector_norm2_3(values: ndarray::ArrayView4<'_, f64>) -> Array3<f64> {
    let (frames, nx, ny, _) = values.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = values[[t, ix, iy, 0]].powi(2)
                    + values[[t, ix, iy, 1]].powi(2)
                    + values[[t, ix, iy, 2]].powi(2);
            }
        }
    }
    out
}

fn trace_square(values: ndarray::ArrayView5<'_, f64>) -> Array3<f64> {
    let (frames, nx, ny, _, _) = values.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..3 {
                    for j in 0..3 {
                        out[[t, ix, iy]] += values[[t, ix, iy, i, j]] * values[[t, ix, iy, i, j]];
                    }
                }
            }
        }
    }
    out
}

fn trace_rank2(values: ndarray::ArrayView5<'_, f64>) -> Array3<f64> {
    let (frames, nx, ny, _, _) = values.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = values[[t, ix, iy, 0, 0]]
                    + values[[t, ix, iy, 1, 1]]
                    + values[[t, ix, iy, 2, 2]];
            }
        }
    }
    out
}

fn divergence_rank2_surface(
    values: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array4<f64>> {
    let (frames, nx, ny, _, _) = values.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for component in 0..2 {
        let mut vector = Array4::<f64>::zeros((frames, nx, ny, 2));
        for t in 0..frames {
            for ix in 0..nx {
                for iy in 0..ny {
                    vector[[t, ix, iy, 0]] = values[[t, ix, iy, component, 0]];
                    vector[[t, ix, iy, 1]] = values[[t, ix, iy, component, 1]];
                }
            }
        }
        let div = fft_ops::divergence_vector(vector.view(), lx, ly)?;
        for t in 0..frames {
            for ix in 0..nx {
                for iy in 0..ny {
                    out[[t, ix, iy, component]] = div[[t, ix, iy]];
                }
            }
        }
    }
    Ok(out)
}

fn advective_derivative_vector(
    advecting: ndarray::ArrayView4<'_, f64>,
    vector: ndarray::ArrayView4<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array4<f64>> {
    let (frames, nx, ny, components) = vector.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, components));
    for component in 0..components {
        let scalar = vector.index_axis(ndarray::Axis(3), component).to_owned();
        let grad = fft_ops::gradient_scalar(scalar.view(), lx, ly)?;
        for t in 0..frames {
            for ix in 0..nx {
                for iy in 0..ny {
                    out[[t, ix, iy, component]] = advecting[[t, ix, iy, 0]] * grad[[t, ix, iy, 0]]
                        + advecting[[t, ix, iy, 1]] * grad[[t, ix, iy, 1]];
                }
            }
        }
    }
    Ok(out)
}

fn grad_div_rank2(p: ndarray::ArrayView4<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array5<f64>> {
    let p_surface = surface_vector(p);
    let grad_div = fft_ops::grad_div_vector(p_surface.view(), lx, ly)?;
    Ok(vector_as_rank2_from_surface(grad_div.view()))
}

fn laplacian_rank2(
    values: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array5<f64>> {
    let shape = values.dim();
    let mut out = Array5::<f64>::zeros(shape);
    for i in 0..shape.3 {
        for j in 0..shape.4 {
            let scalar = values
                .index_axis(ndarray::Axis(3), i)
                .index_axis(ndarray::Axis(3), j)
                .to_owned();
            let lap = fft_ops::laplacian_scalar(scalar.view(), lx, ly)?;
            for t in 0..shape.0 {
                for ix in 0..shape.1 {
                    for iy in 0..shape.2 {
                        out[[t, ix, iy, i, j]] = lap[[t, ix, iy]];
                    }
                }
            }
        }
    }
    Ok(out)
}

fn laplacian_rank3(
    values: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array6<f64>> {
    let rank3 = broadcast_rank2_to_rank3(values);
    laplacian_rank3_full(rank3.view(), lx, ly)
}

fn bilaplacian_rank3(
    values: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array6<f64>> {
    let lap = laplacian_rank3(values, lx, ly)?;
    laplacian_rank3_full(lap.view(), lx, ly)
}

fn laplacian_rank3_full(
    values: ndarray::ArrayView6<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array6<f64>> {
    let shape = values.dim();
    let mut out = Array6::<f64>::zeros(shape);
    for k in 0..shape.3 {
        for i in 0..shape.4 {
            for j in 0..shape.5 {
                let scalar = values
                    .index_axis(ndarray::Axis(3), k)
                    .index_axis(ndarray::Axis(3), i)
                    .index_axis(ndarray::Axis(3), j)
                    .to_owned();
                let lap = fft_ops::laplacian_scalar(scalar.view(), lx, ly)?;
                for t in 0..shape.0 {
                    for ix in 0..shape.1 {
                        for iy in 0..shape.2 {
                            out[[t, ix, iy, k, i, j]] = lap[[t, ix, iy]];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn broadcast_rank2_to_rank3(values: ndarray::ArrayView5<'_, f64>) -> Array6<f64> {
    let (frames, nx, ny, _, _) = values.dim();
    let mut out = Array6::<f64>::zeros((frames, nx, ny, 2, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for i in 0..3 {
                        for j in 0..3 {
                            out[[t, ix, iy, k, i, j]] = values[[t, ix, iy, i, j]];
                        }
                    }
                }
            }
        }
    }
    out
}

fn matrix_product(
    left: ndarray::ArrayView5<'_, f64>,
    right: ndarray::ArrayView5<'_, f64>,
) -> Array5<f64> {
    let (frames, nx, ny, _, _) = left.dim();
    let mut out = Array5::<f64>::zeros((frames, nx, ny, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..3 {
                    for j in 0..3 {
                        for k in 0..3 {
                            out[[t, ix, iy, i, j]] +=
                                left[[t, ix, iy, i, k]] * right[[t, ix, iy, k, j]];
                        }
                    }
                }
            }
        }
    }
    out
}

fn q_a_q(q: ndarray::ArrayView5<'_, f64>, a: ndarray::ArrayView5<'_, f64>) -> Array5<f64> {
    let qa = matrix_product(q, a);
    matrix_product(qa.view(), q)
}

fn q_colon_a(q: ndarray::ArrayView5<'_, f64>, a: ndarray::ArrayView5<'_, f64>) -> Array3<f64> {
    let (frames, nx, ny, _, _) = q.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..3 {
                    for j in 0..3 {
                        out[[t, ix, iy]] += q[[t, ix, iy, i, j]] * a[[t, ix, iy, i, j]];
                    }
                }
            }
        }
    }
    out
}

fn pp_traceless_2d(p: ndarray::ArrayView4<'_, f64>) -> Array5<f64> {
    let (frames, nx, ny, _) = p.dim();
    let p2 = vector_norm2_3(p);
    let mut out = Array5::<f64>::zeros((frames, nx, ny, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..3 {
                    for j in 0..3 {
                        out[[t, ix, iy, i, j]] = p[[t, ix, iy, i]] * p[[t, ix, iy, j]]
                            - 0.5 * p2[[t, ix, iy]] * delta(i, j);
                    }
                }
            }
        }
    }
    out
}

fn aa_traceless_2d(a: ndarray::ArrayView5<'_, f64>) -> Array5<f64> {
    let aa = matrix_product(a, a);
    let tr = trace_rank2(aa.view());
    subtract_half_trace_identity(aa.view(), tr.view())
}

fn qa_plus_aq(q: ndarray::ArrayView5<'_, f64>, a: ndarray::ArrayView5<'_, f64>) -> Array5<f64> {
    let qa = matrix_product(q, a);
    let aq = matrix_product(a, q);
    let shape = qa.dim();
    let mut out = Array5::<f64>::zeros(shape);
    for t in 0..shape.0 {
        for ix in 0..shape.1 {
            for iy in 0..shape.2 {
                for i in 0..3 {
                    for j in 0..3 {
                        out[[t, ix, iy, i, j]] = qa[[t, ix, iy, i, j]] + aq[[t, ix, iy, i, j]];
                    }
                }
            }
        }
    }
    out
}

fn q2_traceless_2d(q: ndarray::ArrayView5<'_, f64>) -> Array5<f64> {
    let q2 = matrix_product(q, q);
    let tr = trace_rank2(q2.view());
    subtract_half_trace_identity(q2.view(), tr.view())
}

fn subtract_half_trace_identity(
    values: ndarray::ArrayView5<'_, f64>,
    trace: ArrayView3<'_, f64>,
) -> Array5<f64> {
    let shape = values.dim();
    let mut out = values.to_owned();
    for t in 0..shape.0 {
        for ix in 0..shape.1 {
            for iy in 0..shape.2 {
                for i in 0..3 {
                    out[[t, ix, iy, i, i]] -= 0.5 * trace[[t, ix, iy]];
                }
            }
        }
    }
    out
}

fn sym_grad_p_traceless_2d(
    p: ndarray::ArrayView4<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array5<f64>> {
    let (frames, nx, ny, _) = p.dim();
    let mut grad = Array5::<f64>::zeros((frames, nx, ny, 3, 3));
    for component in 0..3 {
        let scalar = p.index_axis(ndarray::Axis(3), component).to_owned();
        let derivative = fft_ops::gradient_scalar(scalar.view(), lx, ly)?;
        for t in 0..frames {
            for ix in 0..nx {
                for iy in 0..ny {
                    for k in 0..2 {
                        grad[[t, ix, iy, k, component]] = derivative[[t, ix, iy, k]];
                    }
                }
            }
        }
    }
    let p_surface = surface_vector(p);
    let div = fft_ops::divergence_vector(p_surface.view(), lx, ly)?;
    let mut out = Array5::<f64>::zeros((frames, nx, ny, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..3 {
                    for j in 0..3 {
                        out[[t, ix, iy, i, j]] = grad[[t, ix, iy, i, j]] + grad[[t, ix, iy, j, i]]
                            - div[[t, ix, iy]] * delta(i, j);
                    }
                }
            }
        }
    }
    Ok(out)
}

fn hess_rho_traceless_2d(rho: ArrayView3<'_, f64>, lx: f64, ly: f64) -> CoreResult<Array5<f64>> {
    let (frames, nx, ny) = rho.dim();
    let grad = fft_ops::gradient_scalar(rho, lx, ly)?;
    let mut hess = Array5::<f64>::zeros((frames, nx, ny, 3, 3));
    for component in 0..2 {
        let scalar = grad.index_axis(ndarray::Axis(3), component).to_owned();
        let derivative = fft_ops::gradient_scalar(scalar.view(), lx, ly)?;
        for t in 0..frames {
            for ix in 0..nx {
                for iy in 0..ny {
                    for k in 0..2 {
                        hess[[t, ix, iy, k, component]] = derivative[[t, ix, iy, k]];
                    }
                }
            }
        }
    }
    let lap = fft_ops::laplacian_scalar(rho, lx, ly)?;
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for i in 0..3 {
                    hess[[t, ix, iy, i, i]] -= 0.5 * lap[[t, ix, iy]];
                }
            }
        }
    }
    Ok(hess)
}

fn vector_dot_alpha_traceless(vector: ndarray::ArrayView4<'_, f64>) -> Array6<f64> {
    let (frames, nx, ny, components) = vector.dim();
    let mut out = Array6::<f64>::zeros((frames, nx, ny, 2, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    for i in 0..3 {
                        for j in 0..3 {
                            out[[t, ix, iy, k, i, j]] =
                                vector_component(vector, t, ix, iy, i, components) * delta(k, j)
                                    + vector_component(vector, t, ix, iy, j, components)
                                        * delta(k, i)
                                    - (2.0 / 3.0)
                                        * vector_component(vector, t, ix, iy, k, components)
                                        * delta(i, j);
                        }
                    }
                }
            }
        }
    }
    out
}

fn vector_component(
    vector: ndarray::ArrayView4<'_, f64>,
    t: usize,
    ix: usize,
    iy: usize,
    component: usize,
    components: usize,
) -> f64 {
    if component < components {
        vector[[t, ix, iy, component]]
    } else {
        0.0
    }
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
) -> CoreResult<()> {
    let (frames, particles, coord_components) = coords.dim();
    if coord_components != 3 {
        return Err(CoreError::Shape(
            "coords must have shape (T,N,3)".to_string(),
        ));
    }
    if directions.dim() != (frames, particles, 3) || velocities.dim() != (frames, particles, 2) {
        return Err(CoreError::Shape(
            "directions must have shape (T,N,3) and velocities must have shape (T,N,2)".to_string(),
        ));
    }
    if psi6_abs.dim() != (frames, particles) {
        return Err(CoreError::Shape(
            "psi6_abs must have shape (T,N)".to_string(),
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
