use ndarray::{Array4, Array5, Array6, ArrayD, ArrayView3, ArrayViewD, IxDyn};

use super::math::centered_scalar;
use super::operators::{
    estimate_ubar, q_dot_grad_rho, scalar_times_rank2, scalar_times_rank3, surface_rows_rank2,
    vector_dot_alpha_traceless,
};
use super::validation::validate_grid;
use super::MechanicalLibraries;
use crate::fft_ops;
use crate::{CoreError, CoreResult};

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
    ];
    let y_p_names = vec!["A".to_string(), "rho_delta_psi6sq_A".to_string()];
    let y_q_names = vec!["Ubar_P_dot_alpha_traceless".to_string()];

    let ubar = estimate_ubar(y_p5, a5);
    let p_alpha = vector_dot_alpha_traceless(p4);
    let y_rho = stack_vector_terms(&build_y_rho_terms(rho, q5, lx, ly)?);
    let y_p = stack_rank2_terms(&build_y_p_terms(rho, a5, psi6_sq));
    let y_q = stack_rank3_terms(&build_y_q_terms(ubar.view(), p_alpha.view()));
    Ok(MechanicalLibraries {
        y_rho_names,
        y_p_names,
        y_q_names,
        y_rho,
        y_p,
        y_q,
    })
}

fn build_y_rho_terms(
    rho: ArrayView3<'_, f64>,
    q: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<Array4<f64>>> {
    let lap_rho = fft_ops::laplacian_scalar(rho, lx, ly)?;
    let grad_rho = fft_ops::gradient_scalar(rho, lx, ly)?;
    let grad_lap_rho = fft_ops::gradient_scalar(lap_rho.view(), lx, ly)?;
    let q_grad_rho = q_dot_grad_rho(q, grad_rho.view());
    Ok(vec![grad_rho, grad_lap_rho, q_grad_rho])
}

fn build_y_p_terms(
    rho: ArrayView3<'_, f64>,
    a: ndarray::ArrayView5<'_, f64>,
    psi6_sq: ArrayView3<'_, f64>,
) -> Vec<Array5<f64>> {
    let a_surface = surface_rows_rank2(a);
    let delta_psi6_sq = centered_scalar(psi6_sq);
    vec![
        a_surface.clone(),
        scalar_times_rank2(
            rho,
            scalar_times_rank2(delta_psi6_sq.view(), a_surface.view()).view(),
        ),
    ]
}

fn build_y_q_terms(
    ubar: ArrayView3<'_, f64>,
    p_alpha: ndarray::ArrayView6<'_, f64>,
) -> Vec<Array6<f64>> {
    let ubar_alpha = scalar_times_rank3(ubar, p_alpha);
    vec![ubar_alpha]
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
