use ndarray::{Array3, Array4, Array5, Array6, ArrayD, ArrayView3, ArrayViewD, IxDyn};

use super::operators::{
    estimate_ubar, q_dot_grad_rho, scalar_times_rank2, scalar_times_rank3, surface_rows_rank2,
    vector_dot_alpha_traceless,
};
use super::validation::validate_grid;
use super::MechanicalLibraries;
use crate::fft_ops;
use crate::{CoreError, CoreResult};

/// Build candidate flux libraries for the `Y_rho`, `Y_P`, and `Y_Q` targets.
///
/// Inputs must share `(T, Nx, Ny)` leading axes; `P` has 3 components, `Q`
/// and `A` are `3x3`, and `Y_P` is the measured `(2,3)` flux target used
/// to estimate `Ubar`.
///
/// Example: `build_mechanical_libraries(rho, p.into_dyn(), q.into_dyn(), a.into_dyn(), psi6_sq, y_p.into_dyn(), lx, ly)`.
///
/// Edge cases: coefficient order is defined by the returned name vectors and
/// is expected to stay aligned with Python PDE validation.
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
    let y_p_names = vec![
        "A".to_string(),
        "rho_A".to_string(),
        "psi6sq_A".to_string(),
        "grad_P".to_string(),
        "rho_grad_P".to_string(),
        "grad_lap_P".to_string(),
    ];
    let y_q_names = vec![
        "Ubar_P_dot_alpha_traceless".to_string(),
        "grad_P_symmetric_traceless".to_string(),
        "grad_Q".to_string(),
        "rho_grad_Q".to_string(),
        "grad_lap_Q".to_string(),
    ];

    let ubar = estimate_ubar(y_p5, a5);
    let p_alpha = vector_dot_alpha_traceless(p4);
    let y_rho = stack_vector_terms(&build_y_rho_terms(rho, q5, lx, ly)?);
    let y_p = stack_rank2_terms(&build_y_p_terms(rho, p4, a5, psi6_sq, lx, ly)?);
    let y_q = stack_rank3_terms(&build_y_q_terms(
        rho,
        p4,
        q5,
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
    Ok(vec![
        embed_surface_vector3(grad_rho.view()),
        embed_surface_vector3(grad_lap_rho.view()),
        q_grad_rho,
    ])
}

fn embed_surface_vector3(values: ndarray::ArrayView4<'_, f64>) -> Array4<f64> {
    let (frames, nx, ny, surface_components) = values.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..surface_components.min(2) {
                    out[[t, ix, iy, k]] = values[[t, ix, iy, k]];
                }
            }
        }
    }
    out
}

fn build_y_p_terms(
    rho: ArrayView3<'_, f64>,
    p: ndarray::ArrayView4<'_, f64>,
    a: ndarray::ArrayView5<'_, f64>,
    psi6_sq: ArrayView3<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<Array5<f64>>> {
    let a_surface = surface_rows_rank2(a);
    let grad_p = gradient_vector3(p, lx, ly)?;
    let lap_p = laplacian_vector3(p, lx, ly)?;
    let grad_lap_p = gradient_vector3(lap_p.view(), lx, ly)?;
    Ok(vec![
        a_surface.clone(),
        scalar_times_rank2(rho, a_surface.view()),
        scalar_times_rank2(psi6_sq, a_surface.view()),
        grad_p.clone(),
        scalar_times_rank2(rho, grad_p.view()),
        grad_lap_p,
    ])
}

fn build_y_q_terms(
    rho: ArrayView3<'_, f64>,
    p: ndarray::ArrayView4<'_, f64>,
    q: ndarray::ArrayView5<'_, f64>,
    ubar: ArrayView3<'_, f64>,
    p_alpha: ndarray::ArrayView6<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Vec<Array6<f64>>> {
    let ubar_alpha = scalar_times_rank3(ubar, p_alpha);
    let grad_q = gradient_rank2(q, lx, ly)?;
    let lap_q = laplacian_rank2(q, lx, ly)?;
    let grad_lap_q = gradient_rank2(lap_q.view(), lx, ly)?;
    Ok(vec![
        ubar_alpha,
        grad_p_symmetric_traceless(p, lx, ly)?,
        grad_q.clone(),
        scalar_times_rank3(rho, grad_q.view()),
        grad_lap_q,
    ])
}

fn gradient_vector3(
    values: ndarray::ArrayView4<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array5<f64>> {
    let (frames, nx, ny, components) = values.dim();
    let mut out = Array5::<f64>::zeros((frames, nx, ny, 2, components));
    for component in 0..components {
        let scalar = scalar_component(values, component);
        let grad = fft_ops::gradient_scalar(scalar.view(), lx, ly)?;
        for t in 0..frames {
            for ix in 0..nx {
                for iy in 0..ny {
                    for k in 0..2 {
                        out[[t, ix, iy, k, component]] = grad[[t, ix, iy, k]];
                    }
                }
            }
        }
    }
    Ok(out)
}

fn laplacian_vector3(
    values: ndarray::ArrayView4<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array4<f64>> {
    let (frames, nx, ny, components) = values.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, components));
    for component in 0..components {
        let scalar = scalar_component(values, component);
        let lap = fft_ops::laplacian_scalar(scalar.view(), lx, ly)?;
        for t in 0..frames {
            for ix in 0..nx {
                for iy in 0..ny {
                    out[[t, ix, iy, component]] = lap[[t, ix, iy]];
                }
            }
        }
    }
    Ok(out)
}

fn gradient_rank2(
    values: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array6<f64>> {
    let (frames, nx, ny, rows, cols) = values.dim();
    let mut out = Array6::<f64>::zeros((frames, nx, ny, 2, rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            let scalar = rank2_component(values, row, col);
            let grad = fft_ops::gradient_scalar(scalar.view(), lx, ly)?;
            for t in 0..frames {
                for ix in 0..nx {
                    for iy in 0..ny {
                        for k in 0..2 {
                            out[[t, ix, iy, k, row, col]] = grad[[t, ix, iy, k]];
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn laplacian_rank2(
    values: ndarray::ArrayView5<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array5<f64>> {
    let (frames, nx, ny, rows, cols) = values.dim();
    let mut out = Array5::<f64>::zeros((frames, nx, ny, rows, cols));
    for row in 0..rows {
        for col in 0..cols {
            let scalar = rank2_component(values, row, col);
            let lap = fft_ops::laplacian_scalar(scalar.view(), lx, ly)?;
            for t in 0..frames {
                for ix in 0..nx {
                    for iy in 0..ny {
                        out[[t, ix, iy, row, col]] = lap[[t, ix, iy]];
                    }
                }
            }
        }
    }
    Ok(out)
}

fn grad_p_symmetric_traceless(
    p: ndarray::ArrayView4<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<Array6<f64>> {
    let grad_p = gradient_vector3(p, lx, ly)?;
    let (frames, nx, ny, _, _) = grad_p.dim();
    let mut out = Array6::<f64>::zeros((frames, nx, ny, 2, 3, 3));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    let trace_part = (2.0 / 3.0) * grad_p[[t, ix, iy, k, k]];
                    for a in 0..3 {
                        for b in 0..3 {
                            let grad_ka = grad_p[[t, ix, iy, k, a]];
                            let grad_kb = grad_p[[t, ix, iy, k, b]];
                            out[[t, ix, iy, k, a, b]] = grad_ka * super::math::delta(k, b)
                                + grad_kb * super::math::delta(k, a)
                                - trace_part * super::math::delta(a, b);
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

fn scalar_component(values: ndarray::ArrayView4<'_, f64>, component: usize) -> Array3<f64> {
    let (frames, nx, ny, _) = values.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = values[[t, ix, iy, component]];
            }
        }
    }
    out
}

fn rank2_component(values: ndarray::ArrayView5<'_, f64>, row: usize, col: usize) -> Array3<f64> {
    let (frames, nx, ny, _, _) = values.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = values[[t, ix, iy, row, col]];
            }
        }
    }
    out
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
