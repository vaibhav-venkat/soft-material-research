use ndarray::{Array3, Array4, Array5, Array6, ArrayView3};

use super::math::{delta, vector_component};

pub(super) fn surface_rows_rank2(values: ndarray::ArrayView5<'_, f64>) -> Array5<f64> {
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

pub(super) fn q_dot_grad_rho(
    q: ndarray::ArrayView5<'_, f64>,
    grad: ndarray::ArrayView4<'_, f64>,
) -> Array4<f64> {
    let (frames, nx, ny, _, _) = q.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                for k in 0..2 {
                    out[[t, ix, iy, k]] = q[[t, ix, iy, k, 0]] * grad[[t, ix, iy, 0]]
                        + q[[t, ix, iy, k, 1]] * grad[[t, ix, iy, 1]];
                }
            }
        }
    }
    out
}

pub(super) fn estimate_ubar(
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

pub(super) fn scalar_times_rank2(
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

pub(super) fn scalar_times_rank3(
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

pub(super) fn vector_dot_alpha_traceless(vector: ndarray::ArrayView4<'_, f64>) -> Array6<f64> {
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
