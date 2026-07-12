use ndarray::{Array4, ArrayD, Ix5, Ix6, IxDyn};
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::coarse_grain_burn;
use crate::fitting;
use crate::mechanics::{self, CurrentQField};
use crate::{CoreError, CoreResult};

fn to_py_err(error: CoreError) -> PyErr {
    // Convert Rust core errors into Python ValueError exceptions.
    PyValueError::new_err(error.to_string())
}

fn rank3_field_to_dynamic(field: &CurrentQField) -> ArrayD<f64> {
    let (frames, nx, ntheta, nr) = field.dim();
    let mut out = ArrayD::zeros(IxDyn(&[frames, nx, ntheta, nr, 3, 3, 3]));
    for t in 0..frames {
        for ix in 0..nx {
            for itheta in 0..ntheta {
                for ir in 0..nr {
                    for flux in 0..3 {
                        for row in 0..3 {
                            for col in 0..3 {
                                out[[t, ix, itheta, ir, flux, row, col]] =
                                    field[[t, ix, itheta, ir]][flux][row][col];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn rank3_field_from_dynamic(
    field: ndarray::ArrayViewD<'_, f64>,
    name: &str,
) -> CoreResult<CurrentQField> {
    if field.ndim() != 7 || field.shape()[4..] != [3, 3, 3] {
        return Err(CoreError::Shape(format!(
            "{name} must have shape (T,Nx,Ntheta,Nr,3,3,3)"
        )));
    }
    let shape = field.shape();
    let mut out = Array4::from_elem((shape[0], shape[1], shape[2], shape[3]), [[[0.0; 3]; 3]; 3]);
    for t in 0..shape[0] {
        for ix in 0..shape[1] {
            for itheta in 0..shape[2] {
                for ir in 0..shape[3] {
                    for flux in 0..3 {
                        for row in 0..3 {
                            for col in 0..3 {
                                out[[t, ix, itheta, ir]][flux][row][col] =
                                    field[[t, ix, itheta, ir, flux, row, col]];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[pyfunction]
/// Return the compiled rho-fitting core crate version.
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(signature = (valid_mask, nd, seed, replace))]
/// Sample `(frame, ix, theta, r)` rows from a boolean valid-mask grid.
fn sample_grid_rows(
    py: Python<'_>,
    valid_mask: PyReadonlyArrayDyn<'_, bool>,
    nd: usize,
    seed: u64,
    replace: bool,
) -> PyResult<Py<PyAny>> {
    let rows =
        fitting::sample_grid_rows(valid_mask.as_array(), nd, seed, replace).map_err(to_py_err)?;
    Ok(rows.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
/// Build a finite-value `(T,Nx,Ntheta,Nr)` mask from a sequence of fields.
fn finite_grid_mask(
    py: Python<'_>,
    fields: Vec<PyReadonlyArrayDyn<'_, f64>>,
) -> PyResult<Py<PyAny>> {
    let views = fields
        .iter()
        .map(|field| field.as_array())
        .collect::<Vec<_>>();
    let mask = fitting::finite_grid_mask(&views).map_err(to_py_err)?;
    Ok(mask.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
/// Build the alignment tensor `A = Q + rho I / 3`.
fn build_alignment_tensor(
    py: Python<'_>,
    rho: PyReadonlyArrayDyn<'_, f64>,
    q: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let value = fitting::alignment_tensor(rho.as_array(), q.as_array()).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
/// Contract `A` with a cylindrical density gradient.
fn contract_alignment_gradient(
    py: Python<'_>,
    a: PyReadonlyArrayDyn<'_, f64>,
    grad_rho: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let value =
        fitting::alignment_dot_gradient(a.as_array(), grad_rho.as_array()).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
/// Estimate the scalar speed by projecting `Y_P` onto `A`.
fn estimate_ubar(
    py: Python<'_>,
    y_p: PyReadonlyArrayDyn<'_, f64>,
    a: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let value = fitting::estimate_ubar(y_p.as_array(), a.as_array()).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
/// Build the symmetric traceless P-alignment tensor.
fn build_p_alignment(py: Python<'_>, p: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Py<PyAny>> {
    let value = fitting::p_alignment_traceless(p.as_array()).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
/// Multiply every trailing tensor component by a scalar field.
fn scale_by_scalar(
    py: Python<'_>,
    scalar: PyReadonlyArrayDyn<'_, f64>,
    values: PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let value =
        fitting::scale_by_scalar(scalar.as_array(), values.as_array()).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (values, mode))]
/// Keep tangential (`0`) or radial (`1`) flux directions.
fn project_flux_directions(
    py: Python<'_>,
    values: PyReadonlyArrayDyn<'_, f64>,
    mode: u8,
) -> PyResult<Py<PyAny>> {
    let value = fitting::project_flux_directions(values.as_array(), mode).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
/// Return a coefficient-weighted sum of same-shaped fields.
fn weighted_linear_combination(
    py: Python<'_>,
    fields: Vec<PyReadonlyArrayDyn<'_, f64>>,
    coefficients: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let views = fields
        .iter()
        .map(|field| field.as_array())
        .collect::<Vec<_>>();
    let value =
        fitting::weighted_linear_combination(&views, coefficients.as_array()).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (target_divergence, divergence_library, sample_coordinates, target_flux=None, flux_library=None, flux_weight=0.0))]
/// Assemble finite divergence rows and optional weighted flux rows for regression.
fn assemble_regression_rows(
    py: Python<'_>,
    target_divergence: PyReadonlyArrayDyn<'_, f64>,
    divergence_library: PyReadonlyArrayDyn<'_, f64>,
    sample_coordinates: PyReadonlyArray2<'_, i64>,
    target_flux: Option<PyReadonlyArrayDyn<'_, f64>>,
    flux_library: Option<PyReadonlyArrayDyn<'_, f64>>,
    flux_weight: f64,
) -> PyResult<Py<PyDict>> {
    let rows = fitting::assemble_regression_rows(
        target_divergence.as_array(),
        divergence_library.as_array(),
        sample_coordinates.as_array(),
        target_flux.as_ref().map(|value| value.as_array()),
        flux_library.as_ref().map(|value| value.as_array()),
        flux_weight,
    )
    .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("X", rows.x.into_pyarray(py))?;
    out.set_item("y", rows.y.into_pyarray(py))?;
    out.set_item("divergence_X", rows.divergence_x.into_pyarray(py))?;
    out.set_item("divergence_y", rows.divergence_y.into_pyarray(py))?;
    out.set_item("divergence_rows", rows.divergence_rows.into_pyarray(py))?;
    out.set_item("row_index", rows.row_index.into_pyarray(py))?;
    if let Some(flux_x) = rows.flux_x {
        out.set_item("flux_X", flux_x.into_pyarray(py))?;
    } else {
        out.set_item("flux_X", py.None())?;
    }
    if let Some(flux_y) = rows.flux_y {
        out.set_item("flux_y", flux_y.into_pyarray(py))?;
    } else {
        out.set_item("flux_y", py.None())?;
    }
    if let Some(flux_row_index) = rows.flux_row_index {
        out.set_item("flux_row_index", flux_row_index.into_pyarray(py))?;
    } else {
        out.set_item("flux_row_index", py.None())?;
    }
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature = (coords, directions, velocities, psi6_abs, mask, x_centers, theta_centers, r_centers, lx, theta_period, sigma, gamma, u0))]
/// Python wrapper for Burn Gaussian mechanical coarse-grained fields and current tensors.
fn build_mechanical_fields(
    py: Python<'_>,
    coords: PyReadonlyArray3<'_, f64>,
    directions: PyReadonlyArray3<'_, f64>,
    velocities: PyReadonlyArray3<'_, f64>,
    psi6_abs: PyReadonlyArray2<'_, f64>,
    mask: PyReadonlyArray2<'_, bool>,
    x_centers: PyReadonlyArray1<'_, f64>,
    theta_centers: PyReadonlyArray1<'_, f64>,
    r_centers: PyReadonlyArray1<'_, f64>,
    lx: f64,
    theta_period: f64,
    sigma: f64,
    gamma: f64,
    u0: f64,
) -> PyResult<Py<PyDict>> {
    let fields = coarse_grain_burn::build_mechanical_fields(
        coords.as_array(),
        directions.as_array(),
        velocities.as_array(),
        psi6_abs.as_array(),
        mask.as_array(),
        x_centers.as_array(),
        theta_centers.as_array(),
        r_centers.as_array(),
        lx,
        theta_period,
        sigma,
        gamma,
        u0,
    )
    .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("rho", fields.rho.into_pyarray(py))?;
    out.set_item("P", fields.p.into_pyarray(py))?;
    out.set_item("Q", fields.q.into_pyarray(py))?;
    out.set_item("A", fields.a.into_pyarray(py))?;
    out.set_item("psi6_sq", fields.psi6_sq.into_pyarray(py))?;
    out.set_item("J_rho", fields.j_rho.into_pyarray(py))?;
    out.set_item("J_P", fields.j_p.into_pyarray(py))?;
    out.set_item("J_Q", rank3_field_to_dynamic(&fields.j_q).into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature = (p, j_rho, j_p, j_q, gamma, u0))]
/// Convert filtered currents into mechanical closure targets.
fn build_mechanical_targets(
    py: Python<'_>,
    p: PyReadonlyArrayDyn<'_, f64>,
    j_rho: PyReadonlyArrayDyn<'_, f64>,
    j_p: PyReadonlyArrayDyn<'_, f64>,
    j_q: PyReadonlyArrayDyn<'_, f64>,
    gamma: f64,
    u0: f64,
) -> PyResult<Py<PyDict>> {
    let p = p
        .as_array()
        .into_dimensionality::<Ix5>()
        .map_err(|_| to_py_err(CoreError::Shape("P must have rank 5".to_string())))?;
    let j_rho = j_rho
        .as_array()
        .into_dimensionality::<Ix5>()
        .map_err(|_| to_py_err(CoreError::Shape("J_rho must have rank 5".to_string())))?;
    let j_p = j_p
        .as_array()
        .into_dimensionality::<Ix6>()
        .map_err(|_| to_py_err(CoreError::Shape("J_P must have rank 6".to_string())))?;
    let j_q = rank3_field_from_dynamic(j_q.as_array(), "J_Q").map_err(to_py_err)?;
    let targets =
        mechanics::build_targets(p, j_rho, j_p, j_q.view(), gamma, u0).map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("Y_rho", targets.y_rho.into_pyarray(py))?;
    out.set_item("Y_P", targets.y_p.into_pyarray(py))?;
    out.set_item("Y_Q", rank3_field_to_dynamic(&targets.y_q).into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature = (target, library, sample_indices))]
/// Sample target/library tensor components into regression rows and metadata.
fn sample_component_rows(
    py: Python<'_>,
    target: PyReadonlyArrayDyn<'_, f64>,
    library: PyReadonlyArrayDyn<'_, f64>,
    sample_indices: PyReadonlyArray2<'_, i64>,
) -> PyResult<Py<PyDict>> {
    let (rows, row_index) = fitting::sample_component_rows(
        target.as_array(),
        library.as_array(),
        sample_indices.as_array(),
    )
    .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("rows", rows.into_pyarray(py))?;
    out.set_item("row_index", row_index.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pymodule]
/// Register the Python module functions exposed by the compiled extension.
fn _rho_fitting_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(sample_grid_rows, m)?)?;
    m.add_function(wrap_pyfunction!(finite_grid_mask, m)?)?;
    m.add_function(wrap_pyfunction!(build_alignment_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(contract_alignment_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_ubar, m)?)?;
    m.add_function(wrap_pyfunction!(build_p_alignment, m)?)?;
    m.add_function(wrap_pyfunction!(scale_by_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(project_flux_directions, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_linear_combination, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_regression_rows, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_fields, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_targets, m)?)?;
    m.add_function(wrap_pyfunction!(sample_component_rows, m)?)?;
    Ok(())
}
