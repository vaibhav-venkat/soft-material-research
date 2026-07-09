use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
use crate::coarse_grain_burn;
use crate::mechanics;
use crate::sampling;
use crate::CoreError;

fn to_py_err(error: CoreError) -> PyErr {
    // Convert Rust core errors into Python ValueError exceptions.
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
/// Return the compiled rho-fitting core crate version.
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}


#[pyfunction]
#[pyo3(signature = (valid_mask, nd, seed, replace))]
/// Sample `(frame, ix, iy)` rows from a boolean valid-mask grid.
fn sample_rows(
    py: Python<'_>,
    valid_mask: PyReadonlyArray3<'_, bool>,
    nd: usize,
    seed: u64,
    replace: bool,
) -> PyResult<Py<PyAny>> {
    let rows =
        sampling::sample_rows(valid_mask.as_array(), nd, seed, replace).map_err(to_py_err)?;
    Ok(rows.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (coords, directions, velocities, psi6_abs, mask, x_centers, theta_centers, r_centers, lx, theta_period, sigma, gamma, u0))]
/// Python wrapper for GPU Gaussian mechanical coarse-grained fields and current tensors.
#[allow(unused_variables)]
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
    #[cfg(any(feature = "gpu-metal", feature = "gpu-cuda"))]
    {
        let result = coarse_grain_burn::build_mechanical_fields(
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
        );
        let fields = result.map_err(to_py_err)?;
        let out = PyDict::new(py);
        out.set_item("rho", fields.rho.into_pyarray(py))?;
        out.set_item("P", fields.p.into_pyarray(py))?;
        out.set_item("Q", fields.q.into_pyarray(py))?;
        out.set_item("A", fields.a.into_pyarray(py))?;
        out.set_item("psi6_sq", fields.psi6_sq.into_pyarray(py))?;
        out.set_item("J_rho", fields.j_rho.into_pyarray(py))?;
        out.set_item("J_P", fields.j_p.into_pyarray(py))?;
        out.set_item("J_Q", fields.j_q.into_pyarray(py))?;
        return Ok(out.unbind());
    }

    #[cfg(not(any(feature = "gpu-metal", feature = "gpu-cuda")))]
    {
        Err(PyValueError::new_err(
            "mechanical coarse-graining requires a GPU feature; build with gpu-metal, gpu-cuda, or gpu",
        ))
    }
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
    let (y_rho, y_p, y_q) = mechanics::build_targets(
        p.as_array(),
        j_rho.as_array(),
        j_p.as_array(),
        j_q.as_array(),
        gamma,
        u0,
    )
    .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("Y_rho", y_rho.into_pyarray(py))?;
    out.set_item("Y_P", y_p.into_pyarray(py))?;
    out.set_item("Y_Q", y_q.into_pyarray(py))?;
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
    let (rows, row_index) = mechanics::sample_component_rows(
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
    m.add_function(wrap_pyfunction!(sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_fields, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_targets, m)?)?;
    m.add_function(wrap_pyfunction!(sample_component_rows, m)?)?;
    Ok(())
}
