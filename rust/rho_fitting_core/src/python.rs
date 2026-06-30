use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn not_ready(name: &str) -> PyErr {
    PyNotImplementedError::new_err(format!("{name} is planned for a later rho_fitting step"))
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn coarse_grain_fields(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let _ = py;
    Err(not_ready("coarse_grain_fields"))
}

#[pyfunction]
fn spatial_derivatives(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let _ = py;
    Err(not_ready("spatial_derivatives"))
}

#[pyfunction]
fn sample_rows() -> PyResult<()> {
    Err(not_ready("sample_rows"))
}

#[pyfunction]
fn build_density_library() -> PyResult<()> {
    Err(not_ready("build_density_library"))
}

#[pyfunction]
fn build_polarization_library() -> PyResult<()> {
    Err(not_ready("build_polarization_library"))
}

#[pyfunction]
fn stlsq() -> PyResult<()> {
    Err(not_ready("stlsq"))
}

#[pymodule]
fn _rho_fitting_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(coarse_grain_fields, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(build_density_library, m)?)?;
    m.add_function(wrap_pyfunction!(build_polarization_library, m)?)?;
    m.add_function(wrap_pyfunction!(stlsq, m)?)?;
    Ok(())
}
