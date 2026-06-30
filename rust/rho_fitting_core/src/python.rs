use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::coarse_grain;
use crate::fft_ops;
use crate::CoreError;

fn not_ready(name: &str) -> PyErr {
    PyNotImplementedError::new_err(format!("{name} is planned for a later rho_fitting step"))
}

fn to_py_err(error: CoreError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
#[pyo3(signature = (coords, p_particles, shell_mask, x_centers, y_centers, lx, ly, radius, sigma))]
fn coarse_grain_fields(
    py: Python<'_>,
    coords: PyReadonlyArray3<'_, f64>,
    p_particles: PyReadonlyArray3<'_, f64>,
    shell_mask: PyReadonlyArray2<'_, bool>,
    x_centers: PyReadonlyArray1<'_, f64>,
    y_centers: PyReadonlyArray1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
) -> PyResult<Py<PyDict>> {
    let (rho, p_density) = coarse_grain::coarse_grain_fields(
        coords.as_array(),
        p_particles.as_array(),
        shell_mask.as_array(),
        x_centers.as_array(),
        y_centers.as_array(),
        lx,
        ly,
        radius,
        sigma,
    )
    .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("rho", rho.into_pyarray(py))?;
    out.set_item("P_density", p_density.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature = (rho, p_density, lx, ly, requested))]
fn spatial_derivatives(
    py: Python<'_>,
    rho: PyReadonlyArray3<'_, f64>,
    p_density: PyReadonlyArray4<'_, f64>,
    lx: f64,
    ly: f64,
    requested: Vec<String>,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    for name in requested {
        match name.as_str() {
            "grad_rho" => {
                let value = fft_ops::gradient_scalar(rho.as_array(), lx, ly).map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            "lap_rho" => {
                let value = fft_ops::laplacian_scalar(rho.as_array(), lx, ly).map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            "bilap_rho" => {
                let value =
                    fft_ops::bilaplacian_scalar(rho.as_array(), lx, ly).map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            "div_p" => {
                let value =
                    fft_ops::divergence_vector(p_density.as_array(), lx, ly).map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            "lap_p" => {
                let value =
                    fft_ops::laplacian_vector(p_density.as_array(), lx, ly).map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            "bilap_p" => {
                let value =
                    fft_ops::bilaplacian_vector(p_density.as_array(), lx, ly).map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            "grad_div_p" => {
                let value =
                    fft_ops::grad_div_vector(p_density.as_array(), lx, ly).map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            _ => return Err(PyValueError::new_err(format!("unknown derivative: {name}"))),
        }
    }
    Ok(out.unbind())
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
