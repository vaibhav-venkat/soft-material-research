use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::coarse_grain;
#[cfg(feature = "gpu-metal")]
use crate::coarse_grain_cubecl;
use crate::fft_ops;
use crate::library;
use crate::sampling;
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
    #[cfg(feature = "gpu-metal")]
    let gpu_requested = std::env::var("RHO_FITTING_GPU").is_ok_and(|value| value == "1");
    #[cfg(not(feature = "gpu-metal"))]
    let gpu_requested = false;

    #[cfg(feature = "gpu-metal")]
    let result = if gpu_requested {
        println!("[rho_fitting] GPU requested: RHO_FITTING_GPU=1, gpu-metal feature enabled");
        coarse_grain_cubecl::coarse_grain_fields(
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
        .or_else(|error| {
            eprintln!("[rho_fitting] GPU coarse-grain failed, falling back to CPU: {error}");
            coarse_grain::coarse_grain_fields(
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
        })
    } else {
        println!("[rho_fitting] GPU not requested; using CPU coarse-grain");
        coarse_grain::coarse_grain_fields(
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
    };

    #[cfg(not(feature = "gpu-metal"))]
    if gpu_requested {
        eprintln!("[rho_fitting] RHO_FITTING_GPU=1 ignored; extension was built without gpu-metal");
    } else {
        println!("[rho_fitting] GPU not requested; using CPU coarse-grain");
    }
    #[cfg(not(feature = "gpu-metal"))]
    let result = coarse_grain::coarse_grain_fields(
        coords.as_array(),
        p_particles.as_array(),
        shell_mask.as_array(),
        x_centers.as_array(),
        y_centers.as_array(),
        lx,
        ly,
        radius,
        sigma,
    );

    let (rho, p_density) = result.map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("rho", rho.into_pyarray(py))?;
    out.set_item("P_density", p_density.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature = (rho, lx, ly, requested))]
fn spatial_derivatives(
    py: Python<'_>,
    rho: PyReadonlyArray3<'_, f64>,
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
            _ if lap_order(&name).is_some() => {
                let value = fft_ops::repeated_laplacian_scalar(
                    rho.as_array(),
                    lx,
                    ly,
                    lap_order(&name).unwrap(),
                )
                .map_err(to_py_err)?;
                out.set_item(name, value.into_pyarray(py))?;
            }
            _ => return Err(PyValueError::new_err(format!("unknown derivative: {name}"))),
        }
    }
    Ok(out.unbind())
}

#[pyfunction]
#[pyo3(signature = (valid_mask, nd, seed, replace))]
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
#[pyo3(signature = (rho, p_density, j_density, source_cross, sample_indices, term_names, lx, ly))]
fn build_density_library(
    py: Python<'_>,
    rho: PyReadonlyArray3<'_, f64>,
    p_density: PyReadonlyArray4<'_, f64>,
    j_density: PyReadonlyArray4<'_, f64>,
    source_cross: PyReadonlyArray3<'_, f64>,
    sample_indices: PyReadonlyArray2<'_, i64>,
    term_names: Vec<String>,
    lx: f64,
    ly: f64,
) -> PyResult<Py<PyAny>> {
    for name in &term_names {
        if !library::known_density_term(name) {
            return Err(PyValueError::new_err(format!(
                "unknown density term: {name}"
            )));
        }
    }
    let values = library::build_density_library(
        rho.as_array(),
        p_density.as_array(),
        j_density.as_array(),
        source_cross.as_array(),
        sample_indices.as_array(),
        &term_names,
        lx,
        ly,
    )
    .map_err(to_py_err)?;
    Ok(values.into_pyarray(py).into_any().unbind())
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
    m.add_function(wrap_pyfunction!(stlsq, m)?)?;
    Ok(())
}

fn lap_order(name: &str) -> Option<usize> {
    if name == "lap_rho" {
        return Some(1);
    }
    name.strip_prefix("lap")?
        .strip_suffix("_rho")?
        .parse::<usize>()
        .ok()
}
