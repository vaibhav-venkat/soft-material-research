use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::coarse_grain;
#[cfg(feature = "gpu-metal")]
use crate::coarse_grain_burn;
use crate::fft_ops;
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
#[pyo3(signature = (coords, p_particles, shell_mask, x_centers, y_centers, lx, ly, radius, sigma))]
/// Python wrapper for legacy density and polarization coarse-graining.
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
        println!("[rho_fitting] Burn GPU requested: RHO_FITTING_GPU=1, gpu-metal feature enabled");
        coarse_grain_burn::coarse_grain_fields(
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
            eprintln!("[rho_fitting] Burn GPU coarse-grain failed, falling back to CPU: {error}");
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
        println!("[rho_fitting] Burn GPU not requested; using CPU coarse-grain");
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
/// Return selected spectral derivative fields for a scalar rho grid.
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
/// Python wrapper for mechanical coarse-grained fields and current tensors.
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
    #[cfg(feature = "gpu-metal")]
    let gpu_requested = std::env::var("RHO_FITTING_GPU").is_ok_and(|value| value == "1");
    #[cfg(not(feature = "gpu-metal"))]
    let gpu_requested = false;

    #[cfg(feature = "gpu-metal")]
    let result = if gpu_requested {
        println!("[rho_fitting] Burn GPU requested for mechanical fields");
        coarse_grain_burn::build_mechanical_fields(
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
        .or_else(|error| {
            eprintln!(
                "[rho_fitting] Burn mechanical coarse-grain failed, falling back to CPU: {error}"
            );
            mechanics::build_mechanical_fields(
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
        })
    } else {
        mechanics::build_mechanical_fields(
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
    };

    #[cfg(not(feature = "gpu-metal"))]
    if gpu_requested {
        eprintln!("[rho_fitting] RHO_FITTING_GPU=1 ignored; extension was built without gpu-metal");
    }
    #[cfg(not(feature = "gpu-metal"))]
    let result = mechanics::build_mechanical_fields(
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
#[pyo3(signature = (rho, p, q, a, psi6_sq, y_p, lx, ly))]
/// Build mechanical candidate flux libraries and coefficient-order names.
fn build_mechanical_libraries(
    py: Python<'_>,
    rho: PyReadonlyArray3<'_, f64>,
    p: PyReadonlyArrayDyn<'_, f64>,
    q: PyReadonlyArrayDyn<'_, f64>,
    a: PyReadonlyArrayDyn<'_, f64>,
    psi6_sq: PyReadonlyArray3<'_, f64>,
    y_p: PyReadonlyArrayDyn<'_, f64>,
    lx: f64,
    ly: f64,
) -> PyResult<Py<PyDict>> {
    let libs = mechanics::build_mechanical_libraries(
        rho.as_array(),
        p.as_array(),
        q.as_array(),
        a.as_array(),
        psi6_sq.as_array(),
        y_p.as_array(),
        lx,
        ly,
    )
    .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("Y_rho_names", libs.y_rho_names)?;
    out.set_item("Y_P_names", libs.y_p_names)?;
    out.set_item("Y_Q_names", libs.y_q_names)?;
    out.set_item("Y_rho", libs.y_rho.into_pyarray(py))?;
    out.set_item("Y_P", libs.y_p.into_pyarray(py))?;
    out.set_item("Y_Q", libs.y_q.into_pyarray(py))?;
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
    m.add_function(wrap_pyfunction!(coarse_grain_fields, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(sample_rows, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_fields, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_targets, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_libraries, m)?)?;
    m.add_function(wrap_pyfunction!(sample_component_rows, m)?)?;
    Ok(())
}

fn lap_order(name: &str) -> Option<usize> {
    // Parse derivative names like `lap_rho` and `lap2_rho` into Laplacian order.
    if name == "lap_rho" {
        return Some(1);
    }
    name.strip_prefix("lap")?
        .strip_suffix("_rho")?
        .parse::<usize>()
        .ok()
}
