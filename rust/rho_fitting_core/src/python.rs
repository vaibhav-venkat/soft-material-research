use ndarray::{Array4, ArrayD, Ix5, Ix6, IxDyn};
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::coarse_grain_burn;
use crate::fitting;
use crate::interpolation;
use crate::mechanics::{self, CurrentQField};
use crate::particles;
use crate::regression;
use crate::spectral;
use crate::temporal;
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
/// Sum squared temporal coefficients over every non-mode axis.
fn temporal_power_spectrum(
    py: Python<'_>,
    coefficients: Vec<PyReadonlyArrayDyn<'_, f64>>,
) -> PyResult<Py<PyAny>> {
    let views = coefficients
        .iter()
        .map(|values| values.as_array())
        .collect::<Vec<_>>();
    let result = temporal::power_spectrum(&views).map_err(to_py_err)?;
    Ok(result.into_pyarray(py).into_any().unbind())
}

#[pyclass(name = "TemporalOperators")]
struct PyTemporalOperators {
    inner: temporal::TemporalOperators,
}

#[pyclass(name = "RadialTransfer")]
struct PyRadialTransfer {
    inner: interpolation::RadialTransfer,
}

#[pymethods]
impl PyRadialTransfer {
    #[new]
    fn new(source: PyReadonlyArray1<'_, f64>, target: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: interpolation::RadialTransfer::new(source.as_array(), target.as_array())
                .map_err(to_py_err)?,
        })
    }

    fn matrix(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner
            .matrix()
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    fn apply(
        &self,
        py: Python<'_>,
        values: PyReadonlyArrayDyn<'_, f64>,
        axis: usize,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .apply(values.as_array(), axis)
            .map_err(to_py_err)?;
        Ok(result.into_pyarray(py).into_any().unbind())
    }
}

#[pyclass(name = "CylindricalSpectralOperators")]
struct PyCylindricalSpectralOperators {
    inner: spectral::CylindricalSpectralOperators,
}

#[pymethods]
impl PyCylindricalSpectralOperators {
    #[new]
    #[pyo3(signature = (lx, theta_period, r_min, r_max, nx, ntheta, nr))]
    fn new(
        lx: f64,
        theta_period: f64,
        r_min: f64,
        r_max: f64,
        nx: usize,
        ntheta: usize,
        nr: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: spectral::CylindricalSpectralOperators::new(
                lx,
                theta_period,
                r_min,
                r_max,
                nx,
                ntheta,
                nr,
            )
            .map_err(to_py_err)?,
        })
    }

    fn radial_nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner
            .radial_nodes
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    #[pyo3(signature = (values, direction, grid_offset=0))]
    fn derivative(
        &self,
        py: Python<'_>,
        values: PyReadonlyArrayDyn<'_, f64>,
        direction: usize,
        grid_offset: usize,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .derivative(values.as_array(), grid_offset, direction)
            .map_err(to_py_err)?;
        Ok(result.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(signature = (values, grid_offset=0))]
    fn gradient(
        &self,
        py: Python<'_>,
        values: PyReadonlyArrayDyn<'_, f64>,
        grid_offset: usize,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .gradient(values.as_array(), grid_offset)
            .map_err(to_py_err)?;
        Ok(result.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(signature = (values, grid_offset=0))]
    fn divergence(
        &self,
        py: Python<'_>,
        values: PyReadonlyArrayDyn<'_, f64>,
        grid_offset: usize,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .divergence(values.as_array(), grid_offset)
            .map_err(to_py_err)?;
        Ok(result.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(signature = (values, grid_offset=0))]
    fn laplacian(
        &self,
        py: Python<'_>,
        values: PyReadonlyArrayDyn<'_, f64>,
        grid_offset: usize,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .laplacian(values.as_array(), grid_offset)
            .map_err(to_py_err)?;
        Ok(result.into_pyarray(py).into_any().unbind())
    }

    #[pyo3(signature = (values, grid_offset=0))]
    fn filter_two_thirds(
        &self,
        py: Python<'_>,
        values: PyReadonlyArrayDyn<'_, f64>,
        grid_offset: usize,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .filter_two_thirds(values.as_array(), grid_offset)
            .map_err(to_py_err)?;
        Ok(result.into_pyarray(py).into_any().unbind())
    }
}

#[pyfunction]
fn barycentric_matrix(
    py: Python<'_>,
    source: PyReadonlyArray1<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let result = interpolation::barycentric_matrix(source.as_array(), target.as_array())
        .map_err(to_py_err)?;
    Ok(result.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
fn transfer_radial(
    py: Python<'_>,
    values: PyReadonlyArrayDyn<'_, f64>,
    matrix: PyReadonlyArray2<'_, f64>,
    axis: usize,
) -> PyResult<Py<PyAny>> {
    let transfer =
        interpolation::RadialTransfer::from_matrix(matrix.as_array()).map_err(to_py_err)?;
    let result = transfer.apply(values.as_array(), axis).map_err(to_py_err)?;
    Ok(result.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (x, y, lambda, tolerance, max_iterations, non_positive, non_negative))]
/// Fit one column-normalized, sign-constrained L1 regression problem.
fn fit_constrained_lasso(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    lambda: f64,
    tolerance: f64,
    max_iterations: usize,
    non_positive: PyReadonlyArray1<'_, i64>,
    non_negative: PyReadonlyArray1<'_, i64>,
) -> PyResult<Py<PyDict>> {
    let result = regression::fit_constrained_lasso(
        x.as_array(),
        y.as_array(),
        lambda,
        tolerance,
        max_iterations,
        non_positive.as_array(),
        non_negative.as_array(),
    )
    .map_err(to_py_err)?;
    let out = PyDict::new(py);
    out.set_item("coefficients", result.coefficients.into_pyarray(py))?;
    out.set_item(
        "normalized_coefficients",
        result.normalized_coefficients.into_pyarray(py),
    )?;
    out.set_item("column_norms", result.column_norms.into_pyarray(py))?;
    out.set_item("objective", result.objective)?;
    out.set_item("status", result.status)?;
    out.set_item("iterations", result.iterations)?;
    out.set_item("primal_residual", result.primal_residual)?;
    out.set_item("dual_residual", result.dual_residual)?;
    out.set_item("gap", result.gap)?;
    Ok(out.unbind())
}

#[pymethods]
impl PyTemporalOperators {
    #[new]
    #[pyo3(signature = (steps, timestep, cutoff))]
    fn new(steps: PyReadonlyArray1<'_, i64>, timestep: f64, cutoff: usize) -> PyResult<Self> {
        Ok(Self {
            inner: temporal::TemporalOperators::new(steps.as_array(), timestep, cutoff)
                .map_err(to_py_err)?,
        })
    }

    /// Apply the shared temporal fit/evaluation operators to one contiguous field block.
    fn apply(&self, py: Python<'_>, values: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Py<PyDict>> {
        let result = self.inner.apply(values.as_array()).map_err(to_py_err)?;
        let out = PyDict::new(py);
        out.set_item("cleaned", result.cleaned.into_pyarray(py))?;
        out.set_item("filtered", result.filtered.into_pyarray(py))?;
        out.set_item("derivative", result.derivative.into_pyarray(py))?;
        out.set_item("coefficients", result.coefficients.into_pyarray(py))?;
        Ok(out.unbind())
    }

    /// Apply the same precomputed operators to multiple field blocks.
    fn apply_many(
        &self,
        py: Python<'_>,
        fields: Vec<PyReadonlyArrayDyn<'_, f64>>,
    ) -> PyResult<Py<PyList>> {
        let out = PyList::empty(py);
        for field in fields {
            let result = self.inner.apply(field.as_array()).map_err(to_py_err)?;
            let item = PyDict::new(py);
            item.set_item("cleaned", result.cleaned.into_pyarray(py))?;
            item.set_item("filtered", result.filtered.into_pyarray(py))?;
            item.set_item("derivative", result.derivative.into_pyarray(py))?;
            item.set_item("coefficients", result.coefficients.into_pyarray(py))?;
            out.append(item)?;
        }
        Ok(out.unbind())
    }

    /// Return scaled time coordinates used by the Chebyshev fit.
    fn scaled_times(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner
            .scaled_times
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Return physical time coordinates used by the Chebyshev fit.
    fn times(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner
            .times
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Return ascending Chebyshev-Lobatto nodes used by the diagnostic interpolation.
    fn diagnostic_nodes(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner
            .diagnostic_nodes
            .clone()
            .into_pyarray(py)
            .into_any()
            .unbind()
    }

    /// Fit diagnostic coefficients for values already resampled on the Lobatto nodes.
    fn diagnostic_coefficients(
        &self,
        py: Python<'_>,
        values_at_nodes: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .diagnostic_coefficients(values_at_nodes.as_array())
            .map_err(to_py_err)?;
        Ok(result.into_pyarray(py).into_any().unbind())
    }
}

#[pyfunction]
#[pyo3(signature = (orientation))]
/// Rotate HOOMD quaternions onto the body-frame active axis.
fn particle_active_direction(
    py: Python<'_>,
    orientation: PyReadonlyArray3<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let value =
        particles::active_direction_from_quaternion(orientation.as_array()).map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (vectors, theta))]
/// Project Cartesian vectors to `(x, radial, azimuthal)` components.
fn cartesian_to_cylindrical(
    py: Python<'_>,
    vectors: PyReadonlyArray3<'_, f64>,
    theta: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let value =
        particles::cartesian_to_cylindrical_components(vectors.as_array(), theta.as_array())
            .map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (coords, direction_cylindrical=None, active_direction=None, orientation=None))]
/// Convert particle directions into canonical `(x, e_theta, e_r)` order.
fn particle_directions(
    py: Python<'_>,
    coords: PyReadonlyArray3<'_, f64>,
    direction_cylindrical: Option<PyReadonlyArray3<'_, f64>>,
    active_direction: Option<PyReadonlyArray3<'_, f64>>,
    orientation: Option<PyReadonlyArray3<'_, f64>>,
) -> PyResult<Py<PyAny>> {
    let value = particles::tangential_particle_vectors(particles::DirectionInputs {
        coords: coords.as_array(),
        direction_cylindrical: direction_cylindrical.as_ref().map(|value| value.as_array()),
        active_direction: active_direction.as_ref().map(|value| value.as_array()),
        orientation: orientation.as_ref().map(|value| value.as_array()),
    })
    .map_err(to_py_err)?;
    Ok(value.into_pyarray(py).into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (coords, steps, timestep, lx, theta_period))]
/// Estimate periodic particle velocities in canonical cylindrical component order.
fn particle_surface_velocities(
    py: Python<'_>,
    coords: PyReadonlyArray3<'_, f64>,
    steps: PyReadonlyArray1<'_, i64>,
    timestep: f64,
    lx: f64,
    theta_period: f64,
) -> PyResult<Py<PyAny>> {
    let value = particles::surface_velocities(particles::VelocityInputs {
        coords: coords.as_array(),
        steps: steps.as_array(),
        timestep,
        lx,
        theta_period,
    })
    .map_err(to_py_err)?;
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
    m.add_function(wrap_pyfunction!(temporal_power_spectrum, m)?)?;
    m.add_class::<PyTemporalOperators>()?;
    m.add_class::<PyRadialTransfer>()?;
    m.add_class::<PyCylindricalSpectralOperators>()?;
    m.add_function(wrap_pyfunction!(barycentric_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(transfer_radial, m)?)?;
    m.add_function(wrap_pyfunction!(fit_constrained_lasso, m)?)?;
    m.add_function(wrap_pyfunction!(particle_active_direction, m)?)?;
    m.add_function(wrap_pyfunction!(cartesian_to_cylindrical, m)?)?;
    m.add_function(wrap_pyfunction!(particle_directions, m)?)?;
    m.add_function(wrap_pyfunction!(particle_surface_velocities, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_regression_rows, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_fields, m)?)?;
    m.add_function(wrap_pyfunction!(build_mechanical_targets, m)?)?;
    m.add_function(wrap_pyfunction!(sample_component_rows, m)?)?;
    Ok(())
}
