//! Quadrature for already-sampled real and complex series.

use num_complex::Complex64;

/// Integrate uniformly sampled real values with composite Simpson quadrature.
pub fn simpson_samples(_values: &[f64], _spacing: f64) -> f64 {
    todo!("match SciPy Simpson behavior, including the even-sample correction")
}

/// Integrate uniformly sampled complex values with composite Simpson quadrature.
pub fn simpson_samples_complex(_values: &[Complex64], _spacing: f64) -> Complex64 {
    todo!("apply sampled Simpson weights to real and imaginary components")
}
