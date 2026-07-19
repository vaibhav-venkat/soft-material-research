//! Frequency-domain and model-fit results.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Evaluated complex Laplace transform on an `(omega, r)` grid.
#[derive(Clone, Debug)]
pub struct LaplaceGrid {
    pub r: Vec<f64>,
    pub omega: Vec<f64>,
    pub values: Vec<Complex64>,
    pub shape: [usize; 2],
}

/// Axis along which a preferred transform coordinate is searched.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PreferredAxis {
    Omega,
    R,
}

/// Maximum-log-magnitude estimate along one transform axis.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreferredEstimate {
    pub axis: PreferredAxis,
    pub coordinate: f64,
    pub coordinate_std: f64,
    pub log10_magnitude: f64,
    pub at_lower_boundary: bool,
    pub at_upper_boundary: bool,
    pub replicate_count: usize,
}

/// Parameters and diagnostics for the constrained damped-cosine model.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DampedCosineFit {
    pub amplitude: f64,
    pub rate: f64,
    pub omega: f64,
    pub phase: f64,
    pub offset: f64,
    pub r_squared: f64,
    pub evaluations: usize,
    pub converged: bool,
    pub rate_at_lower_boundary: bool,
    pub rate_at_upper_boundary: bool,
    pub amplitude_at_upper_boundary: bool,
    pub prediction: Vec<f64>,
}
