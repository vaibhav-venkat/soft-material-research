//! Robust constrained fitting of the damped-cosine correlation model.

use crate::backend::AnalysisBackend;
use crate::error::AnalysisResult;
use crate::model::{CorrelationSeries, DampedCosineFit};

/// Numerical controls for bounded soft-L1 fitting.
#[derive(Clone, Copy, Debug)]
pub struct FitConfig {
    pub soft_l1_scale: f64,
    pub tolerance: f64,
    pub maximum_evaluations: usize,
    pub rank_tolerance: f64,
}

/// Evaluate the constrained damped-cosine model.
pub fn damped_cosine(_time: &[f64], _parameters: &[f64; 4]) -> Vec<f64> {
    todo!("evaluate A exp(-r t) cos(omega t + phase) with constrained offset")
}

/// Evaluate the analytic model Jacobian in row-major sample order.
pub fn damped_cosine_jacobian(_time: &[f64], _parameters: &[f64; 4]) -> Vec<f64> {
    todo!("evaluate analytic derivatives for A, r, omega, and phase")
}

/// Fit the averaged correlation with multistart bounded least squares.
pub fn fit_damped_cosine<B: AnalysisBackend>(
    _backend: &B,
    _correlation: &CorrelationSeries,
    _omega_grid: &[f64],
    _config: FitConfig,
) -> AnalysisResult<DampedCosineFit> {
    todo!("run bounded Levenberg-Marquardt and soft-L1 IRLS iterations")
}
