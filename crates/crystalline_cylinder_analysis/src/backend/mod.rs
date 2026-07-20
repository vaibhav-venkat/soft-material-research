//! Numerical backend boundary for future CPU and GPU implementations.

mod cuda;

pub use cuda::{ComputeDevice, DeviceAnalysisBackend};

use crate::model::{CorrelationSeries, LaplaceGrid};

/// Backend operations that benefit from parallel or accelerated execution.
pub trait AnalysisBackend: Send + Sync {
    /// Compute all requested lagged Pearson coefficients.
    fn lagged_pearson(&self, values: &[f64], max_lag: usize) -> Vec<f64>;

    /// Evaluate the complex Laplace transform on the supplied grid.
    fn laplace_grid(
        &self,
        correlation: &CorrelationSeries,
        r: &[f64],
        omega: &[f64],
    ) -> LaplaceGrid;

    /// Solve a dense least-squares problem with a column-major result backend.
    fn linear_least_squares(
        &self,
        matrix_row_major: &[f64],
        rows: usize,
        columns: usize,
        rhs: &[f64],
        rank_tolerance: f64,
    ) -> Vec<f64>;
}
