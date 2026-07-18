//! Rayon and Tenferro CPU backend declaration.

use crate::backend::AnalysisBackend;
use crate::correlation::pearson;
use crate::model::{CorrelationSeries, LaplaceGrid};
use rayon::prelude::*;
use tenferro_cpu::CpuContext;

/// CPU backend backed by Rayon and Tenferro's faer provider.
#[derive(Clone, Debug)]
pub struct CpuAnalysisBackend {
    pub thread_count: usize,
    context: CpuContext,
}

impl CpuAnalysisBackend {
    /// Construct a CPU backend with an optional thread limit.
    pub fn new(thread_count: Option<usize>) -> Self {
        let context = match thread_count {
            Some(0) => panic!("thread count must be positive"),
            Some(count) => CpuContext::with_threads(count),
            None => CpuContext::try_from_env(),
        }
        .expect("CPU backend initialization failed");

        Self {
            thread_count: context.num_threads(),
            context,
        }
    }

    /// Execute work inside the backend's configured Rayon pool.
    pub fn install<R: Send>(&self, operation: impl FnOnce() -> R + Send) -> R {
        self.context.install(operation)
    }
}

impl AnalysisBackend for CpuAnalysisBackend {
    fn lagged_pearson(&self, values: &[f64], max_lag: usize) -> Vec<f64> {
        assert!(values.len() >= 2, "Pearson needs two samples");
        assert!(
            max_lag <= values.len() - 2,
            "lag leaves fewer than two pairs"
        );
        self.install(|| {
            (0..=max_lag)
                .into_par_iter()
                .map(|lag| {
                    if lag == 0 {
                        let _ = pearson(values, values);
                        1.0
                    } else {
                        pearson(&values[..values.len() - lag], &values[lag..])
                    }
                })
                .collect()
        })
    }

    fn laplace_grid(
        &self,
        _correlation: &CorrelationSeries,
        _r: &[f64],
        _omega: &[f64],
    ) -> LaplaceGrid {
        todo!("evaluate independent Laplace grid rows in parallel")
    }

    fn linear_least_squares(
        &self,
        _matrix_row_major: &[f64],
        _rows: usize,
        _columns: usize,
        _rhs: &[f64],
        _rank_tolerance: f64,
    ) -> Vec<f64> {
        todo!("solve through tenferro-linalg SVD or pseudoinverse")
    }
}
