//! Rayon and Tenferro CPU backend declaration.

use crate::backend::AnalysisBackend;
use crate::correlation::pearson;
use crate::integration::simpson_weights;
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
        correlation: &CorrelationSeries,
        r: &[f64],
        omega: &[f64],
    ) -> LaplaceGrid {
        let sample_count = correlation.lag_times.len();
        assert_eq!(
            correlation.pearson_mean.len(),
            sample_count,
            "correlation lengths differ"
        );
        assert!(sample_count >= 2, "Laplace transform needs two lags");
        assert!(!r.is_empty(), "r grid is empty");
        assert!(!omega.is_empty(), "omega grid is empty");
        let spacing = correlation.lag_times[1] - correlation.lag_times[0];
        let weights = simpson_weights(sample_count, spacing);
        let column_count = r.len();
        let values = self.install(|| {
            (0..omega.len() * column_count)
                .into_par_iter()
                .map(|flat_index| {
                    let omega_value = omega[flat_index / column_count];
                    let r_value = r[flat_index % column_count];
                    let mut real = 0.0;
                    let mut real_compensation = 0.0;
                    let mut imaginary = 0.0;
                    let mut imaginary_compensation = 0.0;
                    for ((&time, &correlation_value), &weight) in correlation
                        .lag_times
                        .iter()
                        .zip(&correlation.pearson_mean)
                        .zip(&weights)
                    {
                        let envelope = (r_value * time).exp();
                        assert!(envelope.is_finite(), "Laplace exponential overflow");
                        let (sine, cosine) = (omega_value * time).sin_cos();
                        compensated_add(
                            weight * correlation_value * envelope * cosine,
                            &mut real,
                            &mut real_compensation,
                        );
                        compensated_add(
                            weight * correlation_value * envelope * sine,
                            &mut imaginary,
                            &mut imaginary_compensation,
                        );
                    }
                    let value = num_complex::Complex64::new(real, imaginary);
                    assert!(
                        value.re.is_finite() && value.im.is_finite(),
                        "Laplace result is non-finite"
                    );
                    value
                })
                .collect()
        });
        LaplaceGrid {
            r: r.to_vec(),
            omega: omega.to_vec(),
            values,
            shape: [omega.len(), r.len()],
        }
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

fn compensated_add(value: f64, sum: &mut f64, compensation: &mut f64) {
    let corrected = value - *compensation;
    let updated = *sum + corrected;
    *compensation = (updated - *sum) - corrected;
    *sum = updated;
}
