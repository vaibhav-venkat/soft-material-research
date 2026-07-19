//! Lagged axial-velocity Pearson correlation.

use crate::backend::AnalysisBackend;
use crate::model::{ComSeries, CorrelationSeries};

/// Controls the requested maximum lag index.
#[derive(Clone, Copy, Debug)]
pub struct CorrelationConfig {
    pub max_lag: Option<usize>,
}

/// Compute one pairwise Pearson coefficient using stable accumulation.
pub fn pearson(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len(), "Pearson lengths differ");
    assert!(left.len() >= 2, "Pearson needs two pairs");
    assert!(
        left.iter().all(|value| value.is_finite()),
        "left Pearson input is non-finite"
    );
    assert!(
        right.iter().all(|value| value.is_finite()),
        "right Pearson input is non-finite"
    );

    let count = left.len() as f64;
    let left_mean = compensated_sum(left.iter().copied()) / count;
    let right_mean = compensated_sum(right.iter().copied()) / count;
    let covariance = compensated_sum(
        left.iter()
            .zip(right)
            .map(|(&x, &y)| (x - left_mean) * (y - right_mean)),
    );
    let left_variance = compensated_sum(left.iter().map(|&x| (x - left_mean).powi(2)));
    let right_variance = compensated_sum(right.iter().map(|&y| (y - right_mean).powi(2)));
    assert!(
        left_variance.is_finite() && left_variance > 0.0,
        "left Pearson input is constant"
    );
    assert!(
        right_variance.is_finite() && right_variance > 0.0,
        "right Pearson input is constant"
    );
    let coefficient = covariance / (left_variance * right_variance).sqrt();
    assert!(coefficient.is_finite(), "Pearson result is non-finite");
    coefficient.clamp(-1.0, 1.0)
}

/// Compute the lagged velocity correlation for one COM series.
pub fn analyze_correlation<B: AnalysisBackend>(
    backend: &B,
    com: &ComSeries,
    config: CorrelationConfig,
) -> CorrelationSeries {
    let frame_count = com.elapsed_time.len();
    assert_eq!(
        com.x_velocity_mean.len(),
        frame_count,
        "velocity/time lengths differ"
    );
    assert!(frame_count >= 3, "correlation needs three frames");
    assert!(
        config.max_lag.is_none_or(|lag| lag > 0),
        "max lag must be positive"
    );
    assert!(
        com.elapsed_time.iter().all(|value| value.is_finite()),
        "elapsed time is non-finite"
    );
    assert!(
        com.x_velocity_mean.iter().all(|value| value.is_finite()),
        "COM velocity is non-finite"
    );

    let spacing = com.elapsed_time[1] - com.elapsed_time[0];
    assert!(
        spacing.is_finite() && spacing > 0.0,
        "time spacing must be positive"
    );
    for pair in com.elapsed_time.windows(2) {
        let actual = pair[1] - pair[0];
        let tolerance = 1.0e-12_f64.max(1.0e-10 * spacing.abs().max(actual.abs()));
        assert!(
            (actual - spacing).abs() <= tolerance,
            "correlation requires uniform time"
        );
    }

    // Every Pearson window must retain at least two paired samples.
    let available_maximum = frame_count - 2;
    let maximum_lag = config
        .max_lag
        .unwrap_or(available_maximum)
        .min(available_maximum);
    let pearson_mean = backend.lagged_pearson(&com.x_velocity_mean, maximum_lag);
    let lag_indices = (0..=maximum_lag).collect::<Vec<_>>();
    let lag_times = lag_indices
        .iter()
        .map(|&lag| lag as f64 * spacing)
        .collect::<Vec<_>>();
    let origin_counts = lag_indices
        .iter()
        .map(|&lag| frame_count - lag)
        .collect::<Vec<_>>();
    CorrelationSeries {
        lag_indices,
        lag_times,
        pearson_std: vec![0.0; pearson_mean.len()],
        pearson_mean,
        origin_counts,
        replicate_count: 1,
    }
}

fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut compensation = 0.0;
    for value in values {
        let corrected = value - compensation;
        let updated = sum + corrected;
        compensation = (updated - sum) - corrected;
        sum = updated;
    }
    sum
}
