//! Pointwise aggregation of compatible seed replicas.

use crate::model::{ComSeries, CorrelationSeries};

/// Average compatible COM series on their common elapsed-time prefix.
pub fn average_com_series(replicas: &[ComSeries]) -> ComSeries {
    assert!(!replicas.is_empty(), "no replicas");
    for replica in replicas {
        let count = replica.elapsed_time.len();
        let aligned = replica.x_center_mean.len() == count
            && replica.x_center_std.len() == count
            && replica.x_velocity_mean.len() == count
            && replica.x_velocity_std.len() == count;
        let finite = replica.elapsed_time.iter().all(|value| value.is_finite())
            && replica.x_center_mean.iter().all(|value| value.is_finite())
            && replica
                .x_velocity_mean
                .iter()
                .all(|value| value.is_finite());
        assert!(aligned && finite, "bad replicas");
    }
    let common_length = replicas
        .iter()
        .map(|series| series.elapsed_time.len())
        .min()
        .unwrap_or(0);
    assert!(common_length >= 2, "short replicas");
    let elapsed_time = replicas[0].elapsed_time[..common_length].to_vec();
    for replica in &replicas[1..] {
        for (index, (&expected, &actual)) in elapsed_time
            .iter()
            .zip(&replica.elapsed_time[..common_length])
            .enumerate()
        {
            let tolerance = 1.0e-12 * expected.abs().max(actual.abs()).max(1.0);
            let _ = index;
            assert!((expected - actual).abs() <= tolerance, "bad time grid");
        }
    }
    let (x_center_mean, x_center_std) = pointwise_mean_std(
        replicas
            .iter()
            .map(|series| &series.x_center_mean[..common_length]),
        common_length,
    );
    let (x_velocity_mean, x_velocity_std) = pointwise_mean_std(
        replicas
            .iter()
            .map(|series| &series.x_velocity_mean[..common_length]),
        common_length,
    );
    ComSeries {
        elapsed_time,
        x_center_mean,
        x_center_std,
        x_velocity_mean,
        x_velocity_std,
        replicate_count: replicas.len(),
    }
}

/// Average compatible correlations and sum their time-origin counts.
pub fn average_correlations(_replicas: &[CorrelationSeries]) -> CorrelationSeries {
    todo!("validate lag grids and aggregate seed correlations")
}

fn pointwise_mean_std<'a>(
    series: impl Iterator<Item = &'a [f64]>,
    length: usize,
) -> (Vec<f64>, Vec<f64>) {
    let series: Vec<&[f64]> = series.collect();
    let count = series.len() as f64;
    let mut mean = vec![0.0; length];
    for values in &series {
        for (output, &value) in mean.iter_mut().zip(*values) {
            *output += value / count;
        }
    }
    let mut std = vec![0.0; length];
    if series.len() > 1 {
        for values in &series {
            for ((output, &average), &value) in std.iter_mut().zip(&mean).zip(*values) {
                *output += (value - average).powi(2);
            }
        }
        let denominator = (series.len() - 1) as f64;
        for value in &mut std {
            *value = (*value / denominator).sqrt();
        }
    }
    (mean, std)
}
