//! Pointwise aggregation of compatible seed replicas.

use crate::model::{ComSeries, CorrelationSeries, PreferredEstimate};

/// Average compatible COM series on their common elapsed-time prefix.
pub fn average_com_series(replicas: &[ComSeries]) -> ComSeries {
    assert!(!replicas.is_empty(), "no COM replicas");
    for replica in replicas {
        let count = replica.elapsed_time.len();
        let aligned = replica.x_center_mean.len() == count
            && replica.x_center_std.len() == count
            && replica.x_velocity_mean.len() == count
            && replica.x_velocity_std.len() == count;
        assert!(aligned, "COM replica arrays are misaligned");
        let finite = replica.elapsed_time.iter().all(|value| value.is_finite())
            && replica.x_center_mean.iter().all(|value| value.is_finite())
            && replica
                .x_velocity_mean
                .iter()
                .all(|value| value.is_finite());
        assert!(finite, "COM replica contains non-finite values");
    }
    let common_length = replicas
        .iter()
        .map(|series| series.elapsed_time.len())
        .min()
        .unwrap_or(0);
    assert!(common_length >= 2, "COM replicas need two frames");
    let elapsed_time = replicas[0].elapsed_time[..common_length].to_vec();
    for replica in &replicas[1..] {
        for (index, (&expected, &actual)) in elapsed_time
            .iter()
            .zip(&replica.elapsed_time[..common_length])
            .enumerate()
        {
            let tolerance = 1.0e-12 * expected.abs().max(actual.abs()).max(1.0);
            let _ = index;
            assert!(
                (expected - actual).abs() <= tolerance,
                "COM replica times differ"
            );
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
pub fn average_correlations(replicas: &[CorrelationSeries]) -> CorrelationSeries {
    assert!(!replicas.is_empty(), "no correlation replicas");
    let common_length = replicas
        .iter()
        .map(|series| series.lag_times.len())
        .min()
        .unwrap_or(0);
    assert!(common_length >= 1, "correlation replicas are empty");
    let lag_indices = replicas[0].lag_indices[..common_length].to_vec();
    let lag_times = replicas[0].lag_times[..common_length].to_vec();
    for replica in replicas {
        assert!(
            replica.lag_indices.len() >= common_length
                && replica.pearson_mean.len() >= common_length
                && replica.origin_counts.len() >= common_length,
            "correlation arrays are misaligned"
        );
        assert_eq!(
            &replica.lag_indices[..common_length],
            lag_indices.as_slice(),
            "replicate lag indices differ"
        );
        for (&expected, &actual) in lag_times.iter().zip(&replica.lag_times[..common_length]) {
            let tolerance = 1.0e-12_f64.max(1.0e-10 * expected.abs().max(actual.abs()));
            assert!(
                (expected - actual).abs() <= tolerance,
                "replicate lag times differ"
            );
        }
    }
    let (pearson_mean, pearson_std) = pointwise_mean_std(
        replicas
            .iter()
            .map(|series| &series.pearson_mean[..common_length]),
        common_length,
    );
    let origin_counts = (0..common_length)
        .map(|index| {
            replicas
                .iter()
                .map(|series| series.origin_counts[index])
                .sum()
        })
        .collect();
    CorrelationSeries {
        lag_indices,
        lag_times,
        pearson_mean,
        pearson_std,
        origin_counts,
        replicate_count: replicas.iter().map(|series| series.replicate_count).sum(),
    }
}

/// Average compatible seed preferred-coordinate estimates with sample deviation.
pub fn average_preferred_estimates(replicas: &[PreferredEstimate]) -> PreferredEstimate {
    assert!(!replicas.is_empty(), "no preferred estimates");
    let axis = replicas[0].axis;
    assert!(
        replicas.iter().all(|estimate| estimate.axis == axis),
        "preferred axes differ"
    );
    let count = replicas.len() as f64;
    let coordinate = replicas
        .iter()
        .map(|estimate| estimate.coordinate)
        .sum::<f64>()
        / count;
    let log10_magnitude = replicas
        .iter()
        .map(|estimate| estimate.log10_magnitude)
        .sum::<f64>()
        / count;
    let coordinate_std = if replicas.len() > 1 {
        (replicas
            .iter()
            .map(|estimate| (estimate.coordinate - coordinate).powi(2))
            .sum::<f64>()
            / (replicas.len() - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    PreferredEstimate {
        axis,
        coordinate,
        coordinate_std,
        log10_magnitude,
        at_lower_boundary: replicas.iter().any(|estimate| estimate.at_lower_boundary),
        at_upper_boundary: replicas.iter().any(|estimate| estimate.at_upper_boundary),
        replicate_count: replicas
            .iter()
            .map(|estimate| estimate.replicate_count)
            .sum(),
    }
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
