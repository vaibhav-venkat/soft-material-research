//! Complex Laplace transform and preferred-coordinate searches.

use crate::backend::AnalysisBackend;
use crate::integration::simpson_weights;
use crate::model::{CorrelationSeries, LaplaceGrid, PreferredAxis, PreferredEstimate};
use rayon::prelude::*;

/// Transform-grid controls.
#[derive(Clone, Copy, Debug)]
pub struct LaplaceConfig {
    pub r_min: Option<f64>,
    pub r_max: f64,
    pub r_points: usize,
    pub omega_min: Option<f64>,
    pub omega_max: Option<f64>,
    pub omega_points: usize,
}

/// Construct validated real and imaginary transform axes.
pub fn transform_axes(
    correlations: &[CorrelationSeries],
    config: LaplaceConfig,
) -> (Vec<f64>, Vec<f64>) {
    assert!(!correlations.is_empty(), "no correlations");
    assert!(config.r_points >= 2, "r grid needs two points");
    assert!(config.omega_points >= 2, "omega grid needs two points");
    let mut duration = f64::INFINITY;
    let mut nyquist = f64::INFINITY;
    for correlation in correlations {
        validate_correlation(correlation);
        duration = duration.min(*correlation.lag_times.last().expect("correlation is empty"));
        let spacing = correlation.lag_times[1] - correlation.lag_times[0];
        nyquist = nyquist.min(std::f64::consts::PI / spacing);
    }
    assert!(
        duration.is_finite() && duration > 0.0,
        "correlation duration must be positive"
    );

    let r_min = config.r_min.unwrap_or(-10.0 / duration);
    let r_max = config.r_max;
    assert!(
        r_min.is_finite() && r_max.is_finite(),
        "r bounds are non-finite"
    );
    assert!(r_min < r_max, "r minimum must be below maximum");

    let default_omega_max = nyquist.min(20.0 * std::f64::consts::PI / duration);
    let omega_max = config.omega_max.unwrap_or(default_omega_max);
    let omega_min = config.omega_min.unwrap_or(-default_omega_max);
    assert!(
        omega_min.is_finite() && omega_max.is_finite(),
        "omega bounds are non-finite"
    );
    assert!(omega_min < omega_max, "omega minimum must be below maximum");
    assert!(omega_min >= -nyquist, "omega minimum exceeds Nyquist");
    assert!(omega_max <= nyquist, "omega maximum exceeds Nyquist");

    (
        linspace(r_min, r_max, config.r_points),
        linspace(omega_min, omega_max, config.omega_points),
    )
}

/// Evaluate the complete complex transform grid.
pub fn analyze_laplace<B: AnalysisBackend>(
    backend: &B,
    correlation: &CorrelationSeries,
    r: &[f64],
    omega: &[f64],
) -> LaplaceGrid {
    validate_correlation(correlation);
    assert!(r.len() >= 2, "r grid needs two points");
    assert!(omega.len() >= 2, "omega grid needs two points");
    assert!(
        r.iter().all(|value| value.is_finite()),
        "r grid is non-finite"
    );
    assert!(
        omega.iter().all(|value| value.is_finite()),
        "omega grid is non-finite"
    );
    backend.laplace_grid(correlation, r, omega)
}

/// Locate the maximum log magnitude on the selected transform axis.
pub fn preferred_coordinate(
    correlation: &CorrelationSeries,
    axis: PreferredAxis,
    coordinates: &[f64],
) -> PreferredEstimate {
    validate_correlation(correlation);
    assert!(coordinates.len() >= 2, "preferred grid needs two points");
    assert!(
        coordinates.windows(2).all(|pair| pair[1] > pair[0]),
        "preferred grid is not increasing"
    );
    match axis {
        PreferredAxis::Omega => assert!(
            coordinates
                .iter()
                .all(|value| value.is_finite() && *value > 0.0),
            "preferred omega grid must be positive"
        ),
        PreferredAxis::R => assert!(
            coordinates
                .iter()
                .all(|value| value.is_finite() && *value < 0.0),
            "preferred r grid must be negative"
        ),
    }

    let spacing = correlation.lag_times[1] - correlation.lag_times[0];
    let weights = simpson_weights(correlation.lag_times.len(), spacing);
    let scores = coordinates
        .par_iter()
        .map(|&coordinate| {
            let (r, omega) = match axis {
                PreferredAxis::Omega => (0.0, coordinate),
                PreferredAxis::R => (coordinate, 0.0),
            };
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
                let envelope = (r * time).exp();
                let (sine, cosine) = (omega * time).sin_cos();
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
            let magnitude = real.hypot(imaginary);
            assert!(magnitude.is_finite(), "preferred magnitude is non-finite");
            magnitude.max(f64::MIN_POSITIVE).log10()
        })
        .collect::<Vec<_>>();
    let peak_index = scores
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .expect("preferred grid is empty");
    PreferredEstimate {
        axis,
        coordinate: coordinates[peak_index],
        coordinate_std: 0.0,
        log10_magnitude: scores[peak_index],
        at_lower_boundary: axis == PreferredAxis::R && peak_index == 0,
        at_upper_boundary: peak_index + 1 == coordinates.len(),
        replicate_count: 1,
    }
}

fn validate_correlation(correlation: &CorrelationSeries) {
    let count = correlation.lag_times.len();
    assert!(count >= 2, "Laplace transform needs two lags");
    assert_eq!(
        correlation.pearson_mean.len(),
        count,
        "correlation lengths differ"
    );
    assert!(
        correlation.lag_times.iter().all(|value| value.is_finite()),
        "lag time is non-finite"
    );
    assert!(
        correlation
            .pearson_mean
            .iter()
            .all(|value| value.is_finite()),
        "Pearson input is non-finite"
    );
    let spacing = correlation.lag_times[1] - correlation.lag_times[0];
    assert!(
        spacing.is_finite() && spacing > 0.0,
        "lag spacing must be positive"
    );
    for pair in correlation.lag_times.windows(2) {
        let actual = pair[1] - pair[0];
        let tolerance = 1.0e-12_f64.max(1.0e-10 * spacing.abs().max(actual.abs()));
        assert!(
            (actual - spacing).abs() <= tolerance,
            "Laplace transform requires uniform lags"
        );
    }
}

fn linspace(minimum: f64, maximum: f64, count: usize) -> Vec<f64> {
    let denominator = (count - 1) as f64;
    (0..count)
        .map(|index| minimum + (maximum - minimum) * index as f64 / denominator)
        .collect()
}

fn compensated_add(value: f64, sum: &mut f64, compensation: &mut f64) {
    let corrected = value - *compensation;
    let updated = *sum + corrected;
    *compensation = (updated - *sum) - corrected;
    *sum = updated;
}
