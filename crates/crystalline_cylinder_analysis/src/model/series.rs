//! Time-domain COM and lag-correlation series.

use serde::{Deserialize, Serialize};

/// Aggregated unwrapped axial center-of-mass data.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ComSeries {
    pub elapsed_time: Vec<f64>,
    pub x_center_mean: Vec<f64>,
    pub x_center_std: Vec<f64>,
    pub x_velocity_mean: Vec<f64>,
    pub x_velocity_std: Vec<f64>,
    pub replicate_count: usize,
}

/// Aggregated lagged Pearson correlation for axial COM velocity.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CorrelationSeries {
    pub lag_indices: Vec<usize>,
    pub lag_times: Vec<f64>,
    pub pearson_mean: Vec<f64>,
    pub pearson_std: Vec<f64>,
    pub origin_counts: Vec<usize>,
    pub replicate_count: usize,
}
