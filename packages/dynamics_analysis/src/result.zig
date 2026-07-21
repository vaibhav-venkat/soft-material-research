//! Owned analysis result declarations.

pub const ComSeries = struct {
    elapsed_time: []f64,
    center: []f64,
    velocity: []f64,
};

pub const CorrelationSeries = struct {
    lag_indices: []usize,
    lag_times: []f64,
    pearson: []f64,
    origin_counts: []usize,
};

pub const Result = struct {
    com: ComSeries,
    correlation: CorrelationSeries,
};
