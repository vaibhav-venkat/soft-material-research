use std::f64::consts::PI;

/// Wrap a scalar displacement into the centered minimum-image interval.
pub fn minimum_image(delta: f64, period: f64) -> f64 {
    delta - period * (delta / period).round()
}

/// Evaluate a normalized two-dimensional Gaussian kernel.
pub fn gaussian_2d(dx: f64, dy: f64, sigma: f64) -> f64 {
    let norm = 1.0 / (2.0 * PI * sigma * sigma);
    norm * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
}

/// Evaluate a normalized three-dimensional Gaussian kernel.
pub fn gaussian_3d(dx: f64, dy: f64, dz: f64, sigma: f64) -> f64 {
    let norm = 1.0 / ((2.0 * PI).powf(1.5) * sigma * sigma * sigma);
    norm * (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp()
}
