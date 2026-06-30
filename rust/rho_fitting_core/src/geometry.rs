use std::f64::consts::PI;

pub fn minimum_image(delta: f64, period: f64) -> f64 {
    delta - period * (delta / period).round()
}

pub fn gaussian_2d(dx: f64, dy: f64, sigma: f64) -> f64 {
    let norm = 1.0 / (2.0 * PI * sigma * sigma);
    norm * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()
}
