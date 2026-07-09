/// Standard normal CDF Φ(x). Abramowitz & Stegun 7.1.26 (error ≤ 1.5e-7).
pub(super) fn normal_cdf(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let sign = (x > 0.0) as i32 as f32 - (x < 0.0) as i32 as f32;
    let x_abs = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x_abs);
    let poly = ((((1.061405429_f32).mul_add(t, -1.453152027)
        .mul_add(t, 1.421413741))
        .mul_add(t, -0.284496736))
        .mul_add(t, 0.254829592))
        * t;
    let erf = 1.0 - poly * (-x_abs * x_abs).exp();
    0.5 * (1.0 + sign * erf)
}

/// Z_r(r_i) = ∫_{r_min}^{r_max} exp(-(r-r_i)²/(2σ²)) · r dr    (unnormalized Gaussian, cylindrical measure)
///
/// Analytical result:
///   Z_r = r_i·σ·√(2π)·[Φ(t₂)-Φ(t₁)] - σ²·[exp(-t₂²/2) - exp(-t₁²/2)]
/// where t₁ = (r_min - r_i)/σ, t₂ = (r_max - r_i)/σ
pub(super) fn zr_integral(r_i: f32, r_min: f32, r_max: f32, sigma: f32) -> f32 {
    let t_min = (r_min - r_i) / sigma;
    let t_max = (r_max - r_i) / sigma;
    let cdf_diff = normal_cdf(t_max) - normal_cdf(t_min);
    let exp_diff = (-0.5 * t_max * t_max).exp() - (-0.5 * t_min * t_min).exp();
    r_i * sigma * (2.0 * std::f32::consts::PI).sqrt() * cdf_diff - sigma * sigma * exp_diff
}
