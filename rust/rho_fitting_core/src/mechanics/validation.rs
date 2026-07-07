use ndarray::{ArrayView1, ArrayView2, ArrayView3};

use crate::{CoreError, CoreResult};

pub(super) struct ParticleFieldInputs<'a> {
    /// Borrowed particle arrays and scalar controls validated before mechanical gridding.
    pub coords: ArrayView3<'a, f64>,
    pub directions: ArrayView3<'a, f64>,
    pub velocities: ArrayView3<'a, f64>,
    pub psi6_abs: ArrayView2<'a, f64>,
    pub mask: ArrayView2<'a, bool>,
    pub x_centers: ArrayView1<'a, f64>,
    pub theta_centers: ArrayView1<'a, f64>,
    pub r_centers: ArrayView1<'a, f64>,
    pub lx: f64,
    pub theta_period: f64,
    pub sigma: f64,
    pub gamma: f64,
    pub u0: f64,
}

pub(super) fn validate_grid<D: ndarray::Dimension>(
    rho: ndarray::ArrayView<'_, f64, D>,
    lx: f64,
    theta_period: f64,
) -> CoreResult<()> {
    // Validate non-empty periodic grid axes and positive domain lengths.
    if rho.ndim() < 3 || rho.shape().iter().any(|size| *size == 0) {
        return Err(CoreError::InvalidInput(
            "rho grid axes must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite() && lx > 0.0 && theta_period.is_finite() && theta_period > 0.0) {
        return Err(CoreError::InvalidInput(
            "periodic domain lengths must be positive".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn validate_particle_fields(inputs: ParticleFieldInputs<'_>) -> CoreResult<()> {
    // Validate particle field shapes and finite scalar controls before coarse-graining.
    let ParticleFieldInputs {
        coords,
        directions,
        velocities,
        psi6_abs,
        mask,
        x_centers,
        theta_centers,
        r_centers,
        lx,
        theta_period,
        sigma,
        gamma,
        u0,
    } = inputs;
    let (frames, particles, coord_components) = coords.dim();
    if coord_components != 3 {
        return Err(CoreError::Shape(
            "coords must have shape (T,N,3)".to_string(),
        ));
    }
    if directions.dim() != (frames, particles, 3) || velocities.dim() != (frames, particles, 3) {
        return Err(CoreError::Shape(
            "directions and velocities must have shape (T,N,3)".to_string(),
        ));
    }
    if psi6_abs.dim() != (frames, particles) {
        return Err(CoreError::Shape(
            "psi6_abs must have shape (T,N)".to_string(),
        ));
    }
    if mask.dim() != (frames, particles) {
        return Err(CoreError::Shape("mask must have shape (T,N)".to_string()));
    }
    if x_centers.is_empty() || theta_centers.is_empty() || r_centers.is_empty() {
        return Err(CoreError::InvalidInput(
            "grid centers must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite()
        && lx > 0.0
        && theta_period.is_finite()
        && theta_period > 0.0
        && sigma.is_finite()
        && sigma > 0.0)
    {
        return Err(CoreError::InvalidInput(
            "geometry values must be positive".to_string(),
        ));
    }
    if !r_centers
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(CoreError::InvalidInput(
            "radial centers must be finite and positive".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }
    Ok(())
}
