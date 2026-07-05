use ndarray::{ArrayView1, ArrayView2, ArrayView3};

use crate::{CoreError, CoreResult};

pub(super) struct ParticleFieldInputs<'a> {
    pub coords: ArrayView3<'a, f64>,
    pub directions: ArrayView3<'a, f64>,
    pub velocities: ArrayView3<'a, f64>,
    pub psi6_abs: ArrayView2<'a, f64>,
    pub mask: ArrayView2<'a, bool>,
    pub x_centers: ArrayView1<'a, f64>,
    pub y_centers: ArrayView1<'a, f64>,
    pub lx: f64,
    pub ly: f64,
    pub radius: f64,
    pub sigma: f64,
    pub gamma: f64,
    pub u0: f64,
}

pub(super) fn validate_grid(rho: ArrayView3<'_, f64>, lx: f64, ly: f64) -> CoreResult<()> {
    let (frames, nx, ny) = rho.dim();
    if frames == 0 || nx == 0 || ny == 0 {
        return Err(CoreError::InvalidInput(
            "rho grid axes must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite() && lx > 0.0 && ly.is_finite() && ly > 0.0) {
        return Err(CoreError::InvalidInput(
            "domain lengths must be positive".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn validate_particle_fields(inputs: ParticleFieldInputs<'_>) -> CoreResult<()> {
    let ParticleFieldInputs {
        coords,
        directions,
        velocities,
        psi6_abs,
        mask,
        x_centers,
        y_centers,
        lx,
        ly,
        radius,
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
    if directions.dim() != (frames, particles, 3) || velocities.dim() != (frames, particles, 2) {
        return Err(CoreError::Shape(
            "directions must have shape (T,N,3) and velocities must have shape (T,N,2)".to_string(),
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
    if x_centers.is_empty() || y_centers.is_empty() {
        return Err(CoreError::InvalidInput(
            "grid centers must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite()
        && lx > 0.0
        && ly.is_finite()
        && ly > 0.0
        && radius.is_finite()
        && radius > 0.0
        && sigma.is_finite()
        && sigma > 0.0)
    {
        return Err(CoreError::InvalidInput(
            "geometry values must be positive".to_string(),
        ));
    }
    if !(gamma.is_finite() && u0.is_finite() && u0 != 0.0) {
        return Err(CoreError::InvalidInput(
            "gamma must be finite and u0 must be nonzero".to_string(),
        ));
    }
    Ok(())
}
