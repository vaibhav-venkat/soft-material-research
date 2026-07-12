use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

use crate::{CoreError, CoreResult};

const COMPONENTS: usize = 3;

/// Named particle inputs for direction conversion.
pub(crate) struct DirectionInputs<'a> {
    pub(crate) coords: ArrayView3<'a, f64>,
    pub(crate) direction_cylindrical: Option<ArrayView3<'a, f64>>,
    pub(crate) active_direction: Option<ArrayView3<'a, f64>>,
    pub(crate) orientation: Option<ArrayView3<'a, f64>>,
}

/// Named particle inputs for periodic surface velocity conversion.
pub(crate) struct VelocityInputs<'a> {
    pub(crate) coords: ArrayView3<'a, f64>,
    pub(crate) steps: ArrayView1<'a, i64>,
    pub(crate) timestep: f64,
    pub(crate) lx: f64,
    pub(crate) theta_period: f64,
}

/// Rotate HOOMD `(w, x, y, z)` quaternions onto the body-frame x axis.
pub(crate) fn active_direction_from_quaternion(
    orientation: ArrayView3<'_, f64>,
) -> CoreResult<Array3<f64>> {
    if orientation.shape().last() != Some(&4) {
        return Err(CoreError::Shape(
            "orientation must have shape (T,N,4)".to_string(),
        ));
    }
    let (frames, particles, _) = orientation.dim();
    let mut directions = Array3::<f64>::zeros((frames, particles, COMPONENTS));
    for frame in 0..frames {
        for particle in 0..particles {
            let w = orientation[[frame, particle, 0]];
            let x = orientation[[frame, particle, 1]];
            let y = orientation[[frame, particle, 2]];
            let z = orientation[[frame, particle, 3]];
            let norm = (w * w + x * x + y * y + z * z).sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(CoreError::InvalidInput(
                    "orientation contains a zero-norm or non-finite quaternion".to_string(),
                ));
            }
            let (w, x, y, z) = (w / norm, x / norm, y / norm, z / norm);
            directions[[frame, particle, 0]] = 1.0 - 2.0 * (y * y + z * z);
            directions[[frame, particle, 1]] = 2.0 * (x * y + w * z);
            directions[[frame, particle, 2]] = 2.0 * (x * z - w * y);
        }
    }
    Ok(directions)
}

/// Convert Cartesian vectors to `(x, radial, azimuthal)` components at each angle.
pub(crate) fn cartesian_to_cylindrical_components(
    vectors: ArrayView3<'_, f64>,
    theta: ArrayView2<'_, f64>,
) -> CoreResult<Array3<f64>> {
    let (frames, particles, components) = vectors.dim();
    if theta.dim() != (frames, particles) || components != COMPONENTS {
        return Err(CoreError::Shape(
            "vectors and theta must have matching (T,N,3) shapes".to_string(),
        ));
    }
    let mut out = Array3::<f64>::zeros((frames, particles, COMPONENTS));
    for frame in 0..frames {
        for particle in 0..particles {
            let angle = theta[[frame, particle]];
            let sin = angle.sin();
            let cos = angle.cos();
            let vx = vectors[[frame, particle, 0]];
            let vy = vectors[[frame, particle, 1]];
            let vz = vectors[[frame, particle, 2]];
            out[[frame, particle, 0]] = vx;
            out[[frame, particle, 1]] = vy * sin + vz * cos;
            out[[frame, particle, 2]] = vy * cos - vz * sin;
        }
    }
    Ok(out)
}

/// Build particle directions in the canonical `(x, e_theta, e_r)` moment order.
pub(crate) fn tangential_particle_vectors(inputs: DirectionInputs<'_>) -> CoreResult<Array3<f64>> {
    let DirectionInputs {
        coords,
        direction_cylindrical,
        active_direction,
        orientation,
    } = inputs;
    if coords.shape().last() != Some(&COMPONENTS) {
        return Err(CoreError::Shape(
            "coords must have shape (T,N,3)".to_string(),
        ));
    }
    let (frames, particles, _) = coords.dim();
    let cylindrical = if let Some(values) = direction_cylindrical {
        require_particle_shape(values, frames, particles, "direction_cylindrical")?;
        values.to_owned()
    } else {
        let cartesian = if let Some(values) = active_direction {
            require_particle_shape(values, frames, particles, "active_direction")?;
            values.to_owned()
        } else if let Some(values) = orientation {
            require_orientation_shape(values, frames, particles)?;
            active_direction_from_quaternion(values)?
        } else {
            return Err(CoreError::InvalidInput(
                "missing particle orientation source".to_string(),
            ));
        };
        let mut angles = Array2::<f64>::zeros((frames, particles));
        for frame in 0..frames {
            for particle in 0..particles {
                angles[[frame, particle]] = coords[[frame, particle, 1]];
            }
        }
        cartesian_to_cylindrical_components(cartesian.view(), angles.view())?
    };

    let mut out = Array3::<f64>::zeros((frames, particles, COMPONENTS));
    for frame in 0..frames {
        for particle in 0..particles {
            out[[frame, particle, 0]] = cylindrical[[frame, particle, 0]];
            out[[frame, particle, 1]] = cylindrical[[frame, particle, 2]];
            out[[frame, particle, 2]] = cylindrical[[frame, particle, 1]];
        }
    }
    Ok(out)
}

/// Estimate periodic particle velocities in canonical `(x, e_theta, e_r)` order.
pub(crate) fn surface_velocities(inputs: VelocityInputs<'_>) -> CoreResult<Array3<f64>> {
    let VelocityInputs {
        coords,
        steps,
        timestep,
        lx,
        theta_period,
    } = inputs;
    if coords.shape().last() != Some(&COMPONENTS) {
        return Err(CoreError::Shape(
            "coords must have shape (T,N,3)".to_string(),
        ));
    }
    let (frames, particles, _) = coords.dim();
    if steps.len() != frames || frames < 2 {
        return Err(CoreError::InvalidInput(
            "surface velocities require at least two frames and one step per frame".to_string(),
        ));
    }
    if !(timestep.is_finite()
        && timestep > 0.0
        && lx.is_finite()
        && lx > 0.0
        && theta_period.is_finite()
        && theta_period > 0.0)
    {
        return Err(CoreError::InvalidInput(
            "velocity timestep and periods must be finite and positive".to_string(),
        ));
    }
    let times = steps
        .iter()
        .map(|step| (*step - steps[0]) as f64 * timestep)
        .collect::<Vec<_>>();
    let mut out = Array3::<f64>::zeros((frames, particles, COMPONENTS));
    for frame in 0..frames {
        let left = frame.saturating_sub(1);
        let right = (frame + 1).min(frames - 1);
        let dt = times[right] - times[left];
        if !(dt.is_finite() && dt > 0.0) {
            return Err(CoreError::InvalidInput(
                "steps must increase over time".to_string(),
            ));
        }
        for particle in 0..particles {
            let dx = minimum_image(
                coords[[right, particle, 0]] - coords[[left, particle, 0]],
                lx,
            );
            let dtheta = minimum_image(
                coords[[right, particle, 1]] - coords[[left, particle, 1]],
                theta_period,
            );
            let local_radius = 0.5 * (coords[[right, particle, 2]] + coords[[left, particle, 2]]);
            out[[frame, particle, 0]] = dx / dt;
            out[[frame, particle, 1]] = local_radius * dtheta / dt;
            out[[frame, particle, 2]] =
                (coords[[right, particle, 2]] - coords[[left, particle, 2]]) / dt;
        }
    }
    Ok(out)
}

fn minimum_image(delta: f64, period: f64) -> f64 {
    delta - period * (delta / period).round()
}

fn require_particle_shape(
    values: ArrayView3<'_, f64>,
    frames: usize,
    particles: usize,
    name: &str,
) -> CoreResult<()> {
    if values.dim() != (frames, particles, COMPONENTS) {
        return Err(CoreError::Shape(format!("{name} must have shape (T,N,3)")));
    }
    Ok(())
}

fn require_orientation_shape(
    values: ArrayView3<'_, f64>,
    frames: usize,
    particles: usize,
) -> CoreResult<()> {
    if values.dim() != (frames, particles, 4) {
        return Err(CoreError::Shape(
            "orientation must have shape (T,N,4)".to_string(),
        ));
    }
    Ok(())
}
