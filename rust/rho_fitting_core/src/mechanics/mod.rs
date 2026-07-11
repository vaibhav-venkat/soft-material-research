mod domain;
mod sampling;

use ndarray::{ArrayView1, ArrayView2, ArrayView3};

use crate::geometry::{gaussian_3d, minimum_image};
use crate::CoreResult;

pub(crate) use domain::{
    build_targets, relative_mass_error, CurrentQField, CylindricalGrid, MechanicalFieldSet,
    MechanicalFrame, MechanicalInputViews, PhysicalComponent, TENSOR_COMPONENTS,
};
pub use sampling::sample_component_rows;

/// CPU reference/fallback implementation over the shared mechanical domain model.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_mechanical_fields(
    coords: ArrayView3<'_, f64>,
    directions: ArrayView3<'_, f64>,
    velocities: ArrayView3<'_, f64>,
    psi6_abs: ArrayView2<'_, f64>,
    mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    theta_centers: ArrayView1<'_, f64>,
    r_centers: ArrayView1<'_, f64>,
    lx: f64,
    theta_period: f64,
    sigma: f64,
    gamma: f64,
    u0: f64,
) -> CoreResult<MechanicalFieldSet> {
    let inputs = MechanicalInputViews::new(
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
    )?;
    let (frames, particles, _) = inputs.coords.dim();
    let mut fields = MechanicalFieldSet::zeros(frames, &inputs.grid);
    let cutoff = 4.0 * inputs.sigma;
    let cutoff2 = cutoff * cutoff;

    for t in 0..frames {
        println!(
            "[rho_fitting] CPU reference mechanical coarse-grain frame {}/{}",
            t + 1,
            frames
        );
        let mut frame = MechanicalFrame::new(inputs.grid.len());
        for particle in 0..particles {
            if !inputs.mask[[t, particle]] {
                continue;
            }
            let position = [
                inputs.coords[[t, particle, 0]],
                inputs.coords[[t, particle, 1]],
                inputs.coords[[t, particle, 2]],
            ];
            let direction = [
                inputs.directions[[t, particle, 0]],
                inputs.directions[[t, particle, 1]],
                inputs.directions[[t, particle, 2]],
            ];
            let velocity = [
                inputs.velocities[[t, particle, 0]],
                inputs.velocities[[t, particle, 1]],
                inputs.velocities[[t, particle, 2]],
            ];
            let psi6 = inputs.psi6_abs[[t, particle]];
            if !position
                .iter()
                .chain(direction.iter())
                .chain(velocity.iter())
                .all(|value| value.is_finite())
                || !psi6.is_finite()
            {
                continue;
            }
            for ix in 0..inputs.grid.x.len() {
                let dx = minimum_image(inputs.grid.x[ix] - position[0], inputs.grid.lx);
                if dx.abs() > cutoff {
                    continue;
                }
                for itheta in 0..inputs.grid.theta.len() {
                    let dtheta = minimum_image(
                        inputs.grid.theta[itheta] - position[1],
                        inputs.grid.theta_period,
                    );
                    for ir in 0..inputs.grid.r.len() {
                        let r = inputs.grid.r[ir];
                        let dy = r * dtheta;
                        let dr = r - position[2];
                        if dy.abs() > cutoff
                            || dr.abs() > cutoff
                            || dx * dx + dy * dy + dr * dr > cutoff2
                        {
                            continue;
                        }
                        let kernel = gaussian_3d(dx, dy, dr, inputs.sigma);
                        let flat = inputs.grid.flat_index(ix, itheta, ir);
                        frame.deposit(
                            flat,
                            kernel * inputs.grid.cell_volumes[flat],
                            psi6,
                            direction,
                            velocity,
                        );
                    }
                }
            }
        }
        frame.write_into(t, &inputs.grid, &mut fields);
    }
    Ok(fields)
}
