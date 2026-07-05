use ndarray::{Array3, Array4, ArrayView1, ArrayView2, ArrayView3};

use crate::geometry::{gaussian_2d, minimum_image};
use crate::{CoreError, CoreResult};

/// Coarse-grain scalar density and two-component polarization density.
///
/// `coords` is `(T, N, 3)` with angular coordinate in column 1, `p_particles`
/// is `(T, N, 2)`, and `shell_mask` selects particles to include. Returns
/// `rho` shaped `(T, Nx, Ny)` and `p_density` shaped `(T, Nx, Ny, 2)`.
///
/// Edge cases: non-finite particle positions or directions are skipped, and
/// the Gaussian kernel is truncated at `4*sigma`.
pub fn coarse_grain_fields(
    coords: ArrayView3<'_, f64>,
    p_particles: ArrayView3<'_, f64>,
    shell_mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
) -> CoreResult<(Array3<f64>, Array4<f64>)> {
    validate_inputs(
        coords,
        p_particles,
        shell_mask,
        x_centers,
        y_centers,
        lx,
        ly,
        radius,
        sigma,
    )?;

    let (frames, particles, _) = coords.dim();
    let nx = x_centers.len();
    let ny = y_centers.len();
    let mut rho = Array3::<f64>::zeros((frames, nx, ny));
    let mut p_density = Array4::<f64>::zeros((frames, nx, ny, 2));
    let cutoff = 4.0 * sigma;
    let cutoff2 = cutoff * cutoff;

    for t in 0..frames {
        println!("[rho_fitting] coarse-grain frame {}/{}", t + 1, frames);
        for i in 0..particles {
            if !shell_mask[[t, i]] {
                continue;
            }
            let px = p_particles[[t, i, 0]];
            let py = p_particles[[t, i, 1]];
            let particle_x = coords[[t, i, 0]];
            let particle_y = radius * coords[[t, i, 1]];
            if !(particle_x.is_finite()
                && particle_y.is_finite()
                && px.is_finite()
                && py.is_finite())
            {
                continue;
            }

            for ix in 0..nx {
                let dx = minimum_image(x_centers[ix] - particle_x, lx);
                if dx.abs() > cutoff {
                    continue;
                }
                for iy in 0..ny {
                    let dy = minimum_image(y_centers[iy] - particle_y, ly);
                    if dy.abs() > cutoff || dx * dx + dy * dy > cutoff2 {
                        continue;
                    }
                    let weight = gaussian_2d(dx, dy, sigma);
                    rho[[t, ix, iy]] += weight;
                    p_density[[t, ix, iy, 0]] += weight * px;
                    p_density[[t, ix, iy, 1]] += weight * py;
                }
            }
        }
    }

    Ok((rho, p_density))
}

/// Validate shapes and scalar geometry controls for legacy coarse-graining.
pub fn validate_inputs(
    coords: ArrayView3<'_, f64>,
    p_particles: ArrayView3<'_, f64>,
    shell_mask: ArrayView2<'_, bool>,
    x_centers: ArrayView1<'_, f64>,
    y_centers: ArrayView1<'_, f64>,
    lx: f64,
    ly: f64,
    radius: f64,
    sigma: f64,
) -> CoreResult<()> {
    let (frames, particles, coord_components) = coords.dim();
    if coord_components != 3 {
        return Err(CoreError::Shape(
            "coords must have shape (T, N, 3)".to_string(),
        ));
    }
    if p_particles.dim() != (frames, particles, 2) {
        return Err(CoreError::Shape(
            "p_particles must have shape (T, N, 2)".to_string(),
        ));
    }
    if shell_mask.dim() != (frames, particles) {
        return Err(CoreError::Shape(
            "shell_mask must have shape (T, N)".to_string(),
        ));
    }
    if x_centers.is_empty() || y_centers.is_empty() {
        return Err(CoreError::InvalidInput(
            "grid centers must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite() && lx > 0.0 && ly.is_finite() && ly > 0.0) {
        return Err(CoreError::InvalidInput(
            "surface lengths must be positive".to_string(),
        ));
    }
    if !(radius.is_finite() && radius > 0.0 && sigma.is_finite() && sigma > 0.0) {
        return Err(CoreError::InvalidInput(
            "radius and sigma must be positive".to_string(),
        ));
    }
    Ok(())
}
