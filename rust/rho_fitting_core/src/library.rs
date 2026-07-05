use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4};

use crate::fft_ops;
use crate::{CoreError, CoreResult};

pub const DENSITY_TERM_NAMES: [&str; 4] = [
    "neg_div_grad_rho",
    "neg_div_grad_lap_rho",
    "neg_div_lap_rho_grad_rho",
    "neg_div_grad_rho_cubed",
];

#[non_exhaustive]
pub struct DensityFluxes {
    /// Candidate density flux fields before divergence is applied for scalar fitting.
    pub grad_rho: Array4<f64>,
    pub grad_lap_rho: Array4<f64>,
    pub lap_rho_grad_rho: Array4<f64>,
    pub grad_rho_cubed: Array4<f64>,
}

/// Build the density candidate fluxes from a scalar rho field.
///
/// `rho` is shaped `(T, Nx, Ny)`, and `lx`/`ly` are the periodic surface
/// lengths used by spectral derivatives. The returned fluxes all use shape
/// `(T, Nx, Ny, 2)`.
pub fn build_density_fluxes(
    rho: ArrayView3<'_, f64>,
    lx: f64,
    ly: f64,
) -> CoreResult<DensityFluxes> {
    let grad_rho = fft_ops::gradient_scalar(rho, lx, ly)?;
    let lap_rho = fft_ops::laplacian_scalar(rho, lx, ly)?;
    let grad_lap_rho = fft_ops::gradient_scalar(lap_rho.view(), lx, ly)?;
    let lap_rho_grad_rho = scalar_times_vector(lap_rho.view(), grad_rho.view());
    let grad_rho_cubed = cubic_gradient_flux(grad_rho.view());
    Ok(DensityFluxes {
        grad_rho,
        grad_lap_rho,
        lap_rho_grad_rho,
        grad_rho_cubed,
    })
}

/// Sample named density-library divergence terms at selected grid rows.
///
/// `sample_indices` must be `(N, 3)` with `(frame, ix, iy)` rows. The
/// returned matrix is `(N, term_names.len())` in the same order as
/// `term_names`.
///
/// Edge cases: unknown term names are rejected, and sampled indices are
/// bounds-checked before any term matrix is returned.
pub fn build_density_library(
    rho: ArrayView3<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
    term_names: &[String],
    lx: f64,
    ly: f64,
) -> CoreResult<Array2<f64>> {
    validate_inputs(rho, sample_indices, term_names, lx, ly)?;

    let samples = sample_indices.dim().0;
    let mut out = Array2::<f64>::zeros((samples, term_names.len()));
    let fluxes = build_density_fluxes(rho, lx, ly)?;

    for (column, name) in term_names.iter().enumerate() {
        println!(
            "[rho_fitting] density-library term {}/{}: {}",
            column + 1,
            term_names.len(),
            name
        );
        let field = candidate_field(name, &fluxes, lx, ly)?;
        for row in 0..samples {
            let (t, ix, iy) = index_triplet(sample_indices, row)?;
            out[[row, column]] = field[[t, ix, iy]];
        }
    }

    Ok(out)
}

/// Return whether a density term name is available in this library.
pub fn known_density_term(name: &str) -> bool {
    DENSITY_TERM_NAMES.contains(&name)
}

fn candidate_field(
    name: &str,
    fluxes: &DensityFluxes,
    lx: f64,
    ly: f64,
) -> CoreResult<Array3<f64>> {
    let mut field = match name {
        "neg_div_grad_rho" => fft_ops::divergence_vector(fluxes.grad_rho.view(), lx, ly)?,
        "neg_div_grad_lap_rho" => fft_ops::divergence_vector(fluxes.grad_lap_rho.view(), lx, ly)?,
        "neg_div_lap_rho_grad_rho" => {
            fft_ops::divergence_vector(fluxes.lap_rho_grad_rho.view(), lx, ly)?
        }
        "neg_div_grad_rho_cubed" => {
            fft_ops::divergence_vector(fluxes.grad_rho_cubed.view(), lx, ly)?
        }
        _ => {
            return Err(CoreError::InvalidInput(format!(
                "unknown density term {name}"
            )))
        }
    };
    field.mapv_inplace(|value| -value);
    Ok(field)
}

fn scalar_times_vector(scalar: ArrayView3<'_, f64>, vector: ArrayView4<'_, f64>) -> Array4<f64> {
    let (frames, nx, ny) = scalar.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let scale = scalar[[t, ix, iy]];
                out[[t, ix, iy, 0]] = scale * vector[[t, ix, iy, 0]];
                out[[t, ix, iy, 1]] = scale * vector[[t, ix, iy, 1]];
            }
        }
    }
    out
}

fn cubic_gradient_flux(grad_rho: ArrayView4<'_, f64>) -> Array4<f64> {
    let (frames, nx, ny, _) = grad_rho.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let dx = grad_rho[[t, ix, iy, 0]];
                let dy = grad_rho[[t, ix, iy, 1]];
                let norm2 = dx * dx + dy * dy;
                out[[t, ix, iy, 0]] = norm2 * dx;
                out[[t, ix, iy, 1]] = norm2 * dy;
            }
        }
    }
    out
}

fn validate_inputs(
    rho: ArrayView3<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
    term_names: &[String],
    lx: f64,
    ly: f64,
) -> CoreResult<()> {
    let (frames, nx, ny) = rho.dim();
    if frames == 0 || nx == 0 || ny == 0 {
        return Err(CoreError::InvalidInput(
            "rho axes must be non-empty".to_string(),
        ));
    }
    if !(lx.is_finite() && lx > 0.0 && ly.is_finite() && ly > 0.0) {
        return Err(CoreError::InvalidInput(
            "domain lengths must be positive".to_string(),
        ));
    }
    if sample_indices.dim().1 != 3 {
        return Err(CoreError::Shape(
            "sample_indices must have shape (N, 3)".to_string(),
        ));
    }
    if term_names.is_empty() {
        return Err(CoreError::InvalidInput(
            "term_names must be non-empty".to_string(),
        ));
    }
    for row in 0..sample_indices.dim().0 {
        let (t, ix, iy) = index_triplet(sample_indices, row)?;
        if t >= frames || ix >= nx || iy >= ny {
            return Err(CoreError::InvalidInput(format!(
                "sample index {row} is out of bounds"
            )));
        }
    }
    Ok(())
}

fn index_triplet(
    sample_indices: ArrayView2<'_, i64>,
    row: usize,
) -> CoreResult<(usize, usize, usize)> {
    let t = sample_indices[[row, 0]];
    let ix = sample_indices[[row, 1]];
    let iy = sample_indices[[row, 2]];
    if t < 0 || ix < 0 || iy < 0 {
        return Err(CoreError::InvalidInput(format!(
            "sample index {row} contains a negative coordinate"
        )));
    }
    Ok((t as usize, ix as usize, iy as usize))
}
