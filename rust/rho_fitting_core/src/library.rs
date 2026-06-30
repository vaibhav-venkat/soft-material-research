use std::collections::HashMap;

use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4};

use crate::fft_ops;
use crate::{CoreError, CoreResult};

pub fn build_density_library(
    rho: ArrayView3<'_, f64>,
    p_density: ArrayView4<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
    term_names: &[String],
    lx: f64,
    ly: f64,
) -> CoreResult<Array2<f64>> {
    validate_inputs(rho, p_density, sample_indices, term_names)?;
    let samples = sample_indices.dim().0;
    let mut out = Array2::<f64>::zeros((samples, term_names.len()));
    let mut field_cache: HashMap<String, Array3<f64>> = HashMap::new();

    for (column, name) in term_names.iter().enumerate() {
        println!(
            "[rho_fitting] density-library term {}/{}: {}",
            column + 1,
            term_names.len(),
            name
        );
        let field = match field_cache.get(name) {
            Some(field) => field,
            None => {
                let field = build_term_field(rho, p_density, name, lx, ly)?;
                field_cache.insert(name.clone(), field);
                field_cache.get(name).expect("inserted field must exist")
            }
        };
        for row in 0..samples {
            let (t, ix, iy) = index_triplet(sample_indices, row)?;
            out[[row, column]] = field[[t, ix, iy]];
        }
    }

    Ok(out)
}

pub fn known_density_term(name: &str) -> bool {
    matches!(
        name,
        "div_p"
            | "lap_rho"
            | "lap_rho2"
            | "lap_rho3"
            | "div_rho_p"
            | "div_rho2_p"
            | "div_p_norm2_p"
            | "div_p_perp"
            | "div_rho_p_perp"
            | "div_rho2_p_perp"
            | "div_p_norm2_p_perp"
    )
}

fn validate_inputs(
    rho: ArrayView3<'_, f64>,
    p_density: ArrayView4<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
    term_names: &[String],
) -> CoreResult<()> {
    let (frames, nx, ny) = rho.dim();
    if frames == 0 || nx == 0 || ny == 0 {
        return Err(CoreError::InvalidInput(
            "rho axes must be non-empty".to_string(),
        ));
    }
    if p_density.dim() != (frames, nx, ny, 2) {
        return Err(CoreError::Shape(
            "P_density must have shape (T, Nx, Ny, 2)".to_string(),
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

fn build_term_field(
    rho: ArrayView3<'_, f64>,
    p_density: ArrayView4<'_, f64>,
    name: &str,
    lx: f64,
    ly: f64,
) -> CoreResult<Array3<f64>> {
    match name {
        "div_p" => fft_ops::divergence_vector(p_density, lx, ly),
        "lap_rho" => fft_ops::laplacian_scalar(rho, lx, ly),
        "lap_rho2" => {
            let field = scalar_power(rho, 2);
            fft_ops::laplacian_scalar(field.view(), lx, ly)
        }
        "lap_rho3" => {
            let field = scalar_power(rho, 3);
            fft_ops::laplacian_scalar(field.view(), lx, ly)
        }
        "div_rho_p" => {
            let field = scale_vector_by_rho_power(rho, p_density, 1, false, false);
            fft_ops::divergence_vector(field.view(), lx, ly)
        }
        "div_rho2_p" => {
            let field = scale_vector_by_rho_power(rho, p_density, 2, false, false);
            fft_ops::divergence_vector(field.view(), lx, ly)
        }
        "div_p_norm2_p" => {
            let field = scale_vector_by_rho_power(rho, p_density, 0, false, true);
            fft_ops::divergence_vector(field.view(), lx, ly)
        }
        "div_p_perp" => {
            let field = scale_vector_by_rho_power(rho, p_density, 0, true, false);
            fft_ops::divergence_vector(field.view(), lx, ly)
        }
        "div_rho_p_perp" => {
            let field = scale_vector_by_rho_power(rho, p_density, 1, true, false);
            fft_ops::divergence_vector(field.view(), lx, ly)
        }
        "div_rho2_p_perp" => {
            let field = scale_vector_by_rho_power(rho, p_density, 2, true, false);
            fft_ops::divergence_vector(field.view(), lx, ly)
        }
        "div_p_norm2_p_perp" => {
            let field = scale_vector_by_rho_power(rho, p_density, 0, true, true);
            fft_ops::divergence_vector(field.view(), lx, ly)
        }
        _ => Err(CoreError::InvalidInput(format!(
            "unknown density term {name}"
        ))),
    }
}

fn scalar_power(rho: ArrayView3<'_, f64>, power: i32) -> Array3<f64> {
    let (frames, nx, ny) = rho.dim();
    let mut out = Array3::<f64>::zeros((frames, nx, ny));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                out[[t, ix, iy]] = rho[[t, ix, iy]].powi(power);
            }
        }
    }
    out
}

fn scale_vector_by_rho_power(
    rho: ArrayView3<'_, f64>,
    p_density: ArrayView4<'_, f64>,
    rho_power: i32,
    perpendicular: bool,
    p_norm2_scale: bool,
) -> Array4<f64> {
    let (frames, nx, ny) = rho.dim();
    let mut out = Array4::<f64>::zeros((frames, nx, ny, 2));
    for t in 0..frames {
        for ix in 0..nx {
            for iy in 0..ny {
                let px = p_density[[t, ix, iy, 0]];
                let py = p_density[[t, ix, iy, 1]];
                let (vx, vy) = if perpendicular { (-py, px) } else { (px, py) };
                let mut scale = if rho_power == 0 {
                    1.0
                } else {
                    rho[[t, ix, iy]].powi(rho_power)
                };
                if p_norm2_scale {
                    scale *= px * px + py * py;
                }
                out[[t, ix, iy, 0]] = scale * vx;
                out[[t, ix, iy, 1]] = scale * vy;
            }
        }
    }
    out
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
