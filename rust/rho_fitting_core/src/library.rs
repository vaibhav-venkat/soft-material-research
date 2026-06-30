use std::collections::HashMap;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::fft_ops;
use crate::{CoreError, CoreResult};

pub fn build_density_library(
    rho: ArrayView3<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
    term_names: &[String],
    lx: f64,
    ly: f64,
) -> CoreResult<Array2<f64>> {
    validate_inputs(rho, sample_indices, term_names)?;
    let samples = sample_indices.dim().0;
    let mut out = Array2::<f64>::zeros((samples, term_names.len()));
    let mut lap_cache: HashMap<usize, Array3<f64>> = HashMap::new();

    for (column, name) in term_names.iter().enumerate() {
        if let Some(power) = rho_power(name) {
            for row in 0..samples {
                let (t, ix, iy) = index_triplet(sample_indices, row)?;
                out[[row, column]] = rho[[t, ix, iy]].powi(power as i32);
            }
            continue;
        }

        if let Some(order) = lap_order(name) {
            let field = if order == 0 {
                rho.to_owned()
            } else {
                lap_cache
                    .entry(order)
                    .or_insert(fft_ops::repeated_laplacian_scalar(rho, lx, ly, order)?)
                    .to_owned()
            };
            for row in 0..samples {
                let (t, ix, iy) = index_triplet(sample_indices, row)?;
                out[[row, column]] = field[[t, ix, iy]];
            }
            continue;
        }

        return Err(CoreError::InvalidInput(format!(
            "unknown density term {name}"
        )));
    }

    Ok(out)
}

pub fn known_density_term(name: &str) -> bool {
    rho_power(name).is_some() || lap_order(name).is_some()
}

fn validate_inputs(
    rho: ArrayView3<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
    term_names: &[String],
) -> CoreResult<()> {
    let (frames, nx, ny) = rho.dim();
    if frames == 0 || nx == 0 || ny == 0 {
        return Err(CoreError::InvalidInput(
            "rho axes must be non-empty".to_string(),
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

fn rho_power(name: &str) -> Option<usize> {
    if name == "rho" {
        return Some(1);
    }
    name.strip_prefix("rho")?.parse::<usize>().ok()
}

fn lap_order(name: &str) -> Option<usize> {
    if name == "lap_rho" {
        return Some(1);
    }
    name.strip_prefix("lap")?
        .strip_suffix("_rho")?
        .parse::<usize>()
        .ok()
}
