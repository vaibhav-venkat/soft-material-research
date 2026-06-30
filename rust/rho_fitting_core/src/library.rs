use std::collections::HashMap;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayView4};

use crate::fft_ops;
use crate::{CoreError, CoreResult};

pub fn build_density_library(
    rho: ArrayView3<'_, f64>,
    p_density: ArrayView4<'_, f64>,
    j_density: ArrayView4<'_, f64>,
    source_cross: ArrayView3<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
    term_names: &[String],
    lx: f64,
    ly: f64,
) -> CoreResult<Array2<f64>> {
    validate_inputs(
        rho,
        p_density,
        j_density,
        source_cross,
        sample_indices,
        term_names,
    )?;
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
                let field = build_term_field(rho, j_density, source_cross, name, lx, ly)?;
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
    matches!(name, "source_cross" | "neg_div_j" | "lap_rho")
}

fn validate_inputs(
    rho: ArrayView3<'_, f64>,
    p_density: ArrayView4<'_, f64>,
    j_density: ArrayView4<'_, f64>,
    source_cross: ArrayView3<'_, f64>,
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
    if j_density.dim() != (frames, nx, ny, 2) {
        return Err(CoreError::Shape(
            "J_density must have shape (T, Nx, Ny, 2)".to_string(),
        ));
    }
    if source_cross.dim() != (frames, nx, ny) {
        return Err(CoreError::Shape(
            "S_cross must have shape (T, Nx, Ny)".to_string(),
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
    j_density: ArrayView4<'_, f64>,
    source_cross: ArrayView3<'_, f64>,
    name: &str,
    lx: f64,
    ly: f64,
) -> CoreResult<Array3<f64>> {
    match name {
        "source_cross" => Ok(source_cross.to_owned()),
        "neg_div_j" => {
            let mut field = fft_ops::divergence_vector(j_density, lx, ly)?;
            field.mapv_inplace(|value| -value);
            Ok(field)
        }
        "lap_rho" => fft_ops::laplacian_scalar(rho, lx, ly),
        _ => Err(CoreError::InvalidInput(format!(
            "unknown density term {name}"
        ))),
    }
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
