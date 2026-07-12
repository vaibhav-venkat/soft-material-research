use ndarray::{Array1, Array2, Array4, ArrayView2, ArrayViewD, IxDyn};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{CoreError, CoreResult};

/// Sample valid `(frame, x, theta, r)` rows from a four-dimensional mask.
pub fn sample_grid_rows(
    valid_mask: ArrayViewD<'_, bool>,
    nd: usize,
    seed: u64,
    replace: bool,
) -> CoreResult<Array2<i64>> {
    if valid_mask.ndim() != 4 {
        return Err(CoreError::Shape(
            "valid_mask must have shape (T,Nx,Ntheta,Nr)".to_string(),
        ));
    }
    if nd == 0 {
        return Err(CoreError::InvalidInput("nd must be positive".to_string()));
    }
    let shape = valid_mask.shape();
    let mut valid = Vec::new();
    for t in 0..shape[0] {
        for ix in 0..shape[1] {
            for itheta in 0..shape[2] {
                for ir in 0..shape[3] {
                    if valid_mask[[t, ix, itheta, ir]] {
                        valid.push((t, ix, itheta, ir));
                    }
                }
            }
        }
    }
    if valid.is_empty() {
        return Err(CoreError::InvalidInput(
            "no valid rows to sample".to_string(),
        ));
    }
    let mut rng = seeded_rng(seed);
    let sampled = if replace {
        (0..nd)
            .map(|_| valid[rng.gen_range(0..valid.len())])
            .collect::<Vec<_>>()
    } else {
        valid.shuffle(&mut rng);
        valid.truncate(nd.min(valid.len()));
        valid
    };
    let mut out = Array2::<i64>::zeros((sampled.len(), 4));
    for (row, (t, ix, itheta, ir)) in sampled.into_iter().enumerate() {
        out[[row, 0]] = t as i64;
        out[[row, 1]] = ix as i64;
        out[[row, 2]] = itheta as i64;
        out[[row, 3]] = ir as i64;
    }
    Ok(out)
}

/// Build a shared finite-value mask from scalar and tensor fields.
pub fn finite_grid_mask(fields: &[ArrayViewD<'_, f64>]) -> CoreResult<Array4<bool>> {
    let first = fields.first().ok_or_else(|| {
        CoreError::InvalidInput(
            "at least one field is required for valid-mask construction".to_string(),
        )
    })?;
    if first.ndim() < 4 {
        return Err(CoreError::Shape(
            "fields must have at least four grid axes".to_string(),
        ));
    }
    let grid_shape = &first.shape()[..4];
    for field in fields {
        if field.ndim() < 4 || &field.shape()[..4] != grid_shape {
            return Err(CoreError::Shape(
                "all fields must share shape (T,Nx,Ntheta,Nr,...)".to_string(),
            ));
        }
    }
    let mut mask = Array4::<bool>::from_elem(
        (grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3]),
        true,
    );
    for t in 0..grid_shape[0] {
        for ix in 0..grid_shape[1] {
            for itheta in 0..grid_shape[2] {
                for ir in 0..grid_shape[3] {
                    for field in fields {
                        if !grid_point_is_finite(field, [t, ix, itheta, ir]) {
                            mask[[t, ix, itheta, ir]] = false;
                            break;
                        }
                    }
                }
            }
        }
    }
    Ok(mask)
}

/// Construct the deterministic RNG used for regression row sampling.
pub fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Rust-owned regression rows for one target and its candidate libraries.
pub struct RegressionRows {
    /// Divergence rows followed by weighted flux rows, suitable for fitting.
    pub x: Array2<f64>,
    pub y: Array1<f64>,
    /// Finite divergence rows used for primary evaluation and reporting.
    pub divergence_x: Array2<f64>,
    pub divergence_y: Array1<f64>,
    pub divergence_rows: Array2<f64>,
    pub row_index: Array2<i64>,
    /// Unweighted finite flux rows, when flux fields and a positive weight are supplied.
    pub flux_x: Option<Array2<f64>>,
    pub flux_y: Option<Array1<f64>>,
    pub flux_row_index: Option<Array2<i64>>,
}

/// Assemble contiguous divergence and optional flux regression rows.
///
/// `target_divergence` has grid axes `(T,Nx,Ntheta,Nr)` followed by zero or
/// more component axes. `divergence_library` has a leading candidate-term axis
/// followed by the target shape. Flux fields follow the same convention. Sample
/// coordinates are `(N,4)` in `(T,Nx,Ntheta,Nr)` order. Component metadata is
/// appended to each grid coordinate in row-major component order.
pub fn assemble_regression_rows(
    target_divergence: ArrayViewD<'_, f64>,
    divergence_library: ArrayViewD<'_, f64>,
    sample_coordinates: ArrayView2<'_, i64>,
    target_flux: Option<ArrayViewD<'_, f64>>,
    flux_library: Option<ArrayViewD<'_, f64>>,
    flux_weight: f64,
) -> CoreResult<RegressionRows> {
    validate_design_shapes(
        &target_divergence,
        &divergence_library,
        sample_coordinates,
        "divergence",
    )?;
    if !flux_weight.is_finite() || flux_weight < 0.0 {
        return Err(CoreError::InvalidInput(
            "flux_weight must be finite and non-negative".to_string(),
        ));
    }
    let divergence = collect_rows(target_divergence, divergence_library, sample_coordinates)?;

    let flux = match (target_flux, flux_library, flux_weight > 0.0) {
        (None, None, _) => None,
        (Some(target), Some(library), true) => {
            validate_design_shapes(&target, &library, sample_coordinates, "flux")?;
            Some(collect_rows(target, library, sample_coordinates)?)
        }
        (Some(_), Some(_), false) => None,
        _ => {
            return Err(CoreError::InvalidInput(
                "target_flux and flux_library must be supplied together".to_string(),
            ));
        }
    };

    let divergence_rows = prepend_target(&divergence.y, &divergence.x);
    let (x, y) = if let Some(flux) = &flux {
        let mut x =
            Array2::<f64>::zeros((divergence.x.nrows() + flux.x.nrows(), divergence.x.ncols()));
        let mut y = Array1::<f64>::zeros(divergence.y.len() + flux.y.len());
        copy_rows(&mut x, 0, &divergence.x);
        copy_rows_scaled(&mut x, divergence.x.nrows(), &flux.x, flux_weight);
        copy_values(&mut y, 0, &divergence.y);
        copy_values_scaled(&mut y, divergence.y.len(), &flux.y, flux_weight);
        (x, y)
    } else {
        (divergence.x.clone(), divergence.y.clone())
    };

    Ok(RegressionRows {
        x,
        y,
        divergence_x: divergence.x,
        divergence_y: divergence.y,
        divergence_rows,
        row_index: divergence.row_index,
        flux_x: flux.as_ref().map(|rows| rows.x.clone()),
        flux_y: flux.as_ref().map(|rows| rows.y.clone()),
        flux_row_index: flux.map(|rows| rows.row_index),
    })
}

/// Preserve the existing generic component sampler as a fitting-level helper.
pub fn sample_component_rows(
    target: ArrayViewD<'_, f64>,
    library: ArrayViewD<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
) -> CoreResult<(Array2<f64>, Array2<i64>)> {
    if target.ndim() < 4 {
        return Err(CoreError::Shape(
            "target must have grid axes plus component axes".to_string(),
        ));
    }
    if library.ndim() != target.ndim() + 1 || library.shape()[1..] != *target.shape() {
        return Err(CoreError::Shape(
            "library must have a leading term axis plus matching target axes".to_string(),
        ));
    }
    if sample_indices.dim().1 != 3 {
        return Err(CoreError::Shape("sample_indices must be (N,3)".to_string()));
    }
    let rows = collect_rows_legacy(target, library, sample_indices)?;
    Ok((prepend_target(&rows.y, &rows.x), rows.row_index))
}

struct CollectedRows {
    x: Array2<f64>,
    y: Array1<f64>,
    row_index: Array2<i64>,
}

fn collect_rows(
    target: ArrayViewD<'_, f64>,
    library: ArrayViewD<'_, f64>,
    sample_coordinates: ArrayView2<'_, i64>,
) -> CoreResult<CollectedRows> {
    let component_shape = target.shape()[4..].to_vec();
    collect_rows_with_coordinate_width(
        target,
        library,
        sample_coordinates,
        &component_shape,
        4,
        true,
    )
}

fn collect_rows_legacy(
    target: ArrayViewD<'_, f64>,
    library: ArrayViewD<'_, f64>,
    sample_indices: ArrayView2<'_, i64>,
) -> CoreResult<CollectedRows> {
    let component_shape = target.shape()[3..].to_vec();
    collect_rows_with_coordinate_width(target, library, sample_indices, &component_shape, 3, false)
}

fn collect_rows_with_coordinate_width(
    target: ArrayViewD<'_, f64>,
    library: ArrayViewD<'_, f64>,
    sample_coordinates: ArrayView2<'_, i64>,
    component_shape: &[usize],
    coordinate_width: usize,
    reject_nonfinite: bool,
) -> CoreResult<CollectedRows> {
    let terms = library.shape()[0];
    let component_count = component_shape.iter().product::<usize>();
    let mut values = Vec::new();
    let mut targets = Vec::new();
    let mut metadata = Vec::new();
    let metadata_width = coordinate_width + component_shape.len();
    for row in 0..sample_coordinates.nrows() {
        let mut grid_index = Vec::with_capacity(coordinate_width);
        for axis in 0..coordinate_width {
            grid_index.push(checked_index(
                sample_coordinates[[row, axis]],
                target.shape()[axis],
                axis_name(axis),
            )?);
        }
        for component_flat in 0..component_count {
            let components = unravel_component(component_flat, component_shape);
            let mut target_index = grid_index.clone();
            target_index.extend(components.iter().copied());
            let target_value = target[IxDyn(&target_index)];
            let mut row_values = Vec::with_capacity(terms);
            let mut finite = target_value.is_finite();
            for term in 0..terms {
                let mut library_index = vec![term];
                library_index.extend(target_index.iter().copied());
                let value = library[IxDyn(&library_index)];
                finite &= value.is_finite();
                row_values.push(value);
            }
            if reject_nonfinite && !finite {
                continue;
            }
            targets.push(target_value);
            values.extend(row_values);
            metadata.extend(grid_index.iter().map(|value| *value as i64));
            metadata.extend(components.iter().map(|value| *value as i64));
        }
    }
    let rows = targets.len();
    Ok(CollectedRows {
        x: Array2::from_shape_vec((rows, terms), values).map_err(|error| {
            CoreError::InvalidInput(format!("regression row assembly failed: {error}"))
        })?,
        y: Array1::from_vec(targets),
        row_index: Array2::from_shape_vec((rows, metadata_width), metadata).map_err(|error| {
            CoreError::InvalidInput(format!("regression metadata assembly failed: {error}"))
        })?,
    })
}

fn validate_design_shapes(
    target: &ArrayViewD<'_, f64>,
    library: &ArrayViewD<'_, f64>,
    sample_coordinates: ArrayView2<'_, i64>,
    label: &str,
) -> CoreResult<()> {
    if target.ndim() < 4 {
        return Err(CoreError::Shape(format!(
            "{label} target must have at least four grid axes"
        )));
    }
    if library.ndim() != target.ndim() + 1 || library.shape()[1..] != *target.shape() {
        return Err(CoreError::Shape(format!(
            "{label} library must have shape (terms, ...target shape)"
        )));
    }
    if sample_coordinates.ncols() != 4 {
        return Err(CoreError::Shape(
            "sample_coordinates must have shape (N,4)".to_string(),
        ));
    }
    Ok(())
}

fn prepend_target(y: &Array1<f64>, x: &Array2<f64>) -> Array2<f64> {
    let mut rows = Array2::<f64>::zeros((y.len(), x.ncols() + 1));
    for row in 0..y.len() {
        rows[[row, 0]] = y[row];
        for term in 0..x.ncols() {
            rows[[row, term + 1]] = x[[row, term]];
        }
    }
    rows
}

fn copy_rows(target: &mut Array2<f64>, start: usize, source: &Array2<f64>) {
    for row in 0..source.nrows() {
        for col in 0..source.ncols() {
            target[[start + row, col]] = source[[row, col]];
        }
    }
}

fn copy_rows_scaled(target: &mut Array2<f64>, start: usize, source: &Array2<f64>, scale: f64) {
    for row in 0..source.nrows() {
        for col in 0..source.ncols() {
            target[[start + row, col]] = scale * source[[row, col]];
        }
    }
}

fn copy_values(target: &mut Array1<f64>, start: usize, source: &Array1<f64>) {
    for (offset, value) in source.iter().enumerate() {
        target[start + offset] = *value;
    }
}

fn copy_values_scaled(target: &mut Array1<f64>, start: usize, source: &Array1<f64>, scale: f64) {
    for (offset, value) in source.iter().enumerate() {
        target[start + offset] = scale * value;
    }
}

fn unravel_component(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        out[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    out
}

fn checked_index(value: i64, upper: usize, name: &str) -> CoreResult<usize> {
    if value < 0 || value as usize >= upper {
        return Err(CoreError::InvalidInput(format!(
            "{name} sample index out of bounds"
        )));
    }
    Ok(value as usize)
}

fn grid_point_is_finite(field: &ArrayViewD<'_, f64>, grid_index: [usize; 4]) -> bool {
    let component_shape = &field.shape()[4..];
    let component_count = component_shape.iter().product::<usize>().max(1);
    for component_flat in 0..component_count {
        let components = unravel_component(component_flat, component_shape);
        let mut index = grid_index.to_vec();
        index.extend(components);
        if !field[IxDyn(&index)].is_finite() {
            return false;
        }
    }
    true
}

fn axis_name(axis: usize) -> &'static str {
    match axis {
        0 => "frame",
        1 => "x",
        2 => "theta",
        3 => "radial",
        _ => "grid",
    }
}
