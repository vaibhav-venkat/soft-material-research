use ndarray::{Array2, ArrayView2, ArrayViewD, IxDyn};

use crate::{CoreError, CoreResult};

/// Expand sampled grid rows across all target components into regression rows.
///
/// `target` is `(T, Nx, Ny, ...)`, `library` is `(terms, T, Nx, Ny, ...)`,
/// and `sample_indices` is `(N, 3)`. The returned data matrix stores the
/// target in column 0 and term values in columns 1.., while the metadata
/// matrix stores grid indices followed by component indices.
///
/// Edge cases: component axes are flattened in row-major order and restored
/// only through the returned metadata.
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
    if library.ndim() != target.ndim() + 1 {
        return Err(CoreError::Shape(
            "library must have a leading term axis plus target axes".to_string(),
        ));
    }
    let terms = library.shape()[0];
    if library.shape()[1..] != *target.shape() {
        return Err(CoreError::Shape(
            "library term fields must match target shape".to_string(),
        ));
    }
    if sample_indices.dim().1 != 3 {
        return Err(CoreError::Shape("sample_indices must be (N,3)".to_string()));
    }

    let component_shape = &target.shape()[3..];
    let component_count = component_shape.iter().product::<usize>();
    let mut x = Array2::<f64>::zeros((sample_indices.dim().0 * component_count, terms + 1));
    let mut meta = Array2::<i64>::zeros((
        sample_indices.dim().0 * component_count,
        3 + component_shape.len(),
    ));

    for row in 0..sample_indices.dim().0 {
        let t = checked_index(sample_indices[[row, 0]], target.shape()[0], "frame")?;
        let ix = checked_index(sample_indices[[row, 1]], target.shape()[1], "x")?;
        let iy = checked_index(sample_indices[[row, 2]], target.shape()[2], "y")?;
        for component_flat in 0..component_count {
            let out_row = row * component_count + component_flat;
            let components = unravel_component(component_flat, component_shape);
            let mut target_index = vec![t, ix, iy];
            target_index.extend(components.iter().copied());
            x[[out_row, 0]] = target[IxDyn(&target_index)];
            for term in 0..terms {
                let mut library_index = vec![term, t, ix, iy];
                library_index.extend(components.iter().copied());
                x[[out_row, term + 1]] = library[IxDyn(&library_index)];
            }
            meta[[out_row, 0]] = t as i64;
            meta[[out_row, 1]] = ix as i64;
            meta[[out_row, 2]] = iy as i64;
            for (component_axis, component) in components.iter().enumerate() {
                meta[[out_row, 3 + component_axis]] = *component as i64;
            }
        }
    }

    Ok((x, meta))
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
