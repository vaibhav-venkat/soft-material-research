use ndarray::{ArrayD, ArrayView1, ArrayViewD, IxDyn};

use crate::{CoreError, CoreResult};

const COMPONENTS: usize = 3;

pub(crate) fn alignment_tensor(
    rho: ArrayViewD<'_, f64>,
    q: ArrayViewD<'_, f64>,
) -> CoreResult<ArrayD<f64>> {
    let lead_shape = require_tensor_shape(&rho, &q, 2, "Q")?;
    let mut out = ArrayD::zeros(q.raw_dim());
    for lead in leading_indices(&lead_shape) {
        let rho_value = rho[IxDyn(&lead)];
        for row in 0..COMPONENTS {
            for col in 0..COMPONENTS {
                let mut index = lead.clone();
                index.extend([row, col]);
                out[IxDyn(&index)] = q[IxDyn(&index)]
                    + if row == col {
                        rho_value / COMPONENTS as f64
                    } else {
                        0.0
                    };
            }
        }
    }
    Ok(out)
}

pub(crate) fn alignment_dot_gradient(
    a: ArrayViewD<'_, f64>,
    grad_rho: ArrayViewD<'_, f64>,
) -> CoreResult<ArrayD<f64>> {
    let lead_shape = require_matching_trailing_shape(&a, &grad_rho, 2, 1, "A and grad_rho")?;
    let mut out = ArrayD::zeros(IxDyn(&[lead_shape.as_slice(), &[COMPONENTS]].concat()));
    for lead in leading_indices(&lead_shape) {
        for direction in 0..COMPONENTS {
            let mut value = 0.0;
            for component in 0..COMPONENTS {
                let mut a_index = lead.clone();
                a_index.extend([direction, component]);
                let mut grad_index = lead.clone();
                grad_index.push(component);
                value += a[IxDyn(&a_index)] * grad_rho[IxDyn(&grad_index)];
            }
            let mut out_index = lead.clone();
            out_index.push(direction);
            out[IxDyn(&out_index)] = value;
        }
    }
    Ok(out)
}

pub(crate) fn estimate_ubar(
    y_p: ArrayViewD<'_, f64>,
    a: ArrayViewD<'_, f64>,
) -> CoreResult<ArrayD<f64>> {
    let lead_shape = require_matching_trailing_shape(&y_p, &a, 2, 2, "Y_P and A")?;
    let mut out = ArrayD::zeros(IxDyn(&lead_shape));
    for lead in leading_indices(&lead_shape) {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for row in 0..COMPONENTS {
            for col in 0..COMPONENTS {
                let mut index = lead.clone();
                index.extend([row, col]);
                numerator += y_p[IxDyn(&index)] * a[IxDyn(&index)];
                denominator += a[IxDyn(&index)] * a[IxDyn(&index)];
            }
        }
        out[IxDyn(&lead)] = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
    }
    Ok(out)
}

pub(crate) fn p_alignment_traceless(p: ArrayViewD<'_, f64>) -> CoreResult<ArrayD<f64>> {
    let lead_shape = require_last_axis(&p, 1, "P")?;
    let out_shape = [lead_shape.as_slice(), &[COMPONENTS, COMPONENTS, COMPONENTS]].concat();
    let mut out = ArrayD::zeros(IxDyn(&out_shape));
    for lead in leading_indices(&lead_shape) {
        for flux in 0..COMPONENTS {
            for row in 0..COMPONENTS {
                for col in 0..COMPONENTS {
                    let mut row_index = lead.clone();
                    row_index.push(row);
                    let mut col_index = lead.clone();
                    col_index.push(col);
                    let mut flux_index = lead.clone();
                    flux_index.push(flux);
                    let mut out_index = lead.clone();
                    out_index.extend([flux, row, col]);
                    out[IxDyn(&out_index)] = p[IxDyn(&row_index)]
                        * if flux == col { 1.0 } else { 0.0 }
                        + p[IxDyn(&col_index)] * if flux == row { 1.0 } else { 0.0 }
                        - (2.0 / 3.0) * p[IxDyn(&flux_index)] * if row == col { 1.0 } else { 0.0 };
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn scale_by_scalar(
    scalar: ArrayViewD<'_, f64>,
    values: ArrayViewD<'_, f64>,
) -> CoreResult<ArrayD<f64>> {
    let lead_shape = require_leading_shape(
        &scalar,
        &values,
        values.ndim() - scalar.ndim(),
        "scalar and values",
    )?;
    let mut out = ArrayD::zeros(values.raw_dim());
    let trailing_count = values.shape()[scalar.ndim()..]
        .iter()
        .product::<usize>()
        .max(1);
    for lead in leading_indices(&lead_shape) {
        let scalar_value = scalar[IxDyn(&lead)];
        for trailing_flat in 0..trailing_count {
            let trailing = unravel(trailing_flat, &values.shape()[scalar.ndim()..]);
            let mut index = lead.clone();
            index.extend(trailing);
            out[IxDyn(&index)] = scalar_value * values[IxDyn(&index)];
        }
    }
    Ok(out)
}

pub(crate) fn project_flux_directions(
    values: ArrayViewD<'_, f64>,
    mode: u8,
) -> CoreResult<ArrayD<f64>> {
    if mode > 1 || values.ndim() < 3 || values.shape()[values.ndim() - 3..] != [3, 3, 3] {
        return Err(CoreError::Shape(
            "projected flux values must end with (3,3,3), with mode 0=tangential or 1=radial"
                .to_string(),
        ));
    }
    let direction_axis = values.ndim() - 3;
    let lead_shape = values.shape()[..direction_axis].to_vec();
    let mut out = ArrayD::zeros(values.raw_dim());
    let trailing_count = values.shape()[direction_axis + 1..]
        .iter()
        .product::<usize>();
    let directions: &[usize] = if mode == 0 { &[0, 1] } else { &[2] };
    for lead in leading_indices(&lead_shape) {
        for &direction in directions {
            for trailing_flat in 0..trailing_count {
                let trailing = unravel(trailing_flat, &values.shape()[direction_axis + 1..]);
                let mut index = lead.clone();
                index.push(direction);
                index.extend(trailing);
                out[IxDyn(&index)] = values[IxDyn(&index)];
            }
        }
    }
    Ok(out)
}

pub(crate) fn weighted_linear_combination(
    fields: &[ArrayViewD<'_, f64>],
    coefficients: ArrayView1<'_, f64>,
) -> CoreResult<ArrayD<f64>> {
    if fields.is_empty() || coefficients.len() != fields.len() {
        return Err(CoreError::Shape(
            "linear combination requires one coefficient per field".to_string(),
        ));
    }
    let shape = fields[0].shape();
    if fields.iter().any(|field| field.shape() != shape) {
        return Err(CoreError::Shape(
            "linear-combination fields must have identical shapes".to_string(),
        ));
    }
    let mut out = ArrayD::zeros(fields[0].raw_dim());
    for (index, value) in out.indexed_iter_mut() {
        *value = fields
            .iter()
            .zip(coefficients.iter())
            .map(|(field, coefficient)| coefficient * field[index.clone()])
            .sum();
    }
    Ok(out)
}

fn require_tensor_shape(
    rho: &ArrayViewD<'_, f64>,
    tensor: &ArrayViewD<'_, f64>,
    trailing_axes: usize,
    name: &str,
) -> CoreResult<Vec<usize>> {
    let lead_shape = require_last_axis(tensor, trailing_axes, name)?;
    if rho.shape() != lead_shape.as_slice() {
        return Err(CoreError::Shape(format!(
            "rho and {name} have incompatible leading shapes"
        )));
    }
    Ok(lead_shape)
}

fn require_matching_trailing_shape(
    left: &ArrayViewD<'_, f64>,
    right: &ArrayViewD<'_, f64>,
    left_trailing: usize,
    right_trailing: usize,
    name: &str,
) -> CoreResult<Vec<usize>> {
    if left.ndim() < left_trailing || right.ndim() < right_trailing {
        return Err(CoreError::Shape(format!("{name} have insufficient rank")));
    }
    let left_lead = left.shape()[..left.ndim() - left_trailing].to_vec();
    let right_lead = right.shape()[..right.ndim() - right_trailing].to_vec();
    if left_lead != right_lead
        || left.shape()[left.ndim() - left_trailing..]
            .iter()
            .any(|value| *value != 3)
        || right.shape()[right.ndim() - right_trailing..]
            .iter()
            .any(|value| *value != 3)
    {
        return Err(CoreError::Shape(format!("{name} have incompatible shapes")));
    }
    Ok(left_lead)
}

fn require_leading_shape(
    scalar: &ArrayViewD<'_, f64>,
    values: &ArrayViewD<'_, f64>,
    trailing_axes: usize,
    name: &str,
) -> CoreResult<Vec<usize>> {
    if values.ndim() < trailing_axes
        || scalar.shape() != &values.shape()[..values.ndim() - trailing_axes]
    {
        return Err(CoreError::Shape(format!("{name} have incompatible shapes")));
    }
    Ok(scalar.shape().to_vec())
}

fn require_last_axis(
    values: &ArrayViewD<'_, f64>,
    trailing_axes: usize,
    name: &str,
) -> CoreResult<Vec<usize>> {
    if values.ndim() < trailing_axes
        || values.shape()[values.ndim() - trailing_axes..]
            .iter()
            .any(|value| *value != 3)
    {
        return Err(CoreError::Shape(format!(
            "{name} has incompatible component shape"
        )));
    }
    Ok(values.shape()[..values.ndim() - trailing_axes].to_vec())
}

fn leading_indices(shape: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    let count = shape.iter().product::<usize>().max(1);
    (0..count).map(|flat| unravel(flat, shape))
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    index
}
