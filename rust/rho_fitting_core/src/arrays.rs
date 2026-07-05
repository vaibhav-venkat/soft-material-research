use ndarray::{ArrayViewD, IxDyn};

use crate::{CoreError, CoreResult};

/// Require a dynamic ndarray view to match an exact shape.
pub fn require_shape(array: ArrayViewD<'_, f64>, shape: &[usize]) -> CoreResult<()> {
    if array.shape() == shape {
        Ok(())
    } else {
        Err(CoreError::Shape(format!(
            "expected {:?}, got {:?}",
            shape,
            array.shape()
        )))
    }
}

/// Convert a shape slice into an ndarray dynamic dimension.
pub fn dyn_shape(shape: &[usize]) -> IxDyn {
    IxDyn(shape)
}
