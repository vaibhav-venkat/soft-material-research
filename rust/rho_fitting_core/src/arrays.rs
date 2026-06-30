use ndarray::{ArrayViewD, IxDyn};

use crate::{CoreError, CoreResult};

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

pub fn dyn_shape(shape: &[usize]) -> IxDyn {
    IxDyn(shape)
}
