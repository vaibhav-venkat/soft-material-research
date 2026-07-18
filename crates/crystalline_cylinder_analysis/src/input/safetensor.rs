//! Read-only, field-selective Candle safetensor mapping.

use std::path::Path;

use crate::error::AnalysisResult;

/// Scalar dtypes accepted from trajectory-analysis tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtype {
    F32,
    F64,
    I64,
}

/// Borrowed tensor bytes and logical row-major shape.
#[derive(Clone, Copy, Debug)]
pub struct TensorSlice<'a> {
    pub dtype: TensorDtype,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

/// One memory-mapped safetensor shard whose fields are loaded on demand.
pub struct MappedShard {
    _private: (),
}

impl MappedShard {
    /// Open one immutable shard without materializing its tensor fields.
    pub fn open(_path: &Path) -> AnalysisResult<Self> {
        todo!("wrap candle_core::safetensors::MmapedSafetensors")
    }

    /// Borrow one named tensor directly from the mapped file.
    pub fn tensor<'a>(&'a self, _name: &str) -> AnalysisResult<TensorSlice<'a>> {
        todo!("validate and expose a Candle safetensor view")
    }
}
