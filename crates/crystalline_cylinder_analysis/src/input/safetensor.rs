//! Read-only, field-selective SafeTensors memory mapping.

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

use bytemuck::Pod;
use memmap2::{Mmap, MmapOptions};
use safetensors::Dtype;
use safetensors::SafeTensors;

use crate::error::{AnalysisError, AnalysisResult};

/// Scalar dtypes accepted from trajectory-analysis tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtype {
    F32,
    F64,
    I32,
    I64,
}

/// Borrowed tensor bytes and logical row-major shape.
#[derive(Clone, Debug)]
pub struct SafetensorView<'a> {
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

impl SafetensorView<'_> {
    /// View an F32 tensor as native typed values without copying.
    pub fn as_f32(&self) -> AnalysisResult<&[f32]> {
        self.cast_values(TensorDtype::F32, "F32")
    }

    /// View an F64 tensor as native typed values without copying.
    pub fn as_f64(&self) -> AnalysisResult<&[f64]> {
        self.cast_values(TensorDtype::F64, "F64")
    }

    /// View an I64 tensor as native typed values without copying.
    pub fn as_i64(&self) -> AnalysisResult<&[i64]> {
        self.cast_values(TensorDtype::I64, "I64")
    }

    /// View an I32 tensor as native typed values without copying.
    pub fn as_i32(&self) -> AnalysisResult<&[i32]> {
        self.cast_values(TensorDtype::I32, "I32")
    }

    fn cast_values<T: Pod>(&self, expected: TensorDtype, name: &str) -> AnalysisResult<&[T]> {
        if self.dtype != expected {
            return Err(AnalysisError::ByteLayout(format!(
                "requested {name} values from {:?} tensor",
                self.dtype
            )));
        }
        if cfg!(target_endian = "big") {
            return Err(AnalysisError::ByteLayout(
                "SafeTensors zero-copy decoding requires a little-endian target".to_owned(),
            ));
        }
        bytemuck::try_cast_slice(self.bytes)
            .map_err(|error| AnalysisError::ByteLayout(error.to_string()))
    }
}

/// One memory-mapped safetensor shard whose fields are loaded on demand.
pub struct MappedShard {
    path: PathBuf,
    mapping: Mmap,
    names: HashSet<String>,
}

impl MappedShard {
    /// Open one immutable shard without materializing its tensor fields.
    ///
    /// The mapped file must not be modified or replaced until this value is
    /// dropped. Callers should therefore map immutable, completed shards only.
    #[allow(unsafe_code)]
    pub fn open(path: &Path) -> AnalysisResult<Self> {
        if !path.is_file() {
            return Err(AnalysisError::InvalidTensor {
                path: path.to_path_buf(),
                name: "<file>".to_owned(),
                message: "safetensor shard does not exist or is not a file".to_owned(),
            });
        }

        let file = File::open(path)?;
        // SAFETY: the analysis maps completed inputs read-only and never mutates
        let mapping = unsafe { MmapOptions::new().map(&file) }?;
        let tensors = SafeTensors::deserialize(&mapping)?;
        let names = tensors.iter().map(|(name, _)| name.to_owned()).collect();
        Ok(Self {
            path: path.to_path_buf(),
            mapping,
            names,
        })
    }

    /// Borrow one named tensor directly from the mapped file.
    pub fn tensor<'a>(&'a self, name: &str) -> AnalysisResult<SafetensorView<'a>> {
        let tensors = SafeTensors::deserialize(&self.mapping)?;
        let view = tensors
            .tensor(name)
            .map_err(|error| AnalysisError::InvalidTensor {
                path: self.path.clone(),
                name: name.to_owned(),
                message: error.to_string(),
            })?;
        let dtype = match view.dtype() {
            Dtype::F32 => TensorDtype::F32,
            Dtype::F64 => TensorDtype::F64,
            Dtype::I32 => TensorDtype::I32,
            Dtype::I64 => TensorDtype::I64,
            other => {
                return Err(AnalysisError::InvalidTensor {
                    path: self.path.clone(),
                    name: name.to_owned(),
                    message: format!("unsupported dtype {other:?}; expected F32, F64, I32, or I64"),
                });
            }
        };
        Ok(SafetensorView {
            dtype,
            shape: view.shape().to_vec(),
            bytes: view.data(),
        })
    }

    /// Return whether the mapped shard contains a named tensor.
    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}
