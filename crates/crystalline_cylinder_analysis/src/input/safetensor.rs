//! Read-only, field-selective SafeTensors memory mapping.

use std::collections::HashSet;
use std::fs::File;
use std::path::Path;

use bytemuck::Pod;
use memmap2::{Mmap, MmapOptions};
use safetensors::Dtype;
use safetensors::SafeTensors;

/// Scalar dtypes accepted from trajectory-analysis tensors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorDtype {
    Bool,
    F32,
    F64,
    I8,
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
    pub fn as_f32(&self) -> &[f32] {
        self.cast_values(TensorDtype::F32, "F32")
    }

    /// View an F64 tensor as native typed values without copying.
    pub fn as_f64(&self) -> &[f64] {
        self.cast_values(TensorDtype::F64, "F64")
    }

    /// View an I64 tensor as native typed values without copying.
    pub fn as_i64(&self) -> &[i64] {
        self.cast_values(TensorDtype::I64, "I64")
    }

    /// View an I32 tensor as native typed values without copying.
    pub fn as_i32(&self) -> &[i32] {
        self.cast_values(TensorDtype::I32, "I32")
    }

    /// View an I8 tensor as native typed values without copying.
    pub fn as_i8(&self) -> &[i8] {
        self.cast_values(TensorDtype::I8, "I8")
    }

    /// View a BOOL tensor as its canonical zero/one bytes.
    pub fn as_bool_bytes(&self) -> &[u8] {
        assert_eq!(self.dtype, TensorDtype::Bool, "bad dtype");
        self.bytes
    }

    fn cast_values<T: Pod>(&self, expected: TensorDtype, _name: &str) -> &[T] {
        assert_eq!(self.dtype, expected, "bad dtype");
        const { assert!(cfg!(target_endian = "little"), "bad endian") };
        bytemuck::try_cast_slice(self.bytes).expect("bad bytes")
    }
}

/// One memory-mapped safetensor shard whose fields are loaded on demand.
pub struct MappedShard {
    mapping: Mmap,
    names: HashSet<String>,
}

impl MappedShard {
    /// Open one immutable shard without materializing its tensor fields.
    ///
    /// The mapped file must not be modified or replaced until this value is
    /// dropped. Callers should therefore map immutable, completed shards only.
    #[allow(unsafe_code)]
    pub fn open(path: &Path) -> Self {
        assert!(path.is_file(), "missing shard");

        let file = File::open(path).expect("open shard");
        // SAFETY: the analysis maps completed inputs read-only and never mutates
        // or replaces them while `MappedShard` is alive.
        let mapping = unsafe { MmapOptions::new().map(&file) }.expect("map shard");
        let tensors = SafeTensors::deserialize(&mapping).expect("bad shard");
        let names = tensors.iter().map(|(name, _)| name.to_owned()).collect();
        Self { mapping, names }
    }

    /// Borrow one named tensor directly from the mapped file.
    pub fn tensor<'a>(&'a self, name: &str) -> SafetensorView<'a> {
        let tensors = SafeTensors::deserialize(&self.mapping).expect("bad shard");
        let view = tensors.tensor(name).expect("missing tensor");
        let dtype = match view.dtype() {
            Dtype::BOOL => TensorDtype::Bool,
            Dtype::F32 => TensorDtype::F32,
            Dtype::F64 => TensorDtype::F64,
            Dtype::I8 => TensorDtype::I8,
            Dtype::I32 => TensorDtype::I32,
            Dtype::I64 => TensorDtype::I64,
            _ => panic!("bad dtype"),
        };
        SafetensorView {
            dtype,
            shape: view.shape().to_vec(),
            bytes: view.data(),
        }
    }

    /// Return whether the mapped shard contains a named tensor.
    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}
