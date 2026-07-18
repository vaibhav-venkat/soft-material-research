//! Manifest discovery and lazily mapped safetensor shards.

mod discovery;
mod manifest;
mod safetensor;

pub use discovery::{discover_datasets, group_replicates};
pub use manifest::{load_manifest, validate_manifest};
pub use safetensor::{MappedShard, TensorDtype, TensorSlice};
