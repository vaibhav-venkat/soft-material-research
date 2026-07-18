//! Manifest discovery and lazily mapped safetensor shards.

mod discovery;
mod inspect;
mod manifest;
mod safetensor;

pub use discovery::{discover_datasets, group_replicates};
pub use inspect::{inspect_dataset, DatasetShape, ShardShape};
pub use manifest::{load_manifest, resolve_shard_path, validate_manifest};
pub use safetensor::{MappedShard, TensorDtype, TensorSlice};
