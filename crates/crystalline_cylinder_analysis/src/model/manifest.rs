//! Deserialized analysis-manifest records.

use serde::{Deserialize, Serialize};

use super::CaseMetadata;

/// Top-level manifest emitted by the Python trajectory analysis.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnalysisManifest {
    pub schema: String,
    pub complete: bool,
    pub case: CaseMetadata,
    pub frame_count: usize,
    pub shards: Vec<ShardManifest>,
}

/// One contiguous safetensor frame shard.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ShardManifest {
    pub file: String,
    pub frame_start: usize,
    pub frame_stop: usize,
    #[serde(default)]
    pub steps: Vec<i64>,
    pub bytes: Option<u64>,
}
