//! Typed inputs, intermediate series, and numerical results.

mod case;
mod manifest;
mod results;
mod series;

pub use case::{CaseMetadata, CaseSchema, DiscoveredDataset, ReplicateGroup};
pub use manifest::{AnalysisManifest, ShardManifest};
pub use results::{DampedCosineFit, LaplaceGrid, PreferredAxis, PreferredEstimate};
pub use series::{ComSeries, CorrelationSeries};
