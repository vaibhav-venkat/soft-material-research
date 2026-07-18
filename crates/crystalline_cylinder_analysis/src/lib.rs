//! Streaming numerical analysis for crystalline-cylinder safetensor outputs.

pub mod backend;
pub mod center_of_mass;
pub mod correlation;
pub mod error;
pub mod fit;
pub mod input;
pub mod integration;
pub mod laplace;
pub mod model;
pub mod pipeline;
pub mod replicates;

pub use backend::{AnalysisBackend, CpuAnalysisBackend};
pub use error::{AnalysisError, AnalysisResult};
pub use model::{
    AnalysisManifest, CaseMetadata, CaseSchema, ComSeries, CorrelationSeries, DampedCosineFit,
    DiscoveredDataset, LaplaceGrid, PreferredAxis, PreferredEstimate, ReplicateGroup,
};
