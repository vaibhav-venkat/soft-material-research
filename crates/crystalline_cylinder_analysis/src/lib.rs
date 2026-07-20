//! Streaming numerical analysis for crystalline-cylinder safetensor outputs.

pub mod backend;
pub mod center_of_mass;
pub mod clusters;
pub mod correlation;
pub mod fit;
pub mod input;
pub mod integration;
pub mod laplace;
pub mod model;
pub mod pipeline;
pub mod replicates;

pub use backend::{AnalysisBackend, ComputeDevice, DeviceAnalysisBackend};
pub use clusters::{
    analyze_dataset_clusters, analyze_dataset_clusters_with_snapshots,
    cluster_area_weighted_probability_histogram, cluster_log_probability_histogram,
    cluster_probability_histogram, ClusterConfig, ClusterHistogram, ClusterKind, ClusterRecord,
    ClusterSnapshot, DatasetClusterAnalysis,
};
pub use model::{
    AnalysisManifest, CaseMetadata, CaseSchema, ComSeries, CorrelationSeries, DampedCosineFit,
    DiscoveredDataset, LaplaceGrid, PreferredAxis, PreferredEstimate, ReplicateGroup,
};
