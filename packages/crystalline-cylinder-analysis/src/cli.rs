//! Clap declarations for the public binary interface.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use crystalline_cylinder_analysis::ComputeDevice;

/// Streaming crystalline-cylinder analysis and plotting.
#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Cli {
    #[command(flatten)]
    pub common: CommonArgs,
    #[command(subcommand)]
    pub command: AnalysisCommand,
}

/// Options shared by all analysis command.
#[derive(Clone, Debug, Args)]
pub struct CommonArgs {
    /// Production root(s) containing safetensors_output/
    #[arg(long, global = true)]
    pub input_dir: Vec<PathBuf>,
    /// Destination root for outputs like plots.
    #[arg(long, global = true)]
    pub output_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 1.0e-6, global = true)]
    pub simulation_timestep: f64,

    /// Maximum Rayon worker count; defaults to Rayon policy.
    #[arg(long, global = true)]
    pub threads: Option<usize>,

    /// Tenferro compute device in provider:ordinal form.
    #[arg(long, default_value = "cuda:0", global = true)]
    pub device: ComputeDevice,

    #[arg(long, global = true)]
    pub overwrite: bool,
}

/// Available analysis stages.
#[derive(Clone, Debug, Subcommand)]
pub enum AnalysisCommand {
    /// Validate manifests and mmap required tensors without running analysis.
    Inspect,
    Com(ComArgs),
    Correlation(CorrelationArgs),
    Laplace(LaplaceArgs),
    Preferred(PreferredArgs),
    Fit(FitArgs),
    /// Identify surface structural and coherent-motion clusters.
    Clusters(ClusterArgs),
}

/// Axial center-of-mass controls.
#[derive(Clone, Copy, Debug, Args)]
pub struct ComArgs {
    /// Restrict big-Lx cases to one circumference; omit to include both.
    #[arg(long, value_enum)]
    pub circ: Option<BigLxCircumference>,
}

/// Supported big-Lx circumference families.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum BigLxCircumference {
    #[value(name = "60D")]
    D60,
    #[value(name = "60.5D")]
    D60Point5,
}

impl BigLxCircumference {
    /// Return the circumference in particle diameters.
    pub const fn diameters(self) -> f64 {
        match self {
            Self::D60 => 60.0,
            Self::D60Point5 => 60.5,
        }
    }
}

/// Shared lag-correlation controls.
#[derive(Clone, Copy, Debug, Args)]
pub struct CorrelationArgs {
    /// Restrict big-Lx cases to one circumference; omit to include both.
    #[arg(long, value_enum)]
    pub circ: Option<BigLxCircumference>,
    /// Explicit maximum lag index.
    #[arg(long)]
    pub max_lag: Option<usize>,
}

/// Full transform-grid controls.
#[derive(Clone, Copy, Debug, Args)]
pub struct LaplaceArgs {
    #[command(flatten)]
    pub correlation: CorrelationArgs,
    #[command(flatten)]
    pub transform: TransformArgs,
}

/// Transform-axis controls.
#[derive(Clone, Copy, Debug, Args)]
pub struct TransformArgs {
    #[arg(long)]
    pub r_min: Option<f64>,
    #[arg(long, default_value_t = 0.0)]
    pub r_max: f64,
    #[arg(long, default_value_t = 161)]
    pub r_points: usize,
    #[arg(long)]
    pub omega_min: Option<f64>,
    #[arg(long)]
    pub omega_max: Option<f64>,
    #[arg(long, default_value_t = 241)]
    pub omega_points: usize,
}

/// Preferred-coordinate controls.
#[derive(Clone, Copy, Debug, Args)]
pub struct PreferredArgs {
    #[command(flatten)]
    pub correlation: CorrelationArgs,
    #[arg(long, value_enum, default_value_t = PreferredChoice::Both)]
    pub axis: PreferredChoice,
    #[arg(long)]
    pub r_min: Option<f64>,
    #[arg(long, default_value_t = 0.0)]
    pub r_max: f64,
    #[arg(long, default_value_t = 241)]
    pub r_points: usize,
    #[arg(long)]
    pub omega_max: Option<f64>,
    #[arg(long, default_value_t = 241)]
    pub omega_points: usize,
}

/// Preferred-coordinate selection
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum PreferredChoice {
    Omega,
    R,
    Both,
}

/// Damped-cosine fit controls.
#[derive(Clone, Copy, Debug, Args)]
pub struct FitArgs {
    #[command(flatten)]
    pub correlation: CorrelationArgs,
    #[arg(long)]
    pub omega_max: Option<f64>,
    #[arg(long, default_value_t = 241)]
    pub omega_points: usize,
    #[arg(long, default_value_t = 0.05)]
    pub soft_l1_scale: f64,
    #[arg(long, default_value_t = 1.0e-8)]
    pub tolerance: f64,
    #[arg(long, default_value_t = 20_000)]
    pub maximum_evaluations: usize,
    #[arg(long, default_value_t = 1.0e-12)]
    pub rank_tolerance: f64,
}

/// Surface-cluster controls.
#[derive(Clone, Debug, Args)]
pub struct ClusterArgs {
    /// Restrict big-Lx cases to one circumference; omit to include both.
    #[arg(long, value_enum)]
    pub circ: Option<BigLxCircumference>,
    /// Restrict analysis to these discovered case IDs; repeat as needed.
    #[arg(long)]
    pub case: Vec<String>,
    /// First frame index to analyze (inclusive).
    #[arg(long, default_value_t = 0)]
    pub frame_start: usize,
    /// Frame index at which analysis stops (exclusive); defaults to the trajectory length.
    #[arg(long)]
    pub frame_stop: Option<usize>,
    #[arg(long, default_value_t = 1)]
    pub lag_frames: usize,
    #[arg(long, default_value_t = 50)]
    pub bins: usize,
    #[arg(long, default_value_t = 0.7)]
    pub psi6_min: f64,
    #[arg(long, default_value_t = 5.0)]
    pub misorientation_degrees: f64,
    #[arg(long, default_value_t = 1.7272)]
    pub neighbor_radius_diameters: f64,
    #[arg(long, default_value_t = 0.8)]
    pub motion_cosine_min: f64,
    #[arg(long, default_value_t = 0.1)]
    pub motion_rms_fraction: f64,
    #[arg(long, default_value_t = 0.5)]
    pub motion_magnitude_ratio: f64,
    #[arg(long, default_value_t = 2)]
    pub min_cluster_particles: usize,
    #[arg(long, default_value_t = 64)]
    pub target_shard_mib: usize,
    /// Frame indices to render as static multi-angle cylinder views.
    #[arg(long, value_delimiter = ',', default_value = "200,500,700")]
    pub snapshot_frames: Vec<usize>,
}
