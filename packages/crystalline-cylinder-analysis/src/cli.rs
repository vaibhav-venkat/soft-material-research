//! Clap declarations for the public binary interface.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

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

    #[arg(long, global = true)]
    pub overwrite: bool,
}

/// Available analysis stages.
#[derive(Clone, Debug, Subcommand)]
pub enum AnalysisCommand {
    Com,
    Correlation(CorrelationArgs),
    Laplace(LaplaceArgs),
    Preferred(PreferredArgs),
    Fit(FitArgs),
    All(AllArgs),
}

/// Shared lag-correlation controls.
#[derive(Clone, Copy, Debug, Args)]
pub struct CorrelationArgs {
    /// Minimum paired time origins retained at the greatest lag.
    #[arg(long, default_value_t = 10)]
    pub min_origins: usize,
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
    #[command(flatten)]
    pub transform: TransformArgs,
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

#[derive(Clone, Copy, Debug, Args)]
pub struct AllArgs {
    #[command(flatten)]
    pub correlation: CorrelationArgs,
    #[command(flatten)]
    pub transform: TransformArgs,
    #[arg(long, default_value_t = 0.05)]
    pub soft_l1_scale: f64,
    #[arg(long, default_value_t = 1.0e-8)]
    pub fit_tolerance: f64,
    #[arg(long, default_value_t = 20_000)]
    pub maximum_fit_evaluations: usize,
    #[arg(long, default_value_t = 1.0e-12)]
    pub rank_tolerance: f64,
}
