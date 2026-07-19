//! Translation from CLI declarations to the reusable analysis pipeline.

use crate::cli::{
    AnalysisCommand, BigLxCircumference, Cli, CorrelationArgs, LaplaceArgs, PreferredArgs,
    PreferredChoice,
};
use crate::plots::{
    write_com_plot, write_correlation_plot, write_laplace_plots, write_preferred_plot,
};
use crystalline_cylinder_analysis::center_of_mass::{analyze_replica_com, ComConfig};
use crystalline_cylinder_analysis::correlation::{analyze_correlation, CorrelationConfig};
use crystalline_cylinder_analysis::input::{
    discover_datasets, group_replicates, inspect_dataset, DatasetShape,
};
use crystalline_cylinder_analysis::laplace::{
    analyze_laplace, preferred_coordinate, transform_axes, LaplaceConfig,
};
use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use crystalline_cylinder_analysis::replicates::{
    average_com_series, average_correlations, average_preferred_estimates,
};
use crystalline_cylinder_analysis::{
    CaseSchema, ComSeries, CorrelationSeries, CpuAnalysisBackend, PreferredAxis, ReplicateGroup,
};
use rayon::prelude::*;
use std::path::PathBuf;

/// Validate a command, run the requested stages, and write their artifacts.
pub fn run(cli: Cli) {
    let Cli { common, command } = cli;
    let backend = CpuAnalysisBackend::new(common.threads);
    eprintln!(
        "[debug:init] backend=tenferro-cpu threads={}",
        backend.thread_count,
    );

    match command {
        AnalysisCommand::Inspect => inspect_inputs(&common.input_dir),
        AnalysisCommand::Com(args) => run_com(&backend, &common, args.circ),
        AnalysisCommand::Correlation(args) => run_correlation(&backend, &common, args),
        AnalysisCommand::Laplace(args) => run_laplace(&backend, &common, args),
        AnalysisCommand::Preferred(args) => run_preferred(&backend, &common, args),
        command => panic!("{} not implemented", command_name(&command)),
    }
}

fn run_com(
    backend: &CpuAnalysisBackend,
    common: &crate::cli::CommonArgs,
    circumference: Option<BigLxCircumference>,
) {
    let groups = selected_groups(common, circumference);
    let config = com_config(common);
    let analyses = backend.install(|| {
        groups
            .into_par_iter()
            .map(|group| analyze_group_com(group, config))
            .collect::<Vec<_>>()
    });

    for analysis in &analyses {
        let com = analysis.com.as_ref().expect("analysis has no COM");
        eprintln!(
            "[debug:com] case={} replicates={} elapsed_time={:?} x_center={:?} x_velocity={:?}",
            analysis.group.case.case_id,
            com.replicate_count,
            [com.elapsed_time.len()],
            [com.x_center_mean.len()],
            [com.x_velocity_mean.len()]
        );
    }

    let output = output_path(common, "com", "axial_com_velocity.svg");
    assert_output_available(&output, common.overwrite);
    let written = write_com_plot(&analyses, &output);
    eprintln!(
        "[crystalline-cylinder-analysis] command=com cases={} output={}",
        analyses.len(),
        written.display()
    );
}

fn run_correlation(
    backend: &CpuAnalysisBackend,
    common: &crate::cli::CommonArgs,
    args: CorrelationArgs,
) {
    let analyses = correlation_analyses(backend, common, args);

    for analysis in &analyses {
        let correlation = analysis
            .correlation
            .as_ref()
            .expect("analysis has no correlation");
        eprintln!(
            "[debug:correlation] case={} replicates={} lag_time={:?} pearson={:?} origins={:?}",
            analysis.group.case.case_id,
            correlation.replicate_count,
            [correlation.lag_times.len()],
            [correlation.pearson_mean.len()],
            [correlation.origin_counts.len()]
        );
    }

    let output = output_path(common, "correlation", "axial_velocity_pearson.svg");
    assert_output_available(&output, common.overwrite);
    let written = write_correlation_plot(&analyses, &output);
    eprintln!(
        "[crystalline-cylinder-analysis] command=correlation cases={} output={}",
        analyses.len(),
        written.display()
    );
}

fn run_laplace(backend: &CpuAnalysisBackend, common: &crate::cli::CommonArgs, args: LaplaceArgs) {
    let mut analyses = correlation_analyses(backend, common, args.correlation);
    let correlations = analyses
        .iter()
        .map(|analysis| {
            analysis
                .correlation
                .as_ref()
                .expect("analysis has no correlation")
                .clone()
        })
        .collect::<Vec<_>>();
    let config = LaplaceConfig {
        r_min: args.transform.r_min,
        r_max: args.transform.r_max,
        r_points: args.transform.r_points,
        omega_min: args.transform.omega_min,
        omega_max: args.transform.omega_max,
        omega_points: args.transform.omega_points,
    };
    let (r, omega) = transform_axes(&correlations, config);
    backend.install(|| {
        analyses.par_iter_mut().for_each(|analysis| {
            let correlation = analysis
                .correlation
                .as_ref()
                .expect("analysis has no correlation");
            analysis.laplace = Some(analyze_laplace(backend, correlation, &r, &omega));
        });
    });

    for analysis in &analyses {
        let grid = analysis
            .laplace
            .as_ref()
            .expect("analysis has no Laplace grid");
        eprintln!(
            "[debug:laplace] case={} shape={:?} order=[omega,r] values={}",
            analysis.group.case.case_id,
            grid.shape,
            grid.values.len()
        );
    }

    let output_dir = output_root(common).join("laplace");
    let written = write_laplace_plots(&analyses, &output_dir, common.overwrite);
    eprintln!(
        "[crystalline-cylinder-analysis] command=laplace cases={} outputs={}",
        analyses.len(),
        written.len()
    );
}

fn correlation_analyses(
    backend: &CpuAnalysisBackend,
    common: &crate::cli::CommonArgs,
    args: CorrelationArgs,
) -> Vec<CaseAnalysis> {
    correlation_case_inputs(backend, common, args)
        .into_iter()
        .map(|input| input.analysis)
        .collect()
}

struct CorrelationCaseInput {
    analysis: CaseAnalysis,
    replicas: Vec<CorrelationSeries>,
}

fn correlation_case_inputs(
    backend: &CpuAnalysisBackend,
    common: &crate::cli::CommonArgs,
    args: CorrelationArgs,
) -> Vec<CorrelationCaseInput> {
    let groups = selected_groups(common, args.circ);
    let com_config = com_config(common);
    let correlation_config = CorrelationConfig {
        max_lag: args.max_lag,
    };
    backend.install(|| {
        groups
            .into_par_iter()
            .map(|group| {
                analyze_group_correlation_input(backend, group, com_config, correlation_config)
            })
            .collect()
    })
}

fn run_preferred(
    backend: &CpuAnalysisBackend,
    common: &crate::cli::CommonArgs,
    args: PreferredArgs,
) {
    let mut inputs = correlation_case_inputs(backend, common, args.correlation);
    let correlations = inputs
        .iter()
        .map(|input| {
            input
                .analysis
                .correlation
                .as_ref()
                .expect("analysis has no correlation")
                .clone()
        })
        .collect::<Vec<_>>();
    let (r_grid, omega_grid) = transform_axes(
        &correlations,
        LaplaceConfig {
            r_min: args.r_min,
            r_max: args.r_max,
            r_points: args.r_points,
            omega_min: None,
            omega_max: args.omega_max,
            omega_points: args.omega_points,
        },
    );
    let negative_r = r_grid
        .into_iter()
        .filter(|value| *value < 0.0)
        .collect::<Vec<_>>();
    let positive_omega = omega_grid
        .into_iter()
        .filter(|value| *value > 0.0)
        .collect::<Vec<_>>();
    assert!(
        negative_r.len() >= 2,
        "preferred r grid needs two negative points"
    );
    assert!(
        positive_omega.len() >= 2,
        "preferred omega grid needs two positive points"
    );
    let axes = match args.axis {
        PreferredChoice::Omega => vec![(PreferredAxis::Omega, positive_omega.as_slice())],
        PreferredChoice::R => vec![(PreferredAxis::R, negative_r.as_slice())],
        PreferredChoice::Both => vec![
            (PreferredAxis::Omega, positive_omega.as_slice()),
            (PreferredAxis::R, negative_r.as_slice()),
        ],
    };
    backend.install(|| {
        inputs.par_iter_mut().for_each(|input| {
            for &(axis, coordinates) in &axes {
                let estimates = input
                    .replicas
                    .par_iter()
                    .map(|correlation| preferred_coordinate(correlation, axis, coordinates))
                    .collect::<Vec<_>>();
                input
                    .analysis
                    .preferred
                    .push(average_preferred_estimates(&estimates));
            }
        });
    });
    let analyses = inputs
        .into_iter()
        .map(|input| input.analysis)
        .collect::<Vec<_>>();

    for (axis, _) in axes {
        for analysis in &analyses {
            let estimate = preferred_estimate(analysis, axis);
            eprintln!(
                "[debug:preferred] axis={axis:?} case={} coordinate={} std={} replicates={} lower_boundary={} upper_boundary={}",
                analysis.group.case.case_id,
                estimate.coordinate,
                estimate.coordinate_std,
                estimate.replicate_count,
                estimate.at_lower_boundary,
                estimate.at_upper_boundary
            );
        }
        let file_name = match axis {
            PreferredAxis::Omega => "preferred_omega.svg",
            PreferredAxis::R => "preferred_r.svg",
        };
        let output = output_path(common, "preferred", file_name);
        assert_output_available(&output, common.overwrite);
        write_preferred_plot(&analyses, axis, &output);
    }
    eprintln!(
        "[crystalline-cylinder-analysis] command=preferred cases={} axes={}",
        analyses.len(),
        analyses
            .first()
            .map_or(0, |analysis| analysis.preferred.len())
    );
}

fn preferred_estimate(
    analysis: &CaseAnalysis,
    axis: PreferredAxis,
) -> &crystalline_cylinder_analysis::PreferredEstimate {
    analysis
        .preferred
        .iter()
        .find(|estimate| estimate.axis == axis)
        .expect("analysis has no preferred estimate")
}

fn selected_groups(
    common: &crate::cli::CommonArgs,
    circumference: Option<BigLxCircumference>,
) -> Vec<ReplicateGroup> {
    let groups = select_circumference(discover_replicate_groups(&common.input_dir), circumference);
    assert!(!groups.is_empty(), "no matching cases");
    groups
}

fn discover_replicate_groups(input_dirs: &[PathBuf]) -> Vec<ReplicateGroup> {
    assert!(!input_dirs.is_empty(), "missing --input-dir");
    group_replicates(discover_datasets(input_dirs))
}

fn com_config(common: &crate::cli::CommonArgs) -> ComConfig {
    ComConfig {
        timestep: common.simulation_timestep,
    }
}

fn analyze_group_com(group: ReplicateGroup, config: ComConfig) -> CaseAnalysis {
    let replicas = analyze_com_replicas(&group, config);
    case_analysis(group, average_com_series(&replicas), None)
}

fn analyze_group_correlation_input(
    backend: &CpuAnalysisBackend,
    group: ReplicateGroup,
    com_config: ComConfig,
    correlation_config: CorrelationConfig,
) -> CorrelationCaseInput {
    let com_replicas = analyze_com_replicas(&group, com_config);
    // Correlate each seed before aggregation: mean(C_v), never C_(mean v).
    let correlation_replicas = com_replicas
        .par_iter()
        .map(|com| analyze_correlation(backend, com, correlation_config))
        .collect::<Vec<_>>();
    let com = average_com_series(&com_replicas);
    let correlation = average_correlations(&correlation_replicas);
    CorrelationCaseInput {
        analysis: case_analysis(group, com, Some(correlation)),
        replicas: correlation_replicas,
    }
}

fn analyze_com_replicas(group: &ReplicateGroup, config: ComConfig) -> Vec<ComSeries> {
    group
        .datasets
        .par_iter()
        .map(|dataset| analyze_replica_com(dataset, config))
        .collect()
}

fn case_analysis(
    group: ReplicateGroup,
    com: ComSeries,
    correlation: Option<CorrelationSeries>,
) -> CaseAnalysis {
    CaseAnalysis {
        group,
        com: Some(com),
        correlation,
        laplace: None,
        preferred: Vec::new(),
        fit: None,
    }
}

fn output_path(common: &crate::cli::CommonArgs, directory: &str, file: &str) -> PathBuf {
    output_root(common).join(directory).join(file)
}

fn output_root(common: &crate::cli::CommonArgs) -> PathBuf {
    common.output_dir.clone().unwrap_or_else(|| {
        common
            .input_dir
            .first()
            .expect("missing --input-dir")
            .join("crystalline_cylinder_analysis_output")
    })
}

fn assert_output_available(output: &std::path::Path, overwrite: bool) {
    assert!(
        !output.exists() || overwrite,
        "output exists; use --overwrite"
    );
}

fn select_circumference(
    groups: Vec<ReplicateGroup>,
    circumference: Option<BigLxCircumference>,
) -> Vec<ReplicateGroup> {
    groups
        .into_iter()
        .filter(|group| {
            group.schema != CaseSchema::BigLx
                || circumference.is_none_or(|selected| {
                    group.case.circumference_diameters.map(f64::to_bits)
                        == Some(selected.diameters().to_bits())
                })
        })
        .collect()
}

/// Return a stable name for each public subcommand.
pub fn command_name(command: &AnalysisCommand) -> &'static str {
    match command {
        AnalysisCommand::Inspect => "inspect",
        AnalysisCommand::Com(_) => "com",
        AnalysisCommand::Correlation(_) => "correlation",
        AnalysisCommand::Laplace(_) => "laplace",
        AnalysisCommand::Preferred(_) => "preferred",
        AnalysisCommand::Fit(_) => "fit",
        AnalysisCommand::All(_) => "all",
    }
}

fn inspect_inputs(input_dirs: &[PathBuf]) {
    for group in discover_replicate_groups(input_dirs) {
        let schema = match group.schema {
            CaseSchema::BigLx => "big-lx",
            CaseSchema::Confinement => "confinement",
        };
        eprintln!(
            "[debug:case] case={} schema={} replicates={} expected=[frames, particles, components]",
            group.case.case_id,
            schema,
            group.datasets.len()
        );
        for (replicate_index, dataset) in group.datasets.iter().enumerate() {
            let shape = inspect_dataset(dataset);
            print_dataset_shape(replicate_index, &shape);
        }
    }
}

fn print_dataset_shape(replicate_index: usize, shape: &DatasetShape) {
    for (shard_index, shard) in shape.shards.iter().enumerate() {
        eprintln!(
            "[debug:shard] case={} replicate={} shard={} file={} {}={:?} dtype={:?} step={:?}",
            shape.case_id,
            replicate_index + 1,
            shard_index + 1,
            shard.path.display(),
            shard.coordinate_name,
            shard.coordinate_shape,
            shard.coordinate_dtype,
            shard.step_shape
        );
    }
    eprintln!(
        "[debug:dataset] case={} replicate={} {}={:?} dtype={:?} step={:?}",
        shape.case_id,
        replicate_index + 1,
        shape.coordinate_name,
        shape.coordinate_shape,
        shape.coordinate_dtype,
        shape.step_shape
    );
}
