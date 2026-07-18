//! Translation from CLI declarations to the reusable analysis pipeline.

use crate::cli::{AnalysisCommand, BigLxCircumference, Cli};
use crate::plots::write_com_plot;
use crystalline_cylinder_analysis::center_of_mass::{analyze_replica_com, ComConfig};
use crystalline_cylinder_analysis::input::{
    discover_datasets, group_replicates, inspect_dataset, DatasetShape,
};
use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use crystalline_cylinder_analysis::replicates::average_com_series;
use crystalline_cylinder_analysis::{CaseSchema, CpuAnalysisBackend};
use rayon::prelude::*;

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
        command => panic!("{} not implemented", command_name(&command)),
    }
}

fn run_com(
    backend: &CpuAnalysisBackend,
    common: &crate::cli::CommonArgs,
    circumference: Option<BigLxCircumference>,
) {
    let datasets = discover_datasets(&common.input_dir);
    let groups = group_replicates(datasets)
        .into_iter()
        .filter(|group| {
            group.schema != CaseSchema::BigLx
                || circumference.is_none_or(|selected| {
                    group.case.circumference_diameters.map(f64::to_bits)
                        == Some(selected.diameters().to_bits())
                })
        })
        .collect::<Vec<_>>();
    assert!(!groups.is_empty(), "no cases");
    let config = ComConfig {
        timestep: common.simulation_timestep,
    };
    let analyses = backend.install(|| {
        groups
            .into_par_iter()
            .map(|group| {
                let replicas = group
                    .datasets
                    .par_iter()
                    .map(|dataset| analyze_replica_com(dataset, config))
                    .collect::<Vec<_>>();
                let com = average_com_series(&replicas);
                CaseAnalysis {
                    group,
                    com: Some(com),
                    correlation: None,
                    laplace: None,
                    preferred: Vec::new(),
                    fit: None,
                }
            })
            .collect::<Vec<_>>()
    });

    for analysis in &analyses {
        let com = analysis.com.as_ref().expect("no COM");
        eprintln!(
            "[debug:com] case={} replicates={} elapsed_time={:?} x_center={:?} x_velocity={:?}",
            analysis.group.case.case_id,
            com.replicate_count,
            [com.elapsed_time.len()],
            [com.x_center_mean.len()],
            [com.x_velocity_mean.len()]
        );
    }

    let output_root = common
        .output_dir
        .clone()
        .unwrap_or_else(|| common.input_dir[0].join("crystalline_cylinder_analysis_output"));
    let output = output_root.join("com").join("axial_com_velocity.svg");
    assert!(!output.exists() || common.overwrite, "output exists");
    let written = write_com_plot(&analyses, &output);
    eprintln!(
        "[crystalline-cylinder-analysis] command=com cases={} output={}",
        analyses.len(),
        written.display()
    );
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

fn inspect_inputs(input_dirs: &[std::path::PathBuf]) {
    let datasets = discover_datasets(input_dirs);
    let groups = group_replicates(datasets);
    for group in groups {
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
