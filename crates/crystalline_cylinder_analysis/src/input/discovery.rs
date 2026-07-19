//! Input-root scanning and physical-case grouping.

use std::path::PathBuf;

use crate::model::{CaseSchema, DiscoveredDataset, ReplicateGroup};

use super::{load_manifest, validate_manifest};

/// Discover supported complete manifests beneath all input roots.
pub fn discover_datasets(input_roots: &[PathBuf]) -> Vec<DiscoveredDataset> {
    assert!(!input_roots.is_empty(), "no input");
    let mut datasets = Vec::new();
    for input_root in input_roots {
        let input_root = input_root.canonicalize().expect("bad input");
        let safetensors_root = input_root.join("safetensors_output");
        assert!(safetensors_root.is_dir(), "no safetensors");
        let mut case_dirs = std::fs::read_dir(&safetensors_root)
            .expect("read input")
            .map(|entry| entry.expect("read input"))
            .collect::<Vec<_>>();
        case_dirs.sort_by_key(std::fs::DirEntry::file_name);
        for entry in case_dirs {
            if !entry.file_type().expect("read input").is_dir() {
                continue;
            }
            let manifest_path = entry.path().join("manifest.json");
            if !manifest_path.is_file() {
                continue;
            }
            let manifest_path = manifest_path.canonicalize().expect("bad manifest");
            let (schema, manifest) = load_manifest(&manifest_path);
            validate_manifest(&manifest_path, schema, &manifest);
            datasets.push(DiscoveredDataset {
                input_root: input_root.clone(),
                manifest_path,
                schema,
                manifest,
            });
        }
    }
    assert!(!datasets.is_empty(), "no datasets");
    datasets
}

/// Group compatible datasets by schema and case ID without checking seeds.
pub fn group_replicates(mut datasets: Vec<DiscoveredDataset>) -> Vec<ReplicateGroup> {
    datasets.sort_by(|left, right| {
        schema_order(left.schema)
            .cmp(&schema_order(right.schema))
            .then_with(|| left.manifest.case.case_id.cmp(&right.manifest.case.case_id))
            .then_with(|| left.manifest_path.cmp(&right.manifest_path))
    });
    let mut groups: Vec<ReplicateGroup> = Vec::new();
    for dataset in datasets {
        if let Some(group) = groups.last_mut().filter(|group| {
            group.schema == dataset.schema && group.case.case_id == dataset.manifest.case.case_id
        }) {
            validate_compatible(group, &dataset);
            group.datasets.push(dataset);
        } else {
            groups.push(ReplicateGroup {
                schema: dataset.schema,
                case: dataset.manifest.case.clone(),
                datasets: vec![dataset],
            });
        }
    }
    groups.sort_by(|left, right| {
        schema_order(left.schema)
            .cmp(&schema_order(right.schema))
            .then_with(|| compare_circumference(&left.case, &right.case))
            .then_with(|| left.case.geometry_kind.cmp(&right.case.geometry_kind))
            .then_with(|| left.case.lx_multiplier.cmp(&right.case.lx_multiplier))
            .then_with(|| left.case.case_id.cmp(&right.case.case_id))
    });
    groups
}

fn compare_circumference(
    left: &crate::model::CaseMetadata,
    right: &crate::model::CaseMetadata,
) -> std::cmp::Ordering {
    match (left.circumference_diameters, right.circumference_diameters) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn schema_order(schema: CaseSchema) -> u8 {
    match schema {
        CaseSchema::BigLx => 0,
        CaseSchema::Confinement => 1,
    }
}

fn validate_compatible(group: &ReplicateGroup, dataset: &DiscoveredDataset) {
    let expected = &group.case;
    let actual = &dataset.manifest.case;
    let same = expected.lx.to_bits() == actual.lx.to_bits()
        && expected.n_particles == actual.n_particles
        && expected.lx_multiplier == actual.lx_multiplier
        && expected.circumference_diameters.map(f64::to_bits)
            == actual.circumference_diameters.map(f64::to_bits)
        && expected.geometry_kind == actual.geometry_kind;
    let same = same
        && expected.particle_diameter.map(f64::to_bits)
            == actual.particle_diameter.map(f64::to_bits)
        && expected.radius.map(f64::to_bits) == actual.radius.map(f64::to_bits)
        && expected.circumference.map(f64::to_bits) == actual.circumference.map(f64::to_bits)
        && expected.transverse_span.map(f64::to_bits) == actual.transverse_span.map(f64::to_bits);
    assert!(same, "bad replicas");
}
