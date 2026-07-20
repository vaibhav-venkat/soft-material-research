//! Atomic JSON and safetensor output declarations.

use std::path::{Path, PathBuf};

use crystalline_cylinder_analysis::pipeline::CaseAnalysis;
use crystalline_cylinder_analysis::{
    ClusterConfig, ClusterKind, ClusterRecord, DatasetClusterAnalysis,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// Versioned provenance record written beside every result set.
#[derive(Clone, Debug, Serialize)]
pub struct OutputManifest {
    pub schema: String,
    pub command: String,
    pub input_dirs: Vec<PathBuf>,
    pub timestep: f64,
    pub cases: Vec<OutputCaseRecord>,
}

/// Input and replicate provenance for one physical case.
#[derive(Clone, Debug, Serialize)]
pub struct OutputCaseRecord {
    pub case_id: String,
    pub label: String,
    pub replicate_count: usize,
    pub input_manifests: Vec<PathBuf>,
}

/// Prepare an empty output root or replace it with explicit authorization.
pub fn prepare_output_dir(path: &Path, overwrite: bool) {
    if path.exists() {
        assert!(overwrite, "output exists; use --overwrite");
        std::fs::remove_dir_all(path).expect("remove old output");
    }
    std::fs::create_dir_all(path).expect("create output directory");
}

/// Write a serializable value through a temporary file and atomic rename.
pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> PathBuf {
    let parent = path.parent().expect("output has no parent");
    std::fs::create_dir_all(parent).expect("create output parent");
    let temporary = temporary_path(path);
    let contents = serde_json::to_vec_pretty(value).expect("serialize JSON");
    std::fs::write(&temporary, contents).expect("write JSON");
    std::fs::rename(&temporary, path).expect("publish JSON");
    path.to_path_buf()
}

/// Per-replicate provenance for compact frame-local cluster shards.
#[derive(Clone, Debug, Serialize)]
pub struct ClusterDatasetManifest {
    pub schema: String,
    pub case_id: String,
    pub input_manifest: PathBuf,
    pub frame_count: usize,
    pub particle_count: usize,
    pub config: ClusterConfig,
    pub target_shard_mib: usize,
    pub structural_cluster_count: usize,
    pub motion_cluster_count: usize,
    pub structural_files: Vec<PathBuf>,
    pub motion_files: Vec<PathBuf>,
}

/// Write compact CSR member lists, splitting only between complete clusters.
pub fn write_cluster_dataset(
    output_dir: &Path,
    input_manifest: &Path,
    analysis: &DatasetClusterAnalysis,
    config: ClusterConfig,
    target_shard_mib: usize,
) -> ClusterDatasetManifest {
    assert!(
        target_shard_mib > 0,
        "cluster shard target must be positive"
    );
    std::fs::create_dir_all(output_dir).expect("create cluster dataset output");
    let target_bytes = target_shard_mib * 1024 * 1024;
    let structural_files = write_cluster_kind(
        output_dir,
        "structural",
        ClusterKind::Structural,
        &analysis.structural,
        target_bytes,
    );
    let motion_files = write_cluster_kind(
        output_dir,
        "motion",
        ClusterKind::Motion,
        &analysis.motion,
        target_bytes,
    );
    let manifest = ClusterDatasetManifest {
        schema: "crystalline-cylinder-analysis.clusters.dataset.v3".to_owned(),
        case_id: analysis.case_id.clone(),
        input_manifest: input_manifest.to_path_buf(),
        frame_count: analysis.frame_count,
        particle_count: analysis.particle_count,
        config,
        target_shard_mib,
        structural_cluster_count: analysis.structural.len(),
        motion_cluster_count: analysis.motion.len(),
        structural_files,
        motion_files,
    };
    write_json_atomic(&output_dir.join("manifest.json"), &manifest);
    manifest
}

fn write_cluster_kind(
    output_dir: &Path,
    prefix: &str,
    kind: ClusterKind,
    records: &[ClusterRecord],
    target_bytes: usize,
) -> Vec<PathBuf> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < records.len() {
        let mut stop = start;
        let mut bytes = 0;
        while stop < records.len() {
            let record_bytes = 96 + records[stop].members.len() * size_of::<i64>();
            if stop > start && bytes + record_bytes > target_bytes {
                break;
            }
            bytes += record_bytes;
            stop += 1;
        }
        ranges.push(start..stop);
        start = stop;
    }
    if records.is_empty() {
        ranges.push(0..0);
    }
    ranges
        .into_iter()
        .enumerate()
        .map(|(index, range)| {
            let path = output_dir.join(format!("{prefix}_{:06}.safetensors", index + 1));
            write_cluster_shard(&path, kind, &records[range]);
            path
        })
        .collect()
}

fn write_cluster_shard(path: &Path, kind: ClusterKind, records: &[ClusterRecord]) {
    let count = records.len();
    let frame_index = records
        .iter()
        .map(|record| as_i64(record.frame_index))
        .collect::<Vec<_>>();
    let step = records.iter().map(|record| record.step).collect::<Vec<_>>();
    let destination_frame_index = records
        .iter()
        .map(|record| record.destination_frame_index.map_or(-1, as_i64))
        .collect::<Vec<_>>();
    let destination_step = records
        .iter()
        .map(|record| record.destination_step.unwrap_or(-1))
        .collect::<Vec<_>>();
    let local_id = records
        .iter()
        .map(|record| as_i64(record.local_id))
        .collect::<Vec<_>>();
    let domain_id = records
        .iter()
        .map(|record| i64::from(record.domain_id))
        .collect::<Vec<_>>();
    let centroid = records
        .iter()
        .flat_map(|record| record.centroid)
        .collect::<Vec<_>>();
    let particle_count = records
        .iter()
        .map(|record| as_i64(record.particle_count))
        .collect::<Vec<_>>();
    let occupied_area = records
        .iter()
        .map(|record| record.occupied_area)
        .collect::<Vec<_>>();
    let cylinder_surface_area = records
        .iter()
        .map(|record| record.cylinder_surface_area)
        .collect::<Vec<_>>();
    let equivalent_perimeter = records
        .iter()
        .map(|record| record.equivalent_perimeter)
        .collect::<Vec<_>>();
    let normalized_area = records
        .iter()
        .map(|record| record.normalized_area)
        .collect::<Vec<_>>();
    let mut member_offsets = Vec::with_capacity(count + 1);
    let mut member_particle_ids = Vec::new();
    member_offsets.push(0_i64);
    for record in records {
        member_particle_ids.extend(
            record
                .members
                .iter()
                .map(|&member| i64::try_from(member).expect("particle ID exceeds I64")),
        );
        member_offsets.push(as_i64(member_particle_ids.len()));
    }

    let mut tensors = BTreeMap::new();
    insert_i64(&mut tensors, "frame_index", &[count], &frame_index);
    insert_i64(&mut tensors, "step", &[count], &step);
    insert_i64(
        &mut tensors,
        "destination_frame_index",
        &[count],
        &destination_frame_index,
    );
    insert_i64(
        &mut tensors,
        "destination_step",
        &[count],
        &destination_step,
    );
    insert_i64(&mut tensors, "local_id", &[count], &local_id);
    insert_i64(&mut tensors, "domain_id", &[count], &domain_id);
    insert_f64(&mut tensors, "centroid", &[count, 2], &centroid);
    insert_i64(&mut tensors, "particle_count", &[count], &particle_count);
    insert_f64(&mut tensors, "occupied_area", &[count], &occupied_area);
    insert_f64(
        &mut tensors,
        "cylinder_surface_area",
        &[count],
        &cylinder_surface_area,
    );
    insert_f64(
        &mut tensors,
        "equivalent_perimeter",
        &[count],
        &equivalent_perimeter,
    );
    insert_f64(&mut tensors, "normalized_area", &[count], &normalized_area);
    insert_i64(
        &mut tensors,
        "member_offsets",
        &[member_offsets.len()],
        &member_offsets,
    );
    insert_i64(
        &mut tensors,
        "member_particle_ids",
        &[member_particle_ids.len()],
        &member_particle_ids,
    );
    let temporary = temporary_path(path);
    let metadata = HashMap::from([
        (
            "schema".to_owned(),
            "crystalline-cylinder-analysis.clusters.shard.v2".to_owned(),
        ),
        ("kind".to_owned(), format!("{kind:?}").to_lowercase()),
    ]);
    serialize_to_file(tensors, Some(metadata), &temporary).expect("write cluster safetensors");
    std::fs::rename(&temporary, path).expect("publish cluster safetensors");
}

fn insert_i64<'a>(
    tensors: &mut BTreeMap<&'static str, TensorView<'a>>,
    name: &'static str,
    shape: &[usize],
    values: &'a [i64],
) {
    tensors.insert(
        name,
        TensorView::new(Dtype::I64, shape.to_vec(), bytemuck::cast_slice(values))
            .expect("create I64 tensor"),
    );
}

fn insert_f64<'a>(
    tensors: &mut BTreeMap<&'static str, TensorView<'a>>,
    name: &'static str,
    shape: &[usize],
    values: &'a [f64],
) {
    tensors.insert(
        name,
        TensorView::new(Dtype::F64, shape.to_vec(), bytemuck::cast_slice(values))
            .expect("create F64 tensor"),
    );
}

fn as_i64(value: usize) -> i64 {
    i64::try_from(value).expect("value exceeds I64")
}

fn temporary_path(path: &Path) -> PathBuf {
    let parent = path.parent().expect("output has no parent");
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .expect("bad output name");
    parent.join(format!(".{file_name}.{}.tmp", std::process::id()))
}

/// Write per-case numerical arrays as a versioned safetensor artifact.
pub fn write_case_safetensors(_output_dir: &Path, _analysis: &CaseAnalysis) -> PathBuf {
    todo!("encode result vectors without Python or ndarray dependencies")
}

/// Build provenance records from completed case analyses.
pub fn output_manifest(
    _command: &str,
    _input_dirs: &[PathBuf],
    _timestep: f64,
    _analyses: &[CaseAnalysis],
) -> OutputManifest {
    todo!("collect case metadata and all contributing manifest paths")
}
