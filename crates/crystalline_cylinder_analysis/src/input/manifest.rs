//! Strict manifest parsing and shard-index validation.

use std::path::{Component, Path, PathBuf};

use crate::model::{AnalysisManifest, CaseSchema};

/// Parse a manifest and classify its supported producer schema.
pub fn load_manifest(path: &Path) -> (CaseSchema, AnalysisManifest) {
    let contents = std::fs::read(path).expect("read manifest");
    let manifest: AnalysisManifest = serde_json::from_slice(&contents).expect("bad manifest");
    let schema = match manifest.schema.as_str() {
        "hexatic.big_lx.analysis.v1" => CaseSchema::BigLx,
        "hexatic.confinement_comparison.analysis.v1" => CaseSchema::Confinement,
        _ => panic!("bad schema"),
    };
    (schema, manifest)
}

/// Validate completeness, case fields, contained paths, and contiguous shards.
pub fn validate_manifest(path: &Path, schema: CaseSchema, manifest: &AnalysisManifest) {
    let expected_schema = match schema {
        CaseSchema::BigLx => "hexatic.big_lx.analysis.v1",
        CaseSchema::Confinement => "hexatic.confinement_comparison.analysis.v1",
    };
    assert_eq!(manifest.schema, expected_schema, "bad schema");
    assert!(manifest.complete, "incomplete");
    assert!(!manifest.case.case_id.trim().is_empty(), "bad case");
    assert!(
        manifest.case.lx.is_finite() && manifest.case.lx > 0.0,
        "bad lx"
    );
    assert_ne!(manifest.case.n_particles, 0, "bad particles");
    assert!(manifest.case.lx_multiplier > 0, "bad multiplier");
    for value in [
        manifest.case.particle_diameter,
        manifest.case.radius,
        manifest.case.circumference,
        manifest.case.transverse_span,
    ]
    .into_iter()
    .flatten()
    {
        assert!(value.is_finite() && value > 0.0, "bad case geometry");
    }
    assert!(!manifest.shards.is_empty(), "no shards");

    let mut expected_start = 0;
    let mut previous_step = None;
    for shard in &manifest.shards {
        assert_eq!(shard.frame_start, expected_start, "bad shard range");
        assert!(shard.frame_stop > shard.frame_start, "bad shard range");
        let shard_frames = shard.frame_stop - shard.frame_start;
        assert!(
            shard.steps.is_empty() || shard.steps.len() == shard_frames,
            "bad steps"
        );
        for &step in &shard.steps {
            assert!(
                previous_step.is_none_or(|previous| step > previous),
                "bad steps"
            );
            previous_step = Some(step);
        }
        let _shard_path = resolve_shard_path(path, &shard.file);
        assert!(shard.bytes.is_none_or(|bytes| bytes > 0), "bad bytes");
        expected_start = shard.frame_stop;
    }
    assert_eq!(manifest.frame_count, expected_start, "bad frame count");
}

pub fn resolve_shard_path(manifest_path: &Path, file: &str) -> PathBuf {
    let relative = Path::new(file);
    assert!(!relative.is_absolute(), "bad shard path");
    assert!(
        !relative
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::RootDir)),
        "bad shard path"
    );
    let root = manifest_path.parent().expect("bad manifest path");
    let candidate = root.join(relative);
    let resolved = candidate.canonicalize().expect("missing shard");
    let resolved_root = root.canonicalize().expect("bad root");
    assert!(resolved.starts_with(&resolved_root), "bad shard path");
    assert!(resolved.is_file(), "missing shard");
    resolved
}
