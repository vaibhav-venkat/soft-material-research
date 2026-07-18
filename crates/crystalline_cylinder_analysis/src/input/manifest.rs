//! Strict manifest parsing and shard-index validation.

use std::path::{Component, Path, PathBuf};

use crate::error::AnalysisResult;
use crate::model::{AnalysisManifest, CaseSchema};

/// Parse a manifest and classify its supported producer schema.
pub fn load_manifest(path: &Path) -> AnalysisResult<(CaseSchema, AnalysisManifest)> {
    let contents = std::fs::read(path)?;
    let manifest: AnalysisManifest = serde_json::from_slice(&contents).map_err(|error| {
        crate::error::AnalysisError::InvalidManifest {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    })?;
    let schema = match manifest.schema.as_str() {
        "hexatic.big_lx.analysis.v1" => CaseSchema::BigLx,
        "hexatic.confinement_comparison.analysis.v1" => CaseSchema::Confinement,
        schema => {
            return Err(crate::error::AnalysisError::InvalidManifest {
                path: path.to_path_buf(),
                message: format!("unsupported schema {schema:?}"),
            });
        }
    };
    Ok((schema, manifest))
}

/// Validate completeness, case fields, contained paths, and contiguous shards.
pub fn validate_manifest(
    path: &Path,
    schema: CaseSchema,
    manifest: &AnalysisManifest,
) -> AnalysisResult<()> {
    let invalid = |message: String| crate::error::AnalysisError::InvalidManifest {
        path: path.to_path_buf(),
        message,
    };
    let expected_schema = match schema {
        CaseSchema::BigLx => "hexatic.big_lx.analysis.v1",
        CaseSchema::Confinement => "hexatic.confinement_comparison.analysis.v1",
    };
    assert_eq!(manifest.schema, expected_schema);
    if !manifest.complete {
        return Err(invalid("analysis is not marked complete".to_owned()));
    }
    if manifest.case.case_id.trim().is_empty() {
        return Err(invalid("case_id must not be empty".to_owned()));
    }
    if !manifest.case.lx.is_finite() || manifest.case.lx <= 0.0 {
        return Err(invalid("case lx must be finite and positive".to_owned()));
    }
    if manifest.case.n_particles == 0 {
        return Err(invalid("case n_particles must be positive".to_owned()));
    }
    if manifest.case.lx_multiplier < 1 {
        return Err(invalid("case lx_multiplier must be positive".to_owned()));
    }
    if manifest.shards.is_empty() {
        return Err(invalid("manifest contains no frame shards".to_owned()));
    }

    let mut expected_start = 0;
    let mut previous_step = None;
    for shard in &manifest.shards {
        if shard.frame_start != expected_start || shard.frame_stop <= shard.frame_start {
            return Err(invalid(format!(
                "shards are not contiguous at frame {}",
                shard.frame_start
            )));
        }
        let shard_frames = shard.frame_stop - shard.frame_start;
        if !shard.steps.is_empty() && shard.steps.len() != shard_frames {
            return Err(invalid(format!(
                "shard {:?} declares {} frames but {} steps",
                shard.file,
                shard_frames,
                shard.steps.len()
            )));
        }
        for &step in &shard.steps {
            if previous_step.is_some_and(|previous| step <= previous) {
                return Err(invalid(
                    "manifest steps must be globally increasing".to_owned(),
                ));
            }
            previous_step = Some(step);
        }
        let _shard_path = resolve_shard_path(path, &shard.file)?;
        if shard.bytes.is_some_and(|bytes| bytes == 0) {
            return Err(invalid(format!(
                "shard {:?} declares an empty tensor payload",
                shard.file
            )));
        }
        expected_start = shard.frame_stop;
    }
    if manifest.frame_count != expected_start {
        return Err(invalid(format!(
            "frame_count {} does not match shard stop {expected_start}",
            manifest.frame_count
        )));
    }
    Ok(())
}

pub fn resolve_shard_path(manifest_path: &Path, file: &str) -> AnalysisResult<PathBuf> {
    let relative = Path::new(file);
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::RootDir))
    {
        return Err(crate::error::AnalysisError::InvalidManifest {
            path: manifest_path.to_path_buf(),
            message: format!("shard path escapes the dataset directory: {file:?}"),
        });
    }
    let root =
        manifest_path
            .parent()
            .ok_or_else(|| crate::error::AnalysisError::InvalidManifest {
                path: manifest_path.to_path_buf(),
                message: "manifest has no parent directory".to_owned(),
            })?;
    let candidate = root.join(relative);
    let resolved =
        candidate
            .canonicalize()
            .map_err(|error| crate::error::AnalysisError::InvalidManifest {
                path: manifest_path.to_path_buf(),
                message: format!("cannot resolve shard {file:?}: {error}"),
            })?;
    let resolved_root = root.canonicalize()?;
    if !resolved.starts_with(&resolved_root) || !resolved.is_file() {
        return Err(crate::error::AnalysisError::InvalidManifest {
            path: manifest_path.to_path_buf(),
            message: format!("shard is not a contained file: {file:?}"),
        });
    }
    Ok(resolved)
}
