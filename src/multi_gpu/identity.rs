//! Init identity: proof that every rank started from identical weights.
//!
//! The launcher passes one seed to all ranks. Each rank hashes the parameter
//! bytes it actually loaded and reports the pair `(seed, hash)`; rank 0's pair
//! is the reference. Agreement on both fields is the precondition for averaging
//! weights later: averaging ranks that began from different checkpoints
//! produces a model that never existed.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::multi_gpu::error::MultiGpuError;
use crate::multi_gpu::fnv1a;

/// The `(seed, parameter hash)` pair a rank reports before its first step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InitIdentity {
    pub seed: u64,
    pub param_hash: u64,
}

impl InitIdentity {
    /// Hash the parameter bytes as loaded, in little-endian order.
    ///
    /// The byte encoding, not the host's `f32` layout, is hashed, so ranks on
    /// different machines agree. `-0.0` and `0.0` hash differently, which is
    /// intentional: they are different bytes and identical weights never need
    /// to mix the two.
    pub fn from_params(seed: u64, params: &[f32]) -> Self {
        let mut bytes = Vec::with_capacity(params.len() * 4);
        for param in params {
            bytes.extend_from_slice(&param.to_le_bytes());
        }
        Self::from_bytes(seed, &bytes)
    }

    pub fn from_bytes(seed: u64, bytes: &[u8]) -> Self {
        Self {
            seed,
            param_hash: fnv1a(bytes),
        }
    }

    /// Compare one rank's report against this reference.
    ///
    /// The refusal names the rank and the disagreeing field; the caller does
    /// not have to guess which rank diverged.
    pub fn verify_rank(
        &self,
        rank: usize,
        reported: &InitIdentity,
    ) -> Result<(), MultiGpuError> {
        if reported.seed != self.seed {
            return Err(MultiGpuError::InitIdentityMismatch {
                rank,
                field: "seed",
                expected: self.seed.to_string(),
                actual: reported.seed.to_string(),
            });
        }
        if reported.param_hash != self.param_hash {
            return Err(MultiGpuError::InitIdentityMismatch {
                rank,
                field: "param_hash",
                expected: format!("{:#018x}", self.param_hash),
                actual: format!("{:#018x}", reported.param_hash),
            });
        }
        Ok(())
    }
}

/// Build the run-wide reference from the launcher's seed and the parameter
/// hash rank 0 reported.
///
/// `world_size` is checked here so a plan with no ranks cannot acquire an
/// identity; every rank then calls [`InitIdentity::verify_rank`] with the same
/// reference.
pub fn verify_identity(
    seed: u64,
    hash: u64,
    world_size: usize,
) -> Result<InitIdentity, MultiGpuError> {
    if world_size == 0 {
        return Err(MultiGpuError::ZeroWorldSize);
    }
    Ok(InitIdentity {
        seed,
        param_hash: hash,
    })
}

/// The file rank 0 publishes and every rank compares itself against.
///
/// `run_id` scopes the file to one launch (the launcher passes
/// `FERRISRES_RUN_ID`). Without it the name includes the seed and the starting
/// step, so a fresh run and a resumed run never read each other's reference.
/// A stale file is therefore either impossible (fresh directory) or identical
/// in content (same seed, same start, same parameters).
pub fn identity_path(dir: &Path, run_id: Option<&str>, seed: u64, start_step: u64) -> PathBuf {
    let name = match run_id {
        Some(id) => {
            let safe: String = id
                .chars()
                .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            format!("init-identity-{safe}.json")
        }
        None => format!("init-identity-s{start_step}-seed{seed}.json"),
    };
    dir.join(name)
}

/// Publish the run's reference identity, atomically.
///
/// Called by rank 0 before any rank proceeds. The rename keeps a waiting rank
/// from reading half a JSON document.
pub fn publish_reference(
    dir: &Path,
    identity: &InitIdentity,
    run_id: Option<&str>,
    start_step: u64,
) -> Result<PathBuf, MultiGpuError> {
    fs::create_dir_all(dir).map_err(|source| MultiGpuError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    let path = identity_path(dir, run_id, identity.seed, start_step);
    let temp = dir.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("identity.json"),
        std::process::id()
    ));
    let bytes =
        serde_json::to_vec(identity).map_err(|err| MultiGpuError::Json(err.to_string()))?;
    fs::write(&temp, &bytes).map_err(|source| MultiGpuError::Io {
        path: temp.display().to_string(),
        source,
    })?;
    fs::rename(&temp, &path).map_err(|source| MultiGpuError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(path)
}

/// Wait for the reference identity to appear, then return it.
///
/// A missing file is reported as a timeout that names it; it never blocks
/// forever. A partially written file is not possible because the writer
/// renames, but a malformed file is still refused rather than guessed at.
pub fn await_reference(
    dir: &Path,
    run_id: Option<&str>,
    seed: u64,
    start_step: u64,
    timeout: Duration,
) -> Result<InitIdentity, MultiGpuError> {
    let path = identity_path(dir, run_id, seed, start_step);
    let started = Instant::now();
    loop {
        match fs::read(&path) {
            Ok(bytes) => {
                let identity: InitIdentity = serde_json::from_slice(&bytes)
                    .map_err(|err| MultiGpuError::Json(err.to_string()))?;
                return Ok(identity);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(MultiGpuError::Io {
                    path: path.display().to_string(),
                    source,
                })
            }
        }
        if started.elapsed() >= timeout {
            return Err(MultiGpuError::IdentityTimeout {
                path: path.display().to_string(),
                waited_ms: started.elapsed().as_millis(),
            });
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(test)]
mod tests {
    use super::{verify_identity, InitIdentity};
    use crate::multi_gpu::error::MultiGpuError;

    fn params(scale: f32) -> Vec<f32> {
        vec![0.1 * scale, -0.25 * scale, 1.5 * scale, 0.0]
    }

    #[test]
    fn same_seed_and_same_params_hash_identically() {
        let a = InitIdentity::from_params(42, &params(1.0));
        let b = InitIdentity::from_params(42, &params(1.0));
        assert_eq!(a, b);
        a.verify_rank(1, &b).unwrap();
    }

    #[test]
    fn a_different_seed_changes_the_identity() {
        let a = InitIdentity::from_params(1, &params(1.0));
        let b = InitIdentity::from_params(2, &params(1.0));
        assert_ne!(a.seed, b.seed);
        assert_ne!(a, b);
    }

    #[test]
    fn different_weights_change_the_hash() {
        let a = InitIdentity::from_params(7, &params(1.0));
        let b = InitIdentity::from_params(7, &params(1.01));
        assert_ne!(a.param_hash, b.param_hash);
    }

    #[test]
    fn a_mismatched_param_hash_is_refused_by_name() {
        let reference = InitIdentity::from_params(7, &params(1.0));
        let reported = InitIdentity::from_params(7, &params(2.0));
        let err = reference.verify_rank(1, &reported).unwrap_err();
        match err {
            MultiGpuError::InitIdentityMismatch { rank, field, .. } => {
                assert_eq!(rank, 1);
                assert_eq!(field, "param_hash");
            }
            other => panic!("wrong refusal: {other}"),
        }
    }

    #[test]
    fn a_mismatched_seed_is_refused_by_name() {
        let reference = InitIdentity::from_params(7, &params(1.0));
        let reported = InitIdentity::from_params(8, &params(1.0));
        let err = reference.verify_rank(0, &reported).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("seed"), "must name the field: {text}");
        assert!(text.contains("rank 0"), "must name the rank: {text}");
    }

    #[test]
    fn a_published_reference_is_read_back_verbatim() {
        let dir = std::env::temp_dir().join(format!(
            "ferrisres-init-identity-{}-publish",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let reference = InitIdentity::from_params(42, &params(1.0));
        let path = super::publish_reference(&dir, &reference, Some("run-7"), 0).unwrap();
        assert!(path.is_file());

        let read = super::await_reference(
            &dir,
            Some("run-7"),
            42,
            0,
            std::time::Duration::from_millis(100),
        )
        .unwrap();
        assert_eq!(read, reference);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_reference_times_out_with_its_path() {
        let dir = std::env::temp_dir().join(format!(
            "ferrisres-init-identity-{}-missing",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let err = super::await_reference(
            &dir,
            Some("run-8"),
            1,
            0,
            std::time::Duration::from_millis(20),
        )
        .unwrap_err();
        let text = err.to_string();
        assert!(text.contains("init-identity-run-8.json"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_identity_builds_the_reference_and_rejects_zero_world() {
        let reference = verify_identity(9, 0x1234, 2).unwrap();
        assert_eq!(reference.seed, 9);
        assert_eq!(reference.param_hash, 0x1234);
        assert!(matches!(
            verify_identity(9, 0x1234, 0),
            Err(MultiGpuError::ZeroWorldSize)
        ));
    }
}
