//! Init identity: proof that every rank started from identical weights.
//!
//! The launcher passes one seed to all ranks. Each rank hashes the parameter
//! bytes it actually loaded and reports the pair `(seed, hash)`; rank 0's pair
//! is the reference. Agreement on both fields is the precondition for averaging
//! weights later: averaging ranks that began from different checkpoints
//! produces a model that never existed.

use crate::multi_gpu::error::MultiGpuError;
use crate::multi_gpu::fnv1a;

/// The `(seed, parameter hash)` pair a rank reports before its first step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
