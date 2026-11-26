//! Checkpoint parallelism metadata.
//!
//! A checkpoint written by a `world_size`-way run records its world size and
//! rank. Resuming that checkpoint at a different world size changes how many
//! batches each step sees, so the effective learning rate per optimizer step
//! changes too. This module refuses the resume instead of choosing a scale
//! factor silently.

use serde::{Deserialize, Serialize};

use crate::multi_gpu::error::MultiGpuError;

/// The parallelism recorded in a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunParallelism {
    pub world_size: usize,
    pub rank: usize,
}

impl RunParallelism {
    pub fn new(world_size: usize, rank: usize) -> Result<Self, MultiGpuError> {
        if world_size == 0 {
            return Err(MultiGpuError::ZeroWorldSize);
        }
        if rank >= world_size {
            return Err(MultiGpuError::RankOutOfRange { rank, world_size });
        }
        Ok(Self { world_size, rank })
    }

    /// Refuse a resume whose world size differs from the checkpoint's.
    ///
    /// Scaling the learning rate instead would have to be an explicit,
    /// documented decision; nothing here guesses one.
    pub fn check_resume(&self, requested_world_size: usize) -> Result<(), MultiGpuError> {
        check_resume_world_size(self.world_size, requested_world_size)
    }
}

/// Compare a checkpoint's recorded world size with the requested one.
pub fn check_resume_world_size(
    recorded: usize,
    requested: usize,
) -> Result<(), MultiGpuError> {
    if recorded == requested {
        Ok(())
    } else {
        Err(MultiGpuError::WorldSizeChanged {
            recorded,
            requested,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{check_resume_world_size, RunParallelism};
    use crate::multi_gpu::error::MultiGpuError;

    #[test]
    fn a_matching_world_size_resumes() {
        let recorded = RunParallelism::new(2, 1).unwrap();
        recorded.check_resume(2).unwrap();
        check_resume_world_size(2, 2).unwrap();
    }

    #[test]
    fn a_different_world_size_is_refused_by_name() {
        let recorded = RunParallelism::new(2, 0).unwrap();
        let err = recorded.check_resume(1).unwrap_err();
        match err {
            MultiGpuError::WorldSizeChanged {
                recorded,
                requested,
            } => {
                assert_eq!(recorded, 2);
                assert_eq!(requested, 1);
            }
            other => panic!("wrong refusal: {other}"),
        }
        let text = recorded.check_resume(4).unwrap_err().to_string();
        assert!(text.contains("world_size=2"), "must name the record: {text}");
        assert!(text.contains("world_size=4"), "must name the request: {text}");
        assert!(text.contains("refusing"), "must say it refused: {text}");
    }

    #[test]
    fn metadata_round_trips_through_json() {
        let recorded = RunParallelism::new(2, 1).unwrap();
        let json = serde_json::to_string(&recorded).unwrap();
        let back: RunParallelism = serde_json::from_str(&json).unwrap();
        assert_eq!(recorded, back);
        back.check_resume(2).unwrap();
    }

    #[test]
    fn invalid_metadata_is_refused() {
        assert!(matches!(
            RunParallelism::new(0, 0),
            Err(MultiGpuError::ZeroWorldSize)
        ));
        assert!(matches!(
            RunParallelism::new(2, 2),
            Err(MultiGpuError::RankOutOfRange { .. })
        ));
    }
}
