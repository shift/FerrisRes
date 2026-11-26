//! Multi-GPU coordination: process-per-GPU sharding, init identity, local-SGD
//! weight exchange over `/dev/shm`, and a launcher.
//!
//! This module is the coordination layer only. It does not split tensors or
//! pipelines, does not use NCCL or CUDA IPC (wgpu exposes neither), and never
//! shares a device between processes. Each rank owns one adapter and one
//! process; ranks exchange whole weight vectors through files.
//!
//! Nothing here touches a GPU, so every test runs as threads or fake adapters.

pub mod checkpoint;
pub mod error;
pub mod identity;
pub mod shard;

pub use checkpoint::{check_resume_world_size, RunParallelism};
pub use error::MultiGpuError;
pub use identity::{verify_identity, InitIdentity};
pub use shard::ShardPlan;

/// FNV-1a, 64 bit.
///
/// Used for the parameter hash and the weight-file checksum because it is a
/// fixed, dependency-free function: the same bytes produce the same value on
/// every process and every run, which is the whole point of both uses. It is
/// not a cryptographic hash and does not need to be one.
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
