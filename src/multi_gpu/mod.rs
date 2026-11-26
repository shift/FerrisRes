//! Multi-GPU coordination: process-per-GPU sharding, init identity, local-SGD
//! weight exchange over `/dev/shm`, and a launcher.
//!
//! This module is the coordination layer only. It does not split tensors or
//! pipelines, does not use NCCL or CUDA IPC (wgpu exposes neither), and never
//! shares a device between processes. Each rank owns one adapter and one
//! process; ranks exchange whole weight vectors through files.
//!
//! The pieces are deliberately independent:
//!
//! - [`shard`] decides which batch a `(rank, global step)` pair owns.
//! - [`identity`] proves every rank loaded the same initial weights.
//! - [`exchange`] publishes weights, waits with a deadline, and averages them.
//! - [`launcher`] discovers adapters, spawns one child per adapter, and records
//!   per-rank evidence.
//! - [`checkpoint`] refuses a resume at a different world size.
//!
//! Nothing here touches a GPU, so every test runs as threads or fake adapters.
//! The only GPU-facing call is [`launcher::run_workers`], used by the
//! `ferrisres workers` subcommand and covered by an ignored host test.

pub mod checkpoint;
pub mod error;
pub mod exchange;
pub mod identity;
pub mod launcher;
pub mod shard;

pub use checkpoint::{check_resume_world_size, RunParallelism};
pub use error::MultiGpuError;
pub use exchange::{WeightExchange, DEFAULT_EXCHANGE_TIMEOUT};
pub use identity::{verify_identity, InitIdentity};
pub use launcher::{
    check_capacity, launch, usable_adapters, AdapterSlot, LaunchOptions, LaunchReport,
    WorkerEvidence,
};
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
