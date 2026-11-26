//! Named refusals for the multi-GPU coordination layer.
//!
//! Every failure that a launcher or a rank must distinguish by name (capacity,
//! identity, barrier, checksum, resume) gets a variant here instead of a bare
//! `Device(String)`, so the operator reads the cause from the error alone.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MultiGpuError {
    #[error("invalid shard plan: world size must be greater than zero")]
    ZeroWorldSize,

    #[error("invalid shard plan: rank {rank} is not in 0..{world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },

    #[error(
        "init identity mismatch on rank {rank}: {field} expected {expected}, got {actual}; \
         every rank must start from identical weights"
    )]
    InitIdentityMismatch {
        rank: usize,
        field: &'static str,
        expected: String,
        actual: String,
    },

    #[error(
        "weight exchange timed out after {waited_ms} ms at step {step}; \
         missing rank(s): {missing:?}"
    )]
    BarrierTimeout {
        step: u64,
        waited_ms: u128,
        missing: Vec<usize>,
    },

    #[error("weight average for step {step} did not appear within {waited_ms} ms")]
    AverageTimeout { step: u64, waited_ms: u128 },

    #[error("reference init identity {path} did not appear within {waited_ms} ms")]
    IdentityTimeout { path: String, waited_ms: u128 },

    #[error("weight averaging is rank 0's job; this process is rank {rank}")]
    NotCoordinator { rank: usize },

    #[error("weight file {path} has bad magic (not a FerrisRes shard file)")]
    BadMagic { path: String },

    #[error("weight file {path} is {len} bytes, shorter than the {header} byte header")]
    ShortFile {
        path: String,
        len: u64,
        header: usize,
    },

    #[error("weight file {path}: header {field} is {actual}, expected {expected}")]
    HeaderMismatch {
        path: String,
        field: &'static str,
        expected: String,
        actual: String,
    },

    #[error(
        "weight file {path}: checksum {actual:#018x} does not match the header's {expected:#018x}"
    )]
    ChecksumMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },

    #[error("weight file {path} holds {actual} elements, expected {expected}")]
    ElementCountMismatch {
        path: String,
        expected: u64,
        actual: u64,
    },

    #[error("I/O error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("weight exchange: could not serialize JSON: {0}")]
    Json(String),

    #[error(
        "world size {requested} requested but only {available} usable adapter(s): [{adapters}]"
    )]
    NotEnoughAdapters {
        requested: usize,
        available: usize,
        adapters: String,
    },

    #[error("adapter enumeration failed: {0}")]
    Enumeration(String),

    #[error("launch command is empty; expected `-- <program> [args...]`")]
    EmptyCommand,

    #[error("rank {rank} could not start {program}: {source}")]
    SpawnFailed {
        rank: usize,
        program: String,
        #[source]
        source: std::io::Error,
    },

    #[error(
        "rank {rank} exited with {status}; terminated rank(s) {killed:?}; \
         evidence in {evidence_dir}"
    )]
    WorkerFailed {
        rank: usize,
        status: String,
        killed: Vec<usize>,
        evidence_dir: String,
    },

    #[error(
        "launch timed out after {waited_ms} ms; still running rank(s): {running:?}; \
         terminated them; evidence in {evidence_dir}"
    )]
    LaunchTimeout {
        waited_ms: u128,
        running: Vec<usize>,
        evidence_dir: String,
    },

    #[error(
        "checkpoint records world_size={recorded} but the resume requested \
         world_size={requested}; refusing instead of silently changing the optimisation"
    )]
    WorldSizeChanged {
        recorded: usize,
        requested: usize,
    },
}
