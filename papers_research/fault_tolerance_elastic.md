# Research: Fault Tolerance & Elastic Training

## Task ID: 2a6c76f0-18ca-4616-acf2-e182539bb9a1

## Key Papers & Techniques

### 1. TorchElastic / PyTorch Elastic Training
- **Framework**: PyTorch's elastic training framework (now part of torchrun)
- **Core concepts**:
  - **Rendezvous**: Distributed barrier where workers agree on membership
  - **State dict**: Checkpoint containing model params, optimizer state, RNG state, epoch/step counter
  - **Restart policy**: On failure, restart from last checkpoint with new membership
- **Application to FerrisRes**: Design checkpoint format that includes all distributed state. On rank failure, remaining ranks re-rendezvous and continue.

### 2. Megatron-LM Checkpoint Format
- **Paper**: Shoeybi et al. (2019) "Megatron-LM: Training Multi-Billion Parameter Language Models" [arXiv:1909.08053]
- **Checkpoint structure**: Each rank saves its own shard (parameters, optimizer state, RNG state). On restart, each rank loads its shard.
- **Resharding**: When membership changes (e.g., 4 GPUs → 3 GPUs), redistribute parameter shards across new rank count.

### 3. Asynchronous Checkpointing
- **Paper**: Mohan et al. (2022) "Non-volatile Memory for Checkpointing" techniques
- **Core idea**: Don't block training for checkpoints. Use copy-on-write: fork the process/memory, let the fork write to disk while training continues.
- **Application**: During distillation, save checkpoints in a background thread. Don't block the training loop.

### 4. Heartbeat Protocol
- Each worker sends heartbeat to coordinator every N seconds
- If heartbeat missed for M intervals → mark worker as failed
- Coordinator triggers redistribution of failed worker's shards

### 5. ElasticTrainer Design for FerrisRes

```rust
struct ElasticTrainer {
    rank: usize,
    world_size: usize,
    coordinator: Option<Coordinator>,
    health_monitor: HealthMonitor,
    checkpoint_manager: ElasticCheckpointManager,
}

struct Coordinator {
    workers: HashMap<usize, WorkerState>,
    heartbeat_interval_ms: u64,
    failure_timeout_ms: u64,
}

struct WorkerState {
    rank: usize,
    last_heartbeat: u64,
    status: WorkerStatus,
    assigned_shards: Vec<usize>,
}

enum WorkerStatus {
    Healthy,
    Suspect,     // Missed 1 heartbeat
    Failed,      // Missed N heartbeats
    Recovering,  // Rejoining after failure
}

struct ElasticCheckpointManager {
    checkpoint_dir: PathBuf,
    checkpoint_interval: usize,  // Steps between checkpoints
    async_checkpoint: bool,
}

impl ElasticCheckpointManager {
    /// Save checkpoint with all state needed for restart.
    fn save(&self, state: &TrainingState) -> Result<()> {
        // Atomic write: temp file + rename
        let temp = format!("{}.tmp", self.checkpoint_path());
        let data = serialize(state)?;
        std::fs::write(&temp, data)?;
        std::fs::rename(&temp, self.checkpoint_path())?;
        Ok(())
    }
    
    /// Load checkpoint and reshard if world size changed.
    fn load(&self, new_world_size: usize) -> Result<TrainingState> {
        let state = self.load_raw()?;
        if state.world_size != new_world_size {
            // Reshard: redistribute parameters/optimizer across new rank count
            state.reshard(new_world_size)
        } else {
            Ok(state)
        }
    }
}
```

### 6. Recovery Strategies

1. **Restart from checkpoint**: Simplest. All ranks roll back to last checkpoint. Wastes compute since checkpoint.
2. **Continue with reduced parallelism**: Failed rank's work redistributed among survivors. Slower but doesn't lose progress.
3. **Replace failed worker**: Spin up new worker, transfer state from checkpoint, continue. Requires spare capacity.

### Key References
1. PyTorch TorchElastic documentation
2. [arXiv:1909.08053] Shoeybi 2019 - Megatron-LM
3. Ray Train fault tolerance design
