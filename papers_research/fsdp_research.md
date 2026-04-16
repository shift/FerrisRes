# Research: FSDP — Fully Sharded Data Parallel

## Task ID: 777749d2-219e-4c22-bdf1-fe2d26870b97

## Key Papers & Techniques

### 1. PyTorch FSDP
- **Paper**: Zhao et al. (2023) "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" [arXiv:2304.11277]
- **Core idea**: Shard model parameters, gradients, and optimizer states across data-parallel workers. All-gather parameters for forward, reduce-scatter gradients for backward.
- **Sharding levels**: 
  - FULL_SHARD: shard params + grads + optimizer states (most memory savings)
  - SHARD_GRAD_OP: shard grads + optimizer states only (params replicated)
  - NO_SHARD: equivalent to DDP
- **Wrap policy**: How to partition model into FSDP units. Size-based (params > N) or module-based (per transformer layer).
- **Prefetching**: While computing layer N, all-gather layer N+1's parameters. Overlaps communication with computation.

### 2. FSDP Forward/Backward Flow
```
Forward:
  For each FSDP unit:
    1. All-gather sharded parameters → full parameters
    2. Run forward computation
    3. Discard full parameters (free memory)

Backward:
  For each FSDP unit (reverse order):
    1. All-gather sharded parameters → full parameters
    2. Run backward computation → local gradients
    3. Reduce-scatter gradients → each rank keeps its shard
    4. Discard full parameters and non-sharded gradients
```

### 3. Comparison with Tensor Parallelism
- FSDP: data-parallel, shard across ranks, communicate via all-gather/reduce-scatter
- TP: model-parallel, split weight matrices, communicate via all-reduce after each layer
- Hybrid: FSDP across nodes, TP within node (utilizing NVLink for TP)

### 4. CPU Offloading with FSDP
- Offload optimizer states and sharded parameters to CPU when not in use
- Transfer to GPU only during forward/backward
- Useful when GPU memory is scarce (like Intel iGPU)

### 5. Application to FerrisRes

#### Design: FSDPShardManager
```rust
struct FSDPShardManager {
    world_size: usize,         // Number of data-parallel workers
    rank: usize,               // This worker's rank
    shard_groups: Vec<ShardGroup>, // Groups of layers sharded together
    prefetch_count: usize,     // How many groups ahead to prefetch
}

struct ShardGroup {
    layer_indices: Vec<usize>,
    param_shard: Vec<f32>,     // This rank's parameter shard
    grad_shard: Vec<f32>,      // This rank's gradient shard
    optimizer_shard: OptimizerState, // This rank's optimizer state shard
}

impl FSDPShardManager {
    /// All-gather full parameters for a shard group.
    fn all_gather_params(&self, group_idx: usize) -> Vec<f32> {
        // In single-process: just concatenate all shards
        // In multi-process: MPI/all-gather
        let group = &self.shard_groups[group_idx];
        // ... gather from all ranks
        vec![] // placeholder
    }
    
    /// Reduce-scatter gradients after backward.
    fn reduce_scatter_grads(&mut self, group_idx: usize, full_grads: &[f32]) {
        // Each rank keeps only its shard of the gradient
        let shard_size = full_grads.len() / self.world_size;
        let start = self.rank * shard_size;
        self.shard_groups[group_idx].grad_shard = full_grads[start..start+shard_size].to_vec();
    }
}
```

#### Wrap Policy for FerrisRes
- Each BlockAttnRes block = one FSDP unit (natural boundary)
- Embedding + LM Head = separate FSDP units
- For Intel iGPU: each layer is an FSDP unit with CPU offloading

### 6. Key References
1. [arXiv:2304.11277] Zhao 2023 - PyTorch FSDP
2. Rajbhandari et al. (2020) - ZeRO (FSDP is PyTorch's implementation of ZeRO-3)
3. https://pytorch.org/docs/stable/fsdp.html
