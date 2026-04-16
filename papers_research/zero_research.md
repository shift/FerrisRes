# Research: ZeRO-1/2/3 — Optimizer State / Gradient / Parameter Partitioning

## Task ID: d5359c6e-f8b7-4d72-9dd4-6edcef507651

## Key Papers & Techniques

### 1. ZeRO (Zero Redundancy Optimizer)
- **Paper**: Rajbhandari et al. (2020) "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" [arXiv:1910.02054]
- **Problem**: Standard data parallelism replicates model states across all GPUs. For a 7B model with Adam, that's ~112MB parameters + ~224MB optimizer states × N GPUs.
- **Three stages of progressive memory reduction**:

### Stage 1: Optimizer State Partitioning
- Each rank maintains only 1/Nd of optimizer states (m, v for Adam)
- Memory reduction: 4× (from 16Ψ to 4Ψ + 12Ψ/Nd where Ψ = parameter count)
- Communication: same as DP (all-reduce gradients, but each rank only updates its shard)
- Example: 7B model, 4 GPUs → each GPU holds 1.75B optimizer states

### Stage 2: Gradient Partitioning
- Stage 1 + each rank maintains only 1/Nd of gradients
- Memory reduction: 8× (from 16Ψ to 2Ψ + 14Ψ/Nd)
- Communication: reduce-scatter instead of all-reduce
- Gradient accumulation works on local shard only

### Stage 3: Parameter Partitioning
- Stage 2 + each rank maintains only 1/Nd of parameters
- Memory reduction: Nd× (from 16Ψ to 16Ψ/Nd)
- Communication: all-gather for forward, reduce-scatter for backward
- Requires parameter prefetching to overlap communication

### 2. ZeRO-Infinity (Offload to NVMe)
- **Paper**: Rajbhandari et al. (2021) "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" [arXiv:2104.07857]
- **Core idea**: Offload optimizer states, gradients, AND parameters to NVMe/CPU. Use pinned memory for fast GPU↔CPU transfer.
- **Infinity offload engine**: Bandwidth-optimized PCIe transfer, overlapping computation with transfer.
- **Application to FerrisRes**: Critical for Intel iGPU. Offload optimizer states to system RAM, parameters to NVMe if needed. Keep only active layer on GPU.

### 3. Implementation Design for FerrisRes

```rust
#[derive(Clone, Copy)]
enum ZeroStage {
    None,      // Standard DP
    Stage1,    // Partition optimizer states
    Stage2,    // + Partition gradients  
    Stage3,    // + Partition parameters
}

struct ZeroOptimizer {
    stage: ZeroStage,
    world_size: usize,
    rank: usize,
    
    // Stage 1+: Sharded Adam states
    m_shard: Vec<f32>,  // First moment (this rank's shard)
    v_shard: Vec<f32>,  // Second moment (this rank's shard)
    step_count: u64,
    
    // Stage 2+: Sharded gradients
    grad_shard: Vec<f32>,
    
    // Stage 3: Sharded parameters (full params gathered on demand)
    param_shard: Vec<f32>,
    
    // CPU offload for optimizer states
    cpu_offload: bool,
}

impl ZeroOptimizer {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        match self.stage {
            ZeroStage::None => {
                // Standard Adam on all parameters
                self.adam_step(params, grads);
            }
            ZeroStage::Stage1 => {
                // Reduce-scatter gradients
                let local_grads = self.reduce_scatter(grads);
                // Adam update only on our shard
                let shard_start = self.rank * self.m_shard.len();
                let shard_end = shard_start + self.m_shard.len();
                self.adam_step_shard(
                    &mut params[shard_start..shard_end],
                    &local_grads,
                );
                // All-gather updated params
                self.all_gather_params(params);
            }
            ZeroStage::Stage2 => {
                // Reduce-scatter into local grad shard
                self.grad_shard = self.reduce_scatter(grads);
                // Update param shard
                let shard_start = self.rank * self.m_shard.len();
                let shard_end = shard_start + self.m_shard.len();
                self.adam_step_shard(
                    &mut params[shard_start..shard_end],
                    &self.grad_shard,
                );
                self.all_gather_params(params);
            }
            ZeroStage::Stage3 => {
                // All-gather params for compute already done in forward
                self.grad_shard = self.reduce_scatter(grads);
                self.adam_step_shard(&mut self.param_shard, &self.grad_shard);
                // Params gathered on-demand in next forward
            }
        }
    }
}
```

### 4. Memory Savings Calculator
| Stage | Memory per GPU (Ψ=params) | 7B model, 4 GPUs |
|-------|--------------------------|-------------------|
| None (DP) | 16Ψ | 112 GB |
| Stage 1 | 4Ψ + 12Ψ/N | 32 + 6 = 38 GB |
| Stage 2 | 2Ψ + 14Ψ/N | 16 + 7 = 23 GB |
| Stage 3 | 16Ψ/N | 7 GB |
| Infinity | 16Ψ/N + offload | ~1 GB GPU + CPU/NVMe |

### 5. Key References
1. [arXiv:1910.02054] Rajbhandari 2020 - ZeRO
2. [arXiv:2104.07857] Rajbhandari 2021 - ZeRO-Infinity
3. https://www.deepspeed.ai/tutorials/zero/
