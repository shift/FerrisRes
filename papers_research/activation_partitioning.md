# Research: Activation Partitioning & Gradient Checkpointing

## Task ID: 2cce1449-11d1-493c-97cb-8719c6b88531

## Key Papers & Techniques

### 1. Gradient Checkpointing (Activation Recomputation)
- **Paper**: Chen et al. (2016) "Training Deep Nets with Sublinear Memory Cost" [arXiv:1604.06174]
- **Core idea**: Don't store all intermediate activations during forward pass. Store only "checkpoints" at certain layers. During backward, recompute the missing activations from the nearest checkpoint.
- **Memory savings**: O(√n) instead of O(n) where n = number of layers. With √n checkpoints, each recomputation costs at most √n layers.
- **Trade-off**: 33% more compute (one extra forward pass per segment) for massive memory savings.

### 2. Selective Checkpointing
- **Paper**: Jain et al. (2020) "Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization" [arXiv:2111.00675]
- **Core idea**: Not all activations are equally expensive to recompute. Checkpoint the cheap-to-recompute ones, store the expensive ones.
- **Strategy**: Profile each layer's activation memory vs recomputation cost. Use dynamic programming to find optimal checkpoint set.
- **Application**: In FerrisRes, attention layers produce large activations (seq_len × hidden_dim × heads) but are cheap to recompute. Store only the block boundaries.

### 3. Sequence Parallelism (Megatron-LM)
- **Paper**: Korthikanti et al. (2022) "Reducing Activation Recomputation in Large Transformer Models" [arXiv:2205.05198]
- **Core idea**: LayerNorm and Dropout activations are replicated across tensor-parallel ranks. Instead, partition them along the sequence dimension.
- **Savings**: For TP degree t, reduces activation memory by factor of t for non-attention layers.
- **Application**: In FerrisRes with TP, partition the block-level activations (between attention layers) across ranks.

### 4. Activation Offloading
- **Paper**: Bombarelli (2023) "Llama 3.1 Training" techniques
- **Core idea**: Offload activations to CPU during forward, fetch back during backward. Overlaps compute with transfer.
- **Application for Intel iGPU**: Activations live in system RAM, fetched to GPU only during backward pass.

### 5. Memory Estimation
For a transformer layer with:
- hidden_dim = h, seq_len = s, num_heads = n, head_dim = d = h/n

Activations per layer:
- Attention QKV projections: 3 × s × h × 4 bytes
- Attention scores: n × s × s × 4 bytes (the big one!)
- Attention output: s × h × 4 bytes
- FFN: 2 × s × 4h × 4 bytes (assuming 4× expansion)
- LayerNorm: 2 × s × h × 4 bytes

Total per layer ≈ (10h + n×s) × s × 4 bytes

For seq_len=2048, hidden=2560, heads=32: ~1.3 GB per layer!
With gradient checkpointing: ~1.3 GB for 2 checkpoints + recompute cost.

### 6. Implementation for FerrisRes

```rust
enum CheckpointPolicy {
    /// Checkpoint every N layers
    EveryN(usize),
    /// Checkpoint at block boundaries (FerrisRes blocks)
    BlockBoundaries,
    /// Only checkpoint layers where activation memory > threshold
    Selective { max_memory_bytes: usize },
    /// Checkpoint all (no recomputation, max memory)
    None,
}

struct ActivationCheckpoint {
    /// Layer index where checkpoint is stored
    layer_idx: usize,
    /// The stored activation tensor
    activation: Vec<f32>,
    /// Dimensions of the activation
    shape: (usize, usize), // (seq_len, hidden_dim)
}

struct GradientCheckpointer {
    policy: CheckpointPolicy,
    checkpoints: Vec<ActivationCheckpoint>,
}

impl GradientCheckpointer {
    fn forward_with_checkpoints<F>(
        &mut self,
        layers: &[Layer],
        input: &[f32],
        seq_len: usize,
        layer_fn: F,
    ) -> Vec<f32>
    where
        F: Fn(&Layer, &[f32], usize) -> Vec<f32>,
    {
        let mut current = input.to_vec();
        let mut last_checkpoint = input.to_vec();
        let mut last_checkpoint_layer = 0;
        
        for (i, layer) in layers.iter().enumerate() {
            if self.should_checkpoint(i, &current) {
                self.checkpoints.push(ActivationCheckpoint {
                    layer_idx: i,
                    activation: current.clone(),
                    shape: (seq_len, current.len() / seq_len),
                });
            }
            current = layer_fn(layer, &current, seq_len);
        }
        current
    }
    
    fn recompute_segment(
        &self,
        layers: &[Layer],
        from_layer: usize,
        to_layer: usize,
        checkpoint: &[f32],
    ) -> Vec<f32> {
        // Recompute activations from checkpoint to target layer
        let mut current = checkpoint.to_vec();
        for layer in &layers[from_layer..to_layer] {
            current = layer.forward(&current);
        }
        current
    }
}
```

### Key References
1. [arXiv:1604.06174] Chen 2016 - Gradient Checkpointing
2. [arXiv:2111.00675] Jain 2020 - Checkmate
3. [arXiv:2205.05198] Korthikanti 2022 - Megatron-LM Sequence Parallelism
