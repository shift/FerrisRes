# Research: Optimizer State Sharding — Partitioned Adam, CPU Offload

## Task ID: 41f13c05-05f4-4033-90bb-f5abe92ac16c

## Key Papers & Techniques

### 1. 8-bit Adam (bitsandbytes)
- **Paper**: Dettmers et al. (2022) "8-bit Optimizers via Block-wise Quantization" [arXiv:2110.02861]
- **Core idea**: Quantize optimizer states (m, v) to 8-bit (INT8) dynamically. Dequantize only during the update step.
- **Block-wise quantization**: Divide state vector into blocks of 2048. Quantize each block independently to preserve dynamic range.
- **Memory savings**: 2× for optimizer states (FP32 → INT8 for m and v)
- **Application to FerrisRes**: Critical for Intel iGPU. 8-bit Adam states fit in half the memory. Combined with CPU offload, enables training on devices with limited VRAM.

### 2. LION Optimizer (Simpler States)
- **Paper**: Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms" [arXiv:2302.06675]
- **Core idea**: Discovered via program search. Simpler than Adam: only maintains momentum (no second moment). Update: sign(m_t) instead of m_t / (sqrt(v_t) + ε).
- **Memory savings**: ~50% optimizer state reduction vs Adam (only momentum, no variance)
- **Application**: Replace Adam with LION for on-device training. Half the optimizer memory.

### 3. Adafactor (Factorized Optimizer)
- **Paper**: Shazeer & Stern (2018) "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
- **Core idea**: Factorize the second moment matrix v into row and column statistics. For a weight matrix of shape (m, n), store v as vectors of size m + n instead of matrix m × n.
- **Memory savings**: O(m + n) instead of O(mn) for second moment
- **Application**: For large weight matrices in transformer (QKV projections, FFN), factorized states save significant memory.

### 4. CPU Offload Strategy for FerrisRes
```
Layout:
  GPU (VRAM): Current layer's weights + activations + workspace
  CPU (RAM): All optimizer states (m, v), inactive layer weights
  Disk (NVMe): Checkpoint files, inactive concepts/tools

Training loop:
  For each layer:
    1. Transfer layer weights: CPU → GPU
    2. Forward pass on GPU
    3. Compute loss
    4. Backward pass on GPU → gradients
    5. Transfer gradients: GPU → CPU
    6. Update optimizer states on CPU (m, v update)
    7. Apply weight update on CPU
    8. Transfer updated weights: CPU → GPU (next iteration)
    9. Discard GPU layer weights (free VRAM)
```

### 5. PartitionedAdam Implementation

```rust
struct PartitionedAdam {
    // This partition's slice of optimizer states
    m_partition: Vec<f32>,  // First moment (momentum)
    v_partition: Vec<f32>,  // Second moment (variance)
    
    // Partition boundaries
    param_offset: usize,    // Start index in full parameter array
    param_len: usize,       // Length of this partition
    
    // Hyperparameters
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u64,
    
    // CPU offload
    offloaded: bool,
}

impl PartitionedAdam {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        let local_params = &mut params[self.param_offset..self.param_offset + self.param_len];
        let local_grads = &grads[self.param_offset..self.param_offset + self.param_len];
        
        self.step += 1;
        
        for i in 0..self.param_len {
            let g = local_grads[i];
            self.m_partition[i] = self.beta1 * self.m_partition[i] + (1.0 - self.beta1) * g;
            self.v_partition[i] = self.beta2 * self.v_partition[i] + (1.0 - self.beta2) * g * g;
            
            let m_hat = self.m_partition[i] / (1.0 - self.beta1.powi(self.step as i32));
            let v_hat = self.v_partition[i] / (1.0 - self.beta2.powi(self.step as i32));
            
            local_params[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * local_params[i]);
        }
    }
}
```

### Key References
1. [arXiv:2110.02861] Dettmers 2022 - 8-bit Optimizers
2. [arXiv:2302.06675] Chen 2023 - LION
3. Shazeer & Stern 2018 - Adafactor
4. [arXiv:1910.02054] Rajbhandari 2020 - ZeRO (partitioned optimizer)
