# Research: Large-Scale Cluster Tests — Multi-Node Benchmarks, Scaling Efficiency

## Task ID: 65e8c9d0-4fd3-4d58-bbbf-c25bb1ec5277

## Key Papers & Techniques

### 1. Scaling Laws
- **Paper**: Kaplan et al. (2020) "Scaling Laws for Neural Language Models" [arXiv:2001.08361]
- **Key findings**: Loss scales as power law with model size, data size, and compute. For optimal training, scale all three proportionally.
- **Chinchilla**: Hoffmann et al. (2022) [arXiv:2203.15556] — optimal model size for given compute budget.

### 2. GPT-3 Training Infrastructure
- **Paper**: Brown et al. (2020) "Language Models are Few-Shot Learners" [NeurIPS]
- **Infrastructure**: 10,000 V100 GPUs, 3D parallelism (DP × TP × PP)
- **Communication overhead**: ~35% of compute time at scale
- **Key metric**: Model FLOPs utilization (MFU) — ratio of actual throughput to theoretical peak. GPT-3 achieved ~40% MFU.

### 3. LLaMA Training
- **Paper**: Touvron et al. (2023) "LLaMA: Open and Efficient Foundation Language Models" [arXiv:2302.13971]
- **Infrastructure**: 2,048 A100 GPUs, tensor parallelism within node, pipeline parallelism across nodes
- **Throughput**: ~330 TFLOP/s per GPU (52% MFU)
- **Key optimization**: FlashAttention, fused kernels, efficient gradient checkpointing

### 4. Benchmark Metrics

#### Weak Scaling
- Fixed batch size per GPU, increase GPU count
- Ideal: throughput scales linearly
- Measure: tokens/sec vs GPU count

#### Strong Scaling  
- Fixed total batch size, increase GPU count
- Ideal: time per step decreases linearly
- Measure: step time vs GPU count

#### Communication-to-Compute Ratio
- Compute time per step
- Communication time per step (all-reduce, all-gather, reduce-scatter)
- Ideal: communication < 30% of total time
- Measure: comm_time / total_time

#### Memory Efficiency
- Per-GPU memory usage at different parallelism configs
- ZeRO stage 1/2/3 memory savings
- Activation checkpointing savings

### 5. Cluster Benchmark Design for FerrisRes

```rust
struct ClusterBenchmark {
    gpu_counts: Vec<usize>,  // e.g. [1, 2, 4, 8]
    tp_degrees: Vec<usize>,  // Tensor parallelism degrees
    pp_degrees: Vec<usize>,  // Pipeline parallelism degrees
    zero_stages: Vec<ZeroStage>,
}

struct BenchmarkResult {
    gpu_count: usize,
    tp: usize,
    pp: usize,
    dp: usize,             // Data parallelism = gpu_count / (tp * pp)
    zero_stage: ZeroStage,
    
    // Throughput metrics
    tokens_per_second: f64,
    steps_per_second: f64,
    
    // Memory metrics
    peak_memory_per_gpu: usize,  // bytes
    
    // Efficiency metrics
    mfu: f64,  // Model FLOPs Utilization
    communication_ratio: f64,  // comm_time / total_time
    
    // Scaling efficiency
    weak_scaling_efficiency: f64,  // actual_speedup / ideal_speedup
    strong_scaling_efficiency: f64,
}
```

### 6. NCCL vs Alternatives
- **NCCL**: NVIDIA's collective communications library. Optimized for NVLink/NVSwitch. Default for GPU training.
- **gloo**: Facebook's collective communications. CPU-only, slower than NCCL for GPU.
- **RCCL**: AMD's NCCL equivalent for ROCm GPUs.
- **Custom via wgpu**: FerrisRes uses Vulkan/wgpu. No NCCL available. Options:
  - Use wgpu's host-visible buffers for inter-GPU communication via CPU
  - Use MPI (via `rsmpi` crate) for coordination
  - For single-machine multi-GPU: shared host memory

### Key References
1. [arXiv:2001.08361] Kaplan 2020 - Scaling Laws
2. [arXiv:2203.15556] Hoffmann 2022 - Chinchilla
3. Brown et al. 2020 - GPT-3
4. [arXiv:2302.13971] Touvron 2023 - LLaMA
