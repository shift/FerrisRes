# Distributed Training & Tensor Parallelism Research

## Overview
Research for Phase 9: Distributed multi-GPU training with tensor parallelism.

## Key Approaches

### 1. Tensor Parallelism
Split model weights across GPUs. Each GPU holds partial weights.

**Advantages:**
- Memory savings: O(1/n) per GPU
- Communication: all-reduce only for layer outputs

**Challenges:**
- Need careful synchronization for attention ( Column and Row parallel)
- Load balancing across experts in MoE

### 2. Pipeline Parallelism
Split layers across GPU stages.

**Stages:**
1. Forward pass through stages
2. Gradient backward through stages
3. Pipelining reduces bubble overhead

### 3. Data Parallelism
Replicate model, split data.

**Types:**
- Synchronous (all-reduce gradients)
- Asynchronous (parameter server)

## Key Papers & Resources

### Megatron-LM (NVIDIA)
- Paper: https://arxiv.org/abs/1901.01928
- Tensor parallelism for Transformers
- Row/column parallel Linear layers

### DeepSpeed (Microsoft)
- GitHub: https://github.com/microsoft/DeepSpeed
- ZeRO optimization stages 1-3
- 3D parallelism

### NCCL (NVIDIA Collective Communications)
- Library for GPU collective ops
- All-reduce, all-gather, reduce-scatter
- Ring-based algorithms

## FerrisRes Considerations

### Current Architecture Fit
- BlockAttnRes has intra-block and inter-block passes
- Natural fit for:
  - Intra-block: tensor parallel within block
  - Inter-block: pipeline parallel across blocks

### Implementation Priority
1. **Data Parallelism** - Easiest, replicate model
2. **Pipeline Parallelism** - Split layer blocks
3. **Tensor Parallelism** - Most complex, split weights

### Communication Patterns
- All-reduce for gradient sync
- All-gather for attention (query distribution)
- Point-to-point for pipeline bubbles

## Status: IN PROGRESS

## Key Links
- Megatron-LM: https://arxiv.org/abs/1901.01928
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- NCCL: https://docs.nvidia.com/deeplearning/nccl/