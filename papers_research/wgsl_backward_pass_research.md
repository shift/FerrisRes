# Research: Backward Pass via WGSL Compute Shaders

**Task ID:** Informs Task 8 (backward pass for all ops)

## Question

Has anyone implemented autodifferentiation using WGSL compute shaders in wgpu?
Can we write backward kernels for matmul, softmax, RMSNorm, etc. as WGSL
and have them be competitive?

## Finding: Nobody Does This — But It's Viable

### Survey of Rust ML Frameworks

| Framework | GPU Backend | Backward Strategy |
|-----------|-------------|-------------------|
| candle (HuggingFace) | wgpu CUDA | CPU backward + CUDA kernels (no wgpu backward) |
| burn | wgpu + others | Auto-generated backward via JIT (has wgpu autodiff!) |
| dfdx | CUDA | Hand-written backward kernels in CUDA |
| tch-rs | libtorch | C++ autograd bindings |
| tract | CPU only | CPU autodiff |

**Key finding: burn is the closest analogue.** Burn generates backward WGSL from
forward WGSL using a source-to-source differentiation approach. However, burn's
autodiff is limited — it traces operations at a high level and generates backward
passes automatically, but doesn't support complex custom kernels like flash attention.

### Why Hand-Written WGSL Backward Is The Right Call For FerrisRes

FerrisRes has custom WGSL kernels (flash_decode, prefill_attn, tiled decode, RoPE,
MoE gating) that are not standard matrix ops. No autodiff framework can differentiate
through these. We need to write backward kernels manually for each.

This is exactly what PyTorch does for its C++/CUDA kernels — each op has a manual
`backward()` function registered in the autograd system.

### Feasibility Analysis Per Op

**MatMulOp backward (dL/dA, dL/dB, dL/dbias):**
- Forward: C = A × B
- Backward: dL/dA = dL/dC × B^T, dL/dB = A^T × dL/dC
- These are just more matmuls — reuse the existing tiled matmul kernel!
- No new shader needed — just call `dispatch()` with transposed inputs.
- Estimated effort: trivial (wire existing MatMulOp with swapped bindings).

**RmsNormOp backward:**
- Forward: y = (x / rms(x)) * γ, where rms(x) = sqrt(mean(x²) + ε)
- Backward: dL/dx = γ/rms * (dL/dy - mean(dL/dy * y) * y / rms²)
- One new WGSL kernel needed — same structure as forward (workgroup reduction).
- Estimated effort: small (single kernel, ~60 lines WGSL).

**SoftmaxOp backward:**
- Forward: s_i = exp(x_i) / Σ exp(x_j)
- Backward: dL/dx_i = s_i * (dL/ds_i - Σ_j dL/ds_j * s_j)
- One new WGSL kernel. The reduction Σ dL/ds_j * s_j is a workgroup sum.
- Estimated effort: small.

**FlashDecodeOp backward (attention gradients):**
- Forward: online softmax over KV sequence, weighted V sum.
- Backward: dL/dQ = softmax_weights * V × (dL/dO) scaled, etc.
- This is the standard flash attention backward from FlashAttention-2 paper.
- Needs two passes: (1) recompute softmax stats from saved Q,K, (2) compute dQ,dK,dV.
- Estimated effort: **large** — ~200 lines of new WGSL. The tricky part is the
  online softmax backward which needs saved max and sum from the forward pass.
  Currently flash_decode doesn't save these — forward needs modification to output
  softmax stats alongside the attention output.

**RopeOp backward:**
- RoPE is a rotation: [cos θ, -sin θ; sin θ, cos θ] × [x0, x1]
- Backward: the rotation is orthogonal, so the backward is just the transpose rotation
  (same cos/sin, negated sin). This is literally the same kernel with sin negated.
- Estimated effort: trivial (reuse forward shader, flip sin sign).

**MoEGatingOp backward:**
- Forward: top-k softmax over gate logits.
- Backward: standard softmax backward through the top-k mask.
- Estimated effort: small (extend existing gating kernel).

**MoEExpertOp backward:**
- Forward: selected_expert_weight × expert_up × expert_down
- Backward: gradient through the selected expert's weights.
- Since we're only training LoRA adapters (not full expert weights), the backward
  only needs to flow through the LoRA layers — not the frozen expert weights.
- Estimated effort: medium (need to integrate with LoRA backward path).

### Gradient Accumulation in f32

**Concern:** Accumulating gradients over many layers in f32 can lose precision.

**Analysis:** This is the standard practice in all deep learning frameworks.
PyTorch, TensorFlow, JAX all accumulate in fp32 (or bf16 on newer hardware).
The gradient magnitudes in transformer layers are typically well-conditioned.
For LoRA specifically (rank 4-16), the parameter count per adapter is tiny
(hidden_dim × rank × 2), so gradient accumulation is over very few values.

**No issue for our use case.**

### Memory Cost of Backward

Backward pass requires saving intermediate activations from the forward pass.

For a single transformer layer with hidden_dim=2048:
- Input hidden states: 2048 × 4 = 8 KB
- Attention scores: num_heads × 4 = 32 B (for decode, seq_len=1)
- Post-attention hidden states: 8 KB
- FFN intermediate: intermediate_dim × 4 = 32 KB (with 4× expansion)
- Total per layer: ~50 KB

For 64 layers: 3.2 MB of saved activations per token.
With gradient checkpointing (already partially implemented via CheckpointStore),
only block boundaries need saving: 8 blocks × 50 KB = 400 KB per token.

**Very manageable, even on iGPU.**

### Recommended Implementation Order

1. **MatMul backward** — free (reuse forward kernel with transposed inputs)
2. **RopeOp backward** — free (reuse forward with -sin)
3. **RmsNormOp backward** — small new kernel
4. **SoftmaxOp backward** — small new kernel
5. **FlashDecodeOp backward** — large, but only needed for full training.
   For LoRA-only, gradients flow through the output projection (matmul backward)
   and don't need to differentiate through the attention pattern itself.
6. **MoE backward** — medium, integrate with LoRA path

### Key Insight: LoRA-Only Backward Drastically Reduces Scope

Since self-learning uses LoRA/QLoRA adapters (not full weight training), the
backward pass only needs to compute gradients for the adapter parameters.
The base model weights are frozen.

For a LoRA adapter on a Linear layer (rank r):
- Forward: y = W_frozen @ x + (B @ A) @ x  where B is [out, r], A is [r, in]
- Backward: dL/dA = B^T @ dL/dy @ x^T  (matmul — reuse existing kernel)
             dL/dB = dL/dy @ x^T @ A^T  (matmul — reuse existing kernel)

The entire backward for LoRA is just matmuls + elementwise ops. No need for
flash attention backward, MoE expert backward, etc. Gradients flow through
the base model's frozen forward pass via straight-through.

**This means tasks 1-4 (matmul, RoPE, RMSNorm, softmax backward) are sufficient
for LoRA-based self-learning. Flash attention backward can be deferred.**

## References

- burn framework: https://github.com/tracel-ai/burn — has wgpu autodiff
- FlashAttention-2 backward: https://arxiv.org/abs/2307.08691 Section 3.2
- PyTorch autograd function: https://pytorch.org/docs/stable/notes/extending.html
- Cody & Waite range reduction for trig backward stability
