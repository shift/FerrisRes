# STE Ternary Training Research

## Question
Can we train {-1, 0, +1} weights directly using Straight-Through Estimator (STE),
eliminating the need for LoRA adapters? Or does gradient noise through the
quantization bottleneck destroy the learning signal at 2.66B parameter scale?

## Background

Current approach: frozen ternary base weights + FP32 LoRA adapters on top.
This works but adds ~26M FP32 params and requires separate adapter management.

STE approach: during forward, weights are quantized to ternary. During backward,
gradients pass through as if the quantization didn't happen (identity gradient).
This means:
- Forward: W_q = round(W / α) ∈ {-1, 0, +1}
- Backward: ∂L/∂W ≈ ∂L/∂W_q (identity)
- Weight update: W -= lr × ∂L/∂W (in continuous space)
- Re-quantize for next forward

## Key Papers
- BitNet b1.58 (Ma et al. 2024) — trains ternary from scratch, works at 3B scale
- LLM.int8() (Dettmers et al. 2022) — mixed-precision for inference
- QLoRA (Dettmers et al. 2023) — NF4 base + FP16 LoRA, our current approach
- OneBit (Xu et al. 2024) — 1-bit distillation from dense teacher

## Unknown / Needs Investigation

1. **STE convergence at 2.66B scale**: BitNet trains from scratch. We're
   distilling from a pretrained teacher. Does STE converge when the starting
   point is already close to good (teacher weights), vs random init?

2. **Gradient signal preservation**: Ternary quantization loses ~98% of weight
   information (2 bits vs 16 bits). STE pretends this doesn't happen. How much
   learning signal survives through 35 layers of ternary matmul?

3. **Scale factor adaptation**: The absmean scale α = mean(|W|) / sqrt(2/π) is
   recomputed each step. As weights update, α changes. Does this create
   instability? Do we need per-block scaling?

4. **Hybrid approach**: STE for expert FFN (large, redundant due to MoE) but
   LoRA for attention (small, critical). Is this better than either extreme?

## Experiment Plan

### Experiment 1: STE Baseline
- Take converged LoRA model, merge adapters into base weights
- Re-quantize to ternary
- Train with STE on the ternary weights directly
- Measure loss convergence vs LoRA baseline

### Experiment 2: Hybrid STE + LoRA
- Ternary base (frozen, no STE)
- STE on expert gate/up/down only (the bulk of params)
- LoRA on attention projections (small, quality-critical)
- Compare quality vs full LoRA

### Experiment 3: Expert Count × STE
- 4 full-size experts with STE vs 16 smaller experts with STE
- Same total params, different routing dynamics
- Measure: quality, convergence speed, routing entropy

## Success Criteria
- STE converges to within 5% of LoRA-only distillation loss
- No router collapse (entropy > 0.5 per expert)
- Training stable for 10k+ steps without divergence

## Status: NOT STARTED
## Priority: HIGH (would eliminate ~26M FP32 params from training)

## Update: 2026-04-21 — Shadow Weight Memory Constraint

STE requires maintaining FP32 "shadow" weights alongside ternary during training.
For our 2-expert MoE: shadow weights alone = ~17 GB. With ternary base (2.1 GB)
and teacher overhead, this exceeds 32 GB systems.

**Progressive Distillation** (BitDistill, Oct 2025) is the recommended approach:
1. Start with FP32 weights, slowly harden to ternary over first 10-20% of steps
2. At 2.66B scale, ternary follows same scaling laws as FP32 (enough capacity redundancy)
3. Training can be "spiky" — needs learning rate warmup + gradual quantization

**Practical roadmap for FerrisRes:**
- Phase 1 (now): LoRA on frozen ternary base — 2.2 GB, fits all targets
- Phase 2: Progressive STE on ≥64 GB RAM — shadow weights + gradual hardening
- Phase 3: Pure STE pipeline for production training

**Implementation need:** `SteLinear` struct with both `ternary: Vec<i8>` and
`shadow: Vec<f32>`, with `quantize_step()` method that progressively hardens
from FP32 → ternary based on training step count.

## Update: 2026-04-21 — BF16 Shadow Weights Decision

### Precision architecture for STE path
- **Shadow weights**: BF16 (50% memory savings vs FP32, same dynamic range)
- **Gradients**: FP32 (accumulation precision)
- **Optimizer state**: FP32 (momentum buffers)
- **Base forward**: ternary {-1, 0, +1} (no change)

### Memory impact (2-expert MoE)
| Component | FP32 shadow | BF16 shadow |
|-----------|------------|-------------|
| Shadow weights | ~17 GB | **~8.5 GB** |
| Ternary base | ~2.1 GB | ~2.1 GB |
| Gradients | ~8.5 GB | ~8.5 GB (FP32) |
| Optimizer | ~8.5 GB | ~8.5 GB (FP32) |
| **Total** | **~36.6 GB** | **~27.6 GB** |

BF16 shadow weights make STE feasible on 32 GB systems (barely).
Without BF16, STE requires ≥48 GB RAM.

### Stochastic rounding
Needed for BF16 shadow weights to prevent "frozen weight" problem where
gradient updates are smaller than BF16's minimum representable delta.
Implementation: during weight update, round UP with probability equal to
the fractional part: if update = 0.3 × ε, round to ε with 30% probability.
This is future work — not needed for current LoRA path.

### Implementation blocker
FerrisRes has no `bf16` type support. Need `half` crate or nightly `f16`/`bf16`.
This should be added when STE is implemented, not before.
