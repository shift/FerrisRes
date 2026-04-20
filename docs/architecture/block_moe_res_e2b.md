# Block-MoE-Res Architecture: Gemma 4 E2B Distillation Target

## Overview

Every FerrisRes model is a **Block-MoE-Res** architecture, even when the teacher is dense.
This document specifies how Gemma 4 E2B (35 dense layers, 4GB) is distilled into a
Block-MoE-Res student model with inter-block attention, MoE FFN, and residual connections.

---

## 1. Block Partitioning

### Layer Structure (from Gemma 4 E2B config)

```
Layers 0-3:   sliding_attention, inter_dim=6144
Layer 4:      full_attention,    inter_dim=6144
Layers 5-8:   sliding_attention, inter_dim=6144
Layer 9:      full_attention,    inter_dim=6144
Layers 10-13: sliding_attention, inter_dim=6144
Layer 14:     full_attention,    inter_dim=6144
--- KV sharing boundary (layer 15+) shares KV from layers <15 ---
Layers 15-18: sliding_attention, inter_dim=12288 (double-wide MLP)
Layer 19:     full_attention,    inter_dim=12288
Layers 20-23: sliding_attention, inter_dim=12288
Layer 24:     full_attention,    inter_dim=12288
Layers 25-28: sliding_attention, inter_dim=12288
Layer 29:     full_attention,    inter_dim=12288
Layers 30-33: sliding_attention, inter_dim=12288
Layer 34:     full_attention,    inter_dim=12288
```

### Partitioning Decision: 7 blocks × 5 layers

Full attention layers (4, 9, 14, 19, 24, 29, 34) are natural block boundaries.
Each block contains 4 sliding layers + 1 full attention layer:

```
Block 0: layers 0-4     (4 sliding + 1 full)   inter_dim=6144
Block 1: layers 5-9     (4 sliding + 1 full)   inter_dim=6144
Block 2: layers 10-14   (4 sliding + 1 full)   inter_dim=6144
Block 3: layers 15-19   (4 sliding + 1 full)   inter_dim=12288
Block 4: layers 20-24   (4 sliding + 1 full)   inter_dim=12288
Block 5: layers 25-29   (4 sliding + 1 full)   inter_dim=12288
Block 6: layers 30-34   (4 sliding + 1 full)   inter_dim=12288
```

**Rationale**: Full attention layers have head_dim=512 (vs 256 for sliding) and
process information globally — they are the natural point to accumulate block
representations and run inter-block attention. Each block ends with a full
attention layer that "summarizes" the block before passing to the next.

### KV Sharing Interaction

KV sharing (layers 15-34 reuse KV from layers <15) operates within this structure:
- Blocks 0-2 (layers 0-14): compute their own K/V normally
- Blocks 3-6 (layers 15-34): reference shared K/V from the matching layer type
  in blocks 0-2 (matched by rope_theta — sliding↔sliding, full↔full)

---

## 2. Dense → MoE Conversion

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| num_experts | 4 | Balance between specialization and memory. 8 experts = 8× FFN params per layer (too large for E2B). 2 is barely MoE. 4 is the sweet spot. |
| top_k | 2 | Redundancy prevents single-expert collapse. Each token uses 2 experts (50% of capacity). |
| init_strategy | perturbation | Expert 0 = exact dense copy (preserves teacher knowledge). Experts 1-3 = dense + noise (stddev=0.02). Router initialized small-random so initial routing is near-uniform. |
| load_balance | yes | Auxiliary loss: `balance = num_experts * Σ(f_i × P_i)` where f_i = fraction of tokens to expert i, P_i = mean router prob. Prevents router collapse. |

### Expert Initialization Detail

```
Expert 0: gate_w[0] = dense_gate_w, up_w[0] = dense_up_w, down_w[0] = dense_down_w
Expert 1: gate_w[1] = dense_gate_w + N(0, 0.02), up_w[1] = dense_up_w + N(0, 0.02), ...
Expert 2: gate_w[2] = dense_gate_w + N(0, 0.02), ...
Expert 3: gate_w[3] = dense_gate_w + N(0, 0.02), ...
Router:   gate_weights = N(0, 0.01)  → initial routing ~uniform
```

All experts start functional (close to the dense FFN they came from).
During distillation, LoRA adapters + router training specialize them.

---

## 3. Inter-Block Attention

### Mechanism

Inter-block attention runs at block boundaries (after the full attention layer
at the end of each block). It allows information to flow between blocks without
requiring sequential token-level attention across the full sequence.

### Forward Pass at Block Boundary

After Block N's 5 intra-block layers complete:

```
1. Accumulate block representation:
   block_rep[N] = mean_pool(hidden_states across seq positions)
   
2. Normalize block representations:
   for i in 0..=N:
       normed_reps[i] = rms_norm(block_rep[i])

3. Cross-attention (query = current hidden, keys/values = all block reps):
   scores = current_hidden @ normed_reps^T * (1/sqrt(hd))
   weights = softmax(scores)
   inter_out = weights @ normed_reps

4. Residual add:
   hidden = hidden + inter_out
```

This is the same mechanism as `BlockAttnResLayer::forward_inter_block()` in the
GPU implementation (block_attn_res.rs:955). The `partial_sum` accumulates the
mean of token hidden states within the block, and `block_reps` stores one vector
per completed block plus the initial embedding.

### Pseudo-query

The existing GPU implementation uses a `pseudo_query` GpuBuffer and `attn_res_proj`
linear layer. These are learnable projections that transform the accumulated
partial_sum into a query for inter-block attention. The CPU version should mirror
this with CpuLinear weights initialized to identity.

---

## 4. Full Forward Pass Pseudocode

```
fn forward(tokens: &[u32]) -> Vec<f32> {
    // 1. Embedding (Gemma scaling)
    hidden = embed(tokens) * sqrt(hidden_dim)
    
    // 2. Pre-compute PLE inputs from initial hidden state
    ple_inputs = precompute_ple(hidden, tokens)  // [n_layer, seq, ple_dim]
    
    // 3. Initialize block representations
    block_reps = [hidden.mean_pool()]  // block_reps[0] = mean of initial embedding
    
    // 4. Block loop
    for block_n in 0..7 {
        let layer_start = block_n * 5
        let layer_end = layer_start + 5
        
        // Intra-block: run 5 layers (4 sliding + 1 full attention)
        for layer_idx in layer_start..layer_end {
            ple_slice = ple_inputs[layer_idx]
            shared_kv = get_shared_kv(layer_idx)  // KV sharing for layers 15+
            (k, v) = layer.forward_full(&mut hidden, ple_slice, shared_kv)
            
            // Accumulate into partial_sum for block representation
            partial_sum += hidden.mean_pool()  // accumulate token means
            
            if is_block_boundary(layer_idx) {
                // Store this block's representation
                block_rep = rms_norm(partial_sum)
                block_reps.push(block_rep)
                
                // Inter-block attention: query=current, keys=block_reps
                inter_out = inter_block_attention(hidden, block_reps)
                hidden = hidden + inter_out
                
                // Reset partial_sum for next block
                partial_sum = zero()
            }
        }
    }
    
    // 5. Final norm + LM head + softcapping
    hidden = rms_norm(hidden, final_norm)
    logits = hidden @ lm_head
    logits = tanh(logits / 30.0) * 30.0
    
    logits
}
```

---

## 5. Size Analysis

### Teacher (Gemma 4 E2B Dense)

| Component | Shape | Params | Size (BF16) |
|-----------|-------|--------|-------------|
| embed_tokens | [262144, 1536] | 403M | 806 MB |
| 35 × (Q+K+V+O proj) | 35 × [1536, 2048/4096] | 155M | 310 MB |
| 35 × (gate+up+down) | 35 × [1536, 6144/12288] | 635M | 1,270 MB |
| 35 × norms | 35 × ~[1536] | 0.2M | 0.4 MB |
| PLE (model_proj + per-layer) | [1536, 8960] + per-layer | 14M | 28 MB |
| lm_head (tied) | — | 0 | 0 MB |
| **Total** | | **~1.2B** | **~2.4 GB** |

### Student (Block-MoE-Res, 4 experts)

| Component | Shape | Params | Size (BF16) | Size (NF4) |
|-----------|-------|--------|-------------|-------------|
| embed_tokens | [262144, 1536] | 403M | 806 MB | 201 MB |
| 35 × (Q+K+V+O proj) | 35 × same | 155M | 310 MB | 78 MB |
| 35 × MoE router | 35 × [1536, 4] | 0.2M | 0.4 MB | 0.1 MB |
| 35 × 4 experts × (gate+up+down) | 35 × 4 × [1536, 6144/12288] | 2,540M | 5,080 MB | 635 MB |
| 35 × norms + scalars | same | 0.2M | 0.4 MB | 0.1 MB |
| PLE | same | 14M | 28 MB | 7 MB |
| Inter-block attn | 7 × [1536, 1536] | 11M | 22 MB | 5 MB |
| LoRA adapters (rank=8) | ~175 adapters × (hd×8 + 8×hd) | 4.3M | 9 MB | 9 MB |
| **Total** | | **~3.1B** | **~6.3 GB** | **~935 MB** |

**Key insight**: With NF4 quantization on expert weights (the bulk of the model),
the student fits in ~1 GB — viable for edge devices. With LoRA adapters in FP32
(only 9 MB), fine-tuning memory is minimal.

---

## 6. Distillation Training Strategy

### Phase 1: Weight Transfer
- Convert all Gemma 4 weights to CpuBlockAttnResModel
- Convert dense FFN → MoE (4 experts, perturbation init)
- Add LoRA adapters (rank 8, targets: q_proj, v_proj, expert gates, router, ple_gate)

### Phase 2: Training Loop
- Forward: student with LoRA → logits + hidden states
- Loss: α·KL(logits) + β·MSE(hidden_states) + γ·load_balance
  - α=1.0, β=0.5, γ=0.01
- Backward: chain rule through student, collect LoRA gradients only
- Update: Adam on LoRA params (base weights frozen)

### Phase 3: Merge & Deploy
- Merge LoRA weights into base
- NF4 quantize expert weights
- Upload to GPU for inference
- Validate: teacher KL divergence < threshold

---

## 7. Compatibility Notes

### PLE within Blocks
PLE operates per-layer, not per-block. All 35 layers still get their PLE injection.
Block boundaries only affect where inter-block attention runs — PLE is independent.

### KV Sharing
Layers 15-34 share K/V from layers <15. This works within the block structure:
- Block 0-2 layers compute their own K/V normally
- Block 3-6 layers look up shared K/V from the matching source layer

### Layer Scalar
Per-layer scalar multiplication happens at the end of each layer, same as teacher.
No interaction with block structure.

### Per-Layer Dimensions
Each layer retains its own head_dim (256 or 512) and intermediate_dim (6144 or 12288).
MoE experts inherit the per-layer intermediate_dim — experts in blocks 0-2 use 6144,
experts in blocks 3-6 use 12288.
