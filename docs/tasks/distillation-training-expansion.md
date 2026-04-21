# Distillation Training Expansion Plan

**Goal**: Make Block-MoE-Res distillation actually train the MoE components (router + expert FFN), not just LoRA on Q/V.

**Current state**: Only 1.6M / 2.66B params trainable (LoRA rank 8 on q_proj/v_proj). MoE router and expert weights are frozen. Loss plateaus at ~16-18 and cannot converge.

**Target state**: ~50M trainable params covering router + expert FFN + attention LoRA. Capable of learning expert specialization from teacher signals.

---

## Task 1: Trainable MoE Router Weights — ✅ DONE

**Problem**: Router weights (`gate_weights: Vec<f32>` on `CpuMoELayer`) are frozen. Expert 1-3 get random noise initialization and never learn to specialize.

**Implementation**:
- Register each layer's `moe.gate_weights` with the optimizer
- Tensor name: `router_{layer_idx}` shape `[num_experts, hidden_dim]` = `[4, 1536]` = 6,144 params × 35 layers = 215K params
- In backward pass: compute `d_loss/d_router_weights` from the routing decision
  - `router_logits = hidden · gate_weights^T` → softmax → top-k → weights
  - `dL/d_gate = Σ_t (Σ_topk w_e · d_expert_output_e) · hidden_t` (simplified)
  - Plus load-balance auxiliary loss gradient
- Register with optimizer: `optimizer.register_matrix("router_{i}", 4, 1536)`

**Files**: `src/main.rs` (distillation loop), `src/model/cpu_moe.rs` (add `gate_weights_mut()`)

**Estimated effort**: Medium — need proper routing gradient math

---

## Task 2: LoRA on Expert FFN (gate/up/down per expert) — ✅ DONE

**Problem**: Expert FFN weights (gate, up, down projections for each of 4 experts × 35 layers) are frozen. LoRA only touches q_proj/v_proj.

**Implementation**:
- Extend `attach_lora()` to also target expert FFN projections:
  - `moe.expert.{e}.gate`, `moe.expert.{e}.up`, `moe.expert.{e}.down`
- Each expert has 3 projections × LoRA rank 8:
  - gate: `A=[8, hd], B=[inter_dim, 8]` = 8×1536 + inter_dim×8
  - up: same dimensions
  - down: `A=[8, inter_dim], B=[hd, 8]`
- Per layer: 4 experts × 3 projections × LoRA = lots of adapters
- Params added: ~15M (layers 0-14: inter=6144) + ~30M (layers 15-34: inter=12288) ≈ **~45M params**
- Use LoRA rank 4 for experts (not 8) to keep state manageable: ~22M params
- Update `forward_full_with_lora()` to apply LoRA to expert projections
- Register all new LoRA A/B with optimizer

**Files**: `src/model/cpu_block_attn_res.rs` (`attach_lora`, `forward_full_with_lora`), `src/model/cpu_moe.rs`, `src/main.rs`

**Estimated effort**: Large — LoRA forward integration in MoE expert loop is non-trivial

---

## Task 3: Proper Layer-by-Layer Backward Pass — 📝 IN PROGRESS

**Problem**: Current backward pass is an approximation — computes `d_hidden` from lm_head, then uses it as a proxy gradient for ALL LoRA adapters. No actual per-layer gradient computation. No gradient flow through attention → FFN → MoE routing chain.

**Implementation**:
- Implement proper reverse-mode autodiff through the student forward:
  1. `d_logits` → lm_head backward → `d_hidden_final`
  2. For each layer (reverse order):
     a. Layer scalar: `d_hidden *= scalar`
     b. PLE backward: `d_hidden` through ple_proj, ple_gate
     c. FFN/MoE backward:
        - Dense: `d_down = d_out`, `d_combined = down^T · d_down`, `d_gated = combined * d_combined`, `d_gate = d_gated * gelu'`, `d_up = d_combined * gated`, `d_input += gate^T · d_gate + up^T · d_up + down^T · d_down`
        - MoE: per-expert gradient + routing gradient
     d. Attention backward: `d_attn_out` → `d_V`, `d_K`, `d_Q` → `d_q_proj`, `d_k_proj`, `d_v_proj` → `d_normed`
     e. Residual: `d_input += d_residual`
  3. For each LoRA adapter: extract gradient from the per-layer d activations
- This replaces the current approximate gradient computation

**Files**: `src/main.rs` (distillation backward section), potentially new `src/training/backward.rs`

**Estimated effort**: Large — proper backprop through 35 MoE layers with all Gemma 4 features

---

## Task 4: MoE Load Balance Loss Integration — ✅ DONE

**Problem**: `moe_load_balance_loss()` exists but isn't wired into the distillation loss. Without it, router collapse is likely (all tokens route to same expert).

**Implementation**:
- During student forward, collect per-layer routing decisions (which experts were selected, what weights)
- Add auxiliary loss: `L_balance = num_experts × Σ(f_i × P_i)` where `f_i` = fraction of tokens routed to expert i, `P_i` = mean router probability for expert i
- Weight: `total_loss = KL + 0.5 × MSE + 0.01 × L_balance`
- Route balance loss gradient to router weights

**Files**: `src/main.rs` (forward + loss computation), `src/model/cpu_block_attn_res.rs` (forward returns routing info)

**Estimated effort**: Small — mostly wiring

---

## Task 5: LoRA on Attention O-Projection and K-Projection — ✅ DONE

**Problem**: Only q_proj and v_proj have LoRA. K and O projections are frozen.

**Implementation**:
- Add `"k_proj"`, `"o_proj"` to the `attach_lora()` target list
- Params: k_proj LoRA (rank 8): 8 × 1536 + 1536 × 8 = ~25K per layer, but GQA: kv_dim = head_dim = 256 or 512
  - k_proj: A=[8, hd], B=[kv_dim, 8] ≈ 12K + 2K = 14K per layer × 35 ≈ 490K
  - o_proj: A=[8, q_dim], B=[hd, 8] ≈ varies by head_dim
- Total: ~2M additional params
- Trivial to implement — just extend the target list

**Files**: `src/model/cpu_block_attn_res.rs` (`attach_lora`)

**Estimated effort**: Small — 5 lines of code

---

## Task 6: Checkpoint Resume with Optimizer State — 📝 PLANNED

**Problem**: Mid-training checkpoints save model weights but not optimizer state (SCALE momentum). Resuming from checkpoint would lose training progress.

**Implementation**:
- Extend `BlockMoeResConfig` or add separate optimizer state file
- Save: `{path}_stepN.optimizer.json` with SCALE momentum buffers
- On resume: reload optimizer state alongside model weights
- Or simpler: save optimizer state as safetensors alongside model

**Files**: `src/model/checkpoint.rs`, `src/main.rs` (resume logic)

**Estimated effort**: Medium

---

## Dependency Graph

```
Task 5 (LoRA on K/O) ──→ Task 3 (proper backward)
                              ↑
Task 2 (LoRA on experts) ────┘
                              ↓
Task 1 (trainable router) → Task 4 (balance loss) → Task 6 (checkpoint resume)
```

## Suggested Implementation Order

1. **Task 5** (trivial, 5 minutes) — extend LoRA targets
2. **Task 1** (router trainable, medium) — unlock expert specialization
3. **Task 4** (balance loss, small) — prevent router collapse
4. **Task 2** (expert FFN LoRA, large) — the big one, most params
5. **Task 3** (proper backward, large) — replaces approximate gradients
6. **Task 6** (checkpoint resume, medium) — for long training runs

Tasks 1+4 can ship together. Task 2+3 should ship together. Task 6 can be done anytime.

## Expected Impact

| Config | Trainable | Convergence | Steps to good distill |
|---|---|---|---|
| ~~LoRA q/v only~~ | ~~1.6M~~ | ~~Cannot converge~~ | ~~∞~~ |
| ✅ Tasks 1+4+5 (router + K/O + balance) | 3.6M (0.14%) | Slow | 20,000+ |
| ✅ + Task 2 (LoRA experts r=8) | **~26M (1.0%)** | **Feasible** | **5,000-10,000** |
| 📝 + Task 3 (proper backward) | same | Much faster convergence | 2,000-5,000 |
