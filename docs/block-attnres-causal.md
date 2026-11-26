# Block AttnRes causal inter-block attention

This note records the rule the CPU reference implements, how the GPU
generation path now implements it, what was measured, and what is still
known to be non-equivalent.

## Rule

For token position `t`, at a block boundary:

1. The query is that position's own hidden state `h[t]` (after the boundary
   layer's `layer_scalar`).
2. Entry 0's key is `rms_norm((1/(t+1)) * sum_{s<=t} embed[s], attn_res_norm)`.
3. Entry `b+1`'s key is
   `rms_norm((1/((t+1)*layers_per_block)) * sum_{s<=t} sum_{l in block b} x_l[s], attn_res_norm)`,
   where `x_l[s]` is layer `l`'s post-`layer_scalar` output.
4. `score_e = (h[t] . key_e) / sqrt(hidden_dim)`; a softmax over the entries
   gives the weights; the output is `sum_e w_e * key_e`, added to `h[t]`.

Because every entry is a prefix mean over `0..=t`, position `t` never sees a
later token. The reference implementation is
`CpuBlockAttnResModel::forward` in `src/model/cpu_block_attn_res.rs`.

## GPU paths

* `BlockAttnResModel::forward` — per-position batch path, used by
  `TwoPhaseInference::forward`. It also applies the inter-block rule.
* `BlockAttnResModel::forward_prefill_cached` — the sequence path used by
  `TokenGenerator`. It runs causal self-attention over the prompt, builds
  `[entries, seq, hidden]` per-position summaries, and applies the rule at
  every block boundary.
* `BlockAttnResModel::forward_decode_cached` — one position at a time. It
  extends `[entries, hidden]` running sums, converts them to the current
  position's means, and uses the token's hidden state as the query, so decode
  reproduces the matching prefill row.

`BlockAttnResState` (`BlockAttnState`) owns the running sums. The
`CausalPrefixOp` kernel emits either the causal mean (`normalize = 1`) or the
raw running sum (`normalize = 0`); the raw form is what decode extends.

`layer_scalar` is applied once per layer, on the whole layer output, matching
`CpuBlockAttnResModel`.

## Measured equivalence

`tests/gpu_cpu_equivalence.rs` loads one model's weights into
`CpuBlockAttnResModel` and the GPU model and compares logits. On the Intel
Vulkan ICD on this host:

| Check | Max abs logit difference |
|---|---|
| CPU vs GPU, full sequence | 2.861e-6 |
| CPU vs GPU, prefill(3) + decode(2) | 2.503e-6 |
| GPU full prefill vs prefill(2) + decode(3) | 2.742e-6 |

The RoPE cos/sin table is computed in f64 on the host because the driver's
`sin`/`cos` are only ~5e-5 accurate, which is above the 1e-4 budget on its own.

## Known gaps

* `BlockAttnResModel::forward` treats its `batch` axis as independent
  positions (no cross-position self-attention); it is not a sequence forward.
  The sequence path is `forward_prefill_cached`.
* `src/inference/unified_generator.rs` still drives per-layer forwards, so it
  does not apply inter-block attention, the embedding `sqrt(hidden)` scale or
  the final norm.
* `src/main.rs` does not upload final-norm weights to `TokenGenerator`; the
  setter (`set_final_norm`) exists but is not called.
* The distillation/LoRA gradient path (`forward_train` + `compute_lora_grad`)
  does not backpropagate the inter-block attention. The causal summaries make
  a parameter in an early layer receive a suffix-sum gradient from every later
  boundary position; the path uses `d_hidden` directly and ignores that term
  (and the final-norm and attention Jacobians). `forward_train` now **panics**
  (`lora_gradient_invalid_with_blocks`) when blocks and a LoRA manager are both
  active. `training::grad_audit::max_rel_error_blocked_lora` measures the
  mismatch against a central finite difference (currently 4.89), and
  `examples/grad_audit.rs` prints it as the gate metric
  `metric=grad_quality value=1 - min(error, 1)`.
