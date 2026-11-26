# Mixture-of-Loops Transformer

Status: **assessment, 2026-11-26.** Paper: *Mixture-of-Loops Transformer*,
Zhu et al., ByteDance Seed / HUST / Toronto, arXiv `2611.34923`. Code:
`github.com/hustvl/MoLT`. Not adopted; this records what transfers and what does
not.

## What it claims

| result | number |
|---|---|
| 700M dense, 400B tokens | +1.50% parameters, **1.5×** training-CE speedup, **1.7×** 11-domain validation-CE speedup (matches OLMo3's 400B loss at 232.8B tokens) |
| 700M / 1.5B downstream | average accuracy 59.1/62.6 vs OLMo3 56.6/61.2 |
| 3B MoE (660M active) | 10-domain ppl −0.75, 10-task accuracy +2.1 pp |
| Qwen3.5-0.8B continued, 100B tokens | CE down on 11/11 domains, 1.4× mean token efficiency |
| S-size ~113M latency | 996.6 ms vs 1,000.6 ms for non-looped OLMo3 at 4k with MoL-Hybrid-2 |

The latency result is **parity**, not a win: in fixed 2-loop mode U-MoLT is
slower than a 1-loop baseline (1.4675 vs 1.0006 s). It matches non-looped
latency only because the hybrid schedule **skips decoder traversals**. The
quality/efficiency gain is against a looped baseline at equal tokens.

## The mechanism

Looped models reuse shared layers; the question is what passes between loops.

- **MoLA (Mixture-of-Loops Attention)** treats keys and values as memory that can
  be read across depths and loops, selectively, in the same softmax as local KV.
- **MoLT-R** keeps the residual stream and reads current- and preceding-loop
  local KV. Expressivity; fixed N-loop; N× persistent KV.
- **MoLT-KV** restarts from the token embeddings each loop and reads
  **strict-prefix** memory KV from the previous loop (`j < t`). It **eliminates
  same-position cross-loop dependence**, which allows *loop forcing* (parallel
  training over loops) and *single-traversal* decoding. 1× persistent KV.
- **U-MoLT** splits the stack: an encoder that builds memory by looping downward,
  and a cross-only decoder that reads the normalised encoder output and produces
  the prediction. Intermediate and non-final decoder traversals can be skipped
  **without changing predictions**.
- **MoL-Hybrid-k** is the schedule that turns the skipped traversals into the
  latency parity.

## Why it matters here

1. **Our looped middle is the same design family.** The tiers already use looped
   depth; this is a stronger, more efficient version of the same idea, with a
   released reference configuration at 6+6 layers, 768 hidden, 12/3 GQA — very
   close to Small's 768 hidden.
2. **The backward defect is a cross-block dependence problem.** The 4.89 error is
   the suffix-sum term: a boundary summary depends on later positions, so the
   gradient must be summed back across the boundary. MoLT-KV's design explicitly
   **removes same-position cross-loop dependence**; U-MoLT's encoder/decoder
   split makes the readout a separate causal attention. If the suffix-sum fix
   stays intractable, this is the **architectural alternative**, not only an
   optimisation.
3. **Token efficiency is Kaggle quota.** 1.7× at 700M/400B tokens, and 1.4× on a
   0.8B continued-pretrain, would matter on a 30 h/week budget and a 2B-token
   corpus. It has to be measured at 111M/2B, not assumed from 400B.
4. **KV as memory vs our block summary.** The block summary compresses a suffix
   into a boundary state; MoLA reads KV directly across loops. The former is
   cheaper per token, the latter keeps exact memory. A comparison at equal KV
   bytes is the real question.

## What does not transfer

| item | why |
|---|---|
| **Code** | the repository has **no licence** (GitHub reports `license: null`). Ideas and architecture can be reimplemented; code cannot be copied. Treat it as all-rights-reserved and write a clean-room implementation from the paper. |
| **Triton kernel** | fused MoLA is NVIDIA sm80+, BF16/FP16. We run wgpu; a WGSL kernel is new work, and the fused path does not port. |
| **Released variants** | only a minimal **U-MoLT** is released. MoLT-R, MoLT-KV and MoL-Hybrid decoding are TODOs. The exact variants we would want must be built from the description. |
| **Scale** | 400B tokens and 700M–3B. Our budget is 2B tokens and 111M. The token-efficiency claim is a hypothesis here until measured. |
| **Muon** | the reference config trains with Muon (lr 1e-3); we use Adam. An optimizer change is a separate experiment. |
| **Hardware** | their latency is PyTorch on NVIDIA; ours is Rust/wgpu on an i5 and a T4. No number is comparable without running. |

## Adoption plan

| phase | work | gate |
|---|---|---|
| **0** | clean-room note, licence recorded, no code copied | legal note in the branch; citation in NOTICE |
| **1** | U-MoLT encoder/decoder split in the **f32 reference**, tiny config | finite-difference gradients pass; logits match a hand-built reference |
| **2** | equal-token comparison at 111M/2B on our corpus vs the current looped block | lower validation CE at equal tokens and equal parameters (±5%), or drop it |
| **3** | MoLA KV memory layout + readout skipping in the runtime | decode tok/s and KV bytes at 4k; skipped traversals change no prediction |
| **4** | loop forcing / parallel loop training | integration with the GPU loop and multi-GPU averaging |
| **5** | promote | a major model version (`frs-s-2.0.0`), because it is an architecture change |

Phase 1 is the cheapest decisive step: a CPU reference that either reproduces
the claimed mechanism or does not, before any WGSL work.

## Gates

| gate | pass |
|---|---|
| gradient correctness (finite difference) | relative error < 1e-2 on every parameter group |
| validation CE at equal tokens, 111M, our corpus | strictly better than the current looped block |
| parameters | +≤5% versus the current tier (the paper reports +1.5% at 700M) |
| decode tok/s at 4k on the target | at least parity with the current tier when skipping |
| KV bytes | reported; 1× persistent KV is a claim to verify |
| tool-call validity and eval suite | no regression |
| licence | clean-room; no copied code or weights |

## Recommendation

Do **not** pivot the architecture before the current gradient fix lands. The
suffix-sum fix is still the gate to every distillation run, and MoLT does not
make that work disappear — it may make the backward simpler, which is an
argument for testing it in the f32 reference **in parallel**, not for replacing
the shipping path.

The one experiment worth starting as soon as a build slot frees: the U-MoLT
encoder/decoder split in the f32 reference, at the tiny parity config, with a
finite-difference gate. If it is correct and wins on equal tokens, it becomes
`frs-s-2.0.0`; if not, the note says why.
