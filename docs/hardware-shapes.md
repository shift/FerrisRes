# Hardware-shaped model variants

Status: **design space, 2026-11-26.** The four tiers in `model-tiers.md` are one
point in a larger space. This maps the bottlenecks to the shape levers and names
which variants are worth building.

## The rule

A shape is chosen by the **bottleneck**, not by parameter count:

- decode is **memory-bandwidth bound**: bytes read per token set the ceiling;
- prefill is **compute bound**: attention is quadratic before the KV cache fills;
- training is **memory + compute bound**, and on a T4 an f32 path wastes the
  tensor cores;
- a small L3 cache makes a looped core re-read RAM every pass.

Every number below is calculated from bandwidth and geometry, as in
`model-tiers.md`, not measured on the device.

## Hardware profiles

| profile | bandwidth | memory | binding constraint | shape that fits |
|---|---:|---:|---|---|
| **Pi 5 / ARM board** | ~17 GB/s | 4–8 GB shared | bandwidth, single-thread | 40–80M, 24–32k vocab, GQA 4:1, sliding window 1k + 2 global, ternary, tied embeddings |
| **phone / NPU** | ~50 GB/s | 8–12 GB shared | static shapes, no branching | 0.5–1.5B, 4-bit, GQA, no MoE, fixed seq buckets, int8 KV |
| **i5 + HD 530** | ~6–34 GB/s | 32 GB, 6 MB L3 | L3 residency of the loop core | the current Small/Base, loop core ≤ 6 MB, iGPU for prefill |
| **Strix Halo** | ~256 GB/s | 64–128 GB unified | bandwidth is generous, compute modest | 1–4B, top-2 MoE, 32k, NF4, hybrid attention |
| **DGX Spark** | ~273 GB/s | 128 GB unified | memory is abundant, bandwidth moderate | 8–30B MoE (16–64 experts, top-2), 64k, MLA or hybrid SSM+attention |
| **2× T4** | 320 GB/s each | 16 GB each | compute + fp32 waste; no NVLink | 1–7B dense or 8–13B MoE, bf16, 8k, batch > 1, local-SGD |
| **teacher host** | — | — | quality, not latency | Qwen3/Gemma open weights; not a FerrisRes shape |

## The shape levers

| lever | buys | costs | best for |
|---|---|---|---|
| **width / depth** | quality | bandwidth per token | every profile |
| **looped / recurrent depth** | params without bytes | compute passes; harder to train | bandwidth-starved devices (Pi, i5 L3) |
| **top-1 / top-2 MoE** | capacity at fixed active bytes | storage; load balancing; bad on NPUs | unified-memory boxes (Strix, Spark) |
| **GQA vs MQA** | quality at 2–4× KV | more KV bytes | everything except the tightest RAM |
| **MLA / latent KV** | long context cheaply | a new kernel and training path | Spark, T4 long context |
| **sliding window + global** | linear prefill at long context | recall of the far past | all local profiles |
| **hybrid SSM / linear attention** | O(1) decode state, no KV growth | weaker exact recall | phones, long sessions |
| **ternary / MatMul-free** | 2–4× bandwidth and compute | needs from-scratch training; accuracy risk | Pi, phone, CPU |
| **NF4 / int8 weights** | half the bytes | a small quality loss | Base/Large, Spark |
| **3-bit / int8 KV** | 4–8× context | kernel complexity | all |
| **tied embeddings** | head + embedding share bytes | a small quality loss | < 300M, tight RAM |
| **vocab 24–32k vs 151k** | the head is 66% of Base's bytes at 151k | fertility on code | every small tier |
| **early exit / mixture-of-depths** | per-token compute | training complexity, latency jitter | instant mode, phones |
| **draft + verify** | latency | a second model | Pi, phone, CPU |
| **retrieval-first** | knowledge without parameters | a tool loop | all, especially Nano |

## Variants worth building

| id | shape | target | why it is a separate variant |
|---|---|---|---|
| `frs-n` | 46M, looped 6/7, MQA, 32k vocab, ternary, tied | Pi 5, phone | the current Nano; the smallest useful assistant with tools |
| `frs-n-gqa` | 46M, **GQA 4:1** instead of MQA | Pi 5 | quality at 4× KV, still under 4 MB at 8k |
| `frs-s` | 111M, looped 8/10, 32k, ternary | i5 | first shipping assistant |
| `frs-s-code` | 111M with a **code-aware 32k tokenizer** | i5, Rust/Nix work | fertility on Rust/Nix beats a 151k vocab for a small model |
| `frs-s-draft` | 20–30M draft for speculative decoding | all | latency, not quality |
| `frs-b-hybrid` | 598M, hybrid sliding+global, 32k | Strix | long context without quadratic prefill |
| `frs-l-moe` | 1–3B, 16 experts top-2, NF4, MLA | Spark | capacity at low active bytes |
| `frs-s` in **bf16** | the same shape, bf16 weights and training (not a new id: precision is an artefact property) | 2× T4 | the f32 path wastes the T4; this is the training shape |

## What each profile should not do

- **Pi / phone:** no dense 1B, no MoE, no 151k vocab, no 32k context training.
- **i5:** no loop core larger than L3; no batch training on the iGPU.
- **T4:** no f32 training, no model larger than one card, no per-step all-reduce.
- **Spark:** no dense model that ignores the memory; sparsity is the point.
- **Anywhere:** no chain-of-thought shape that trades tokens for accuracy without
  measuring the accuracy.

## Shape gates

A variant is worth shipping only if it passes all of these on its target device:

| gate | measure |
|---|---|
| decode tok/s at 8k context | measured, not calculated |
| prefill tok/s at 8k | measured |
| peak RSS / VRAM | measured |
| quality on the eval suite at equal compute | no worse than the variant it replaces |
| long-context recall at 8k/32k | the needle test, by position |
| tool-call validity | unchanged from the base |
| training hours for 2B tokens on the target | measured once |
| quantisation loss | perplexity delta at the chosen quant |

## What the current plan already commits to

The four tiers fix: looped depth, top-1 MoE from Small up, MQA, 3-bit KV,
ternary experts, NF4 head, 32k vocab for Nano/Small and 151k for Base/Large.
The variants above change one lever at a time, so a regression can be attributed.
Changing two at once (vocab **and** attention) is how a comparison stops being
one.

## Order of experiments

1. **GQA vs MQA at 46M** — cheap, one attention change, immediate KV/quality
   answer on the Pi.
2. **Code tokenizer at 32k** — measured in tokens per Rust/Nix file, then
   quality; this is likely the largest small-model gain.
3. **Hybrid sliding+global** on Small/Base — prefill and memory at 8k.
4. **bf16 on T4** — the training shape; without it the two-T4 plan is 2–3×
   slower than it needs to be.
5. **MoE on Spark** — only after the dense tiers have a quality baseline.
6. **Ternary from scratch** — a research bet; do not mix it into a shipping
   tier until it wins at equal compute.
