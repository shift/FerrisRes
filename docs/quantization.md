# Quantization options

Status: **decision record, 2026-11-26.** No quantised model has been evaluated;
the numbers below are layout arithmetic and vendor-reported results.

## Current scheme

| tensor | format | bits |
|---|---|---|
| MoE experts | ternary `{-1,0,+1}` | 1.58 information bits, **packed at 2 bits/weight** (4 values/byte) |
| attention, LM head | NF4 | 4 |
| router, norms | BF16 | 16 |
| KV cache | TurboQuant | 3 |

Training keeps f32/bf16 shadow weights and quantises at export; the packed
decode kernel already reads 2 bits per expert weight, so "1.x bit" describes
the number of levels, not the storage.

For Base (598M, 151k vocab): experts ~104 MB, head ~78 MB, attention ~13 MB,
≈195 MB total.

## Why the head decides the order

At 151k vocabulary the head is 156M parameters and 78 MB in NF4 — **66% of the
bytes per token** at 8k (78 of 117 MB). Compressing the experts from 2.0 to
1.0 bits saves ~52 MB of storage and ~19 MB per token, which is 117 → 98 MB,
about **16% faster decode**. Factorising the head (rank 256) takes 156M to
~39M and the token from 117 to ~60 MB, about **2×**. The head comes first.

## LittleBit-2

Samsung Research, ICML 2026 (arXiv `2603.00042`); predecessor LittleBit,
NeurIPS 2025 (`2506.13771`).

- Factorises each matrix into two **binary** (±1) low-rank factors plus FP16
  row/latent/column scales. The rank sets the effective rate: 1.0 to 0.1 bits
  per weight, below the 1-bit floor of one-bit-per-entry formats.
- LittleBit-2 changes only the **initialisation** (Internal Latent Rotation +
  Joint-ITQ) so the SVD factors align with the binary hypercube; inference is
  unchanged from LittleBit.
- It is **QAT + knowledge distillation**, not post-training quantisation.

Reported quality at 1.0 bpp on Llama-3-8B: 11.53 WikiText-2 perplexity against
6.10 for BF16; 0.55 bpp → 14.01; 0.1 bpp → 23.74. Vendor-reported, not
independently checked. Those are 7–8B models; a 0.6B has far less redundant
capacity to spend.

| concern | assessment |
|---|---|
| storage win | real: 2.0 → 1.0 bits halves the experts, but only ~16% of a token until the head is factorised |
| compute | two narrow binary matmuls + scalings; the engine has no binary-matmul kernel and wgpu will not use tensor cores here |
| quality at 0.6B | the sub-1-bit regime is where small models lose the most; at 1.0 bpp it is already far below BF16 at 8B |
| licence | the reference implementation is **CC BY-NC 4.0**; a commercially licensed engine cannot ship it, and a reimplementation is a legal review |
| effort | new layer type, new kernel, QAT+KD pipeline, on top of a stub `train` and a broken distill LoRA gradient |

**Decision: no, not now.**

## NF2: the experiment that does make sense

Our packed ternary already spends 2 bits per weight. A **4-level symmetric
quantiser with per-group scales** fits the same packing and the same decode
kernel with a different table. That is a representation change at equal
storage, not a new architecture.

Gate: perplexity at equal bytes per token, on the same checkpoint and data. If
the 4-level variant does not beat ternary, ternary stays.

## Other options, ranked

1. **Factorised / cluster head** — the 66% term; 2× per-token bytes.
2. **NF2 experts at equal storage** — free quality if it wins.
3. **QAT with straight-through estimation** — closes the f32 → ternary gap at
   export; the `BitLinear` shadow path exists.
4. **Low-rank binary (LittleBit-shaped)** — only after 1–3, with the licence
   resolved and expert bytes still binding.

## What is not being considered

- 2-bit KV: KV is 5.4 MB at 32k; it is not the bottleneck and the recall cost
  is measurable.
- 2-bit attention/head: the head is the term we are trying to shrink by
  factorisation, not by precision.
- FP8: wgpu exposes no FP8 path.

## References

- LittleBit-2, *Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via
  Latent Geometry Alignment*, ICML 2026, arXiv `2603.00042`.
- LittleBit, *Ultra Low-Bit Quantization via Latent Factorization*, NeurIPS
  2025, arXiv `2506.13771`.
- `SamsungLabs/LittleBit` — reference code, **CC BY-NC 4.0**.
- BitNet b1.58 — ternary QAT, the operating point we currently use.
