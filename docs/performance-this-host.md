# Performance on this host

Host: i5-6500, 4 cores, 6 MB L3, DDR4 (see the measured bus below), Intel HD 530
(shared memory, Vulkan), 32 GB RAM.

## The measured bus, and what it says about the GPU path

Measured with `intel_gpu_top` while training on the HD 530 (2026-11-26):

| counter | value |
|---|---|
| IMC reads | up to **7800 MiB/s** (≈8.2 GB/s) |
| IMC writes | 750–1900 MiB/s |
| Render/3D engine busy | **~30%** |

Two conclusions, both measured rather than assumed:

1. **The 6.3 GB/s used by the tier tables is too low.** It was an estimate; the
   controller serves at least 8.2 GB/s of reads. Decode figures computed at 6.3
   are pessimistic by ~30%: Small becomes ~228 tok/s rather than ~163.
2. **GPU training here is memory-bound, not dispatch-bound.** Bandwidth is at the
   ceiling while the EUs sit at 30%. So more threads or wider SIMD buy almost
   nothing on this host, and cutting bytes buys up to ~3.3× before compute takes
   over: f16 activations ≈ 2×, f16 plus fused elementwise ops ≈ 3×.

This is also why the T4 result (326 tok/s, dispatch-bound) and this host cannot be
compared directly: the two are limited by different things.
 Every number is labelled **measured**,
**calculated** or **unknown**; a calculated number is not a promise.

## Decode (generation)

`model-tiers.md` calculates from the layout at 6.3 GB/s, 8k context. The one
measured run on a loaded machine was **2.5× below the bus ceiling**, so the
realistic column applies that factor; it is still an estimate.

| tier | bytes/token (layout) | calculated tok/s | realistic tok/s | status |
|---|---:|---:|---:|---|
| **Nano** 46M | 17 MB | ~420 | ~170 | calculated |
| **Small** 111M | 36 MB | ~163 | ~65 | calculated |
| **Base** 598M | 195 MB | ~49 | ~20 | calculated |
| **Large** 1B | 299 MB (top-1) | ~49 | ~20 | calculated |

Measured on this host, but not on a FerrisRes tier:

- Qwen3-0.6B **f32 CPU**: **7.73 tok/s** at 128 tokens, **1.90 tok/s** at 512
  (quadratic prefill, no quantisation benefit because the loader dequantises).
- A small test model after the kernel work: **8.1× decode FFN**, **~7.7×**
  end-to-end decode, 2.6× prefill. These are speedups against the previous code,
  not a tier's tok/s.
- The current `infer` path maps a teacher into an untrained student and took
  ~96 s for 8 tokens. It is not a model you can use; do not quote it as one.

## Prefill and first-token latency

Decode is bandwidth-bound; prefill is compute-bound. On this CPU the only
measured prefill is the 0.6B f32 teacher: 1.90 tok/s at 512. A 111M ternary
student is ~5× smaller and quantised, so 20–40 tok/s is a plausible range at
512 — **unknown**, and it is the number the smoke must measure.

The UX consequence is larger than decode:

| prompt | at 20 tok/s prefill | at 40 tok/s |
|---:|---:|---:|
| 1k tokens | ~50 s | ~25 s |
| 4k tokens | ~3.4 min | ~1.7 min |
| 8k tokens | ~6.8 min | ~3.4 min |

So **the prefix cache is not an optimisation; it is the difference between
usable and not**. Warm the soul/tool prefix, cache the conversation prefix, and
compact before the window fills. A cold 8k prefill on this host is minutes.

## The iGPU

The HD 530 shares DDR4. Decode stays bandwidth-bound, so the iGPU does not
multiply decode; prefill is the only place it can help, and it is unmeasured.
Expect a modest prefill gain at best, not a different class of machine. Vulkan
works with the loader/ICD env in `hardware-matrix.md`.

## Training

The working trainer is the f32 CPU reference. A step costs roughly `6 · N` FLOPs
per token (forward + backward); at an effective 20–40 GFLOP/s on four cores that
is ~30–60 tok/s for a 111M model. A 2B-token run is therefore **hundreds of
days**, not hours. This host is for correctness and smoke tests, not training.

Measured training throughput that does exist:

- **GPU loop, Intel HD 530**: tiny config, 200 steps, loss 3.557 → 0.0026 in
  **32.8 s** on the wgpu path (`73a6902`). No T4 measurement yet; attention is
  one invocation per head per token, so the T4 number is the next thing to record.
- Armor L1 (2.2M, 128 tokens): full corpus, ~10k examples, **3,997 s**
  (~2.5 examples/s) on this host.
- `train` learns and resumes (loss 3.65 → 0.0009 on the overfit test), but no
  tokens/s at scale is recorded.

## Specialists

| model | expected | status |
|---|---|---|
| Armor L1 2.2M | ~2.5 examples/s scan | measured |
| Embedding head (768→256) | milliseconds per text on CPU, batch-dependent | unknown |
| Draft (Nano) | adds 20–50% decode overhead, recovers 2–3× | predicted |
| Router / decision head | sub-millisecond | predicted |

## How to improve prefill

The levers, the roofline and the measurement plan are in
`prefill-performance.md`. The prefix KV cache is implemented (`9960441`):
2,000 → 500 prefilled on a 1,500-token shared prefix, equality 1.19e-7, but not
wired to `infer` yet.

## What to measure next

1. `benchmark` on an exported Small: decode tok/s at 128, 1k, 8k.
2. Prefill tok/s for Small at 512/2k/8k — the number that decides UX.
3. The training smoke (`train_learns`) extrapolated to a real config, for
   tokens/s on this CPU.
4. The iGPU prefill with the Vulkan env, against the CPU.

Until those exist, the honest summary is: **Nano ~170, Small ~65, Base/Large ~20
tok/s decode if the 2.5× gap holds; first-token latency is dominated by prefill
and needs the prefix cache; training on this host is not viable.**
