# Prefill performance

Status: **analysis, 2026-11-26.** Prefill is compute-bound. The important finding
is that the current path is far below the machine, so kernel and thread work can
beat the prompt-cache work in absolute terms — and both compound.

## The roofline, first

Prefill FLOPs ≈ `2 × active_params × prompt_tokens` (forward only).

| model | 8k prompt | at 5 GFLOP/s | at 40 GFLOP/s |
|---|---:|---:|---:|
| Small 111M | 1.82 TFLOP | ~6 min | ~45 s |
| Base 598M | 9.8 TFLOP | ~33 min | ~4 min |

The measured 0.6B teacher runs at **1.9 tok/s at 512**, which is ~2.3 GFLOP/s.
An i5-6500 with AVX2 has a theoretical fp32 peak near 100–200 GFLOP/s; 20–40
GFLOP/s effective is a reasonable target for blocked, multi-threaded kernels.
So today's prefill is roughly **10–50× off the machine**, not off the model.

That changes the priority: caching avoids the work, but **the work itself is
also badly underpriced**.

## The levers, ranked

| # | lever | expected | effort | evidence |
|---|---|---|---|---|
| 1 | **Don't prefill: prefix/KV cache** | **implemented** at `9960441`: 2,000 → 500 prefilled, equality 1.19e-7; CLI wiring open | done (library) | `runtime-caches.md`; the NVIDIA context-API paper reports 15% from suffix reuse alone |
| 2 | **Fewer tokens** (retrieval top-k, compact tool results, skeleton reads, tool-block selection, compaction) | ~linear in FFN cost | hours–days | `token-economics.md`, `reasoning-and-retention.md` |
| 3 | **Parallel matmul over 4 cores** for prefill, not only decode | their comments claim 4–8× on rayon paths; the status says the **MoE prefill path is serial** | hours | `cpu_moe.rs:forward_parallel` has no prefill call site; `cpu_linear.rs`, `ternary.rs` are rayon-parallel |
| 4 | **Fused/tiled kernels**: RMSNorm+QKV, fused attention (`PrefillAttnOp`), blocked scores, avoid materialising the full matrix | 2–4× prefill plausible | days | kernel work already gave 2.6× prefill; the 512-token quadratic is visible in the teacher numbers |
| 5 | **Native quantised compute** (ternary/NF4) instead of dequantising GGUF to f32 | 2–4× compute, less cache pressure | days–weeks | `gguf.rs` dequantises on load; 0.6B takes 5.9 GB RSS |
| 6 | **iGPU prefill, CPU decode** (split by stage) | 1.5–3× prefill if dispatch overhead is low | days | HD 530 shares DDR4; prefill is compute-bound, decode is not |
| 7 | **Chunked prefill** (512-token chunks, emit the first token after the first chunk) | total unchanged; TTFT cut to one chunk | hours | pairs with streaming |
| 8 | **Attention shape**: sliding window + a few global layers for long prompts | attention cost from O(n²) to ~O(n·w) | weeks | `hardware-shapes.md`; at 8k attention is a large share |
| 9 | **Scheduling hygiene**: rayon pool = 4, no build running, no oversubscription | removes the 2.5× load penalty | minutes | the measured 2.5× gap was on a loaded host |
| 10 | **Batching** multiple requests' prefills | throughput, not single-user latency | days | only matters for a server with concurrency |

Levers 1 and 3 are the immediate pair: one avoids the work, the other makes the
remaining work 4–8× cheaper. Lever 2 is the product-level equivalent of 1.

## Concrete code sites

- `src/model/cpu_moe.rs`: `forward_parallel` exists and is rayon-parallel, but
  the prefill path does not call it — wire it for `seq > 1` and benchmark.
- `src/model/cpu_linear.rs` / `src/model/ternary.rs`: parallel matmul exists;
  check the prefill call sites use it.
- `src/model/gguf.rs`: dequantisation to f32 at load; a native quantised prefill
  would remove a copy and a cache miss per weight.
- `src/compute/kernels/prefill_attn.rs`: confirm the prefill attention is tiled
  and does not materialise a full `seq × seq` score buffer.
- `src/model/block_attn_res.rs:forward_prefill`: the place to instrument
  per-stage timing.

## Measure before and after

Break one 2k-token prefill into embedding, per-layer norm, QKV, scores, softmax,
value, O, FFN, and the final norm; report ms and effective GFLOP/s per stage.
Then report tokens/s at 512/2k/8k, on an idle host, with the rayon pool size and
the core count recorded. Without the breakdown, "prefill is slow" is an opinion.

## What not to do

- **Speculative prefill.** No reliable analogue of draft decoding for a prompt.
- **Bigger batches on a single-user path.** It raises TTFT.
- **Caching a partial prefill across different prefixes.** Key the cache by the
  exact token prefix; a mismatch is a wrong answer, not a slow one.
- **Optimising before the cache.** A fast prefill of a prompt you could have
  cached is wasted work.
