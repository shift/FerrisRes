# FerrisRes model tiers

Four sizes, all Block AttnRes with a looped middle, top-1 MoE, one KV head
(MQA), 3-bit KV, ternary experts and an NF4 head. The numbers are **calculated
from the layout** at 6.3 GB/s, ~40 GFLOP/s effective, 8k context — not
measured. The one measured run on a loaded machine was 2.5× below the bus
ceiling, so treat the tok/s column as a design constraint until `benchmark`
runs on a quiet host.

| tier | hidden | stored / visits | experts | vocab | params | disk | tok/s (i5, 8k) | KV @32k | train 2B tok |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **Nano** | 512 | 6 / 7 | 2 | 32k | 46M | 17 MB | ~420 | 0.9 MB | 8 h |
| **Small** | 768 | 8 / 10 | 2 | 32k | 111M | 36 MB | ~163 | 2.7 MB | 18 h |
| **Base** | 1024 | 12 / 16 | 4 | 151k | 598M | 195 MB | ~49 | 5.4 MB | 100 h |
| **Large** | 1024 | 12 / 16 | 8 | 151k | 1013M | 299 MB | ~49 | 5.4 MB | 169 h |

Training hours assume 2B tokens and 20 TFLOP/s effective on a T4; Kaggle's
30 h/week quota therefore means ~0.6, 3.3 and 5.6 weeks for Small, Base and
Large.

## What each tier is for

- **Nano** — router, tool selector, Armor-class scanner, embeddings, and the
  draft model for speculative decoding. Not a chat model.
- **Small** — the first shipping assistant: instant on a Telegram turn, 36 MB,
  fits a Pi 5.
- **Base** — quality tier for reasoning, longer answers and code.
- **Large** — capacity at the same decode speed, because top-1 reads one expert
  per visit. Worth it only if Base plateaus on quality and RAM allows.

## The three rules that set the sizes

1. **Vocabulary sets the head, and the head sets the floor.** At 151k vocab the
   head is 78 MB per token — 66% of Base's bytes at 8k and 26% of its
   parameters. At 111M it would be half the model, so Small and Nano use 32k.
2. **The loop core must fit L3.** Nano's one 512-wide layer (~1.5 MB) fits a
   Pi 5's 2 MB; Small's two 768-wide layers (~4.6 MB) fit the i5's 6 MB but not
   the Pi; Base's two 1024-wide layers (~5.9 MB) fit the i5 only. A core that
   does not fit re-reads RAM every pass.
3. **Top-1 experts trade storage for capacity, not speed.** This is why Large
   exists at the same tok/s as Base, and why the family is defined by width and
   depth first, experts second.

## Context

Hybrid attention: sliding window 1024 with 1–4 global visits. KV with MQA and
3-bit is small (0.9–10.6 MB at 32k across the tiers), so context is limited by
**compute**, not memory: Base is ~36 tok/s at 32k and ~26 at 64k. Training at
2k–4k and extending with YaRN is the plan; claiming 32k without the extension
phase would be a number the model cannot use.

## Target hardware (calculated)

Decode is memory-bound on both; the context penalty that dominates the i5
disappears at 256 GB/s.

| tier | MB/token | Strix Halo 256 GB/s | DGX Spark 273 GB/s |
|---|---:|---:|---:|
| Nano | 13.3 | ~9,600–13,500 tok/s | ~10,300–14,400 tok/s |
| Small | 27.1 | ~4,700–6,600 | ~5,000–7,100 |
| Base | 117.2 | ~1,100–1,500 | ~1,200–1,600 |
| Large | 117.3 | ~1,100–1,500 | ~1,200–1,600 |

Training 2B tokens at 50% of FP16 peak: Small ~12 h / 7 h, Base ~66 h / 40 h,
Large ~113 h / 68 h (Strix / Spark). With 128 GB unified, Base with Adam fits
without offload. These are roofline estimates; the engine's Vulkan path on
RDNA 3.5 and Blackwell is unverified and does not use tensor cores.

## What not to build

- 3B and above: ~2 tok/s at 6.3 GB/s, and 2B tokens of from-scratch training is
  20+ weeks of free quota.
- 7B/27B/75B: explicitly out.
- A 151k-vocab 111M: half the model would be a lookup table.

## Specialists

Armor L1 (~2.5M), a draft model (Nano), an embedding model, and a ranker/router.
Each is acquired by capability through the model runtime, each has its own card
and gate, and none is referenced by filename.
