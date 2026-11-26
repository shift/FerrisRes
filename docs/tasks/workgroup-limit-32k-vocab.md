# A 32k vocabulary cannot be trained: the dispatch exceeds 65535 workgroups

Status: **blocks every real training run on every device.** Found 2026-11-26 on a
T4 with `0.2.7`, `--hidden-dim 768 --tokenizer <32k> --seq-len 512 --batch-size 4`.

## The failure

```
In a CommandEncoder, label = 'gpu_train_backward'
  In a dispatch command, indirect:false
    Each current dispatch group size dimension ([98304, 1, 1]) must be less or equal to 65535
```

`98_304 × 256 = 25_165_824 = 32_768 × 768` — the **lm_head weight gradient**
(`vocab × hidden`). The op dispatches one workgroup per 256 elements in **x
only**, and x is capped at 65,535 by the spec. It fails the moment a gradient
tensor exceeds `65_535 × 256 = 16.7M` elements:

| vocab | hidden | elements | groups | result |
|---:|---:|---:|---:|---|
| 259 | 768 | 198,912 | 777 | works (why this was never seen) |
| 8,192 | 768 | 6.3M | 24,576 | works |
| **32,768** | **768** | **25.2M** | **98,304** | **panic** |
| 151,552 | 1,024 | 155M | 605,000 | would panic |

So the engine's usable vocabulary at hidden 768 is **~21,800**, and the first
student is specified at 32k.

## The fix

Dispatch a **2D grid** and let the shader flatten it. In `TrainOps::dispatch`
(`src/autodiff/gpu_ops.rs`), replace the one-dimensional count with

```rust
let gx = groups.min(65535);
let gy = (groups + 65535 - 1) / 65535;
```

and pass `gx` to the shader in its params so the flat index becomes

```wgsl
let flat = (gid.y * p.grid_x + gid.x) * 256u + lid.x;
```

Affected ops are the ones whose element count scales with `vocab × hidden` or
`seq × hidden × batch`: the elementwise family, `mul_bcast_grad_w`, `silu_mul`,
the rope/gqa helpers, and whatever emits the lm_head weight gradient. Each has
its own params struct, so this is per-shader work, not one line — but it is
mechanical and each fix is testable in isolation.

## The test

Name it `no_dispatch_exceeds_the_spec_limit`. It must run one forward+backward on
the smallest model that still exceeds the limit (hidden 768, vocab 32,768,
seq 8, batch 1) and assert every dispatch's group count is within 65,535. On a
host without a GPU it reports "skipped: no adapter", not a pass.

A cheaper unit-level guard belongs in `TrainOps::dispatch` itself: assert
`gx <= 65_535 && gy <= 65_535` with the tensor name in the message, so the next
oversized tensor names itself instead of panicking in wgpu.

## Workaround, not a fix

Train at a vocabulary at or below ~21,800 at hidden 768 — for example a 16k BPE.
That works today and would let a session run, but the student is specified at 32k,
so the 2D grid is the real answer.

## Out of scope

The iGPU device-lost at 768×256 (`igpu-device-lost.md`), which is a different
defect: that one has no validation error and happens with the 259 vocabulary.
