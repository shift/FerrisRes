# FlashAttention-3 Async Principles for WGSL Research

**Task ID:** `f4c0a839`
**Phase:** Cross-cutting — Phases 3, 6, 7 (Inference Engine + Architecture + Multimodal)
**Status:** DONE ✅
**Priority:** MEDIUM — architectural quality improvement; not a blocker for any Phase 7 feature,
but compounds all of them by reducing CPU-to-GPU latency on the 800 MHz shared bus

---

## Problem Statement

FerrisRes currently dispatches WGSL compute passes synchronously: each kernel submits a command
buffer to the wgpu queue, the CPU calls `queue.submit(...)`, and the next kernel is enqueued
only after the previous one is submitted. This is correct but does not exploit:

1. **Asynchronous data movement**: While the GPU executes kernel N, the CPU could be uploading
   the inputs for kernel N+1 to GPU-side staging buffers.
2. **Pipelined decode**: During the `flash_decode` attention pass (token by token), the CPU
   currently waits idle for each token's KV cache read to complete before issuing the next
   matmul. On a 800 MHz shared-memory bus this idle time is a significant fraction of wall
   clock time.
3. **Overlapping TurboQuant dequantization**: `turboquant_kernels.rs` dequantizes KV blocks
   just before they're consumed by attention. With async pipelining, dequantization of block K+1
   could overlap with attention over block K.

FlashAttention-3 (Dao et al., 2023) — designed for H100 Tensor Core hardware — introduces
**warp specialisation** and **asynchronous data movement** as first-class design principles.
The H100-specific WGMMA instructions cannot be backported to WGSL, but the **structural
patterns** — interleaving compute and data movement across independent parallel units — are
architecture-agnostic and directly applicable to wgpu's command encoder model.

---

## Key Paper

**"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"**
- ArXiv: https://arxiv.org/abs/2407.08608
- Authors: Shah et al., Princeton / Together AI, 2024
- Core ideas:
  1. **Producer-Consumer warp specialisation**: Different warps handle data load vs. compute.
  2. **Software pipelining of GEMM stages**: Prefetch next tile while computing current tile.
  3. **Pingpong buffering**: Double-buffer shared memory so load and compute never stall each
     other.
- Hardware target: NVIDIA H100 (Hopper). The WGMMA / TMA instructions are H100-only.
- **Backportable principle**: Double-buffering and overlapping dispatch are achievable in wgpu
  via multiple command encoders and `queue.submit` with callbacks.

### Companion / Related Reading

| Paper | URL | Why |
|---|---|---|
| FlashAttention (original) | https://arxiv.org/abs/2205.14135 | Block-tiled online softmax — already influences `flash_decode.rs` |
| FlashAttention-2 | https://arxiv.org/abs/2307.08691 | Work partitioning improvements |
| "Efficient GPU Kernels for N:M Sparse Weights" | https://arxiv.org/abs/2104.01136 | Double-buffering pattern in CUDA; WGSL analogue research |
| wgpu Async Compute | https://docs.rs/wgpu/latest/wgpu/struct.Queue.html | wgpu submit / callback API |
| "Dissecting the Ampere GPU Architecture" | https://arxiv.org/abs/2208.11174 | Async copy + compute overlap on consumer GPUs |

---

## Core Concepts to Backport

### 1. Double-Buffered Tile Loading

In `flash_decode.rs` and `matmul.rs`, the inner tiling loop loads a tile from global memory
into `var<workgroup>`, computes, then loads the next tile. Each load must complete before
compute begins (enforced by `workgroupBarrier()`).

With double buffering:
```
// Pseudocode
tile_A_current, tile_A_next: workgroup arrays

// Prefetch tile 0
load(tile_A_current, global[0])
workgroupBarrier()

for t in 1..n_tiles:
    // Start prefetch of tile t+1 while computing tile t-1
    load_async(tile_A_next, global[t])   // hypothetical
    compute(tile_A_current)
    workgroupBarrier()
    swap(tile_A_current, tile_A_next)
```

**WGSL reality**: WGSL does not have explicit async load primitives (`ldmatrix` equivalent).
However, the *compiler* on Metal/Vulkan backends may issue async loads when it detects
independent load + compute sequences. The programmer's job is to **structure the WGSL so the
dependency chain is explicit** and the backend can optimise it.

Concretely: split the single workgroup loop in `matmul.rs` into a two-stage approach that
maximises distance between the `array<workgroup>` write and read, giving the compiler maximum
freedom to schedule the load early.

### 2. Multi-Encoder Pipelining at the wgpu Level

wgpu's command encoder model allows multiple encoders to be recorded in parallel (on different
CPU threads) and submitted as a batch:

```rust
// Current pattern (sequential):
let mut enc = device.create_command_encoder(&desc);
dequant_op.dispatch(&mut enc, kv_block_n);
flash_decode_op.dispatch(&mut enc, kv_block_n);
queue.submit([enc.finish()]);

// Pipelined pattern (overlapping N and N+1):
let mut enc_n   = device.create_command_encoder(&desc);
let mut enc_n1  = device.create_command_encoder(&desc);

dequant_op.dispatch(&mut enc_n, kv_block_n);
flash_decode_op.dispatch(&mut enc_n, kv_block_n);
dequant_op.dispatch(&mut enc_n1, kv_block_n_plus_1);   // prefetch dequant

queue.submit([enc_n.finish(), enc_n1.finish()]);
// GPU can overlap flash_decode(n) with dequant(n+1) if it has independent execution units
```

On integrated GPUs with a single compute queue this may not provide wall-clock speedup, but on
discrete GPUs with separate copy and compute queues it can. The code structure is correct either
way — the driver decides parallelism.

### 3. TurboQuant Dequant / Attention Overlap

The most concrete immediate win: in the `TwoPhaseInference` decode loop:

```
for each KV block n:
    dequantize(block n)     ← turboquant_kernels.rs
    flash_decode(block n)   ← flash_decode.rs
```

These are currently sequential. If dequantize(n+1) is dispatched *before* flash_decode(n)
completes — in a separate encoder / buffer slot — the GPU can pipeline them on architectures
where the dequant and attention shaders use different functional units. This is the
FlashAttention-3 producer-consumer specialisation, adapted for wgpu.

---

## FerrisRes Integration Points

### What Changes

| Module | Change |
|---|---|
| `compute/kernels/flash_decode.rs` | Restructure inner tile loop for double-buffer friendliness |
| `compute/kernels/matmul.rs` | Same: explicit prefetch structure in tile loop |
| `inference/two_phase.rs` | Overlapping dequant + flash_decode command encoder submission |
| `compute/pipeline.rs` | Add `AsyncComputePipeline` wrapper that manages double-encoder dispatch |
| `device/profile.rs` | Add `supports_async_copy: bool` field; set based on GPU vendor detection |

### New: `AsyncComputePipeline`

```rust
pub struct AsyncComputePipeline {
    /// Double-buffered command encoders
    enc_a: Option<wgpu::CommandEncoder>,
    enc_b: Option<wgpu::CommandEncoder>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl AsyncComputePipeline {
    /// Dispatch op_n on current buffer, op_n1 on lookahead buffer
    pub fn dispatch_pipelined(
        &mut self,
        op_n:   &dyn DispatchableOp,
        op_n1:  &dyn DispatchableOp,
        args_n: &OpArgs,
        args_n1: &OpArgs,
    ) -> Result<()> { ... }

    /// Flush both buffers to queue in order
    pub fn flush(&mut self) -> Result<()> { ... }
}
```

### WGSL Double-Buffer Sketch (matmul)

The current `matmul.rs` has a single `var<workgroup>` double-array. The double-buffer variant
adds a second slot and explicitly indexes between them:

```wgsl
var<workgroup> tile_a: array<f32, 2 * 16 * 16>;   // slot 0 = current, slot 1 = prefetch
var<workgroup> tile_b: array<f32, 2 * 16 * 16>;

@compute @workgroup_size(16, 16)
fn matmul_double_buf(...) {
    // Prefetch tile 0 into slot 0
    tile_a[0 * 256 + local_row * 16 + local_col] = load_a(0);
    tile_b[0 * 256 + local_row * 16 + local_col] = load_b(0);
    workgroupBarrier();

    for (var t: u32 = 0u; t < num_tiles - 1u; t++) {
        let cur  = t % 2u;
        let next = (t + 1u) % 2u;

        // Prefetch next tile into 'next' slot (compiler may overlap with compute below)
        tile_a[next * 256 + local_row * 16 + local_col] = load_a(t + 1u);
        tile_b[next * 256 + local_row * 16 + local_col] = load_b(t + 1u);

        // Compute with current slot
        for (var i: u32 = 0u; i < 16u; i++) {
            acc += tile_a[cur * 256 + local_row * 16 + i]
                 * tile_b[cur * 256 + i * 16 + local_col];
        }
        workgroupBarrier();
    }
    // Last tile
    let cur = (num_tiles - 1u) % 2u;
    for (var i: u32 = 0u; i < 16u; i++) {
        acc += tile_a[cur * 256 + local_row * 16 + i]
             * tile_b[cur * 256 + i * 16 + local_col];
    }
}
```

Note: `var<workgroup>` doubles in size (2 × 256 × 4B = 2 KB per matrix vs 1 KB). Still well
within the 32 KB workgroup memory limit of integrated GPUs.

---

## Expected Benefits on Integrated GPU (X1 Yoga)

| Technique | Expected gain | Caveat |
|---|---|---|
| Double-buffer matmul | 5–15% throughput | Depends on compiler; Mesa/ANV vary |
| Pipelined dequant+decode | 10–20% latency | Single CU may not overlap; verify per-GPU |
| Async encoder submission | Architecture-dependent | May be no-op on single-queue integrated GPU |

These are modest gains — not the 2–3× improvements FlashAttention-3 achieves on H100s. The
value is in **latency smoothing**: removing the CPU idle stalls at the 800 MHz bus boundary
makes the generation feel more responsive even when aggregate throughput is unchanged.

---

## Research Questions to Resolve

1. **Do Mesa/ANV (Vulkan on Intel) and RadeonSI actually pipeline independent compute and
   copy commands within a single submit?** Needs micro-benchmark to confirm.
2. **Does wgpu's `BufferAsyncError` path allow overlap or does it fence?** Check wgpu source.
3. **Is there a WGSL `prefetch` hint or memory order annotation available in the WGSL 2.0
   spec?** Check the spec for `unordered` load semantics.
4. **What is the workgroup memory size limit on Intel Iris Xe (X1 Yoga's GPU)?** Confirm 32 KB
   to validate the double-buffer tiling above.

---

## Testing Plan

| Test | Method |
|---|---|
| Numerical correctness | Double-buffered matmul output == single-buffered output to 1e-5 |
| No regression on `kernel_tests.rs` | Run all 75 existing tests |
| Latency benchmark | `benches/turboquant_benchmarks.rs` — add double-buffered decode bench |
| GPU vendor conditional | Assert `Integrated`/Intel uses pipelined path; HighEnd same |

---

## Dependencies / Blockers

- No Phase 7 features required. This task is purely infrastructure.
- Can be developed and benchmarked today on the existing decode path.
- Requires a micro-benchmark harness for measuring GPU-side latency (not just CPU-side timing).
  The existing `criterion` bench setup in `Cargo.toml` is sufficient.

---

## Estimated Effort

| Subtask | Estimate |
|---|---|
| Double-buffer WGSL for `matmul.rs` | 3 h |
| Double-buffer WGSL for `flash_decode.rs` | 3 h |
| `AsyncComputePipeline` wrapper | 3 h |
| Pipelined dequant+decode in `two_phase.rs` | 2 h |
| Research Q resolution + micro-benchmarks | 4 h |
| Tests + correctness validation | 2 h |
| **Total** | **~17 h** |
