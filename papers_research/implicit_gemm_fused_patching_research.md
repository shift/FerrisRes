# Implicit GEMM / Fused Patching Research

**Task ID:** `b3f74a12`
**Phase:** 7 (Multimodal Tokenization)
**Status:** DONE ✅
**Priority:** HIGH — directly replaces the memory-blowing `im2col` buffer on integrated GPU

---

## Problem Statement

FerrisRes already has a working `im2col` WGSL kernel (`compute/kernels/im2col.rs`). The approach is
correct and passes tests, but it carries an explicit **N × K²** memory expansion:

- An `H×W×C` image with patch size `P` produces `(H/P)×(W/P)` patches.
- Each patch is `P × P × C` floats.
- `im2col` materialises the entire patch matrix as a flat buffer: `(H/P)*(W/P) * P*P*C` f32 values.
- For a 224×224 RGB image at patch size 16: `(14×14) × (16×16×3)` = **14.7 M floats = ~59 MB**.
- On the Lenovo X1 Yoga with shared DRAM and 800 MHz integrated GPU bus, this creates measurable
  bandwidth pressure *before* the matmul projection even starts.

The goal is to **eliminate the intermediate buffer** by fusing patch extraction directly into the
linear projection dispatch.

---

## Key Paper

**"Anatomy of High-Performance Deep Learning Convolutions on GPUs"**
- ArXiv: https://arxiv.org/abs/1803.05594
- Authors: Tao et al., NVIDIA, 2018
- Core technique: Implicit GEMM — the patch extraction coordinate mapping is performed *inside*
  the GEMM kernel rather than as a separate data rearrangement pass.

### Companion / Related Reading
| Paper | URL | Why |
|---|---|---|
| cuDNN Implicit GEMM (NVIDIA) | https://docs.nvidia.com/deeplearning/cudnn/developer-guide/ | Reference implementation concepts |
| "Fast Algorithms for Convolutional Neural Networks" (Lavin & Gray 2016) | https://arxiv.org/abs/1509.09308 | Winograd reference, useful contrast |
| ViT "An Image is Worth 16×16 Words" | https://arxiv.org/abs/2010.11929 | Establishes that `im2col`+linear is the canonical ViT patch embedding |

---

## Core Idea

In a standard patch embedding:

```
image [H, W, C]
   ↓  im2col (explicit copy)
patches [N_patches, P*P*C]          ← THIS IS THE EXPENSIVE BUFFER
   ↓  matmul with weight [P*P*C, D]
embeddings [N_patches, D]
```

In Implicit GEMM:

```
image [H, W, C]                     ← stays in place, no copy
   ↓  single fused kernel
embeddings [N_patches, D]           ← only the output is written
```

Inside the fused kernel, each output element `[patch_idx, embed_dim]` is computed by:
1. Decoding `patch_idx` → `(row, col)` in patch-grid coordinates.
2. Iterating over the `P*P*C` inner dimension, computing `img[row*P+dy, col*P+dx, c]` on the fly.
3. Accumulating `img_pixel * weight[p, embed_dim]`.

---

## FerrisRes Integration Points

### What Changes

| Module | Current | After |
|---|---|---|
| `compute/kernels/im2col.rs` | Stand-alone extraction kernel | Keep for backwards-compat / testing; mark as `im2col_explicit` |
| `compute/kernels/fused_patch_embed.rs` | Does not exist | **New kernel** — Implicit GEMM patch embedding |
| `model/image_preprocessor.rs` | Outputs raw f32 tensor | Unchanged; feeds directly into fused kernel |
| `model/config.rs` | `patch_size` field used for `im2col` | Add `use_implicit_gemm: bool` toggle |
| Phase 7 `VisionEncoder` | Planned: `im2col` → matmul | Replace with single fused dispatch |

### WGSL Sketch

```wgsl
struct Params {
    height: u32,          // image H
    width: u32,           // image W
    channels: u32,        // image C
    patch_size: u32,      // P
    embed_dim: u32,       // D (projection output)
    n_patches: u32,       // (H/P) * (W/P)
}

@group(0) @binding(0) var<storage, read>       img:     array<f32>;  // [H, W, C]
@group(0) @binding(1) var<storage, read>       weight:  array<f32>;  // [P*P*C, D]
@group(0) @binding(2) var<storage, read_write> out:     array<f32>;  // [N_patches, D]
@group(0) @binding(3) var<uniform>             params:  Params;

@compute @workgroup_size(16, 16)
fn fused_patch_embed(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let patch_idx  = gid.x;   // which patch
    let embed_idx  = gid.y;   // which output dimension

    if (patch_idx >= params.n_patches || embed_idx >= params.embed_dim) { return; }

    let patches_per_row = params.width / params.patch_size;
    let p_row = patch_idx / patches_per_row;
    let p_col = patch_idx % patches_per_row;

    var acc: f32 = 0.0;
    let kk = params.patch_size * params.patch_size * params.channels;
    for (var k: u32 = 0u; k < kk; k = k + 1u) {
        let c        = k % params.channels;
        let local_xy = k / params.channels;
        let ly       = local_xy / params.patch_size;
        let lx       = local_xy % params.patch_size;
        let img_y    = p_row * params.patch_size + ly;
        let img_x    = p_col * params.patch_size + lx;
        let img_idx  = (img_y * params.width + img_x) * params.channels + c;
        acc = acc + img[img_idx] * weight[k * params.embed_dim + embed_idx];
    }
    out[patch_idx * params.embed_dim + embed_idx] = acc;
}
```

### Memory Savings (224×224, C=3, P=16, D=768)

| Step | `im2col` path | Implicit GEMM |
|---|---|---|
| Intermediate patch buffer | **~59 MB** | **0 MB** |
| Output embedding buffer | 14×14×768 × 4B ≈ 0.6 MB | 0.6 MB |
| Peak VRAM during embed | ~59.6 MB | **~0.6 MB** |

For an 800 MHz integrated GPU with shared DRAM, eliminating 59 MB of write-then-read halves the
time spent on patch embedding.

---

## Integration Constraints

1. **Tiling**: The inner `P*P*C` loop in the WGSL sketch is correct but not tiled. For `P=16,
   C=3` the inner loop is 768 iterations — small enough to skip tiling. For larger C or P, add
   workgroup-shared tiling matching `matmul.rs`'s 16×16 tile strategy.
2. **wgpu compute limits**: The workgroup dispatch `(N_patches, D)` must stay within
   `max_compute_workgroup_size_x`. Use `dispatch_workgroups_indirect` or chunked dispatch for
   high-res images (>512×512).
3. **DeviceProfile awareness**: `Integrated` profile should prefer implicit GEMM unconditionally.
   `HighEnd` may still benefit but the absolute savings are proportionally smaller.
4. **Backward pass (training)**: The implicit GEMM kernel currently targets inference. If Phase 7
   includes fine-tuning the vision encoder, a matching backward kernel must be written or the
   explicit `im2col`+`matmul` path retained for training only.

---

## Testing Plan

| Test | Method |
|---|---|
| Numerical equivalence | Run `im2col` + `matmul` and `fused_patch_embed`; compare outputs to 1e-4 tolerance |
| Memory pressure | Use `sysinfo` to confirm no intermediate allocation above baseline |
| Benchmark | `benches/` entry: measure μs/image for 224×224, 336×336, 448×448 on integrated profile |
| Edge cases | Non-divisible H/W (padding), P=32, channels=4 |

---

## Dependencies / Blockers

- Phase 7 `VisionEncoder` struct not yet implemented.
- `fused_patch_embed.rs` can be developed standalone and wired in once `VisionEncoder` lands.
- No external Rust crate changes required.

---

## Estimated Effort

| Subtask | Estimate |
|---|---|
| WGSL kernel (non-tiled, correctness) | 2–4 h |
| Tiled variant for large patches | 2–3 h |
| Rust wrapper + pipeline + bind group | 2 h |
| Tests + benchmark | 2 h |
| ROADMAP / CHANGELOG update | 30 min |
| **Total** | **~9–12 h** |
