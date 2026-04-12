# Token Merging (ToMe) Research

**Task ID:** `c9e2d541`
**Phase:** 7 (Multimodal Tokenization)
**Status:** DONE ✅
**Priority:** HIGH — addresses Token Explosion on the X1 Yoga before it happens

---

## Problem Statement

When Phase 7 lands a `VisionEncoder`, a single 224×224 image at patch size 16 produces **196
visual tokens** fed into `BlockAttnResModel`. A 336×336 image at the same patch size produces
**441 tokens**. At block attention complexity O(n) these are individually survivable, but:

- In a multimodal agent session, the model might receive *multiple* images per turn.
- A 5-image turn at 224×224 → **980 visual tokens + text tokens**. On an 800 MHz integrated
  GPU with shared DRAM KV cache this saturates `TurboQuant`'s 2-bit budget fast.
- Many visual tokens encode *redundant* information (e.g., uniform sky, text background, blank
  table surface). Attending to all of them wastes compute for no quality gain.

**Token Merging (ToMe)** solves this by merging the most similar (redundant) token pairs at each
transformer layer without retraining.

---

## Key Paper

**"Token Merging: Your ViT But Faster"**
- ArXiv: https://arxiv.org/abs/2210.09461
- Authors: Bolya et al., ICLR 2023
- GitHub: https://github.com/facebookresearch/ToMe
- Core technique: At each transformer layer, find the top-r most similar token pairs using a
  bipartite soft matching algorithm on the key vectors, merge them (averaged), and proceed with
  fewer tokens. **No retraining required.**

### Companion / Related Reading

| Paper | URL | Why |
|---|---|---|
| DynamicViT | https://arxiv.org/abs/2106.02034 | Token pruning (requires training); useful contrast |
| EViT (Efficient ViT) | https://arxiv.org/abs/2202.07800 | Token reorganization; training required |
| SPViT | https://arxiv.org/abs/2111.11802 | Soft pruning with training |
| LLaVA-1.5 token reduction | https://arxiv.org/abs/2310.03744 | Practical multimodal token budget |

**Key advantage of ToMe over alternatives:** zero retraining. This aligns perfectly with
FerrisRes's "drop-in at inference" philosophy (cf. TurboQuant, LoRA merge/unmerge).

---

## Algorithm Summary

### Bipartite Soft Matching

Given N tokens at a transformer layer:
1. Split tokens into two sets A (odd positions) and B (even positions), each of size N/2.
2. For each token in A, compute cosine similarity to all tokens in B using their **key vectors**
   (reusing the existing QKV computation — no extra projection needed).
3. Greedily match each A token to its most similar B token (no two A tokens share a B token).
4. Merge the top-r matched pairs: replace each pair with the average of their values, weighted by
   the token sizes (to maintain a correct average when multiple merges chain).
5. Propagate the merged tokens through the rest of the layer.

The hyperparameter `r` controls the speed/quality tradeoff:
- `r=0` → original ViT, no change.
- `r=N/2` → maximum merging, maximum speedup, small quality drop.
- Typical sweet spot: `r ≈ 8–16` per layer, reducing token count by ~30–50% across 12 layers.

### Token Size Tracking

After merging, each surviving token carries a **size** (count of original tokens it represents).
During attention, the softmax is weighted by token sizes so larger merged tokens contribute
proportionally to the attention scores. This requires a small adjustment to the `prefill_attn`
and `flash_decode` WGSL kernels.

---

## FerrisRes Integration Points

### What Changes

| Module | Change |
|---|---|
| `model/config.rs` | Add `tome_r: Option<u32>` — tokens to merge per layer (None = disabled) |
| `model/block_attn_res.rs` | After QKV projection, before attention: call `ToMeMerger::merge(keys, r)` |
| `compute/kernels/tome.rs` | **New**: WGSL kernel for bipartite matching + merge |
| `compute/kernels/prefill_attn.rs` | Accept optional `token_sizes` buffer; weight softmax accordingly |
| `inference/kv_cache.rs` | Handle variable-length sequence after merging |
| `model/image_preprocessor.rs` | Unchanged |

### New Struct: `ToMeMerger`

```rust
pub struct ToMeMerger {
    r: u32,                        // tokens to merge per layer
    pipeline: wgpu::ComputePipeline,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl ToMeMerger {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>, r: u32) -> Self { ... }

    /// Returns (merged_tokens, token_sizes)
    /// merged_tokens: [N - r, D]
    /// token_sizes:   [N - r]     (f32 weights for size-weighted attention)
    pub fn merge(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &GpuBuffer,    // [N, D_head]
        values: &GpuBuffer,  // [N, D]
        n_tokens: u32,
        d_head: u32,
    ) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer)> { ... }
}
```

### WGSL Strategy

The bipartite matching involves an argmax over similarities — not trivially parallelisable, but
for the typical case N ≤ 500 tokens the matching can be done in a single workgroup using
workgroup-shared memory:

1. **Similarity kernel** (`tome_similarity.wgsl`): compute A×B cosine similarity matrix
   `[N/2, N/2]`. Dispatch `(N/2, N/2)`.
2. **Greedy match kernel** (`tome_match.wgsl`): single-thread greedy assignment (CPU-side for
   small N; GPU-side `workgroup_size(1)` with shared mem for N > 256).
3. **Merge kernel** (`tome_merge.wgsl`): scatter-add matched pairs into output buffer, write
   `token_sizes`.

For typical visual tokens (N ≤ 441 at 336×336), a CPU-side matching pass with GPU merge is
acceptable latency. Profile to decide.

---

## Performance Projections

Based on the ToMe paper (ViT-B/16 on ImageNet):

| r per layer | Throughput gain | Accuracy drop (ImageNet) |
|---|---|---|
| 8 | ~1.2× | ~0.1% |
| 16 | ~1.5× | ~0.3% |
| 32 | ~2.0× | ~1.0% |

For FerrisRes on integrated GPU, the gains are amplified because:
- Fewer tokens → smaller KV cache → less DRAM pressure on the 800 MHz bus.
- `TurboQuant` compresses fewer vectors → smaller compressed KV block.
- `BlockAttnRes` inter-block attention query cost scales with token count — merging compounds
  savings across all blocks.

---

## Integration Constraints

1. **Variable-length sequences**: After ToMe, sequence lengths are no longer uniform across
   images in a batch. The existing `BlockAttnResModel` operates on fixed shapes. Either:
   a. Restrict ToMe to single-image inference (simplest).
   b. Add padding + mask support to `prefill_attn.rs`.
2. **Token unmerging for output tasks**: For dense prediction (segmentation, depth) you'd need
   to unmerge. For FerrisRes's text generation pipeline this is irrelevant — the visual tokens
   are only consumed as KV context, not decoded back to pixels.
3. **KV cache invalidation**: If ToMe is applied at every layer, the KV cache layout changes per
   layer. `ModelKVCache` must store per-layer `n_tokens_after_merge` to size buffers correctly.
4. **Phase 6 HullKVCache interaction**: The 2D attention heads rely on a fixed query head
   structure. ToMe changes the token count but not the head dimension — compatible, but verify.

---

## Testing Plan

| Test | Method |
|---|---|
| Bipartite match correctness | CPU reference implementation; compare to WGSL output |
| Token size weighted attention | Numerical comparison with unmerged reference |
| Quality regression | Run inference on a captioning task; measure BLEU vs r=0 baseline |
| Benchmark | tokens/sec vs `r` for 224×224 and 336×336 on `Integrated` profile |
| Edge case: r ≥ N/2 | Clamp r to N/2 - 1; assert no panic |

---

## Dependencies / Blockers

- Phase 7 `VisionEncoder` must exist (same blocker as Implicit GEMM task `b3f74a12`).
- `prefill_attn.rs` needs `token_sizes` support — can be added as optional binding.
- No external crate changes needed; ToMe is pure compute.

---

## Estimated Effort

| Subtask | Estimate |
|---|---|
| Similarity + greedy match (CPU-side) | 2 h |
| Merge WGSL kernel | 3 h |
| Rust wrapper `ToMeMerger` | 2 h |
| `prefill_attn` size-weighted softmax | 2 h |
| Config + wiring into `BlockAttnResLayer` | 1 h |
| Tests + benchmark | 2 h |
| **Total** | **~12 h** |
