# Patch-to-Cluster Attention Research

**Task ID:** `e5b92f76`
**Phase:** 7 (Multimodal Tokenization) — Vision encoder architecture
**Status:** TODO
**Priority:** MEDIUM — architectural synergy with HullKVCache (Task `9059364b`); deferred until
HullKVCache lands but research should precede it

---

## Problem Statement

The standard ViT patch pipeline (via `im2col` or Implicit GEMM) produces a **flat sequence** of
`N_patches` tokens with no explicit spatial grouping. Every token attends to every other token
in O(N²) standard attention, or O(N) in `BlockAttnRes`. Both are sound, but neither exploits
the **spatial coherence** of images:

- Adjacent patches are highly correlated (grass, sky, skin tones form regions, not scattered
  points).
- A "flat sequence" view forces the attention heads to learn spatial proximity implicitly through
  positional embeddings — unnecessary work that consumes capacity.
- For tasks like document understanding or diagram parsing (relevant to FerrisRes agents reading
  code/docs), spatial cluster structure carries direct semantic meaning.

**Patch-to-Cluster Attention (PaCa)** reformulates the visual token sequence as a set of
*learned spatial clusters*. Tokens within a cluster attend cheaply; cross-cluster attention is
compressed. This maps directly onto FerrisRes's block partitioning philosophy already used for
text in `BlockAttnResModel`.

---

## Key Paper

**"Patch-to-Cluster Attention for Efficient Vision Transformers"**
- ArXiv: https://arxiv.org/abs/2203.11942
- Authors: Grainger et al., 2022
- Core technique: Learn a differentiable soft-assignment of patches to clusters using a
  lightweight slot attention mechanism; perform local attention within clusters and global
  attention between cluster centroids.

### Companion / Related Reading

| Paper | URL | Why |
|---|---|---|
| Slot Attention | https://arxiv.org/abs/2006.15055 | Differentiable clustering mechanism used internally |
| Swin Transformer | https://arxiv.org/abs/2103.14030 | Window-based local attention (fixed windows vs. learned clusters) |
| ClusterFormer | https://arxiv.org/abs/2206.07745 | Related clustering approach; post-PaCa |
| EfficientViT | https://arxiv.org/abs/2305.07027 | Cascaded group attention (related spatial efficiency) |

---

## Algorithm Summary

### Two-Level Attention Hierarchy

```
Input patches: [N, D]               (196 patches from 224×224/16)
        ↓ Slot Attention
Clusters: [K, D]                    (K ≪ N, e.g. K=14 clusters from 196 patches)
        ↓ Local attention within clusters
Updated patches: [N, D]
        ↓ Global attention between cluster centroids
Updated clusters: [K, D]
        ↓ Scatter back
Output: [N, D]
```

Key parameters:
- `K`: number of clusters. Typically `K ≈ sqrt(N)`. For N=196, K=14.
- Cluster assignment is soft (each patch belongs to each cluster with a weight).
- The **Slot Attention** module produces cluster centroids by iterative attention over the
  patch sequence — it is differentiable and trained jointly.

### Complexity

| Attention type | Complexity |
|---|---|
| Full ViT | O(N²) = O(196²) ≈ 38k ops |
| Standard PaCa | O(N×K + K²) = O(196×14 + 196) ≈ 2.9k ops |
| BlockAttnRes PaCa | O(N×K / B + K) where B is block size |

For N=196, K=14: **13× fewer attention operations** vs vanilla ViT. On the 800 MHz integrated
GPU bus this directly reduces memory bandwidth usage during the vision encoder forward pass.

---

## FerrisRes Integration Points

### What Changes

| Module | Change |
|---|---|
| `model/config.rs` | Add `n_clusters: Option<u32>` — None = flat ViT, Some(K) = PaCa |
| `model/block_attn_res.rs` | Add `PaCaLayer` variant that wraps `SlotAttention` + local/global attention |
| `compute/kernels/slot_attn.rs` | **New**: WGSL slot attention kernel |
| `compute/kernels/prefill_attn.rs` | Accept cluster mask for local attention |
| `model/image_preprocessor.rs` | Unchanged |

### New Struct: `PaCaLayer`

```rust
pub struct PaCaLayer {
    /// Soft-assignment: patches → clusters
    slot_attn: SlotAttentionModule,     // produces [K, D]
    /// Local attention: within-cluster self-attention
    local_attn: LocalPrefillAttnOp,     // operates on [n_per_cluster, D]
    /// Global attention: between cluster centroids [K, D]
    global_attn: PrefillAttnOp,
    n_clusters: u32,
}
```

### WGSL: Slot Attention

Slot attention is iterative (typically 3 iterations):
```
iter 0: centroids = learned_init [K, D]
for t in 0..n_iters:
    attn = softmax(patches @ centroids^T / sqrt(D))   # [N, K]
    centroids = attn^T @ patches                       # [K, D]
    centroids = LayerNorm(GRU(centroids, prev))        # update with gating
```

The GRU update makes it non-trivial in WGSL. Options:
1. **Simplified**: Replace GRU with LayerNorm + additive residual. Loses some expressivity but
   is easy to implement in WGSL and still produces good clusters per the ablations in the paper.
2. **Full GRU in WGSL**: Implement a minimal GRU cell (2 weight matrices, sigmoid + tanh).
   Feasible but ~3× more kernel code.

Recommendation: Start with the simplified variant for FerrisRes's integrated GPU target.

### Connection to HullKVCache (Task `9059364b`)

The cluster centroids `[K, D]` output by PaCa are *semantically compressed representations of
image regions*. These are structurally identical to the block representations `b_n` accumulated
by `BlockAttnResModel` for text. This means:

- The `HullKVCache` O(log N) lookup structure could be extended to visual clusters.
- Instead of a flat sequence of visual tokens competing for attention, the model sees `K=14`
  cluster centroids that can be indexed with the same convex hull binary search as text blocks.
- This makes Phase 7 vision tokens first-class citizens in the `HullKVCache` — same O(log K)
  lookup cost as text blocks.

This is the "Patch-to-Cluster as the visual equivalent of HullKVCache" design the prompt
alludes to.

---

## Integration Constraints

1. **Training requirement**: Unlike ToMe (`c9e2d541`), PaCa's Slot Attention module contains
   *learned parameters* (`init_centroids`, optionally GRU weights). A pre-trained ViT cannot
   use PaCa zero-shot — it must be either:
   a. Trained from scratch with PaCa layers, or
   b. Fine-tuned from a ViT checkpoint with PaCa layers added (1–5% accuracy drop during
      adaptation converges within 10k steps per the paper).
2. **Cluster count stability**: K=14 from K=sqrt(196). For 336×336 images N=441, K≈21. The
   `BlockAttnResConfig.n_clusters` should default to `None` and be set at model init time as
   `(n_patches as f32).sqrt() as u32`.
3. **KV cache**: After PaCa, the "sequence" seen by subsequent transformer layers is `[K, D]`,
   not `[N, D]`. The `ModelKVCache` must be sized for `K`, not `N`. This is a significant
   structural change to the inference pipeline that must be coordinated with ToMe (`c9e2d541`).
4. **No production PaCa weights available**: Unlike ToMe which works on existing ViT weights,
   PaCa requires dedicated training. The Phase 9 model loader will not find off-the-shelf PaCa
   weights in Safetensors/GGUF. This task is a **green-field training task**.

---

## Research Questions to Resolve

Before implementation begins, answer:

1. **What is the quality impact of the simplified (non-GRU) slot attention on CIFAR / ImageNet
   benchmarks?** The paper reports numbers for the full GRU variant — we need the simplified
   variant numbers.
2. **Does `K = sqrt(N)` hold for non-square patch grids?** The X1 Yoga is 16:10; images may
   not be square.
3. **Does PaCa's cluster attention compose correctly with `BlockAttnRes`'s inter-block
   pseudo-query mechanism?** Block AttnRes expects the hidden states to be token-aligned with
   the block representation sums. Clusters break this alignment.
4. **HullKVCache compatibility**: Can the convex hull construction from `9059364b` be applied
   over cluster centroids from PaCa in the same pass?

---

## Testing Plan

| Test | Method |
|---|---|
| Slot attention convergence | Verify centroids are stable after 3 iters on random patch input |
| Cluster assignment sum-to-one | Softmax columns of assignment matrix sum to 1.0 |
| Local attention masking | Patches in different clusters do not cross-attend |
| Full layer forward | Compare output shape `[N, D]` to expected |
| Integration with `BlockAttnResLayer` | PaCa wrapping one block layer; forward pass completes |

---

## Dependencies / Blockers

- **Blocked by**: Phase 7 `VisionEncoder` (same as tasks `b3f74a12`, `c9e2d541`).
- **Strongly recommended after**: HullKVCache (`9059364b`) is at least designed, so PaCa
  cluster centroids can be architected to feed directly into it.
- No external Rust crate changes needed.

---

## Estimated Effort

| Subtask | Estimate |
|---|---|
| Simplified slot attention WGSL | 4 h |
| Local/global attention mask handling | 3 h |
| `PaCaLayer` Rust wrapper | 3 h |
| Config + wiring | 1 h |
| Tests | 3 h |
| Research questions resolution (reading) | 3 h |
| **Total** | **~17 h** |
