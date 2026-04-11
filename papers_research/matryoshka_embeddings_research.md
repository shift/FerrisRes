# Matryoshka Representation Learning (MRL) Research

**Task ID:** `d7a18c03`
**Phase:** 7 (Multimodal Tokenization) / 4 (RAG / Engram)
**Status:** TODO
**Priority:** MEDIUM-HIGH — elastic embedding dimensions align directly with the tiered
`DeviceProfile` (`Integrated` → `LowEnd` → `MidRange` → `HighEnd`) already in `device/profile.rs`

---

## Problem Statement

FerrisRes's RAG pipeline (`inference/rag.rs`) currently stores dense embeddings at a fixed
dimension (whatever the model's `hidden_dim` is — e.g., 768 or 1024). On the X1 Yoga:

- A cosine similarity search over K documents computes K dot products of dimension D.
- For D=768, K=10,000: 10,000 × 768 = **7.68M multiply-adds** per query.
- With `TurboQuant` KV compression at 2-bit, the embedding store itself is already space-
  efficient — but the *compute* for RAG lookup still scales with D.
- More critically, the **Engram agent memory** (`.engram/`) could benefit from storing embeddings
  at the lowest dimensionality that still achieves acceptable recall, saving both disk and
  lookup time.

**Matryoshka Representation Learning (MRL)** trains a single embedding model to produce
embeddings where **the first d dimensions are already a high-quality d-dimensional embedding**
for any `d ∈ {32, 64, 128, 256, 512, 1024, ...}`. This means you can truncate to `[:64]` for
fast coarse retrieval and then re-rank with `[:768]` only when needed.

---

## Key Paper

**"Matryoshka Representation Learning"**
- ArXiv: https://arxiv.org/abs/2205.13147
- Authors: Kusupati et al., NeurIPS 2022
- Core contribution: Multi-granularity training loss that forces the first `m` dimensions of
  every embedding to be independently meaningful for any `m` in a predefined set `M`.

### Companion / Related Reading

| Paper | URL | Why |
|---|---|---|
| MRL-E (Efficient MRL) | https://arxiv.org/abs/2309.06944 | Training efficiency improvements |
| Adaptive Retrieval with MRL | https://arxiv.org/abs/2205.13147 (§4.2) | Coarse-to-fine RAG strategy |
| OpenAI text-embedding-3 | https://platform.openai.com/docs/guides/embeddings | Production deployment of MRL |
| MTEB Leaderboard | https://huggingface.co/spaces/mteb/leaderboard | Benchmark for embedding quality |

**Production signal**: OpenAI's `text-embedding-3-small` and `text-embedding-3-large` use MRL
internally. You can truncate to any dimension with only marginal quality loss.

---

## Training Mechanism

### Loss Function

Standard embedding training loss:
```
L_full = ContrastiveLoss(embed[:D], label)
```

MRL loss for a set of nesting dimensions `M = {32, 64, 128, 256, 512, 768}`:
```
L_MRL = Σ_{m ∈ M}  w_m × ContrastiveLoss(embed[:m], label)
```

Where `w_m` is a weight (often uniform: `1/|M|`). The key insight: this forces the *prefix*
subspace `[:m]` to be informative at every resolution.

The loss can be bolted onto FerrisRes's existing `CrossEntropyLoss` / contrastive training by:
1. Adding a `MatryoshkaProjection` head that produces `[B, D_max]`.
2. Summing `len(M)` loss terms, each using a different prefix slice.
3. Backpropagating the summed gradient — the autodiff engine in `autodiff/` handles this
   automatically as long as the sliced projections are differentiable operations.

### LoRA Compatibility

MRL works with LoRA adapters: train the LoRA adapter with the MRL loss. The frozen base weights
provide the bulk of the representation; the adapter specialises the nesting structure for the
target domain.

---

## FerrisRes Integration Points

### What Changes

| Module | Change |
|---|---|
| `model/config.rs` | Add `matryoshka_dims: Option<Vec<usize>>` to `BlockAttnResConfig` |
| `model/lm_head.rs` | Add `MatryoshkaHead` variant that projects to `D_max` and exposes slice |
| `training/optimizer.rs` | Add `MatryoshkaLoss` that sums contrastive losses across nesting dims |
| `inference/rag.rs` | Add `ElasticRagStore` — index at D_max, retrieve at configurable dim |
| `device/profile.rs` | Map `DeviceProfile` → recommended query dim |

### New: `ElasticRagStore`

```rust
pub struct ElasticRagStore {
    /// Full D_max embeddings on disk / in memory
    embeddings: Vec<Vec<f32>>,          // [K, D_max]
    /// Current query dimension (can be changed at runtime)
    query_dim: usize,
    /// Nesting dimensions supported
    matryoshka_dims: Vec<usize>,        // e.g., [32, 64, 128, 256, 768]
}

impl ElasticRagStore {
    /// Set query dimension based on device profile
    pub fn set_query_dim_for_profile(&mut self, profile: &DeviceProfile) {
        self.query_dim = match profile {
            DeviceProfile::Integrated => 64,
            DeviceProfile::LowEnd     => 128,
            DeviceProfile::MidRange   => 256,
            DeviceProfile::HighEnd    => self.matryoshka_dims.last().copied().unwrap_or(768),
        };
    }

    /// Cosine similarity search using only [:query_dim] of stored embeddings
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> { ... }
}
```

### Device-Profile → Embedding Dim Mapping

| DeviceProfile | RAG query dim | Recall vs full | DRAM bytes/lookup (K=10k) |
|---|---|---|---|
| Integrated (X1 Yoga) | 64 | ~95% | 0.64 MB |
| LowEnd | 128 | ~97% | 1.28 MB |
| MidRange | 256 | ~98.5% | 2.56 MB |
| HighEnd | 768+ | ~100% | 7.68 MB |

The "95% recall at dim=64" figure comes from the MRL paper (BEIR benchmark). For agent memory
(Engram) workloads where precision@1 matters more than recall, use dim=128 as the floor.

---

## Engram / Agent Memory Application

The `.engram/` workspace already stores agent context, reasoning chains, and knowledge. If these
are backed by a vector store, MRL enables:

1. **Fast approximate lookup at d=64** for "does this memory exist?" queries.
2. **Full d=768 re-rank** only for the top-10 candidates.
3. **Adaptive storage**: store only `d=128` for old/low-importance memories; `d=768` for
   recent/high-importance ones. The same MRL model generates both.

This mirrors the "Elastic Memory" design pattern described in the FerrisRes project framing.

---

## Training Requirements

MRL is a **training-time** modification. For FerrisRes Phase 7 inference-first approach:

- **Option A (Train from scratch)**: Use MRL loss in `training/optimizer.rs` when training a
  Phase 7 multimodal embedding model. Natural fit.
- **Option B (Post-hoc via linear distillation)**: Apply a lightweight linear transform to
  re-order dimensions of an existing embedding model's output to approximate MRL structure.
  Less accurate but requires no retraining.
- **Option C (Use existing MRL model weights)**: Load a pre-trained MRL embedding model
  (e.g., OpenAI `text-embedding-3`, or a HuggingFace MTEB model with MRL support) via the
  Phase 9 Safetensors loader. FerrisRes does the inference; someone else did the training.

For the X1 Yoga use-case, **Option C** is the fastest path to value.

---

## Integration Constraints

1. **Contrastive loss**: FerrisRes currently has `CrossEntropyLoss` but not a contrastive /
   InfoNCE loss. Required for Option A training.
2. **Phase 9 dependency**: Loading a pre-trained MRL model (Option C) requires the
   `SafetensorsLoader` from Phase 9. That task is currently un-started.
3. **Dimension ordering**: MRL requires the model to be *trained* with the loss. A vanilla
   embedding model does not produce nested subspaces by default. Do not apply MRL truncation
   to a non-MRL-trained model's output — the subspace will be garbage.

---

## Testing Plan

| Test | Method |
|---|---|
| Loss computation | Verify `MatryoshkaLoss` sums `len(M)` terms correctly |
| Subspace quality | Embed test sentences; verify recall@10 at d=64 ≥ 90% of d=768 |
| `ElasticRagStore` search correctness | Compare results at each dim to brute-force reference |
| Profile-dim mapping | Assert `Integrated` → 64, `HighEnd` → 768 |
| Dim out of bounds | Assert error on `query_dim` not in `matryoshka_dims` |

---

## Dependencies / Blockers

- `MatryoshkaLoss` requires a contrastive loss implementation (new).
- `ElasticRagStore` can be implemented today as a RAG module extension — no Phase 7 blocker.
- Phase 9 Safetensors loader needed for Option C.
- Training the MRL head with LoRA requires LoRA already merged into the base model or LoRA
  adapter targeting the projection layer — both already supported.

---

## Estimated Effort

| Subtask | Estimate |
|---|---|
| `MatryoshkaLoss` (sum of contrastive terms) | 3 h |
| `MatryoshkaHead` in `lm_head.rs` | 1 h |
| `ElasticRagStore` with profile mapping | 3 h |
| Config + wiring | 1 h |
| Tests | 2 h |
| **Total** | **~10 h** |
