# Changelog

All notable changes to FerrisRes are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
FerrisRes uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0 onwards.
Pre-1.0 minor versions may contain breaking changes.

---

## [Unreleased]

### Added
- `FusedPatchEmbedOp` — Implicit GEMM fused patch embedding WGSL kernel; eliminates the
  explicit `im2col` intermediate buffer (~59 MB at 224×224 RGB) by computing patch pixels
  on-the-fly inside a tiled 16×16 workgroup shader (`compute/kernels/fused_patch_embed.rs`)
  (task `b3f74a12`)
- `MatMulDoubleBufferOp` — double-buffered tiled matmul variant using FlashAttention-3
  async structural principles; prefetches tile `t+1` into a second shared-memory slot while
  accumulating over tile `t`, maximising compiler freedom to hide global-memory latency
  (`compute/kernels/matmul.rs`) (task `f4c0a839`)
- `ElasticRagStore` — Matryoshka-aware RAG document store; searches using only the leading
  `query_dim` dimensions of stored `d_max` embeddings, where `query_dim` is mapped from
  `EmbedProfile` (`Integrated`→64, `LowEnd`→128, `MidRange`→256, `HighEnd`→d_max)
  (`inference/rag.rs`) (task `d7a18c03`)
- `EmbedProfile` enum — hardware-tier enum mirroring `DeviceProfile` for use in pure CPU
  RAG/Engram contexts without importing the GPU device module (`inference/rag.rs`)
- Two-stage coarse-to-fine search on `ElasticRagStore` — `search_coarse_then_fine()` scans
  all docs at `coarse_dim` then re-ranks top-k candidates at full `d_max`
- `.cargo/config.toml` — sets `RUST_TEST_THREADS=1` to prevent GPU context race in async
  kernel tests
- Research task documents for five new Phase-7 / cross-cutting investigation areas:
  `implicit_gemm_fused_patching_research.md`, `token_merging_tome_research.md`,
  `matryoshka_embeddings_research.md`, `patch_to_cluster_attention_research.md`,
  `flashattn3_async_wgsl_research.md` (all in `papers_research/`)
- Researcher agent YAMLs: `researcher-implicit-gemm`, `researcher-token-merging`,
  `researcher-matryoshka`, `researcher-patch-to-cluster`, `researcher-flashattn3-wgsl`
  (`.engram/agents/`)

### Changed
- `compute/kernels/mod.rs` — exposes `fused_patch_embed` module
- `compute/mod.rs` — re-exports `FusedPatchEmbedOp` and `MatMulDoubleBufferOp`
- `lib.rs` — adds `FusedPatchEmbedOp`, `MatMulDoubleBufferOp`, `ElasticRagStore`,
  `EmbedProfile` to the public crate API
- `ROADMAP.md` — Phase 7 progress updated; new Cross-Cutting WGSL Efficiency section;
  Phase 6 task table given explicit ID column

### Tests
- 11 new tests in `tests/kernel_tests.rs`: fused patch embed (no-bias, with-bias, byte-size),
  double-buffer matmul (2×2, 32×64×32 numerical equivalence), ElasticRagStore (profile dims,
  top-1 search, reduced-dim search, coarse-then-fine, apply_profile, empty)
- Total: **188 tests passing** (was 75)
 two-stage KV cache quantisation engine — 2-bit, 2.5-bit, 3-bit, 4-bit compression modes with outlier channel splitting (`compute/turboquant.rs`, `compute/kernels/turboquant_kernels.rs`)
- `LoRA` adapter — low-rank fine-tuning with merge/unmerge and per-module targeting (`training/lora.rs`)
- Composable logit processor pipeline — repetition penalty, frequency/presence penalty, temperature, top-k, top-p (`inference/logit_processors.rs`)
- Prompt template support — ChatML, Llama 2, Mistral, Alpaca, Raw formats (`inference/prompt_templates.rs`)
- Context extension — YaRN NTK-aware RoPE scaling, StreamingLLM attention sinks, position interpolation (`inference/context_extension.rs`)
- RAG pipeline — dense (cosine), sparse (TF-IDF), hybrid retrieval with document chunking (`inference/rag.rs`)
- BPE tokeniser with domain vocabulary extension and adaptive entropy-based patching (`model/tokenizer.rs`)
- `AsyncGradientOffload` — multi-stage GPU→CPU gradient staging pool (`training/async_offload.rs`)
- `CpuGradientBuffer` — CPU-side gradient accumulation for integrated GPUs (`training/cpu_offload.rs`)
- MoE expert dispatch/gather WGSL kernels (`compute/kernels/moe.rs`)
- im2col image-patch extraction kernel for vision inputs (`compute/kernels/im2col.rs`)
- `ImagePreprocessor` — resize and normalise inputs for vision pipeline (`model/image_preprocessor.rs`)
- `DeviceProfile` auto-detection with four tiers: `Integrated`, `LowEnd`, `MidRange`, `HighEnd`
- `ModelShard` / `ShardManager` for tensor-parallel layer distribution

### Changed
- Repository reorganised for public release: docs moved to `docs/`, research notes consolidated under `papers_research/`
- Internal agent tooling (`.engram/`, `skills/`) removed from version control

---

## [0.1.0] — Initial development snapshot

### Added
- wgpu/Vulkan GPU backend with `GpuBuffer` and `GpuTensor` abstractions
- WGSL compute shaders: tiled MatMul, RMSNorm, Softmax, RoPE, flash-decode attention, causal mask, prefill attention, elementwise ops
- `BlockAttnResModel` and `BlockAttnResLayer` — O(n) linear-time transformer via two-level block attention hierarchy
- `TwoPhaseInference` prefill + decode pipeline with per-layer `ModelKVCache`
- `TokenGenerator` with `generate_stream` channel for streaming token delivery
- `Sampler` — argmax, temperature, top-k, top-p on CPU from read-back logits
- Reverse-mode autodiff engine — computation graph, backward pass, gradient accumulation (`autodiff/`)
- `SgdOptimizer`, `AdamOptimizer`, `CrossEntropyLoss` — GPU-side parameter updates
- `CheckpointStore` — activation checkpointing at `PerBlock` / `PerLayer` / `PerAttention` granularity
- CLI with `train`, `infer`, `benchmark`, `info` subcommands
- Nix dev-shell with pinned Rust toolchain and Vulkan validation layers
