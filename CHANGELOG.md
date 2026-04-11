# Changelog

All notable changes to FerrisRes are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
FerrisRes uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0 onwards.
Pre-1.0 minor versions may contain breaking changes.

---

## [Unreleased]

### Added
- `TurboQuant` two-stage KV cache quantisation engine — 2-bit, 2.5-bit, 3-bit, 4-bit compression modes with outlier channel splitting (`compute/turboquant.rs`, `compute/kernels/turboquant_kernels.rs`)
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
