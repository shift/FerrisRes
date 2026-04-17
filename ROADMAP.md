# FerrisRes Technical Roadmap

> **FerrisRes** is a Rust-native AI inference and training engine built around **Block AttnRes**, a linear-time transformer architecture. It runs on any GPU via wgpu (Vulkan, Metal, DX12, WebGPU), auto-adapts to hardware, and has zero Python dependency.

## Current Status

| Metric | Value |
|---|---|
| Source code | ~65,000 lines across 118 modules |
| Test suites | 1216 lib tests passing, 0 failures |
| Language | 100% Rust (safe + WGSL compute shaders) |
| GPU backends | Vulkan, Metal, DX12, WebGPU via wgpu |
| Tasks completed | **212 / 212 (all complete)** |
| License | AGPL-3.0-or-later |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (clap)                           │
│     train --lora-rank / infer --template --image --yarn     │
├─────────────────────────────────────────────────────────────┤
│                     Inference Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ TokenGenerator│ │ UnifiedToken │  │ Prompt           │  │
│  │ (BlockAttnRes)│ │ Generator    │  │ Templates        │  │
│  │              │ │ (AnyModel:   │  │ (ChatML/Llama2   │  │
│  │              │ │  Standard or │  │  /Mistral/Alpaca  │  │
│  │              │ │  BlockAttnRes)│ │  /Raw)           │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────┴───────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Logit        │  │ Sampling     │  │ KV Cache         │  │
│  │ Processors   │  │ (argmax/temp │  │ (Per-layer GPU   │  │
│  │ (repetition→ │  │  /top-k/top-p│  │  + TurboQuant    │  │
│  │  freq/pres→  │  │              │  │  2-bit compress  │  │
│  │  temp→topk→  │  │              │  │  + PagedAttention│  │
│  │  topp→sample)│  │              │  │  + compaction)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ RAG Pipeline │  │ Tool Search  │  │ DECS             │  │
│  │ (dense/sparse│  │ Registry     │  │ (reasoning token │  │
│  │  hybrid +    │  │ (keyword/    │  │  optimizer,      │  │
│  │  Matryoshka  │  │  embedding/  │  │  plateau detect) │  │
│  │  elastic)    │  │  hybrid)     │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ ToMeMerger   │  │ HullKVCache  │  │ LLM-Computer     │  │
│  │ (bipartite   │  │ (2D convex   │  │ (CALM VM: LookUp │  │
│  │  soft match, │  │  hull attn,  │  │  → Compute →     │  │
│  │  token merge)│  │  O(log n))   │  │  BranchIf)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Speculative  │  │ PagedAttention│  │ Cross-Modal      │  │
│  │ Decoding     │  │ (vLLM-style  │  │ Attention        │  │
│  │ (n-gram draft│  │  block pool, │  │ (text/vision/    │  │
│  │  + verify)   │  │  COW, prefix │  │  audio fusion)   │  │
│  │              │  │  sharing)    │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Video Token  │  │ Streaming    │                         │
│  │ Compression  │  │ I/O Pipelines│                         │
│  │ (temporal    │  │ (image/audio │                         │
│  │  redundancy, │  │  /video:     │                         │
│  │  motion comp,│  │  progressive │                         │
│  │  4-8× reduce)│  │  decode)     │                         │
│  └──────────────┘  └──────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│                      Model Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ BlockAttnRes │  │ Standard     │  │ Architecture     │  │
│  │ Model/Layer  │  │ Transformer  │  │ Dispatcher       │  │
│  │ O(n) + back  │  │ O(n²) compat │  │ (auto-detect:   │  │
│  │              │  │ mode         │  │  safetensors/GGUF│  │
│  └──────────────┘  └──────────────┘  │  → AnyModel)     │  │
│  ┌──────────────┐  ┌──────────────┐  └──────────────────┘  │
│  │ Safetensors  │  │ GGUF Loader  │  ┌──────────────────┐  │
│  │ (F32/F16/    │  │ (v2/v3, Q8_0 │  │ Audio Encoder    │  │
│  │  BF16, shard)│  │  Q4_0/Q4_K/  │  │ (EnCodec-style,  │  │
│  │              │  │  Q5_K/Q6_K)  │  │  RVQ codebooks)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ BPE Tokenizer│  │ BLT Tokenizer│  │ QA-Token         │  │
│  │ + Domain     │  │ (byte-level, │  │ (quality-aware)  │  │
│  │ Vocabulary   │  │  entropy     │  │                  │  │
│  │              │  │  patching)   │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ VisionEncoder│  │ VQ-VAE       │  │ ModelShard       │  │
│  │ (Implicit    │  │ Codebook     │  │ (F32/F16/I8/I4)  │  │
│  │  GEMM + ToMe)│  │ (EMA, multi- │  │                  │  │
│  │              │  │  codebook)   │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Training Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ LoRA         │  │ Gradient     │  │ Autodiff         │  │
│  │ (merge/      │  │ Checkpointing│  │ (ComputationGraph│  │
│  │  unmerge,    │  │ (closure-    │  │  + backward)     │  │
│  │  hot-swap)   │  │  based       │  │                  │  │
│  │              │  │  recompute)  │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Adam/SGD     │  │ Tile-Based   │  │ Partial          │  │
│  │ Optimizers   │  │ Gradient     │  │ Backpropagation  │  │
│  │              │  │ Accumulation │  │ (layer freeze,   │  │
│  │              │  │              │  │  selective bwd)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Compute Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ wgpu Runtime │  │ TurboQuant   │  │ Flash Attention  │  │
│  │ (Vulkan/     │  │ (Outlier     │  │ (FlashDecode +   │  │
│  │  Metal/DX12/ │  │  Channel     │  │  PrefillAttn)    │  │
│  │  WebGPU)     │  │  Split,      │  │                  │  │
│  │              │  │  2.5-bit)    │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Tensor       │  │ Pipeline     │  │ 3D Factored      │  │
│  │ Parallelism  │  │ Parallelism  │  │ Convolution      │  │
│  │ (weight      │  │ (GPipe/1F1B  │  │ (temporal +      │  │
│  │  sharding)   │  │  schedules)  │  │  spatial)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Device Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ DeviceProfile│  │ Capability   │  │ Hardware Tuning  │  │
│  │ (GPU vendor  │  │ Detection    │  │ (workgroup size, │  │
│  │  + memory)   │  │              │  │  coalescing,     │  │
│  │              │  │              │  │  compute params) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                 Distributed / Hardware Layer                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Cloud GPU    │  │ ANE / NPU    │  │ RDMA / DirectGPU │  │
│  │ Orchestrator │  │ Op Placement │  │ (NVLink/RoCE/    │  │
│  │ (workers,    │  │ (auto route  │  │  InfiniBand/TCP) │  │
│  │  fault tol,  │  │  ops to GPU  │  │                  │  │
│  │  cost sched) │  │  or ANE)     │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Features (all implemented)

### Core
- **Block AttnRes transformer**: O(n) inference via block-partitioned attention with full backward pass
- **Standard transformer compatibility**: O(n²) mode for loading LLaMA/Mistral/Gemma models
- **Architecture dispatcher**: auto-detect model type from weights, `--arch` CLI override
- **AnyModel enum**: unified interface for BlockAttnRes + Standard transformer
- **wgpu GPU runtime**: Vulkan, Metal, DX12, WebGPU — runs on NVIDIA, AMD, Intel, Apple, Qualcomm

### Model Loading
- **Safetensors loader**: F32/F16/BF16, multi-shard, architecture detection
- **GGUF loader**: v2/v3, Q8_0/Q4_0/Q4_K/Q5_K/Q6_K dequantization, name mapping
- **QuantizedBuffer**: F32/F16/INT8/INT4 with real bit-packing and per-block scales

### Inference Optimizations
- **TurboQuant**: Outlier Channel Splitting, 2.5-bit fractional precision, GPU kernels
- **FlashDecode**: single-query O(n) decode via tiled KV scan
- **PrefillAttn**: batched O(n²) causal self-attention with online softmax
- **YaRN context extension**: NTK-aware frequency scaling for extended context
- **StreamingLLM**: attention sinks + segmented KV cache compaction
- **Speculative decoding**: n-gram draft model + rejection sampling verification
- **PagedAttention**: vLLM-style block management, copy-on-write, prefix sharing
- **ToMe token merging**: bipartite soft-match merging for vision tokens
- **Circular KV buffer**: virtual ring buffer for streaming KV cache

### Training
- **Gradient checkpointing**: closure-based recompute (ADR-010)
- **LoRA**: merge/unmerge, auto-populate, hot-swap adapters
- **Autodiff**: ComputationGraph + backward pass
- **Adam/SGD optimizers**
- **Tile-based gradient accumulation**: memory-efficient large-batch training
- **Partial backpropagation**: layer freeze, selective backward, gradual unfreezing

### Multimodal
- **VisionEncoder**: implicit GEMM fused patch embedding + ToMe merge
- **EnCodec audio encoder**: strided conv encoder + residual vector quantization (8 codebooks)
- **Cross-modal attention**: Q from text, K/V from vision/audio, early/mid/late fusion
- **Modality type embeddings**: text/vision/audio learnable type IDs
- **VQ-VAE codebook**: nearest-neighbor lookup, EMA updates, commitment+codebook loss, multi-codebook modes
- **Streaming image I/O**: progressive patch extraction, tiled reading for large images
- **Streaming audio I/O**: chunked window processing, SPSC ring buffer, streaming EnCodec encoding
- **Streaming video I/O**: frame sampling, temporal buffering, progressive decode
- **Video token compression**: temporal redundancy removal, motion-compensated residuals, cross-frame token merging (4-8× reduction)
- **3D/factored convolution**: temporal (T×1×1) + spatial (1×H×W) decomposition with WGSL kernels

### Tokenizers
- **BPE tokenizer**: byte-pair encoding with DomainVocabulary for specialized tokens
- **BLT tokenizer**: Byte Latent Transformer — raw UTF-8 bytes, entropy-based dynamic patching, cross-patch attention
- **QA-Token**: quality-aware tokenization with confidence-weighted vocabulary

### Distributed & Hardware
- **Tensor parallelism**: split weight matrices across N GPUs, all-reduce after attention/FFN
- **Pipeline parallelism**: assign layers to different GPUs, GPipe and 1F1B schedules
- **Weight sharding**: split_rows/cols with reconstruct, scatter/gather primitives
- **Cloud GPU orchestration**: worker registration, shard assignment, gradient aggregation, fault tolerance, cost-aware spot scheduling
- **Apple Neural Engine (ANE)**: automatic op placement (GPU for matmul/attention, ANE for BN/activation), unified memory buffers
- **RDMA/DirectGPU**: NVLink, RoCE, InfiniBand, TCP fallback with bandwidth/latency estimates

### WGSL Compute Kernels
- Tiled matmul (16×16 + double-buffer)
- RMSNorm, Softmax (online), CausalMask
- RoPE (in-place), Elementwise (add/scale/ReLU/copy)
- FlashDecode + Tiled, PrefillAttn (batched causal)
- FusedPatchEmbed (implicit GEMM), im2col
- MoE routing, ToMeMerge
- TurboQuant (rotation, quantize, dequantize, QJL)
- FFT + Mel-spectrogram
- Temporal/Spatial Conv3D
- Circular KV buffer

### Distillation
- **Gemma 4 → Block AttnRes**: structural linearization from O(n²) to O(n)
- **Real model verified**: 9.6 GB Gemma 4 27B Multimodal IT (2.66B params, 35 layers, GQA)
- **Memory-mapped loader**: `MmapedSafetensors` avoids loading entire model into RAM
- **GQA support**: 8 query heads / 1 KV head with correct Q/K/V dimensions
- **Teacher-student memory optimization**: pre-compute teacher logits, drop teacher, reload for student (fits in 16 GB RAM)
- **GPU-accelerated forward**: DeviceProfile-aware JIT weight uploads, hybrid CPU/GPU matmul
- **KL divergence loss**: temperature-scaled soft target matching
- **CLI distill command**: `--config e2b/e4b/27b-mm`, `--gpu`, `--seq-len`, `--steps`

### Self-Improvement Loop
- **WASM sandbox**: wasmi runtime, embedded brace-checker module, fuel limits, memory bounds — zero-trust tool execution
- **LSP-as-Oracle**: JSON-RPC LSP client for rust-analyzer/pyright/clangd, fallback syntax checker, `compiler_error_loss` for autodiff
- **Mirror Test**: recursive self-verification — model generates code, generates tests, executes tests, failures → backprop loss
- **Speculative Block Decoding**: tiny BlockDraftModel (~10M params) predicts block summaries, main model verifies, 8x token throughput
- **Persistent Concept Memory**: `ConceptMap` with embedding-based retrieval, quality scoring, LRU eviction, JSON persistence, `ConceptHullBridge` for Hull-KV integration
- **Host tools**: web_fetch, math_eval, file_read/write, shell_exec, search, code_interpreter — 7 tools with dispatch router

## Phase Completion

| Phase | Status | Description |
|---|---|---|
| 1–3 | ✅ Done | wgpu foundation, BlockAttnRes model, tiered compute, caching |
| 4 | ✅ Done | Autodiff, training, tokenizer, embedding, benches |
| 5 | ✅ Done | Streaming inference, RoPE, KV cache, flash-decode, logit processors |
| 6 | ✅ Done | TurboQuant, LoRA, RAG, YaRN, templates, DECS, HullKVCache, LLM-Computer |
| 7 | ✅ Done | Vision, audio, video, cross-modal, streaming I/O, VQ-VAE, BLT, 3D convolution, video compression |
| 8 | ✅ Done | Distributed tensor/pipeline parallelism, cloud GPU, RDMA, ANE/NPU |
| 9 | ✅ Done | Weight loading (safetensors, GGUF), standard transformer, architecture dispatcher |
| 10 | ✅ Done | Gemma 4 distillation pipeline, GPU forward pass, mmap loader, GQA, teacher-student memory optimization |
| 11 | ✅ Done | Self-improvement loop: WASM sandbox, LSP-as-Oracle, Mirror Test, Speculative Block Decoding, Concept Memory, host tools |
| 12 | ✅ Done | v0.2.0: Block AttnRes edge-case benchmarks, API stabilisation, quickstart/architecture/deployment/API-reference docs |
| 13 | ✅ Done | Output modalities: VisionHead, SpeechHead, VideoHead, TTS stream, VLA ActionHead, Scientific (SMILES/Mesh/GCode), TactileHead |
| 14 | ✅ Done | FerrisRes Armor: L0 regex+bloom, L1 neural scanner, L2 RepE probe, L3 sanitizer, GPU-accelerated distillation |
| 15 | ✅ Done | Profile-driven dispatch: DispatchPlan per-op CPU/GPU, Intel iGPU detection, auto-tiling, --gpu flag removed |
| 16 | ✅ Done | Real distillation verified: Gemma 4 27B on Intel HD 530, 27M tokens, checkpoint resilience |

**All tasks complete — 1189 tests passing, 0 failures.**

## Cognitive Architecture

FerrisRes includes a full cognitive architecture for self-improving AI inference:

```
Layer 0: CognitivePipeline wiring    ✅ PR #30
Layer 1: Memory & Learning           ✅ PR #31
Layer 2: Autonomy                    ✅ PR #32
Layer 3: Self-Improvement            ✅ PR #33
Layer 4: Emergence                   ✅ PR #34
```

### Cognitive Pipeline (Layer 0)
The cognitive pipeline orchestrates all cognitive components:
- **ConceptMap**: Persistent learned patterns with embedding-based retrieval
- **LlmComputer (CALM VM)**: Deterministic computation via LookUp → Compute → BranchIf
- **MirrorTest**: Recursive self-verification — model generates code, tests, and loss signals
- **HullKVCache**: 2D convex hull attention with O(log n) lookups
- **WasmSandbox**: Zero-trust tool execution with wasmi runtime

### Memory & Learning (Layer 1)
- **EpisodicMemory**: Event-based experience storage (not token-based). Stores (prompt, tool traces, outcome, quality, importance). Content-based retrieval via cosine similarity + recency bias. Compression merges similar episodes.
- **DifferentiableLlmComputer**: Gumbel-Softmax op selection with Straight-Through Estimator. NTM-style DiffMemoryBank for gradient flow through memory. Temperature annealing.
- **ToolTriggeredLora**: On-the-fly LoRA weight updates from the 'learn' tool. Elastic Weight Consolidation (Fisher diagonal) prevents catastrophic forgetting. Progressive adapter stacking.

### Autonomy (Layer 2)
- **ToolCreationPipeline**: Model generates tool specs via `[tool_create]` blocks. 6-stage validation (name, code size, structure, safety, syntax, semantics). Refinement loop with max 3 retries.
- **PlanExecutor**: Multi-step `[plan]` with `$N` reference resolution. Condition evaluation, retry on failure, replanning from failed step.
- **ToolUsageTracker**: Per-tool + per-context EMA quality tracking. Contextual bandit for best-tool recommendation. JSON persistence.

### Self-Improvement (Layer 3)
- **AbstractionEngine**: Scans concepts for clusters (cosine > 0.8), computes centroid meta-concepts, compresses N → 1. Hierarchical levels: Instance → Pattern → Principle → MetaPrinciple.
- **IntrinsicMotivation**: Per-concept uncertainty tracking (entropy + quality + distance). Zone of Proximal Development goal selection. Learning progress tracking. Mastery detection.
- **ProactiveController**: 4-level autonomy (Reactive → Suggestive → SemiAutonomous → FullyAutonomous). Initiative signals: concept degradation, tool obsolescence, knowledge gaps, memory pressure. Action logging with rollback.

### Emergence Measurement (Layer 4)
- **EmergenceBenchmark**: Quantitative emergence measurement across 6 categories:
  - **Skill Acquisition**: improvement rate (baseline vs augmented) over repeated attempts
  - **Self-Correction**: MirrorTest error recurrence rate, correction rate
  - **Self-Extension**: self-created tool count × average reuse × max chain depth
  - **Cognitive Scaffolding**: concept count vs task diversity correlation
  - **Planning**: plan success rate and depth improvement over time
  - **Abstraction**: compression ratio × abstraction levels reached
- Baseline (no pipeline) vs augmented (with pipeline) delta = emergent capability
- Trend analysis: increasing, stable, decreasing, insufficient data
- Emergence index: composite 0.0–1.0 score across all categories
- JSON persistence for cross-session measurement
- 27 tests
