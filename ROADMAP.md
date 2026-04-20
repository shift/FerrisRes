# FerrisRes Technical Roadmap

> **FerrisRes** is a Rust-native AI inference and training engine built around **Block AttnRes**, a linear-time transformer architecture. It runs on any GPU via wgpu (Vulkan, Metal, DX12, WebGPU), auto-adapts to hardware, and has zero Python dependency.

## Current Status

| Metric | Value |
|---|---|
| Source code | ~94,000 lines across 158 modules |
| Test suites | 1448 lib tests passing, 0 failures |
| Language | 100% Rust (safe + WGSL compute shaders) |
| GPU backends | Vulkan, Metal, DX12, WebGPU via wgpu |
| Tasks completed | **444 / 493** |
| Tasks in progress | **0** |
| Tasks planned | **41** |
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
- **QLoRA**: NF4 quantized base weights + LoRA adapters
- **Autodiff**: ComputationGraph + backward pass
- **Adam/SGD optimizers**
- **SCALE optimizer**: 12 MB state for edge devices (column-normalized gradients, last-layer momentum)
- **AdaMeM optimizer**: 181 MB state for capable hardware (power iteration SVD, low-rank momentum)
- **WeightOptimizer trait**: `optimizer_for_profile(DeviceProfile)` factory routes SCALE vs AdaMeM
- **Tile-based gradient accumulation**: memory-efficient large-batch training
- **Partial backpropagation**: layer freeze, selective backward, gradual unfreezing
- **1.58-bit ternary quantization**: absmean scaling, 2-bit packing (16× vs FP32), block-wise quantization, STE
- **Block-MoE-Res distillation**: KL logits + MSE hidden states, dense→MoE conversion, checkpoint serialization

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
| 17 | ✅ Done | Cognitive architecture: Layer 0-4 (pipeline wiring, memory & learning, autonomy, self-improvement, emergence measurement) |
| 18 | ✅ Done | Phase 8 integration: consolidation engine, quality propagation, uncertainty feedback, tool exploration, safe learn tool, GGUF CPU inference, API server |
| 19 | ✅ Done | Block-MoE-Res architecture: ternary quantization, SCALE/AdaMeM optimizers, inter-block attention, MoE conversion, distillation, checkpoint serialization |
| 20 | ✅ Done | 1.58-bit inference stack: ternary matmul (6 variants), TernaryLinear, TernaryMoELayer, STE, TernaryBlockAttnResModel |
| 21 | ✅ Done | Edge I/O: 2:4 sparse ternary, expert mmap (.stm), 3-bit TurboQuant KV, recurrent block summary KV, pruning pipeline |
| 22 | 📝 Planned | Expert I/O Pipeline (PreScope + BuddyMoE) |
| 23 | ✅ Done | CPU/GPU Backend Abstraction (YaRN, GpuCapabilities, WGSL) |
| 24 | ✅ Done | Inference pipeline wiring |
| 25 | 📝 Planned | Autonomous Learning Loop (CoVo + SkillKB + FDAL) |
| 26 | 📝 Planned | Elastic Inference (E2B/E4B switching) |
| 27 | 📝 Planned | GPU BlockAttnResLayer (Gemma 4 features + backward kernels) |

**444 tasks done, 0 in progress, 41 planned — 1456 tests passing.**

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

---

## Phase 18: Integration & Real Inference (v0.2.3)

### Consolidation Engine
- **ConsolidationEngine**: Sleep-like memory replay. Selects important unconsolidated episodes, scores by quality, forms new concepts from episode clusters (cosine > 0.8), prunes stale consolidated memories. Preserves high-quality exemplars (quality > 0.9). 16 tests.

### Quality Propagation
- MirrorTest quality now fans out to ALL cognitive modules after each generation:
  - **ConceptMap**: strengthen/weaken retrieved concepts based on quality
  - **IntrinsicMotivation**: record observations per concept with entropy + quality
  - **ProactiveController**: quality degradation alerts when trend drops below 0.4
  - **Episode importance**: quality extremity factor (remember extremes, forget mediocre)

### Uncertainty Feedback
- Text-based entropy proxy (diversity, repetition, hedging) feeds IntrinsicMotivation
- High uncertainty triggers practice goal generation
- Practice goal queue (max 5) for self-directed learning

### Tool-Learning Policy
- ε-greedy tool exploration with autonomy-adjusted epsilon (Reactive=0, FullyAutonomous=0.15)
- Logs suboptimal model choices when tracker data suggests better alternatives

### Safe Learn Tool
- `learn` tool registered in ToolRegistry, dispatched via `execute_tool()`
- 5-layer safety: autonomy gate, rate limiting (5/session, 20/hour, 60s cooldown), quality gate (EWC), audit log

### GGUF/safetensors CPU Inference
- `infer --model-path` loads GGUF or safetensors and runs full autoregressive generation on CPU
- `serve --model-path` loads model once at startup, serves real generation via OpenAI-compatible API
- Supports all 13 model configs: e2b, e4b, 12b, 27b, 27b-mm, 26b-a4b, 31b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b

### API Server
- New `serve` subcommand with `--host`, `--port`, `--model-name`, `--model-path`, `--model-format`, `--tokenizer`
- Real CPU inference on `/v1/chat/completions` and `/v1/completions` when model loaded
- Placeholder responses with instructions when no model loaded

---

## Phase 19: Block-MoE-Res Architecture (In Progress)

### Multimodal City of Experts
The "Multimodal City of Experts" describes the system FerrisRes already is: many input encoders (vision, audio, text, video, cross-modal), many output heads (VisionHead, SpeechHead, VideoHead, TTS, ActionHead, MeshHead, TactileHead, GCode), MoE expert routing (top-2 from 4 experts per token), and resource-adaptive scaling via DeviceProfile.

Phase 19 adds the Block-MoE-Res reasoning engine: inter-block attention for hierarchical reasoning, MoE conversion for expert specialization, ternary quantization for 16× compression, and hardware-aware optimizers for edge training.

### Ternary Quantization (BitNet b1.58) — ✅ Done
- **Absmean quantization**: α = mean(|W|) / √(2/π), W_q = clamp(round(W/α), -1, 0, +1)
- **2-bit packing**: 4 values per byte, 16× size reduction vs FP32
- **Block-wise scaling**: per-block scale for non-uniform weight distributions
- **STE (Straight-Through Estimator)**: forward uses quantized, backward passes through FP32
- **Quality metrics**: MSE, cosine similarity, SNR(dB)
- 13 tests all passing

### SCALE + AdaMeM Optimizers — ✅ Done
- **SCALE**: 12.1 MB state for edge (column-normalized gradients, last-layer momentum)
- **AdaMeM**: 181.6 MB state for capable hardware (power iteration SVD, low-rank momentum)
- **DeviceProfile routing**: `optimizer_for_profile()` selects optimizer based on hardware
- 14 optimizer tests all passing

### Block-MoE-Res Model — ✅ Done
- **CpuBlockAttnResLayer**: All 10 Gemma 4 features (PLE, GQA 8Q/1KV, KV sharing, logit softcapping, sliding/full attention, dual head_dim, dual RoPE theta, pre-norm, SwiGLU, double-wide MLP)
- **CpuBlockAttnResModel**: 7×5 block structure, inter-block attention, PLE pre-computation
- **CpuMoELayer**: 4-expert MoE with top-2 routing, load balance loss, SwiGLU/GeLU toggle
- **Dense→MoE conversion**: every FFN becomes MoE; Expert 0 = exact copy, others = dense + noise

### LoRA Wiring — ✅ Done
- `attach_lora()` on CpuBlockAttnResModel (q_proj, v_proj)
- `forward_full_with_lora()`, two-phase backward for borrow checker
- LoRA A/B registered with optimizer via `register_matrix()`

### Distillation Pipeline — ✅ Done
- `gemma4_to_block_attnres()`: weight mapping with all Gemma 4 features
- `dense_ffn_to_moe(4, 2)`: convert every FFN to 4-expert MoE
- Multi-objective loss: KL(logits) + 0.5 × MSE(hidden_states)
- Actual weight updates via optimizer.step()

### Checkpoint Serialization — ✅ Done
- `BlockMoeResConfig` + `LayerConfig`: serializable architecture metadata
- BF16 safetensors output + JSON sidecar
- Tensor naming: `layers.{i}.{q,k,v,out}_proj`, `layers.{i}.moe.expert.{e}.{gate,up,down}`

---

## Phase 20: 1.58-bit Inference Stack (Planned)

The full inference stack for ternary-weighted models:

- **Ternary matmul** (e2080d12): add/subtract only — no multiplies. `output = scale × Σ(sign × activation)` where sign ∈ {-1, 0, +1}
- **2:4 sparse ternary** (97138344): guarantee 50% zeros, halve compute again. Effective ~1 bit/weight
- **TernaryLinear** (f2124f4e): quantized linear layer wrapping packed ternary weights
- **TernaryMoELayer** (c7e4d0b7): quantized MoE for fast inference — ternary experts with top-k routing
- **STE training integration** (eeb0f414): forward in ternary, backward through FP32, weight update, re-quantize
- **Post-training quantization pipeline** (0d0651c2): FP32 → ternary conversion with quality checkpoints
- **Post-training pruning** (8ce99a48): FP32/ternary → 2:4 sparse structure
- **WGSL GPU kernels** (48f9cfce, 903351d7): ternary matmul, sparse ternary matmul on GPU
- **Benchmarks** (4d93e889, bd1811b2): FP32 vs NF4 vs ternary vs sparse ternary quality/latency comparison

Target: ~700 MB model on disk, ~25 MB working set, ~20ms/token on RPi 5

---

## Phase 21: Edge I/O and KV Compression (Planned)

- **Expert mmap loading** (28975a4f): per-token expert loading from disk via mmap. USB SSD (~5ms), NVMe (~1.2ms)
- **3-bit TurboQuant KV cache** (4fbcad61): random orthogonal rotation → Lloyd-Max 3-bit codebook. 6× KV compression, training-free
- **Recurrent KV with block summaries** (7aa4ae75): recent 512 tokens full KV, older context replaced by block summary representations. ~3.5 MB regardless of context length. Validated by PyramidKV, StreamingLLM, Memformer research

---

## Phase 22: Expert I/O Pipeline (Planned)

- **PreScope predictive prefetch** (6975e040, ad091165): quasi-hidden state from post-attention residual predicts next-layer expert needs. Async mmap read overlaps I/O with compute
- **BuddyMoE expert fallback** (9c539330, 09bf41d0): offline calibration identifies co-activation patterns. Prefetch miss → substitute buddy expert already in memory. Eliminates synchronous I/O stalls

Source: arXiv:2509.23638 (PreScope), arXiv:2511.10054 (BuddyMoE)

---

## Phase 23: CPU/GPU Backend Abstraction (Planned)

- **YaRN-aware RoPE WGSL shaders** (d7d9f140): NTK-aware frequency scaling for extended context on GPU
- **GpuCapabilities** (7bccb096): expand capability detection for F64, SUBGROUPS, COOPERATIVE_MATRIX, storage buffer limits
- **Cooperative matrix MatMul** (447008c4): exploit Intel/AMD/NVIDIA tensor core features where available
- **Subgroup-aware FlashDecode** (89e83b10): subgroup-optimized decode kernel for capable GPUs
- **WGSL backward passes** (eb600679, c14a8c52, f21de9f5, 4f4ad031, ba70a9dd): RMSNorm backward, softmax backward, matmul backward, RoPE backward, LoRA backward
- **Parameter arena** (61208bb5): replace per-dispatch uniform buffers with single async arena

---

## Phase 24: Inference Pipeline Wiring (Planned)

- **Wire CpuBlockAttnResModel into KV cache, prefill, decode** (9eb37569): incremental decode (one token at a time), prefill (parallel prompt processing), KV cache management, block structure in decode, YaRN context extension

---

## Phase 25: Autonomous Learning Loop (Planned)

- **CoVo self-rewarding RL** (e751fbfa, b0aecc14): model evaluates reasoning trajectories via consistency + volatility. Rewards weight LoRA gradient updates. Implementation in `intrinsic_motivation.rs`
- **SkillKB** (6b733946, acb4188c): three-tier hierarchical experience (strategic plans → functional skills → atomic skills). Storage as LoRA diffs or PLE edits. Cosine similarity retrieval
- **FDAL active learning** (a02aa661, 9a71c993): small sampler network (~256 hidden, <1 MB) prioritizes high-uncertainty samples for edge LoRA training budget

Source: NeurIPS 2025 (CoVo), ICCV 2025 (FDAL)

---

## Phase 26: Elastic Inference (Planned)

- **Gemma 4 ISWA research** (45a51b6e): study how E2B/E4B share checkpoint but activate different parameter counts
- **Elastic expert activation** (1962a899): DeviceProfile controls active expert count. Integrated → top-1 from 2, LowEnd → top-2 from 4, MidRange/HighEnd → full with prefetch. Runtime switching between forward passes

---

## Phase 27: GPU BlockAttnResLayer (Planned)

- **GPU BlockAttnResLayer with all 10 Gemma 4 features** (66878bea): bring CPU features (PLE, GQA, KV sharing, logit softcapping, dual attention modes) to GPU compute shaders
- **Wire CPU-trained model to GPU** (eb112113): upload distilled Block-MoE-Res model weights to GPU for inference
