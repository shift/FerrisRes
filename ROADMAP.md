# FerrisRes Technical Roadmap

> **FerrisRes** is a Rust-native AI inference and training engine built around **Block AttnRes**, a linear-time transformer architecture. It runs on any GPU via wgpu (Vulkan, Metal, DX12, WebGPU), auto-adapts to hardware, and has zero Python dependency.

## Current Status

| Metric | Value |
|---|---|
| Source code | ~15,800 lines across 51 modules |
| Test suites | 75 unit tests passing |
| Language | 100% Rust (safe + WGSL compute shaders) |
| GPU backends | Vulkan, Metal, DX12, WebGPU via wgpu |
| License | AGPL-3.0-or-later |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (clap)                           │
│              train / infer / benchmark / info               │
├─────────────────────────────────────────────────────────────┤
│                     Inference Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Two-Phase    │  │ Autoregressive│  │ Prompt Templates │  │
│  │ Inference    │  │ Generator     │  │ (ChatML/Llama2/  │  │
│  │              │  │ (stream)      │  │  Mistral/Alpaca) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘  │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────────────────┐  │
│  │ KV Cache     │  │ Logit        │  │ Context Extension │  │
│  │ (compressed) │  │ Processors   │  │ (YaRN/Streaming)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ RAG Pipeline │  │ Sampling     │  │ TurboQuant       │  │
│  │ (dense/sparse│  │ (top-k/top-p)│  │ (2-bit KV cache) │  │
│  │  hybrid)     │  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      Model Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ BlockAttnRes │  │ Token        │  │ MoE Linear       │  │
│  │ Model/Layer  │  │ Embedding    │  │ (top-k gating)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Linear       │  │ LM Head      │  │ Image            │  │
│  │ (GPU matmul) │  │ (logits)     │  │ Preprocessor     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ BPE Tokenizer│  │ Domain       │                         │
│  │ + Adaptive   │  │ Vocabulary   │                         │
│  │ Patching     │  │ Extension    │                         │
│  └──────────────┘  └──────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│                    Training Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Autodiff     │  │ SGD/Adam     │  │ Gradient         │  │
│  │ Engine       │  │ Optimizers   │  │ Checkpointing    │  │
│  │ (graph,      │  │              │  │ (PerBlock/       │  │
│  │  backward)   │  │              │  │  PerLayer)       │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ CPU Gradient │  │ Async        │  │ LoRA Adapter     │  │
│  │ Offload      │  │ Gradient     │  │ (low-rank        │  │
│  │              │  │ Offload      │  │  fine-tuning)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                   Compute / Device Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ GpuBuffer    │  │ WGSL Kernel  │  │ Device Profile   │  │
│  │ (read/write/ │  │ Registry     │  │ (Integrated/     │  │
│  │  map)        │  │              │  │  Low/Mid/High)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Block Cache  │  │ Memory Pool  │  │ GPU Vendor       │  │
│  │ (tiled)      │  │ & Borrowed   │  │ Detection        │  │
│  │              │  │ Buffers      │  │ (NV/AMD/Intel)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                        WGSL Kernels                         │
│  MatMul │ RMSNorm │ Softmax │ RoPE │ FlashDecode │ CausalMask │
│  Elementwise │ im2col │ MoE dispatch/gather │ TurboQuant    │
├─────────────────────────────────────────────────────────────┤
│                         wgpu                                │
│            Vulkan │ Metal │ DX12 │ WebGPU                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Implemented Features

### Phase 1: Foundation (Complete)

| Component | Module | Description | Tests |
|---|---|---|---|
| **Project setup** | `main.rs`, `lib.rs` | Cargo workspace, CLI with `train`/`infer`/`benchmark`/`info` subcommands | — |
| **Device detection** | `device/capability.rs` | GPU vendor detection (NVIDIA/AMD/Intel/Apple/Qualcomm), VRAM query, `GpuKind` enum | — |
| **Device profiles** | `device/profile.rs` | 4-tier auto-tuning: `Integrated`/`LowEnd`/`MidRange`/`HighEnd`, workgroup size tuning | — |
| **GpuBuffer** | `compute/buffer.rs` | GPU buffer abstraction: read/write/map, zero-initialized buffers | — |
| **GpuTensor** | `tensor/gpu_tensor.rs` | Tensor wrapper over GpuBuffer with shape tracking | — |
| **WGSL MatMul** | `compute/kernels/matmul.rs` | Tiled matrix multiply compute shader | ✅ |
| **WGSL RMSNorm** | `compute/kernels/rmsnorm.rs` | Row-wise RMS normalization | ✅ |
| **WGSL Softmax** | `compute/kernels/softmax.rs` | Numerically stable online softmax | ✅ |
| **WGSL RoPE** | `compute/kernels/rope.rs` | Rotary position embeddings with position offset | ✅ |
| **WGSL Elementwise** | `compute/kernels/elementwise.rs` | Add, scale, ReLU, copy, strided bias add | ✅ |

### Phase 2: Model & Training (Complete)

| Component | Module | Description | Tests |
|---|---|---|---|
| **BlockAttnResConfig** | `model/config.rs` | Architecture hyperparameters: hidden_dim, block_size, num_layers, MoE config, adaptive patching | — |
| **BlockAttnResLayer** | `model/block_attn_res.rs` | Full layer: pre-norm → Q/K/V → RoPE → attention → residual → FFN/MoE | — |
| **BlockAttnResModel** | `model/block_attn_res.rs` | Multi-layer model with intra-block and inter-block attention, prefill and decode paths | — |
| **TokenEmbedding** | `model/embedding.rs` | Learned token embeddings with GPU lookup | — |
| **LMHead** | `model/lm_head.rs` | Linear projection to vocabulary logits | — |
| **Linear** | `model/linear.rs` | GPU-resident linear layer with matmul + bias | — |
| **MoELinear** | `model/moe_linear.rs` | Mixture-of-Experts with top-k gating | — |
| **ModelShard** | `model/shard.rs` | Tensor-parallel shard with QuantizedBuffer (F32/F16/Int8/Int4) | — |
| **Autodiff engine** | `autodiff/` | Computation graph, backward pass, gradient accumulation, matmul/elementwise grad kernels | — |
| **Optimizers** | `training/optimizer.rs` | SGD and Adam with GPU-side parameter updates, cross-entropy loss | ✅ |
| **Gradient checkpointing** | `training/checkpointing.rs` | Activation checkpointing at PerBlock/PerLayer/PerAttention granularity | — |
| **CPU gradient offload** | `training/cpu_offload.rs` | CPU-side gradient accumulation for integrated GPUs | ✅ |
| **Async gradient offload** | `training/async_offload.rs` | Multi-staged async GPU→CPU gradient transfer | ✅ |
| **LoRA** | `training/lora.rs` | Low-rank adaptation: merge/unmerge, per-module targeting, auto-populate | 9 tests |

### Phase 3: Inference Engine (Complete)

| Component | Module | Description | Tests |
|---|---|---|---|
| **Two-phase inference** | `inference/two_phase.rs` | Prefill + decode pipeline with block-level caching | — |
| **Autoregressive generator** | `inference/generator.rs` | `generate_stream` channel, KV cache management | — |
| **KV cache** | `inference/kv_cache.rs` | Per-layer GPU-resident key/value buffers with atomic position tracking | — |
| **Flash decode** | `compute/kernels/flash_decode.rs` | Single-query decode attention over full KV cache | ✅ |
| **Causal mask** | `compute/kernels/causal_mask.rs` | Upper-triangular masking for prefill attention | ✅ |
| **Prefill attention** | `compute/kernels/prefill_attn.rs` | Batched multi-head attention for prompt processing | ✅ |
| **Sampling** | `inference/sampling.rs` | Argmax, temperature, top-k, top-p sampling on CPU | — |

### Phase 4: Advanced Features (Complete)

| Component | Module | Description | Tests |
|---|---|---|---|
| **Logit processors** | `inference/logit_processors.rs` | Composable pipeline: repetition penalty → frequency/presence penalty → temperature → top-k → top-p → sample | 12 tests |
| **Prompt templates** | `inference/prompt_templates.rs` | ChatML, Llama 2, Mistral, Alpaca, Raw formats with system prompt override | 10 tests |
| **Context extension** | `inference/context_extension.rs` | YaRN (NTK-aware RoPE scaling), StreamingLLM (attention sinks), position interpolation | 11 tests |
| **RAG pipeline** | `inference/rag.rs` | Dense (cosine similarity), sparse (TF-IDF), hybrid retrieval; document chunking, in-context learning | 10 tests |
| **TurboQuant** | `compute/turboquant.rs` | Two-stage vector quantization for KV cache: outlier channel splitting, 2-bit/3-bit/4-bit compression | 6 tests |
| **BPE tokenizer** | `model/tokenizer.rs` | Byte-pair encoding with configurable vocab size, learned merges from corpus | — |
| **Domain vocabulary** | `model/tokenizer.rs` | `DomainVocabulary` for specialized tokens (SMILES, genomics), longest-match encoding | — |
| **Adaptive patching** | `model/config.rs` | Entropy-based patch boundary prediction: high-entropy → smaller patches, low-entropy → larger | — |
| **MoE kernels** | `compute/kernels/moe.rs` | Expert routing (top-k gating), dispatch, gather in WGSL | ✅ |
| **im2col kernel** | `compute/kernels/im2col.rs` | Image patch extraction for vision inputs | ✅ |
| **Image preprocessor** | `model/image_preprocessor.rs` | Resize + normalize for vision inputs | — |

### Hardware Optimizations (Complete)

| Feature | Module | Description |
|---|---|---|
| **Device profiles** | `device/profile.rs` | Auto-detection: Integrated (shared RAM), LowEnd (<4GB), MidRange (4-8GB), HighEnd (>8GB) |
| **GPU vendor detection** | `device/capability.rs` | `GpuVendor` enum (NVIDIA, AMD, Intel, Apple, Qualcomm) with vendor-specific tuning parameters |
| **Dynamic workgroups** | `device/profile.rs` | `recommended_workgroup_size()` and `recommended_tile_size()` per profile |
| **Memory coalescing** | `compute/memory.rs` | `MemoryCoalescingConfig` tuned per GPU type (discrete: 256-byte align + double buffering; integrated: 64-byte) |
| **Borrowed buffer pool** | `compute/memory.rs` | KV cache buffer reuse for gradients on integrated GPUs (shared DRAM) |
| **Quantized buffers** | `model/shard.rs` | F32, F16, Int8, Int4 storage with dequantize-on-read |
| **WGSL shader variants** | `compute/pipeline.rs` | `ComputeParams` for runtime kernel tuning based on device profile |

---

## Planned / In Progress

The following features have been researched and designed but not yet implemented in code.

### Near-Term: Advanced Inference

#### Tool Search Registry & Dynamic Loading
**Task:** `2c6aacbf`

Dynamically discover and load relevant tools (3-8) instead of scanning an entire library (50-100+). Cuts prompt costs by ~50% in agentic workflows. Requires:
- `ToolRegistry` with semantic tool descriptions
- Embedding-based tool matching
- Runtime tool injection into prompt context

#### DECS Reasoning Token Optimizer
**Task:** `72fb66b3`

50%+ reasoning token reduction (ICLR 2026). Identifies and penalizes redundant reasoning tokens using token-level reward signals. Requires:
- Token-level reward model integration
- Redundancy detector in the sampling loop
- Quality-preserving early stopping

#### QA-Token Quality-Aware Tokenization
**Task:** `882b4c58`

15-20% token count reduction vs BPE. Incorporates data reliability (e.g., sequencing confidence) into vocabulary construction. Zero inference overhead. Requires:
- Quality signal integration in BPE merge training
- Confidence-weighted vocabulary construction
- No changes to inference path (vocabulary is static)

### Long-Term: Architecture Research

#### 2D Attention with HullKVCache
**Task:** `9059364b`

Restricts lookup heads to 2D head dimension, turning linear scans into O(log n) queries. Enables millions of exact execution steps with perfect deterministic accuracy. Requires:
- 2D head dimension restructuring in attention
- Convex hull construction on attention heads
- Binary search over hull boundaries

#### LLM-Computer Architecture (WASM in Transformer Weights)
**Task:** `a1965c61`

Treats the transformer as a programmable target. CALM DSL compiles to attention (LookUp gates) and FFN (ReGLU gates). Full WASM interpreter embedded in weights. 30k+ tokens/sec on CPU. Requires:
- CALM DSL parser and compiler
- LookUp/ReGLU gate implementations in WGSL
- WASM interpreter weight encoding scheme
- Integration with HullKVCache for O(log n) lookups

---

## Module Index

```
src/
├── main.rs                          # CLI entry point (train/infer/benchmark/info)
├── lib.rs                           # Public API re-exports
├── error.rs                         # FerrisResError enum
│
├── device/                          # Hardware detection & adaptation
│   ├── mod.rs
│   ├── capability.rs                # GPU vendor detection, VRAM, adapter info
│   └── profile.rs                   # 4-tier DeviceProfile + workgroup tuning
│
├── tensor/                          # Tensor abstractions
│   ├── mod.rs
│   └── gpu_tensor.rs                # GpuTensor shape wrapper
│
├── compute/                         # GPU compute infrastructure
│   ├── mod.rs                       # Re-exports
│   ├── buffer.rs                    # GpuBuffer (wgpu buffer wrapper)
│   ├── cache.rs                     # BlockCache (tiled compute cache)
│   ├── memory.rs                    # MemoryPool, BorrowedBufferPool, MemoryCoalescingConfig
│   ├── pipeline.rs                  # ComputeParams, dispatch helpers
│   ├── turboquant.rs                # TurboQuant engine (outlier channel splitting, quantization)
│   └── kernels/                     # WGSL compute shaders
│       ├── mod.rs
│       ├── matmul.rs                # Tiled matrix multiply
│       ├── rmsnorm.rs               # RMS normalization
│       ├── softmax.rs               # Online softmax
│       ├── rope.rs                  # Rotary position embeddings
│       ├── flash_decode.rs          # Single-query decode attention
│       ├── causal_mask.rs           # Causal masking
│       ├── prefill_attn.rs          # Batched multi-head attention
│       ├── elementwise.rs           # Add, scale, ReLU, copy
│       ├── im2col.rs                # Image patch extraction
│       ├── moe.rs                   # MoE expert routing + gather
│       └── turboquant_kernels.rs    # TurboQuant rotation/quantize/dequantize
│
├── model/                           # Neural network components
│   ├── mod.rs
│   ├── config.rs                    # BlockAttnResConfig, AdaptivePatchingConfig, EntropyPredictor
│   ├── block_attn_res.rs            # BlockAttnResModel + BlockAttnResLayer (core architecture)
│   ├── linear.rs                    # GPU linear layer
│   ├── moe_linear.rs                # Mixture-of-Experts FFN
│   ├── embedding.rs                 # Token embedding lookup
│   ├── lm_head.rs                   # Output projection to logits
│   ├── shard.rs                     # ModelShard + QuantizedBuffer (F32/F16/Int8/Int4)
│   ├── tokenizer.rs                 # SimpleTokenizer, BpeTokenizer, DomainVocabulary
│   └── image_preprocessor.rs        # Image resize/normalize
│
├── inference/                       # Inference pipeline
│   ├── mod.rs
│   ├── two_phase.rs                 # TwoPhaseInference (prefill + decode)
│   ├── generator.rs                 # AutoregressiveGenerator (streaming)
│   ├── kv_cache.rs                  # KV cache (standard + TurboQuant compressed)
│   ├── sampling.rs                  # Basic sampling (argmax, temperature, top-k, top-p)
│   ├── logit_processors.rs          # Full logit pipeline (repetition/temp/top-k/top-p/frequency/presence)
│   ├── prompt_templates.rs          # ChatML, Llama2, Mistral, Alpaca, Raw
│   ├── context_extension.rs         # YaRN, StreamingLLM, position interpolation
│   └── rag.rs                       # RAG pipeline (dense/sparse/hybrid retrieval, in-context learning)
│
├── training/                        # Training infrastructure
│   ├── mod.rs                       # TrainingState, TrainingConfig, CheckpointGranularity
│   ├── optimizer.rs                 # SGD, Adam, CrossEntropyLoss
│   ├── checkpointing.rs             # CheckpointStore (PerBlock/PerLayer/PerAttention)
│   ├── cpu_offload.rs               # CPU gradient accumulation buffer
│   ├── async_offload.rs             # Async GPU→CPU gradient transfer
│   └── lora.rs                      # LoRA adapter (low-rank fine-tuning)
│
└── autodiff/                        # Automatic differentiation
    ├── mod.rs
    ├── graph.rs                     # Computation graph (node tracking, shape propagation)
    ├── backward.rs                  # Reverse-mode autodiff (matmul/elementwise grads in WGSL)
    └── accumulator.rs               # Gradient accumulator
```

---

## Test Coverage Summary

```
Module                                    Tests
─────────────────────────────────────────────────
compute::turboquant                        6
training::lora                             9
inference::logit_processors               12
inference::prompt_templates               10
inference::context_extension              11
inference::rag                            10
training::cpu_offload                      5
training::async_offload                    5
training::optimizer                        2
autodiff                                   5
─────────────────────────────────────────────────
Total                                     75
```

---

## Development Phases

```
Phase 1: Vulkan Tensor & Autodiff Foundation     ████████████ DONE
Phase 2: Training Engine & Cross-Stage Caching   ████████████ DONE
Phase 3: Inference Engine & Two-Phase Compute    ████████████ DONE
Phase 4: End-to-End Trainable System             ████████████ DONE
Phase 5: Advanced Inference Features             ████████████ DONE
Phase 6: Architecture Extensions                 ██████░░░░░░ IN PROGRESS
Phase 7: Multimodal Tokenization                 ██░░░░░░░░░░ RESEARCHED
Phase 8: Distributed Training                    ██░░░░░░░░░░ RESEARCHED
Phase 9: Model Format Loading                    ██░░░░░░░░░░ RESEARCHED
```

### Phase 6: Architecture Extensions (In Progress)

Remaining implementation tasks:

| Task | Description | Status |
|---|---|---|
| Tool Search Registry | Dynamic tool discovery for agentic workflows | 📝 Todo |
| DECS Token Optimizer | Reasoning token reduction via redundancy detection | 📝 Todo |
| QA-Token | Quality-aware tokenization for noisy domains | 📝 Todo |
| 2D Attention + HullKVCache | O(log n) exact lookups | 📝 Todo |
| LLM-Computer | WASM interpreter in transformer weights | 📝 Todo |

### Phase 7: Multimodal Tokenization

Research complete for:
- Vision encoder design (ViT-style patch tokenization via im2col)
- Audio codec integration (EnCodec-style)
- Unified multimodal embedding space
- Cross-modal attention

Implementation requires:
- `MultimodalTokenizer` combining text/vision/audio tokenizers
- `VisionEncoder` with patch embedding
- `AudioEncoder` with spectral features
- Cross-modal attention layers

### Phase 8: Distributed Training

Research complete for:
- Tensor parallelism (weight sharding across GPUs)
- Pipeline parallelism with cross-stage caching
- NCCL vs custom collective ops
- RDMA/DirectGPU for multi-GPU

Implementation requires:
- `DistributedCommunicator` abstraction
- Gradient synchronization (all-reduce)
- Pipeline stage management
- Multi-node coordinator

### Phase 9: Model Format Loading

Research complete for:
- **Safetensors**: HuggingFace format (secure, no pickle, efficient tensor storage)
- **GGUF**: llama.cpp format (quantized weights Q4_K_M, Q5_K_S, metadata, vocabulary)

Implementation requires:
- `SafetensorsLoader` (header parsing, tensor extraction, dtype mapping)
- `GgufLoader` (metadata parsing, quantization type handling, weight dequantization)
- Integration with `ModelShard` for shard loading

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **wgpu over CUDA** | Universal GPU support: NVIDIA, AMD, Intel, Apple Silicon, Qualcomm |
| **WGSL shaders** | Runtime compilation via naga; no SPIR-V build step |
| **Block AttnRes** | Linear O(n) attention via block partitioning vs O(n²) standard attention |
| **CPU gradient offload** | Enables training on 8GB integrated GPUs |
| **TurboQuant on KV cache** | 16x compression (2-bit) on the largest inference memory consumer |
| **LoRA merge/unmerge** | Zero-cost inference after merging; hot-swap adapters without retraining |
| **Composable logit pipeline** | Chain any combination of temperature/top-k/top-p/penalties in any order |
| **YaRN over naive PI** | Better perplexity at extended context (NTK-aware frequency scaling) |

---

## Performance Characteristics

| Component | Throughput / Metric |
|---|---|
| MatMul kernel | Tiled with workgroup-optimized dispatch |
| Flash decode | Single-pass attention over full KV cache |
| TurboQuant compression | 2-bit: 16x, 2.5-bit: 12.8x, 3-bit: 10.7x, 4-bit: 8x memory reduction |
| LoRA overhead | <0.5% parameter increase at rank=8 on q_proj/v_proj |
| Context extension | 4x context (4k→16k) with YaRN, streaming with attention sinks |
| Gradient checkpointing | PerBlock: ~2x memory savings at ~33% recomputation cost |

---

## Contributing

FerrisRes is in active development. The architecture is stabilizing but not yet 1.0. Key areas for contribution:

1. **WGSL kernel optimization** — better tiling, cooperative matrix multiply
2. **Model format loaders** — Safetensors and GGUF
3. **Distributed training** — tensor/pipeline parallelism
4. **Multimodal integration** — vision and audio encoders
5. **Benchmarks** — systematic profiling across DeviceProfile tiers
