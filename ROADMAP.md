# FerrisRes Technical Roadmap

> **FerrisRes** is a Rust-native AI inference and training engine built around **Block AttnRes**, a linear-time transformer architecture. It runs on any GPU via wgpu (Vulkan, Metal, DX12, WebGPU), auto-adapts to hardware, and has zero Python dependency.

## Current Status

| Metric | Value |
|---|---|
| Source code | ~29,065 lines across 80+ modules |
| Test suites | 279 lib tests passing, 0 failures |
| Language | 100% Rust (safe + WGSL compute shaders) |
| GPU backends | Vulkan, Metal, DX12, WebGPU via wgpu |
| Tasks completed | 195 / 212 (17 low-priority remaining) |
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
│  │              │  │  Q5_K/Q6_K)  │  │  RVQ codecbooks) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ BPE Tokenizer│  │ QA-Token     │  │ VisionEncoder    │  │
│  │ + Domain     │  │ (quality-    │  │ (Implicit GEMM   │  │
│  │ Vocabulary   │  │  aware)      │  │  + ToMe merge)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ ModelShard   │  │ Image        │                         │
│  │ (F32/F16/    │  │ Preprocessor │                         │
│  │  I8/I4)      │  │              │                         │
│  └──────────────┘  └──────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│                    Training Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ LoRA         │  │ Gradient     │  │ Autodiff         │  │
│  │ (merge/      │  │ Checkpointing│  │ (ComputationGraph│  │
│  │  unmerge,    │  │ (closure-    │  │  + backward)     │  │
│  │  hot-swap)   │  │  based       │  │                  │  │
│  │              │  │  recompute)  │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Adam/SGD     │  │ CPU/Async    │                         │
│  │ Optimizers   │  │ Offload      │                         │
│  └──────────────┘  └──────────────┘                         │
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
│  │ MatMul +     │  │ RoPE +       │  │ RMSNorm +        │  │
│  │ FusedPatch   │  │ InPlace      │  │ Elementwise      │  │
│  │ Embed WGSL   │  │ RoPE WGSL    │  │ WGSL             │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Device Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ DeviceProfile│  │ Capability   │  │ Hardware Tuning  │  │
│  │ (GPU vendor  │  │ Detection    │  │ (workgroup size, │  │
│  │  + memory)   │  │              │  │  coalescing,     │  │
│  │              │  │              │  │  compute params) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Features (implemented)

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

### Training
- **Gradient checkpointing**: closure-based recompute (ADR-010)
- **LoRA**: merge/unmerge, auto-populate, hot-swap adapters
- **Autodiff**: ComputationGraph + backward pass
- **Adam/SGD optimizers**

### Multimodal
- **VisionEncoder**: implicit GEMM fused patch embedding + ToMe merge
- **EnCodec audio encoder**: strided conv encoder + residual vector quantization (8 codebooks)
- **Cross-modal attention**: Q from text, K/V from vision/audio, early/mid/late fusion
- **Modality type embeddings**: text/vision/audio learnable type IDs

### Tools & Pipeline
- **Logit processors**: temperature → top-k → top-p → repetition/frequency/presence penalty
- **Prompt templates**: ChatML, Llama2, Mistral, Alpaca, Raw
- **RAG pipeline**: dense/sparse/hybrid retrieval + Matryoshka elastic embeddings
- **Tool search**: keyword/embedding/hybrid registry with `[tool_call]` detection
- **DECS**: reasoning token optimization with plateau detection
- **HullKVCache**: 2D convex hull attention O(log n)
- **LLM-Computer**: CALM virtual machine (LookUp → Compute → BranchIf)

## Remaining Tasks (17, all Low priority)

| Task | Description |
|---|---|
| Remove deprecated AutoregressiveGenerator | Cleanup |
| Audit 43 dead_code annotations | Cleanup |
| Virtual circular KV buffer | Future optimization |
| Partial backpropagation | Future training |
| Cloud GPU server training | Infrastructure |
| Apple Neural Engine (ANE) | Hardware |
| Tile-based gradient accumulation | Training |
| RDMA/DirectGPU multi-node | Distributed |
| Streaming image/video/audio I/O | I/O pipelines |
| 3D/factored convolution | Multimodal |
| Video token compression | Multimodal |
| Byte Latent Transformer (BLT) | Architecture |
| VQ-VAE codebook | Multimodal |
| WGSL FFT for audio | Audio |
| Distributed tensor parallelism | Distributed |
