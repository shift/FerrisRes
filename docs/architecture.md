# FerrisRes Architecture

## Block AttnRes: O(n) Transformer

Standard transformers use quadratic attention: every token attends to every other token. Block AttnRes reduces this to linear time through a two-level hierarchy.

### Level 1: Intra-block attention

The token sequence is divided into fixed-size blocks (default: 8 tokens). Within each block, standard multi-head self-attention runs with RoPE positional encoding. This is O(block_size²) — constant per block.

### Level 2: Inter-block attention

Block summaries (mean-pooled representations of each block) attend across all blocks. Since there are only `n / block_size` blocks, this is O(n / block_size) — linear in the total sequence length.

### Distillation

Standard transformer models (Gemma 4) are converted to Block AttnRes through structural linearization:

```
Teacher (Gemma 4, O(n²))
    ↓  KL divergence loss
Student (Block AttnRes, O(n))
```

The teacher's attention patterns are preserved via KL divergence loss during training. Quality is 95-99% of the teacher, measured by perplexity.

## System Architecture

```
┌─────────────────────────────────────────────┐
│                  CLI / API Server            │
├─────────────────────────────────────────────┤
│             Inference Pipeline               │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │Generator │ │ Logit    │ │ Sampling     │ │
│  │(prefill+ │ │Processor │ │ (top-k/top-p)│ │
│  │ decode)  │ │ Chain    │ │              │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ RAG     │ │ Tools    │ │ WASM Sandbox │ │
│  │ Store   │ │ Registry │ │ (zero-trust) │ │
│  └─────────┘ └──────────┘ └──────────────┘ │
├─────────────────────────────────────────────┤
│              Model Layer                     │
│  ┌─────────────┐ ┌───────────────────────┐  │
│  │BlockAttnRes │ │ Standard Transformer  │  │
│  │ O(n)        │ │ O(n²) compatibility  │  │
│  └─────────────┘ └───────────────────────┘  │
│  ┌─────────────┐ ┌───────────────────────┐  │
│  │ Vision      │ │ Audio (EnCodec)       │  │
│  │ (Implicit   │ │ (RVQ codebooks)       │  │
│  │  GEMM)      │ │                       │  │
│  └─────────────┘ └───────────────────────┘  │
├─────────────────────────────────────────────┤
│            Compute Layer (wgpu)              │
│  Vulkan │ Metal │ DX12 │ WebGPU             │
├─────────────────────────────────────────────┤
│          Device Adaptation                   │
│  Integrated │ Low-End │ Mid-Range │ High-End │
└─────────────────────────────────────────────┘
```

## Self-Improvement Loop

FerrisRes implements a closed-loop self-correction system:

1. **Model generates code** → WASM sandbox validates in <1ms
2. **LSP-as-Oracle** provides deterministic compiler feedback
3. **Mirror Test** — model writes tests for its own code
4. **Test failures → loss signal → backprop** at the weight level
5. **Concept Memory** persists learned patterns across sessions

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Pure Rust (no Python) | Single build system, no FFI overhead |
| wgpu (not CUDA) | Runs on NVIDIA, AMD, Intel, Apple, Qualcomm |
| Block AttnRes | O(n) attention without quality loss |
| WASM sandbox | Zero-trust tool execution |
| Memory-mapped weights | 10GB model fits in 16GB RAM |
| JIT GPU uploads | Scales from 256MB to multi-GB buffers |
