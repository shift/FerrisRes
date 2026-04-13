# FerrisRes

FerrisRes is a Rust-native AI inference and training engine built around **Block AttnRes** — a novel linear-time transformer architecture that replaces the quadratic attention bottleneck of standard transformers. It runs on any GPU or iGPU via [wgpu](https://github.com/gfx-rs/wgpu) (Vulkan, Metal, DX12, WebGPU), adapts automatically to the hardware it finds, and is written entirely in safe Rust with no Python dependency.

> ⚠️ **Research project — work in progress.** FerrisRes is an active research project exploring novel transformer architectures and heterogeneous GPU runtimes. It is **not production-ready**. APIs are unstable and breaking changes should be expected. It is shared publicly for transparency and to invite early feedback.

---

## Why FerrisRes?

| Problem | FerrisRes approach |
|---|---|
| Quadratic attention cost at long context | Block AttnRes: linear-time with intra- and inter-block attention |
| Python-only ML ecosystem | Pure Rust — no Python runtime, no C extension chain |
| Fixed hardware assumptions | `DeviceProfile` auto-tunes for integrated GPU through data-centre |
| Training only on high-end GPUs | Gradient checkpointing + CPU offload for 8 GB iGPUs and below |
| KV cache memory blowout | TurboQuant 2-bit compression: 16× memory reduction |
| Rigid inference pipeline | Composable: LoRA hot-swap, YaRN context extension, RAG, tool-calling |
| Single-modality only | Vision, audio, video with streaming I/O and cross-modal attention |
| Single-GPU bottleneck | Tensor parallelism, pipeline parallelism, RDMA/NVLink, cloud orchestration |

---

## Architecture: Block AttnRes

Standard transformers apply full self-attention over every token, giving O(n²) cost in sequence length. Block AttnRes reduces this to **O(n)** through a two-level attention hierarchy:

### Intra-block attention

The token sequence is divided into fixed-size **blocks** (default: 8 tokens per block). Within each block, standard multi-head self-attention runs with RoPE positional encoding. This produces a per-block *partial sum* — a compressed representation of that block's content.

### Inter-block attention

Once all block representations are collected, a second attention pass attends *across* blocks. The current query attends over the sequence of block representations, selecting which blocks are relevant. Because there are only k = n / block_size blocks, this second pass is O(k) = O(n / block_size) — linear in the original sequence length.

### Two-phase inference

1. **Prefill** — process the entire prompt in parallel, populate per-layer KV cache, produce logits for the first output token.
2. **Decode** — autoregressively generate one token per step. Each step appends new K/V to the per-layer cache and runs flash-decode attention via a dedicated WGSL kernel.

The `TokenGenerator` orchestrates both phases and exposes `generate_stream` for streaming delivery.

---

## Feature Overview

### Inference Pipeline

- **TokenGenerator** — full prefill+decode pipeline with `generate()`, `generate_stream()`, `generate_with_rag()`, `generate_with_tools()`
- **UnifiedTokenGenerator** — supports both BlockAttnRes and standard transformer (LLaMA/Mistral/Gemma) via `AnyModel` enum
- **Logit processors** — composable chain: repetition → frequency/presence penalty → temperature → top-k → top-p → sample
- **Prompt templates** — ChatML, Llama 2, Mistral, Alpaca, Raw (CLI `--template` flag)
- **Context extension** — YaRN (NTK-aware RoPE scaling) and StreamingLLM (attention sinks), effective position computed per decode step
- **RAG pipeline** — dense (cosine similarity), sparse (TF-IDF), hybrid retrieval with in-context learning
- **Tool search** — keyword/embedding/hybrid tool discovery, `[tool_call]` detection, result injection, continuation generation
- **DECS** — reasoning token optimizer with plateau detection and quality-preserving early stopping
- **Matryoshka elastic RAG** — adaptive embedding dimensions per device profile (32/64/128/256/768)
- **Token merging (ToMe)** — CPU bipartite soft matching for training-free visual token reduction
- **HullKVCache** — 2D convex hull attention with O(log n) lookups
- **LLM-Computer** — CALM virtual machine: LookUp → Compute → BranchIf instruction set
- **Speculative decoding** — n-gram draft model + rejection sampling verification
- **PagedAttention** — vLLM-style block management, copy-on-write, prefix sharing

### Multimodal

- **VisionEncoder** — ViT-style with Implicit GEMM (0 MB intermediate) or legacy im2col + ToMe
- **EnCodec audio encoder** — strided conv encoder + residual vector quantization (8 codebooks)
- **Cross-modal attention** — text/vision/audio fusion with early/mid/late fusion modes
- **VQ-VAE codebook** — nearest-neighbor lookup, EMA updates, multi-codebook (multi-head + residual)
- **Streaming image I/O** — progressive patch extraction, tiled reading for large images
- **Streaming audio I/O** — chunked window processing, ring buffer capture, streaming EnCodec
- **Streaming video I/O** — frame sampling, temporal buffering, progressive decode
- **Video token compression** — temporal redundancy removal, motion-compensated residuals, cross-frame merging (4-8× reduction)
- **3D/factored convolution** — temporal (T×1×1) + spatial (1×H×W) decomposition with WGSL kernels

### Training

- **Autodiff engine** — computation graph, reverse-mode backward pass, gradient accumulation
- **SGD / Adam optimizers** — GPU-side parameter updates
- **Cross-entropy loss** — GPU loss computation
- **LoRA adapters** — low-rank fine-tuning with merge/unmerge, auto-populate, hot-swap, merge_all()
- **QLoRA** — quantized-weight training: NF4 base + LoRA adapters, only adapters trainable
- **Gradient checkpointing** — PerBlock/PerLayer/PerAttention with recompute_block() (ADR-010)
- **CPU/Async gradient offload** — CPU-side accumulation and async GPU→CPU transfer for iGPUs
- **Tile-based gradient accumulation** — split batch into GPU-sized tiles, accumulate partials
- **Partial backpropagation** — layer freeze, selective backward, gradual unfreezing, LoRA integration

### Tokenizers

- **BPE tokenizer** — byte-pair encoding with DomainVocabulary for specialized tokens
- **QA-Token** — quality-aware tokenization with confidence-weighted vocabulary
- **BLT tokenizer** — Byte Latent Transformer: raw UTF-8 bytes, entropy-based dynamic patching, cross-patch attention

### Model Loading

- **Safetensors** — F32/F16/BF16, multi-shard, architecture detection
- **GGUF** — v2/v3, Q8_0/Q4_0/Q4_K/Q5_K/Q6_K dequantization, name mapping
- **Standard transformer** — O(n²) compatibility mode for LLaMA/Mistral/Gemma
- **Architecture dispatcher** — auto-detect model type from weights, `AnyModel` unified interface

### Distributed & Hardware

- **Tensor parallelism** — split weight matrices across N GPUs, all-reduce after attention/FFN
- **Pipeline parallelism** — assign layers to different GPUs, GPipe and 1F1B schedules
- **Weight sharding** — split_rows/cols with reconstruct, scatter/gather primitives
- **Cloud GPU orchestration** — worker registration, shard assignment, gradient aggregation, fault tolerance, cost-aware spot scheduling
- **Apple Neural Engine (ANE)** — automatic op placement (GPU for matmul/attention, ANE for BN/activation), unified memory buffers
- **RDMA/DirectGPU** — NVLink, RoCE, InfiniBand, TCP fallback with bandwidth/latency estimates

### Compute Kernels (WGSL)

| Kernel | Purpose |
|---|---|
| Tiled MatMul | 16×16 workgroup tiling + double-buffer variant |
| RMSNorm | Row-wise normalization |
| Softmax | Numerically stable online softmax |
| RoPE | Rotary position embeddings |
| FlashDecode + Tiled | Single-query decode attention, tiled with online softmax |
| CausalMask | Upper-triangular masking |
| Elementwise | Add, scale, ReLU, copy |
| im2col | Image patch extraction (legacy) |
| FusedPatchEmbed | Implicit GEMM — fused patch extraction + projection, 0 MB intermediate |
| MoE | Expert routing and gather |
| TurboQuant | Rotation, quantize, dequantize, QJL projection |
| ToMeMerge | Scatter-merge for token reduction |
| FFT | Fast Fourier Transform for audio spectrograms |
| Mel-spectrogram | Log-mel filterbank from FFT output |
| Temporal/Spatial Conv | 3D factored convolution (video processing) |
| Circular KV | Virtual circular buffer for KV cache |
| TurboQuant kernels | Rotation, quantize, dequantize, QJL projection |

### Hardware Adaptation

| Profile | VRAM | Default batch | KV cache |
|---|---|---|---|
| `Integrated` | shared / iGPU | 1 | 2 GB |
| `LowEnd` | < 4 GB | 2 | 4 GB |
| `MidRange` | 4–8 GB | 4 | 8 GB |
| `HighEnd` | > 8 GB | 8 | 16 GB |

Auto-detects at startup. Override via `FERRIS_DEVICE_PROFILE=integrated cargo run`.

---

## CLI

```
# Inference
cargo run -- infer --prompt "Explain transformers" --template chatml --max-tokens 128

# Multimodal
cargo run -- infer --prompt "Describe this image" --image photo.jpg

# Extended context
cargo run -- infer --prompt "Long document..." --yarn-scale 4.0

# Training with LoRA
cargo run -- train --epochs 3 --lora-rank 8 --data training.txt

# Benchmark
cargo run -- benchmark --iterations 100 --hidden-dim 512
```

---

## Getting Started

> The API is not yet stable. Method signatures may change before 1.0.

Add FerrisRes to your `Cargo.toml`:

```toml
[dependencies]
ferrisres = { git = "https://github.com/shift/FerrisRes" }
```

### Minimal inference example

```rust
use ferrisres::{
    BlockAttnResConfig, TokenEmbedding, LMHead,
    inference::generator::{TokenGenerator, GenerateConfig},
    model::BlockAttnResModel,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let vocab_size = 32_000;
    let config = BlockAttnResConfig::new(512);

    let model = Arc::new(BlockAttnResModel::new(
        Arc::clone(&device), Arc::clone(&queue),
        config.clone(), vocab_size,
    )?);
    let embedding = TokenEmbedding::new(
        Arc::clone(&device), Arc::clone(&queue),
        vocab_size, config.hidden_dim,
    )?;
    let lm_head = LMHead::new(
        Arc::clone(&device), Arc::clone(&queue),
        config.hidden_dim, vocab_size,
    )?;

    let generator = TokenGenerator::new(
        model, lm_head, embedding,
        Arc::clone(&device), Arc::clone(&queue),
        2048,
    )?;

    let prompt_tokens: Vec<u32> = vec![1, 42, 7];
    let output = generator.generate(
        &prompt_tokens,
        &GenerateConfig { max_tokens: 64, ..Default::default() },
    )?;
    println!("Generated token ids: {:?}", output);

    Ok(())
}
```

### Streaming generation

```rust
let rx = Arc::new(generator).generate_stream(
    prompt_tokens, /*max_new_tokens=*/ 128,
);
for token_id in rx {
    print!("{token_id} ");
}
```

### RAG-augmented generation

```rust
use ferrisres::inference::rag::RagStore;

let rag_store = RagStore::default_store();
// ... add documents ...
let output = generator.generate_with_rag(
    "What is attention?",
    &rag_store,
    &GenerateConfig::default(),
)?;
```

---

## API Server

FerrisRes includes an OpenAI-compatible HTTP API server:

```
# Start the API server
cargo run -- serve --port 8080
```

Endpoints:
- `POST /v1/chat/completions` — chat with messages
- `POST /v1/completions` — text completion
- `GET /v1/models` — list models
- `GET /health` — health check

Supports SSE streaming, CORS for browser integration, and works with any
OpenAI-compatible client (Open WebUI, curl, etc.).

---

## Building

FerrisRes requires a working Vulkan driver. On Linux the recommended path is through the provided Nix dev-shell:

```bash
nix develop          # enters the dev shell with Rust + Vulkan layers
cargo build
cargo test            # 637 tests
cargo bench
```

---

## Project Structure

```
src/
├── main.rs              # CLI (train/infer/benchmark/info)
├── lib.rs               # Public API re-exports
├── autodiff/             # Reverse-mode autodiff graph
├── compute/
│   ├── kernels/          # 17+ WGSL compute shaders
│   │   ├── matmul.rs     # Tiled + double-buffer matmul
│   │   ├── flash_decode  # Single-query decode attention
│   │   ├── rope.rs       # RoPE in-place
│   │   ├── fft.rs        # FFT for audio
│   │   ├── conv3d.rs     # 3D factored convolution (temporal + spatial)
│   │   └── ...           # RMSNorm, softmax, causal, elementwise, etc.
│   ├── buffer.rs         # GpuBuffer
│   ├── turboquant.rs     # TurboQuant engine
│   ├── distributed.rs    # Tensor/pipeline parallelism, weight sharding
│   ├── hardware.rs       # Cloud GPU, ANE/NPU, RDMA/DirectGPU
│   └── async_pipeline.rs # FA3 double-buffer dispatch
├── device/               # GPU detection + DeviceProfile + hardware tuning
├── inference/
│   ├── generator.rs      # TokenGenerator (generate/stream/rag/tools)
│   ├── unified_generator # UnifiedTokenGenerator (AnyModel)
│   ├── speculative.rs    # Speculative decoding (n-gram draft)
│   ├── paged_attention   # vLLM-style block pool, COW, prefix sharing
│   ├── cross_modal.rs    # Text/vision/audio cross-attention fusion
│   ├── video_compression # Temporal redundancy, motion compensation, merging
│   ├── logit_processors.rs
│   ├── prompt_templates.rs
│   ├── context_extension.rs
│   ├── rag.rs / matryoshka.rs / tool_search.rs
│   ├── token_merging.rs / paca.rs
│   ├── decs.rs / hull_kv_cache.rs / llm_computer.rs
│   ├── circular_kv.rs    # Virtual circular KV buffer
│   └── kv_cache.rs / sampling.rs
├── model/
│   ├── model.rs          # BlockAttnResModel (forward + backward)
│   ├── block_attn_res.rs # BlockAttnResLayer
│   ├── standard_transformer.rs  # O(n²) compatibility mode
│   ├── dispatcher.rs     # Architecture auto-detection (AnyModel)
│   ├── safetensors.rs    # Safetensors loader
│   ├── gguf.rs           # GGUF v2/v3 loader
│   ├── tokenizer.rs      # BPE + DomainVocabulary
│   ├── blt.rs            # Byte Latent Transformer tokenizer
│   ├── qa_tokenizer.rs   # QA-Token
│   ├── vision.rs         # VisionEncoder (Implicit GEMM + ToMe)
│   ├── audio.rs          # EnCodec audio encoder (RVQ)
│   ├── vqvae.rs          # VQ-VAE codebook (EMA, multi-codebook)
│   ├── streaming_image.rs # Progressive patch extraction
│   ├── streaming_audio.rs # Chunked audio processing + ring buffer
│   ├── streaming_video.rs # Frame sampling + temporal buffering
│   └── shard.rs          # ModelShard + QuantizedBuffer
├── tensor/               # GpuTensor
└── training/
    ├── optimizer.rs      # SGD, Adam, CrossEntropyLoss
    ├── checkpointing.rs  # CheckpointStore (recompute_block)
    ├── lora.rs           # LoRA adapter
    ├── gradient_accum.rs # Tile-based gradient accumulation
    ├── partial_backprop  # Layer freeze, selective backward
    └── cpu_offload.rs / async_offload.rs
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1–3 | ✅ Done | wgpu foundation, BlockAttnRes model, tiered compute, caching |
| 4 | ✅ Done | Autodiff, training, tokenizer, embedding, benches |
| 5 | ✅ Done | Streaming inference, RoPE, KV cache, flash-decode, logit processors |
| 6 | ✅ Done | TurboQuant, LoRA, RAG, YaRN, templates, DECS, HullKVCache, LLM-Computer |
| 7 | ✅ Done | Vision (Implicit GEMM, ToMe, PaCa), Matryoshka, audio, cross-modal, streaming I/O, VQ-VAE, BLT, video compression, 3D convolution |
| 8 | ✅ Done | Distributed tensor/pipeline parallelism, cloud GPU orchestration, RDMA/DirectGPU, ANE/NPU |
| 9 | ✅ Done | Weight loading (safetensors, GGUF), standard transformer compatibility, architecture dispatcher |

**All 212 tasks complete.**

See [ROADMAP.md](ROADMAP.md) for full technical details.

---

## Contributing

The project is not yet open for external contributions while core development is ongoing. Watch this repository for updates.

---

## License

FerrisRes is dual-licensed:

**AGPL-3.0-or-later** for free and open-source use. See [`LICENSE`](LICENSE) for the full terms.

**Commercial license** for use in proprietary or commercial products. Contact: shift+licensing@someone.section.me
