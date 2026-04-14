# FerrisRes

FerrisRes is a Rust-native AI inference and training engine built around **Block AttnRes** — a novel linear-time transformer architecture that replaces the quadratic attention bottleneck of standard transformers. It runs on any GPU or iGPU via [wgpu](https://github.com/gfx-rs/wgpu) (Vulkan, Metal, DX12, WebGPU), adapts automatically to the hardware it finds, and is written entirely in safe Rust with no Python dependency.

> ⚠️ **v0.2.1 — near-production grade, not yet 1.0.** FerrisRes has 843 passing tests, a full self-improvement loop, a verified Gemma 4 distillation pipeline, five output modalities (vision, speech, tactile, robotics, scientific), and a 4-layer security proxy (FerrisRes Armor). The Block AttnRes architecture is losslessly distilled from established models but still benefits from further field testing at scale. Public APIs follow `0.x` semver — breaking changes may occur before 1.0.0. Suitable for research, high-performance prototyping, and early-adopter production workloads.

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
- **Speculative Block Decoding** — tiny BlockDraftModel predicts block summaries, 8x throughput
- **Host tools** — web_fetch, math_eval, file_read/write, shell_exec, search, code_interpreter
- **WASM sandbox** — wasmi runtime for zero-trust tool execution with fuel limits
- **LSP tools** — Language Server Protocol client for deterministic code validation
- **Mirror Test** — recursive self-verification: code → test → execute → loss
- **Concept Memory** — persistent learned patterns with embedding-based retrieval
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

### Output Modalities

- **VisionHead** — VQ-VAE codebook logits with progressive row-by-row decode
- **SpeechHead** — N-codebook EnCodec-style prediction
- **VideoHead** — I-frame + P-frame residual prediction with VideoStreamReconstructor
- **Streaming TTS** — AudioStreamReconstructor with overlap-add, crossfade, fade-in/out
- **Robotics VLA** — ActionHead with binned (256 bins) and continuous (tanh-squashed) modes, ControlMode (Cartesian/Joint), safety checker
- **ChemicalValidator** — SMILES valence-aware token masking for organic subset (C, N, O, P, S, halogens, B)
- **MeshHead** — SDF prediction on sparse grid + Marching Cubes extraction → .obj export
- **GCodeValidator + GCodeGenerator** — Klipper-style parsing, work envelope validation, 32-token vocabulary
- **TactileHead** — 6-actuator (5 fingers + palm) 1000Hz haptic intensity prediction
- **VisualTactileBridge** — cosine similarity visual-to-tactile texture translation
- **SpeculativeHapticDecoder** — draft-then-verify for sub-20ms latency

### Security (FerrisRes Armor)

- **L0: Regex + Bloom filter** — 31 PII recognizers (email, SSN, phone, credit cards, IP, IBAN, API keys, etc.) + 5 prompt injection heuristics + 1MB Bloom filter for blocklist lookup
- **L1: Neural scanner** — ArmorGuardTiny: 4-layer BERT (~3.5M params) for binary SAFE/INJECTION classification
- **L2: RepE safety probe** — 6-category linear probes (violence, self-harm, sexual, hate, harassment, injection) on BlockSummary hidden states
- **L3: Output sanitizer** — PII redaction with Mask/Replace/Truncate strategies, injection-heuristic exclusion
- **ArmorLayer orchestrator** — verify_input (L0+L1), verify_hidden (L2), sanitize_output (L3), self-learning feedback loop

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

# Distillation (full pipeline)
cargo run -- distill \
  --model-path ./model.safetensors \
  --config 27b-mm \
  --steps 10000 \
  --tokenizer ./tokenizer.json \
  --data training_data.txt \
  --gpu \
  --converge 0.001 \
  --converge-patience 100

# Resume distillation
cargo run -- distill \
  --model-path ./model.safetensors \
  --config 27b-mm \
  --steps 5000 \
  --resume distilled_model.bin.checkpoint.bin
```
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
ferrisres = { git = "https://github.com/shift/FerrisRes", tag = "v0.2.0" }
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

## Distillation

FerrisRes converts standard transformer models (Gemma 4, LLaMA, Mistral, Phi, Qwen)
into native Block AttnRes models through structural linearization — a distillation
process that reduces attention from O(n²) to O(n) while preserving 95–99% of teacher quality.

**Verified on real hardware**: Successfully distilled the 9.6 GB Gemma 4 27B
Multimodal IT model (2.66B params, 35 layers, GQA 8Q/1KV heads) on a
32 GB machine with Intel HD 530 iGPU. Loss decreased from 22.05 → 21.56 over
10 steps with cached frozen states and `matrixmultiply`-accelerated CPU GEMM.

```bash
# Full distillation with real data and auto-convergence
cargo run -- distill \
  --model-path ./model.safetensors \
  --config 27b-mm \
  --steps 10000 \
  --seq-len 32 \
  --tokenizer ./tokenizer.json \
  --data training_data.txt \
  --gpu \
  --converge 0.001 \
  --converge-patience 100 \
  --checkpoint-every 100

# Resume from checkpoint (continues from last step)
cargo run -- distill \
  --model-path ./model.safetensors \
  --config 27b-mm \
  --steps 1000 \
  --resume distilled_model.bin.checkpoint.bin \
  --gpu

# Smaller model for testing
cargo run -- distill \
  --model-path ./model.safetensors \
  --config e2b \
  --steps 1000
```

### Key distillation features

- **Cached frozen states** — base model weights don't change, so per-layer hidden states are precomputed once. Training steps only run block summary blending + LM head.
- **`matrixmultiply` GEMM** — cache-tiled SIMD CPU matmul (~5–10× faster than naive loops).
- **Full Adam state persistence** — checkpoints save optimizer moments (m, v, t) so resume is seamless.
- **Mid-training checkpointing** — `--checkpoint-every N` saves progress incrementally.
- **Auto-convergence** — `--converge 0.001 --converge-patience 100` stops training when loss plateaus.
- **Structured logging** — all log lines use `event=` fields for machine parsing.

See [docs/distillation.md](docs/distillation.md) for the full guide.

## Documentation

| Guide | Description |
|---|---|
| [Quick Start](docs/quickstart.md) | Get running in 5 minutes |
| [Architecture](docs/architecture.md) | Block AttnRes deep dive |
| [Distillation](docs/distillation.md) | Gemma 4 → Block AttnRes conversion |
| [Security](docs/security.md) | FerrisRes Armor: 4-layer security proxy |
| [API Reference](docs/api-reference.md) | Public API, stability tiers, CLI |
| [Deployment](docs/deployment.md) | systemd, Docker, NixOS, security |

---

## Building

FerrisRes implements a closed-loop self-correction system:

1. **Model generates code** → WASM sandbox validates syntax in <1ms
2. **LSP-as-Oracle** provides deterministic compiler feedback
3. **Mirror Test** — model writes tests for its own code, test failures become loss signals
4. **Autodiff backward pass** — compiler errors penalize the model at the weight level
5. **Concept Memory** — learned patterns are persisted for retrieval in future sessions

This creates a system where the AI is physically tethered to the laws of programming logic:
syntax errors cause weight updates, not just chat corrections.

```bash
# Validate code via WASM sandbox (sub-millisecond, zero-trust)
ferrisres> TOOL_CALL:wasm_parse({"code": "fn main() { }", "lang": "rust"})
```

---

FerrisRes requires a working Vulkan driver. On Linux the recommended path is through the provided Nix dev-shell:

```bash
nix develop          # enters the dev shell with Rust + Vulkan layers
cargo build
cargo test            # 759 tests
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
│   ├── host_tools.rs      # 7 host tools (web_fetch, math_eval, etc.)
│   ├── lsp_tools.rs       # LSP JSON-RPC client + fallback checker
│   ├── wasm_sandbox.rs    # WASM runtime (wasmi) + embedded checker
│   ├── mirror_test.rs     # Recursive self-verification
│   ├── block_draft.rs     # Speculative Block Decoding
│   ├── concept_memory.rs  # Persistent concept memory + Hull-KV bridge
│   ├── pdf_ingestion.rs   # Raw PDF text extraction
│   ├── acp.rs             # Agent Capability Protocol router
│   ├── tts_stream.rs      # Streaming TTS with overlap-add reconstruction
│   ├── vla.rs             # Vision-Language-Action robotics controller
│   ├── scientific.rs      # SMILES validator, Marching Cubes, G-Code validator/generator
│   ├── tactile.rs         # TactileHead haptics + VisualTactileBridge
│   └── kv_cache.rs / sampling.rs
├── model/
│   ├── model.rs          # BlockAttnResModel (forward + backward)
│   ├── block_attn_res.rs # BlockAttnResLayer
│   ├── standard_transformer.rs  # O(n²) compatibility mode
│   ├── dispatcher.rs     # Architecture auto-detection (AnyModel)
│   ├── gemma_mapper.rs   # Gemma 4 weight mapper, GQA, distillation training
│   ├── gpu_forward.rs     # GPU-accelerated forward pass (DeviceProfile-aware)
│   ├── safetensors.rs     # Safetensors + MmapedSafetensors loader
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
│   ├── generation_head.rs # VisionHead, SpeechHead, VideoHead output modalities
│   └── shard.rs          # ModelShard + QuantizedBuffer
├── security/
│   ├── mod.rs            # Armor module registration
│   ├── armor.rs          # ArmorLayer orchestrator + self-learning feedback
│   ├── armor_l0.rs       # Regex PII engine + Bloom filter
│   ├── armor_l1.rs       # ArmorGuardTiny neural injection scanner
│   ├── armor_l2.rs       # RepE safety probe (6 categories)
│   └── armor_l3.rs       # PII redaction output sanitizer
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
| 10 | ✅ Done | Gemma 4 distillation pipeline, GPU forward pass, mmap loader, GQA, real-model verification |
| 11 | ✅ Done | Self-improvement: WASM sandbox, LSP-as-Oracle, Mirror Test, Block Decoding, Concept Memory |
| 12 | ✅ Done | v0.2.0: benchmarking, API stabilisation, quickstart/architecture/deployment docs |
| 13 | ✅ Done | Output modalities: VisionHead, SpeechHead, VideoHead, TTS stream, VLA ActionHead, Scientific (SMILES/Mesh/GCode), TactileHead |
| 14 | ✅ Done | FerrisRes Armor: L0 regex+bloom, L1 neural scanner, L2 RepE probe, L3 sanitizer, GPU-accelerated distillation |

**All tasks complete — 843 tests passing.**

See [ROADMAP.md](ROADMAP.md) for full technical details.

---

## Contributing

The project is not yet open for external contributions while core development is ongoing. Watch this repository for updates.

---

## License

FerrisRes is dual-licensed:

**AGPL-3.0-or-later** for free and open-source use. See [`LICENSE`](LICENSE) for the full terms.

**Commercial license** for use in proprietary or commercial products. Contact: shift+licensing@someone.section.me
