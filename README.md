# FerrisRes

FerrisRes is a Rust-native AI inference and training engine built around **Block AttnRes** — a novel linear-time transformer architecture that replaces the quadratic attention bottleneck of standard transformers. It runs on any GPU or iGPU via [wgpu](https://github.com/gfx-rs/wgpu) (Vulkan, Metal, DX12, WebGPU), adapts automatically to the hardware it finds, and is written entirely in safe Rust with no Python dependency.

> **Status:** Active development. The architecture and core APIs are stabilising but not yet 1.0. Expect breaking changes between minor versions until the 1.0 release.

---

## Why FerrisRes?

| Problem | FerrisRes approach |
|---|---|
| Quadratic attention cost at long context | Block AttnRes: linear-time with intra- and inter-block attention |
| Python-only ML ecosystem | Pure Rust — no Python runtime, no C extension chain |
| Fixed hardware assumptions | `DeviceProfile` auto-tunes for integrated GPU through data-centre |
| Training only on high-end GPUs | Gradient checkpointing + CPU offload for 8 GB iGPUs and below |
| Monolithic model format | `ModelShard` + `ShardManager` for tensor-parallel split across devices |

---

## Architecture: Block AttnRes

Standard transformers apply full self-attention over every token, giving O(n²) cost in sequence length. Block AttnRes reduces this to **O(n)** through a two-level attention hierarchy:

### Intra-block attention

The token sequence is divided into fixed-size **blocks** (default: 8 tokens per block). Within each block, standard multi-head self-attention runs with RoPE positional encoding. This produces a per-block *partial sum* — a compressed representation of that block's content.

```
tokens: [t₀ t₁ … t₇] [t₈ t₉ … t₁₅] … [tₙ₋₈ … tₙ₋₁]
                 ↓ intra-block attn
block reps:   [b₀]          [b₁]      …       [bₖ]
```

Each block's partial sum is produced by accumulating the residual-connected attention outputs across every layer within the block (`forward_intra_block`). This can be parallelised across blocks.

### Inter-block attention

Once all block representations are collected, a second attention pass attends *across* blocks. The current query (or the running hidden state in decode) attends over the sequence of block representations, selecting which blocks are relevant and blending their summaries. This is the `forward_inter_block` pass.

Because there are only k = n / block_size blocks, this second pass is O(k) = O(n / block_size) — linear in the original sequence length.

### Two-phase inference

Generation follows the standard prefill → decode split:

1. **Prefill** — process the entire prompt in parallel, populate the per-layer KV cache (keys and values projected through RoPE), and produce logits for the first output token.
2. **Decode** — autoregressively generate one token per step using `forward_decode_token`. Each step appends the new K/V to the per-layer cache and runs flash-decode attention (single query against full cache) via a dedicated WGSL kernel.

The `TokenGenerator` orchestrates both phases and exposes a `generate_stream` channel for streaming token delivery.

---

## Feature Overview

### Compute kernels (WGSL)

All GPU computation is expressed in hand-written WGSL shaders compiled at runtime through naga:

| Kernel | File | Purpose |
|---|---|---|
| Tiled matrix multiply | `compute/kernels/matmul.rs` | General matmul with workgroup tiling |
| RMS normalisation | `compute/kernels/rmsnorm.rs` | Pre-norm before attention and FFN |
| Softmax | `compute/kernels/softmax.rs` | Row-wise softmax for attention weights |
| RoPE | `compute/kernels/rope.rs` | Rotary position embeddings on Q and K |
| Flash decode | `compute/kernels/flash_decode.rs` | Single-query decode attention over KV cache |
| Causal mask | `compute/kernels/causal_mask.rs` | Upper-triangular -inf masking for prefill |
| Elementwise | `compute/kernels/elementwise.rs` | Add, scale, ReLU, copy |
| im2col | `compute/kernels/im2col.rs` | Image-patch extraction for vision inputs |
| MoE dispatch/gather | `compute/kernels/moe.rs` | Expert routing and result assembly |

### Model components

- `BlockAttnResLayer` — single layer: pre-norm → Q/K/V projection → RoPE → attention → output projection → residual → FFN (dense or MoE)
- `BlockAttnResModel` — stack of layers organised into blocks; exposes `forward`, `forward_prefill`, `forward_decode_token`
- `MoELinear` — Mixture-of-Experts feed-forward with top-k gating, GPU dispatch and gather
- `TokenEmbedding` — learned token embeddings with GPU lookup
- `LMHead` — linear projection to vocabulary logits
- `ImagePreprocessor` — resize + normalise input images; im2col patch tokenisation for vision

### Inference

- `TwoPhaseInference` — block-level forward with optional block-representation caching
- `TokenGenerator` — full prefill+decode pipeline with per-layer `ModelKVCache`
- `KVCache` / `LayerKVCache` / `ModelKVCache` — GPU-resident key/value buffers with atomic position tracking
- `Sampler` — argmax, temperature, top-k, top-p sampling on CPU from read-back logits
- `generate_stream` — `mpsc::Receiver<u32>` channel for streaming tokens to the caller

### Training

- `SgdOptimizer`, `AdamOptimizer` — GPU-side optimisers
- `CrossEntropyLoss` — GPU loss computation
- `CheckpointStore` — activation checkpointing at configurable granularity (`PerBlock`, `PerLayer`, `PerAttention`)
- `AsyncGradientOffload` — multi-stage staging pool for async GPU→CPU gradient transfer
- `CpuGradientBuffer` — CPU-side gradient accumulation for `CpuOffload` mode
- `TrainingConfig::apply_device_profile` — auto-promotes checkpoint granularity and gradient accumulation steps for integrated-GPU devices

### Hardware adaptation (`DeviceProfile`)

FerrisRes auto-detects VRAM and GPU kind at startup and selects one of four profiles:

| Profile | VRAM | Compute mode | Default batch | KV cache |
|---|---|---|---|---|
| `Integrated` | shared / iGPU | `CpuOffload` | 1 | 2 GB |
| `LowEnd` | < 4 GB | `Tiled` | 2 | 4 GB |
| `MidRange` | 4–8 GB | `FullGpu` | 4 | 8 GB |
| `HighEnd` | > 8 GB | `FullGpu` | 8 | 16 GB |

Override via environment variable for testing:
```
FERRIS_DEVICE_PROFILE=integrated cargo run
```

Compile-time feature flags also force a profile:
```
cargo build --features integrated_gpu_profile
cargo build --features high_end_profile
```

### Model sharding

`ModelShard` and `ShardManager` split the layer stack across multiple devices for tensor-parallel inference and training. Each shard holds a contiguous range of layers and exposes the same `forward` interface.

---

## Getting Started

> The API is not yet stable. The snippet below reflects the current state; method signatures may change before 1.0.

Add FerrisRes to your `Cargo.toml`:

```toml
[dependencies]
ferrisres = { git = "https://github.com/shift/FerrisRes" }
```

### Minimal inference example

```rust
use ferrisres::{
    BlockAttnResConfig, DeviceProfile, Capability,
    inference::{TokenGenerator, GenerateConfig},
    model::{BlockAttnResModel, TokenEmbedding, LMHead},
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialise wgpu device
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("no adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let vocab_size = 32_000;
    let config = BlockAttnResConfig::new(512); // hidden_dim=512

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
        /*max_seq_len=*/ 2048,
    )?;

    let prompt_tokens: Vec<u32> = vec![1, 42, 7]; // pre-tokenised
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
let rx = generator.generate_stream(
    prompt_tokens,
    GenerateConfig { max_tokens: 128, temperature: 0.8, ..Default::default() },
);
for token_id in rx {
    print!("{token_id} ");
}
```

### Training

```rust
use ferrisres::training::{TrainingConfig, TrainingState};

let mut train_cfg = TrainingConfig::new(/*epochs=*/3, /*batch=*/4, /*lr=*/1e-3);
train_cfg.apply_device_profile(DeviceProfile::Integrated); // auto-tunes for iGPU
let mut state = TrainingState::with_profile(&DeviceProfile::Integrated, Some(Arc::clone(&device)));
```

---

## Building

FerrisRes requires a working Vulkan driver. On Linux the recommended path is through the provided Nix dev-shell:

```bash
nix develop          # enters the dev shell with Rust + Vulkan layers
cargo build
cargo test
cargo bench
```

Feature flags:
```
--features vulkan      # default
--features metal       # macOS
--features dx12        # Windows
--features webgpu      # browser / WASM target
```

---

## Project Structure

```
src/
├── autodiff/        # Reverse-mode autodiff graph and gradient accumulator
├── compute/
│   ├── kernels/     # WGSL compute shaders (matmul, RoPE, softmax, MoE, …)
│   ├── buffer.rs    # GpuBuffer — typed wgpu buffer wrapper
│   ├── cache.rs     # BlockCache for block representations
│   ├── memory.rs    # MemoryBudget, MemoryPool, BorrowedBufferPool
│   └── pipeline.rs  # ComputeDispatcher, TiledCompute
├── device/
│   ├── capability.rs # GPU kind detection (integrated / discrete)
│   └── profile.rs    # DeviceProfile + ComputeMode
├── inference/
│   ├── generator.rs  # TokenGenerator — prefill + decode pipeline
│   ├── kv_cache.rs   # LayerKVCache / ModelKVCache
│   ├── sampling.rs   # argmax, temperature, top-k, top-p
│   └── two_phase.rs  # TwoPhaseInference + legacy KVCache / Sampler
├── model/
│   ├── block_attn_res.rs  # BlockAttnResLayer (intra/inter/decode/prefill)
│   ├── config.rs          # BlockAttnResConfig
│   ├── embedding.rs       # TokenEmbedding
│   ├── image_preprocessor.rs  # Vision input pipeline
│   ├── lm_head.rs         # LMHead linear projection
│   ├── moe_linear.rs      # MoELinear with gating
│   ├── shard.rs           # ModelShard / ShardManager
│   └── tokenizer.rs       # SimpleTokenizer
├── tensor/
│   └── gpu_tensor.rs      # GpuTensor high-level wrapper
└── training/
    ├── async_offload.rs   # AsyncGradientOffload staging pool
    ├── checkpointing.rs   # CheckpointStore + CheckpointGranularity
    ├── cpu_offload.rs     # CpuGradientBuffer
    └── optimizer.rs       # SGD, Adam, CrossEntropyLoss
```

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | Done | wgpu foundation, device detection, GPU tensors, compute kernels |
| 2–3 | Done | BlockAttnRes model, tiered compute, caching, batch inference |
| 4 | Done | Autodiff, training, tokenizer, embedding, benches |
| 4.1 | Done | Backward pass, async offload, gradient checkpointing, CPU offload |
| 4.2 | Done | MoE gating, expert dispatch/gather kernels, MoELinear |
| 5 | Done | Streaming inference (RoPE, per-layer KV cache, prefill, flash-decode) |
| 6 | In progress | Test harness, integration tests, public API stabilisation |
| 7 | Planned | Weight loading (safetensors / GGUF), real tokeniser integration |
| 8 | Planned | Vision encoder (ViT + im2col), multimodal input pipeline |
| 9 | Planned | Distributed / multi-GPU training, tensor parallelism |

---

## Contributing

The project is not yet open for external contributions while core development is ongoing. Watch this repository for updates — a contribution guide will be published alongside the public release.

---

## License

FerrisRes is dual-licensed:

**AGPL-3.0-or-later** for free and open-source use. If you use FerrisRes in a
product or service you must publish the complete corresponding source code of
that product or service under the AGPL-3.0. See [`LICENSE`](LICENSE) for the
full terms.

**Commercial license** for use in proprietary or commercial products and
services that do not comply with the AGPL-3.0. To obtain a commercial license,
contact: shift+licensing@someone.section.me

### Contributor License Agreement

Because FerrisRes is dual-licensed, any external code contribution must be
accompanied by a signed Contributor License Agreement (CLA) that grants the
project owner the right to distribute your contribution under both the AGPL-3.0
and a commercial license. A CLA will be published alongside the contribution
guide at the time of the public release.
