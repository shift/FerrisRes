# FerrisRes Quick Start

Get FerrisRes running in 5 minutes.

## Prerequisites

- Linux (NixOS recommended) or macOS
- Vulkan driver (Linux) or Metal (macOS)
- [Nix](https://nixos.org/) with flakes enabled

## 1. Enter the dev shell

```bash
git clone https://github.com/shift/FerrisRes.git
cd FerrisRes
nix develop
```

This gives you Rust, Vulkan layers, and all dependencies.

## 2. Build and test

```bash
cargo build --release
cargo test --lib           # 742 tests
```

## 3. Run inference

```bash
# Basic text generation
cargo run --release -- infer --prompt "Explain transformers" --max-tokens 64

# With a template
cargo run --release -- infer --prompt "What is attention?" --template chatml

# Multimodal
cargo run --release -- infer --prompt "Describe this" --image photo.jpg
```

## 4. Start the API server

```bash
cargo run --release -- serve --port 8080
```

Then call it like OpenAI:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ferrisres","messages":[{"role":"user","content":"Hello"}]}'
```

## 5. Distill a model

If you have a Gemma 4 safetensors file:

```bash
cargo run --release -- distill \
  --model-path ./model.safetensors \
  --config e2b \
  --steps 100 \
  --gpu
```

## 6. Use as a library

```toml
[dependencies]
ferrisres = { git = "https://github.com/shift/FerrisRes", tag = "v0.2.0" }
```

```rust
use ferrisres::inference::sampling::sample_top_k;

let logits = vec![1.0, 2.0, 3.0, 0.5];
let token_id = sample_top_k(&logits, 2, 0.8);
println!("Generated token: {}", token_id);
```

## Next steps

- [Architecture deep dive](architecture.md)
- [Distillation guide](distillation.md)
- [API reference](api-reference.md)
- [Deployment guide](deployment.md)
