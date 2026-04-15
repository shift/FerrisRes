# FerrisRes Deployment Guide

## Hardware Requirements

| Configuration | Minimum RAM | Recommended GPU | Use Case |
|---|---|---|---|
| E2B (2B params) | 8 GB | Any Vulkan/Metal GPU | Development, testing |
| E4B (4B params) | 16 GB | 4 GB VRAM | Production inference |
| 27B MM | 16 GB | 8 GB VRAM | High-quality inference |

FerrisRes auto-detects hardware via `DeviceProfile` and adapts:
- **Integrated GPU**: Shared memory, minimal GPU usage, CPU-heavy
- **Low-End GPU** (<4GB VRAM): JIT weight uploads, small batch sizes
- **Mid-Range GPU** (4-8GB): Partial GPU offload, moderate batching
- **High-End GPU** (>8GB): Full GPU acceleration, large batches

## Running as a Service

### systemd (Linux)

```ini
[Unit]
Description=FerrisRes Inference Server
After=network.target

[Service]
Type=simple
User=ferrisres
WorkingDirectory=/opt/ferrisres
ExecStart=/opt/ferrisres/ferrisres serve --port 8080
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Docker

```dockerfile
FROM rust:1.94 as build
COPY . /app
WORKDIR /app
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=build /app/target/release/ferrisres /usr/local/bin/
EXPOSE 8080
CMD ["ferrisres", "serve", "--port", "8080"]
```

### NixOS

```nix
# flake.nix or configuration.nix
systemd.services.ferrisres = {
  wantedBy = [ "multi-user.target" ];
  serviceConfig.ExecStart = "${pkgs.ferrisres}/bin/ferrisres serve --port 8080";
};
```

## Model Management

### Downloading models

```bash
# Gemma 4 from HuggingFace
huggingface-cli download google/gemma-4-27b-mm-it \
  --include "*.safetensors" \
  --local-dir ./models/gemma-4-27b
```

### Memory-mapped loading

FerrisRes uses `mmap` by default for safetensors files. The model stays on disk
and is loaded on-demand, keeping RAM usage low:

```
# 10GB model on disk, only ~2GB in RAM during inference
./ferrisres infer --model-path ./models/gemma-4-27b/model.safetensors
```

### Quantized models (GGUF)

```bash
# Load a quantized GGUF model
./ferrisres infer --model-path ./models/llama-7b-q4_k.gguf
```

## Monitoring

### Health check

```bash
curl http://localhost:8080/health
```

### Metrics

FerrisRes logs structured metrics via `tracing`:

```
INFO ferrisres: Step 0: loss=0.000000 bridge_w=0.0000 lr=0.000000
INFO ferrisres::model::gpu_forward: GPU initialized: NVIDIA RTX 4090 (vulkan), profile=HighEnd
```

### Performance tuning

- Set `FERRIS_DEVICE_PROFILE=high_end` to override auto-detection
- Use `for distillation to get 10-20x speedup
- Adjust `--seq-len` based on available RAM (longer = more RAM)
- Use TurboQuant 2-bit compression for 16x KV cache reduction

## Security

### WASM Sandbox

All tool execution runs in a WASM sandbox with:
- No filesystem access
- No network access
- Bounded memory (16 MB default)
- Bounded CPU (10M instruction fuel)
- Sub-millisecond execution

### Shell execution

The `shell_exec` tool uses a blocklist to prevent dangerous commands:
```bash
# Blocked: rm, chmod, sudo, dd, mkfs, etc.
# Allowed: ls, cat, echo, grep, python3, gcc, etc.
```

### Production hardening

1. Run as non-root user
2. Use a reverse proxy (nginx/caddy) for TLS
3. Rate-limit the API endpoint
4. Keep model files read-only
5. Audit `host_tools.rs` for your threat model
