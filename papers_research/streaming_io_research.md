# Streaming I/O Research

## Overview
Research for efficient streaming of image, audio, and video data.

## Image Streaming (e74071b8)
- Format: JPEG, WebP, PNG streaming
- Implementation: Chunked decode
- wgpu texture update pipeline
- Latency target: <50ms per frame

## Audio Streaming (d1477929)
- Format: PCM, Opus, EnCodec
- Buffer: Ring buffer for latency
- Sample rate: 16kHz-44.1kHz
- Latency target: <20ms

## Readback Latency (a6c9abed)
- GPU→CPU transfer bottleneck
- Options:
  - Mapped memory (lowest latency)
  - Staging buffer (medium)
  - Async Pipeline (best throughput)

## wgpu Implementation
```rust
// Async readback pattern
let buffer = device.create_buffer(...);
queue.write_buffer(&buffer, data);
// Async map
buffer.map_async(|_| callback);
```

## Status: DONE
