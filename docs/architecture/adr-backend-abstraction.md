# ADR: Backend Abstraction Strategy

**Status**: Accepted  
**Date**: 2026-04-20  
**Task**: `6aedcf76`

## Context

FerrisRes needs to run on a spectrum of hardware: Raspberry Pi → Android WebGPU → iOS → Mac Metal → NVIDIA/AMD → multi-GPU. We need a backend abstraction that:

1. Supports GPU inference via wgpu (Vulkan, Metal, DX12, WebGPU)
2. Supports CPU-only inference for edge devices without GPUs
3. Exploits platform-specific GPU features (subgroups, cooperative matrix, f16, etc.)
4. Doesn't add runtime overhead from abstraction layers
5. Is maintainable for a small team

Two approaches were considered:

### Option A: Trait-based TensorBackend

```rust
trait TensorBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn rmsnorm(&self, x: &Tensor, weight: &Tensor) -> Tensor;
    // ... 20+ methods
}
struct CpuBackend;
struct WgpuBackend { device: Arc<Device> }
```

**Pros**: Clean abstraction, testable, swap backends at runtime  
**Cons**: Dynamic dispatch overhead, doubles API surface, hides platform-specific features, requires generic tensor types, monomorphization bloat

### Option B: wgpu-only GPU + separate CPU path (CHOSEN)

```rust
// GPU path: direct wgpu calls, platform features via GpuCapabilities
let caps = GpuCapabilities::from_device(&device);
if caps.can_subgroup_flash_decode {
    // use subgroup variant
} else {
    // use basic variant
}

// CPU path: direct native Rust (no abstraction)
fn cpu_attention(q: &[f32], k: &[f32], ...) -> Vec<f32> { ... }
```

**Pros**: Zero overhead, full platform feature access, simple mental model, existing code already does this  
**Cons**: Two separate code paths to maintain (but they serve fundamentally different hardware)

## Decision

**Option B: wgpu-only GPU + separate CPU path.**

### Rationale

1. **Existing code already uses this pattern**: `CpuBlockAttnResLayer` (CPU) and `BlockAttnResLayer` (GPU) are separate structs, not trait impls. This works well.

2. **DeviceProfile/DispatchPlan already handles routing**: The `src/device/` infrastructure decides CPU vs GPU at the model level, not per-operation. No need for per-op traits.

3. **Platform-specific features are critical**: Subgroup ops, cooperative matrix, f16, f64 — these are accessed through wgpu feature flags, not abstract tensor operations. A trait would either hide them (losing optimization) or expose them (defeating the abstraction).

4. **Two backends, not ten**: We only have CPU and wgpu. A trait system adds complexity for a 2-backend world.

5. **Performance is paramount**: On RPi, every nanosecond matters. Dynamic dispatch or generic tensor abstractions add overhead we can't afford.

6. **wgsl kernels ARE the abstraction**: WGSL compiles to Vulkan SPIR-V, Metal MSL, DX12 DXIL, and WebGPU. wgpu IS the portable backend. Adding another abstraction on top is redundant.

### Architecture

```
┌─────────────────────────────────────────────┐
│              DeviceProfile                   │
│  (detects hardware, chooses dispatch plan)  │
└────────────────┬────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
    ┌────▼────┐    ┌─────▼─────┐
    │ CPU Path│    │  GPU Path │
    │         │    │           │
    │ Native  │    │  wgpu     │
    │ Rust    │    │  + WGSL   │
    │ (f32)   │    │  kernels  │
    └─────────┘    └───────────┘
                        │
                ┌───────┴───────┐
                │ GpuCapabilities│
                │ (feature flags)│
                └───────┬───────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    Basic path    Subgroup path   CoopMatrix path
    (all devices)  (Intel/NVIDIA)  (Intel iGPU)
```

### What's shared between CPU and GPU

- **Model weights**: `CpuBlockAttnResModel` → `BlockAttnResModel` via `upload.rs`
- **Model structure**: Same layer indices, block boundaries, PLE config
- **Quantized formats**: TernaryLinear, TernaryMoELayer (both paths use same packed format)
- **Training**: LoRA/QLoRA on CPU, deploy to GPU
- **Tokenizer, config, safetensors loading**: All platform-independent

### What's different between CPU and GPU

- **Attention**: CPU does direct f32 matmul; GPU uses FlashDecode/PrefillAttn WGSL
- **FFN**: CPU does direct matmul; GPU uses MoELinear/Linear WGSL dispatch
- **KV cache**: CPU uses Vec<f32>; GPU uses wgpu::Buffer with optional compression
- **RoPE**: CPU uses f64 precision; GPU uses Cody-Waite f32 WGSL

## Consequences

- No `TensorBackend` trait — use `CpuBlockAttnResLayer` and `BlockAttnResLayer` directly
- `DeviceProfile` chooses which path; model upload bridges between them
- Platform-specific optimizations go directly into WGSL kernels or CPU functions
- New platforms get a new WGSL kernel variant, not a new trait impl
