# Research: wgpu 29 Platform-Specific Feature Exploitation

**Task ID:** Cross-cutting — informs DeviceProfile expansion for all tasks

## Key Finding: wgpu 29 Feature Landscape

wgpu 29.0.1 exposes platform-specific GPU features that FerrisRes should exploit
per DeviceProfile. Currently the codebase uses `Features::empty()` or `IMMEDIATES`
only. Every shader uses the same path regardless of what the hardware supports.

## Feature Availability by Platform

| Feature | Intel iGPU | NVIDIA | AMD | Apple Metal | Android | RPi 4/5 |
|---------|-----------|--------|-----|-------------|---------|---------|
| SHADER_F64 | ✅ | ✅ (slow) | ✅ (slow) | ❌ | ❌ | ❌ |
| SUBGROUPS | ✅ (8-32) | ✅ (32) | ✅ (32/64) | ✅ (32) | ❌ | ❌ |
| COOPERATIVE_MATRIX | ✅ | ✅ | ✅ (RDNA3+) | ✅ (M3+) | ❌ | ❌ |
| SHADER_F16 | ✅ | ✅ | ✅ | ✅ | partial | ❌ |
| SHADER_INT64 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| PUSH_CONSTANTS | ✅ 256B | ✅ 256B | ✅ 256B | ✅ 256B | ✅ 128B | ✅ 128B |
| IMMEDIATES | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FLOAT32_ATOMIC | ✅ | ✅ | ✅ | varies | varies | ❌ |
| TEXTURE_ATOMIC | ✅ | ✅ | ✅ | varies | ❌ | ❌ |
| SHADER_INT64_ATOMIC | ✅ | ✅ | partial | ❌ | ❌ | ❌ |

## How This Changes the Architecture

### 1. RoPE Precision (Task 4) — SHADER_F64 Path

**Before:** Assumed f64 not available, planned range reduction for everything.
**After:** Three-tier strategy based on features:

- **F64 available** (Intel iGPU, NVIDIA, AMD): Use `f64` intermediates in WGSL.
  `let theta_f64 = f64(pos) * f64(freq); let cos_t = f32(cos(theta_f64));`
  Require `Features::SHADER_F64` at device creation.
  Performance: negligible cost on Intel iGPU (full-rate FP64 on Gen9+),
  ~1/64 rate on NVIDIA (still fine for RoPE — it's a tiny fraction of total compute).

- **F16 available, no F64** (Apple Metal, some Android): Use `f32` with Payne-Hanek
  range reduction. This is the standard libm approach — reduce theta to [-π/4, π/4]
  before computing cos/sin. No precision loss for any practical position.
  F16 can be used for the intermediate expansion terms.

- **Neither F64 nor F16** (RPi 4/5): Plain f32 with Cody-Waite range reduction.
  Adequate up to ~2M positions (well beyond 512k target).

### 2. Attention Kernels — SUBGROUPS Path

Subgroup operations allow warp-level primitives without shared memory overhead.

**Impact on FlashDecodeTiledOp:**
- Current: uses `var<workgroup>` shared memory + `workgroupBarrier()`.
- With SUBGROUPS: use `subgroupShuffle()` + `subgroupAdd()` for reduction.
  Eliminates shared memory allocation and barriers. ~2x faster on NVIDIA/AMD.
- Apple Metal: `simdgroup` is the Metal equivalent. wgpu maps SUBGROUPS to it.
- Fallback: current workgroup-shared path for non-subgroup devices.

**New shader variants needed:**
```
flash_decode_subgroup.wgsl  — for SUBGROUP devices
flash_decode_tiled.wgsl     — current, for non-SUBGROUP devices
```

### 3. Matmul — COOPERATIVE_MATRIX Path

VK_KHR_cooperative_matrix / Metal simd_matrix provide matrix-multiply accumulate
at the warp level. This is essentially a tensor core / matrix core abstraction.

**Impact on MatMulOp:**
- Current: tiled matmul with shared memory (16×16 tiles).
- With COOPERATIVE_MATRIX: use `cooperativeMatrixMulAdd()` builtins.
  On NVIDIA: maps to HMMA instructions (tensor cores).
  On Intel: maps to DPAS instructions (Xe matrix engine).
  On Apple M3+: maps to simd_matrix instructions.
- 2-8x throughput improvement for matmul-heavy workloads.
- Fallback: current tiled matmul for non-cooperative devices.

**Availability:** Intel iGPU (confirmed), NVIDIA, AMD RDNA3+, Apple M3+.
Not available on Android or RPi.

### 4. Uniform Binding — PUSH_CONSTANTS / IMMEDIATES

Currently RoPE params, softmax scales, etc. use full uniform buffers (with
buffer creation + write per dispatch). This is wasteful.

- **PUSH_CONSTANTS** (128-256 bytes): Zero-allocation way to pass small params.
  RoPE params = 16 bytes. Flash decode params = 16 bytes. All fit easily.
  Eliminates the per-dispatch buffer allocation for params.
- **IMMEDIATES** (wgpu 29 new): Even simpler API — `set_immediates()` on the
  compute pass. No bind group needed. The params are embedded in the command buffer.
  Already partially plumbed in `src/compute/pipeline.rs`.

### 5. Proposed DeviceProfile Expansion

```rust
enum DeviceProfile {
    // Existing
    Integrated,  // Intel/AMD iGPU
    LowEnd,      // RPi 4/5, old mobile
    MidRange,    // Mid-tier discrete, modern mobile
    HighEnd,     // RTX 40xx, Apple M3+

    // NEW: More granular profiles
    WebGPU,      // Browser sandbox — WebGPU spec minimum
    EdgeNoGPU,   // CPU-only via lavapipe
}

struct GpuCapabilities {
    profile: DeviceProfile,
    // Feature flags — checked at device creation
    has_f64: bool,
    has_subgroups: bool,
    subgroup_size: u32,
    has_cooperative_matrix: bool,
    has_f16: bool,
    has_push_constants: bool,
    has_immediates: bool,
    has_float32_atomics: bool,
    // Limits
    max_storage_buffer: u64,
    max_workgroup_size: u32,
    max_push_constant_size: u32,
    // Vendor tuning
    vendor: GpuVendor,
    vendor_tuning: VendorTuning,
}
```

Each kernel dispatch then:
```rust
fn dispatch_rope(&self, ..., caps: &GpuCapabilities) {
    match (caps.has_f64, caps.has_f16) {
        (true, _) => self.dispatch_rope_f64(...),
        (false, true) => self.dispatch_rope_f16_range_reduce(...),
        (false, false) => self.dispatch_rope_f32_cody_waite(...),
    }
}

fn dispatch_flash_decode(&self, ..., caps: &GpuCapabilities) {
    if caps.has_subgroups {
        self.dispatch_flash_decode_subgroup(...)
    } else {
        self.dispatch_flash_decode_tiled(...)
    }
}
```

## Key wgpu 29 New Features

1. **IMMEDIATES**: Push-constant-like data for compute shaders without bind groups.
   `compute_pass.set_immediates(offset, data)`. Already partially used in pipeline.rs.

2. **EXPERIMENTAL_COOPERATIVE_MATRIX**: Matrix multiply at warp level.
   Requires `Features::EXPERIMENTAL_COOPERATIVE_MATRIX`. Query via
   `adapter.cooperative_matrix_properties()`.

3. **EXPERIMENTAL_PASSTHROUGH_SHADERS**: Pre-compiled SPIR-V passthrough.
   Could speed up pipeline creation for fixed kernels.

4. **SUBGROUP**: Warp-level vote, shuffle, arithmetic, ballot operations.
   Available on all desktop GPUs and Apple Metal. NOT available in WebGPU.

## Research References

- wgpu 29 changelog: https://github.com/gfx-rs/wgpu/releases/tag/v29.0.0
- VK_KHR_cooperative_matrix: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_cooperative_matrix.html
- Metal simdgroup: https://developer.apple.com/documentation/metal/simd
- Payne-Hanek range reduction: Payne & Hanek (1983), "Radian Reduction for Trigonometric Functions"
- Cody-Waite range reduction: Cody & Waite (1980), "Software Manual for the Elementary Functions"
