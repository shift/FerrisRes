//! Immediate data helpers for small kernel parameters.
//!
//! wgpu 29's `IMMEDIATES` feature allows small parameters (< 32 bytes typically)
//! to be passed to shaders without creating uniform buffers and bind groups.
//! This eliminates per-dispatch allocation overhead.
//!
//! In WGSL: `var<immediate> params: Params;`
//! In Rust: `pass.set_immediates(0, bytemuck::cast_slice(&params_data));`
//!
//! Fallback: If IMMEDIATES not supported, use the traditional uniform buffer path.

use wgpu::{Device, ComputePass};

/// Check if the device supports IMMEDIATES.
pub fn device_supports_immediates(device: &Device) -> bool {
    device.features().contains(wgpu::Features::IMMEDIATES)
}

/// Set immediate data on a compute pass.
///
/// This is a no-op if the data is empty.
pub fn set_immediates(pass: &mut ComputePass, data: &[u8]) {
    if !data.is_empty() {
        pass.set_immediates(0, data);
    }
}

/// Pack a slice of u32 into bytes for immediate data.
pub fn pack_u32s(data: &[u32]) -> Vec<u8> {
    bytemuck::cast_slice(data).to_vec()
}

/// Create a uniform buffer from u32 params (fallback path).
///
/// Returns a buffer that must be kept alive for the duration of the compute pass.
pub fn create_params_buffer(device: &Device, data: &[u32]) -> wgpu::Buffer {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Params Buffer"),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytes);
    buffer.unmap();
    buffer
}

/// Small kernel params that can use either immediates or uniform buffers.
///
/// This enum abstracts over the two paths, allowing kernels to use
/// the fast path when available and fall back gracefully.
pub enum SmallParams {
    /// Use immediate data (fast path, no allocation).
    Immediate(Vec<u8>),
    /// Use uniform buffer (fallback, requires allocation).
    Uniform(wgpu::Buffer),
}

impl SmallParams {
    /// Create params, choosing the best path based on device capabilities.
    pub fn new(device: &Device, data: &[u32]) -> Self {
        if device_supports_immediates(device) {
            SmallParams::Immediate(pack_u32s(data))
        } else {
            SmallParams::Uniform(create_params_buffer(device, data))
        }
    }

    /// Apply to a compute pass: either set immediates or use as bind resource.
    pub fn apply_to_pass(&self, pass: &mut ComputePass) {
        if let SmallParams::Immediate(ref data) = self {
            set_immediates(pass, data);
        }
        // For Uniform variant, caller adds to bind group entries
    }

    /// Get the buffer resource (for bind group), if using uniform path.
    pub fn as_binding_resource(&self) -> Option<wgpu::BindingResource<'_>> {
        match self {
            SmallParams::Uniform(buf) => Some(buf.as_entire_binding()),
            SmallParams::Immediate(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_u32s() {
        let data = [1u32, 2, 3, 4];
        let bytes = pack_u32s(&data);
        assert_eq!(bytes.len(), 16);
        // Verify round-trip
        let roundtrip: &[u32] = bytemuck::cast_slice(&bytes);
        assert_eq!(roundtrip, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_small_params_immediate_path() {
        // Just test the packing logic; actual GPU test needs device
        let data = [100u32, 200, 300];
        let bytes = pack_u32s(&data);
        assert_eq!(bytes.len(), 12);
    }
}
