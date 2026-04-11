use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::compute::memory::MemoryPool;
use crate::error::{FerrisResError, Result};
use crate::model::config::BlockAttnResConfig;

pub struct ModelShard {
    shard_id: usize,
    layer_start: usize,
    layer_end: usize,
    loaded: AtomicBool,
    weight_buffers: Vec<GpuBuffer>,
    memory_bytes: u64,
}

impl ModelShard {
    pub fn new(
        shard_id: usize,
        layer_start: usize,
        layer_end: usize,
        device: &Device,
    ) -> Result<Self> {
        let layer_count = layer_end - layer_start;
        let bytes_per_layer: usize = 4 * 1024;
        let shard_bytes = layer_count as u64 * bytes_per_layer as u64;

        let mut weight_buffers = Vec::with_capacity(layer_count);
        for i in 0..layer_count {
            let buffer = GpuBuffer::new(
                device,
                bytes_per_layer,
                Some(&format!("shard{}_weight{}", shard_id, i)),
            )?;
            weight_buffers.push(buffer);
        }

        Ok(Self {
            shard_id,
            layer_start,
            layer_end,
            loaded: AtomicBool::new(false),
            weight_buffers,
            memory_bytes: shard_bytes,
        })
    }

    pub fn layer_range(&self) -> Range<usize> {
        self.layer_start..self.layer_end
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::SeqCst)
    }

    pub fn mark_loaded(&self) {
        self.loaded.store(true, Ordering::SeqCst);
    }

    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    pub fn weights(&self) -> &[GpuBuffer] {
        &self.weight_buffers
    }
}

pub struct ShardManager {
    shards: Vec<ModelShard>,
    current_shard: AtomicUsize,
    prefetch_ahead: usize,
    #[allow(dead_code)]
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    #[allow(dead_code)]
    memory_pool: Arc<MemoryPool>,
}

impl ShardManager {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: &BlockAttnResConfig,
        memory_pool: &MemoryPool,
    ) -> Result<Self> {
        let total_layers = config.total_layers();
        let available = memory_pool.budget().available();
        let bytes_per_layer: u64 = 4 * 1024;
        let layers_per_shard = if available > bytes_per_layer {
            let max_layers = (available / bytes_per_layer).max(1) as usize;
            config.block_size.min(max_layers)
        } else {
            1
        };

        let num_shards = (total_layers + layers_per_shard - 1) / layers_per_shard;
        let mut shards = Vec::with_capacity(num_shards);

        for i in 0..num_shards {
            let start = i * layers_per_shard;
            let end = (start + layers_per_shard).min(total_layers);
            let shard = ModelShard::new(i, start, end, &device)?;
            shards.push(shard);
        }

        let prefetch_ahead = if num_shards >= 2 { 1 } else { 0 };

        let pool_device = device.clone();
        Ok(Self {
            shards,
            current_shard: AtomicUsize::new(0),
            prefetch_ahead,
            device,
            queue,
            memory_pool: Arc::new(MemoryPool::new(
                pool_device,
                &crate::device::Capability::detect(),
            )),
        })
    }

    pub fn load_shard(&self, shard_id: usize) -> Result<()> {
        let shard = self
            .shards
            .get(shard_id)
            .ok_or_else(|| FerrisResError::Device(format!("Invalid shard id: {}", shard_id)))?;
        shard.mark_loaded();
        Ok(())
    }

    pub fn ensure_loaded(&self, layer_idx: usize) -> Result<()> {
        let shard = self.shard_for_layer(layer_idx);
        if !shard.is_loaded() {
            self.load_shard(shard.shard_id)?;
        }
        self.current_shard.store(shard.shard_id, Ordering::SeqCst);
        Ok(())
    }

    pub fn prefetch(&self, current_layer: usize) {
        if let Some(current_shard) = self
            .shards
            .iter()
            .find(|s| current_layer >= s.layer_start && current_layer < s.layer_end)
        {
            for offset in 1..=self.prefetch_ahead {
                let target_id = current_shard.shard_id + offset;
                if target_id < self.shards.len() {
                    let _ = self.load_shard(target_id);
                }
            }
        }
    }

    pub fn current_shard(&self) -> usize {
        self.current_shard.load(Ordering::SeqCst)
    }

    pub fn shard_for_layer(&self, layer_idx: usize) -> &ModelShard {
        self.shards
            .iter()
            .find(|s| layer_idx >= s.layer_start && layer_idx < s.layer_end)
            .expect("layer_idx out of range")
    }

    pub fn total_shards(&self) -> usize {
        self.shards.len()
    }

    pub fn memory_footprint(&self) -> u64 {
        self.shards
            .iter()
            .filter(|s| s.is_loaded())
            .map(|s| s.memory_bytes())
            .sum()
    }
}

pub enum QuantDtype {
    F32,
    F16,
    Int8,
    Int4,
}

pub struct QuantizedBuffer {
    buffer: GpuBuffer,
    #[allow(dead_code)]
    scale: GpuBuffer,
    #[allow(dead_code)]
    zero_point: GpuBuffer,
    dtype: QuantDtype,
}

impl QuantizedBuffer {
    /// Create new f32 buffer
    pub fn new_f32(device: &Device, size: usize) -> Result<Self> {
        let buffer = GpuBuffer::new(device, size, Some("quant_f32"))?;
        let scale = GpuBuffer::new(device, size, Some("quant_f32_scale"))?;
        let zero_point = GpuBuffer::new(device, size, Some("quant_f32_zp"))?;
        Ok(Self {
            buffer,
            scale,
            zero_point,
            dtype: QuantDtype::F32,
        })
    }
    
    /// Create new int8 buffer (with scale for dequantization)
    pub fn new_int8(device: &Device, size: usize) -> Result<Self> {
        // Int8 uses 1 byte per element vs 4 for f32
        let buffer_size = size * std::mem::size_of::<i8>();
        let scale_size = size * std::mem::size_of::<f32>();
        
        let buffer = GpuBuffer::new(device, buffer_size, Some("quant_i8"))?;
        let scale = GpuBuffer::new(device, scale_size, Some("quant_i8_scale"))?;
        let zero_point = GpuBuffer::new(device, scale_size, Some("quant_i8_zp"))?;
        
        Ok(Self {
            buffer,
            scale,
            zero_point,
            dtype: QuantDtype::Int8,
        })
    }
    
    /// Create new int4 buffer (4-bit, packed)
    pub fn new_int4(device: &Device, size: usize) -> Result<Self> {
        // Int4 packs 2 values per byte
        let buffer_size = (size + 1) / 2;
        let scale_size = size * std::mem::size_of::<f32>();
        
        let buffer = GpuBuffer::new(device, buffer_size, Some("quant_i4"))?;
        let scale = GpuBuffer::new(device, scale_size, Some("quant_i4_scale"))?;
        let zero_point = GpuBuffer::new(device, scale_size, Some("quant_i4_zp"))?;
        
        Ok(Self {
            buffer,
            scale,
            zero_point,
            dtype: QuantDtype::Int4,
        })
    }
    
    /// Create new fp16 buffer
    pub fn new_f16(device: &Device, size: usize) -> Result<Self> {
        let buffer_size = size * std::mem::size_of::<u16>(); // f16 as u16 repr
        let scale_size = size * std::mem::size_of::<f32>(); // scale for upcast
        
        let buffer = GpuBuffer::new(device, buffer_size, Some("quant_f16"))?;
        let scale = GpuBuffer::new(device, scale_size, Some("quant_f16_scale"))?;
        let zero_point = GpuBuffer::new(device, scale_size, Some("quant_f16_zp"))?;
        
        Ok(Self {
            buffer,
            scale,
            zero_point,
            dtype: QuantDtype::F16,
        })
    }

    pub fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn dtype(&self) -> &QuantDtype {
        &self.dtype
    }


    /// Quantize raw f32 data into a QuantizedBuffer on the GPU.
    /// `device` and `queue` are needed to upload the quantized data.
    #[allow(dead_code)]
    pub fn quantize_data(
        device: &Device,
        queue: &wgpu::Queue,
        data: &[f32],
        target: QuantDtype,
    ) -> Result<QuantizedBuffer> {
        match target {
            QuantDtype::F32 => {
                let bytes = bytemuck::cast_slice(data);
                let buffer = GpuBuffer::new(device, bytes.len(), Some("quant_f32"))?;
                queue.write_buffer(buffer.buffer(), 0, bytes);
                let scale = GpuBuffer::zeros(device, queue, data.len() * 4, Some("quant_f32_scale"))?;
                let zp = GpuBuffer::zeros(device, queue, data.len() * 4, Some("quant_f32_zp"))?;
                Ok(QuantizedBuffer { buffer, scale, zero_point: zp, dtype: QuantDtype::F32 })
            }
            QuantDtype::F16 => {
                let n = data.len();
                let mut f16_bytes = Vec::with_capacity(n * 2);
                let scales = vec![1.0f32; n];
                let zero_points = vec![0.0f32; n];

                for &val in data {
                    let bits = f32_to_f16_bits(val);
                    f16_bytes.push((bits & 0xFF) as u8);
                    f16_bytes.push(((bits >> 8) & 0xFF) as u8);
                }

                let buffer = GpuBuffer::new(device, f16_bytes.len(), Some("quant_f16"))?;
                queue.write_buffer(buffer.buffer(), 0, &f16_bytes);
                let scale = GpuBuffer::zeros(device, queue, n * 4, Some("quant_f16_scale"))?;
                queue.write_buffer(scale.buffer(), 0, bytemuck::cast_slice(&scales));
                let zp = GpuBuffer::zeros(device, queue, n * 4, Some("quant_f16_zp"))?;
                queue.write_buffer(zp.buffer(), 0, bytemuck::cast_slice(&zero_points));

                Ok(QuantizedBuffer { buffer, scale, zero_point: zp, dtype: QuantDtype::F16 })
            }
            QuantDtype::Int8 => {
                let n = data.len();
                let mut i8_data = Vec::with_capacity(n);
                let mut scales = vec![0.0f32; n];
                let zero_points = vec![0.0f32; n];

                for (i, &val) in data.iter().enumerate() {
                    let abs_max = val.abs().max(1e-8);
                    let scale = abs_max / 127.0;
                    let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
                    i8_data.push(q);
                    scales[i] = scale;
                }

                let buffer = GpuBuffer::new(device, n, Some("quant_i8"))?;
                queue.write_buffer(buffer.buffer(), 0, bytemuck::cast_slice(&i8_data));
                let scale = GpuBuffer::zeros(device, queue, n * 4, Some("quant_i8_scale"))?;
                queue.write_buffer(scale.buffer(), 0, bytemuck::cast_slice(&scales));
                let zp = GpuBuffer::zeros(device, queue, n * 4, Some("quant_i8_zp"))?;
                queue.write_buffer(zp.buffer(), 0, bytemuck::cast_slice(&zero_points));

                Ok(QuantizedBuffer { buffer, scale, zero_point: zp, dtype: QuantDtype::Int8 })
            }
            QuantDtype::Int4 => {
                let n = data.len();
                let block_size = 32;
                let num_blocks = (n + block_size - 1) / block_size;

                let mut scales = vec![0.0f32; n];
                let zero_points = vec![0.0f32; n];
                let mut nibbles = vec![0u8; n];

                for block_idx in 0..num_blocks {
                    let start = block_idx * block_size;
                    let end = (start + block_size).min(n);
                    let block = &data[start..end];

                    let abs_max = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-8);
                    let scale = abs_max / 7.0;

                    for (j, &val) in block.iter().enumerate() {
                        let idx = start + j;
                        let q = (val / scale).round().clamp(-8.0, 7.0) as i8;
                        nibbles[idx] = (q & 0x0F) as u8;
                        scales[idx] = scale;
                    }
                }

                // Pack two nibbles per byte
                let packed_len = (n + 1) / 2;
                let mut packed = vec![0u8; packed_len];
                for i in 0..n {
                    let byte_idx = i / 2;
                    if i % 2 == 0 {
                        packed[byte_idx] = nibbles[i] & 0x0F;
                    } else {
                        packed[byte_idx] |= (nibbles[i] & 0x0F) << 4;
                    }
                }

                let buffer = GpuBuffer::new(device, packed_len, Some("quant_i4"))?;
                queue.write_buffer(buffer.buffer(), 0, &packed);
                let scale = GpuBuffer::zeros(device, queue, n * 4, Some("quant_i4_scale"))?;
                queue.write_buffer(scale.buffer(), 0, bytemuck::cast_slice(&scales));
                let zp = GpuBuffer::zeros(device, queue, n * 4, Some("quant_i4_zp"))?;
                queue.write_buffer(zp.buffer(), 0, bytemuck::cast_slice(&zero_points));

                Ok(QuantizedBuffer { buffer, scale, zero_point: zp, dtype: QuantDtype::Int4 })
            }
        }
    }

    /// Get compression ratio vs f32
    #[allow(dead_code)]
    pub fn compression_ratio(&self) -> f32 {
        match self.dtype {
            QuantDtype::F32 => 1.0,
            QuantDtype::F16 => 2.0,
            QuantDtype::Int8 => 4.0,
            QuantDtype::Int4 => 8.0,
        }
    }
}

/// Convert f32 to f16 bits (IEEE 754 half-precision).
/// Returns the 16-bit representation as u16.
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x007F_FFFF) as u32;

    if exp == 0 {
        return sign;
    }
    if exp == 0xFF {
        if mant != 0 {
            return sign | 0x7FFF; // NaN
        }
        return sign | 0x7C00; // Inf
    }

    // Normal number: rebias exponent from 127 to 15
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign;
    }
    if new_exp >= 0x1F {
        return sign | 0x7C00;
    }

    let new_mant = (mant >> 13) as u16;
    sign | ((new_exp as u16) << 10) | new_mant
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_zero() {
        assert_eq!(f32_to_f16_bits(0.0), 0u16);
        assert_eq!(f32_to_f16_bits(-0.0), 0x8000u16);
    }

    #[test]
    fn test_f32_to_f16_inf() {
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7C00u16);
        assert_eq!(f32_to_f16_bits(f32::NEG_INFINITY), 0xFC00u16);
    }

    #[test]
    fn test_f32_to_f16_one() {
        // 1.0 in f16 = 0x3C00 (sign=0, exp=15, mant=0)
        assert_eq!(f32_to_f16_bits(1.0), 0x3C00u16);
    }

    #[test]
    fn test_f32_to_f16_two() {
        // 2.0 in f16 = 0x4000 (sign=0, exp=16, mant=0)
        assert_eq!(f32_to_f16_bits(2.0), 0x4000u16);
    }

    #[test]
    fn test_f32_to_f16_half() {
        // 0.5 in f16 = 0x3800 (sign=0, exp=14, mant=0)
        assert_eq!(f32_to_f16_bits(0.5), 0x3800u16);
    }

    #[test]
    fn test_f32_to_f16_neg() {
        // -1.0 in f16 = 0xBC00 (sign=1, exp=15, mant=0)
        assert_eq!(f32_to_f16_bits(-1.0), 0xBC00u16);
    }

    #[test]
    fn test_f32_to_f16_nan() {
        let nan_bits = f32_to_f16_bits(f32::NAN);
        // NaN has exponent = 0x1F and mantissa != 0
        assert_ne!(nan_bits & 0x7C00, 0); // exponent is all 1s
        assert_ne!(nan_bits & 0x03FF, 0); // mantissa is nonzero
    }

    #[test]
    fn test_f32_to_f16_large() {
        // 65504.0 is the largest representable f16
        assert_eq!(f32_to_f16_bits(65504.0), 0x7BFFu16);
        // Beyond that overflows to inf
        assert_eq!(f32_to_f16_bits(131072.0), 0x7C00u16);
    }

    #[test]
    fn test_int8_quantization_values() {
        // Test quantization math directly
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        
        for &val in &data {
            let abs_max: f32 = val.abs().max(1e-8);
            let scale: f32 = abs_max / 127.0;
            let q = (val / scale).round().clamp(-128.0_f32, 127.0_f32) as i8;
            // Dequantize
            let recovered = q as f32 * scale;
            let error: f32 = (val - recovered).abs();
            assert!(error < 0.02, "Int8 roundtrip error too large: {} -> {} -> {} (err={})", val, q, recovered, error);
        }
    }

    #[test]
    fn test_int4_quantization_values() {
        let data: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5];
        
        let block_size = 32;
        let mut block = vec![0.0_f32; block_size];
        block[..data.len()].copy_from_slice(&data);

        let abs_max: f32 = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max).max(1e-8);
        let scale: f32 = abs_max / 7.0;

        for &val in &data {
            let q = (val / scale).round().clamp(-8.0_f32, 7.0_f32) as i8;
            let recovered = q as f32 * scale;
            let error: f32 = (val - recovered).abs();
            assert!(error < 0.2, "Int4 roundtrip error too large: {} -> {} -> {} (err={})", val, q, recovered, error);
        }
    }
}
