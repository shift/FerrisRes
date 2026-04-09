use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use wgpu::Device;

use crate::compute::buffer::GpuBuffer;
use crate::error::Result;

pub struct LayerKVCache {
    key_cache: GpuBuffer,
    value_cache: GpuBuffer,
    current_len: AtomicU32,
    max_seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    device: Arc<Device>,
}

impl LayerKVCache {
    pub fn new(
        device: Arc<Device>,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let elem_count = max_seq_len as usize * num_heads as usize * head_dim as usize;
        let byte_size = elem_count * std::mem::size_of::<f32>();
        let key_cache = GpuBuffer::zeros(&device, byte_size, Some("KVCache Layer Keys"))?;
        let value_cache = GpuBuffer::zeros(&device, byte_size, Some("KVCache Layer Values"))?;

        Ok(Self {
            key_cache,
            value_cache,
            current_len: AtomicU32::new(0),
            max_seq_len,
            num_heads,
            head_dim,
            device,
        })
    }

    pub fn update(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        new_k: &GpuBuffer,
        new_v: &GpuBuffer,
    ) -> Result<u32> {
        let pos = self.current_len.load(Ordering::Relaxed);
        if pos >= self.max_seq_len {
            return Err(crate::error::FerrisResError::Shape(format!(
                "KVCache overflow: current_len {} >= max_seq_len {}",
                pos, self.max_seq_len
            )));
        }

        let per_head_dim = self.head_dim as usize * std::mem::size_of::<f32>();
        let per_pos_size = self.num_heads as usize * per_head_dim;
        let dst_offset = pos as u64 * per_pos_size as u64;
        let copy_size = (new_k.size() as u64).min(per_pos_size as u64);

        encoder.copy_buffer_to_buffer(
            new_k.buffer(),
            0,
            self.key_cache.buffer(),
            dst_offset,
            Some(copy_size),
        );

        encoder.copy_buffer_to_buffer(
            new_v.buffer(),
            0,
            self.value_cache.buffer(),
            dst_offset,
            Some(copy_size),
        );

        self.current_len.fetch_add(1, Ordering::Relaxed);
        Ok(self.current_len.load(Ordering::Relaxed))
    }

    pub fn current_len(&self) -> u32 {
        self.current_len.load(Ordering::Relaxed)
    }

    pub fn key_buffer(&self) -> &GpuBuffer {
        &self.key_cache
    }

    pub fn value_buffer(&self) -> &GpuBuffer {
        &self.value_cache
    }

    pub fn update_batch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        new_k: &GpuBuffer,
        new_v: &GpuBuffer,
        num_tokens: u32,
    ) -> Result<u32> {
        let pos = self.current_len.load(Ordering::Relaxed);
        if pos + num_tokens > self.max_seq_len {
            return Err(crate::error::FerrisResError::Shape(format!(
                "KVCache overflow: current_len {} + num_tokens {} > max_seq_len {}",
                pos, num_tokens, self.max_seq_len
            )));
        }

        let per_pos_size = self.num_heads as u64 * self.head_dim as u64 * std::mem::size_of::<f32>() as u64;
        let dst_offset = pos as u64 * per_pos_size;
        let copy_size = num_tokens as u64 * per_pos_size;

        encoder.copy_buffer_to_buffer(
            new_k.buffer(),
            0,
            self.key_cache.buffer(),
            dst_offset,
            Some(copy_size),
        );

        encoder.copy_buffer_to_buffer(
            new_v.buffer(),
            0,
            self.value_cache.buffer(),
            dst_offset,
            Some(copy_size),
        );

        self.current_len.fetch_add(num_tokens, Ordering::Relaxed);
        Ok(self.current_len.load(Ordering::Relaxed))
    }

    pub fn reset(&self) {
        self.current_len.store(0, Ordering::Relaxed);
    }

    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }

    pub fn num_heads(&self) -> u32 {
        self.num_heads
    }

    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

pub struct ModelKVCache {
    layers: Vec<LayerKVCache>,
}

impl ModelKVCache {
    pub fn new(
        device: Arc<Device>,
        num_layers: u32,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            layers.push(LayerKVCache::new(
                Arc::clone(&device),
                max_seq_len,
                num_heads,
                head_dim,
            ).map_err(|e| crate::error::FerrisResError::Device(format!(
                    "Failed to allocate KV cache for layer {}: {}", i, e
                )))?);
        }

        Ok(Self { layers })
    }

    pub fn layer(&self, idx: usize) -> &LayerKVCache {
        &self.layers[idx]
    }

    pub fn reset_all(&self) {
        for layer in &self.layers {
            layer.reset();
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}
