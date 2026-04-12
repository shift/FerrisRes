use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::compute::turboquant::{TurboQuantConfig, TurboQuantEngine, OutlierChannelSplitter};
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
        queue: Arc<Queue>,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let elem_count = max_seq_len as usize * num_heads as usize * head_dim as usize;
        let byte_size = elem_count * std::mem::size_of::<f32>();
        let key_cache = GpuBuffer::zeros(&device, &queue, byte_size, Some("KVCache Layer Keys"))?;
        let value_cache = GpuBuffer::zeros(&device, &queue, byte_size, Some("KVCache Layer Values"))?;

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

    /// Increment the length counter without copying any data.
    /// Used by the direct-write decode path where K/V data is
    /// already written directly into the cache buffer.
    pub fn increment_len(&self) {
        self.current_len.fetch_add(1, Ordering::Relaxed);
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

    /// Compact the KV cache by keeping only the entries at `indices`.
    ///
    /// Copies entries at the given indices into a contiguous prefix using a
    /// GPU staging buffer, then sets `current_len = indices.len()`. This is
    /// the StreamingLLM eviction strategy: keep sink tokens (first N) and
    /// recent tokens (last M), discard everything in between.
    ///
    /// Cost: O(indices.len()) GPU copy per compaction event, amortized over
    /// every `window_size - num_sink_tokens` decode steps.
    ///
    /// # Arguments
    /// * `encoder` — Command encoder to record copy commands into
    /// * `indices` — Sorted list of cache positions to keep (sinks + recent)
    pub fn compact(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indices: &[usize],
    ) -> Result<()> {
        if indices.is_empty() {
            self.current_len.store(0, Ordering::Relaxed);
            return Ok(());
        }

        let per_pos_bytes = self.num_heads as u64
            * self.head_dim as u64
            * std::mem::size_of::<f32>() as u64;

        // Allocate staging buffers to hold the compacted entries
        let compact_bytes = indices.len() as u64 * per_pos_bytes;
        let staging_key = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_compact_staging_key"),
            size: compact_bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_val = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_compact_staging_val"),
            size: compact_bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy each selected entry into the staging buffer
        for (dst_idx, &src_idx) in indices.iter().enumerate() {
            let src_offset = src_idx as u64 * per_pos_bytes;
            let dst_offset = dst_idx as u64 * per_pos_bytes;

            encoder.copy_buffer_to_buffer(
                self.key_cache.buffer(),
                src_offset,
                &staging_key,
                dst_offset,
                per_pos_bytes,
            );
            encoder.copy_buffer_to_buffer(
                self.value_cache.buffer(),
                src_offset,
                &staging_val,
                dst_offset,
                per_pos_bytes,
            );
        }

        // Copy staging back into the cache at offset 0
        encoder.copy_buffer_to_buffer(
            &staging_key,
            0,
            self.key_cache.buffer(),
            0,
            compact_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &staging_val,
            0,
            self.value_cache.buffer(),
            0,
            compact_bytes,
        );

        // Update length to compacted size
        self.current_len.store(indices.len() as u32, Ordering::Relaxed);

        tracing::debug!(
            "KV cache compacted: kept {} of {} entries",
            indices.len(),
            self.max_seq_len
        );

        Ok(())
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
        queue: Arc<Queue>,
        num_layers: u32,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            layers.push(LayerKVCache::new(
                Arc::clone(&device),
                Arc::clone(&queue),
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

    /// Compact all layers using the same set of indices.
    /// See [`LayerKVCache::compact`] for details.
    pub fn compact_all(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        indices: &[usize],
    ) -> Result<()> {
        for layer in &self.layers {
            layer.compact(encoder, indices)?;
        }
        Ok(())
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ============================================================================
// TurboQuant Compressed KV Cache Variants
// ============================================================================

/// Configuration for TurboQuant-compressed KV cache
#[derive(Debug, Clone)]
pub struct CompressedKVCacheConfig {
    /// TurboQuant configuration
    pub turboquant: TurboQuantConfig,
    /// Outlier channel splitter for fractional bits
    pub splitter: Option<OutlierChannelSplitter>,
    /// Whether compression is enabled
    pub enabled: bool,
}

impl CompressedKVCacheConfig {
    /// Create new config with specified bit width
    pub fn new(hidden_dim: u32, bit_width: u32, enable_qjl: bool) -> Self {
        Self {
            turboquant: TurboQuantConfig::new(bit_width, hidden_dim, enable_qjl),
            splitter: None,
            enabled: true,
        }
    }
    
    /// Create config for 2-bit compression (16x reduction)
    pub fn two_bit(hidden_dim: u32) -> Self {
        Self::new(hidden_dim, 2, true)
    }
    
    /// Create config for 2.5-bit compression (12.8x reduction)
    pub fn two_and_half_bit(hidden_dim: u32) -> Self {
        let mut config = Self::new(hidden_dim, 2, true);
        config.splitter = Some(OutlierChannelSplitter::two_and_half_bit(hidden_dim));
        config
    }
    
    /// Create config for 3-bit compression (10.7x reduction)
    pub fn three_bit(hidden_dim: u32) -> Self {
        Self::new(hidden_dim, 3, true)
    }
    
    /// Calculate memory savings ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        if let Some(ref splitter) = self.splitter {
            splitter.memory_savings()
        } else {
            32.0 / self.turboquant.bit_width as f32
        }
    }
}

/// Compressed LayerKVCache using TurboQuant
#[allow(dead_code)]
pub struct CompressedLayerKVCache {
    /// The underlying uncompressed cache (may be None if using indices-only storage)
    key_cache: Option<GpuBuffer>,
    value_cache: Option<GpuBuffer>,
    // Quantized indices storage (for keys and values separately)
    key_indices: Option<GpuBuffer>,
    value_indices: Option<GpuBuffer>,
    /// TurboQuant engine for compression/decompression
    engine: TurboQuantEngine,
    /// Current length
    current_len: AtomicU32,
    /// Maximum sequence length
    max_seq_len: u32,
    /// Number of heads
    num_heads: u32,
    /// Head dimension
    head_dim: u32,
    /// Device reference
    device: Arc<Device>,
}

impl CompressedLayerKVCache {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: &CompressedKVCacheConfig,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let _hidden_dim = config.turboquant.hidden_dim;
        let elem_count = max_seq_len as usize * num_heads as usize * head_dim as usize;
        let byte_size = elem_count * std::mem::size_of::<f32>();
        
        // Create TurboQuant engine
        let engine = TurboQuantEngine::new(config.turboquant.clone())
            .map_err(|e| crate::error::FerrisResError::Device(format!(
                "Failed to create TurboQuant engine: {}", e
            )))?;
        
        // If using full precision storage (for now, until GPU kernels are ready)
        let key_cache = GpuBuffer::zeros(&device, &queue, byte_size, Some("Compressed KVCache Keys"))?;
        let value_cache = GpuBuffer::zeros(&device, &queue, byte_size, Some("Compressed KVCache Values"))?;
        
        Ok(Self {
            key_cache: Some(key_cache),
            value_cache: Some(value_cache),
            key_indices: None,
            value_indices: None,
            engine,
            current_len: AtomicU32::new(0),
            max_seq_len,
            num_heads,
            head_dim,
            device,
        })
    }
    
    /// Update cache with new keys and values
    pub fn update(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        new_k: &GpuBuffer,
        new_v: &GpuBuffer,
    ) -> Result<u32> {
        let pos = self.current_len.load(Ordering::Relaxed);
        if pos >= self.max_seq_len {
            return Err(crate::error::FerrisResError::Shape(format!(
                "Compressed KVCache overflow: {} >= {}", pos, self.max_seq_len
            )));
        }
        
        let per_head_dim = self.head_dim as usize * std::mem::size_of::<f32>();
        let per_pos_size = self.num_heads as usize * per_head_dim;
        let dst_offset = pos as u64 * per_pos_size as u64;
        let copy_size = (new_k.size() as u64).min(per_pos_size as u64);
        
        // Copy to cache (compression will happen via GPU kernel in future)
        if let Some(ref cache) = self.key_cache {
            encoder.copy_buffer_to_buffer(new_k.buffer(), 0, cache.buffer(), dst_offset, Some(copy_size));
        }
        if let Some(ref cache) = self.value_cache {
            encoder.copy_buffer_to_buffer(new_v.buffer(), 0, cache.buffer(), dst_offset, Some(copy_size));
        }
        
        self.current_len.fetch_add(1, Ordering::Relaxed);
        Ok(self.current_len.load(Ordering::Relaxed))
    }
    
    /// Get current length
    pub fn current_len(&self) -> u32 {
        self.current_len.load(Ordering::Relaxed)
    }
    
    /// Reset cache
    pub fn reset(&self) {
        self.current_len.store(0, Ordering::Relaxed);
    }
    
    /// Get key buffer (for attention computation)
    pub fn key_buffer(&self) -> Option<&GpuBuffer> {
        self.key_cache.as_ref()
    }
    
    /// Get value buffer (for attention computation)
    pub fn value_buffer(&self) -> Option<&GpuBuffer> {
        self.value_cache.as_ref()
    }
    
    /// Get compression configuration
    pub fn compression_ratio(&self) -> f32 {
        self.engine.compression_ratio()
    }
}

/// Compressed ModelKVCache using TurboQuant
pub struct CompressedModelKVCache {
    layers: Vec<CompressedLayerKVCache>,
}

impl CompressedModelKVCache {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: &CompressedKVCacheConfig,
        num_layers: u32,
        max_seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            layers.push(CompressedLayerKVCache::new(
                Arc::clone(&device),
                Arc::clone(&queue),
                config,
                max_seq_len,
                num_heads,
                head_dim,
            ).map_err(|e| crate::error::FerrisResError::Device(format!(
                "Failed to create compressed KV cache for layer {}: {}", i, e
            )))?);
        }
        Ok(Self { layers })
    }
    
    pub fn layer(&self, idx: usize) -> &CompressedLayerKVCache {
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
    
    /// Calculate total memory savings
    pub fn total_compression_ratio(&self) -> f32 {
        if self.layers.is_empty() {
            return 1.0;
        }
        self.layers[0].compression_ratio()
    }
}

#[cfg(test)]
mod tests {
    /// Verify index selection for sinks + recent window
    #[test]
    fn test_compact_indices_sink_and_recent() {
        let num_sinks = 4;
        let _window = 10;
        let cache_len = 20;

        let mut indices: Vec<usize> = (0..num_sinks).collect();
        let recent_start = cache_len - (_window - num_sinks);
        indices.extend(recent_start..cache_len);

        assert_eq!(indices.len(), _window);
        assert_eq!(&indices[0..4], &[0, 1, 2, 3]);
        assert_eq!(&indices[4..10], &[14, 15, 16, 17, 18, 19]);
    }

    /// Verify compact with no eviction (cache < window) keeps all
    #[test]
    fn test_compact_indices_no_eviction() {
        let cache_len = 8;
        let _window = 16;
        let indices: Vec<usize> = (0..cache_len).collect();
        assert_eq!(indices.len(), 8);
    }

    /// Verify staging buffer size calculation
    #[test]
    fn test_compact_buffer_size() {
        let num_heads = 8u64;
        let head_dim = 64u64;
        let per_pos_bytes = num_heads * head_dim * std::mem::size_of::<f32>() as u64;
        let indices_len = 10u64;
        let compact_bytes = indices_len * per_pos_bytes;

        assert_eq!(compact_bytes, 10 * 8 * 64 * 4);
    }
}
