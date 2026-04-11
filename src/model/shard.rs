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

    pub fn quantize(&self, _target: QuantDtype) -> Result<QuantizedBuffer> {
        Err(FerrisResError::Unsupported(
            "Quantization not yet implemented".into(),
        ))
    }
    
    /// Get compression ratio vs f32
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
