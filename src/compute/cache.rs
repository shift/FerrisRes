use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::ops::Range;
use wgpu::{Device, Queue, CommandEncoder};

use crate::compute::buffer::GpuBuffer;
use crate::error::Result;
use crate::model::config::BlockAttnResConfig;

pub struct BlockCache {
    buffer: GpuBuffer,
    cache_capacity: usize,
    hidden_dim: usize,
    head: AtomicUsize,
    count: AtomicUsize,
    #[allow(dead_code)]
    device: Arc<Device>,
}

impl BlockCache {
    pub fn new(
        device: Arc<Device>,
        hidden_dim: usize,
        cache_capacity: usize,
    ) -> Result<Self> {
        let total_size = cache_capacity * hidden_dim * std::mem::size_of::<f32>();
        let buffer = GpuBuffer::new(&device, total_size, Some("ferris_block_cache"))?;
        Ok(Self {
            buffer,
            cache_capacity,
            hidden_dim,
            head: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            device,
        })
    }

    /// Reconstruct a `BlockCache` from an existing [`GpuBuffer`].
    ///
    /// Used by [`BorrowedBufferPool::transition_to_inference`] to restore
    /// the KV cache after gradient buffers are returned.  The buffer **must**
    /// be large enough to hold `cache_capacity * hidden_dim * sizeof(f32)` bytes;
    /// the cache state (head / count) is reset to zero.
    pub fn from_buffer(
        device: Arc<Device>,
        buffer: GpuBuffer,
        hidden_dim: usize,
        cache_capacity: usize,
    ) -> Self {
        Self {
            buffer,
            cache_capacity,
            hidden_dim,
            head: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
            device,
        }
    }

    /// Consume the cache and return the underlying [`GpuBuffer`].
    ///
    /// Used by [`BorrowedBufferPool::transition_to_training`] to extract the
    /// backing memory for reuse as gradient scratch space.
    pub fn into_buffer(self) -> GpuBuffer {
        self.buffer
    }

    pub fn push(
        &self,
        encoder: &mut CommandEncoder,
        block_rep: &GpuBuffer,
    ) {
        let head = self.head.load(Ordering::Relaxed);
        let block_bytes = self.hidden_dim * std::mem::size_of::<f32>();
        let src_size = block_rep.size().min(block_bytes);
        let dst_offset = head * block_bytes;

        encoder.copy_buffer_to_buffer(
            block_rep.buffer(),
            0,
            self.buffer.buffer(),
            dst_offset as u64,
            Some(src_size as u64),
        );

        let next_head = (head + 1) % self.cache_capacity;
        self.head.store(next_head, Ordering::Relaxed);
        let prev = self.count.fetch_add(1, Ordering::Relaxed);
        if prev >= self.cache_capacity {
            self.count.store(self.cache_capacity, Ordering::Relaxed);
        }
    }

    pub fn get_all(&self) -> (u32, u32) {
        (
            self.count.load(Ordering::Relaxed) as u32,
            self.hidden_dim as u32,
        )
    }

    pub fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn len(&self) -> usize {
        let c = self.count.load(Ordering::Relaxed);
        c.min(self.cache_capacity)
    }

    pub fn is_full(&self) -> bool {
        self.count.load(Ordering::Relaxed) >= self.cache_capacity
    }

    pub fn clear(&self) {
        self.head.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }

    pub fn capacity(&self) -> usize {
        self.cache_capacity
    }
}

pub struct PipelineStage {
    stage_id: usize,
    layer_start: usize,
    layer_end: usize,
    device: Option<Arc<Device>>,
    queue: Option<Arc<Queue>>,
}

impl PipelineStage {
    pub fn new(stage_id: usize, layer_start: usize, layer_end: usize) -> Self {
        Self {
            stage_id,
            layer_start,
            layer_end,
            device: None,
            queue: None,
        }
    }

    pub fn with_device(self, device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            device: Some(device),
            queue: Some(queue),
            ..self
        }
    }

    pub fn layer_range(&self) -> Range<usize> {
        self.layer_start..self.layer_end
    }

    pub fn stage_id(&self) -> usize {
        self.stage_id
    }

    pub fn device(&self) -> Option<&Arc<Device>> {
        self.device.as_ref()
    }

    pub fn queue(&self) -> Option<&Arc<Queue>> {
        self.queue.as_ref()
    }
}

pub struct PipelineScheduler {
    stages: Vec<PipelineStage>,
    cache: BlockCache,
    transfer_buffers: Vec<GpuBuffer>,
}

impl PipelineScheduler {
    pub fn new(
        device: Arc<Device>,
        config: &BlockAttnResConfig,
        num_stages: usize,
    ) -> Result<Self> {
        let total_layers = config.total_layers();
        let layers_per_stage = (total_layers + num_stages - 1) / num_stages;

        let stages: Vec<PipelineStage> = (0..num_stages)
            .map(|i| {
                let start = i * layers_per_stage;
                let end = (start + layers_per_stage).min(total_layers);
                PipelineStage::new(i, start, end)
            })
            .collect();

        let cache = BlockCache::new(
            Arc::clone(&device),
            config.hidden_dim,
            16,
        )?;

        let transfer_buf_size = config.hidden_dim * std::mem::size_of::<f32>();
        let transfer_buffers: Vec<GpuBuffer> = (0..num_stages)
            .map(|i| {
                GpuBuffer::new(
                    &device,
                    transfer_buf_size,
                    Some(&format!("ferris_transfer_stage_{}", i)),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            stages,
            cache,
            transfer_buffers,
        })
    }

    pub fn stage_for_layer(&self, layer_idx: usize) -> &PipelineStage {
        self.stages
            .iter()
            .find(|s| layer_idx >= s.layer_start && layer_idx < s.layer_end)
            .expect("layer index out of range")
    }

    pub fn cache(&self) -> &BlockCache {
        &self.cache
    }

    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    pub fn prepare_transfer(
        &self,
        encoder: &mut CommandEncoder,
        source_stage: usize,
        dest_stage: usize,
        block_rep: &GpuBuffer,
    ) {
        if source_stage >= self.transfer_buffers.len() || dest_stage >= self.transfer_buffers.len() {
            return;
        }
        let copy_size = block_rep.size().min(self.transfer_buffers[dest_stage].size());
        encoder.copy_buffer_to_buffer(
            block_rep.buffer(),
            0,
            self.transfer_buffers[dest_stage].buffer(),
            0,
            Some(copy_size as u64),
        );
    }

    pub fn transfer_buffer(&self, stage_idx: usize) -> Option<&GpuBuffer> {
        self.transfer_buffers.get(stage_idx)
    }
}
