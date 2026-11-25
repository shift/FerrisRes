use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::error::Result;
use crate::model::Linear;

pub struct LMHead {
    linear: Linear,
    vocab_size: usize,
}

impl LMHead {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        hidden_dim: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let linear = Linear::new(device, queue, hidden_dim, vocab_size, false)?;
        Ok(Self { linear, vocab_size })
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden: &GpuBuffer,
        logits: &GpuBuffer,
        batch_size: u32,
    ) -> Result<()> {
        self.linear.forward(encoder, hidden, logits, batch_size)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Upload LM-head weights in `[hidden_dim, vocab_size]` row-major layout,
    /// which is the layout the GPU matmul reads and the CPU model stores.
    pub fn set_weight(&self, queue: &Queue, data: &[f32]) {
        self.linear.set_weight(queue, data);
    }

    pub fn linear(&self) -> &Linear {
        &self.linear
    }
}
