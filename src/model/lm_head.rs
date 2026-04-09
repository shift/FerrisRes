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
}
