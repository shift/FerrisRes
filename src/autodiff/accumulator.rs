use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::autodiff::graph::NodeId;
use crate::error::Result;

pub struct GradientAccumulator {
    accumulators: HashMap<NodeId, GpuBuffer>,
    counts: HashMap<NodeId, u32>,
    elementwise: ElementWiseOp,
    device: Arc<Device>,
    queue: Arc<Queue>,
    /// How many micro-batches have been accumulated since the last optimizer step.
    pub micro_batch_count: u32,
    /// How many micro-batches to accumulate before an optimizer step.
    pub accumulation_steps: u32,
}

impl GradientAccumulator {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        tracing::info!("Creating GradientAccumulator");
        let elementwise = ElementWiseOp::new(&device, &queue);
        Self {
            accumulators: HashMap::new(),
            counts: HashMap::new(),
            elementwise,
            device,
            queue,
            micro_batch_count: 0,
            accumulation_steps: 1,
        }
    }

    /// Create a GradientAccumulator with a specific number of accumulation steps.
    pub fn with_accumulation_steps(device: Arc<Device>, queue: Arc<Queue>, accumulation_steps: u32) -> Self {
        assert!(accumulation_steps >= 1, "accumulation_steps must be >= 1");
        tracing::info!("Creating GradientAccumulator with {} accumulation steps", accumulation_steps);
        let elementwise = ElementWiseOp::new(&device, &queue);
        Self {
            accumulators: HashMap::new(),
            counts: HashMap::new(),
            elementwise,
            device,
            queue,
            micro_batch_count: 0,
            accumulation_steps,
        }
    }

    /// Returns true when enough micro-batches have been accumulated to trigger an optimizer step.
    /// Specifically returns true when `micro_batch_count % accumulation_steps == 0` and
    /// `micro_batch_count > 0`.
    pub fn should_step(&self) -> bool {
        self.micro_batch_count > 0 && self.micro_batch_count % self.accumulation_steps == 0
    }

    /// Divides all accumulated gradient buffers by `accumulation_steps` in-place using GPU dispatch.
    /// Must be called before the optimizer step to correctly average gradients across micro-batches.
    pub fn normalize(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        if self.accumulation_steps <= 1 {
            return Ok(());
        }
        let scale = 1.0 / self.accumulation_steps as f32;
        for (id, accum) in &self.accumulators {
            let numel = accum.size() as u32 / std::mem::size_of::<f32>() as u32;
            self.elementwise.dispatch_scale(encoder, accum, accum, scale, numel)?;
            tracing::debug!(
                "GradientAccumulator::normalize: scaled {:?} by {:.6} (accumulation_steps={})",
                id, scale, self.accumulation_steps
            );
        }
        Ok(())
    }

    pub fn register(&mut self, id: NodeId, size: usize) -> Result<()> {
        let buf = GpuBuffer::zeros(&self.device, &self.queue, size, Some(&format!("grad_accum_{:?}", id)))?;
        self.accumulators.insert(id, buf);
        self.counts.insert(id, 0);
        tracing::debug!("GradientAccumulator: registered {:?} size={}", id, size);
        Ok(())
    }

    pub fn accumulate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        id: NodeId,
        grad: &GpuBuffer,
    ) -> Result<()> {
        let accum = self.accumulators.get(&id).ok_or_else(|| {
            crate::error::FerrisResError::Device(format!("gradient accumulator not registered for {:?}", id))
        })?;

        let numel = grad.size() as u32 / std::mem::size_of::<f32>() as u32;
        self.elementwise.dispatch_add(encoder, accum, grad, accum, numel)?;

        let count = self.counts.entry(id).or_insert(0);
        *count += 1;

        tracing::debug!("GradientAccumulator: accumulated {:?} count={}", id, count);
        Ok(())
    }

    /// Increment the micro-batch counter. Call once per micro-batch (after all
    /// parameter gradients have been accumulated for that micro-batch).
    pub fn increment_micro_batch(&mut self) {
        self.micro_batch_count += 1;
        tracing::debug!(
            "GradientAccumulator: micro_batch_count={} accumulation_steps={}",
            self.micro_batch_count,
            self.accumulation_steps
        );
    }

    pub fn averaged(&self, id: NodeId) -> Option<(&GpuBuffer, u32)> {
        let accum = self.accumulators.get(&id)?;
        let count = *self.counts.get(&id)?;
        Some((accum, count))
    }

    pub fn reset(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        for (id, accum) in &self.accumulators {
            let numel = accum.size() as u32 / std::mem::size_of::<f32>() as u32;
            let zero = GpuBuffer::zeros(&self.device, &self.queue, accum.size(), Some("grad_accum_zero"))?;
            self.elementwise.dispatch_copy(encoder, &zero, accum, numel)?;
            self.counts.insert(*id, 0);
        }
        self.micro_batch_count = 0;
        tracing::debug!("GradientAccumulator: reset all accumulators");
        Ok(())
    }

    pub fn get(&self, id: NodeId) -> Option<&GpuBuffer> {
        self.accumulators.get(&id)
    }
}
