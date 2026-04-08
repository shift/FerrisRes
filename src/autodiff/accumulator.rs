use std::collections::HashMap;
use std::sync::Arc;
use wgpu::Device;
use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::autodiff::graph::NodeId;
use crate::error::Result;

pub struct GradientAccumulator {
    accumulators: HashMap<NodeId, GpuBuffer>,
    counts: HashMap<NodeId, u32>,
    elementwise: ElementWiseOp,
    device: Arc<Device>,
}

impl GradientAccumulator {
    pub fn new(device: Arc<Device>) -> Self {
        tracing::info!("Creating GradientAccumulator");
        let elementwise = ElementWiseOp::new(&device);
        Self {
            accumulators: HashMap::new(),
            counts: HashMap::new(),
            elementwise,
            device,
        }
    }

    pub fn register(&mut self, id: NodeId, size: usize) -> Result<()> {
        let buf = GpuBuffer::zeros(&self.device, size, Some(&format!("grad_accum_{:?}", id)))?;
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

    pub fn averaged(&self, id: NodeId) -> Option<(&GpuBuffer, u32)> {
        let accum = self.accumulators.get(&id)?;
        let count = *self.counts.get(&id)?;
        Some((accum, count))
    }

    pub fn reset(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        for (id, accum) in &self.accumulators {
            let numel = accum.size() as u32 / std::mem::size_of::<f32>() as u32;
            let zero = GpuBuffer::zeros(&self.device, accum.size(), Some("grad_accum_zero"))?;
            self.elementwise.dispatch_copy(encoder, &zero, accum, numel)?;
            self.counts.insert(*id, 0);
        }
        tracing::debug!("GradientAccumulator: reset all accumulators");
        Ok(())
    }

    pub fn get(&self, id: NodeId) -> Option<&GpuBuffer> {
        self.accumulators.get(&id)
    }
}
