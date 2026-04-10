use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::kernels::moe::{MoEGatingOp, MoEExpertOp};
use crate::compute::GpuBuffer;
use crate::error::Result;
use crate::model::Linear;

pub struct MoELinear {
    gate_proj: Linear,
    expert_up_weights: GpuBuffer,
    expert_down_weights: GpuBuffer,
    moe_gating_op: MoEGatingOp,
    moe_expert_op: MoEExpertOp,
    gate_logits_buf: Option<GpuBuffer>,
    selected_experts_buf: Option<GpuBuffer>,
    expert_weights_buf: Option<GpuBuffer>,
    output_buf: Option<GpuBuffer>,
    scratch_buf: Option<GpuBuffer>,
    hidden_dim: usize,
    intermediate_dim: usize,
    num_experts: usize,
    top_k: usize,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl MoELinear {
    pub fn new(
        device: &Arc<Device>,
        queue: &Arc<wgpu::Queue>,
        hidden_dim: usize,
        intermediate_dim: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        let gate_proj = Linear::new(
            Arc::clone(device),
            Arc::clone(queue),
            hidden_dim,
            num_experts,
            false,
        )?;

        let up_bytes = num_experts * intermediate_dim * hidden_dim * std::mem::size_of::<f32>();
        let down_bytes = num_experts * hidden_dim * intermediate_dim * std::mem::size_of::<f32>();
        let expert_up_weights = GpuBuffer::zeros(device, queue, up_bytes, Some("MoE Expert Up Weights"))?;
        let expert_down_weights = GpuBuffer::zeros(device, queue, down_bytes, Some("MoE Expert Down Weights"))?;

        let moe_gating_op = MoEGatingOp::new(device)?;
        let moe_expert_op = MoEExpertOp::new(device)?;

        Ok(Self {
            gate_proj,
            expert_up_weights,
            expert_down_weights,
            moe_gating_op,
            moe_expert_op,
            gate_logits_buf: None,
            selected_experts_buf: None,
            expert_weights_buf: None,
            output_buf: None,
            scratch_buf: None,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        })
    }

    pub fn forward(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        batch_size: usize,
    ) -> Result<&GpuBuffer> {
        let f32_size = std::mem::size_of::<f32>();
        let u32_size = std::mem::size_of::<u32>();

        let gate_logits_bytes = batch_size * self.num_experts * f32_size;
        let selected_experts_bytes = batch_size * self.top_k * u32_size;
        let expert_weights_bytes = batch_size * self.top_k * f32_size;
        let output_bytes = batch_size * self.top_k * self.hidden_dim * f32_size;
        let scratch_bytes = batch_size * self.top_k * self.intermediate_dim * f32_size;

        if self.gate_logits_buf.as_ref().map_or(true, |b| b.size() < gate_logits_bytes) {
            self.gate_logits_buf = Some(GpuBuffer::zeros(&self.device, &self.queue, gate_logits_bytes, Some("MoE Gate Logits"))?);
        }
        if self.selected_experts_buf.as_ref().map_or(true, |b| b.size() < selected_experts_bytes) {
            self.selected_experts_buf = Some(GpuBuffer::zeros(&self.device, &self.queue, selected_experts_bytes, Some("MoE Selected Experts"))?);
        }
        if self.expert_weights_buf.as_ref().map_or(true, |b| b.size() < expert_weights_bytes) {
            self.expert_weights_buf = Some(GpuBuffer::zeros(&self.device, &self.queue, expert_weights_bytes, Some("MoE Expert Weights"))?);
        }
        if self.output_buf.as_ref().map_or(true, |b| b.size() < output_bytes) {
            self.output_buf = Some(GpuBuffer::zeros(&self.device, &self.queue, output_bytes, Some("MoE Output"))?);
        }
        if self.scratch_buf.as_ref().map_or(true, |b| b.size() < scratch_bytes) {
            self.scratch_buf = Some(GpuBuffer::zeros(&self.device, &self.queue, scratch_bytes, Some("MoE Scratch"))?);
        }

        let gate_logits = self.gate_logits_buf.as_ref().unwrap();
        let selected_experts = self.selected_experts_buf.as_ref().unwrap();
        let expert_weights = self.expert_weights_buf.as_ref().unwrap();
        let output = self.output_buf.as_ref().unwrap();
        let scratch = self.scratch_buf.as_ref().unwrap();

        let bs = batch_size as u32;
        let ne = self.num_experts as u32;
        let tk = self.top_k as u32;
        let hd = self.hidden_dim as u32;
        let id = self.intermediate_dim as u32;

        self.gate_proj.forward(encoder, input, gate_logits, bs)?;

        self.moe_expert_op.dispatch_top_k(
            encoder,
            gate_logits,
            selected_experts,
            expert_weights,
            bs, ne, tk, hd,
        )?;

        self.moe_expert_op.gather_expert_outputs(
            encoder,
            input,
            selected_experts,
            expert_weights,
            &self.expert_up_weights,
            &self.expert_down_weights,
            output,
            scratch,
            bs, ne, tk, hd, id,
        )?;

        Ok(output)
    }

    pub fn expert_up_weights(&self) -> &GpuBuffer {
        &self.expert_up_weights
    }

    pub fn expert_down_weights(&self) -> &GpuBuffer {
        &self.expert_down_weights
    }

    pub fn gate_proj(&self) -> &Linear {
        &self.gate_proj
    }

    pub fn moe_gating_op(&self) -> &MoEGatingOp {
        &self.moe_gating_op
    }

    pub fn moe_expert_op(&self) -> &MoEExpertOp {
        &self.moe_expert_op
    }
}
