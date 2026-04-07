use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::compute::kernels::matmul::MatMulOp;
use crate::compute::kernels::rmsnorm::RmsNormOp;
use crate::compute::kernels::softmax::SoftmaxOp;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::model::config::BlockAttnResConfig;
use crate::model::linear::Linear;
use crate::error::Result;

pub struct BlockAttnResLayer {
    layer_number: usize,

    #[allow(dead_code)]
    attn_norm: RmsNormOp,
    #[allow(dead_code)]
    attn_qkv: Linear,
    #[allow(dead_code)]
    attn_out: Linear,

    #[allow(dead_code)]
    ff_norm: RmsNormOp,
    #[allow(dead_code)]
    ff_up: Linear,
    #[allow(dead_code)]
    ff_down: Linear,

    #[allow(dead_code)]
    pseudo_query: GpuBuffer,
    #[allow(dead_code)]
    attn_res_proj: Linear,
    #[allow(dead_code)]
    attn_res_norm: RmsNormOp,

    elementwise: ElementWiseOp,
    #[allow(dead_code)]
    softmax: SoftmaxOp,
    #[allow(dead_code)]
    matmul: MatMulOp,

    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    hidden_dim: usize,
}

impl BlockAttnResLayer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: &BlockAttnResConfig,
        layer_number: usize,
    ) -> Result<Self> {
        tracing::info!(
            "Creating BlockAttnResLayer {}: hidden_dim={} intermediate_dim={}",
            layer_number, config.hidden_dim, config.intermediate_dim
        );

        let attn_norm = RmsNormOp::new(&device)?;
        let ff_norm = RmsNormOp::new(&device)?;
        let attn_res_norm = RmsNormOp::new(&device)?;

        let _head_dim = config.hidden_dim / config.attention_heads;
        let qkv_total = 3 * config.hidden_dim;
        let attn_qkv = Linear::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            config.hidden_dim,
            qkv_total,
            false,
        )?;
        let attn_out = Linear::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            config.hidden_dim,
            config.hidden_dim,
            false,
        )?;

        let ff_up = Linear::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            config.hidden_dim,
            config.intermediate_dim,
            false,
        )?;
        let ff_down = Linear::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            config.intermediate_dim,
            config.hidden_dim,
            false,
        )?;

        let pseudo_query_bytes = config.hidden_dim * std::mem::size_of::<f32>();
        let pseudo_query = GpuBuffer::zeros(&device, pseudo_query_bytes, Some("Pseudo Query"))?;

        let attn_res_proj = Linear::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            config.hidden_dim,
            2 * config.hidden_dim,
            false,
        )?;

        let elementwise = ElementWiseOp::new(&device);
        let softmax = SoftmaxOp::new(&device)?;
        let matmul = MatMulOp::new(&device);

        let hidden_dim = config.hidden_dim;

        tracing::info!("BlockAttnResLayer {} created successfully", layer_number);

        Ok(Self {
            layer_number,
            attn_norm,
            attn_qkv,
            attn_out,
            ff_norm,
            ff_up,
            ff_down,
            pseudo_query,
            attn_res_proj,
            attn_res_norm,
            elementwise,
            softmax,
            matmul,
            device,
            queue,
            hidden_dim,
        })
    }

    pub fn forward_intra_block(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuBuffer,
        output: &GpuBuffer,
        partial_sum: &GpuBuffer,
        batch_size: u32,
    ) -> Result<()> {
        tracing::debug!(
            "BlockAttnResLayer::forward_intra_block layer={} batch={}",
            self.layer_number, batch_size
        );

        let hidden_dim = self.hidden_dim as u32;
        let intermediate_dim = self.ff_up.out_features() as u32;
        let numel = batch_size * hidden_dim;

        let normed = GpuBuffer::new(
            &self.device,
            batch_size as usize * self.hidden_dim * std::mem::size_of::<f32>(),
            Some("intra_normed"),
        )?;

        self.attn_norm.dispatch(
            &self.device,
            encoder,
            hidden_states,
            &normed,
            batch_size,
            hidden_dim,
        )?;

        let qkv_buf = GpuBuffer::new(
            &self.device,
            batch_size as usize * 3 * self.hidden_dim * std::mem::size_of::<f32>(),
            Some("intra_qkv"),
        )?;
        self.attn_qkv.forward(encoder, &normed, &qkv_buf, batch_size)?;

        let attn_intermediate = GpuBuffer::new(
            &self.device,
            batch_size as usize * self.hidden_dim * std::mem::size_of::<f32>(),
            Some("intra_attn_intermediate"),
        )?;
        self.attn_out.forward(encoder, &qkv_buf, &attn_intermediate, batch_size)?;

        self.elementwise.dispatch_add(encoder, hidden_states, &attn_intermediate, output, numel)?;

        let ff_normed = GpuBuffer::new(
            &self.device,
            batch_size as usize * self.hidden_dim * std::mem::size_of::<f32>(),
            Some("intra_ff_normed"),
        )?;
        self.ff_norm.dispatch(
            &self.device,
            encoder,
            output,
            &ff_normed,
            batch_size,
            hidden_dim,
        )?;

        let ff_hidden = GpuBuffer::new(
            &self.device,
            batch_size as usize * self.ff_up.out_features() * std::mem::size_of::<f32>(),
            Some("intra_ff_hidden"),
        )?;
        self.ff_up.forward(encoder, &ff_normed, &ff_hidden, batch_size)?;

        self.elementwise.dispatch_scale(
            encoder,
            &ff_hidden,
            &ff_hidden,
            0.5f32,
            batch_size * intermediate_dim,
        )?;

        let ff_out = GpuBuffer::new(
            &self.device,
            batch_size as usize * self.hidden_dim * std::mem::size_of::<f32>(),
            Some("intra_ff_out"),
        )?;
        self.ff_down.forward(encoder, &ff_hidden, &ff_out, batch_size)?;

        self.elementwise.dispatch_add(encoder, output, &ff_out, output, numel)?;
        self.elementwise.dispatch_add(encoder, partial_sum, output, partial_sum, numel)?;

        tracing::debug!(
            "BlockAttnResLayer::forward_intra_block layer={} complete",
            self.layer_number
        );

        Ok(())
    }

    pub fn forward_inter_block(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        block_reps: &GpuBuffer,
        partial_sum: &GpuBuffer,
        attn_output: &GpuBuffer,
        num_blocks: u32,
        batch_size: u32,
    ) -> Result<()> {
        tracing::debug!(
            "BlockAttnResLayer::forward_inter_block layer={} num_blocks={} batch={}",
            self.layer_number, num_blocks, batch_size
        );

        let hidden_dim = self.hidden_dim as u32;
        let total_entries = num_blocks + 1;

        let normed_keys = GpuBuffer::new(
            &self.device,
            total_entries as usize * self.hidden_dim * std::mem::size_of::<f32>(),
            Some("inter_normed_keys"),
        )?;
        self.attn_res_norm.dispatch(
            &self.device,
            encoder,
            block_reps,
            &normed_keys,
            total_entries,
            hidden_dim,
        )?;

        let scale_factor = 1.0f32 / (self.hidden_dim as f32).sqrt();

        let scores = GpuBuffer::new(
            &self.device,
            batch_size as usize * total_entries as usize * std::mem::size_of::<f32>(),
            Some("inter_scores"),
        )?;

        self.matmul.dispatch(
            encoder,
            partial_sum,
            &normed_keys,
            &scores,
            batch_size,
            hidden_dim,
            total_entries,
        )?;

        self.elementwise.dispatch_scale(
            encoder,
            &scores,
            &scores,
            scale_factor,
            batch_size * total_entries,
        )?;

        let softmax_out = GpuBuffer::new(
            &self.device,
            batch_size as usize * total_entries as usize * std::mem::size_of::<f32>(),
            Some("inter_softmax"),
        )?;
        self.softmax.dispatch(encoder, &scores, &softmax_out, batch_size, total_entries)?;

        self.matmul.dispatch(
            encoder,
            &softmax_out,
            &normed_keys,
            attn_output,
            batch_size,
            total_entries,
            hidden_dim,
        )?;

        tracing::debug!(
            "BlockAttnResLayer::forward_inter_block layer={} complete",
            self.layer_number
        );

        Ok(())
    }
}
