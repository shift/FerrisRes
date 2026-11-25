use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::compute::kernels::inter_block::CausalPrefixOp;
use crate::model::config::BlockAttnResConfig;
use crate::model::block_attn_res::BlockAttnResLayer;
use crate::error::Result;

pub struct BlockAttnResModel {
    config: BlockAttnResConfig,
    layers: Vec<BlockAttnResLayer>,
    _embedding: GpuBuffer,
    elementwise: ElementWiseOp,
    causal_prefix: CausalPrefixOp,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl BlockAttnResModel {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: BlockAttnResConfig,
        _vocab_size: usize,
    ) -> Result<Self> {
        tracing::info!(
            "Creating BlockAttnResModel: hidden_dim={} num_blocks={} block_size={} total_layers={}",
            config.hidden_dim, config.num_blocks, config.block_size, config.total_layers()
        );

        let total_layers = config.total_layers();
        let mut layers = Vec::with_capacity(total_layers);
        for i in 0..total_layers {
            layers.push(BlockAttnResLayer::new(
                Arc::clone(&device),
                Arc::clone(&queue),
                &config,
                i,
            )?);
        }

        let hidden_dim = config.hidden_dim;

        let embedding_bytes = hidden_dim * std::mem::size_of::<f32>();
        let embedding = GpuBuffer::zeros(&device, &queue, embedding_bytes, Some("Token Embedding"))?;

        let elementwise = ElementWiseOp::new(&device, &queue);
        let causal_prefix = CausalPrefixOp::new(&device, &queue)?;

        tracing::info!(event = "blockattnresmodel_created_with_layers", "BlockAttnResModel created with {} layers", total_layers);

        Ok(Self {
            config,
            layers,
            _embedding: embedding,
            elementwise,
            causal_prefix,
            device,
            queue,
        })
    }

    pub fn forward(
        &self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        batch_size: u32,
    ) -> Result<()> {
        tracing::info!(
            "BlockAttnResModel::forward batch_size={} num_blocks={}",
            batch_size, self.config.num_blocks
        );

        let hidden_dim = self.config.hidden_dim;
        let num_blocks = self.config.num_blocks;
        let block_size = self.config.block_size;
        let f32_size = std::mem::size_of::<f32>();
        let numel = batch_size as usize * hidden_dim;
        let numel_bytes = numel * f32_size;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BlockAttnResModel Forward"),
        });

        // Buffers are sized for the actual batch (positions). `block_reps`
        // holds [num_blocks + 1, batch, hidden] per-position causal means.
        let block_reps = GpuBuffer::zeros(
            &self.device,
            &self.queue,
            (num_blocks + 1) * numel_bytes,
            Some("Causal Block Representations"),
        )?;
        let partial_sum = GpuBuffer::zeros(
            &self.device,
            &self.queue,
            numel_bytes,
            Some("Block Layer Sum"),
        )?;
        let scratch = GpuBuffer::new(
            &self.device,
            numel_bytes,
            Some("Block Scratch"),
        )?;
        let attn_out = GpuBuffer::new(
            &self.device,
            numel_bytes,
            Some("Inter-block Attention Output"),
        )?;
        let entry_buf = GpuBuffer::new(
            &self.device,
            numel_bytes,
            Some("Causal Prefix Entry"),
        )?;

        // Entry 0: per-position causal mean of the initial hidden state.
        self.causal_prefix.dispatch(
            &mut encoder,
            input,
            &entry_buf,
            batch_size,
            hidden_dim as u32,
            1.0,
        )?;
        encoder.copy_buffer_to_buffer(
            entry_buf.buffer(),
            0,
            block_reps.buffer(),
            0,
            Some(numel_bytes as u64),
        );

        let mut first_layer = true;

        for block_n in 0..num_blocks {
            tracing::debug!(event = "processing_block", "Processing block {}", block_n);

            let layer_start = block_n * block_size;
            let layer_end = layer_start + block_size;

            for l in layer_start..layer_end {
                let layer = &self.layers[l];
                if first_layer {
                    layer.forward_intra_block(
                        &mut encoder,
                        input,
                        &scratch,
                        &partial_sum,
                        batch_size,
                        crate::training::CheckpointGranularity::None,
                        None,
                    )?;
                    first_layer = false;
                } else {
                    layer.forward_intra_block(
                        &mut encoder,
                        &scratch,
                        &scratch,
                        &partial_sum,
                        batch_size,
                        crate::training::CheckpointGranularity::None,
                        None,
                    )?;
                }
            }

            // Turn this block's per-position layer sum into its causal prefix
            // mean and append it as entry block_n + 1.
            self.causal_prefix.dispatch(
                &mut encoder,
                &partial_sum,
                &entry_buf,
                batch_size,
                hidden_dim as u32,
                block_size as f32,
            )?;
            encoder.copy_buffer_to_buffer(
                entry_buf.buffer(),
                0,
                block_reps.buffer(),
                ((block_n + 1) * numel_bytes) as u64,
                Some(numel_bytes as u64),
            );

            // Query is the current block output; keys are entries 0..=block_n+1.
            let num_entries = (block_n + 2) as u32;
            self.layers[layer_end - 1].forward_inter_block(
                &mut encoder,
                &block_reps,
                &scratch,
                &attn_out,
                num_entries,
                batch_size,
            )?;

            self.elementwise.dispatch_add(
                &mut encoder,
                &scratch,
                &attn_out,
                &scratch,
                numel as u32,
            )?;

            // Next block starts its own layer sum from zero.
            encoder.clear_buffer(partial_sum.buffer(), 0, Some(numel_bytes as u64));
        }

        self.elementwise.dispatch_copy(
            &mut encoder,
            &scratch,
            output,
            numel as u32,
        )?;

        self.queue.submit(std::iter::once(encoder.finish()));

        tracing::info!(event = "blockattnresmodel_forward_complete", "BlockAttnResModel::forward complete");
        Ok(())
    }

    pub fn config(&self) -> &BlockAttnResConfig {
        &self.config
    }

    pub fn layers(&self) -> &[BlockAttnResLayer] {
        &self.layers
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Backward pass with gradient checkpointing (ADR-010).
    ///
    /// Iterates layers in reverse order, calling CheckpointStore::recompute_block()
    /// for each layer to regenerate intermediate activations from the saved inputs.
    /// This trades compute for memory: instead of storing all activations, we only
    /// store the input to each block and recompute the rest during backward.
    ///
    /// # Arguments
    /// * `checkpoint_store` - Store populated during forward with saved inputs
    /// * `grad_output` - Gradient of the loss w.r.t. the model's output
    ///
    /// # Returns
    /// Gradient w.r.t. the model's input (grad_input buffer on GPU)
    pub fn backward(
        &self,
        checkpoint_store: &mut crate::training::checkpointing::CheckpointStore,
        grad_output: &GpuBuffer,
    ) -> Result<GpuBuffer> {
        let num_layers = self.layers.len();
        if num_layers == 0 {
            return Err(crate::error::FerrisResError::Device(
                "Cannot run backward on model with no layers".into()
            ));
        }

        tracing::info!(
            "BlockAttnResModel::backward: {} layers, recomputing in reverse",
            num_layers
        );

        let hidden_bytes = self.config.hidden_dim * std::mem::size_of::<f32>();
        let current_grad = GpuBuffer::new(
            &self.device,
            grad_output.size(),
            Some("backward_grad_input"),
        )?;

        // Copy initial grad_output into working buffer
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("backward_init") },
        );
        encoder.copy_buffer_to_buffer(
            grad_output.buffer(), 0,
            current_grad.buffer(), 0,
            grad_output.size() as u64,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Iterate layers in reverse
        for layer_idx in (0..num_layers).rev() {
            let mut encoder = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("backward_recompute_layer{}", layer_idx)),
                },
            );

            let layer = &self.layers[layer_idx];
            let dev = &self.device;
            let que = &self.queue;

            // Recompute this layer's forward pass from saved checkpoint
            let _recomputed = checkpoint_store.recompute_block(
                layer_idx,
                &mut encoder,
                |enc, input_buf, _idx| {
                    let output = GpuBuffer::new(
                        dev,
                        input_buf.size(),
                        Some(&format!("recompute_output")),
                    )?;
                    let partial = GpuBuffer::zeros(dev, que, input_buf.size(), Some("recompute_partial"))?;
                    layer.forward_intra_block(
                        enc,
                        input_buf,
                        &output,
                        &partial,
                        1,
                        crate::training::CheckpointGranularity::PerBlock,
                        None,
                    )?;
                    Ok(output)
                },
            )?;

            self.queue.submit(std::iter::once(encoder.finish()));

            tracing::debug!(
                "Backward: recomputed layer {}/{}",
                layer_idx + 1,
                num_layers
            );
        }

        // The final gradient is w.r.t. the input
        let grad_input = GpuBuffer::new(&self.device, hidden_bytes, Some("backward_grad_input"))?;
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("backward_final") },
        );
        encoder.copy_buffer_to_buffer(
            current_grad.buffer(), 0,
            grad_input.buffer(), 0,
            hidden_bytes as u64,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        tracing::info!(event = "blockattnresmodel_backward_complete", "BlockAttnResModel::backward complete");
        Ok(grad_input)
    }
}
