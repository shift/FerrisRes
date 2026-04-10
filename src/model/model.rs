use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::model::config::BlockAttnResConfig;
use crate::model::block_attn_res::BlockAttnResLayer;
use crate::error::Result;

pub struct BlockAttnResModel {
    config: BlockAttnResConfig,
    layers: Vec<BlockAttnResLayer>,
    _embedding: GpuBuffer,
    block_reps: GpuBuffer,
    partial_sum: GpuBuffer,
    attn_keys_buf: GpuBuffer,
    attn_out: GpuBuffer,
    scratch_a: GpuBuffer,
    elementwise: ElementWiseOp,
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
        let num_blocks = config.num_blocks;

        let embedding_bytes = hidden_dim * std::mem::size_of::<f32>();
        let embedding = GpuBuffer::zeros(&device, &queue, embedding_bytes, Some("Token Embedding"))?;

        let block_reps_bytes = (num_blocks + 1) * hidden_dim * std::mem::size_of::<f32>();
        let block_reps = GpuBuffer::zeros(&device, &queue, block_reps_bytes, Some("Block Representations"))?;

        let partial_sum_bytes = hidden_dim * std::mem::size_of::<f32>();
        let partial_sum = GpuBuffer::new(
            &device,
            partial_sum_bytes,
            Some("Partial Sum (single row)"),
        )?;

        let attn_keys_bytes = (num_blocks + 1) * hidden_dim * std::mem::size_of::<f32>();
        let attn_keys_buf = GpuBuffer::new(
            &device,
            attn_keys_bytes,
            Some("Attention Keys Buffer"),
        )?;

        let attn_out_bytes = hidden_dim * std::mem::size_of::<f32>();
        let attn_out = GpuBuffer::new(
            &device,
            attn_out_bytes,
            Some("Inter-block Attention Output"),
        )?;

        let scratch_a = GpuBuffer::new(
            &device,
            hidden_dim * std::mem::size_of::<f32>(),
            Some("Scratch Buffer A"),
        )?;

        let elementwise = ElementWiseOp::new(&device, &queue);

        tracing::info!("BlockAttnResModel created with {} layers", total_layers);

        Ok(Self {
            config,
            layers,
            _embedding: embedding,
            block_reps,
            partial_sum,
            attn_keys_buf,
            attn_out,
            scratch_a,
            elementwise,
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
        let hidden_dim_bytes = (hidden_dim * std::mem::size_of::<f32>()) as u64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BlockAttnResModel Forward"),
        });

        let partial_sum_size = hidden_dim * std::mem::size_of::<f32>();

        encoder.clear_buffer(self.partial_sum.buffer(), 0, Some(partial_sum_size as u64));

        let hidden = input.buffer();
        let block_reps_buf = self.block_reps.buffer();

        encoder.copy_buffer_to_buffer(
            hidden,
            0,
            block_reps_buf,
            0,
            Some(hidden_dim_bytes),
        );

        let mut current_hidden = hidden;

        for block_n in 0..num_blocks {
            tracing::debug!("Processing block {}", block_n);

            let layer_start = block_n * block_size;
            let layer_end = layer_start + block_size;

            for l in layer_start..layer_end {
                let layer = &self.layers[l];

                if current_hidden == hidden {
                    layer.forward_intra_block(
                        &mut encoder,
                        input,
                        &self.scratch_a,
                        &self.partial_sum,
                        batch_size,
                        crate::training::CheckpointGranularity::None,
                        None,
                    )?;
                    current_hidden = self.scratch_a.buffer();
                } else {
                    layer.forward_intra_block(
                        &mut encoder,
                        &self.scratch_a,
                        &self.scratch_a,
                        &self.partial_sum,
                        batch_size,
                        crate::training::CheckpointGranularity::None,
                        None,
                    )?;
                    current_hidden = self.scratch_a.buffer();
                }
            }

            let block_slot = (block_n + 1) as u64;

            encoder.copy_buffer_to_buffer(
                self.partial_sum.buffer(),
                0,
                block_reps_buf,
                block_slot * hidden_dim_bytes,
                Some(hidden_dim_bytes),
            );

            let entries_to_copy = block_n + 2;
            let src_block_reps = self.block_reps.buffer();

            for i in 0..entries_to_copy {
                let src_offset = (i as u64) * hidden_dim_bytes;
                let dst_offset = (i as u64) * hidden_dim_bytes;
                encoder.copy_buffer_to_buffer(
                    src_block_reps,
                    src_offset,
                    self.attn_keys_buf.buffer(),
                    dst_offset,
                    Some(hidden_dim_bytes),
                );
            }

            let last_layer = &self.layers[layer_end - 1];
            let num_keys = entries_to_copy as u32;

            last_layer.forward_inter_block(
                &mut encoder,
                &self.attn_keys_buf,
                &self.partial_sum,
                &self.attn_out,
                num_keys,
                batch_size,
            )?;

            let numel = batch_size * hidden_dim as u32;
            self.elementwise.dispatch_add(
                &mut encoder,
                &self.scratch_a,
                &self.attn_out,
                &self.scratch_a,
                numel,
            )?;

            current_hidden = self.scratch_a.buffer();
        }

        self.elementwise.dispatch_copy(
            &mut encoder,
            &self.scratch_a,
            output,
            batch_size * hidden_dim as u32,
        )?;

        self.queue.submit(std::iter::once(encoder.finish()));

        tracing::info!("BlockAttnResModel::forward complete");
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
}
