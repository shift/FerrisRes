use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::compute::cache::BlockCache;
use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::model::config::BlockAttnResConfig;
use crate::model::model::BlockAttnResModel;
use crate::error::Result;

#[derive(Debug, Clone)]
pub struct TwoPhaseConfig {
    pub max_batch_inference: u32,
    pub cache_block_reps: bool,
    pub use_online_softmax: bool,
}

impl Default for TwoPhaseConfig {
    fn default() -> Self {
        Self {
            max_batch_inference: 1,
            cache_block_reps: true,
            use_online_softmax: true,
        }
    }
}

pub struct TwoPhaseInference {
    config: TwoPhaseConfig,
    model: BlockAttnResModel,
    block_cache: BlockCache,
    lse_buffer: GpuBuffer,
    elementwise: ElementWiseOp,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl TwoPhaseInference {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        model_config: BlockAttnResConfig,
        inference_config: TwoPhaseConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        tracing::info!(
            "Creating TwoPhaseInference: hidden_dim={} num_blocks={} cache_block_reps={}",
            model_config.hidden_dim,
            model_config.num_blocks,
            inference_config.cache_block_reps,
        );

        let model = BlockAttnResModel::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            model_config.clone(),
            vocab_size,
        )?;

        let cache_capacity = if inference_config.cache_block_reps {
            model_config.num_blocks + 1
        } else {
            1
        };

        let block_cache = BlockCache::new(
            Arc::clone(&device),
            model_config.hidden_dim,
            cache_capacity,
        )?;

        let lse_bytes = model_config.hidden_dim * std::mem::size_of::<f32>();
        let lse_buffer = GpuBuffer::zeros(&device, lse_bytes, Some("LSE Buffer"))?;

        let elementwise = ElementWiseOp::new(&device);

        tracing::info!("TwoPhaseInference created successfully");

        Ok(Self {
            config: inference_config,
            model,
            block_cache,
            lse_buffer,
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
            "TwoPhaseInference::forward batch_size={} cache_block_reps={}",
            batch_size,
            self.config.cache_block_reps,
        );

        let model_config = self.model.config();
        let hidden_dim = model_config.hidden_dim;
        let num_blocks = model_config.num_blocks;
        let hidden_dim_bytes = (hidden_dim * std::mem::size_of::<f32>()) as u64;

        if self.config.cache_block_reps {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TwoPhase Cache Clear"),
            });
            encoder.clear_buffer(self.lse_buffer.buffer(), 0, Some(hidden_dim_bytes));
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        self.model.forward(input, output, batch_size)?;

        if self.config.cache_block_reps {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TwoPhase Cache Reps"),
            });

            let rep_slot_size = hidden_dim * std::mem::size_of::<f32>();
            let cache_buf = self.block_cache.buffer().buffer();

            for block_n in 0..num_blocks {
                let slot_offset = (block_n + 1) as u64 * hidden_dim_bytes;
                let cache_offset = block_n as u64 * rep_slot_size as u64;

                encoder.copy_buffer_to_buffer(
                    output.buffer(),
                    slot_offset,
                    cache_buf,
                    cache_offset,
                    hidden_dim_bytes,
                );
            }

            self.queue.submit(std::iter::once(encoder.finish()));
        }

        tracing::info!("TwoPhaseInference::forward complete");
        Ok(())
    }

    pub fn forward_block(
        &self,
        block_idx: usize,
        input: &GpuBuffer,
        output: &GpuBuffer,
        batch_size: u32,
    ) -> Result<()> {
        tracing::info!(
            "TwoPhaseInference::forward_block block_idx={} batch_size={}",
            block_idx,
            batch_size,
        );

        let model_config = self.model.config();
        let hidden_dim = model_config.hidden_dim;
        let block_size = model_config.block_size;
        let hidden_dim_bytes = (hidden_dim * std::mem::size_of::<f32>()) as u64;

        if block_idx >= model_config.num_blocks {
            return Err(crate::error::FerrisResError::Shape(format!(
                "block_idx {} >= num_blocks {}",
                block_idx, model_config.num_blocks
            )));
        }

        let layer_start = block_idx * block_size;
        let layer_end = layer_start + block_size;

        let scratch = GpuBuffer::new(
            &self.device,
            batch_size as usize * hidden_dim * std::mem::size_of::<f32>(),
            Some("block_scratch"),
        )?;

        let partial_sum = GpuBuffer::zeros(
            &self.device,
            hidden_dim * std::mem::size_of::<f32>(),
            Some("block_partial_sum"),
        )?;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("Forward Block {}", block_idx)),
        });

        let partial_sum_size = hidden_dim * std::mem::size_of::<f32>();
        encoder.clear_buffer(partial_sum.buffer(), 0, Some(partial_sum_size as u64));

        self.elementwise.dispatch_copy(
            &mut encoder,
            input,
            &scratch,
            batch_size * hidden_dim as u32,
        )?;

        for l in layer_start..layer_end {
            let layer = &self.model.layers()[l];

            layer.forward_intra_block(
                &mut encoder,
                &scratch,
                &scratch,
                &partial_sum,
                batch_size,
            )?;
        }

        if self.config.cache_block_reps {
            let slot = block_idx as u64 * hidden_dim_bytes;
            encoder.copy_buffer_to_buffer(
                partial_sum.buffer(),
                0,
                self.block_cache.buffer().buffer(),
                slot,
                hidden_dim_bytes,
            );
        }

        self.elementwise.dispatch_copy(
            &mut encoder,
            &scratch,
            output,
            batch_size * hidden_dim as u32,
        )?;

        self.queue.submit(std::iter::once(encoder.finish()));

        tracing::info!(
            "TwoPhaseInference::forward_block {} complete",
            block_idx,
        );
        Ok(())
    }

    pub fn config(&self) -> &TwoPhaseConfig {
        &self.config
    }

    pub fn model(&self) -> &BlockAttnResModel {
        &self.model
    }

    pub fn block_cache(&self) -> &BlockCache {
        &self.block_cache
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

pub struct AutoregressiveGenerator {
    inference: TwoPhaseInference,
    #[allow(dead_code)]
    kv_cache: GpuBuffer,
    generated_tokens: usize,
    max_tokens: usize,
}

impl AutoregressiveGenerator {
    pub fn new(inference: TwoPhaseInference, max_tokens: usize) -> Self {
        let model_config = inference.model().config();
        let hidden_dim = model_config.hidden_dim;
        let cache_size = max_tokens * hidden_dim * std::mem::size_of::<f32>();

        let kv_cache = GpuBuffer::new(
            inference.device(),
            cache_size,
            Some("Autoregressive KV Cache"),
        ).expect("failed to allocate KV cache");

        Self {
            inference,
            kv_cache,
            generated_tokens: 0,
            max_tokens,
        }
    }

    pub fn generate(
        &mut self,
        _prompt: &GpuBuffer,
        max_new_tokens: usize,
    ) -> Result<GpuBuffer> {
        tracing::info!(
            "AutoregressiveGenerator::generate max_new_tokens={} (stub)",
            max_new_tokens,
        );

        let model_config = self.inference.model().config();
        let hidden_dim = model_config.hidden_dim;
        let output_bytes = hidden_dim * std::mem::size_of::<f32>();

        let output = GpuBuffer::zeros(
            self.inference.device(),
            output_bytes,
            Some("Generator Output (stub)"),
        )?;

        self.generated_tokens += max_new_tokens;

        tracing::warn!("AutoregressiveGenerator::generate is a stub - returns zeroed buffer");

        Ok(output)
    }

    pub fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn inference(&self) -> &TwoPhaseInference {
        &self.inference
    }
}
