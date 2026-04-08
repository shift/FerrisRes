use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::compute::cache::BlockCache;
use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::model::config::BlockAttnResConfig;
use crate::model::embedding::TokenEmbedding;
use crate::model::model::BlockAttnResModel;
use crate::model::tokenizer::SimpleTokenizer;
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
                crate::training::CheckpointGranularity::None,
                None,
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

pub struct KVCache {
    key_cache: GpuBuffer,
    value_cache: GpuBuffer,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    current_len: AtomicUsize,
    #[allow(dead_code)]
    device: Arc<Device>,
}

impl KVCache {
    pub fn new(
        device: Arc<Device>,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let layer_size = max_seq_len * num_heads * head_dim * std::mem::size_of::<f32>();
        let key_cache = GpuBuffer::zeros(&device, layer_size, Some("KVCache Keys"))?;
        let value_cache = GpuBuffer::zeros(&device, layer_size, Some("KVCache Values"))?;

        Ok(Self {
            key_cache,
            value_cache,
            max_seq_len,
            num_heads,
            head_dim,
            current_len: AtomicUsize::new(0),
            device,
        })
    }

    pub fn update(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        _layer_idx: usize,
        keys: &GpuBuffer,
        values: &GpuBuffer,
        seq_len: u32,
    ) {
        let pos = seq_len as usize;
        if pos >= self.max_seq_len {
            return;
        }

        let per_head_dim = self.head_dim * std::mem::size_of::<f32>();
        let per_pos_size = self.num_heads * per_head_dim;
        let dst_offset = pos * per_pos_size;

        let copy_size = (keys.size() as u64).min(per_pos_size as u64);

        encoder.copy_buffer_to_buffer(
            keys.buffer(),
            0,
            self.key_cache.buffer(),
            dst_offset as u64,
            Some(copy_size),
        );

        encoder.copy_buffer_to_buffer(
            values.buffer(),
            0,
            self.value_cache.buffer(),
            dst_offset as u64,
            Some(copy_size),
        );
    }

    pub fn get(&self, _layer_idx: usize, seq_len: u32) -> (u64, u64) {
        let per_pos_size = self.num_heads * self.head_dim * std::mem::size_of::<f32>();
        let key_offset = (seq_len as usize) * per_pos_size;
        let value_offset = key_offset;
        (key_offset as u64, value_offset as u64)
    }

    pub fn current_len(&self) -> usize {
        self.current_len.load(Ordering::Relaxed)
    }

    pub fn advance(&self) {
        self.current_len.fetch_add(1, Ordering::Relaxed);
    }

    pub fn key_cache(&self) -> &GpuBuffer {
        &self.key_cache
    }

    pub fn value_cache(&self) -> &GpuBuffer {
        &self.value_cache
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

pub struct Sampler {
    temperature: f32,
    #[allow(dead_code)]
    top_k: Option<usize>,
    #[allow(dead_code)]
    top_p: Option<f32>,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: Option<usize>, top_p: Option<f32>) -> Self {
        Self {
            temperature: if temperature > 0.0 { temperature } else { 1.0 },
            top_k,
            top_p,
        }
    }

    pub fn sample(&self, logits: &[f32]) -> usize {
        if logits.is_empty() {
            return 0;
        }

        if self.temperature != 1.0 {
            let scaled: Vec<f32> = logits.iter().map(|&l| l / self.temperature).collect();
            return greedy_decode(&scaled);
        }

        greedy_decode(logits)
    }
}

fn greedy_decode(logits: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    best_idx
}

pub struct GenerationState {
    pub token_ids: Vec<u32>,
    pub finished: bool,
    pub finish_reason: String,
}

fn readback_buffer(device: &Device, queue: &Queue, buffer: &GpuBuffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generation_readback"),
        size: buffer.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(
        buffer.buffer(),
        0,
        &staging,
        0,
        Some(buffer.size() as u64),
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Poll);
    let _ = rx.recv().unwrap();

    let data = slice.get_mapped_range();
    let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    floats
}

pub struct AutoregressiveGenerator {
    model: BlockAttnResModel,
    embedding: TokenEmbedding,
    tokenizer: SimpleTokenizer,
    kv_cache: KVCache,
    sampler: Sampler,
    #[allow(dead_code)]
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    #[allow(dead_code)]
    elementwise: ElementWiseOp,
}

impl AutoregressiveGenerator {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        model_config: BlockAttnResConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        tracing::info!(
            "Creating AutoregressiveGenerator: hidden_dim={} vocab_size={}",
            model_config.hidden_dim,
            vocab_size,
        );

        let model = BlockAttnResModel::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            model_config.clone(),
            vocab_size,
        )?;

        let embedding = TokenEmbedding::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            vocab_size,
            model_config.hidden_dim,
        )?;

        let tokenizer = SimpleTokenizer::new();

        let num_heads = model_config.attention_heads;
        let head_dim = model_config.hidden_dim / num_heads;
        let max_seq_len = 2048;

        let kv_cache = KVCache::new(
            Arc::clone(&device),
            num_heads,
            head_dim,
            max_seq_len,
        )?;

        let sampler = Sampler::new(1.0, None, None);

        let elementwise = ElementWiseOp::new(&device);

        tracing::info!("AutoregressiveGenerator created successfully");

        Ok(Self {
            model,
            embedding,
            tokenizer,
            kv_cache,
            sampler,
            device,
            queue,
            elementwise,
        })
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerationState> {
        tracing::info!(
            "AutoregressiveGenerator::generate prompt_len={} max_new_tokens={}",
            prompt.len(),
            max_new_tokens,
        );

        let hidden_dim = self.model.config().hidden_dim;
        let eos_id = self.tokenizer.eos_id();
        let mut token_ids = self.tokenizer.encode(prompt);

        if token_ids.is_empty() {
            return Ok(GenerationState {
                token_ids: vec![],
                finished: true,
                finish_reason: "empty_prompt".to_string(),
            });
        }

        let embed_bytes = hidden_dim * std::mem::size_of::<f32>();
        let input_ids_buf = GpuBuffer::new(
            &self.device,
            std::mem::size_of::<u32>(),
            Some("gen_input_ids"),
        )?;
        let embed_out_buf = GpuBuffer::new(
            &self.device,
            embed_bytes,
            Some("gen_embed_out"),
        )?;
        let model_out_buf = GpuBuffer::new(
            &self.device,
            embed_bytes,
            Some("gen_model_out"),
        )?;

        let mut finished = false;
        let mut finish_reason = String::from("max_tokens");

        for _step in 0..max_new_tokens {
            let last_token = *token_ids.last().unwrap();

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gen_step"),
            });

            {
                let mut id_mapped = input_ids_buf.buffer().slice(..).get_mapped_range_mut();
                id_mapped.copy_from_slice(&last_token.to_le_bytes());
                drop(id_mapped);
            }

            self.embedding.forward(&mut encoder, &input_ids_buf, &embed_out_buf, 1)?;

            self.model.forward(&embed_out_buf, &model_out_buf, 1)?;

            self.kv_cache.advance();

            self.queue.submit(std::iter::once(encoder.finish()));

            let output_data = readback_buffer(&self.device, &self.queue, &model_out_buf);

            let logits = &output_data[..hidden_dim.min(output_data.len())];
            let next_token = self.sampler.sample(logits) as u32;

            token_ids.push(next_token);

            if next_token == eos_id {
                finished = true;
                finish_reason = "eos".to_string();
                break;
            }
        }

        tracing::info!(
            "AutoregressiveGenerator::generate complete: {} tokens, finished={}, reason={}",
            token_ids.len(),
            finished,
            finish_reason,
        );

        Ok(GenerationState {
            token_ids,
            finished,
            finish_reason,
        })
    }

    pub fn model(&self) -> &BlockAttnResModel {
        &self.model
    }

    pub fn embedding(&self) -> &TokenEmbedding {
        &self.embedding
    }

    pub fn tokenizer(&self) -> &SimpleTokenizer {
        &self.tokenizer
    }

    pub fn kv_cache(&self) -> &KVCache {
        &self.kv_cache
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}
