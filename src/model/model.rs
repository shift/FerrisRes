use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::compute::kernels::inter_block::CausalPrefixOp;
use crate::inference::kv_cache::ModelKVCache;
use crate::model::config::BlockAttnResConfig;
use crate::model::block_attn_res::BlockAttnResLayer;
use crate::error::Result;

/// Per-entry causal running sums for the cached (prefill + decode) path.
///
/// `sums[e]` is `[hidden]` and holds `sum_{s < token_count} x_e[s]`, i.e. the
/// raw prefix sum of entry `e` over every processed position. Entry 0 is the
/// initial embedding; entry `b + 1` is block `b`'s layer outputs. The causal
/// mean at the current position is `sums[e] / (token_count * divisor(e))`.
pub struct BlockAttnState {
    sums: Vec<GpuBuffer>,
    num_entries: usize,
    hidden_dim: usize,
    token_count: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl BlockAttnState {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        num_entries: usize,
        hidden_dim: usize,
    ) -> Result<Self> {
        let bytes = hidden_dim * std::mem::size_of::<f32>();
        let sums = (0..num_entries)
            .map(|_| GpuBuffer::zeros(&device, &queue, bytes, Some("Block Attn Sum")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            sums,
            num_entries,
            hidden_dim,
            token_count: 0,
            device,
            queue,
        })
    }

    pub fn token_count(&self) -> u32 {
        self.token_count
    }

    pub fn num_entries(&self) -> usize {
        self.num_entries
    }

    pub fn sums(&self) -> &[GpuBuffer] {
        &self.sums
    }

    /// Reset to an empty prefix (zero sums, zero positions).
    pub fn reset(&mut self) {
        let bytes = self.hidden_dim * std::mem::size_of::<f32>();
        for i in 0..self.sums.len() {
            if let Ok(fresh) = GpuBuffer::zeros(
                &self.device,
                &self.queue,
                bytes,
                Some("Block Attn Sum"),
            ) {
                self.sums[i] = fresh;
            }
        }
        self.token_count = 0;
    }
}

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

        Self::from_layers(device, queue, config, layers)
    }

    /// Build a model from caller-provided layers (used for weight loading and
    /// for tests that need non-default per-layer parameters).
    pub fn from_layers(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: BlockAttnResConfig,
        layers: Vec<BlockAttnResLayer>,
    ) -> Result<Self> {
        let hidden_dim = config.hidden_dim;

        let embedding_bytes = hidden_dim * std::mem::size_of::<f32>();
        let embedding = GpuBuffer::zeros(&device, &queue, embedding_bytes, Some("Token Embedding"))?;

        let elementwise = ElementWiseOp::new(&device, &queue);
        let causal_prefix = CausalPrefixOp::new(&device, &queue)?;

        tracing::info!(
            event = "blockattnresmodel_created_with_layers",
            "BlockAttnResModel created with {} layers",
            layers.len()
        );

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
        self.forward_with_summaries(input, output, batch_size, None)
    }

    /// Full-sequence forward that can also export the per-position causal
    /// block summaries it computed.
    ///
    /// `summaries`, when provided, must be `[(num_blocks + 1), batch, hidden]`.
    /// Slot 0 receives entry 0's causal mean at each position; slot `b + 1`
    /// receives block `b`'s. This is the same tensor the decode path extends.
    pub fn forward_with_summaries(
        &self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        batch_size: u32,
        summaries: Option<&GpuBuffer>,
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
        if let Some(summaries) = summaries {
            encoder.copy_buffer_to_buffer(
                entry_buf.buffer(),
                0,
                summaries.buffer(),
                0,
                Some(numel_bytes as u64),
            );
        }

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
            if let Some(summaries) = summaries {
                encoder.copy_buffer_to_buffer(
                    entry_buf.buffer(),
                    0,
                    summaries.buffer(),
                    ((block_n + 1) * numel_bytes) as u64,
                    Some(numel_bytes as u64),
                );
            }

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

    /// Prefill the cached path: run every layer over `seq_len` positions,
    /// populate each layer's KV cache, apply causal inter-block attention at
    /// every block boundary, and return the per-entry running sums that decode
    /// extends.
    ///
    /// `input`/`output` are `[seq_len, hidden]`. This is the path
    /// [`crate::inference::generator::TokenGenerator`] uses; it is the same
    /// architecture as [`Self::forward`], only with a KV cache.
    pub fn forward_prefill_cached(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        kv_cache: &ModelKVCache,
        seq_len: u32,
    ) -> Result<BlockAttnState> {
        let hidden_dim = self.config.hidden_dim;
        let num_blocks = self.config.num_blocks;
        let block_size = self.config.block_size;
        let num_entries = num_blocks + 1;
        let f32_size = std::mem::size_of::<f32>();
        let numel = seq_len as usize * hidden_dim;
        let numel_bytes = numel * f32_size;
        let row_bytes = hidden_dim * f32_size;

        let mut state = BlockAttnState::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            num_entries,
            hidden_dim,
        )?;

        let block_reps = GpuBuffer::zeros(
            &self.device,
            &self.queue,
            num_entries * numel_bytes,
            Some("Prefill Block Reps"),
        )?;
        let partial_sum = GpuBuffer::zeros(
            &self.device,
            &self.queue,
            numel_bytes,
            Some("Prefill Block Sum"),
        )?;
        let entry_buf = GpuBuffer::new(&self.device, numel_bytes, Some("Prefill Entry Mean"))?;
        let entry_raw = GpuBuffer::new(&self.device, numel_bytes, Some("Prefill Entry Sum"))?;
        let attn_out = GpuBuffer::new(&self.device, numel_bytes, Some("Prefill Inter Block Out"))?;

        // Entry 0: the initial hidden state. Keep both the raw prefix sum (for
        // the decode state) and the causal mean (for the attention keys).
        self.causal_prefix.dispatch_with_normalize(
            encoder, input, &entry_raw, seq_len, hidden_dim as u32, 1.0, false,
        )?;
        encoder.copy_buffer_to_buffer(
            entry_raw.buffer(),
            ((seq_len as usize - 1) * row_bytes) as u64,
            state.sums[0].buffer(),
            0,
            Some(row_bytes as u64),
        );
        self.causal_prefix.dispatch(
            encoder, input, &entry_buf, seq_len, hidden_dim as u32, 1.0,
        )?;
        encoder.copy_buffer_to_buffer(
            entry_buf.buffer(),
            0,
            block_reps.buffer(),
            0,
            Some(numel_bytes as u64),
        );

        let mut carry: Option<GpuBuffer> = None;
        for block_n in 0..num_blocks {
            let layer_start = block_n * block_size;
            let layer_end = layer_start + block_size;

            let mut cur: Option<GpuBuffer> = carry.take();
            for l in layer_start..layer_end {
                let inp = cur.as_ref().unwrap_or(input);
                let out = self.layers[l].forward_prefill(
                    encoder, inp, kv_cache.layer(l), seq_len,
                )?;
                // Every layer in the block contributes to the block summary.
                self.elementwise.dispatch_add(
                    encoder, &partial_sum, &out, &partial_sum, numel as u32,
                )?;
                cur = Some(out);
            }
            let block_out = cur.expect("block must contain at least one layer");

            // Turn the accumulated layer sum into both the raw prefix sum and
            // the causal mean.
            self.causal_prefix.dispatch_with_normalize(
                encoder, &partial_sum, &entry_raw, seq_len, hidden_dim as u32,
                block_size as f32, false,
            )?;
            encoder.copy_buffer_to_buffer(
                entry_raw.buffer(),
                ((seq_len as usize - 1) * row_bytes) as u64,
                state.sums[block_n + 1].buffer(),
                0,
                Some(row_bytes as u64),
            );
            self.causal_prefix.dispatch(
                encoder, &partial_sum, &entry_buf, seq_len, hidden_dim as u32,
                block_size as f32,
            )?;
            encoder.copy_buffer_to_buffer(
                entry_buf.buffer(),
                0,
                block_reps.buffer(),
                ((block_n + 1) * numel_bytes) as u64,
                Some(numel_bytes as u64),
            );

            let num_entries_here = (block_n + 2) as u32;
            self.layers[layer_end - 1].forward_inter_block(
                encoder,
                &block_reps,
                &block_out,
                &attn_out,
                num_entries_here,
                seq_len,
            )?;
            self.elementwise.dispatch_add(encoder, &block_out, &attn_out, &block_out, numel as u32)?;

            encoder.clear_buffer(partial_sum.buffer(), 0, Some(numel_bytes as u64));
            carry = Some(block_out);
        }

        let final_out = carry.expect("model must contain at least one block");
        self.elementwise.dispatch_copy(encoder, &final_out, output, numel as u32)?;
        state.token_count = seq_len;
        Ok(state)
    }

    /// Decode one position through the cached path, extending `state`.
    ///
    /// `input`/`output` are `[hidden]`. The inter-block query is this token's
    /// own hidden state, and every key is the causal mean over positions
    /// `0..=pos`, so decode produces the same value as the corresponding
    /// prefill row.
    pub fn forward_decode_cached(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        kv_cache: &ModelKVCache,
        state: &mut BlockAttnState,
        effective_pos: Option<u32>,
    ) -> Result<()> {
        let hidden_dim = self.config.hidden_dim;
        let num_blocks = self.config.num_blocks;
        let block_size = self.config.block_size;
        let num_entries = num_blocks + 1;
        let f32_size = std::mem::size_of::<f32>();
        let row_bytes = hidden_dim * f32_size;

        state.token_count += 1;
        let count = state.token_count;

        // Extend entry 0 with this position's embedding.
        self.elementwise.dispatch_add(
            encoder, &state.sums[0], input, &state.sums[0], hidden_dim as u32,
        )?;

        let keys = GpuBuffer::new(
            &self.device,
            num_entries * hidden_dim * f32_size,
            Some("Decode Inter Block Keys"),
        )?;
        let mean = GpuBuffer::new(&self.device, row_bytes, Some("Decode Block Mean"))?;
        let attn_out = GpuBuffer::new(&self.device, row_bytes, Some("Decode Inter Block Out"))?;

        let mut carry: Option<GpuBuffer> = None;
        for block_n in 0..num_blocks {
            let layer_start = block_n * block_size;
            let layer_end = layer_start + block_size;

            let mut cur: Option<GpuBuffer> = carry.take();
            for l in layer_start..layer_end {
                let inp = cur.as_ref().unwrap_or(input);
                let out = self.layers[l].forward_decode_token_direct(
                    encoder, inp, kv_cache.layer(l), effective_pos,
                )?;
                // Every layer in the block contributes to the block summary.
                self.elementwise.dispatch_add(
                    encoder,
                    &state.sums[block_n + 1],
                    &out,
                    &state.sums[block_n + 1],
                    hidden_dim as u32,
                )?;
                cur = Some(out);
            }
            let block_out = cur.expect("block must contain at least one layer");

            let last_entry = block_n + 1;
            let lpb = block_size.max(1) as f32;
            for e in 0..=last_entry {
                let divisor = if e == 0 { 1.0 } else { lpb };
                let scale = 1.0 / (count as f32 * divisor);
                self.elementwise.dispatch_scale(
                    encoder, &state.sums[e], &mean, scale, hidden_dim as u32,
                )?;
                encoder.copy_buffer_to_buffer(
                    mean.buffer(),
                    0,
                    keys.buffer(),
                    (e * row_bytes) as u64,
                    Some(row_bytes as u64),
                );
            }

            self.layers[layer_end - 1].forward_inter_block(
                encoder,
                &keys,
                &block_out,
                &attn_out,
                (last_entry + 1) as u32,
                1,
            )?;
            self.elementwise.dispatch_add(
                encoder, &block_out, &attn_out, &block_out, hidden_dim as u32,
            )?;

            carry = Some(block_out);
        }

        let final_out = carry.expect("model must contain at least one block");
        self.elementwise.dispatch_copy(encoder, &final_out, output, hidden_dim as u32)?;
        Ok(())
    }

    pub fn config(&self) -> &BlockAttnResConfig {
        &self.config
    }

    pub fn layers(&self) -> &[BlockAttnResLayer] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [BlockAttnResLayer] {
        &mut self.layers
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
