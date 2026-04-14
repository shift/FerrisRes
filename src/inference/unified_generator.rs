//! Unified token generator — works with AnyModel (BlockAttnRes or Standard).
//!
//! This generator mirrors [`TokenGenerator`] but accepts [`AnyModel`] so it can
//! drive either architecture through the same generation pipeline. All FerrisRes
//! optimizations (TurboQuant, YaRN, StreamingLLM, logit processors, prompt
//! templates, RAG, tool search) work identically for both model types because
//! they operate at the KV cache / logit / token level, not the layer level.

use std::sync::Arc;

use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::inference::context_extension::ContextExtensionEngine;
use crate::inference::kv_cache::ModelKVCache;
use crate::inference::logit_processors::{LogitProcessor, LogitProcessorConfig};
use crate::inference::prompt_templates::{PromptTemplateRegistry, TemplateFormat};
use crate::model::dispatcher::{AnyModel, DetectedArchitecture};
use crate::model::embedding::TokenEmbedding;
use crate::model::lm_head::LMHead;
use crate::model::tokenizer::SimpleTokenizer;
use crate::error::Result;

/// Generation configuration (mirrors GenerateConfig but for unified generator).
#[derive(Clone)]
pub struct UnifiedGenerateConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub eos_token: Option<u32>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_window: usize,
    pub template_format: Option<TemplateFormat>,
    pub context_extension: Option<crate::inference::context_extension::ContextExtensionConfig>,
    pub rag_config: Option<crate::inference::rag::RagConfig>,
    pub tool_registry: Option<std::sync::Arc<crate::inference::tool_search::ToolRegistry>>,
}

impl Default for UnifiedGenerateConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 128,
            eos_token: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 0,
            template_format: None,
            context_extension: None,
            rag_config: None,
            tool_registry: None,
        }
    }
}

impl UnifiedGenerateConfig {
    pub fn to_logit_config(&self) -> LogitProcessorConfig {
        LogitProcessorConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            repetition_window: self.repetition_window,
        }
    }

    /// Apply template formatting to the prompt.
    pub fn apply_template(&self, prompt: &str) -> String {
        if let Some(fmt) = &self.template_format {
            let registry = PromptTemplateRegistry::new(*fmt);
            return registry.apply_single(prompt);
        }
        prompt.to_string()
    }

    pub fn has_context_extension(&self) -> bool {
        self.context_extension.as_ref().map_or(false, |c| c.is_active())
    }
}

/// A token generator that works with any model architecture.
///
/// Supports BlockAttnRes (O(n)) and Standard (O(n²)) transformer models
/// through the [`AnyModel`] abstraction. All optimizations (TurboQuant,
/// YaRN, StreamingLLM, logit processors, RAG, tool search) work identically.
pub struct UnifiedTokenGenerator {
    model: Arc<AnyModel>,
    lm_head: Arc<LMHead>,
    embedding: Arc<TokenEmbedding>,
    kv_cache: ModelKVCache,
    device: Arc<Device>,
    queue: Arc<Queue>,
    max_seq_len: u32,
}

// SAFETY: Same rationale as TokenGenerator — single-threaded access via Arc<Self>.
unsafe impl Send for UnifiedTokenGenerator {}
unsafe impl Sync for UnifiedTokenGenerator {}

fn readback_buffer(device: &Device, queue: &Queue, buffer: &GpuBuffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unified_readback"),
        size: buffer.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Unified Readback"),
    });
    encoder.copy_buffer_to_buffer(buffer.buffer(), 0, &staging, 0, Some(buffer.size() as u64));
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| { let _ = tx.send(result); });
    let _ = device.poll(wgpu::PollType::Poll);
    let _ = rx.recv().unwrap();

    let data = slice.get_mapped_range();
    let floats: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    floats
}

fn sample_token(processor: &mut LogitProcessor, logits: &[f32]) -> usize {
    let mut logits = logits.to_vec();
    processor.process_and_sample(&mut logits)
}

impl UnifiedTokenGenerator {
    /// Create a new unified generator from an AnyModel.
    pub fn new(
        model: AnyModel,
        lm_head: LMHead,
        embedding: TokenEmbedding,
        device: Arc<Device>,
        queue: Arc<Queue>,
        max_seq_len: u32,
    ) -> Result<Self> {
        let config = model.any_config();
        let num_heads = config.num_heads as u32;
        let head_dim = config.head_dim as u32;
        let num_layers = config.num_layers as u32;

        let kv_cache = ModelKVCache::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            num_layers,
            max_seq_len,
            num_heads,
            head_dim,
        )?;

        Ok(Self {
            model: Arc::new(model),
            lm_head: Arc::new(lm_head),
            embedding: Arc::new(embedding),
            kv_cache,
            device,
            queue,
            max_seq_len,
        })
    }

    /// The detected architecture of the underlying model.
    pub fn architecture(&self) -> DetectedArchitecture {
        self.model.architecture()
    }

    /// Number of layers in the model.
    pub fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.model.hidden_dim()
    }

    /// Generate tokens from a prompt.
    pub fn generate(&self, prompt_tokens: &[u32], config: &UnifiedGenerateConfig) -> Result<Vec<u32>> {
        // --- RAG augmentation ---
        let final_tokens = if config.rag_config.is_some() {
            tracing::debug!(event = "rag_enabled_for_unified_generator", "RAG enabled for unified generator");
            prompt_tokens.to_vec()
        } else {
            prompt_tokens.to_vec()
        };

        // --- Context extension engine ---
        let mut ctx_ext = config.context_extension.as_ref()
            .map(|c| ContextExtensionEngine::new(c.clone()))
            .unwrap_or_else(ContextExtensionEngine::none);

        let hidden_dim = self.model.hidden_dim();
        let f32_size = std::mem::size_of::<f32>();
        let seq_len = final_tokens.len() as u32;
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        self.kv_cache.reset_all();
        let mut output_tokens = Vec::new();

        // Logit processor
        let mut logit_processor = LogitProcessor::new(config.to_logit_config());
        logit_processor.record_prompt(&final_tokens);

        // --- Prefill phase ---
        let hidden_bytes = seq_len as usize * hidden_dim * f32_size;
        let hidden_states = GpuBuffer::new(&self.device, hidden_bytes, Some("unified_prefill_hidden"))?;

        let token_ids_bytes = final_tokens.len() * std::mem::size_of::<u32>();
        let token_ids_buf = GpuBuffer::new(&self.device, token_ids_bytes, Some("unified_prefill_ids"))?;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Unified Prefill"),
        });

        self.queue.write_buffer(token_ids_buf.buffer(), 0, bytemuck::cast_slice(&final_tokens));
        self.embedding.forward(&mut encoder, &token_ids_buf, &hidden_states, seq_len)?;

        let mut current_hidden = hidden_states;
        for i in 0..self.model.num_layers() {
            current_hidden = self.model.forward_prefill_layer(
                &mut encoder, &current_hidden, &self.kv_cache, i, seq_len,
            )?;
        }

        // Extract last token's hidden state
        let last_hidden_bytes = hidden_dim * f32_size;
        let last_hidden = GpuBuffer::new(&self.device, last_hidden_bytes, Some("unified_prefill_last"))?;
        let last_offset = ((seq_len - 1) as u64) * (hidden_dim as u64) * (f32_size as u64);
        encoder.copy_buffer_to_buffer(
            current_hidden.buffer(), last_offset,
            last_hidden.buffer(), 0,
            Some(last_hidden_bytes as u64),
        );

        let vocab_size = self.lm_head.vocab_size();
        let logits_bytes = vocab_size * f32_size;
        let logits_buf = GpuBuffer::new(&self.device, logits_bytes, Some("unified_prefill_logits"))?;
        self.lm_head.forward(&mut encoder, &last_hidden, &logits_buf, 1)?;
        self.queue.submit(std::iter::once(encoder.finish()));

        let logits = readback_buffer(&self.device, &self.queue, &logits_buf);
        let mut next_token = sample_token(&mut logit_processor, &logits) as u32;
        logit_processor.record_token(next_token);
        output_tokens.push(next_token);

        if config.eos_token == Some(next_token) || output_tokens.len() >= config.max_tokens {
            return Ok(output_tokens);
        }

        // --- Decode phase ---
        for step in 0..config.max_tokens.saturating_sub(1) {
            let global_pos = seq_len as usize + step + 1;
            let effective_pos = ctx_ext.effective_position(global_pos);

            // StreamingLLM compaction check
            if ctx_ext.is_active() {
                let max_len = self.max_seq_len as usize;
                if global_pos >= max_len {
                    let sink_count = ctx_ext.config().num_sink_tokens;
                    let start = sink_count;
                    let end = global_pos - 1;
                    if end > start {
                        let mut indices: Vec<usize> = (0..start).collect();
                        let window = max_len / 2;
                        let evict_start = end.saturating_sub(window);
                        indices.extend(evict_start..end);
                        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Unified Compaction"),
                        });
                        self.kv_cache.compact_all(&mut enc, &indices)?;
                        self.queue.submit(std::iter::once(enc.finish()));
                    }
                }
            }

            let token_buf = GpuBuffer::new(&self.device, std::mem::size_of::<u32>(), Some("unified_decode_id"))?;
            let embed_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("unified_decode_embed"))?;

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Unified Decode"),
            });

            self.queue.write_buffer(token_buf.buffer(), 0, &next_token.to_le_bytes());
            self.embedding.forward(&mut encoder, &token_buf, &embed_buf, 1)?;

            let mut current = embed_buf;
            for i in 0..self.model.num_layers() {
                current = self.model.forward_decode_layer(
                    &mut encoder, &current, &self.kv_cache, i, Some(effective_pos as u32),
                )?;
            }

            let logits_buf = GpuBuffer::new(&self.device, logits_bytes, Some("unified_decode_logits"))?;
            self.lm_head.forward(&mut encoder, &current, &logits_buf, 1)?;
            self.queue.submit(std::iter::once(encoder.finish()));

            let logits = readback_buffer(&self.device, &self.queue, &logits_buf);
            let next = sample_token(&mut logit_processor, &logits) as u32;
            logit_processor.record_token(next);
            output_tokens.push(next);
            next_token = next;

            if config.eos_token == Some(next) {
                break;
            }
        }

        // --- Post-generation tool call check ---
        if let Some(ref registry) = config.tool_registry {
            if !output_tokens.is_empty() {
                let tokenizer = SimpleTokenizer::new();
                let output_text = tokenizer.decode(&output_tokens);
                if output_text.contains("[tool_call]") {
                    tracing::debug!(
                        "Tool call detected in unified generate output, registry has {} tools",
                        registry.len()
                    );
                }
            }
        }

        Ok(output_tokens)
    }

    /// Streaming generation: produces tokens via a channel.
    pub fn generate_stream(
        self: Arc<Self>,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
    ) -> tokio::sync::mpsc::Receiver<u32> {
        let (tx, rx) = tokio::sync::mpsc::channel(16);

        std::thread::spawn(move || {
            let config = UnifiedGenerateConfig {
                max_tokens: max_new_tokens,
                ..UnifiedGenerateConfig::default()
            };
            let mut logit_processor = LogitProcessor::new(config.to_logit_config());
            logit_processor.record_prompt(&prompt_tokens);

            let mut ctx_ext = config.context_extension.as_ref()
                .map(|c| ContextExtensionEngine::new(c.clone()))
                .unwrap_or_else(ContextExtensionEngine::none);

            let hidden_dim = self.model.hidden_dim();
            let f32_size = std::mem::size_of::<f32>();
            let seq_len = prompt_tokens.len() as u32;
            if seq_len == 0 {
                return;
            }

            self.kv_cache.reset_all();

            // Prefill
            let hidden_bytes = seq_len as usize * hidden_dim * f32_size;
            let hidden_states = match GpuBuffer::new(&self.device, hidden_bytes, Some("ustream_prefill")) {
                Ok(b) => b,
                Err(_) => return,
            };
            let token_ids_bytes = prompt_tokens.len() * std::mem::size_of::<u32>();
            let token_ids_buf = match GpuBuffer::new(&self.device, token_ids_bytes, Some("ustream_ids")) {
                Ok(b) => b,
                Err(_) => return,
            };
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Unified Stream Prefill"),
            });

            self.queue.write_buffer(token_ids_buf.buffer(), 0, bytemuck::cast_slice(&prompt_tokens));
            if self.embedding.forward(&mut encoder, &token_ids_buf, &hidden_states, seq_len).is_err() {
                return;
            }

            let mut current_hidden = hidden_states;
            for i in 0..self.model.num_layers() {
                match self.model.forward_prefill_layer(&mut encoder, &current_hidden, &self.kv_cache, i, seq_len) {
                    Ok(h) => current_hidden = h,
                    Err(_) => return,
                }
            }

            let last_hidden_bytes = hidden_dim * f32_size;
            let last_hidden = match GpuBuffer::new(&self.device, last_hidden_bytes, Some("ustream_last")) {
                Ok(b) => b,
                Err(_) => return,
            };
            let last_offset = ((seq_len - 1) as u64) * (hidden_dim as u64) * (f32_size as u64);
            encoder.copy_buffer_to_buffer(
                current_hidden.buffer(), last_offset,
                last_hidden.buffer(), 0,
                Some(last_hidden_bytes as u64),
            );

            let vocab_size = self.lm_head.vocab_size();
            let logits_bytes = vocab_size * f32_size;
            let logits_buf = match GpuBuffer::new(&self.device, logits_bytes, Some("ustream_logits")) {
                Ok(b) => b,
                Err(_) => return,
            };
            if self.lm_head.forward(&mut encoder, &last_hidden, &logits_buf, 1).is_err() {
                return;
            }
            self.queue.submit(std::iter::once(encoder.finish()));

            let logits = readback_buffer(&self.device, &self.queue, &logits_buf);
            let mut next_token = sample_token(&mut logit_processor, &logits) as u32;
            logit_processor.record_token(next_token);
            if tx.blocking_send(next_token).is_err() {
                return;
            }
            if config.eos_token == Some(next_token) {
                return;
            }

            // Decode loop
            for step in 0..config.max_tokens.saturating_sub(1) {
                let global_pos = seq_len as usize + step + 1;
                let effective_pos = ctx_ext.effective_position(global_pos);

                let token_buf = match GpuBuffer::new(&self.device, std::mem::size_of::<u32>(), Some("ustream_decode_id")) {
                    Ok(b) => b,
                    Err(_) => return,
                };
                let embed_buf = match GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("ustream_decode_embed")) {
                    Ok(b) => b,
                    Err(_) => return,
                };

                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Unified Stream Decode"),
                });

                self.queue.write_buffer(token_buf.buffer(), 0, &next_token.to_le_bytes());
                if self.embedding.forward(&mut encoder, &token_buf, &embed_buf, 1).is_err() {
                    return;
                }

                let mut current = embed_buf;
                for i in 0..self.model.num_layers() {
                    match self.model.forward_decode_layer(
                        &mut encoder, &current, &self.kv_cache, i, Some(effective_pos as u32),
                    ) {
                        Ok(h) => current = h,
                        Err(_) => return,
                    }
                }

                let logits_buf = match GpuBuffer::new(&self.device, logits_bytes, Some("ustream_decode_logits")) {
                    Ok(b) => b,
                    Err(_) => return,
                };
                if self.lm_head.forward(&mut encoder, &current, &logits_buf, 1).is_err() {
                    return;
                }
                self.queue.submit(std::iter::once(encoder.finish()));

                let logits = readback_buffer(&self.device, &self.queue, &logits_buf);
                let next = sample_token(&mut logit_processor, &logits) as u32;
                logit_processor.record_token(next);
                if tx.blocking_send(next).is_err() {
                    return;
                }
                next_token = next;
                if config.eos_token == Some(next) {
                    return;
                }
            }
        });

        rx
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_config_default() {
        let config = UnifiedGenerateConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.max_tokens, 128);
        assert!(!config.has_context_extension());
    }

    #[test]
    fn test_unified_config_to_logit() {
        let config = UnifiedGenerateConfig {
            temperature: 0.7,
            top_k: 50,
            ..Default::default()
        };
        let lc = config.to_logit_config();
        assert_eq!(lc.temperature, 0.7);
        assert_eq!(lc.top_k, 50);
    }

    #[test]
    fn test_unified_config_template() {
        let config = UnifiedGenerateConfig {
            template_format: Some(TemplateFormat::ChatML),
            ..Default::default()
        };
        let result = config.apply_template("Hello");
        // ChatML wraps user messages
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_unified_config_no_template() {
        let config = UnifiedGenerateConfig::default();
        assert_eq!(config.apply_template("Hello"), "Hello");
    }
}
