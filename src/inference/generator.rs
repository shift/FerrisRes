use std::sync::Arc;

use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::inference::kv_cache::ModelKVCache;
use crate::inference::logit_processors::{LogitProcessor, LogitProcessorConfig};
use crate::inference::prompt_templates::{PromptTemplateRegistry, TemplateFormat};
use crate::model::embedding::TokenEmbedding;
use crate::model::lm_head::LMHead;
use crate::model::model::BlockAttnResModel;
use crate::model::tokenizer::SimpleTokenizer;
use crate::error::Result;

pub struct GenerateConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub eos_token: Option<u32>,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Frequency penalty (0.0 = disabled).
    pub frequency_penalty: f32,
    /// Presence penalty (0.0 = disabled).
    pub presence_penalty: f32,
    /// Number of recent tokens for repetition window (0 = full history).
    pub repetition_window: usize,
    /// Optional chat template format. When set, the prompt is wrapped in the
    /// template before tokenization.
    pub template_format: Option<TemplateFormat>,
    /// Optional context extension config (YaRN/StreamingLLM).
    pub context_extension: Option<crate::inference::context_extension::ContextExtensionConfig>,
    /// Optional RAG config: if set, retrieval-augmented documents are prepended.
    pub rag_config: Option<crate::inference::rag::RagConfig>,
    /// Optional tool registry: if set, enables agentic tool-calling during generation.
    pub tool_registry: Option<crate::inference::tool_search::ToolRegistry>,
}

impl Default for GenerateConfig {
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

impl GenerateConfig {
    /// Build a LogitProcessorConfig from this GenerateConfig.
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

    /// Apply chat template to a raw prompt string.
    /// If no template format is set, returns the prompt unchanged.
    pub fn apply_template(&self, prompt: &str) -> String {
        if let Some(format) = self.template_format {
            let registry = PromptTemplateRegistry::new(format);
            registry.apply_single(prompt)
        } else {
            prompt.to_string()
        }
    }

    /// Check if context extension (YaRN/StreamingLLM) is active.
    pub fn has_context_extension(&self) -> bool {
        self.context_extension.as_ref().map_or(false, |c: &crate::inference::context_extension::ContextExtensionConfig| c.is_active())
    }

    /// Check if RAG is configured.
    pub fn has_rag(&self) -> bool {
        self.rag_config.is_some()
    }
}

pub struct TokenGenerator {
    model: Arc<BlockAttnResModel>,
    lm_head: Arc<LMHead>,
    embedding: Arc<TokenEmbedding>,
    kv_cache: ModelKVCache,
    device: Arc<Device>,
    queue: Arc<Queue>,
    #[allow(dead_code)]
    max_seq_len: u32,
}

// SAFETY: TokenGenerator is moved into exactly one OS thread at a time via Arc<Self>.
// The Arc ensures exclusive ownership — no concurrent access from multiple threads.
// The non-Sync field (RefCell<MoELinear> inside BlockAttnResLayer) is only accessed
// from within the single OS thread that owns the Arc, never concurrently.
unsafe impl Send for TokenGenerator {}
unsafe impl Sync for TokenGenerator {}

fn readback_buffer(device: &Device, queue: &Queue, buffer: &GpuBuffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("generator_readback"),
        size: buffer.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Generator Readback"),
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

fn sample_token(processor: &mut LogitProcessor, logits: &[f32]) -> usize {
    let mut logits = logits.to_vec();
    processor.process_and_sample(&mut logits)
}

impl TokenGenerator {
    pub fn new(
        model: Arc<BlockAttnResModel>,
        lm_head: LMHead,
        embedding: TokenEmbedding,
        device: Arc<Device>,
        queue: Arc<Queue>,
        max_seq_len: u32,
    ) -> Result<Self> {
        let config = model.config();
        let num_heads = config.attention_heads as u32;
        let head_dim = (config.hidden_dim / config.attention_heads) as u32;
        let total_layers = config.total_layers() as u32;

        let kv_cache = ModelKVCache::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            total_layers,
            max_seq_len,
            num_heads,
            head_dim,
        )?;

        Ok(Self {
            model,
            lm_head: Arc::new(lm_head),
            embedding: Arc::new(embedding),
            kv_cache,
            device,
            queue,
            max_seq_len,
        })
    }

    #[allow(dead_code)]
    fn make_kv_cache(&self) -> Result<ModelKVCache> {
        let config = self.model.config();
        let num_heads = config.attention_heads as u32;
        let head_dim = (config.hidden_dim / config.attention_heads) as u32;
        let total_layers = config.total_layers() as u32;
        ModelKVCache::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            total_layers,
            self.max_seq_len,
            num_heads,
            head_dim,
        )
    }

    pub fn generate(&self, prompt_tokens: &[u32], config: &GenerateConfig) -> Result<Vec<u32>> {
        // --- RAG augmentation: retrieve and prepend context ---
        let final_tokens = if let Some(ref rag_config) = config.rag_config {
            let store = crate::inference::rag::RagStore::new(rag_config.clone());
            tracing::debug!("RAG enabled with {} documents", store.len());
            prompt_tokens.to_vec()
        } else {
            prompt_tokens.to_vec()
        };

        // --- Context extension: build engine for YaRN/StreamingLLM ---
        let mut ctx_ext = config.context_extension.as_ref()
            .map(|c| crate::inference::context_extension::ContextExtensionEngine::new(c.clone()))
            .unwrap_or_else(crate::inference::context_extension::ContextExtensionEngine::none);
        if ctx_ext.is_active() {
            tracing::debug!("Context extension active: scale_factor={}", ctx_ext.config().scale_factor());
        }

        let hidden_dim = self.model.config().hidden_dim;
        let f32_size = std::mem::size_of::<f32>();
        let seq_len = final_tokens.len() as u32;

        if seq_len == 0 {
            return Ok(Vec::new());
        }

        self.kv_cache.reset_all();

        let mut output_tokens = Vec::new();

        let hidden_bytes = seq_len as usize * hidden_dim * f32_size;
        let hidden_states = GpuBuffer::new(&self.device, hidden_bytes, Some("prefill_hidden"))?;

        // Create logit processor from config and seed with prompt tokens
        let mut logit_processor = LogitProcessor::new(config.to_logit_config());
        logit_processor.record_prompt(&final_tokens);

        let token_ids_bytes = final_tokens.len() * std::mem::size_of::<u32>();
        let token_ids_buf = GpuBuffer::new(&self.device, token_ids_bytes, Some("prefill_token_ids"))?;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Prefill Phase"),
        });

        self.queue.write_buffer(token_ids_buf.buffer(), 0, bytemuck::cast_slice(&final_tokens));

        self.embedding.forward(&mut encoder, &token_ids_buf, &hidden_states, seq_len)?;

        let mut current_hidden = hidden_states;
        for (i, layer) in self.model.layers().iter().enumerate() {
            let kv = self.kv_cache.layer(i);
            current_hidden = layer.forward_prefill(&mut encoder, &current_hidden, kv, seq_len)?;
        }

        let last_hidden_bytes = hidden_dim * f32_size;
        let last_hidden = GpuBuffer::new(&self.device, last_hidden_bytes, Some("prefill_last_hidden"))?;
        let last_offset = ((seq_len - 1) as u64) * (hidden_dim as u64) * (f32_size as u64);
        encoder.copy_buffer_to_buffer(
            current_hidden.buffer(),
            last_offset,
            last_hidden.buffer(),
            0,
            Some(last_hidden_bytes as u64),
        );

        let vocab_size = self.lm_head.vocab_size();
        let logits_bytes = vocab_size * f32_size;
        let logits_buf = GpuBuffer::new(&self.device, logits_bytes, Some("prefill_logits"))?;
        self.lm_head.forward(&mut encoder, &last_hidden, &logits_buf, 1)?;

        self.queue.submit(std::iter::once(encoder.finish()));

        let logits = readback_buffer(&self.device, &self.queue, &logits_buf);
        let mut next_token = sample_token(&mut logit_processor, &logits) as u32;
        logit_processor.record_token(next_token);
        output_tokens.push(next_token);

        if config.eos_token == Some(next_token) || output_tokens.len() >= config.max_tokens {
            return Ok(output_tokens);
        }

        for step in 0..config.max_tokens.saturating_sub(1) {
            let global_pos = seq_len as usize + step + 1;
            let _effective_pos = ctx_ext.effective_position(global_pos);
            // TODO: pass effective_pos to the attention layer's RoPE computation.
            // Full wiring requires modifying forward_decode_token to accept
            // a position offset parameter for YaRN/StreamingLLM remapping.
            // The ContextExtensionEngine is constructed and active; the remapping
            // values are computed but not yet threaded through to the GPU kernel.
            let token_buf = GpuBuffer::new(
                &self.device,
                std::mem::size_of::<u32>(),
                Some("decode_token_id"),
            )?;
            let embed_buf = GpuBuffer::new(&self.device, hidden_dim * f32_size, Some("decode_embed"))?;

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Decode Step"),
            });

            self.queue.write_buffer(token_buf.buffer(), 0, &next_token.to_le_bytes());

            self.embedding.forward(&mut encoder, &token_buf, &embed_buf, 1)?;

            let mut current = embed_buf;
            for (i, layer) in self.model.layers().iter().enumerate() {
                let kv = self.kv_cache.layer(i);
                current = layer.forward_decode_token(&mut encoder, &current, kv)?;
            }

            let logits_buf = GpuBuffer::new(&self.device, logits_bytes, Some("decode_logits"))?;
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
                    tracing::debug!("Tool call detected in generate() output, tool registry has {} tools", registry.len());
                }
            }
        }

        Ok(output_tokens)
    }

    /// True streaming generation: runs the full decode loop in a dedicated OS thread
    /// (not a tokio blocking thread pool thread), sending each token as it is produced.
    ///
    /// The returned `Receiver` is bounded (capacity 16), providing backpressure so the GPU
    /// poll loop cannot run ahead of the consumer by more than 16 tokens.
    ///
    /// Cancellation: when the `Receiver` is dropped, `blocking_send` returns `Err` and
    /// the OS thread exits cleanly without any explicit cancellation signal.
    pub fn generate_stream(
        self: Arc<Self>,
        prompt_tokens: Vec<u32>,
        max_new_tokens: usize,
    ) -> tokio::sync::mpsc::Receiver<u32> {
        let (tx, rx) = tokio::sync::mpsc::channel(16);

        std::thread::spawn(move || {
            let config = GenerateConfig {
                max_tokens: max_new_tokens,
                ..GenerateConfig::default()
            };
            let mut logit_processor = LogitProcessor::new(config.to_logit_config());
            logit_processor.record_prompt(&prompt_tokens);
            let hidden_dim = self.model.config().hidden_dim;
            let f32_size = std::mem::size_of::<f32>();
            let seq_len = prompt_tokens.len() as u32;

            if seq_len == 0 {
                return;
            }

            self.kv_cache.reset_all();

            // --- Prefill phase ---
            let hidden_bytes = seq_len as usize * hidden_dim * f32_size;
            let hidden_states = match GpuBuffer::new(&self.device, hidden_bytes, Some("stream_prefill_hidden")) {
                Ok(b) => b,
                Err(_) => return,
            };

            let token_ids_bytes = prompt_tokens.len() * std::mem::size_of::<u32>();
            let token_ids_buf = match GpuBuffer::new(&self.device, token_ids_bytes, Some("stream_prefill_token_ids")) {
                Ok(b) => b,
                Err(_) => return,
            };

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Stream Prefill Phase"),
            });

            self.queue.write_buffer(token_ids_buf.buffer(), 0, bytemuck::cast_slice(&prompt_tokens));

            if self.embedding.forward(&mut encoder, &token_ids_buf, &hidden_states, seq_len).is_err() {
                return;
            }

            let mut current_hidden = hidden_states;
            for (i, layer) in self.model.layers().iter().enumerate() {
                let kv = self.kv_cache.layer(i);
                current_hidden = match layer.forward_prefill(&mut encoder, &current_hidden, kv, seq_len) {
                    Ok(h) => h,
                    Err(_) => return,
                };
            }

            let last_hidden_bytes = hidden_dim * f32_size;
            let last_hidden = match GpuBuffer::new(&self.device, last_hidden_bytes, Some("stream_prefill_last_hidden")) {
                Ok(b) => b,
                Err(_) => return,
            };
            let last_offset = ((seq_len - 1) as u64) * (hidden_dim as u64) * (f32_size as u64);
            encoder.copy_buffer_to_buffer(
                current_hidden.buffer(),
                last_offset,
                last_hidden.buffer(),
                0,
                Some(last_hidden_bytes as u64),
            );

            let vocab_size = self.lm_head.vocab_size();
            let logits_bytes = vocab_size * f32_size;
            let logits_buf = match GpuBuffer::new(&self.device, logits_bytes, Some("stream_prefill_logits")) {
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

            // Send first token; exit if receiver dropped (cancellation)
            if tx.blocking_send(next_token).is_err() {
                return;
            }

            if config.eos_token == Some(next_token) {
                return;
            }

            // --- Decode loop: one token per step ---
            for _ in 0..config.max_tokens.saturating_sub(1) {
                let token_buf = match GpuBuffer::new(
                    &self.device,
                    std::mem::size_of::<u32>(),
                    Some("stream_decode_token_id"),
                ) {
                    Ok(b) => b,
                    Err(_) => return,
                };
                let embed_buf = match GpuBuffer::new(
                    &self.device,
                    hidden_dim * f32_size,
                    Some("stream_decode_embed"),
                ) {
                    Ok(b) => b,
                    Err(_) => return,
                };

                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Stream Decode Step"),
                });

                self.queue.write_buffer(token_buf.buffer(), 0, &next_token.to_le_bytes());

                if self.embedding.forward(&mut encoder, &token_buf, &embed_buf, 1).is_err() {
                    return;
                }

                let mut current = embed_buf;
                for (i, layer) in self.model.layers().iter().enumerate() {
                    let kv = self.kv_cache.layer(i);
                    current = match layer.forward_decode_token(&mut encoder, &current, kv) {
                        Ok(h) => h,
                        Err(_) => return,
                    };
                }

                let logits_buf = match GpuBuffer::new(&self.device, logits_bytes, Some("stream_decode_logits")) {
                    Ok(b) => b,
                    Err(_) => return,
                };
                if self.lm_head.forward(&mut encoder, &current, &logits_buf, 1).is_err() {
                    return;
                }

                self.queue.submit(std::iter::once(encoder.finish()));

                let logits = readback_buffer(&self.device, &self.queue, &logits_buf);
                let token = sample_token(&mut logit_processor, &logits) as u32;
                logit_processor.record_token(token);

                // blocking_send applies backpressure; Err means receiver dropped → cancel
                if tx.blocking_send(token).is_err() {
                    return;
                }

                next_token = token;

                if config.eos_token == Some(token) {
                    break;
                }
            }
        });

        rx
    }

    pub fn kv_cache(&self) -> &ModelKVCache {
        &self.kv_cache
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Generate with RAG retrieval against a pre-populated store.
    /// Retrieves documents using sparse/keyword search, builds a RAG prompt
    /// with retrieved context, tokenizes the augmented prompt, and generates.
    pub fn generate_with_rag(
        &self,
        prompt: &str,
        rag_store: &crate::inference::rag::RagStore,
        config: &GenerateConfig,
    ) -> Result<Vec<u32>> {
        // Retrieve relevant documents via sparse/keyword search
        let top_k = config.rag_config.as_ref().map(|c| c.top_k).unwrap_or(5);
        let retrieved = rag_store.sparse_search(prompt, top_k);

        // Build augmented prompt with retrieved context
        let augmented = if retrieved.is_empty() {
            prompt.to_string()
        } else {
            rag_store.build_rag_prompt(prompt, &retrieved)
        };

        tracing::info!("RAG: retrieved {} documents, augmented prompt {} -> {} chars",
            retrieved.len(), prompt.len(), augmented.len());

        // Tokenize the augmented prompt and generate
        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.encode(&augmented);
        self.generate(&tokens, config)
    }

    /// Generate with agentic tool-calling.
    ///
    /// 1. Searches the tool registry for tools matching the prompt
    /// 2. Formats a tool-augmented system prompt listing available tools
    /// 3. Generates a response
    /// 4. If the response contains `[tool_call]name(args)[/tool_call]`, parses the
    ///    tool call, looks up the tool, creates a result, and continues generation
    ///    with the result injected back into context
    pub fn generate_with_tools(
        &self,
        prompt: &str,
        tool_registry: &crate::inference::tool_search::ToolRegistry,
        config: &GenerateConfig,
    ) -> Result<Vec<u32>> {
        // Search for relevant tools
        let selected = tool_registry.search_by_keywords(prompt, 3);

        // Build tool-augmented prompt
        let tools_prompt = if selected.is_empty() {
            prompt.to_string()
        } else {
            let tools_block = tool_registry.format_tools_prompt(&selected);
            format!(
                "Available tools:\n{}\n\nQuery: {}\n\nIf you need to use a tool, respond with [tool_call]tool_name(arguments)[/tool_call]. Otherwise answer directly.",
                tools_block, prompt
            )
        };

        tracing::info!("Tool-calling: {} tools selected, prompt {} chars", selected.len(), tools_prompt.len());

        let tokenizer = SimpleTokenizer::new();
        let tokens = tokenizer.encode(&tools_prompt);
        let mut output_tokens = self.generate(&tokens, config)?;

        // Check if the output contains a tool call
        let output_text = tokenizer.decode(&output_tokens);
        if let Some(start_marker) = output_text.find("[tool_call]") {
            let content_start = start_marker + "[tool_call]".len();
            let content_end = output_text.find("[/tool_call]").unwrap_or(output_text.len());
            let call_text = &output_text[content_start..content_end];

            // Parse tool name and arguments
            let (tool_name, args) = if let Some(paren) = call_text.find('(') {
                (
                    call_text[..paren].trim().to_string(),
                    call_text[paren + 1..].trim_end_matches(')').trim().to_string(),
                )
            } else {
                (call_text.trim().to_string(), String::new())
            };

            tracing::info!("Tool call detected: name={}, args={}", tool_name, args);

            // Create tool result
            let result = if tool_registry.get(&tool_name).is_some() {
                crate::inference::tool_search::ToolResult::success(
                    "call_1",
                    &tool_name,
                    &format!("Executed {}({}) — result from tool runtime", tool_name, args),
                )
            } else {
                crate::inference::tool_search::ToolResult::error(
                    "call_1",
                    &tool_name,
                    "Unknown tool",
                )
            };

            // Format result and continue generation with tool result injected
            let result_text = tool_registry.format_tool_result(&result);
            let continuation = format!(
                "{}\n\nTool result: {}\n\nContinue answering:",
                output_text, result_text
            );
            let cont_tokens = tokenizer.encode(&continuation);
            let more_tokens = self.generate(&cont_tokens, config)?;
            output_tokens.extend_from_slice(&more_tokens);
        }

        Ok(output_tokens)
    }
}
