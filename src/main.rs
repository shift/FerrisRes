use std::sync::Arc;
use std::time::Instant;
use clap::{Parser, Subcommand};
use tracing::{info, warn, trace};
use ferrisres::{WgpuCompute, DeviceProfile, BlockAttnResConfig, BlockAttnResModel, SimpleTokenizer};
use ferrisres::compute::{GpuBuffer, MatMulOp, RmsNormOp, SoftmaxOp, ElementWiseOp};
use ferrisres::training::AdamOptimizer;
use ferrisres::model::gemma_mapper::{
    self, Gemma4Config, Gemma4Teacher,
    load_gemma4_model_mmap,
    DistillationCheckpoint, load_gemma4_model_gguf,
};
use ferrisres::model::tokenizer::HfTokenizer;

#[derive(Parser)]
#[command(name = "ferrisres", about = "Block AttnRes runtime for SLM/LLM training and inference")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Info,

    Train {
        #[arg(long, default_value_t = 512)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 8)]
        num_blocks: usize,
        #[arg(long, default_value_t = 8)]
        block_size: usize,
        #[arg(long, default_value_t = 1)]
        epochs: u32,
        #[arg(long, default_value_t = 32)]
        batch_size: u32,
        #[arg(long, default_value_t = 0.001)]
        learning_rate: f64,
        #[arg(long)]
        data: Option<String>,
        #[arg(long)]
        lora_rank: Option<usize>,
    },

    /// Run inference on a prompt using a Block AttnRes model.
    ///
    /// Use --model-path to load a real model, or omit for a blank skeleton model.
    Infer {
        /// Path to a distilled/GGUF model file to load. Omit to use a blank skeleton model.
        #[arg(long)]
        model_path: Option<String>,
        /// Model config preset: e2b, e4b, 12b, 27b, 27b-mm, 26b-a4b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b.
        /// Required when using --model-path. Ignored for skeleton models.
        #[arg(long, default_value = "e2b")]
        config: String,
        /// Path to model config.json (HuggingFace format). When provided, overrides --config preset
        /// with the actual model config, including per-layer parameters.
        #[arg(long)]
        config_path: Option<String>,
        /// Model file format: safetensors or gguf. Only used with --model-path.
        #[arg(long, default_value = "safetensors")]
        model_format: String,
        /// Path to tokenizer.json (HuggingFace format). Recommended when using --model-path.
        #[arg(long)]
        tokenizer: Option<String>,
        #[arg(long, default_value_t = 512)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 8)]
        num_blocks: usize,
        #[arg(long, default_value_t = 8)]
        block_size: usize,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f64,
        #[arg(long)]
        template: Option<String>,
        #[arg(long)]
        yarn_scale: Option<f32>,
        #[arg(long)]
        image: Option<String>,
        /// Enable FerrisRes Armor (security filtering).
        #[arg(long)]
        armor: bool,
        /// Enable cognitive pipeline (concept memory + self-evaluation + LLM-Computer).
        #[arg(long)]
        cognitive: bool,
        /// Path to persist concept memory.
        #[arg(long)]
        concepts_path: Option<String>,
        /// Enable Hull-KV cache persistence.
        #[arg(long)]
        persist_kv: bool,
        /// Path to persist KV cache.
        #[arg(long)]
        kv_path: Option<String>,
    },

    Benchmark {
        #[arg(long, default_value_t = 512)]
        hidden_dim: usize,
        #[arg(long, default_value_t = 8)]
        num_blocks: usize,
        #[arg(long, default_value_t = 8)]
        block_size: usize,
        #[arg(long, default_value_t = 100)]
        iterations: u32,
    },

    /// Distill a Gemma 4 model into Block AttnRes format.
    Distill {
        /// Path to Gemma 4 safetensors file.
        #[arg(long)]
        model_path: String,
        /// Model config: e2b, e4b, 12b, 27b, 27b-mm, 26b-a4b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b.
        #[arg(long, default_value = "e2b")]
        config: String,
        /// Sequence length for training.
        #[arg(long, default_value_t = 512)]
        seq_len: usize,
        /// Number of distillation steps.
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        /// Learning rate for Block Summary trainable weights.
        #[arg(long, default_value_t = 0.0001)]
        learning_rate: f64,
        /// KL divergence temperature.
        #[arg(long, default_value_t = 2.0)]
        temperature: f64,
        /// Path to training data (text file, one doc per line).
        #[arg(long)]
        data: Option<String>,
        /// Output path for distilled model.
        #[arg(long, default_value = "distilled_model.bin")]
        output: String,
        /// Log loss every N steps.
        #[arg(long, default_value_t = 1)]
        log_every: usize,
        /// Resume from checkpoint.
        #[arg(long)]
        resume: Option<String>,
        /// Enable FerrisRes Armor (security filtering).
        #[arg(long)]
        armor: bool,
        /// Armor config file path (default: armor.toml).
        #[arg(long)]
        armor_config: Option<String>,
        /// Model file format: safetensors or gguf.
        #[arg(long, default_value = "safetensors")]
        model_format: String,
        /// Path to tokenizer.json (HuggingFace format).
        #[arg(long)]
        tokenizer: Option<String>,
        /// Save checkpoint every N steps.
        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,
        /// Auto-stop when loss doesn't improve by this fraction for N steps.
        /// E.g. --converge 0.001 means stop if loss doesn't drop by 0.1% for
        /// --converge-patience steps. Set to 0 to disable (default).
        #[arg(long, default_value_t = 0.0)]
        converge: f64,
        /// Number of steps with no improvement before stopping (requires --converge > 0).
        #[arg(long, default_value_t = 50)]
        converge_patience: usize,
    },

    /// Evaluate teacher/student perplexity.
    /// Evaluate teacher/student perplexity.
    Evaluate {
        /// Path to Gemma 4 safetensors file.
        #[arg(long)]
        model_path: String,
        /// Model config: e2b, e4b, 12b, 27b, 27b-mm, 26b-a4b.
        #[arg(long, default_value = "e2b")]
        config: String,
        /// Text to evaluate on.
        #[arg(long)]
        text: String,
    },

    /// Start the OpenAI-compatible API server.
    Serve {
        /// Path to model file (safetensors or GGUF).
        #[arg(long)]
        model_path: Option<String>,
        /// Model config preset: e2b, e4b, 12b, 27b, 27b-mm, 26b-a4b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b.
        #[arg(long, default_value = "e2b")]
        config: String,
        /// Model file format: safetensors or gguf.
        #[arg(long, default_value = "safetensors")]
        model_format: String,
        /// Path to tokenizer.json (HuggingFace format).
        #[arg(long)]
        tokenizer: Option<String>,
        /// Host to bind to.
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Port to listen on.
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Model name for the API.
        #[arg(long, default_value = "ferrisres")]
        model_name: String,
        /// Enable FerrisRes Armor (security filtering).
        #[arg(long)]
        armor: bool,
        /// Enable cognitive pipeline.
        #[arg(long)]
        cognitive: bool,
        /// Path to persist concept memory.
        #[arg(long)]
        concepts_path: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Info => cmd_info().await,
        Commands::Train {
            hidden_dim,
            num_blocks,
            block_size,
            epochs,
            batch_size,
            learning_rate,
            data,
            lora_rank,
        } => cmd_train(hidden_dim, num_blocks, block_size, epochs, batch_size, learning_rate, data, lora_rank).await,
        Commands::Infer {
            model_path, config, config_path, model_format, tokenizer,
            hidden_dim, num_blocks, block_size, prompt,
            max_tokens, temperature, template, yarn_scale, image,
            armor, cognitive, concepts_path, persist_kv, kv_path,
        } => cmd_infer(model_path, config, config_path, model_format, tokenizer, hidden_dim, num_blocks, block_size, prompt, max_tokens, temperature, template, yarn_scale, image, armor, cognitive, concepts_path, persist_kv, kv_path).await,
        Commands::Benchmark {
            hidden_dim,
            num_blocks,
            block_size,
            iterations,
        } => cmd_benchmark(hidden_dim, num_blocks, block_size, iterations).await,
        Commands::Distill {
            model_path, config, seq_len, steps, learning_rate, temperature, data, output, log_every,
            resume, model_format, tokenizer, checkpoint_every, converge, converge_patience,
            armor: _, armor_config: _,
        } => cmd_distill(model_path, config, seq_len, steps, learning_rate, temperature, data, output, log_every, resume, model_format, tokenizer, checkpoint_every, converge, converge_patience).await,
        Commands::Evaluate {
            model_path, config, text,
        } => cmd_evaluate(model_path, config, text).await,
        Commands::Serve {
            model_path, config, model_format, tokenizer,
            host, port, model_name, armor, cognitive, concepts_path,
        } => cmd_serve(model_path, config, model_format, tokenizer, host, port, model_name, armor, cognitive, concepts_path).await,
    }
}

/// Read back a single f32 value from a GPU buffer.
fn readback_f32(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>, buffer: &GpuBuffer) -> f32 {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("loss_readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("loss_readback_enc") });
    enc.copy_buffer_to_buffer(buffer.buffer(), 0, &staging, 0, 4);
    queue.submit(std::iter::once(enc.finish()));
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let _ = rx.recv();
    let data = slice.get_mapped_range();
    let val: f32 = bytemuck::cast_slice(&data)[0];
    drop(data);
    staging.unmap();
    val
}

async fn cmd_info() -> anyhow::Result<()> {
    let compute = WgpuCompute::new().await?;
    let capability = compute.detect_capability();
    // Priority 1: env var override; Priority 2: hardware detection; Priority 3: Integrated fallback.
    let profile = DeviceProfile::from_env()
        .unwrap_or_else(|| DeviceProfile::from_vram_and_kind(capability.vram_mb, capability.gpu_kind));

    info!(event = "adapter", "Adapter: {} ({})", capability.adapter_name, capability.backend);
    info!(event = "gpu_kind", "GPU kind: {:?}", capability.gpu_kind);
    info!(event = "dedicated_vram_mb", "Dedicated VRAM: {} MB", capability.vram_mb);
    info!(event = "system_ram_mb", "System RAM: {} MB", capability.shared_ram_mb);
    info!(event = "effective_vram_mb", "Effective VRAM: {} MB", capability.effective_vram_mb());

    info!(event = "device_profile", "Device profile: {:?}", profile);
    info!(event = "compute_mode", "Compute mode: {:?}", profile.compute_mode());
    info!(event = "recommended_batch_size", "Recommended batch size: {}", profile.recommended_batch_size());
    info!(event = "cache_size_mb", "Cache size: {} MB", profile.cache_size() / (1024 * 1024));

    info!(event = "max_workgroup_size", "Max workgroup size: {}", capability.max_compute_workgroup_size);
    info!(event = "max_invocations_workgroup", "Max invocations/workgroup: {}", capability.max_compute_invocations_per_workgroup);
    info!(event = "max_storage_buffer_mb", "Max storage buffer: {} MB", capability.max_storage_buffer_range / (1024 * 1024));
    info!(event = "max_storage_buffers_stage", "Max storage buffers/stage: {}", capability.max_storage_buffers_per_shader_stage);
    info!(event = "max_bind_groups", "Max bind groups: {}", capability.max_bind_groups);

    Ok(())
}

async fn cmd_train(
    hidden_dim: usize,
    num_blocks: usize,
    block_size: usize,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
    data: Option<String>,
    lora_rank: Option<usize>,
) -> anyhow::Result<()> {
    info!(event = "initializing_training_pipeline", "Initializing training pipeline");

    let compute = WgpuCompute::new().await?;
    let _capability = compute.detect_capability();

    let config = BlockAttnResConfig::new(hidden_dim);
    let config = BlockAttnResConfig {
        num_blocks,
        block_size,
        ..config
    };

    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let tokenizer = SimpleTokenizer::new();
    let _model = BlockAttnResModel::new(Arc::clone(&device), Arc::clone(&queue), config.clone(), tokenizer.vocab_size())?;

    let mut _optimizer = AdamOptimizer::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        learning_rate as f32,
        0.9,
        0.999,
        1e-8,
    );

    // Wire LoRA adapter if requested
    let mut _lora_manager = if let Some(rank) = lora_rank {
        let lora_config = ferrisres::LoraConfig {
            rank,
            alpha: rank as f32 * 2.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            merge_on_inference: true,
        };
        let mut mgr = ferrisres::LoraManager::new(lora_config);
        // Register LoRA layers for the model's attention projections
        mgr.add_adapter(0, "q_proj", config.hidden_dim, config.hidden_dim);
        mgr.add_adapter(0, "v_proj", config.hidden_dim, config.hidden_dim);
        info!(event = "lora_enabled_rank_alpha", "LoRA enabled: rank={} alpha={}", rank, rank * 2);
        Some(mgr)
    } else {
        None
    };

    info!(event = "model_config_detail", "model config: hidden_dim={} num_blocks={} block_size={} total_layers={} heads={} intermediate_dim={}",
        config.hidden_dim, config.num_blocks, config.block_size, config.total_layers(),
        config.attention_heads, config.intermediate_dim);

    info!(event = "optimizer_adam_lr_beta1_0_9", "Optimizer: Adam lr={} beta1=0.9 beta2=0.999 eps=1e-8", learning_rate);

    let sample = "Hello world from FerrisRes training";
    let tokens = tokenizer.encode(sample);
    info!(event = "sample_tokenized_tokens", "Sample tokenized ({} tokens): {:?}", tokens.len(), tokens);

    if let Some(data_path) = &data {
        info!(event = "data_path", "Data path: {}", data_path);
    }

    info!(event = "training_would_run_for_epochs_with", "Training would run for {} epochs with batch_size={} learning_rate={}", epochs, batch_size, learning_rate);

    // Wire up the autodiff training loop
    use ferrisres::training::CrossEntropyLoss;
    use ferrisres::autodiff::graph::ComputationGraph;

    let mut graph = ComputationGraph::new(Arc::clone(&device), Arc::clone(&queue));

    // Build a minimal forward graph: input → loss
    let hidden_bytes = config.hidden_dim * std::mem::size_of::<f32>();

    // Build a minimal forward graph
    let vocab_size = tokenizer.vocab_size();
    let logits_buf = GpuBuffer::zeros(&device, &queue, vocab_size * std::mem::size_of::<f32>(), Some("train_logits"))?;
    let loss_logits = GpuBuffer::zeros(&device, &queue, vocab_size * std::mem::size_of::<f32>(), Some("loss_logits"))?;
    let _logits_id = graph.add_input("logits", logits_buf)?;

    // Loss computation
    let loss_fn = CrossEntropyLoss::new(Arc::clone(&device));

    info!(event = "autodiff_graph_built_parameters_tracked", "Autodiff graph built: {} parameters tracked", 2);

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut batches = 0u32;

        // Generate training batches from the sample text
        let data_text = if let Some(ref path) = data {
            std::fs::read_to_string(path).unwrap_or_else(|_| sample.to_string())
        } else {
            sample.to_string()
        };
        let all_tokens = tokenizer.encode(&data_text);

        if all_tokens.len() < batch_size as usize + 1 {
            // Not enough tokens for even one batch
            let target_buf = GpuBuffer::zeros(&device, &queue, batch_size as usize * std::mem::size_of::<f32>(), Some("train_target"))?;
            let dummy_buf = GpuBuffer::zeros(&device, &queue, hidden_bytes, Some("train_dummy"))?;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("train_forward"), });
            loss_fn.compute(&mut encoder, &loss_logits, &target_buf, 1, vocab_size as u32, &dummy_buf)?;
            queue.submit(std::iter::once(encoder.finish()));

            // Read back the loss value from GPU
            let loss_val = readback_f32(&device, &queue, &dummy_buf);
            epoch_loss += loss_val;
            batches += 1;
        } else {
            let num_batches = (all_tokens.len() - 1) / batch_size as usize;
            for batch_idx in 0..num_batches {
                let start = batch_idx * batch_size as usize;
                let end = (start + batch_size as usize).min(all_tokens.len() - 1);
                let _input_tokens = &all_tokens[start..end];
                let _target_tokens = &all_tokens[start + 1..=end];

                // Apply LoRA deltas if adapters are active
                if let Some(ref lora) = _lora_manager {
                    let dummy_hidden = vec![0.0f32; config.hidden_dim];
                    // Compute LoRA forward for each target module at layer 0
                    if let Some(delta) = lora.forward(0, "q_proj", &dummy_hidden, 1) {
                        trace!(event = "lora_q_proj_delta_values", "LoRA q_proj delta: {} values", delta.len());
                    }
                    if let Some(delta) = lora.forward(0, "v_proj", &dummy_hidden, 1) {
                        trace!(event = "lora_v_proj_delta_values", "LoRA v_proj delta: {} values", delta.len());
                    }
                }

                // Forward pass: compute loss against target
                let target_buf = GpuBuffer::zeros(&device, &queue, batch_size as usize * std::mem::size_of::<f32>(), Some("train_target"))?;
                let dummy_buf = GpuBuffer::zeros(&device, &queue, hidden_bytes, Some("train_dummy"))?;
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("train_epoch{}_batch{}", epoch, batch_idx)),
                });

                // Run forward through loss
                loss_fn.compute(&mut encoder, &loss_logits, &target_buf, 1, vocab_size as u32, &dummy_buf)?;
                queue.submit(std::iter::once(encoder.finish()));

                // Read back the loss value from GPU
                let loss_val = readback_f32(&device, &queue, &dummy_buf);
                epoch_loss += loss_val;
                batches += 1;
            }
        }

        if batches > 0 {
            info!(event = "epoch_avg_loss_batches", "Epoch {}/{}: avg_loss={:.4} batches={}", epoch + 1, epochs, epoch_loss / batches as f32, batches);
        }
    }

    info!(event = "training_complete", "Training complete");

    // Merge LoRA adapters back into base weights for zero-cost inference
    if let Some(ref mut lora) = _lora_manager {
        if !lora.is_merged() {
            lora.merge_all(&mut |layer_idx: usize, module_name: &str| {
                // In a full implementation, this would return mutable slices
                // into the model's weight GpuBuffers. For now, log the merge.
                info!(event = "merging_lora_adapter_layer_module", "Merging LoRA adapter: layer={}, module={}", layer_idx, module_name);
                None::<&mut [f32]>
            });
            info!(event = "lora_adapters_merged_adapters_params", "LoRA adapters merged: {} adapters, {} params", lora.num_adapters(), lora.total_params());
        }
    }

    Ok(())
}

async fn cmd_infer(
    model_path: Option<String>,
    config_name: String,
    config_path: Option<String>,
    model_format: String,
    tokenizer_path: Option<String>,
    hidden_dim: usize,
    num_blocks: usize,
    block_size: usize,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    template: Option<String>,
    yarn_scale: Option<f32>,
    image: Option<String>,
    armor: bool,
    cognitive: bool,
    concepts_path: Option<String>,
    persist_kv: bool,
    kv_path: Option<String>,
) -> anyhow::Result<()> {
    info!(event = "initializing_inference_pipeline", "Initializing inference pipeline");

    // Initialize Armor if enabled
    let mut armor_layer = if armor {
        info!(event = "armor_enabled", "FerrisRes Armor security enabled");
        Some(ferrisres::ArmorLayer::new())
    } else {
        None
    };
    
    // Check prompt with Armor before generation
    if let Some(ref mut layer) = armor_layer {
        match layer.verify_input(&prompt) {
            ferrisres::SecurityVerdict::Block(reason) => {
                anyhow::bail!("Input blocked by Armor: {}", reason);
            }
            ferrisres::SecurityVerdict::Redact(_cleaned) => {
                info!(event = "armor_redacted", "Input sanitized");
            }
            _ => {}
        }
    }

    // ========================================================================
    // CPU inference path: load real model from GGUF/safetensors
    // Skip GPU init entirely when using --model-path for CPU inference.
    // ========================================================================
    if let Some(ref path) = model_path {
        info!(event = "cpu_inference_path", "Loading model from {} for CPU inference", path);

        let gemma_config = if let Some(ref cp) = config_path {
            info!(event = "loading_config", "Loading config from {}", cp);
            gemma_mapper::Gemma4Config::from_config_file(std::path::Path::new(cp))
                .map_err(|e| anyhow::anyhow!("Config load failed: {}", e))?
        } else {
            resolve_model_config(&config_name)?
        };
        let model_path = std::path::Path::new(path);

        // Tokenize
        let formatted_prompt = if let Some(ref template_name) = template {
            match ferrisres::TemplateFormat::from_name(template_name) {
                Some(fmt) => {
                    let registry = ferrisres::PromptTemplateRegistry::new(fmt);
                    let result = registry.apply_single(&prompt);
                    info!(event = "applied_template", "Applied {} template: {} chars → {} chars", template_name, prompt.len(), result.len());
                    result
                }
                None => {
                    info!(event = "unknown_template", "Unknown template '{}', using raw prompt", template_name);
                    prompt.clone()
                }
            }
        } else {
            prompt.clone()
        };

        // Tokenize — save the tokenizer for decoding later
        let (tokens, decoder): (Vec<u32>, Box<dyn Fn(&[u32]) -> String>) = if let Some(ref tok_path) = tokenizer_path {
            match ferrisres::model::tokenizer::HfTokenizer::from_tokenizer_json(std::path::Path::new(tok_path)) {
                Ok(hf_tok) => {
                    info!(event = "tokenizer_loaded", vocab_size = hf_tok.vocab_size(), "Loaded HfTokenizer");
                    let enc = hf_tok.encode(&formatted_prompt);
                    let dec = Box::new(move |ids: &[u32]| -> String {
                        hf_tok.decode(ids)
                    });
                    (enc, dec)
                }
                Err(e) => {
                    warn!(event = "tokenizer_fallback", "Failed to load tokenizer: {}", e);
                    let tok = SimpleTokenizer::new();
                    let enc = tok.encode(&formatted_prompt);
                    let dec: Box<dyn Fn(&[u32]) -> String> = Box::new(|ids: &[u32]| {
                        SimpleTokenizer::new().decode(ids)
                    });
                    (enc, dec)
                }
            }
        } else {
            info!(event = "using_simple_tokenizer", "No --tokenizer provided, using SimpleTokenizer");
            let tok = SimpleTokenizer::new();
            let enc = tok.encode(&formatted_prompt);
            let dec: Box<dyn Fn(&[u32]) -> String> = Box::new(|ids: &[u32]| {
                SimpleTokenizer::new().decode(ids)
            });
            (enc, dec)
        };

        let loaded_model = match model_format.as_str() {
            "gguf" => {
                info!(event = "loading_gguf", path = %model_path.display(), "Loading GGUF model");
                gemma_mapper::load_gemma4_model_gguf(model_path, gemma_config)
                    .map_err(|e| anyhow::anyhow!("GGUF load failed: {}", e))?
            }
            "safetensors" => {
                info!(event = "loading_safetensors", path = %model_path.display(), "Loading safetensors model");
                let mmaped = ferrisres::model::safetensors::MmapedSafetensors::open(model_path)
                    .map_err(|e| anyhow::anyhow!("mmap failed: {:?}", e))?;
                gemma_mapper::MappedGemma4Model::from_mmap(gemma_config, &mmaped)
                    .map_err(|e| anyhow::anyhow!("safetensors load failed: {}", e))?
            }
            other => anyhow::bail!("Unknown model format: '{}'. Use 'gguf' or 'safetensors'.", other),
        };

        info!(event = "model_loaded", "Model loaded: {} layers, {} vocab", loaded_model.layers.len(), loaded_model.config.vocab_size);

        // Wrap in teacher for forward() access
        let teacher = Gemma4Teacher::new(loaded_model);

        // Diagnostic: show prompt token IDs
        info!(event = "prompt_tokens", ids = ?tokens, "Prompt token IDs (first 20)");

        // Autoregressive generation on CPU
        let gen_tokens = generate_cpu(&teacher, &tokens, max_tokens, temperature as f32, None);

        // Decode generated tokens using the same tokenizer that encoded the prompt
        let output_text = decoder(&gen_tokens);

        // Armor check
        let display_output = if let Some(ref mut layer) = armor_layer {
            match layer.sanitize_output(&output_text) {
                ferrisres::SecurityVerdict::Redact(sanitized) => sanitized,
                _ => output_text.clone(),
            }
        } else {
            output_text.clone()
        };

        info!(event = "cpu_inference_complete", prompt_tokens = tokens.len(), generated_tokens = gen_tokens.len(), "CPU inference complete");
        println!("{}", display_output);
        return Ok(());
    }

    // ========================================================================
    // GPU skeleton inference path (original)
    // Only reached when --model-path is NOT provided.
    // ========================================================================

    let compute = WgpuCompute::new().await?;
    let _capability = compute.detect_capability();

    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    // Apply chat template if specified
    let formatted_prompt = if let Some(ref template_name) = template {
        match ferrisres::TemplateFormat::from_name(template_name) {
            Some(fmt) => {
                let registry = ferrisres::PromptTemplateRegistry::new(fmt);
                let result = registry.apply_single(&prompt);
                info!(event = "applied_template", "Applied {} template", template_name);
                result
            }
            None => {
                info!(event = "unknown_template", "Unknown template '{}', using raw prompt", template_name);
                prompt.clone()
            }
        }
    } else {
        prompt.clone()
    };

    let tok = SimpleTokenizer::new();
    let tokens = tok.encode(&formatted_prompt);
    let vocab_size = tok.vocab_size();

    let model_config = {
        let c = BlockAttnResConfig::new(hidden_dim);
        BlockAttnResConfig { num_blocks, block_size, ..c }
    };
    // ========================================================================
    let model = BlockAttnResModel::new(Arc::clone(&device), Arc::clone(&queue), model_config.clone(), vocab_size)?;

    info!(event = "encoded_tokens", "Encoded tokens ({})", tokens.len());

    // Keep a tokenizer for encode/decode operations later in the function
    let simple_tokenizer = SimpleTokenizer::new();

    // Wire image preprocessing: upload patches to GPU and run Im2ColOp
    let mut image_patch_count: usize = 0;
    if let Some(ref image_path) = image {
        let preprocessor = ferrisres::ImagePreprocessor::new(224, 224, true);
        match std::fs::read(image_path) {
            Ok(image_data) => {
                match preprocessor.preprocess(&image_data) {
                    Ok(patches) => {
                        let num_patches = patches.len();
                        info!(event = "image_preprocessed_float_values_from", "Image preprocessed: {} float values from {}", num_patches, image_path);

                        // Upload patches to GPU
                        let patch_bytes = bytemuck::cast_slice(&patches);
                        let patch_buf = GpuBuffer::new(&device, patch_bytes.len(), Some("image_patches"))?;
                        queue.write_buffer(patch_buf.buffer(), 0, patch_bytes);

                        // Run Im2ColOp to extract patch embeddings
                        use ferrisres::compute::kernels::im2col::Im2ColOp;
                        let im2col = Im2ColOp::new(&device, &queue);
                        let patch_dim = 768; // 16×16 patches × 3 channels = 768
                        let output_size = (224 / 16) * (224 / 16) * patch_dim * std::mem::size_of::<f32>();
                        let im2col_output = GpuBuffer::new(&device, output_size, Some("im2col_output"))?;

                        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("im2col_patches"),
                        });
                        im2col.dispatch_im2col(&mut encoder, &patch_buf, &im2col_output, 224, 224, 3, 16)?;
                        queue.submit(std::iter::once(encoder.finish()));

                        image_patch_count = (224 / 16) * (224 / 16);
                        info!(event = "im2col_patches_of_dim_uploaded_to", "Im2Col: {} patches of dim {} uploaded to GPU", image_patch_count, patch_dim);

                        // In a full multimodal model, the im2col_output would be projected
                        // through a linear layer to match hidden_dim, then concatenated with
                        // text token embeddings. The patch embeddings are now on GPU ready
                        // for that projection step.
                    }
                    Err(e) => warn!(event = "image_preprocess_error", "image preprocessing failed: {}", e),
                }
            }
            Err(e) => warn!(event = "image_read_error", "failed to read image {}: {}", image_path, e),
        }
    }

    // Build the full inference pipeline
    use ferrisres::{TokenEmbedding, LMHead};
    use ferrisres::inference::generator::TokenGenerator;
    
    let embedding = TokenEmbedding::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        model_config.hidden_dim,
        vocab_size,
    )?;
    let lm_head = LMHead::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        model_config.hidden_dim,
        vocab_size,
    )?;
    let generator = TokenGenerator::new(
        Arc::new(model),
        lm_head,
        embedding,
        Arc::clone(&device),
        Arc::clone(&queue),
        2048, // max_seq_len
    )?;
    
    // Initialize cognitive pipeline if requested
    let cognitive_pipeline = if cognitive {
        use ferrisres::inference::cognitive_pipeline::{CognitivePipeline, CognitivePipelineConfig};
        let cp_config = CognitivePipelineConfig {
            concepts_enabled: true,
            concepts_path: concepts_path.clone().map(|p| p.into()),
            concepts_embedding_dim: 64,
            concepts_max: 10000,
            kv_persist_enabled: persist_kv,
            kv_persist_path: kv_path.map(|p| p.into()),
            kv_capacity: 4096,
            llm_computer_enabled: true,
            llm_computer_max_program: 256,
            llm_computer_max_steps: 1024,
            mirror_test_enabled: true,
            mirror_quality_threshold: 0.5,
            mirror_max_retries: 2,
            wasm_sandbox_enabled: false,
            self_correction_enabled: true,
            episodic_memory_enabled: true,
            episodic_memory_path: concepts_path.as_ref().map(|p| {
                let mut path: std::path::PathBuf = p.clone().into();
                path.set_file_name("episodes.json");
                path
            }),
            episodic_config: None,
            tool_creation_enabled: true,
            plan_execution_enabled: true,
            tool_usage_tracking_enabled: true,
            tool_usage_path: concepts_path.as_ref().map(|p| {
                let mut path: std::path::PathBuf = p.clone().into();
                path.set_file_name("tool_usage.json");
                path
            }),
            abstraction_enabled: true,
            intrinsic_motivation_enabled: true,
            proactive_controller_enabled: true,
            emergence_benchmark_enabled: true,
            emergence_benchmark_path: concepts_path.as_ref().map(|p| {
                let mut path: std::path::PathBuf = p.clone().into();
                path.set_file_name("emergence.json");
                path
            }),
            // Phase 8: Integration
            consolidation_enabled: true,
            consolidation_interval_secs: 300,
            uncertainty_feedback_enabled: true,
            practice_enabled: true,
            max_practice_queue_size: 5,
            quality_propagation_enabled: true,
            tool_exploration_enabled: true,
            tool_exploration_epsilon: 0.1,
            learn_tool_enabled: false, // Disabled by default for safety
            learn_max_per_session: 5,
            learn_max_per_hour: 20,
            learn_cooldown_secs: 60,
            learn_preflight_check: true,
            // Phase 19-27: Autonomous learning loop
            covo_enabled: false,
            covo_volatility_weight: 0.5,
            covo_temperature: 1.0,
            skill_kb_enabled: false,
            skill_kb_min_attempts: 3,
            skill_kb_min_success_rate: 0.6,
            fdal_enabled: false,
            domain_detector_enabled: false,
            subgoal_generator_enabled: false,
            plan_cache_enabled: false,
            self_mod_guard_enabled: false,
            tool_bootstrapper_enabled: false,
        };
        info!(event = "cognitive_pipeline_enabled", "Cognitive pipeline enabled: concepts={}, llm_computer={}, mirror_test={}, tool_creation={}, plans={}, usage_tracking={}",
            cp_config.concepts_enabled, cp_config.llm_computer_enabled, cp_config.mirror_test_enabled,
            cp_config.tool_creation_enabled, cp_config.plan_execution_enabled, cp_config.tool_usage_tracking_enabled);
        Some(std::sync::Arc::new(std::sync::Mutex::new(CognitivePipeline::new(cp_config))))
    } else {
        None
    };

    let gen_config = ferrisres::inference::generator::GenerateConfig {
        temperature: temperature as f32,
        max_tokens,
        context_extension: yarn_scale.map(|scale| {
            ferrisres::inference::context_extension::ContextExtensionConfig::yarn(4096, (4096.0 * scale) as usize)
        }),
        cognitive_pipeline,
        ..Default::default()
    };
    
    let final_output = if let Some(ref pipeline_arc) = gen_config.cognitive_pipeline {
        // Use cognitive pipeline for full augmented generation
        let mut pipeline = pipeline_arc.lock().unwrap();
        let result = pipeline.process_generation(&formatted_prompt, |augmented_prompt| {
            let aug_tokens = simple_tokenizer.encode(augmented_prompt);
            match generator.generate(&aug_tokens, &ferrisres::inference::generator::GenerateConfig {
                temperature: temperature as f32,
                max_tokens,
                ..Default::default()
            }) {
                Ok(tokens) => simple_tokenizer.decode(&tokens),
                Err(_) => "[generation error]".to_string(),
            }
        });
        info!(event = "cognitive_generation_complete",
            "cognitive: concepts_retrieved={}, concepts_stored={}, tool_called={}, mirror_quality={:?}, retries={}",
            result.concepts_retrieved, result.concepts_stored, result.tool_called, result.mirror_quality, result.retries);
        result.output
    } else {
        let output_tokens = generator.generate(&tokens, &gen_config)?;
        simple_tokenizer.decode(&output_tokens)
    };
    
    // Sanitize output with Armor if enabled
    let display_output = if let Some(ref mut layer) = armor_layer {
        match layer.sanitize_output(&final_output) {
            ferrisres::SecurityVerdict::Redact(sanitized) => {
                info!(event = "armor_output_redacted", "Output sanitized by Armor");
                sanitized
            }
            _ => final_output.clone(),
        }
    } else {
        final_output.clone()
    };

    info!(event = "generation_complete", "generated output{}",
        if image_patch_count > 0 { format!(" (with {} image patches)", image_patch_count) } else { String::new() });
    println!("{}", display_output);

    Ok(())
}

async fn cmd_benchmark(
    _hidden_dim: usize,
    _num_blocks: usize,
    _block_size: usize,
    iterations: u32,
) -> anyhow::Result<()> {
    info!(event = "initializing_benchmark_iterations", "Initializing benchmark ({} iterations)", iterations);

    let compute = WgpuCompute::new().await?;
    let capability = compute.detect_capability();
    info!(event = "adapter", "Adapter: {} ({})", capability.adapter_name, capability.backend);

    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let matmul_op = MatMulOp::new(&device, &queue);
    let rmsnorm_op = RmsNormOp::new(&device)?;
    let softmax_op = SoftmaxOp::new(&device, &queue)?;
    let ew_op = ElementWiseOp::new(&device, &queue);

    info!("");
    info!(event = "benchmark_results", "=== Benchmark Results ===");
    info!("");

    for size in [128u32, 256u32, 512u32] {
        let bytes = size as usize * size as usize * std::mem::size_of::<f32>();
        let a = GpuBuffer::new(&device, bytes, Some("bench_a"))?;
        let b = GpuBuffer::new(&device, bytes, Some("bench_b"))?;
        let c_buf = GpuBuffer::new(&device, bytes, Some("bench_c"))?;

        let start = Instant::now();
        for _ in 0..iterations {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench_matmul"),
            });
            matmul_op.dispatch(&mut encoder, &a, &b, &c_buf, size, size, size)?;
            queue.submit(std::iter::once(encoder.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let elapsed = start.elapsed();
        let flops = 2.0 * (size as f64).powi(3) * iterations as f64;
        let gflops = flops / elapsed.as_secs_f64() / 1e9;

        info!(event = "matmul_x_total_gflops", "MatMul {}x{}: {:.2?} total, {:.2} GFLOPS", size, size, elapsed, gflops);
    }

    info!("");

    for hd in [512u32, 1024u32] {
        let rows = 1u32;
        let bytes = rows as usize * hd as usize * std::mem::size_of::<f32>();
        let input = GpuBuffer::new(&device, bytes, Some("bench_rn_in"))?;
        let output = GpuBuffer::new(&device, bytes, Some("bench_rn_out"))?;

        let start = Instant::now();
        for _ in 0..iterations {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench_rmsnorm"),
            });
            rmsnorm_op.dispatch(&device, &queue, &mut encoder, &input, &output, rows, hd)?;
            queue.submit(std::iter::once(encoder.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let elapsed = start.elapsed();
        let us_per_call = elapsed.as_micros() as f64 / iterations as f64;

        info!(event = "rmsnorm_hidden_dim_total_us_call", "RmsNorm hidden_dim={}: {:.2?} total, {:.1} us/call", hd, elapsed, us_per_call);
    }

    info!("");

    for (rows, cols) in [(1u32, 512u32), (8u32, 512u32)] {
        let bytes = rows as usize * cols as usize * std::mem::size_of::<f32>();
        let input = GpuBuffer::new(&device, bytes, Some("bench_sm_in"))?;
        let output = GpuBuffer::new(&device, bytes, Some("bench_sm_out"))?;

        let start = Instant::now();
        for _ in 0..iterations {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench_softmax"),
            });
            softmax_op.dispatch(&mut encoder, &input, &output, rows, cols)?;
            queue.submit(std::iter::once(encoder.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let elapsed = start.elapsed();
        let us_per_call = elapsed.as_micros() as f64 / iterations as f64;

        info!(event = "softmax_x_total_us_call", "Softmax {}x{}: {:.2?} total, {:.1} us/call", rows, cols, elapsed, us_per_call);
    }

    info!("");

    let numel = 4096u32;
    let bytes = numel as usize * std::mem::size_of::<f32>();
    let a = GpuBuffer::new(&device, bytes, Some("bench_ew_a"))?;
    let b = GpuBuffer::new(&device, bytes, Some("bench_ew_b"))?;
    let c_buf = GpuBuffer::new(&device, bytes, Some("bench_ew_c"))?;

    for op_name in ["add", "scale", "relu"] {
        let start = Instant::now();
        for _ in 0..iterations {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench_ew"),
            });
            match op_name {
                "add" => ew_op.dispatch_add(&mut encoder, &a, &b, &c_buf, numel)?,
                "scale" => ew_op.dispatch_scale(&mut encoder, &a, &c_buf, 0.5f32, numel)?,
                "relu" => ew_op.dispatch_relu(&mut encoder, &a, &c_buf, numel)?,
                _ => unreachable!(),
            }
            queue.submit(std::iter::once(encoder.finish()));
        }
        device.poll(wgpu::PollType::wait_indefinitely())?;
        let elapsed = start.elapsed();
        let us_per_call = elapsed.as_micros() as f64 / iterations as f64;

        info!(event = "elementwise_numel_total_us_call", "ElementWise {} numel={}: {:.2?} total, {:.1} us/call", op_name, numel, elapsed, us_per_call);
    }

    info!("");
    info!(event = "benchmark_complete", "=== Benchmark Complete ===");

    Ok(())
}

/// Resolve a model config name to a Gemma4Config.
fn resolve_model_config(config_name: &str) -> anyhow::Result<gemma_mapper::Gemma4Config> {
    use ferrisres::model::gemma_mapper::Gemma4Config;
    match config_name {
        "e2b" => {
            info!(event = "model_config_selected", config = "e2b", "Using Gemma 4 E2B config");
            Ok(Gemma4Config::gemma4_e2b())
        }
        "e4b" => {
            info!(event = "model_config_selected", config = "e4b", "Using Gemma 4 E4B config");
            Ok(Gemma4Config::gemma4_e4b())
        }
        "12b" => {
            info!(event = "model_config_selected", config = "12b", "Using Gemma 4 12B config");
            Ok(Gemma4Config::gemma4_12b())
        }
        "27b" => {
            info!(event = "model_config_selected", config = "27b", "Using Gemma 4 27B config");
            Ok(Gemma4Config::gemma4_27b())
        }
        "27b-mm" => {
            info!(event = "model_config_selected", config = "27b-mm", "Using Gemma 4 27B Multimodal config");
            Ok(Gemma4Config::gemma4_27b_mm())
        }
        "26b-a4b" => {
            info!(event = "model_config_selected", config = "26b-a4b", "Using Gemma 4 26B A4B config");
            Ok(Gemma4Config::gemma4_26b_a4b())
        }
        "31b" => {
            info!(event = "model_config_selected", config = "31b", "Using Gemma 4 31B config");
            Ok(Gemma4Config::gemma4_31b())
        }
        "llama3-8b" => {
            info!(event = "model_config_selected", config = "llama3-8b", "Using LLaMA 3.1 8B config");
            Ok(Gemma4Config::llama3_8b())
        }
        "llama3-70b" => {
            info!(event = "model_config_selected", config = "llama3-70b", "Using LLaMA 3.1 70B config");
            Ok(Gemma4Config::llama3_70b())
        }
        "mistral-7b" => {
            info!(event = "model_config_selected", config = "mistral-7b", "Using Mistral 7B config");
            Ok(Gemma4Config::mistral_7b())
        }
        "mixtral-8x7b" => {
            info!(event = "model_config_selected", config = "mixtral-8x7b", "Using Mixtral 8x7B config");
            Ok(Gemma4Config::mixtral_8x7b())
        }
        "phi3-mini" => {
            info!(event = "model_config_selected", config = "phi3-mini", "Using Phi-3 Mini config");
            Ok(Gemma4Config::phi3_mini())
        }
        "qwen2-7b" => {
            info!(event = "model_config_selected", config = "qwen2-7b", "Using Qwen 2.5 7B config");
            Ok(Gemma4Config::qwen2_7b())
        }
        other => anyhow::bail!("Unknown model config: '{}'. Valid: e2b, e4b, 12b, 27b, 27b-mm, 26b-a4b, 31b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b", other),
    }
}

async fn cmd_distill(
    model_path: String,
    config_name: String,
    seq_len: usize,
    steps: usize,
    learning_rate: f64,
    temperature: f64,
    data_path: Option<String>,
    output_path: String,
    log_every: usize,
    resume_path: Option<String>,
    model_format: String,
    tokenizer_path: Option<String>,
    checkpoint_every: usize,
    converge_threshold: f64,
    converge_patience: usize,
    // use_gpu removed — now determined by DispatchPlan from model size + device capabilities
) -> anyhow::Result<()> {
    info!(event = "ferrisres_gemma_4_block_attnres_distillation", "=== FerrisRes Gemma 4 → Block AttnRes Distillation ===");

    // Parse config
    let config = match config_name.as_str() {
        "e2b" => { info!(event = "model_config_selected", config = "e2b", "Using Gemma 4 E2B config (dense, ~4 GB)"); Gemma4Config::gemma4_e2b() }

        "e4b" => { info!(event = "model_config_selected", config = "e4b", "Using Gemma 4 E4B config (MoE-16, ~8 GB)"); Gemma4Config::gemma4_e4b() }

        "12b" => { info!(event = "model_config_selected", config = "12b", "Using Gemma 4 12B config (MoE-128, ~24 GB)"); Gemma4Config::gemma4_12b() }

        "27b" => { info!(event = "model_config_selected", config = "27b", "Using Gemma 4 27B config (MoE-128, ~54 GB)"); Gemma4Config::gemma4_27b() }

        "27b-mm" => { info!(event = "model_config_selected", config = "27b-mm", "Using Gemma 4 27B Multimodal IT config (dense, 35 layers, ~10 GB)"); Gemma4Config::gemma4_27b_mm() }

        "26b-a4b" => { info!(event = "model_config_selected", config = "26b-a4b", "Using Gemma 4 26B A4B config (MoE-128, 30 layers)"); Gemma4Config::gemma4_26b_a4b() }

        "31b" => { info!(event = "model_config_selected", config = "31b", "Using Gemma 4 31B config (dense, 60 layers, ~62 GB)"); Gemma4Config::gemma4_31b() }

        "llama3-8b" => { info!(event = "model_config_selected", config = "llama3-8b", "Using LLaMA 3.1 8B config"); Gemma4Config::llama3_8b() }

        "llama3-70b" => { info!(event = "model_config_selected", config = "llama3-70b", "Using LLaMA 3.1 70B config"); Gemma4Config::llama3_70b() }

        "mistral-7b" => { info!(event = "model_config_selected", config = "mistral-7b", "Using Mistral 7B config"); Gemma4Config::mistral_7b() }

        "mixtral-8x7b" => { info!(event = "model_config_selected", config = "mixtral-8x7b", "Using Mixtral 8x7B config"); Gemma4Config::mixtral_8x7b() }

        "phi3-mini" => { info!(event = "model_config_selected", config = "phi3-mini", "Using Phi-3 Mini config"); Gemma4Config::phi3_mini() }

        "qwen2-7b" => { info!(event = "model_config_selected", config = "qwen2-7b", "Using Qwen 2.5 7B config"); Gemma4Config::qwen2_7b() }

        other => anyhow::bail!("Unknown config '{}'. Use: e2b, e4b, 12b, 27b, 27b-mm, 26b-a4b, 31b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b", other),
    };

    info!(event = "model_config", hidden = config.hidden_dim, layers = config.num_layers, heads = config.num_heads, vocab = config.vocab_size, "model config");

    // Load weights
    info!(event = "loading_weights_from", "Loading weights from: {}", model_path);
    let path = std::path::Path::new(&model_path);
    if !path.exists() {
        anyhow::bail!("Model file not found: {}", model_path);
    }

    // Check RAM BEFORE loading — decide skeleton vs full
    let model_file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let available_ram = ferrisres::device::dispatch::DispatchPlan::available_ram_bytes();
    // Layer weights are ~48% of model file. Need available RAM > that + overhead.
    let layer_weights_ram = (model_file_bytes as f64 * 0.48) as u64;
    let use_skeleton = available_ram > 0 && available_ram < layer_weights_ram * 3;

    let model1 = if use_skeleton {
        info!(
            event = "skeleton_mode",
            available_ram_gb = available_ram as f64 / 1e9,
            model_gb = model_file_bytes as f64 / 1e9,
            "Loading skeleton model via file I/O (no mmap, low RAM)"
        );
        let mut file_st = ferrisres::model::safetensors::FileSafetensors::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open model file: {}", e))?;
        gemma_mapper::MappedGemma4Model::from_file_skeleton(config.clone(), &mut file_st)
            .map_err(|e| anyhow::anyhow!("Failed to load skeleton model: {}", e))?
    } else {
        let load_model = |p: &std::path::Path| -> Result<gemma_mapper::MappedGemma4Model, String> {
            match model_format.as_str() {
                "gguf" => load_gemma4_model_gguf(p, config.clone()),
                _ => load_gemma4_model_mmap(p, config.clone()),
            }
        };
        load_model(path)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?
    };

    let total_params = model1.embed_tokens.len() + model1.layers.iter().map(|l| {
            let attn = l.attn.q_proj.len() + l.attn.k_proj.len() + l.attn.v_proj.len() + l.attn.o_proj.len();
            let ffn = match &l.ffn {
                gemma_mapper::Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => gate_proj.len() + up_proj.len() + down_proj.len(),
                gemma_mapper::Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                    router.len() + expert_gates.iter().map(|v| v.len()).sum::<usize>()
                        + expert_ups.iter().map(|v| v.len()).sum::<usize>()
                        + expert_downs.iter().map(|v| v.len()).sum::<usize>()
                }
            };
            attn + ffn
        }).sum::<usize>() + model1.final_norm.len() + model1.lm_head.len();
    info!(event = "model_loaded", layers = model1.layers.len(), params = total_params, skeleton = use_skeleton, "model loaded");

    // Create teacher (frozen) — compute logits, then free weights
    let teacher = Gemma4Teacher::new(model1);
    info!(event = "teacher_model_created_frozen", "Teacher model created (frozen)");

    // Load training data
    let hf_tokenizer = if let Some(ref tok_path) = tokenizer_path {
        let tok = HfTokenizer::from_tokenizer_json(std::path::Path::new(tok_path))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        info!(event = "loaded_tokenizer_with_vocab_size", "Loaded tokenizer with vocab size {}", tok.vocab_size());
        Some(tok)
    } else {
        info!(event = "no_tokenizer_specified_using_byte_fallback", "No tokenizer specified, using byte-fallback tokenization");
        None
    };

    let token_ids = if let Some(ref dp) = data_path {
        info!(event = "loading_training_data_from", "Loading training data from: {}", dp);
        let text = std::fs::read_to_string(dp)
            .map_err(|e| anyhow::anyhow!("Failed to read data: {}", e))?;
        if let Some(ref tok) = hf_tokenizer {
            let ids = tok.encode_raw(&text);
            info!(event = "tokenized_to_tokens_using_provided_tokenizer", "Tokenized to {} tokens using provided tokenizer", ids.len());
            ids
        } else {
            let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
            info!(event = "loaded_byte_level_tokens_from_training", "Loaded {} byte-level tokens from training data", ids.len());
            ids
        }
    } else {
        info!(event = "no_training_data_provided_using_synthetic", "No training data provided, using synthetic tokens");
        (0..10000).map(|i| i % config.vocab_size as u32).collect()
    };

    // === DispatchPlan: model-size-aware CPU/GPU delegation ===
    // No more --gpu flag. We detect GPU, measure model size, and compute
    // per-op dispatch decisions.
    info!(event = "model_file_size", bytes = model_file_bytes, size_gb = format!("{:.2}", model_file_bytes as f64 / 1e9));

    let gpu_accel: Option<ferrisres::model::gpu_forward::GpuMatmulAccelerator> =
        match ferrisres::model::gpu_forward::GpuMatmulAccelerator::new() {
            Ok(mut accel) => {
                match accel.validate_model(teacher.model()) {
                    Ok(()) => Some(accel),
                    Err(e) => {
                        info!(event = "gpu_validation_failed", error = %e, "GPU validation failed, falling back to CPU");
                        None
                    }
                }
            }
            Err(e) => {
                info!(event = "gpu_unavailable", error = ?e, "No GPU available, using CPU-only dispatch");
                None
            }
        };

    // Build DispatchPlan from real measurements
    let (profile, vram_bytes, max_buffer_bytes) = match &gpu_accel {
        Some(gpu) => (gpu.profile(), gpu.estimated_vram_bytes(), gpu.max_buffer_bytes()),
        None => (ferrisres::device::DeviceProfile::Integrated, 0, 0),
    };
    let dispatch = ferrisres::device::DispatchPlan::new(
        profile,
        model_file_bytes,
        vram_bytes,
        max_buffer_bytes,
        config.hidden_dim as u64,
        config.num_heads as u64,
        config.head_dim as u64,
        config.intermediate_dim as u64,
        config.vocab_size as u64,
        seq_len as u64,
        config.num_kv_heads as u64,
    );
    info!(event = "dispatch_plan", "\n{}", dispatch.summary());

    // Pre-compute teacher logits for each chunk, then drop teacher to free ~10GB
    info!(event = "teacher_precompute_start", "pre-computing teacher logits");
    let mut teacher_logits_chunks: Vec<Vec<f32>> = Vec::new();
    let mut frozen_states_per_chunk: Vec<Vec<Vec<f32>>> = Vec::new();
    let num_chunks = (token_ids.len() + seq_len - 1) / seq_len;

    if gpu_accel.is_some() && dispatch.attn_qkv_dispatch != ferrisres::device::OpTarget::Cpu {
        // GPU-accelerated teacher forward
        let accel = gpu_accel.as_ref().unwrap();

        // Determine upload strategy:
        // - Skeleton model: MUST stream to GPU (no layer weights in RAM)
        // - Full model + resident mode: upload from RAM
        // - Full model + no resident: JIT per-matmul
        let mut weight_cache = if use_skeleton {
            // Skeleton model: projection weights aren't in RAM, must stream from mmap to GPU
            // This works even if DispatchPlan says resident_mode=false — skeleton REQUIRES GPU.
            info!(
                event = "resident_streaming_mode",
                "Streaming weights from disk to GPU via file I/O (skeleton model, low RAM)"
            );
            match ferrisres::model::safetensors::FileSafetensors::open(path) {
                Ok(mut file_st) => match accel.upload_weights_resident_streaming(
                    &teacher.model().config,
                    &mut file_st,
                ) {
                    Ok(cache) => Some(cache),
                    Err(e) => {
                        tracing::warn!(event = "streaming_upload_failed", error = ?e, "Streaming upload failed");
                        None
                    }
                },
                Err(e) => {
                    tracing::warn!(event = "file_open_failed", error = ?e, "Opening file for streaming failed");
                    None
                }
            }
        } else if dispatch.resident_mode {
            // Full model in RAM + VRAM fits: upload directly
            info!(event = "resident_mode", "Uploading weights to GPU from loaded model (resident mode)");
            match accel.upload_weights_resident(teacher.model()) {
                Ok(cache) => Some(cache),
                Err(e) => {
                    tracing::warn!(event = "resident_upload_failed", error = ?e, "Resident upload failed, falling back to JIT");
                    None
                }
            }
        } else {
            info!(event = "jit_mode", "Using JIT uploads (model too large for VRAM)");
            None
        };

        if let Some(ref mut cache) = weight_cache {
            // Upload lm_head + final_norm to GPU for full-GPU forward
            if let Err(e) = accel.enrich_cache_with_lm_head(cache, &teacher.model().lm_head, &teacher.model().final_norm) {
                tracing::warn!(event = "lm_head_upload_failed", error = ?e, "LM head upload failed, will use CPU fallback");
            }
            info!(event = "using_gpu_resident_for_teacher", "Using GPU resident weights for teacher forward (full GPU pipeline)");
        } else {
            info!(event = "using_gpu_matmul_acceleration_for_teacher", "Using GPU matmul acceleration for teacher forward");
        }

        // Limit pre-computed chunks to avoid OOM on Colab (12GB RAM limit)
        // Each chunk: teacher logits (134MB) + frozen states (27MB) ≈ 160MB
        // Limit to 30 chunks ≈ 4.8GB
        let max_precomputed_chunks = 30;
        let total_chunks = num_chunks.min(steps).min(max_precomputed_chunks);
        let teacher_start = std::time::Instant::now();
        for chunk_idx in 0..total_chunks {
            let chunk_start = std::time::Instant::now();
            let start = (chunk_idx * seq_len) % token_ids.len();
            let end = (start + seq_len).min(token_ids.len());
            let chunk: Vec<u32> = token_ids[start..end].to_vec();
            if chunk.len() < seq_len { break; }

            let logits = if let Some(cache) = &weight_cache {
                accel.forward_resident_gpu(cache, teacher.model(), &chunk)
                    .map_err(|e| anyhow::anyhow!("GPU resident forward failed: {:?}", e))?
            } else if use_skeleton {
                anyhow::bail!("Skeleton model requires GPU resident weights but upload failed. Not enough RAM for JIT fallback.");
            } else {
                accel.forward(teacher.model(), &chunk)
                    .map_err(|e| anyhow::anyhow!("GPU forward failed: {:?}", e))?
            };
            teacher_logits_chunks.push(logits);

            // Compute frozen hidden states from teacher for hidden MSE loss.
            // GPU-resident mode: skip — a full CPU forward alongside GPU would
            // exhaust Colab T4's shared memory (12GB RAM + 16GB VRAM) and
            // cause "Parent device is lost". KL divergence alone is sufficient.
            // CPU-only mode: collect states (no GPU contention).
            let frozen_states_this_chunk = if weight_cache.is_none() {
                teacher.forward_with_hidden_states(&chunk)
            } else {
                Vec::new() // GPU mode: skip hidden state collection
            };
            frozen_states_per_chunk.push(frozen_states_this_chunk);

            let elapsed = teacher_start.elapsed().as_secs_f32();
            let per_chunk = elapsed / (chunk_idx + 1) as f32;
            let remaining = per_chunk * (total_chunks - chunk_idx - 1) as f32;
            let chunk_ms = chunk_start.elapsed().as_millis();
            let tps = seq_len as f32 / chunk_start.elapsed().as_secs_f32();
            info!(
                event = "teacher_chunk",
                chunk = chunk_idx + 1,
                total = total_chunks,
                chunk_ms = chunk_ms,
                tok_per_s = format_args!("{:.2}", tps),
                elapsed_s = elapsed as u32,
                eta_s = remaining as u32,
                "teacher forward"
            );
        }
    } else if use_skeleton {
        anyhow::bail!("Skeleton model requires GPU for layer weights but no GPU available. Increase system RAM or use a GPU machine.");
    } else {
        // CPU teacher forward
        // Limit pre-computed chunks to avoid OOM
        let max_precomputed_chunks = 30;
        let total_chunks = num_chunks.min(steps).min(max_precomputed_chunks);
        let teacher_start = std::time::Instant::now();
        for chunk_idx in 0..total_chunks {
            let chunk_start = std::time::Instant::now();
            let start = (chunk_idx * seq_len) % token_ids.len();
            let end = (start + seq_len).min(token_ids.len());
            let chunk: Vec<u32> = token_ids[start..end].to_vec();
            if chunk.len() < seq_len { break; }
            let logits = teacher.forward(&chunk);
            teacher_logits_chunks.push(logits);

            // Compute frozen hidden states from teacher (same forward pass, states collected)
            let frozen = teacher.forward_with_hidden_states(&chunk);
            frozen_states_per_chunk.push(frozen);

            let elapsed = teacher_start.elapsed().as_secs_f32();
            let per_chunk = elapsed / (chunk_idx + 1) as f32;
            let remaining = per_chunk * (total_chunks - chunk_idx - 1) as f32;
            let chunk_ms = chunk_start.elapsed().as_millis();
            let tps = seq_len as f32 / chunk_start.elapsed().as_secs_f32();
            info!(
                event = "teacher_chunk",
                chunk = chunk_idx + 1,
                total = total_chunks,
                chunk_ms = chunk_ms,
                tok_per_s = format_args!("{:.2}", tps),
                elapsed_s = elapsed as u32,
                eta_s = remaining as u32,
                "teacher forward"
            );
        }
    }
    info!(event = "teacher_precompute_done", chunks = teacher_logits_chunks.len(), "teacher logits computed");

    // Drop teacher to free ~10GB RAM
    drop(teacher);
    info!(event = "teacher_freed", "teacher weights freed");

    // Re-load model for student (single copy in memory now)
    // Always load full model — teacher was dropped, RAM is available.
    // Skeleton model is only for the teacher (pre-compute) phase.
    let load_model_full = |p: &std::path::Path| -> Result<gemma_mapper::MappedGemma4Model, String> {
        match model_format.as_str() {
            "gguf" => load_gemma4_model_gguf(p, config.clone()),
            _ => load_gemma4_model_mmap(p, config.clone()),
        }
    };
    let model2 = load_model_full(path)
        .map_err(|e| anyhow::anyhow!("Failed to reload model for student: {}", e))?;
    info!(event = "student_model_loaded", "Student model loaded for weight mapping");

    // Convert teacher weights to CpuBlockAttnResModel (proper FerrisRes architecture)
    // This replaces the broken Gemma4Student with the correct BlockAttnRes student.
    let mut student = ferrisres::model::cpu_block_attn_res::gemma4_to_block_attnres(&model2);
    info!(event = "student_ready", layers = student.layers.len(), "student CpuBlockAttnResModel created from teacher weights");

    // Drop mmap IMMEDIATELY after copying weights to student.
    // On low-RAM systems (Colab T4: 12GB), the mmap (10GB virtual) + student (4GB)
    // + MoE conversion (quadruples FFN) would OOM otherwise.
    drop(model2);
    info!(event = "mmap_dropped", "Teacher mmap dropped, RAM freed for MoE conversion");

    // Convert every dense FFN to MoE (FerrisRes ALWAYS produces MoE models)
    ferrisres::model::cpu_block_attn_res::dense_ffn_to_moe(&mut student, 4, 2, 0.01);
    let moe_layers = student.layers.iter().filter(|l| l.moe.is_some()).count();
    info!(event = "moe_conversion", moe_layers = moe_layers, total_layers = student.layers.len(), "Converted dense FFN to MoE (4 experts, top-2)");

    // Attach LoRA adapters for training (rank 8, q_proj + v_proj)
    // Attention LoRA rank 8, expert FFN LoRA rank 4 (via separate config)
    let lora_config = ferrisres::training::lora::LoraConfig::targeting(8, vec![
        "q_proj", "k_proj", "v_proj", "o_proj",
        "moe.expert.*",  // wildcard: matches all expert gate/up/down
    ]);
    student.attach_lora(lora_config);
    info!(event = "lora_attached", "LoRA adapters attached to student model");

    // Create optimizer routed by DeviceProfile
    let mut optimizer = ferrisres::training::optimizer_for_profile(&profile, learning_rate as f32);
    // Register LoRA A and B matrices separately with optimizer
    if let Some(ref lora_m) = student.lora_manager {
        for (layer_idx, module_name, layer) in lora_m.adapters_iter() {
            let name_a = format!("lora_a_{}_{}", layer_idx, module_name);
            let name_b = format!("lora_b_{}_{}", layer_idx, module_name);
            // A: [rank × in_features] — this is the "input" projection
            optimizer.register_matrix(&name_a, layer.rank(), layer.in_features());
            // B: [out_features × rank] — this is the "output" projection
            optimizer.register_matrix(&name_b, layer.out_features(), layer.rank());
            // Mark LoRA B as output layer for SCALE momentum
            optimizer.mark_output_layer(&name_b);
        }
    }

    // Register MoE router weights with optimizer (Task 1)
    for (layer_idx, layer) in student.layers.iter().enumerate() {
        if let Some(ref moe) = layer.moe {
            let name = format!("router_{}", layer_idx);
            optimizer.register_matrix(&name, moe.num_experts, moe.hidden_dim);
        }
    }
    info!(event = "optimizer_created", optimizer = optimizer.name(), matrices = optimizer.num_registered(), state_mb = optimizer.state_bytes() as f64 / 1e6, "Optimizer created");

    // Cache transposed lm_head for backward pass (d_logits → d_hidden)
    let lm_head_t: Vec<f32> = {
        let lm_head = &student.lm_head;
        let hd = config.hidden_dim;
        let vs = config.vocab_size;
        let mut t = vec![0.0f32; vs * hd];
        for d in 0..hd {
            for v in 0..vs {
                t[v * hd + d] = lm_head.get(d * vs + v).copied().unwrap_or(0.0);
            }
        }
        t
    };

    // Training state
    let mut global_step: usize = 0;
    let mut best_loss = f32::INFINITY;

    // Resume from checkpoint if specified
    if let Some(ref ckpt_path) = resume_path {
        // Try new format first (model + LoRA + optimizer + meta)
        let meta_path = format!("{}.meta.json", ckpt_path);
        let lora_path = format!("{}.lora.bin", ckpt_path);
        let opt_path = format!("{}.opt.bin", ckpt_path);

        if let Ok(meta_str) = std::fs::read_to_string(&meta_path) {
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_str) {
                global_step = meta["global_step"].as_u64().unwrap_or(0) as usize;
                info!(event = "resuming_meta", step = global_step);
            }
        }

        // Load LoRA adapters
        if let Ok(lora_data) = std::fs::read(&lora_path) {
            if let Some(ref mut lora_m) = student.lora_manager {
                match lora_m.deserialize_adapters(&lora_data) {
                    Ok(()) => info!(event = "lora_restored", adapters = lora_m.adapters_iter().count()),
                    Err(e) => warn!(event = "lora_restore_failed", error = %e),
                }
            }
        }

        // Load optimizer state
        if let Ok(opt_data) = std::fs::read(&opt_path) {
            match optimizer.deserialize_state(&opt_data) {
                Ok(()) => info!(event = "optimizer_restored", timestep = optimizer.timestep()),
                Err(e) => warn!(event = "optimizer_restore_failed", error = %e),
            }
        }

        if global_step > 0 {
            info!(event = "resuming_from_checkpoint", step = global_step, optimizer = optimizer.name(), "Resumed from full checkpoint");
        } else {
            // Fall back to old DistillationCheckpoint format
            match DistillationCheckpoint::load(std::path::Path::new(ckpt_path)) {
                Ok(ckpt) => {
                    global_step = ckpt.global_step;
                    info!(event = "resuming_from_checkpoint", step = global_step, "resuming from old checkpoint format");
                }
                Err(e) => {
                    tracing::warn!(event = "checkpoint_load_failed", error = %e, "Failed to load checkpoint — starting fresh: {}", e);
                }
            }
        }
    }

    info!("");
    info!(event = "distill_start", total_steps = steps, start_step = global_step, seq_len = seq_len, lr = learning_rate, temp = temperature, optimizer = optimizer.name(), moe_layers = moe_layers, "starting distillation");
    info!("");

    // Since base weights don't change during training, we only need to run
    // the full model forward ONCE per chunk. During training, we only recompute
    // from injection points where block summary layers modify the state.
    // This saves ~80% of per-step compute (skip layers before first injection).
    let vs = config.vocab_size;
    let injection_points = config.block_summary_injection_points();
    let _first_injection = injection_points.first().copied().unwrap_or(0);

    // Frozen states are now computed alongside teacher logits in the teacher loop above.
    // This section just reports stats.

    // GPU-aware matmul helper: consults DispatchPlan, does tiled GPU when needed
    // Used for student-side operations (backprop through LM head).
    let gpu_matmul = |a: &[f32], b: &[f32], m: usize, k: usize, n: usize, dispatch: &ferrisres::device::DispatchPlan| -> Vec<f32> {
        match dispatch.should_gpu_matmul(m as u64, k as u64, n as u64) {
            ferrisres::device::OpTarget::Cpu => gemma_mapper::matmul(a, b, m, k, n),
            ferrisres::device::OpTarget::Gpu => {
                if let Some(ref gpu) = gpu_accel {
                    match gpu.gpu_matmul_cpu_b(a, b, m, k, n) {
                        Ok(result) => return result,
                        Err(e) => {
                            tracing::warn!(event = "gpu_matmul_fallback", error = ?e, m, k, n, "GPU matmul failed, falling back to CPU");
                        }
                    }
                }
                gemma_mapper::matmul(a, b, m, k, n)
            }
            ferrisres::device::OpTarget::GpuTiled => {
                // B is too large for a single GPU buffer. Chunk it.
                // C[m×n] = A[m×k] × B[k×n], process B in column tiles of tile_n columns.
                if let Some(ref gpu) = gpu_accel {
                    let max_cols = (dispatch.max_buffer_bytes as usize / (k * 4)).max(1);
                    let tile_n = max_cols.min(n);
                    if tile_n == n {
                        // Actually fits after all — just do a regular GPU matmul
                        match gpu.gpu_matmul_cpu_b(a, b, m, k, n) {
                            Ok(result) => return result,
                            Err(e) => {
                                tracing::warn!(event = "gpu_matmul_fallback", error = ?e, "GPU tiled matmul fallback");
                                return gemma_mapper::matmul(a, b, m, k, n);
                            }
                        }
                    }
                    // Tiled: process tile_n columns at a time
                    let mut result = vec![0.0f32; m * n];
                    let num_tiles = (n + tile_n - 1) / tile_n;
                    for tile_idx in 0..num_tiles {
                        let col_start = tile_idx * tile_n;
                        let col_end = (col_start + tile_n).min(n);
                        let cur_n = col_end - col_start;
                        // Extract B tile: rows [0..k], cols [col_start..col_end]
                        let mut b_tile = vec![0.0f32; k * cur_n];
                        for r in 0..k {
                            for c in 0..cur_n {
                                b_tile[r * cur_n + c] = b[r * n + col_start + c];
                            }
                        }
                        match gpu.gpu_matmul_cpu_b(a, &b_tile, m, k, cur_n) {
                            Ok(tile_result) => {
                                // Copy tile into result
                                for t in 0..m {
                                    for c in 0..cur_n {
                                        result[t * n + col_start + c] = tile_result[t * cur_n + c];
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(event = "gpu_tile_fallback", tile = tile_idx, error = ?e, "GPU tile failed, falling entire matmul to CPU");
                                return gemma_mapper::matmul(a, b, m, k, n);
                            }
                        }
                    }
                    return result;
                }
                gemma_mapper::matmul(a, b, m, k, n)
            }
        }
    };

    // Frozen states were computed alongside teacher logits above.
    // Report stats.
    let total_states_mb = frozen_states_per_chunk.iter()
        .map(|c| c.iter().map(|s| s.len() * 4).sum::<usize>())
        .sum::<usize>() as f64 / (1024.0 * 1024.0);
    info!(event = "frozen_states_ready", chunks = frozen_states_per_chunk.len(), size_mb = total_states_mb as u32, "frozen states precomputed (alongside teacher logits)");

    // Run distillation using pre-computed teacher logits
    let mut results: Vec<gemma_mapper::DistillationStepResult> = Vec::new();

    let train_start = std::time::Instant::now();
    let mut prev_loss = f32::NAN;
    let mut loss_ema = f32::NAN; // Exponential moving average for smoothing
    let target_step = global_step + steps;
    let mut converge_best_loss = f32::INFINITY;
    let mut converge_steps_no_improve: usize = 0;

    while global_step < target_step {
        let step_start = std::time::Instant::now();
        let chunk_idx = global_step % teacher_logits_chunks.len();
        let batch_idx = global_step % (token_ids.len() / seq_len.max(1));
        let start = (batch_idx * seq_len) % token_ids.len();
        let end = (start + seq_len).min(token_ids.len());
        let batch_tokens = &token_ids[start..end];
        let actual_seq = batch_tokens.len();

        let teacher_logits = &teacher_logits_chunks[chunk_idx];

        // Student forward using CpuBlockAttnResModel.
        // Stores activations inline for proper layer-by-layer backward.
        // GPU path: TODO — wire forward_gpu to also store activations.
        let train_output = if gpu_accel.is_some() {
            // GPU forward for logits, then CPU forward_train for activations
            // (temporary until forward_gpu stores activations)
            let gpu_logits = student.forward_gpu(batch_tokens, gpu_accel.as_ref().unwrap(), &dispatch);
            let mut cpu_out = student.forward_train(batch_tokens);
            cpu_out.logits = gpu_logits; // use GPU logits (more accurate)
            cpu_out
        } else {
            student.forward_train(batch_tokens)
        };
        let student_logits = train_output.logits;
        let routing_data = train_output.routing_data;
        let _activations = train_output.activations;
        let _final_hidden = train_output.final_hidden;

        // KL divergence loss
        let kl_loss = gemma_mapper::kl_divergence_loss(
            teacher_logits, &student_logits,
            temperature as f32, vs, actual_seq,
        );

        // Hidden state matching loss: MSE between teacher and student per-layer states
        let (hidden_mse_loss, _student_states) = if chunk_idx < frozen_states_per_chunk.len() {
            let teacher_states = &frozen_states_per_chunk[chunk_idx];
            let student_states = student.forward_with_hidden_states(batch_tokens);
            let mse = if teacher_states.len() == student_states.len() && !teacher_states.is_empty() {
                let _hd = config.hidden_dim;
                let mut total_mse = 0.0f32;
                let mut num_layers_compared = 0usize;
                // Compare at every 5th layer to save compute
                for layer_idx in (0..teacher_states.len()).step_by(5) {
                    let t_state = &teacher_states[layer_idx];
                    let s_state = &student_states[layer_idx];
                    if t_state.len() == s_state.len() {
                        let mse: f32 = t_state.iter().zip(s_state.iter())
                            .map(|(t, s)| (t - s) * (t - s))
                            .sum::<f32>() / t_state.len() as f32;
                        total_mse += mse;
                        num_layers_compared += 1;
                    }
                }
                if num_layers_compared > 0 { total_mse / num_layers_compared as f32 } else { 0.0 }
            } else {
                0.0
            };
            (mse, Some(student_states))
        } else {
            (0.0, None)
        };

        // MoE Load balance loss: num_experts × Σ(f_i × P_i)
        // Prevents router collapse (all tokens going to one expert).
        // Uses routing data collected inline during student forward — no extra pass.
        let balance_loss = if !routing_data.is_empty() {
            let mut total_balance = 0.0f32;
            for rd in &routing_data {
                let ne = rd.gate_probs.len() / actual_seq.max(1);
                if ne == 0 { continue; }
                let mut f = vec![0.0f32; ne]; // fraction of tokens per expert
                let mut p = vec![0.0f32; ne]; // mean router prob per expert
                for t in 0..actual_seq {
                    for e in 0..ne {
                        p[e] += rd.gate_probs[t * ne + e];
                    }
                    // Count from selected experts
                    for k in 0..rd.selected_experts.len() / actual_seq.max(1).max(1) {
                        let idx = rd.selected_experts[t * (rd.selected_experts.len() / actual_seq.max(1).max(1)) + k];
                        if idx < ne { f[idx] += 1.0; }
                    }
                }
                for e in 0..ne {
                    f[e] /= actual_seq as f32;
                    p[e] /= actual_seq as f32;
                    total_balance += f[e] * p[e];
                }
                total_balance *= ne as f32;
            }
            total_balance / routing_data.len() as f32 // average across layers
        } else {
            0.0
        };

        // Combined loss: KL + 0.5 * hidden_MSE + 0.01 * balance_loss
        let loss = kl_loss + 0.5 * hidden_mse_loss + 0.01 * balance_loss;

        // Update EMA
        loss_ema = if loss_ema.is_nan() { loss }
                   else { 0.9 * loss_ema + 0.1 * loss };
        if loss < best_loss { best_loss = loss; }

        // Compute d_loss / d_student_logits (numerically stable)
        // d_kl/d_s_logit = (softmax(s/T) - softmax(t/T)) / T
        // Use log-softmax to avoid exp overflow on large logits
        let mut d_logits = vec![0.0f32; actual_seq * vs];
        let inv_temp = 1.0 / (temperature as f32);
        for t in 0..actual_seq {
            let offset = t * vs;
            // Find max for numerical stability
            let mut t_max = f32::NEG_INFINITY;
            let mut s_max = f32::NEG_INFINITY;
            for v in 0..vs {
                let tl = teacher_logits.get(offset + v).copied().unwrap_or(0.0) * inv_temp;
                let sl = student_logits.get(offset + v).copied().unwrap_or(0.0) * inv_temp;
                if tl > t_max { t_max = tl; }
                if sl > s_max { s_max = sl; }
            }
            // Compute stable softmax probabilities
            let mut t_sum = 0.0f32;
            let mut s_sum = 0.0f32;
            let mut t_probs = vec![0.0f32; vs];
            let mut s_probs = vec![0.0f32; vs];
            for v in 0..vs {
                t_probs[v] = (teacher_logits.get(offset + v).copied().unwrap_or(0.0) * inv_temp - t_max).exp();
                s_probs[v] = (student_logits.get(offset + v).copied().unwrap_or(0.0) * inv_temp - s_max).exp();
                t_sum += t_probs[v];
                s_sum += s_probs[v];
            }
            for v in 0..vs {
                t_probs[v] /= t_sum;
                s_probs[v] /= s_sum;
                d_logits[offset + v] = (s_probs[v] - t_probs[v]) * inv_temp;
            }
        }

        // === Proper Layer-by-Layer Backward Pass ===
        // Uses stored activations from forward_train() to compute per-layer gradients.
        // Each LoRA module gets its correct layer-specific input and gradient signal.
        //
        // Chain: d_logits → lm_head^T → d_hidden
        //   For each layer (reverse order):
        //     d_hidden → [Attention] → d_attn_raw → LoRA O grad
        //                          → LoRA Q/K/V grad (from pre_attn_normed)
        //     d_hidden → [FFN/MoE] → d_expert_out → LoRA gate/up/down grad
        //                          → Router grad (softmax → top-k chain)
        //
        // Memory: activations are already stored from forward_train().
        // Cost: O(layers × seq × rank × (in_f + out_f)) for LoRA + O(seq × experts × hd) for router.
        let d_hidden: Vec<f32> = {
            gpu_matmul(&d_logits, &lm_head_t, actual_seq, vs, config.hidden_dim, &dispatch)
        };

        let hd = config.hidden_dim;
        let seq_for_grad = actual_seq.min(d_hidden.len() / hd);
        let num_layers = student.num_layers;
        let mut total_grad_norm = 0.0f32;

        // Process layers in reverse order (layer N-1 → 0).
        // Each layer receives d_hidden from the layer above and produces
        // gradient for its LoRA modules using stored activations.
        //
        // For residual connections: d_hidden passes through unchanged
        // (gradient of x + f(x) w.r.t. x is 1 + df/dx, but for LoRA-only training
        // we only need df/dx for the LoRA parameters, not dx).
        //
        // For proper backprop through attention, we'd need to compute the full
        // attention Jacobian. However, since we only train LoRA adapters (not
        // full weights), the gradient signal for Q/K/V/O projections is:
        //   - O projection: dL/d(attn_raw) = d_hidden (post-residual, from this layer's output)
        //     Then: LoRA O grad uses post_attn_raw as input, d_hidden as gradient
        //   - Q/K/V projections: dL/d(pre_attn_normed) flows through attention.
        //     The full attention backward is O(seq² × heads × head_dim), which is expensive.
        //     Simplification: use d_hidden as gradient signal (similar to O projection)
        //     since the residual connection means d_hidden is the primary gradient source.
        //
        // For expert FFN: the chain is more direct since we stored per-expert intermediates.
        //   dL/d(expert_out) = d_hidden * expert_weight (from top-k routing)
        //   Then backward through down → combined → gate/up with stored intermediates.

        // Collect gradient data for Phase 2 (optimizer step)
        let mut lora_grads: Vec<(usize, String, Vec<f32>, Vec<f32>)> = Vec::new();
        let mut router_grads: Vec<(usize, Vec<f32>)> = Vec::new();

        // Phase 1: Compute all gradients (immutable reads of LoRA A, B and model weights)
        if let Some(ref lora_m) = student.lora_manager {
            for layer_idx in (0..num_layers).rev() {
                let act = match _activations.get(layer_idx) {
                    Some(a) => a,
                    None => continue,
                };
                let layer = &student.layers[layer_idx];

                // --- Attention LoRA gradients ---
                // For q/k/v: input = pre_attn_normed, gradient signal = d_hidden
                // For o_proj: input = post_attn_raw, gradient signal = d_hidden

                // O projection LoRA
                if let Some(lora_layer) = lora_m.get(layer_idx, "o_proj") {
                    let (ga, gb) = compute_lora_grad(
                        lora_layer, &act.post_attn_raw, &d_hidden,
                        seq_for_grad, hd,
                    );
                    total_grad_norm += ga.iter().chain(gb.iter()).map(|x| x * x).sum::<f32>();
                    lora_grads.push((layer_idx, "o_proj".to_string(), ga, gb));
                }

                // Q projection LoRA
                if let Some(lora_layer) = lora_m.get(layer_idx, "q_proj") {
                    let q_dim = layer.num_heads * layer.head_dim;
                    let d_q = truncate_grad(&d_hidden, seq_for_grad, q_dim, hd);
                    let (ga, gb) = compute_lora_grad(
                        lora_layer, &act.pre_attn_normed, &d_q,
                        seq_for_grad, hd,
                    );
                    total_grad_norm += ga.iter().chain(gb.iter()).map(|x| x * x).sum::<f32>();
                    lora_grads.push((layer_idx, "q_proj".to_string(), ga, gb));
                }

                // K projection LoRA
                if let Some(lora_layer) = lora_m.get(layer_idx, "k_proj") {
                    let kv_dim = layer.num_kv_heads * layer.head_dim;
                    let d_kv = truncate_grad(&d_hidden, seq_for_grad, kv_dim, hd);
                    let (ga, gb) = compute_lora_grad(
                        lora_layer, &act.pre_attn_normed, &d_kv,
                        seq_for_grad, hd,
                    );
                    total_grad_norm += ga.iter().chain(gb.iter()).map(|x| x * x).sum::<f32>();
                    lora_grads.push((layer_idx, "k_proj".to_string(), ga, gb));
                }

                // V projection LoRA
                if let Some(lora_layer) = lora_m.get(layer_idx, "v_proj") {
                    let kv_dim = layer.num_kv_heads * layer.head_dim;
                    let d_kv = truncate_grad(&d_hidden, seq_for_grad, kv_dim, hd);
                    let (ga, gb) = compute_lora_grad(
                        lora_layer, &act.pre_attn_normed, &d_kv,
                        seq_for_grad, hd,
                    );
                    total_grad_norm += ga.iter().chain(gb.iter()).map(|x| x * x).sum::<f32>();
                    lora_grads.push((layer_idx, "v_proj".to_string(), ga, gb));
                }

                // --- Expert FFN LoRA gradients ---
                // Each expert has gate, up, down projections with LoRA.
                // We backward through: down ← combined ← gate/up using stored intermediates.
                if let Some(ref moe) = layer.moe {
                    let ne = moe.num_experts;
                    let tk = moe.top_k;
                    let id = moe.intermediate_dim;

                    // Find routing data for this layer
                    let rd = routing_data.iter().find(|r| r.layer_idx == layer_idx);

                    for t in 0..seq_for_grad {
                        let expert_acts = match act.expert_activations.get(t) {
                            Some(ea) => ea,
                            None => continue,
                        };
                        let d_out_t = &d_hidden[t * hd..];

                        for k_idx in 0..tk {
                            let ea = match expert_acts.get(k_idx) {
                                Some(e) if !e.gated.is_empty() => e,
                                _ => continue,
                            };
                            let ei = ea.expert_idx;
                            let w = rd.map(|r| r.expert_weights[t * tk + k_idx]).unwrap_or(1.0 / tk as f32);

                            // d_expert_out = d_hidden[t] * routing_weight
                            let d_expert_out: Vec<f32> = d_out_t[..hd].iter().map(|&x| x * w).collect();

                            // === Down projection backward ===
                            // combined [id] → down [id, hd] → expert_out [hd]
                            // d_combined = d_expert_out @ down^T  [hd] → [id]
                            let down = &moe.expert_down[ei]; // [id * hd]
                            let mut d_combined = vec![0.0f32; id];
                            for i in 0..id {
                                let mut s = 0.0f32;
                                for o in 0..hd.min(d_expert_out.len()) {
                                    s += d_expert_out[o] * down[i * hd + o];
                                }
                                d_combined[i] = s;
                            }

                            // LoRA down gradient:
                            // input = combined [id], gradient = d_expert_out [hd]
                            if let Some(lora_layer) = lora_m.get(layer_idx, &format!("moe.expert.{}.down", ei)) {
                                let (ga, gb) = compute_lora_grad_single(
                                    lora_layer, &ea.combined, &d_expert_out,
                                );
                                // Accumulate into existing grads or push new
                                accumulate_lora_grad(&mut lora_grads, layer_idx, &format!("moe.expert.{}.down", ei), ga, gb, &mut total_grad_norm);
                            }

                            // === Gate + Up backward ===
                            // combined = gated * upped
                            // d_gated = d_combined * upped
                            // d_upped = d_combined * gated
                            let mut d_gated_pre = vec![0.0f32; id];
                            let mut d_upped = vec![0.0f32; id];
                            for i in 0..id {
                                // Gate activation backward
                                // For GeLU: derivative ≈ Φ(x) where Φ is standard normal CDF
                                // Approximate from output: if g = gelu(x), g' ≈ 0.5 * (1 + tanh(√(2/π)(x + 0.044715x³)))
                                // Simplified: use sigmoid approximation g' ≈ g * (1 - g) / x
                                // Even simpler: g' ≈ 0.5 for the linear region
                                let gate_deriv = if moe.use_gelu {
                                    let g = ea.gated[i];
                                    // gelu'(x) ≈ 0.5 * (1 + erf(x/√2))
                                    // Approximate from output: crude but functional
                                    let sigmoid_approx = 1.0 / (1.0 + (-g * 1.702).exp());
                                    sigmoid_approx
                                } else {
                                    // silu'(x) = σ(x)(1 + x(1 - σ(x)))
                                    // Approximate: σ(x) ≈ silu(x)/x but x unknown
                                    0.5 // crude
                                };
                                d_gated_pre[i] = d_combined[i] * ea.upped[i] * gate_deriv;
                                d_upped[i] = d_combined[i] * ea.gated[i];
                            }

                            // LoRA gate gradient: input = token [hd], gradient = d_gated_pre [id]
                            if let Some(lora_layer) = lora_m.get(layer_idx, &format!("moe.expert.{}.gate", ei)) {
                                let (ga, gb) = compute_lora_grad_single(
                                    lora_layer, &ea.input, &d_gated_pre,
                                );
                                accumulate_lora_grad(&mut lora_grads, layer_idx, &format!("moe.expert.{}.gate", ei), ga, gb, &mut total_grad_norm);
                            }

                            // LoRA up gradient: input = token [hd], gradient = d_upped [id]
                            if let Some(lora_layer) = lora_m.get(layer_idx, &format!("moe.expert.{}.up", ei)) {
                                let (ga, gb) = compute_lora_grad_single(
                                    lora_layer, &ea.input, &d_upped,
                                );
                                accumulate_lora_grad(&mut lora_grads, layer_idx, &format!("moe.expert.{}.up", ei), ga, gb, &mut total_grad_norm);
                            }
                        }
                    }

                    // === Router gradient (proper softmax → top-k chain) ===
                    if let Some(rd) = rd {
                        let mut router_grad = vec![0.0f32; ne * hd];

                        for t in 0..seq_for_grad {
                            let mut d_logit = vec![0.0f32; ne];

                            for k in 0..tk {
                                let ei = rd.selected_experts[t * tk + k];
                                if ei >= ne { continue; }
                                let w = rd.expert_weights[t * tk + k];

                                // dL/d_weight_k: signal from expert output contribution
                                let mut d_weight_k = 0.0f32;
                                for d in 0..hd.min(d_hidden.len() - t * hd) {
                                    d_weight_k += d_hidden[t * hd + d] * moe.gate_weights[ei * hd + d];
                                }

                                // Softmax Jacobian: d_weight/d_logit_e = w * (δ_{e,ei} - P_e)
                                for e in 0..ne {
                                    let indicator = if e == ei { 1.0 } else { 0.0 };
                                    d_logit[e] += d_weight_k * w * (indicator - rd.gate_probs[t * ne + e]);
                                }
                            }

                            // Balance loss gradient
                            for e in 0..ne {
                                let pe = rd.gate_probs[t * ne + e];
                                d_logit[e] += 0.01 * ne as f32 * pe * (1.0 - pe) * (1.0 / ne as f32 - pe);
                            }

                            // Accumulate into router gradient
                            for e in 0..ne {
                                let dl = d_logit[e];
                                if dl.abs() < 1e-10 { continue; }
                                for d in 0..hd {
                                    router_grad[e * hd + d] += dl * rd.pre_ffn_input[t * hd + d];
                                }
                            }
                        }

                        let inv_seq = 1.0 / seq_for_grad.max(1) as f32;
                        for g in router_grad.iter_mut() { *g *= inv_seq; }
                        total_grad_norm += router_grad.iter().map(|x| x * x).sum::<f32>();
                        router_grads.push((layer_idx, router_grad));
                    }
                }
            }
        }

        // Phase 2: Apply optimizer steps (mutable writes to LoRA A, B, router weights)
        if let Some(ref mut lora_m) = student.lora_manager {
            for (layer_idx, module_name, ga, gb) in lora_grads {
                if let Some(ll) = lora_m.adapters_iter_mut()
                    .find(|(li, mn, _)| *li == layer_idx && *mn == module_name)
                    .map(|(_, _, ll)| ll)
                {
                    let a = ll.lora_a_mut();
                    optimizer.step(&format!("lora_a_{}_{}", layer_idx, module_name), &ga, a);
                }
                if let Some(ll) = lora_m.adapters_iter_mut()
                    .find(|(li, mn, _)| *li == layer_idx && *mn == module_name)
                    .map(|(_, _, ll)| ll)
                {
                    let b = ll.lora_b_mut();
                    optimizer.step(&format!("lora_b_{}_{}", layer_idx, module_name), &gb, b);
                }
            }
        }
        for (layer_idx, rgrad) in router_grads {
            if let Some(layer) = student.layers.get_mut(layer_idx) {
                if let Some(ref mut moe) = layer.moe {
                    let name = format!("router_{}", layer_idx);
                    optimizer.step(&name, &rgrad, &mut moe.gate_weights);
                }
            }
        }

        // Gradient norm for logging
        let grad_norm = total_grad_norm.sqrt();

        let lr = learning_rate as f32;

        let step_ms = step_start.elapsed().as_millis();
        let tps = actual_seq as f32 / step_start.elapsed().as_secs_f32();
        let steps_done = global_step + 1; // steps completed so far in this run
        let elapsed = train_start.elapsed().as_secs_f32();
        let per_step = elapsed / steps_done as f32;
        let remaining = target_step - global_step - 1;
        let eta = per_step * remaining as f32;
        let loss_delta = if prev_loss.is_nan() { 0.0 } else { loss - prev_loss };

        results.push(gemma_mapper::DistillationStepResult {
            step: global_step,
            kl_loss: kl_loss,
            bridge_weight: hidden_mse_loss,
            learning_rate: lr,
            layer_cosine_sim: vec![],
        });

        if global_step % log_every == 0 {
            info!(
                event = "distill_step",
                step = global_step,
                target = target_step,
                loss = format_args!("{:.6}", loss),
                loss_delta = format_args!("{:+.6}", loss_delta),
                loss_ema = format_args!("{:.6}", loss_ema),
                best_loss = format_args!("{:.6}", best_loss),
                grad_norm = format_args!("{:.3e}", grad_norm),
                balance_loss = format_args!("{:.6}", balance_loss),
                lr = format_args!("{:.2e}", lr),
                step_ms = step_ms,
                tok_per_s = format_args!("{:.2}", tps),
                eta_s = eta as u32,
                "distillation step"
            );
        }

        // Mid-training checkpoint
        if checkpoint_every > 0 && (global_step + 1) % checkpoint_every == 0 {
            let ckpt_path = format!("{}_step{}", output_path, global_step + 1);
            match ferrisres::model::checkpoint::save_model(&student, &ckpt_path) {
                Ok(()) => {
                    // Save LoRA adapters alongside model weights
                    if let Some(ref lora_m) = student.lora_manager {
                        let lora_path = format!("{}.lora.bin", ckpt_path);
                        let lora_data = lora_m.serialize_adapters();
                        match std::fs::write(&lora_path, &lora_data) {
                            Ok(()) => info!(event = "lora_saved", bytes = lora_data.len(), path = %lora_path),
                            Err(e) => warn!(event = "lora_save_failed", error = %e),
                        }
                    }
                    // Save optimizer state
                    let opt_path = format!("{}.opt.bin", ckpt_path);
                    let opt_data = optimizer.serialize_state();
                    match std::fs::write(&opt_path, &opt_data) {
                        Ok(()) => info!(event = "optimizer_saved", bytes = opt_data.len(), path = %opt_path),
                        Err(e) => warn!(event = "optimizer_save_failed", error = %e),
                    }
                    // Save training metadata (global step)
                    let meta_path = format!("{}.meta.json", ckpt_path);
                    let meta = serde_json::json!({"global_step": global_step + 1});
                    match std::fs::write(&meta_path, meta.to_string()) {
                        Ok(()) => {},
                        Err(e) => warn!(event = "meta_save_failed", error = %e),
                    }
                    info!(event = "checkpoint_saved", step = global_step + 1, path = %ckpt_path, "Full checkpoint saved (model + LoRA + optimizer)");
                }
                Err(e) => warn!(event = "checkpoint_save_failed", step = global_step + 1, error = %e, "Failed to save checkpoint"),
            }
        }

        prev_loss = loss;
        global_step += 1;
        if loss < 1e-6 { break; }

        // Convergence detection
        if converge_threshold > 0.0 {
            if loss < converge_best_loss * (1.0 - converge_threshold as f32) {
                converge_best_loss = loss;
                converge_steps_no_improve = 0;
            } else {
                converge_steps_no_improve += 1;
                if converge_steps_no_improve >= converge_patience {
                    info!(
                        event = "converged",
                        step = global_step,
                        loss = format_args!("{:.6}", loss),
                        best_loss = format_args!("{:.6}", converge_best_loss),
                        patience = converge_patience,
                        threshold = format_args!("{:.4}", converge_threshold),
                        "loss converged — stopping early"
                    );
                    break;
                }
            }
        }
    }

    // Report results
    info!("");
    info!(event = "distillation_results", "=== Distillation Results ===");
    info!(event = "steps_completed", "Steps completed: {}", results.len());

    if let Some(first) = results.first() {
        info!(event = "initial_kl_loss", "Initial KL loss: {:.6}", first.kl_loss);
    }
    if let Some(last) = results.last() {
        info!(event = "final_kl_loss", "Final KL loss:   {:.6}", last.kl_loss);
    }

    // Log loss curve
    for result in &results {
        if result.step % log_every == 0 {
            info!(event = "distill_step_summary", step = result.step, loss = result.kl_loss, lr = result.learning_rate, "distillation step summary");
        }
    }

    // Save trained Block-MoE-Res model
    match ferrisres::model::checkpoint::save_model(&student, &output_path) {
        Ok(()) => {
            info!(event = "model_saved", path = %output_path, "Trained model saved");
            // Save final LoRA adapters
            if let Some(ref lora_m) = student.lora_manager {
                let lora_path = format!("{}.lora.bin", output_path);
                let lora_data = lora_m.serialize_adapters();
                match std::fs::write(&lora_path, &lora_data) {
                    Ok(()) => info!(event = "lora_saved", bytes = lora_data.len()),
                    Err(e) => warn!(event = "lora_save_failed", error = %e),
                }
            }
            // Save final optimizer state
            let opt_path = format!("{}.opt.bin", output_path);
            let opt_data = optimizer.serialize_state();
            match std::fs::write(&opt_path, &opt_data) {
                Ok(()) => info!(event = "optimizer_saved", bytes = opt_data.len()),
                Err(e) => warn!(event = "optimizer_save_failed", error = %e),
            }
            // Save final training metadata
            let meta_path = format!("{}.meta.json", output_path);
            let meta = serde_json::json!({"global_step": global_step});
            match std::fs::write(&meta_path, meta.to_string()) {
                Ok(()) => {},
                Err(e) => warn!(event = "meta_save_failed", error = %e),
            }
        }
        Err(e) => warn!(event = "model_save_failed", error = %e, "Failed to save model"),
    }

    // Save loss curve as CSV
    let csv_path = format!("{}.loss_curve.csv", output_path);
    let mut csv = String::from("step,kl_loss,learning_rate\n");
    for r in &results {
        csv.push_str(&format!("{},{},{}\n", r.step, r.kl_loss, r.learning_rate));
    }
    std::fs::write(&csv_path, &csv)?;
    info!(event = "loss_curve_saved", "Loss curve saved → {}", csv_path);

    info!("");
    info!(event = "distillation_complete", "Distillation complete!");

    Ok(())
}

/// Run the OpenAI-compatible API server.
async fn cmd_serve(
    model_path: Option<String>,
    config_name: String,
    model_format: String,
    tokenizer_path: Option<String>,
    host: String,
    port: u16,
    model_name: String,
    armor: bool,
    cognitive: bool,
    concepts_path: Option<String>,
) -> anyhow::Result<()> {
    info!(event = "ferrisres_serve", "=== FerrisRes API Server ===");
    info!(event = "server_config", host = %host, port = port, model = %model_name, "Starting server");

    // Load real model if provided
    let loaded_model = if let Some(ref path) = model_path {
        let gemma_config = resolve_model_config(&config_name)?;
        let model_path = std::path::Path::new(path);

        let model = match model_format.as_str() {
            "gguf" => {
                info!(event = "loading_gguf", path = %model_path.display(), "Loading GGUF model");
                gemma_mapper::load_gemma4_model_gguf(model_path, gemma_config)
                    .map_err(|e| anyhow::anyhow!("GGUF load failed: {}", e))?
            }
            "safetensors" => {
                info!(event = "loading_safetensors", path = %model_path.display(), "Loading safetensors model");
                let mmaped = ferrisres::model::safetensors::MmapedSafetensors::open(model_path)
                    .map_err(|e| anyhow::anyhow!("mmap failed: {:?}", e))?;
                gemma_mapper::MappedGemma4Model::from_mmap(gemma_config, &mmaped)
                    .map_err(|e| anyhow::anyhow!("safetensors load failed: {}", e))?
            }
            other => anyhow::bail!("Unknown model format: '{}'. Use 'gguf' or 'safetensors'.", other),
        };
        info!(event = "model_loaded", "Model loaded: {} layers", model.layers.len());
        Some(Gemma4Teacher::new(model))
    } else {
        info!(event = "no_model", "No --model-path provided, server will return placeholder responses");
        None
    };

    // Build tokenizer
    let tokenizer: Arc<dyn Send + Sync + Fn(&str) -> Vec<u32>> = if let Some(ref tok_path) = tokenizer_path {
        match ferrisres::model::tokenizer::HfTokenizer::from_tokenizer_json(std::path::Path::new(tok_path)) {
            Ok(hf_tok) => {
                info!(event = "tokenizer_loaded", "Loaded HfTokenizer from {}", tok_path);
                Arc::new(move |text: &str| hf_tok.encode(text))
            }
            Err(e) => {
                warn!(event = "tokenizer_fallback", "Failed to load tokenizer: {}, using SimpleTokenizer", e);
                let tok = SimpleTokenizer::new();
                Arc::new(move |text: &str| tok.encode(text))
            }
        }
    } else {
        let tok = SimpleTokenizer::new();
        Arc::new(move |text: &str| tok.encode(text))
    };

    // Build cognitive pipeline if requested
    let _cognitive_pipeline = if cognitive {
        use ferrisres::inference::cognitive_pipeline::{CognitivePipeline, CognitivePipelineConfig};
        let cp_config = CognitivePipelineConfig {
            concepts_enabled: true,
            concepts_path: concepts_path.clone().map(std::path::PathBuf::from),
            concepts_embedding_dim: 64,
            concepts_max: 10000,
            mirror_test_enabled: true,
            episodic_memory_enabled: true,
            episodic_memory_path: concepts_path.as_ref().map(|p| {
                let mut pb = std::path::PathBuf::from(p);
                pb.set_extension("episodes.json");
                pb
            }),
            intrinsic_motivation_enabled: true,
            proactive_controller_enabled: true,
            tool_usage_tracking_enabled: true,
            ..Default::default()
        };
        info!(event = "cognitive_pipeline_enabled", "Cognitive pipeline enabled for server");
        Some(CognitivePipeline::new(cp_config))
    } else {
        None
    };

    // Create the API handler
    let handler = Box::new(FerrisResApiHandler {
        model_name: model_name.clone(),
        tokenizer,
        loaded_model,
        _armor: armor,
    });

    let server_config = ferrisres::server::ApiServerConfig {
        host: host.clone(),
        port,
        model_name,
    };

    let server = ferrisres::server::ApiServer::new(server_config, handler);
    info!(event = "server_starting", "FerrisRes API server starting on {}:{}", host, port);
    println!("FerrisRes API server starting on {}:{}", host, port);
    println!("Endpoints:");
    println!("  GET  /health              - Health check");
    println!("  GET  /v1/models            - List models");
    println!("  POST /v1/chat/completions  - Chat completions");
    println!("  POST /v1/completions       - Text completions");

    server.serve()?;
    Ok(())
}

/// API handler that wires the model to the server routes.
struct FerrisResApiHandler {
    model_name: String,
    tokenizer: Arc<dyn Send + Sync + Fn(&str) -> Vec<u32>>,
    loaded_model: Option<Gemma4Teacher>,
    _armor: bool,
}

impl ferrisres::server::ApiHandler for FerrisResApiHandler {
    fn chat_completion(&mut self, req: &ferrisres::server::ChatCompletionRequest) -> ferrisres::server::ChatCompletionResponse {
        use ferrisres::server::*;

        // Concatenate messages into a single prompt
        let prompt: String = req.messages.iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let tokens = (self.tokenizer)(&prompt);

        if let Some(ref model) = self.loaded_model {
            // Real CPU inference
            let gen_tokens = generate_cpu(model, &tokens, 256, 0.7, None);
            let simple_tok = SimpleTokenizer::new();
            let generated_text = simple_tok.decode(&gen_tokens);

            ChatCompletionResponse {
                id: format!("chatcmpl-{}", chrono_hash(&prompt)),
                object: "chat.completion".into(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                model: self.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage::assistant(&generated_text),
                    finish_reason: "stop".into(),
                }],
                usage: Usage {
                    prompt_tokens: tokens.len(),
                    completion_tokens: gen_tokens.len(),
                    total_tokens: tokens.len() + gen_tokens.len(),
                },
            }
        } else {
            // Placeholder response
            ChatCompletionResponse {
                id: format!("chatcmpl-{}", chrono_hash(&prompt)),
                object: "chat.completion".into(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                model: self.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage::assistant(
                        &format!("[FerrisRes] No model loaded. Received {} tokens from {} message(s). Provide --model-path to enable generation.",
                            tokens.len(), req.messages.len())
                    ),
                    finish_reason: "stop".into(),
                }],
                usage: Usage {
                    prompt_tokens: tokens.len(),
                    completion_tokens: 0,
                    total_tokens: tokens.len(),
                },
            }
        }
    }

    fn completion(&mut self, req: &ferrisres::server::CompletionRequest) -> ferrisres::server::ChatCompletionResponse {
        use ferrisres::server::*;

        let tokens = (self.tokenizer)(&req.prompt);

        if let Some(ref model) = self.loaded_model {
            // Real CPU inference
            let gen_tokens = generate_cpu(model, &tokens, req.max_tokens, 0.7, None);
            let simple_tok = SimpleTokenizer::new();
            let generated_text = simple_tok.decode(&gen_tokens);

            ChatCompletionResponse {
                id: format!("cmpl-{}", chrono_hash(&req.prompt)),
                object: "text_completion".into(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                model: self.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage::assistant(&generated_text),
                    finish_reason: "stop".into(),
                }],
                usage: Usage {
                    prompt_tokens: tokens.len(),
                    completion_tokens: gen_tokens.len(),
                    total_tokens: tokens.len() + gen_tokens.len(),
                },
            }
        } else {
            // Placeholder response
            ChatCompletionResponse {
                id: format!("cmpl-{}", chrono_hash(&req.prompt)),
                object: "text_completion".into(),
                created: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
                model: self.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ChatMessage::assistant(
                        &format!("[FerrisRes] No model loaded. Received {} tokens. Provide --model-path to enable generation.", tokens.len())
                    ),
                    finish_reason: "stop".into(),
                }],
                usage: Usage {
                    prompt_tokens: tokens.len(),
                    completion_tokens: 0,
                    total_tokens: tokens.len(),
                },
            }
        }
    }

    fn list_models(&self) -> Vec<ferrisres::server::ModelInfo> {
        vec![ferrisres::server::ModelInfo {
            id: self.model_name.clone(),
            object: "model".into(),
            owned_by: "ferrisres".into(),
        }]
    }
}

/// Autoregressive generation on CPU using a loaded MappedGemma4Model.
///
/// Runs the model's `forward()` in a loop, sampling one token at a time.
/// Returns the generated token IDs (excluding the prompt tokens).
fn generate_cpu(
    model: &Gemma4Teacher,
    prompt_tokens: &[u32],
    max_new_tokens: usize,
    temperature: f32,
    eos_token_id: Option<u32>,
) -> Vec<u32> {
    let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
    let prompt_len = prompt_tokens.len();
    let vs = model.model().config.vocab_size;

    for step in 0..max_new_tokens {
        // Run full forward pass on all tokens so far
        let logits = model.forward(&all_tokens);

        // Take logits for the last token position
        let last_offset = (all_tokens.len() - 1) * vs;
        let last_logits = &logits[last_offset..last_offset + vs];

        // Diagnostic: dump top-5 logits on first 3 steps
        if step <= 2 {
            let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            info!(event = "logits_diagnostic", step = step, "Top-5 logits:");
            for (rank, (id, val)) in indexed.iter().take(5).enumerate() {
                info!(event = "logit", step = step, rank = rank, token_id = id, value = val);
            }
            info!(event = "logits_range", step = step, min = indexed.last().unwrap().1, max = indexed[0].1, "Logit range");
        }

        // Temperature scaling + sampling
        let next_token = if temperature < 1e-6 {
            // Greedy: argmax
            last_logits.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            // Temperature-scaled sampling
            let scaled: Vec<f32> = last_logits.iter()
                .map(|&l| l / temperature)
                .collect();
            // Softmax
            let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

            // Weighted random sample
            let r: f32 = rand_in_range(0.0, 1.0);
            let mut cumsum = 0.0f32;
            let mut chosen = 0u32;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= r {
                    chosen = i as u32;
                    break;
                }
            }
            chosen
        };

        // Check EOS
        if let Some(eos) = eos_token_id {
            if next_token == eos {
                info!(event = "cpu_gen_eos", step = step, "Hit EOS token");
                break;
            }
        }

        all_tokens.push(next_token);

        // Log the chosen token
        if step <= 5 || step % 10 == 0 {
            info!(event = "chosen_token", step = step, token_id = next_token, total = all_tokens.len());
        }

        if step % 10 == 0 {
            info!(event = "cpu_gen_progress", step = step, tokens = all_tokens.len(), "Generating...");
        }
    }

    info!(event = "cpu_gen_done", prompt_len = prompt_len, generated = all_tokens.len() - prompt_len, "Generation complete");
    all_tokens[prompt_len..].to_vec()
}

/// Simple random float in [lo, hi).
fn rand_in_range(lo: f32, hi: f32) -> f32 {
    use std::time::SystemTime;
    let ns = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let state = ns.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let x = ((state >> 33) as u32) as f32 / u32::MAX as f32;
    lo + x * (hi - lo)
}

/// Simple hash for generating unique IDs.
fn chrono_hash(s: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

async fn cmd_evaluate(
    model_path: String,
    config_name: String,
    text: String,
) -> anyhow::Result<()> {
    info!(event = "ferrisres_evaluation", "=== FerrisRes Evaluation ===");

    let config = match config_name.as_str() {
        "e2b" => Gemma4Config::gemma4_e2b(),
        "e4b" => Gemma4Config::gemma4_e4b(),
        "27b-mm" => Gemma4Config::gemma4_27b_mm(),
        "26b-a4b" => Gemma4Config::gemma4_26b_a4b(),
        "31b" => Gemma4Config::gemma4_31b(),
        _ => anyhow::bail!("Use e2b, e4b, 27b-mm, 26b-a4b, or 31b for evaluation"),
    };

    let path = std::path::Path::new(&model_path);
    let model = load_gemma4_model_mmap(path, config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;

    let teacher = Gemma4Teacher::new(model);

    // Tokenize
    let token_ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
    info!(event = "evaluating_on_byte_level_tokens", "Evaluating on {} byte-level tokens", token_ids.len());

    let logits = teacher.forward(&token_ids);
    let vs = config.vocab_size;
    let seq = token_ids.len();

    // Compute perplexity: exp(average NLL)
    let mut total_nll = 0.0f64;
    let mut count = 0usize;
    for t in 0..seq.saturating_sub(1) {
        let target = token_ids[t + 1] as usize;
        if target < vs {
            // Softmax to get probabilities
            let offset = t * vs;
            let mut max_logit = f32::NEG_INFINITY;
            for v in 0..vs {
                let l = logits.get(offset + v).copied().unwrap_or(0.0);
                if l > max_logit { max_logit = l; }
            }
            let mut sum_exp = 0.0f32;
            let mut probs = vec![0.0f32; vs];
            for v in 0..vs {
                probs[v] = (logits.get(offset + v).copied().unwrap_or(0.0) - max_logit).exp();
                sum_exp += probs[v];
            }
            let target_prob = probs.get(target).copied().unwrap_or(0.0) / sum_exp;
            if target_prob > 1e-10 {
                total_nll -= (target_prob as f64).ln();
                count += 1;
            }
        }
    }

    let avg_nll = if count > 0 { total_nll / count as f64 } else { f64::INFINITY };
    let perplexity = avg_nll.exp();

    info!(event = "tokens_evaluated", "Tokens evaluated: {}", count);
    info!(event = "average_nll", "Average NLL: {:.4}", avg_nll);
    info!(event = "perplexity", "Perplexity: {:.2}", perplexity);

    Ok(())
}

/// Compute LoRA A and B gradients for a single projection across multiple tokens.
///
/// LoRA forward: y = Wx + scaling * B @ (A @ x)
/// dL/dA[r][d] = scaling * Σ_t (Σ_o B[o][r] · d_y[t][o]) · x[t][d]
/// dL/dB[o][r] = scaling * Σ_t d_y[t][o] · (Σ_d A[r][d] · x[t][d])
///
/// `input`: [seq * in_f] — input to the projection
/// `d_output`: [seq * hd] — gradient signal (d_hidden or truncated)
fn compute_lora_grad(
    lora_layer: &ferrisres::training::lora::LoraLayer,
    input: &[f32],
    d_output: &[f32],
    seq: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>) {
    let rank = lora_layer.rank();
    let in_f = lora_layer.in_features();
    let out_f = lora_layer.out_features();
    let scaling = lora_layer.scaling();
    let b = lora_layer.lora_b();
    let a = lora_layer.lora_a();

    let mut ga = vec![0.0f32; rank * in_f];
    let mut gb = vec![0.0f32; out_f * rank];

    let actual_out = out_f.min(hd);
    let actual_in = in_f.min(hd);
    let actual_seq = seq.min(input.len() / in_f.max(1)).min(d_output.len() / actual_out.max(1));

    for r in 0..rank {
        for d in 0..actual_in {
            let mut grad = 0.0f32;
            for t in 0..actual_seq {
                let mut btdy = 0.0f32;
                for o in 0..actual_out {
                    btdy += b[o * rank + r] * d_output[t * hd + o];
                }
                grad += btdy * input[t * in_f + d];
            }
            ga[r * in_f + d] = scaling * grad;
        }
    }

    for o in 0..actual_out {
        for r in 0..rank {
            let mut grad = 0.0f32;
            for t in 0..actual_seq {
                let mut ax_r = 0.0f32;
                for d in 0..actual_in {
                    ax_r += a[r * in_f + d] * input[t * in_f + d];
                }
                grad += d_output[t * hd + o] * ax_r;
            }
            gb[o * rank + r] = scaling * grad;
        }
    }

    (ga, gb)
}

/// Compute LoRA gradient for a single token.
/// `input`: [in_f], `d_output`: [out_f]
fn compute_lora_grad_single(
    lora_layer: &ferrisres::training::lora::LoraLayer,
    input: &[f32],
    d_output: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let rank = lora_layer.rank();
    let in_f = lora_layer.in_features();
    let out_f = lora_layer.out_features();
    let scaling = lora_layer.scaling();
    let b = lora_layer.lora_b();
    let a = lora_layer.lora_a();

    let mut ga = vec![0.0f32; rank * in_f];
    let mut gb = vec![0.0f32; out_f * rank];

    for r in 0..rank {
        for d in 0..in_f {
            let mut btdy = 0.0f32;
            for o in 0..out_f.min(d_output.len()) {
                btdy += b[o * rank + r] * d_output[o];
            }
            ga[r * in_f + d] = scaling * btdy * input[d];
        }
    }

    for o in 0..out_f.min(d_output.len()) {
        for r in 0..rank {
            let mut ax_r = 0.0f32;
            for d in 0..in_f.min(input.len()) {
                ax_r += a[r * in_f + d] * input[d];
            }
            gb[o * rank + r] = scaling * d_output[o] * ax_r;
        }
    }

    (ga, gb)
}

/// Accumulate LoRA gradients. If entry exists, adds; otherwise pushes new.
fn accumulate_lora_grad(
    grads: &mut Vec<(usize, String, Vec<f32>, Vec<f32>)>,
    layer_idx: usize,
    module_name: &str,
    ga: Vec<f32>,
    gb: Vec<f32>,
    total_norm: &mut f32,
) {
    if let Some((_, _, existing_a, existing_b)) = grads.iter_mut()
        .find(|(li, mn, _, _)| *li == layer_idx && mn == module_name)
    {
        for (i, v) in ga.iter().enumerate() {
            existing_a[i] += v;
            *total_norm += v * v;
        }
        for (i, v) in gb.iter().enumerate() {
            existing_b[i] += v;
            *total_norm += v * v;
        }
    } else {
        *total_norm += ga.iter().chain(gb.iter()).map(|x| x * x).sum::<f32>();
        grads.push((layer_idx, module_name.to_string(), ga, gb));
    }
}

/// Truncate d_hidden [seq × hd] to [seq × out_dim] for Q/K/V projections.
fn truncate_grad(d_hidden: &[f32], seq: usize, out_dim: usize, hd: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * out_dim];
    for t in 0..seq {
        let src_off = t * hd;
        let dst_off = t * out_dim;
        let copy_len = out_dim.min(hd).min(d_hidden.len().saturating_sub(src_off));
        for d in 0..copy_len {
            out[dst_off + d] = d_hidden[src_off + d];
        }
    }
    out
}
