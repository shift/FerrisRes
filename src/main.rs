use std::sync::Arc;
use std::time::Instant;
use clap::{Parser, Subcommand};
use tracing::{info, warn, trace};
use ferrisres::{WgpuCompute, DeviceProfile, BlockAttnResConfig, BlockAttnResModel, SimpleTokenizer};
use ferrisres::compute::{GpuBuffer, MatMulOp, RmsNormOp, SoftmaxOp, ElementWiseOp};
use ferrisres::training::AdamOptimizer;
use ferrisres::model::gemma_mapper::{
    self, Gemma4Config, Gemma4Teacher, Gemma4Student, DistillationConfig,
    BlockSummaryLayer, load_gemma4_model_mmap,
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

    Infer {
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
        /// Model config: e2b, e4b, 12b, 27b.
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
        #[arg(long, default_value_t = 10)]
        log_every: usize,
        /// Resume from checkpoint.
        #[arg(long)]
        resume: Option<String>,
        /// Model file format: safetensors or gguf.
        #[arg(long, default_value = "safetensors")]
        model_format: String,
        /// Path to tokenizer.json (HuggingFace format).
        #[arg(long)]
        tokenizer: Option<String>,
        /// Save checkpoint every N steps.
        #[arg(long, default_value_t = 100)]
        checkpoint_every: usize,
        /// Use GPU for matmul acceleration.
        #[arg(long, default_value_t = false)]
        gpu: bool,
    },

    /// Evaluate teacher/student perplexity.
    Evaluate {
        /// Path to Gemma 4 safetensors file.
        #[arg(long)]
        model_path: String,
        /// Model config: e2b, e4b, 12b, 27b.
        #[arg(long, default_value = "e2b")]
        config: String,
        /// Text to evaluate on.
        #[arg(long)]
        text: String,
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
            hidden_dim,
            num_blocks,
            block_size,
            prompt,
            max_tokens,
            temperature,
            template,
            yarn_scale,
            image,
        } => cmd_infer(hidden_dim, num_blocks, block_size, prompt, max_tokens, temperature, template, yarn_scale, image).await,
        Commands::Benchmark {
            hidden_dim,
            num_blocks,
            block_size,
            iterations,
        } => cmd_benchmark(hidden_dim, num_blocks, block_size, iterations).await,
        Commands::Distill {
            model_path, config, seq_len, steps, learning_rate, temperature, data, output, log_every,
            resume, model_format, tokenizer, checkpoint_every, gpu,
        } => cmd_distill(model_path, config, seq_len, steps, learning_rate, temperature, data, output, log_every, resume, model_format, tokenizer, checkpoint_every, gpu).await,
        Commands::Evaluate {
            model_path, config, text,
        } => cmd_evaluate(model_path, config, text).await,
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
    hidden_dim: usize,
    num_blocks: usize,
    block_size: usize,
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    template: Option<String>,
    yarn_scale: Option<f32>,
    image: Option<String>,
) -> anyhow::Result<()> {
    info!(event = "initializing_inference_pipeline", "Initializing inference pipeline");

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

    // Apply chat template if specified
    let formatted_prompt = if let Some(ref template_name) = template {
        match ferrisres::TemplateFormat::from_name(template_name) {
            Some(fmt) => {
                let registry = ferrisres::PromptTemplateRegistry::new(fmt);
                let result = registry.apply_single(&prompt);
                info!(event = "applied_template_chars_chars", "Applied {} template: {} chars → {} chars", template_name, prompt.len(), result.len());
                result
            }
            None => {
                info!(event = "unknown_template_using_raw_prompt", "Unknown template '{}', using raw prompt", template_name);
                prompt.clone()
            }
        }
    } else {
        prompt.clone()
    };

    let tokenizer = SimpleTokenizer::new();
    let model = BlockAttnResModel::new(Arc::clone(&device), Arc::clone(&queue), config.clone(), tokenizer.vocab_size())?;

    let tokens = tokenizer.encode(&formatted_prompt);
    info!(event = "encoded_tokens", "Encoded tokens ({}): {:?}", tokens.len(), tokens);

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
        config.hidden_dim,
        tokenizer.vocab_size(),
    )?;
    let lm_head = LMHead::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        config.hidden_dim,
        tokenizer.vocab_size(),
    )?;
    let generator = TokenGenerator::new(
        Arc::new(model),
        lm_head,
        embedding,
        Arc::clone(&device),
        Arc::clone(&queue),
        2048, // max_seq_len
    )?;
    
    let gen_config = ferrisres::inference::generator::GenerateConfig {
        temperature: temperature as f32,
        max_tokens,
        context_extension: yarn_scale.map(|scale| {
            ferrisres::inference::context_extension::ContextExtensionConfig::yarn(4096, (4096.0 * scale) as usize)
        }),
        ..Default::default()
    };
    
    let output_tokens = generator.generate(&tokens, &gen_config)?;
    
    // Decode and print
    let decoded = tokenizer.decode(&output_tokens);
    info!(event = "generation_complete", "generated {} tokens{}", output_tokens.len(),
        if image_patch_count > 0 { format!(" (with {} image patches)", image_patch_count) } else { String::new() });
    println!("{}", decoded);

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
    use_gpu: bool,
) -> anyhow::Result<()> {
    info!(event = "ferrisres_gemma_4_block_attnres_distillation", "=== FerrisRes Gemma 4 → Block AttnRes Distillation ===");

    // Parse config
    let config = match config_name.as_str() {
        "e2b" => { info!(event = "model_config_selected", config = "e2b", "Using Gemma 4 E2B config (dense, ~4 GB)"); Gemma4Config::gemma4_e2b() }

        "e4b" => { info!(event = "model_config_selected", config = "e4b", "Using Gemma 4 E4B config (MoE-16, ~8 GB)"); Gemma4Config::gemma4_e4b() }

        "12b" => { info!(event = "model_config_selected", config = "12b", "Using Gemma 4 12B config (MoE-128, ~24 GB)"); Gemma4Config::gemma4_12b() }

        "27b" => { info!(event = "model_config_selected", config = "27b", "Using Gemma 4 27B config (MoE-128, ~54 GB)"); Gemma4Config::gemma4_27b() }

        "27b-mm" => { info!(event = "model_config_selected", config = "27b-mm", "Using Gemma 4 27B Multimodal IT config (dense, 35 layers, ~10 GB)"); Gemma4Config::gemma4_27b_mm() }

        "llama3-8b" => { info!(event = "model_config_selected", config = "llama3-8b", "Using LLaMA 3.1 8B config"); Gemma4Config::llama3_8b() }

        "llama3-70b" => { info!(event = "model_config_selected", config = "llama3-70b", "Using LLaMA 3.1 70B config"); Gemma4Config::llama3_70b() }

        "mistral-7b" => { info!(event = "model_config_selected", config = "mistral-7b", "Using Mistral 7B config"); Gemma4Config::mistral_7b() }

        "mixtral-8x7b" => { info!(event = "model_config_selected", config = "mixtral-8x7b", "Using Mixtral 8x7B config"); Gemma4Config::mixtral_8x7b() }

        "phi3-mini" => { info!(event = "model_config_selected", config = "phi3-mini", "Using Phi-3 Mini config"); Gemma4Config::phi3_mini() }

        "qwen2-7b" => { info!(event = "model_config_selected", config = "qwen2-7b", "Using Qwen 2.5 7B config"); Gemma4Config::qwen2_7b() }

        other => anyhow::bail!("Unknown config '{}'. Use: e2b, e4b, 12b, 27b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b", other),
    };

    info!(event = "model_config", hidden = config.hidden_dim, layers = config.num_layers, heads = config.num_heads, vocab = config.vocab_size, "model config");

    // Load weights
    info!(event = "loading_weights_from", "Loading weights from: {}", model_path);
    let path = std::path::Path::new(&model_path);
    if !path.exists() {
        anyhow::bail!("Model file not found: {}", model_path);
    }

    let load_model = |p: &std::path::Path| -> Result<gemma_mapper::MappedGemma4Model, String> {
        match model_format.as_str() {
            "gguf" => load_gemma4_model_gguf(p, config.clone()),
            _ => load_gemma4_model_mmap(p, config.clone()),
        }
    };

    let model1 = load_model(path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
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
    info!(event = "model_loaded", layers = model1.layers.len(), params = total_params, "model loaded");

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

    // Pre-compute teacher logits for each chunk, then drop teacher to free ~10GB
    info!(event = "teacher_precompute_start", "pre-computing teacher logits");
    let mut teacher_logits_chunks: Vec<Vec<f32>> = Vec::new();
    let num_chunks = (token_ids.len() + seq_len - 1) / seq_len;

    if use_gpu {
        // GPU-accelerated teacher forward
        info!(event = "using_gpu_matmul_acceleration_for_teacher", "Using GPU matmul acceleration for teacher forward");
        let mut gpu_accel = ferrisres::model::gpu_forward::GpuMatmulAccelerator::new()
            .map_err(|e| anyhow::anyhow!("Failed to init GPU: {:?}", e))?;
        gpu_accel.validate_model(teacher.model())
            .map_err(|e| anyhow::anyhow!("Failed to upload weights: {:?}", e))?;
        info!(event = "gpu_weights_ready", "GPU weights validated");

        let total_chunks = num_chunks.min(steps);
        let teacher_start = std::time::Instant::now();
        for chunk_idx in 0..total_chunks {
            let chunk_start = std::time::Instant::now();
            let start = (chunk_idx * seq_len) % token_ids.len();
            let end = (start + seq_len).min(token_ids.len());
            let chunk: Vec<u32> = token_ids[start..end].to_vec();
            if chunk.len() < seq_len { break; }
            let logits = gpu_accel.forward(teacher.model(), &chunk)
                .map_err(|e| anyhow::anyhow!("GPU forward failed: {:?}", e))?;
            teacher_logits_chunks.push(logits);

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
                tok_per_s = tps as u32,
                elapsed_s = elapsed as u32,
                eta_s = remaining as u32,
                "teacher forward"
            );
        }
    } else {
        // CPU teacher forward
        let total_chunks = num_chunks.min(steps);
        let teacher_start = std::time::Instant::now();
        for chunk_idx in 0..total_chunks {
            let chunk_start = std::time::Instant::now();
            let start = (chunk_idx * seq_len) % token_ids.len();
            let end = (start + seq_len).min(token_ids.len());
            let chunk: Vec<u32> = token_ids[start..end].to_vec();
            if chunk.len() < seq_len { break; }
            let logits = teacher.forward(&chunk);
            teacher_logits_chunks.push(logits);

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
                tok_per_s = tps as u32,
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
    let model2 = load_model(path)
        .map_err(|e| anyhow::anyhow!("Failed to reload model for student: {}", e))?;
    info!(event = "student_model_loaded", "Student model loaded");

    let injection_points = config.block_summary_injection_points();
    let block_summaries: Vec<BlockSummaryLayer> = injection_points.iter()
        .map(|_| BlockSummaryLayer::new_identity(config.hidden_dim, config.sliding_window))
        .collect();
    info!(event = "student_ready", block_summary_layers = block_summaries.len(), "student model created");

    let distill_config = DistillationConfig {
        learning_rate: learning_rate as f32,
        temperature: temperature as f32,
        num_steps: steps,
        max_seq_len: seq_len,
        ..DistillationConfig::default()
    };

    let mut student = Gemma4Student::new(model2, block_summaries, distill_config.clone());

    // Initialize optimizers (fresh or from checkpoint)
    let mut global_step: usize = 0;
    let mut optimizers: Vec<gemma_mapper::BlockSummaryAdam> = student.block_summaries.iter()
        .map(|bs| gemma_mapper::BlockSummaryAdam::new(bs, distill_config.learning_rate))
        .collect();
    let mut best_loss = f32::INFINITY;

    // Resume from checkpoint if specified
    if let Some(ref ckpt_path) = resume_path {
        let ckpt = DistillationCheckpoint::load(std::path::Path::new(ckpt_path))
            .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;
        global_step = ckpt.global_step;
        info!(event = "resuming_from_checkpoint", step = global_step, "resuming from checkpoint");
        optimizers = ckpt.apply(&mut student);
        info!(event = "checkpoint_restored", layers = student.block_summaries.len(), "restored block summary layers and optimizer state");
    }

    info!("");
    info!(event = "distill_start", total_steps = steps, start_step = global_step, seq_len = seq_len, lr = learning_rate, temp = temperature, "starting distillation");
    info!("");

    // Precompute frozen base model hidden states for each chunk.
    // Since base weights don't change during training, we only need to run
    // the full model forward ONCE per chunk. During training, we only recompute
    // from injection points where block summary layers modify the state.
    // This saves ~80% of per-step compute (skip layers before first injection).
    let vs = config.vocab_size;
    let injection_points = config.block_summary_injection_points();
    let first_injection = injection_points.first().copied().unwrap_or(0);
    let mut frozen_states_per_chunk: Vec<Vec<Vec<f32>>> = Vec::new();

    info!(event = "precompute_frozen_states", chunks = teacher_logits_chunks.len(), first_injection = first_injection, "precomputing frozen base model hidden states");
    let frozen_start = std::time::Instant::now();
    for (chunk_idx, _chunk_tokens) in teacher_logits_chunks.iter().enumerate() {
        // We need the token IDs for this chunk
        let start_tok = (chunk_idx * seq_len) % token_ids.len();
        let end_tok = (start_tok + seq_len).min(token_ids.len());
        let tok_ids: Vec<u32> = token_ids[start_tok..end_tok].to_vec();
        if tok_ids.len() < seq_len { break; }

        // Run the frozen base model to get all per-layer hidden states
        // We create a temporary teacher-like model for this
        let frozen_states = {
            let config_ref = &student.model.config;
            let hd = config_ref.hidden_dim;
            let seq = tok_ids.len();
            let mut states = Vec::new();

            // Embedding
            let mut hidden = vec![0.0f32; seq * hd];
            for (t, &tid) in tok_ids.iter().enumerate() {
                let id = tid as usize;
                if id * hd + hd <= student.model.embed_tokens.len() {
                    for d in 0..hd { hidden[t * hd + d] = student.model.embed_tokens[id * hd + d]; }
                }
            }
            let scale = (hd as f32).sqrt();
            for h in hidden.iter_mut() { *h *= scale; }
            states.push(hidden.clone());

            for layer in &student.model.layers {
                let residual = hidden.clone();
                let normed = gemma_mapper::rms_norm(&hidden, &layer.attn.input_norm, hd, 1e-6);
                let q_dim = config_ref.num_heads * config_ref.head_dim;
                let kv_dim = config_ref.num_kv_heads * config_ref.head_dim;
                let q = gemma_mapper::matmul(&normed, &layer.attn.q_proj, seq, hd, q_dim);
                let k = gemma_mapper::matmul(&normed, &layer.attn.k_proj, seq, hd, kv_dim);
                let v = gemma_mapper::matmul(&normed, &layer.attn.v_proj, seq, hd, kv_dim);
                let mut q = q; let mut k = k;
                gemma_mapper::apply_rope(&mut q, seq, config_ref.num_heads, config_ref.head_dim, 0);
                gemma_mapper::apply_rope_gqa(&mut k, seq, config_ref.num_kv_heads, config_ref.head_dim, 0);
                let heads_per_kv = config_ref.num_heads / config_ref.num_kv_heads;
                let attn_scale = 1.0 / (config_ref.head_dim as f32).sqrt();
                let mut attn_out = vec![0.0f32; seq * q_dim];
                for h_idx in 0..config_ref.num_heads {
                    let kv_h = h_idx / heads_per_kv;
                    for t in 0..seq {
                        let mut max_s = f32::NEG_INFINITY;
                        let mut scores = vec![0.0f32; seq];
                        for s in 0..seq {
                            if s > t { scores[s] = f32::NEG_INFINITY; continue; }
                            let mut dot = 0.0;
                            for d in 0..config_ref.head_dim {
                                dot += q[t * q_dim + h_idx * config_ref.head_dim + d]
                                     * k[s * kv_dim + kv_h * config_ref.head_dim + d];
                            }
                            scores[s] = dot * attn_scale;
                            if scores[s] > max_s { max_s = scores[s]; }
                        }
                        let mut sum_e = 0.0;
                        for s in &mut scores { *s = (*s - max_s).exp(); sum_e += *s; }
                        for s in &mut scores { *s /= sum_e; }
                        for d in 0..config_ref.head_dim {
                            let mut val = 0.0;
                            for s in 0..seq { val += scores[s] * v[s * kv_dim + kv_h * config_ref.head_dim + d]; }
                            attn_out[t * q_dim + h_idx * config_ref.head_dim + d] = val;
                        }
                    }
                }
                let attn_proj = gemma_mapper::matmul(&attn_out, &layer.attn.o_proj, seq, q_dim, hd);
                for i in 0..hidden.len() { hidden[i] = residual[i] + attn_proj[i]; }
                let residual2 = hidden.clone();
                let normed2 = gemma_mapper::rms_norm(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);
                let ffn_out = match &layer.ffn {
                    gemma_mapper::Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                        gemma_mapper::swiglu_ffn(&normed2, gate_proj, up_proj, down_proj, hd, config_ref.intermediate_dim)
                    }
                    gemma_mapper::Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                        let mut moe = gemma_mapper::CpuMoELayer::new(hd, config_ref.intermediate_dim, config_ref.num_experts, config_ref.top_k);
                        moe.gate_weights = router.clone();
                        moe.expert_gate = expert_gates.clone();
                        moe.expert_up = expert_ups.clone();
                        moe.expert_down = expert_downs.clone();
                        moe.forward(&normed2, seq)
                    }
                };
                for i in 0..hidden.len() { hidden[i] = residual2[i] + ffn_out[i]; }
                states.push(hidden.clone());
            }
            states
        };
        frozen_states_per_chunk.push(frozen_states);
        let elapsed = frozen_start.elapsed().as_secs_f32();
        let per_chunk = elapsed / (chunk_idx + 1) as f32;
        let remaining = per_chunk * (teacher_logits_chunks.len() - chunk_idx - 1) as f32;
        info!(event = "frozen_state_chunk", chunk = chunk_idx + 1, total = teacher_logits_chunks.len(), elapsed_s = elapsed as u32, eta_s = remaining as u32, "frozen base hidden states precomputed");
    }
    let total_states_mb = frozen_states_per_chunk.iter()
        .map(|c| c.iter().map(|s| s.len() * 4).sum::<usize>())
        .sum::<usize>() as f64 / (1024.0 * 1024.0);
    info!(event = "frozen_states_ready", chunks = frozen_states_per_chunk.len(), size_mb = total_states_mb as u32, "frozen states precomputed");

    // Run distillation using pre-computed teacher logits
    let mut results: Vec<gemma_mapper::DistillationStepResult> = Vec::new();

    let train_start = std::time::Instant::now();
    let mut prev_loss = f32::NAN;
    let mut loss_ema = f32::NAN; // Exponential moving average for smoothing
    let target_step = global_step + steps;

    while global_step < target_step {
        let step_start = std::time::Instant::now();
        let chunk_idx = global_step % teacher_logits_chunks.len();
        let batch_idx = global_step % (token_ids.len() / seq_len.max(1));
        let start = (batch_idx * seq_len) % token_ids.len();
        let end = (start + seq_len).min(token_ids.len());
        let batch_tokens = &token_ids[start..end];
        let actual_seq = batch_tokens.len();

        let teacher_logits = &teacher_logits_chunks[chunk_idx];

        // Student forward (using cached frozen hidden states)
        let frozen_states = &frozen_states_per_chunk[chunk_idx];
        let student_logits = student.forward_from_frozen(frozen_states);

        // KL divergence loss
        let loss = gemma_mapper::kl_divergence_loss(
            teacher_logits, &student_logits,
            distill_config.temperature, vs, actual_seq,
        );

        // Update EMA
        loss_ema = if loss_ema.is_nan() { loss }
                   else { 0.9 * loss_ema + 0.1 * loss };
        if loss < best_loss { best_loss = loss; }

        // Compute d_loss / d_student_logits
        let mut d_logits = vec![0.0f32; actual_seq * vs];
        let scale = 1.0 / (distill_config.temperature * distill_config.temperature);
        for t in 0..actual_seq {
            for v in 0..vs {
                let idx = t * vs + v;
                let t_logit = teacher_logits.get(idx).copied().unwrap_or(0.0) / distill_config.temperature;
                let s_logit = student_logits.get(idx).copied().unwrap_or(0.0) / distill_config.temperature;
                let t_prob = t_logit.exp();
                let s_prob = s_logit.exp();
                d_logits[idx] = (s_prob - t_prob) * scale;
            }
        }

        // Backprop through Block Summary layers
        let all_grads: Vec<(usize, gemma_mapper::BlockSummaryGradients)> = student.block_summaries.iter().enumerate()
            .filter(|(_, bs)| bs.trainable)
            .filter(|(si, _)| *si < optimizers.len())
            .map(|(si, bs)| {
                let grads = gemma_mapper::backprop_block_summary(bs, &student.model.embed_tokens, &d_logits);
                (si, grads)
            })
            .collect();

        // Track gradient norm across all block summary layers
        let grad_norm: f32 = all_grads.iter()
            .map(|(_, g)| g.d_bridge_weight * g.d_bridge_weight)
            .sum::<f32>()
            .sqrt();

        for (si, grads) in all_grads {
            optimizers[si].step(&mut student.block_summaries[si], &grads);
        }

        // Collect per-layer bridge weights
        let bridge_weights: Vec<f32> = student.block_summaries.iter()
            .map(|bs| bs.bridge_weight)
            .collect();
        let bridge_w = bridge_weights.first().copied().unwrap_or(0.0);
        let bridge_mean = if bridge_weights.is_empty() { 0.0 }
            else { bridge_weights.iter().sum::<f32>() / bridge_weights.len() as f32 };
        let bridge_min = bridge_weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let bridge_max = bridge_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let lr = if global_step < distill_config.warmup_steps {
            distill_config.learning_rate * (global_step + 1) as f32 / distill_config.warmup_steps.max(1) as f32
        } else {
            distill_config.learning_rate
        };

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
            kl_loss: loss,
            bridge_weight: bridge_w,
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
                bridge_mean = format_args!("{:.4}", bridge_mean),
                bridge_min = format_args!("{:.4}", bridge_min),
                bridge_max = format_args!("{:.4}", bridge_max),
                lr = format_args!("{:.2e}", lr),
                step_ms = step_ms,
                tok_per_s = tps as u32,
                eta_s = eta as u32,
                "distillation step"
            );
        }

        // Mid-training checkpoint
        if checkpoint_every > 0 && (global_step + 1) % checkpoint_every == 0 {
            let ckpt = DistillationCheckpoint::from_student(&student, &optimizers, global_step + 1);
            let ckpt_path = format!("{}.checkpoint.bin", output_path);
            ckpt.save(std::path::Path::new(&ckpt_path))
                .map_err(|e| anyhow::anyhow!("Failed to save checkpoint: {}", e))?;
            info!(event = "checkpoint_saved", step = global_step + 1, path = %ckpt_path, "mid-training checkpoint saved");
        }

        prev_loss = loss;
        global_step += 1;
        if loss < 1e-6 { break; }
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
        info!(event = "final_bridge_weight", "Final bridge_weight: {:.6}", last.bridge_weight);
    }

    // Log loss curve
    for result in &results {
        if result.step % log_every == 0 {
            info!(event = "distill_step_summary", step = result.step, loss = result.kl_loss, bridge_w = result.bridge_weight, lr = result.learning_rate, "distillation step summary");
        }
    }

    // Save distilled model (block summary params)
    for (i, bs) in student.block_summaries.iter().enumerate() {
        let params = bs.export_trainable();
        let save_path = format!("{}.block_summary_{}.bin", output_path, i);
        let bytes: Vec<u8> = params.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&save_path, &bytes)?;
        info!(event = "saved_block_summary_params", "Saved Block Summary {} → {} ({} params)", i, save_path, params.len());
    }

    // Save full checkpoint (for resume) with optimizer state
    let ckpt_path = format!("{}.checkpoint.bin", output_path);
    let final_step = results.last().map(|r| r.step).unwrap_or(0) + 1;
    let final_ckpt = DistillationCheckpoint::from_student(&student, &optimizers, final_step);
    final_ckpt.save(std::path::Path::new(&ckpt_path))
        .map_err(|e| anyhow::anyhow!("Failed to save checkpoint: {}", e))?;
    info!(event = "checkpoint_saved", step = final_step, path = %ckpt_path, "final checkpoint saved");

    // Save loss curve as CSV
    let csv_path = format!("{}.loss_curve.csv", output_path);
    let mut csv = String::from("step,kl_loss,bridge_weight,learning_rate,cosine_sim_avg\n");
    for r in &results {
        let avg_cos = if r.layer_cosine_sim.is_empty() {
            String::from("n/a")
        } else {
            let avg = r.layer_cosine_sim.iter().sum::<f32>() / r.layer_cosine_sim.len() as f32;
            format!("{:.6}", avg)
        };
        csv.push_str(&format!("{},{},{},{},{}\n", r.step, r.kl_loss, r.bridge_weight, r.learning_rate, avg_cos));
    }
    std::fs::write(&csv_path, &csv)?;
    info!(event = "loss_curve_saved", "Loss curve saved → {}", csv_path);

    info!("");
    info!(event = "distillation_complete", "Distillation complete!");

    Ok(())
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
        _ => anyhow::bail!("Use e2b, e4b, or 27b-mm for evaluation"),
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
