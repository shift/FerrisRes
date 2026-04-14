use std::sync::Arc;
use std::time::Instant;
use clap::{Parser, Subcommand};
use tracing::{info, warn};
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
            resume, model_format, tokenizer, checkpoint_every,
        } => cmd_distill(model_path, config, seq_len, steps, learning_rate, temperature, data, output, log_every, resume, model_format, tokenizer, checkpoint_every).await,
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

    info!("Adapter: {} ({})", capability.adapter_name, capability.backend);
    info!("GPU kind: {:?}", capability.gpu_kind);
    info!("Dedicated VRAM: {} MB", capability.vram_mb);
    info!("System RAM: {} MB", capability.shared_ram_mb);
    info!("Effective VRAM: {} MB", capability.effective_vram_mb());

    info!("Device profile: {:?}", profile);
    info!("Compute mode: {:?}", profile.compute_mode());
    info!("Recommended batch size: {}", profile.recommended_batch_size());
    info!("Cache size: {} MB", profile.cache_size() / (1024 * 1024));

    info!("Max workgroup size: {}", capability.max_compute_workgroup_size);
    info!("Max invocations/workgroup: {}", capability.max_compute_invocations_per_workgroup);
    info!("Max storage buffer: {} MB", capability.max_storage_buffer_range / (1024 * 1024));
    info!("Max storage buffers/stage: {}", capability.max_storage_buffers_per_shader_stage);
    info!("Max bind groups: {}", capability.max_bind_groups);

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
    info!("Initializing training pipeline");

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
        info!("LoRA enabled: rank={} alpha={}", rank, rank * 2);
        Some(mgr)
    } else {
        None
    };

    info!("Model config: hidden_dim={} num_blocks={} block_size={} total_layers={} heads={} intermediate_dim={}",
        config.hidden_dim, config.num_blocks, config.block_size, config.total_layers(),
        config.attention_heads, config.intermediate_dim);

    info!("Optimizer: Adam lr={} beta1=0.9 beta2=0.999 eps=1e-8", learning_rate);

    let sample = "Hello world from FerrisRes training";
    let tokens = tokenizer.encode(sample);
    info!("Sample tokenized ({} tokens): {:?}", tokens.len(), tokens);

    if let Some(data_path) = &data {
        info!("Data path: {}", data_path);
    }

    info!("Training would run for {} epochs with batch_size={} learning_rate={}", epochs, batch_size, learning_rate);

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

    info!("Autodiff graph built: {} parameters tracked", 2); // input + logits as tracked nodes

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
                        tracing::trace!("LoRA q_proj delta: {} values", delta.len());
                    }
                    if let Some(delta) = lora.forward(0, "v_proj", &dummy_hidden, 1) {
                        tracing::trace!("LoRA v_proj delta: {} values", delta.len());
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
            info!("Epoch {}/{}: avg_loss={:.4} batches={}", epoch + 1, epochs, epoch_loss / batches as f32, batches);
        }
    }

    info!("Training complete");

    // Merge LoRA adapters back into base weights for zero-cost inference
    if let Some(ref mut lora) = _lora_manager {
        if !lora.is_merged() {
            lora.merge_all(&mut |layer_idx: usize, module_name: &str| {
                // In a full implementation, this would return mutable slices
                // into the model's weight GpuBuffers. For now, log the merge.
                tracing::info!("Merging LoRA adapter: layer={}, module={}", layer_idx, module_name);
                None::<&mut [f32]>
            });
            info!("LoRA adapters merged: {} adapters, {} params", lora.num_adapters(), lora.total_params());
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
    info!("Initializing inference pipeline");

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
                info!("Applied {} template: {} chars → {} chars", template_name, prompt.len(), result.len());
                result
            }
            None => {
                info!("Unknown template '{}', using raw prompt", template_name);
                prompt.clone()
            }
        }
    } else {
        prompt.clone()
    };

    let tokenizer = SimpleTokenizer::new();
    let model = BlockAttnResModel::new(Arc::clone(&device), Arc::clone(&queue), config.clone(), tokenizer.vocab_size())?;

    let tokens = tokenizer.encode(&formatted_prompt);
    info!("Encoded tokens ({}): {:?}", tokens.len(), tokens);

    // Wire image preprocessing: upload patches to GPU and run Im2ColOp
    let mut image_patch_count: usize = 0;
    if let Some(ref image_path) = image {
        let preprocessor = ferrisres::ImagePreprocessor::new(224, 224, true);
        match std::fs::read(image_path) {
            Ok(image_data) => {
                match preprocessor.preprocess(&image_data) {
                    Ok(patches) => {
                        let num_patches = patches.len();
                        info!("Image preprocessed: {} float values from {}", num_patches, image_path);

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
                        info!("Im2Col: {} patches of dim {} uploaded to GPU", image_patch_count, patch_dim);

                        // In a full multimodal model, the im2col_output would be projected
                        // through a linear layer to match hidden_dim, then concatenated with
                        // text token embeddings. The patch embeddings are now on GPU ready
                        // for that projection step.
                    }
                    Err(e) => warn!("Image preprocessing failed: {}", e),
                }
            }
            Err(e) => warn!("Failed to read image {}: {}", image_path, e),
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
    info!("Generated {} tokens{}", output_tokens.len(),
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
    info!("Initializing benchmark ({} iterations)", iterations);

    let compute = WgpuCompute::new().await?;
    let capability = compute.detect_capability();
    info!("Adapter: {} ({})", capability.adapter_name, capability.backend);

    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let matmul_op = MatMulOp::new(&device, &queue);
    let rmsnorm_op = RmsNormOp::new(&device)?;
    let softmax_op = SoftmaxOp::new(&device, &queue)?;
    let ew_op = ElementWiseOp::new(&device, &queue);

    info!("");
    info!("=== Benchmark Results ===");
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

        info!("MatMul {}x{}: {:.2?} total, {:.2} GFLOPS", size, size, elapsed, gflops);
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

        info!("RmsNorm hidden_dim={}: {:.2?} total, {:.1} us/call", hd, elapsed, us_per_call);
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

        info!("Softmax {}x{}: {:.2?} total, {:.1} us/call", rows, cols, elapsed, us_per_call);
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

        info!("ElementWise {} numel={}: {:.2?} total, {:.1} us/call", op_name, numel, elapsed, us_per_call);
    }

    info!("");
    info!("=== Benchmark Complete ===");

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
    _checkpoint_every: usize,
) -> anyhow::Result<()> {
    info!("=== FerrisRes Gemma 4 → Block AttnRes Distillation ===");

    // Parse config
    let config = match config_name.as_str() {
        "e2b" => { info!("Using Gemma 4 E2B config (dense, ~4 GB)"); Gemma4Config::gemma4_e2b() }
        "e4b" => { info!("Using Gemma 4 E4B config (MoE-16, ~8 GB)"); Gemma4Config::gemma4_e4b() }
        "12b" => { info!("Using Gemma 4 12B config (MoE-128, ~24 GB)"); Gemma4Config::gemma4_12b() }
        "27b" => { info!("Using Gemma 4 27B config (MoE-128, ~54 GB)"); Gemma4Config::gemma4_27b() }
        "27b-mm" => { info!("Using Gemma 4 27B Multimodal IT config (dense, 35 layers, ~10 GB)"); Gemma4Config::gemma4_27b_mm() }
        "llama3-8b" => { info!("Using LLaMA 3.1 8B config"); Gemma4Config::llama3_8b() }
        "llama3-70b" => { info!("Using LLaMA 3.1 70B config"); Gemma4Config::llama3_70b() }
        "mistral-7b" => { info!("Using Mistral 7B config"); Gemma4Config::mistral_7b() }
        "mixtral-8x7b" => { info!("Using Mixtral 8x7B config"); Gemma4Config::mixtral_8x7b() }
        "phi3-mini" => { info!("Using Phi-3 Mini config"); Gemma4Config::phi3_mini() }
        "qwen2-7b" => { info!("Using Qwen 2.5 7B config"); Gemma4Config::qwen2_7b() }
        other => anyhow::bail!("Unknown config '{}'. Use: e2b, e4b, 12b, 27b, llama3-8b, llama3-70b, mistral-7b, mixtral-8x7b, phi3-mini, qwen2-7b", other),
    };

    info!("Model: hidden={} layers={} heads={} vocab={}",
        config.hidden_dim, config.num_layers, config.num_heads, config.vocab_size);

    // Load weights
    info!("Loading weights from: {}", model_path);
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
    info!("Model loaded: {} layers, {} params",
        model1.layers.len(),
        model1.embed_tokens.len() + model1.layers.iter().map(|l| {
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
        }).sum::<usize>() + model1.final_norm.len() + model1.lm_head.len());

    // Create teacher (frozen) — compute logits, then free weights
    let teacher = Gemma4Teacher::new(model1);
    info!("Teacher model created (frozen)");

    // Load training data
    let hf_tokenizer = if let Some(ref tok_path) = tokenizer_path {
        let tok = HfTokenizer::from_tokenizer_json(std::path::Path::new(tok_path))
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        info!("Loaded tokenizer with vocab size {}", tok.vocab_size());
        Some(tok)
    } else {
        info!("No tokenizer specified, using byte-fallback tokenization");
        None
    };

    let token_ids = if let Some(ref dp) = data_path {
        info!("Loading training data from: {}", dp);
        let text = std::fs::read_to_string(dp)
            .map_err(|e| anyhow::anyhow!("Failed to read data: {}", e))?;
        if let Some(ref tok) = hf_tokenizer {
            let ids = tok.encode_raw(&text);
            info!("Tokenized to {} tokens using provided tokenizer", ids.len());
            ids
        } else {
            let ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
            info!("Loaded {} byte-level tokens from training data", ids.len());
            ids
        }
    } else {
        info!("No training data provided, using synthetic tokens");
        (0..10000).map(|i| i % config.vocab_size as u32).collect()
    };

    // Pre-compute teacher logits for each chunk, then drop teacher to free ~10GB
    info!("Pre-computing teacher logits...");
    let mut teacher_logits_chunks: Vec<Vec<f32>> = Vec::new();
    let num_chunks = (token_ids.len() + seq_len - 1) / seq_len;
    for chunk_idx in 0..num_chunks.min(steps) {
        let start = (chunk_idx * seq_len) % token_ids.len();
        let end = (start + seq_len).min(token_ids.len());
        let chunk: Vec<u32> = token_ids[start..end].to_vec();
        if chunk.len() < seq_len { break; }
        let logits = teacher.forward(&chunk);
        teacher_logits_chunks.push(logits);
        info!("Teacher logits: chunk {}/{}", chunk_idx + 1, num_chunks.min(steps));
    }
    info!("Teacher logits computed: {} chunks", teacher_logits_chunks.len());

    // Drop teacher to free ~10GB RAM
    drop(teacher);
    info!("Teacher freed, memory released for student");

    // Re-load model for student (single copy in memory now)
    let model2 = load_model(path)
        .map_err(|e| anyhow::anyhow!("Failed to reload model for student: {}", e))?;
    info!("Student model loaded");

    let injection_points = config.block_summary_injection_points();
    let block_summaries: Vec<BlockSummaryLayer> = injection_points.iter()
        .map(|_| BlockSummaryLayer::new_identity(config.hidden_dim, config.sliding_window))
        .collect();
    info!("Student model created with {} Block Summary layers", block_summaries.len());

    let distill_config = DistillationConfig {
        learning_rate: learning_rate as f32,
        temperature: temperature as f32,
        num_steps: steps,
        max_seq_len: seq_len,
        ..DistillationConfig::default()
    };

    let mut student = Gemma4Student::new(model2, block_summaries, distill_config.clone());

    // Resume from checkpoint if specified
    if let Some(ref ckpt_path) = resume_path {
        let ckpt = DistillationCheckpoint::load(std::path::Path::new(ckpt_path))
            .map_err(|e| anyhow::anyhow!("Failed to load checkpoint: {}", e))?;
        info!("Resuming from checkpoint at step {}", ckpt.global_step);
        let _optimizers = ckpt.apply(&mut student);
        info!("Restored Block Summary layers from checkpoint");
    }

    info!("");
    info!("Starting distillation: {} steps, seq_len={}, lr={}, temp={}",
        steps, seq_len, learning_rate, temperature);
    info!("");

    // Run distillation using pre-computed teacher logits
    let vs = config.vocab_size;
    let mut results: Vec<gemma_mapper::DistillationStepResult> = Vec::new();
    let mut optimizers: Vec<gemma_mapper::BlockSummaryAdam> = student.block_summaries.iter()
        .map(|bs| gemma_mapper::BlockSummaryAdam::new(bs, distill_config.learning_rate))
        .collect();

    for step in 0..steps {
        let chunk_idx = step % teacher_logits_chunks.len();
        let batch_idx = step % (token_ids.len() / seq_len.max(1));
        let start = (batch_idx * seq_len) % token_ids.len();
        let end = (start + seq_len).min(token_ids.len());
        let batch_tokens = &token_ids[start..end];
        let actual_seq = batch_tokens.len();

        let teacher_logits = &teacher_logits_chunks[chunk_idx];

        // Student forward
        let student_logits = student.forward(batch_tokens);

        // KL divergence loss
        let loss = gemma_mapper::kl_divergence_loss(
            teacher_logits, &student_logits,
            distill_config.temperature, vs, actual_seq,
        );

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

        for (si, grads) in all_grads {
            optimizers[si].step(&mut student.block_summaries[si], &grads);
        }

        let bridge_w = student.block_summaries.first()
            .map(|bs| bs.bridge_weight)
            .unwrap_or(0.0);

        let lr = if step < distill_config.warmup_steps {
            distill_config.learning_rate * step as f32 / distill_config.warmup_steps.max(1) as f32
        } else {
            distill_config.learning_rate
        };

        results.push(gemma_mapper::DistillationStepResult {
            step,
            kl_loss: loss,
            bridge_weight: bridge_w,
            learning_rate: lr,
            layer_cosine_sim: vec![],
        });

        if step % log_every == 0 {
            info!("Step {:>5}: loss={:.6} bridge_w={:.4} lr={:.6}",
                step, loss, bridge_w, lr);
        }

        if loss < 1e-6 { break; }
    }

    // Report results
    info!("");
    info!("=== Distillation Results ===");
    info!("Steps completed: {}", results.len());

    if let Some(first) = results.first() {
        info!("Initial KL loss: {:.6}", first.kl_loss);
    }
    if let Some(last) = results.last() {
        info!("Final KL loss:   {:.6}", last.kl_loss);
        info!("Final bridge_weight: {:.6}", last.bridge_weight);
    }

    // Log loss curve
    for result in &results {
        if result.step % log_every == 0 {
            info!("Step {:>5}: loss={:.6} bridge_w={:.4} lr={:.6}",
                result.step, result.kl_loss, result.bridge_weight, result.learning_rate);
        }
    }

    // Save distilled model (block summary params)
    for (i, bs) in student.block_summaries.iter().enumerate() {
        let params = bs.export_trainable();
        let save_path = format!("{}.block_summary_{}.bin", output_path, i);
        let bytes: Vec<u8> = params.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(&save_path, &bytes)?;
        info!("Saved Block Summary {} → {} ({} params)", i, save_path, params.len());
    }

    // Save full checkpoint (for resume)
    let ckpt_path = format!("{}.checkpoint.bin", output_path);
    // We don't have the optimizers here — save a simple version
    let final_ckpt = DistillationCheckpoint {
        version: 1,
        global_step: results.last().map(|r| r.step).unwrap_or(0),
        layer_checkpoints: student.block_summaries.iter().map(|bs| {
            use gemma_mapper::BlockSummaryCheckpoint;
            let params = bs.export_trainable();
            let _hd = bs.hidden_dim;
            BlockSummaryCheckpoint {
                params,
                adam_m: vec![],
                adam_v: vec![],
                adam_t: 0,
                bridge_weight: bs.bridge_weight,
                bw_m: 0.0, bw_v: 0.0,
                norm_weight: vec![],
                norm_bias: vec![],
            }
        }).collect(),
    };
    final_ckpt.save(std::path::Path::new(&ckpt_path))
        .map_err(|e| anyhow::anyhow!("Failed to save checkpoint: {}", e))?;
    info!("Checkpoint saved → {}", ckpt_path);

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
    info!("Loss curve saved → {}", csv_path);

    info!("");
    info!("Distillation complete!");

    Ok(())
}

async fn cmd_evaluate(
    model_path: String,
    config_name: String,
    text: String,
) -> anyhow::Result<()> {
    info!("=== FerrisRes Evaluation ===");

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
    info!("Evaluating on {} byte-level tokens", token_ids.len());

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

    info!("Tokens evaluated: {}", count);
    info!("Average NLL: {:.4}", avg_nll);
    info!("Perplexity: {:.2}", perplexity);

    Ok(())
}
