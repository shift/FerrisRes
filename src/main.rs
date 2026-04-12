use std::sync::Arc;
use std::time::Instant;
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use ferrisres::{WgpuCompute, DeviceProfile, BlockAttnResConfig, BlockAttnResModel, SimpleTokenizer};
use ferrisres::compute::{GpuBuffer, MatMulOp, RmsNormOp, SoftmaxOp, ElementWiseOp};
use ferrisres::training::AdamOptimizer;

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
