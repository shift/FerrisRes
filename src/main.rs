use std::sync::Arc;
use std::time::Instant;
use clap::{Parser, Subcommand};
use ferrisres::{WgpuCompute, DeviceProfile, BlockAttnResConfig, BlockAttnResModel, SimpleTokenizer};
use ferrisres::compute::{GpuBuffer, MatMulOp, RmsNormOp, SoftmaxOp, ElementWiseOp};
use ferrisres::training::AdamOptimizer;
use tracing::info;

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
        } => cmd_train(hidden_dim, num_blocks, block_size, epochs, batch_size, learning_rate, data).await,
        Commands::Infer {
            hidden_dim,
            num_blocks,
            block_size,
            prompt,
            max_tokens,
            temperature,
        } => cmd_infer(hidden_dim, num_blocks, block_size, prompt, max_tokens, temperature, template).await,
        Commands::Benchmark {
            hidden_dim,
            num_blocks,
            block_size,
            iterations,
        } => cmd_benchmark(hidden_dim, num_blocks, block_size, iterations).await,
    }
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
    info!("Training stub: forward pass only (full training loop requires autodiff wiring)");

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

    let input_bytes = config.hidden_dim * std::mem::size_of::<f32>();
    let input = GpuBuffer::zeros(&device, &queue, input_bytes, Some("Infer Input"))?;
    let output = GpuBuffer::new(&device, input_bytes, Some("Infer Output"))?;

    model.forward(&input, &output, 1)?;

    info!("Forward pass complete");
    info!("Config: max_tokens={} temperature={}", max_tokens, temperature);
    info!("Generated {} tokens (stub: autoregressive generation not yet wired)", max_tokens);

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
