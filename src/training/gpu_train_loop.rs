//! CLI/daemon policy for the GPU training path.
//!
//! Mirrors [`crate::training::train_loop`] step for step — windows in order,
//! `batch_size` windows averaged into one Adam step, the same checkpoint cadence
//! — but the forward, backward and optimizer run on the GPU graph. The
//! checkpoint is still `FRTRAIN1`: GPU weights and Adam moments are read back
//! into an [`F32Model`] and written by the existing writer. There is no second
//! format, so `--device cpu` can resume a GPU run and vice versa.

use std::path::PathBuf;

use crate::compute::pipeline::WgpuCompute;
use crate::error::{FerrisResError, Result};
use crate::training::f32_trainer::{Adam, F32BlockConfig, F32Model};
use crate::training::gpu_trainer::GpuModel;
use crate::training::train_loop::{load_checkpoint, save_checkpoint, validate, TrainOptions, TrainReport};

/// Train on the GPU and write the same `FRTRAIN1` checkpoint the CPU path does.
pub async fn run(options: &TrainOptions, tokens: &[u32]) -> Result<TrainReport> {
    validate(options, tokens)?;
    let cfg = options.config;

    let compute = WgpuCompute::new()
        .await
        .map_err(|e| FerrisResError::Device(format!("GPU device for --device gpu: {}", e)))?;
    let info = compute.adapter_info();
    tracing::info!(
        event = "gpu_train_device",
        "GPU training on {} ({:?})",
        info.name,
        info.backend
    );
    let device = std::sync::Arc::new(compute.device().clone());
    let queue = std::sync::Arc::new(compute.queue().clone());

    let (mut model, start_step) = match &options.resume {
        Some(path) => {
            let (cpu_model, cpu_adam, step) = load_checkpoint(path)?;
            if cpu_model.config != cfg {
                return Err(FerrisResError::Shape(format!(
                    "checkpoint {} has config {:?} but the CLI flags build {:?}",
                    path.display(),
                    cpu_model.config,
                    cfg
                )));
            }
            if cpu_model.block_config != F32BlockConfig::plain() {
                return Err(FerrisResError::Unsupported(format!(
                    "checkpoint {} uses block boundaries; the GPU trainer is plain-only",
                    path.display()
                )));
            }
            let mut model = GpuModel::from_f32_model(
                std::sync::Arc::clone(&device),
                std::sync::Arc::clone(&queue),
                &cpu_model,
            )?;
            restore_adam(&mut model, &cpu_model, &cpu_adam)?;
            tracing::info!(
                event = "resuming_checkpoint_step",
                "Resuming {} at step {}",
                path.display(),
                step
            );
            (model, step)
        }
        None => (
            GpuModel::new(
                std::sync::Arc::clone(&device),
                std::sync::Arc::clone(&queue),
                cfg,
                options.seed,
            )?,
            0,
        ),
    };

    let windows_per_epoch = (tokens.len() - 1) / options.seq_len;
    let total_step = start_step + options.epochs as u64 * windows_per_epoch as u64;
    let before = model.read_param("embed")?;

    let mut initial_loss = f32::NAN;
    let mut final_loss = f32::NAN;

    for step in (start_step + 1)..=total_step {
        let step_started = std::time::Instant::now();
        let mut ms_forward = 0.0f64;
        let mut ms_read_logits = 0.0f64;
        let mut ms_cross_entropy = 0.0f64;
        let mut ms_backward = 0.0f64;
        let mut ms_accumulate = 0.0f64;
        let mut ms_optimizer = 0.0f64;
        let mut ms_checkpoint = 0.0f64;
        let first_window = ((step - 1) % windows_per_epoch as u64) as usize;
        let mut step_loss = 0.0f32;

        for b in 0..options.batch_size {
            let window = (first_window + b) % windows_per_epoch;
            let begin = window * options.seq_len;
            let input = &tokens[begin..begin + options.seq_len];
            let target = &tokens[begin + 1..begin + options.seq_len + 1];

            let t = std::time::Instant::now();
            let (logits_node, loss_node) = model.forward(input)?;
            ms_forward += t.elapsed().as_secs_f64() * 1e3;

            // The logits readback is the phase that makes a GPU step cost like
            // a CPU one: seq * vocab * 4 bytes per window, then a CPU softmax.
            let t = std::time::Instant::now();
            let logits = model.read_logits(logits_node, options.seq_len * cfg.vocab)?;
            ms_read_logits += t.elapsed().as_secs_f64() * 1e3;

            let t = std::time::Instant::now();
            let (loss, _) = F32Model::cross_entropy(&logits, target, cfg.vocab);
            ms_cross_entropy += t.elapsed().as_secs_f64() * 1e3;
            if !loss.is_finite() {
                return Err(FerrisResError::Shape(format!(
                    "non-finite loss at step {}: {}",
                    step, loss
                )));
            }
            step_loss += loss;

            let t = std::time::Instant::now();
            model.backward(loss_node)?;
            ms_backward += t.elapsed().as_secs_f64() * 1e3;

            let t = std::time::Instant::now();
            model.accumulate_grads()?;
            ms_accumulate += t.elapsed().as_secs_f64() * 1e3;
        }

        let mean_loss = step_loss / options.batch_size as f32;
        if step == start_step + 1 {
            initial_loss = mean_loss;
        }
        final_loss = mean_loss;

        let t = std::time::Instant::now();
        model.step_accumulated(options.learning_rate, options.batch_size as u32)?;
        model.reset_accumulators()?;
        ms_optimizer += t.elapsed().as_secs_f64() * 1e3;

        if options.checkpoint_every > 0 && step % options.checkpoint_every == 0 {
            let t = std::time::Instant::now();
            save_gpu_checkpoint(&options.output, &model, step)?;
            ms_checkpoint += t.elapsed().as_secs_f64() * 1e3;
        }

        let total_ms = step_started.elapsed().as_secs_f64() * 1e3;
        let tokens = (options.batch_size as u64) * (options.seq_len as u64);
        if options.log_every > 0 && (step % options.log_every == 0 || step == total_step) {
            tracing::info!(
                event = "train_step_loss",
                "step {}/{} loss={:.6}",
                step,
                total_step,
                mean_loss
            );
            tracing::info!(
                event = "train_step_timing",
                step,
                total_ms,
                forward_ms = ms_forward,
                read_logits_ms = ms_read_logits,
                cross_entropy_ms = ms_cross_entropy,
                backward_ms = ms_backward,
                accumulate_ms = ms_accumulate,
                optimizer_ms = ms_optimizer,
                checkpoint_ms = ms_checkpoint,
                tokens_per_s = tokens as f64 / (total_ms / 1e3),
                "step timing: total={:.1}ms fwd={:.1} readback={:.1} ce={:.1} bwd={:.1} acc={:.1} opt={:.1} ckpt={:.1}",
                total_ms,
                ms_forward,
                ms_read_logits,
                ms_cross_entropy,
                ms_backward,
                ms_accumulate,
                ms_optimizer,
                ms_checkpoint
            );
        }
    }

    save_gpu_checkpoint(&options.output, &model, total_step)?;

    let after = model.read_param("embed")?;
    let report = TrainReport {
        initial_loss,
        final_loss,
        steps_run: total_step - start_step,
        resumed_from: start_step,
        total_step,
        params_changed: before != after,
        checkpoint: options.output.clone(),
    };
    tracing::info!(
        event = "training_complete_initial_final_loss",
        "Training complete: steps={} initial_loss={:.6} final_loss={:.6}",
        report.steps_run,
        report.initial_loss,
        report.final_loss
    );
    Ok(report)
}

/// Build the `FRTRAIN1` writer's [`Adam`] from the GPU optimizer state. Reads
/// one tensor at a time; only called at checkpoint boundaries.
fn save_gpu_checkpoint(path: &PathBuf, model: &GpuModel, step: u64) -> Result<()> {
    let cpu = model.to_f32_model()?;
    let mut m = Vec::with_capacity(cpu.param_count());
    let mut v = Vec::with_capacity(cpu.param_count());
    for (name, values) in cpu.tensors() {
        let (mt, vt) = model.adam.read_state(&name, values.len())?;
        m.extend_from_slice(&mt);
        v.extend_from_slice(&vt);
    }
    let adam = Adam {
        m,
        v,
        t: model.adam.timestep(),
    };
    save_checkpoint(path, &cpu, &adam, step)?;
    tracing::info!(
        event = "checkpoint_saved_step_path",
        "Checkpoint saved: step={} path={}",
        step,
        path.display()
    );
    Ok(())
}

fn restore_adam(model: &mut GpuModel, cpu: &F32Model, adam: &Adam) -> Result<()> {
    let mut offset = 0usize;
    for (name, values) in cpu.tensors() {
        let n = values.len();
        if offset + n > adam.m.len() || offset + n > adam.v.len() {
            return Err(FerrisResError::Shape(format!(
                "checkpoint Adam state is shorter than the model at tensor `{}`",
                name
            )));
        }
        model
            .adam
            .load_state(&name, &adam.m[offset..offset + n], &adam.v[offset..offset + n])?;
        offset += n;
    }
    model.adam.set_timestep(adam.t);
    Ok(())
}
