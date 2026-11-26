//! Real training loop over the f32 reference model.
//!
//! [`crate::training::f32_trainer`] owns the maths (forward, hand-written
//! backward, Adam) but deliberately knows nothing about files or the CLI.
//! This module is the thin policy layer between them: next-token windows,
//! gradient accumulation, checkpointing and resume.
//!
//! The forward pass has no batch axis, so `batch_size` means "windows whose
//! gradients are averaged into one Adam step". Training is deterministic for a
//! fixed seed and a fixed token stream: windows are visited in order, with no
//! shuffling.
//!
//! Block AttnRes boundaries are not used here. `F32BlockConfig::plain()` keeps
//! the path identical to the verified reference trainer; the block+LoRA path
//! has a separate open gradient defect.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::error::{FerrisResError, Result};

use super::f32_trainer::{Adam, F32BlockConfig, F32Config, F32Model, Gradients};

/// Everything the loop needs that is not the token stream itself.
#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub config: F32Config,
    /// Window length in tokens. One step predicts `tokens[i + 1]` from
    /// `tokens[..=i]` inside the window.
    pub seq_len: usize,
    /// Windows averaged into a single Adam step.
    pub batch_size: usize,
    /// Epochs to run *from the resume point*. A fresh run and a resumed run
    /// therefore both train for this many passes over the data.
    pub epochs: u32,
    pub learning_rate: f32,
    pub seed: u64,
    /// Save a checkpoint every N optimizer steps. Zero disables periodic
    /// saves; the final checkpoint is always written.
    pub checkpoint_every: u64,
    /// Checkpoint path. Also the path written at the end of a run.
    pub output: PathBuf,
    /// Restore model and Adam state from this checkpoint before training.
    pub resume: Option<PathBuf>,
    /// Log one loss line every N steps. Zero logs only the first and last.
    pub log_every: u64,
}

/// What one invocation did. Enough for the CLI to report and for a test to
/// assert that learning happened without parsing logs.
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Mean loss of the first step, before its update.
    pub initial_loss: f32,
    /// Mean loss of the last step, before its update.
    pub final_loss: f32,
    /// Optimizer steps run by this invocation.
    pub steps_run: u64,
    /// Step restored from the checkpoint (0 when training from scratch).
    pub resumed_from: u64,
    /// Step the run finished at.
    pub total_step: u64,
    /// Whether at least one parameter tensor changed.
    pub params_changed: bool,
    /// Path of the final checkpoint.
    pub checkpoint: PathBuf,
}

const CHECKPOINT_MAGIC: &[u8; 8] = b"FRTRAIN1";
const CHECKPOINT_VERSION: u32 = 1;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Train `tokens` with the reference model and return the measured losses.
///
/// `tokens` must contain at least `seq_len + 1` values, and every value must
/// be a valid index into `options.config.vocab`.
pub fn run(options: &TrainOptions, tokens: &[u32]) -> Result<TrainReport> {
    validate(options, tokens)?;
    let cfg = options.config;

    let (mut model, mut adam, start_step) = match &options.resume {
        Some(path) => {
            let (model, adam, step) = load_checkpoint(path)?;
            if model.config != cfg {
                return Err(FerrisResError::Shape(format!(
                    "checkpoint {} has config {:?} but the CLI flags build {:?}",
                    path.display(),
                    model.config,
                    cfg
                )));
            }
            if model.block_config != F32BlockConfig::plain() {
                return Err(FerrisResError::Unsupported(format!(
                    "checkpoint {} uses block boundaries; this trainer only resumes plain checkpoints",
                    path.display()
                )));
            }
            tracing::info!(
                event = "resuming_checkpoint_step",
                "Resuming {} at step {}",
                path.display(),
                step
            );
            (model, adam, step)
        }
        None => {
            let model = F32Model::new_random(cfg, options.seed);
            let adam = Adam::new(&model);
            (model, adam, 0)
        }
    };

    let windows_per_epoch = (tokens.len() - 1) / options.seq_len;
    let total_step = start_step + options.epochs as u64 * windows_per_epoch as u64;
    let before = params_fingerprint(&model);

    let mut initial_loss = f32::NAN;
    let mut final_loss = f32::NAN;

    for step in (start_step + 1)..=total_step {
        let window = ((step - 1) % windows_per_epoch as u64) as usize;
        let mut grads = Gradients::zeros(&cfg);
        let mut step_loss = 0.0f32;

        for b in 0..options.batch_size {
            let window = (window + b) % windows_per_epoch;
            let begin = window * options.seq_len;
            let input = &tokens[begin..begin + options.seq_len];
            let target = &tokens[begin + 1..begin + options.seq_len + 1];

            let (logits, cache) = model.forward(input);
            let (loss, d_logits) = F32Model::cross_entropy(&logits, target, cfg.vocab);
            if !loss.is_finite() {
                return Err(FerrisResError::Shape(format!(
                    "non-finite loss at step {}: {}",
                    step, loss
                )));
            }
            step_loss += loss;
            let gradients = model.backward(&cache, &d_logits);
            grads.accumulate(&gradients, 1.0 / options.batch_size as f32);
        }

        let mean_loss = step_loss / options.batch_size as f32;
        if step == start_step + 1 {
            initial_loss = mean_loss;
        }
        final_loss = mean_loss;

        adam.step(&mut model, &grads, options.learning_rate, ADAM_BETA1, ADAM_BETA2, ADAM_EPS);

        if options.log_every > 0 && (step % options.log_every == 0 || step == total_step) {
            tracing::info!(
                event = "train_step_loss",
                "step {}/{} loss={:.6}",
                step,
                total_step,
                mean_loss
            );
        }

        if options.checkpoint_every > 0 && step % options.checkpoint_every == 0 {
            save_checkpoint(&options.output, &model, &adam, step)?;
            tracing::info!(
                event = "checkpoint_saved_step_path",
                "Checkpoint saved: step={} path={}",
                step,
                options.output.display()
            );
        }
    }

    save_checkpoint(&options.output, &model, &adam, total_step)?;

    let report = TrainReport {
        initial_loss,
        final_loss,
        steps_run: total_step - start_step,
        resumed_from: start_step,
        total_step,
        params_changed: params_fingerprint(&model) != before,
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

pub(crate) fn validate(options: &TrainOptions, tokens: &[u32]) -> Result<()> {
    if options.seq_len == 0 || options.seq_len > options.config.max_seq {
        return Err(FerrisResError::Shape(format!(
            "seq_len {} must be in 1..={} (the model's max_seq)",
            options.seq_len, options.config.max_seq
        )));
    }
    if options.batch_size == 0 {
        return Err(FerrisResError::Shape("batch_size must be > 0".to_string()));
    }
    if options.epochs == 0 {
        return Err(FerrisResError::Shape("epochs must be > 0".to_string()));
    }
    if tokens.len() < options.seq_len + 1 {
        return Err(FerrisResError::Shape(format!(
            "need at least {} tokens for one window, got {}",
            options.seq_len + 1,
            tokens.len()
        )));
    }
    if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= options.config.vocab) {
        return Err(FerrisResError::Shape(format!(
            "token {} is outside vocab {}; pass a matching --vocab",
            bad, options.config.vocab
        )));
    }
    Ok(())
}

/// FNV-1a over every parameter's bit pattern. Used to tell the caller whether
/// the optimizer touched anything, without keeping a second copy of the model.
fn params_fingerprint(model: &F32Model) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for (_, values) in model.tensors() {
        for value in values {
            hash ^= value.to_bits() as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

// ---------------------------------------------------------------------------
// Checkpoint format
// ---------------------------------------------------------------------------
//
// Version 1, little-endian throughout:
//
//   magic[8] = "FRTRAIN1", version u32, step u64
//   config: 8 x u64 (hidden, layers, heads, kv_heads, head_dim, intermediate,
//           vocab, max_seq) + rope_theta f32 + eps f32
//   block: num_blocks u64, layers_per_block u64, boundary count u64, boundaries
//   tensors: count u64, then name (len u64 + UTF-8), element count u64, f32s
//   adam: t u32, m (len u64 + f32s), v (len u64 + f32s)
//
// The GPU [`super::checkpointing::CheckpointStore`] is activation
// checkpointing for the wgpu model, so it does not fit this CPU trainer; the
// parameter layout below is the same stable one `F32Model::tensors` exposes.

pub(crate) fn save_checkpoint(path: &Path, model: &F32Model, adam: &Adam, step: u64) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(CHECKPOINT_MAGIC)?;
    write_u32(&mut w, CHECKPOINT_VERSION)?;
    write_u64(&mut w, step)?;
    write_config(&mut w, &model.config)?;
    write_u64(&mut w, model.block_config.num_blocks as u64)?;
    write_u64(&mut w, model.block_config.layers_per_block as u64)?;
    write_u64(&mut w, model.block_config.boundary_layers.len() as u64)?;
    for &boundary in &model.block_config.boundary_layers {
        write_u64(&mut w, boundary as u64)?;
    }

    let tensors = model.tensors();
    write_u64(&mut w, tensors.len() as u64)?;
    for (name, values) in tensors {
        write_u64(&mut w, name.len() as u64)?;
        w.write_all(name.as_bytes())?;
        write_f32_slice(&mut w, values)?;
    }

    write_u32(&mut w, adam.t)?;
    write_f32_slice(&mut w, &adam.m)?;
    write_f32_slice(&mut w, &adam.v)?;
    w.flush()?;
    Ok(())
}

pub(crate) fn load_checkpoint(path: &Path) -> Result<(F32Model, Adam, u64)> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != CHECKPOINT_MAGIC {
        return Err(FerrisResError::Shape(format!(
            "{} is not a FerrisRes f32 checkpoint",
            path.display()
        )));
    }
    let version = read_u32(&mut r)?;
    if version != CHECKPOINT_VERSION {
        return Err(FerrisResError::Unsupported(format!(
            "checkpoint version {} is not supported (expected {})",
            version, CHECKPOINT_VERSION
        )));
    }
    let step = read_u64(&mut r)?;
    let config = read_config(&mut r)?;
    let block_config = F32BlockConfig {
        num_blocks: read_u64(&mut r)? as usize,
        layers_per_block: read_u64(&mut r)? as usize,
        boundary_layers: {
            let n = read_u64(&mut r)? as usize;
            let mut boundaries = Vec::with_capacity(n);
            for _ in 0..n {
                boundaries.push(read_u64(&mut r)? as usize);
            }
            boundaries
        },
    };

    let mut model = F32Model::new_random_with_blocks(config, block_config, 0);
    let tensor_count = read_u64(&mut r)? as usize;
    {
        let mut tensors = model.tensors_mut();
        if tensor_count != tensors.len() {
            return Err(FerrisResError::Shape(format!(
                "checkpoint has {} tensors, model expects {}",
                tensor_count,
                tensors.len()
            )));
        }
        for (expected_name, values) in tensors.iter_mut() {
            let name_len = read_u64(&mut r)? as usize;
            let mut name = vec![0u8; name_len];
            r.read_exact(&mut name)?;
            let name = String::from_utf8(name)
                .map_err(|_| FerrisResError::Shape("tensor name is not UTF-8".to_string()))?;
            if &name != expected_name {
                return Err(FerrisResError::Shape(format!(
                    "checkpoint tensor `{}` does not match expected `{}`",
                    name, expected_name
                )));
            }
            read_f32_slice(&mut r, values, &name)?;
        }
    }

    let adam_t = read_u32(&mut r)?;
    let mut m = vec![0.0f32; model.param_count()];
    let mut v = vec![0.0f32; model.param_count()];
    read_f32_slice(&mut r, &mut m, "adam.m")?;
    read_f32_slice(&mut r, &mut v, "adam.v")?;

    let adam = Adam { m, v, t: adam_t };
    Ok((model, adam, step))
}

fn write_config<W: Write>(w: &mut W, cfg: &F32Config) -> std::io::Result<()> {
    for value in [
        cfg.hidden,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.intermediate,
        cfg.vocab,
        cfg.max_seq,
    ] {
        write_u64(w, value as u64)?;
    }
    write_f32(w, cfg.rope_theta)?;
    write_f32(w, cfg.eps)
}

fn read_config<R: Read>(r: &mut R) -> Result<F32Config> {
    Ok(F32Config {
        hidden: read_u64(r)? as usize,
        num_layers: read_u64(r)? as usize,
        num_heads: read_u64(r)? as usize,
        num_kv_heads: read_u64(r)? as usize,
        head_dim: read_u64(r)? as usize,
        intermediate: read_u64(r)? as usize,
        vocab: read_u64(r)? as usize,
        max_seq: read_u64(r)? as usize,
        rope_theta: read_f32(r)?,
        eps: read_f32(r)?,
    })
}

fn write_u32<W: Write>(w: &mut W, value: u32) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    r.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn write_u64<W: Write>(w: &mut W, value: u64) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn write_f32<W: Write>(w: &mut W, value: f32) -> std::io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut bytes = [0u8; 4];
    r.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn write_f32_slice<W: Write>(w: &mut W, values: &[f32]) -> std::io::Result<()> {
    write_u64(w, values.len() as u64)?;
    for &value in values {
        write_f32(w, value)?;
    }
    Ok(())
}

fn read_f32_slice<R: Read>(r: &mut R, values: &mut [f32], label: &str) -> Result<()> {
    let count = read_u64(r)? as usize;
    if count != values.len() {
        return Err(FerrisResError::Shape(format!(
            "{} has {} elements, model expects {}",
            label,
            count,
            values.len()
        )));
    }
    for value in values.iter_mut() {
        *value = read_f32(r)?;
    }
    Ok(())
}
