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
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::{FerrisResError, Result};
use crate::multi_gpu::{
    check_resume_world_size, identity, InitIdentity, RunParallelism, ShardPlan, WeightExchange,
    DEFAULT_EXCHANGE_TIMEOUT,
};

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
    /// This process's rank in `0..world_size`. Zero for a single process.
    pub rank: usize,
    /// Number of cooperating processes. One keeps the unsharded schedule.
    ///
    /// For `world_size > 1` the schedule shards *windows*, not optimizer steps:
    /// every rank runs the same global optimizer steps, and at step `s` the
    /// ranks own `world_size` consecutive windows (`batch_size` of them per
    /// rank). The union covers each corpus window exactly as often as the
    /// single-process loop does, and [`ShardPlan`] makes the mapping a pure
    /// function of `(seed, step, batch slot)`, so a resume reproduces the tail.
    pub world_size: usize,
    /// Directory for the weight-exchange files and per-rank evidence. Required
    /// when `world_size > 1`.
    pub exchange_dir: Option<PathBuf>,
    /// Publish and average parameters every N optimizer steps. Zero disables
    /// the exchange. The last step always exchanges when the world size is
    /// greater than one.
    pub exchange_every: u64,
}

/// One rank's evidence file, `<exchange-dir>/rank-<i>.json`.
///
/// `adapter` is `"cpu"` because this path is the f32 CPU trainer; there is no
/// GPU adapter behind it and the field exists so a consumer cannot mistake it
/// for wgpu training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankEvidence {
    pub rank: usize,
    pub world_size: usize,
    /// Optimizer steps run by this invocation.
    pub steps: u64,
    pub resumed_from: u64,
    pub total_step: u64,
    pub initial_loss: f32,
    pub final_loss: f32,
    /// The last step's loss, repeated for consumers that want one number.
    pub loss: f32,
    /// `steps * batch_size * seq_len` for this rank.
    pub tokens: u64,
    pub tokens_per_second: f64,
    pub elapsed_seconds: f64,
    pub adapter: &'static str,
    pub checkpoint: String,
}

/// Header of a checkpoint, without loading it into a model.
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub step: u64,
    pub parallelism: RunParallelism,
    pub param_count: usize,
    pub config: F32Config,
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
/// Version 2 adds the `(world_size, rank)` pair so a resume can refuse a
/// changed world size. Version 1 files still load as `(1, 0)`.
const CHECKPOINT_VERSION: u32 = 2;
const ADAM_BETA1: f32 = 0.9;
const ADAM_BETA2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Train `tokens` with the reference model and return the measured losses.
///
/// `tokens` must contain at least `seq_len + 1` values, and every value must
/// be a valid index into `options.config.vocab`.
pub fn run(options: &TrainOptions, tokens: &[u32]) -> Result<TrainReport> {
    validate(options, tokens)?;
    let started = Instant::now();
    let cfg = options.config;
    let world_size = options.world_size;
    let parallelism = RunParallelism::new(world_size, options.rank)?;

    // `world_size == 1` takes exactly the schedule this loop had before the
    // multi-GPU layer existed: no plan, no exchange, one output path.
    let plan = if world_size > 1 {
        Some(ShardPlan::new(world_size, options.rank, options.seed)?)
    } else {
        None
    };
    let exchange = match &options.exchange_dir {
        Some(dir) => Some(WeightExchange::with_default_timeout(dir, world_size, options.rank)?),
        None if world_size > 1 => {
            return Err(FerrisResError::Shape(format!(
                "--world-size {} needs --exchange-dir: the ranks average parameters \
                 through files in one shared directory",
                world_size
            )))
        }
        None => None,
    };

    let (mut model, mut adam, start_step) = match &options.resume {
        Some(path) => {
            let (model, adam, step, recorded) = load_checkpoint(path)?;
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
            // Restored first, then refused: the model and Adam state are known
            // to be loadable before the parallelism is judged.
            check_resume_world_size(recorded.world_size, world_size)?;
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

    if world_size > 1 {
        verify_init_identity(&model, options, start_step, exchange.as_ref())?;
    }

    let windows_per_epoch = (tokens.len() - 1) / options.seq_len;
    let total_step = match plan {
        None => start_step + options.epochs as u64 * windows_per_epoch as u64,
        Some(_) => {
            // Each global optimizer step consumes `batch_size` windows per
            // rank. One epoch's windows therefore divide across the ranks, not
            // across batch_size, so that is the only divisibility that matters.
            let wanted = options.epochs as u64 * windows_per_epoch as u64;
            if wanted % world_size as u64 != 0 {
                return Err(FerrisResError::Shape(format!(
                    "{} window(s) over {} epoch(s) do not divide across world_size {}; \
                     adjust --epochs, --seq-len or --data",
                    wanted, options.epochs, world_size
                )));
            }
            start_step + wanted / world_size as u64
        }
    };
    let before = params_fingerprint(&model);

    let mut initial_loss = f32::NAN;
    let mut final_loss = f32::NAN;

    for step in (start_step + 1)..=total_step {
        let mut grads = Gradients::zeros(&cfg);
        let mut step_loss = 0.0f32;

        for b in 0..options.batch_size {
            let window = match &plan {
                // A shard step is `(optimizer step, batch slot)`, so the plan's
                // contiguous group of modes still covers consecutive windows.
                Some(plan) => {
                    let shard_step = (step - 1) * options.batch_size as u64 + b as u64;
                    (plan.batch_index(shard_step) % windows_per_epoch as u64) as usize
                }
                None => ((step - 1) as usize + b) % windows_per_epoch,
            };
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

        if plan.is_some()
            && options.exchange_every > 0
            && (step % options.exchange_every == 0 || step == total_step)
        {
            if let Some(exchange) = &exchange {
                // Local SGD: parameters are averaged, Adam moments are not.
                // The final step always exchanges so the ranks end bit-identical
                // and a checkpoint written now is a valid resume point.
                let local = model.flat_params();
                let averaged = exchange.round(step, &local)?;
                model.set_flat_params(&averaged)?;
                tracing::info!(
                    event = "weight_exchange_complete_step",
                    "rank {}/{} averaged {} parameters at step {}",
                    options.rank,
                    world_size,
                    averaged.len(),
                    step
                );
            }
        }

        if options.log_every > 0 && (step % options.log_every == 0 || step == total_step) {
            tracing::info!(
                event = "train_step_loss",
                "step {}/{} loss={:.6}",
                step,
                total_step,
                mean_loss
            );
        }

        let checkpoint = checkpoint_path(&options.output, world_size, options.rank);
        if options.checkpoint_every > 0 && step % options.checkpoint_every == 0 {
            save_checkpoint(&checkpoint, &model, &adam, step, parallelism)?;
            tracing::info!(
                event = "checkpoint_saved_step_path",
                "Checkpoint saved: step={} path={}",
                step,
                checkpoint.display()
            );
        }
    }

    let checkpoint = checkpoint_path(&options.output, world_size, options.rank);
    save_checkpoint(&checkpoint, &model, &adam, total_step, parallelism)?;

    let report = TrainReport {
        initial_loss,
        final_loss,
        steps_run: total_step - start_step,
        resumed_from: start_step,
        total_step,
        params_changed: params_fingerprint(&model) != before,
        checkpoint,
    };
    tracing::info!(
        event = "training_complete_initial_final_loss",
        "Training complete: steps={} initial_loss={:.6} final_loss={:.6}",
        report.steps_run,
        report.initial_loss,
        report.final_loss
    );

    // The launcher reads this file to aggregate throughput; it is optional so
    // a plain `ferrisres train` invocation stays file-free.
    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64().max(1e-9);
    let tokens = report.steps_run * options.batch_size as u64 * options.seq_len as u64;
    if let Ok(path) = std::env::var("FERRISRES_TOKENS_FILE") {
        fs::write(path, format!("{:.6}\n", tokens as f64 / seconds))?;
    }
    if let Some(exchange) = &exchange {
        write_rank_evidence(exchange.dir(), options, &report, seconds)?;
    }
    Ok(report)
}

/// Publish rank 0's identity, then hold every rank to it.
///
/// Rank 0 publishes its own `(seed, hash)`; the others wait for that file and
/// compare. A disagreement is a named refusal, never a warning: averaging
/// ranks that started from different weights produces a model that never
/// existed.
fn verify_init_identity(
    model: &F32Model,
    options: &TrainOptions,
    start_step: u64,
    exchange: Option<&WeightExchange>,
) -> Result<()> {
    let exchange = exchange.ok_or_else(|| {
        FerrisResError::Shape("parallel training has no exchange directory".to_string())
    })?;
    let flat = model.flat_params();
    let local = InitIdentity::from_params(options.seed, &flat);
    let run_id = std::env::var("FERRISRES_RUN_ID").ok();
    let reference = if options.rank == 0 {
        identity::publish_reference(exchange.dir(), &local, run_id.as_deref(), start_step)?;
        local
    } else {
        identity::await_reference(
            exchange.dir(),
            run_id.as_deref(),
            options.seed,
            start_step,
            DEFAULT_EXCHANGE_TIMEOUT,
        )?
    };
    reference.verify_rank(options.rank, &local)?;
    tracing::info!(
        event = "init_identity_verified_rank",
        "rank {}/{} init identity verified (seed={}, hash={:#018x})",
        options.rank,
        options.world_size,
        local.seed,
        local.param_hash
    );
    Ok(())
}

/// `world_size > 1` writes one checkpoint per rank so the processes do not
/// clobber each other. A single process keeps the exact path it was given.
fn checkpoint_path(base: &Path, world_size: usize, rank: usize) -> PathBuf {
    if world_size > 1 {
        let mut name = base.as_os_str().to_os_string();
        name.push(format!(".rank-{rank}"));
        PathBuf::from(name)
    } else {
        base.to_path_buf()
    }
}

fn write_rank_evidence(
    dir: &Path,
    options: &TrainOptions,
    report: &TrainReport,
    seconds: f64,
) -> Result<()> {
    let tokens = report.steps_run * options.batch_size as u64 * options.seq_len as u64;
    let evidence = RankEvidence {
        rank: options.rank,
        world_size: options.world_size,
        steps: report.steps_run,
        resumed_from: report.resumed_from,
        total_step: report.total_step,
        initial_loss: report.initial_loss,
        final_loss: report.final_loss,
        loss: report.final_loss,
        tokens,
        tokens_per_second: tokens as f64 / seconds,
        elapsed_seconds: seconds,
        adapter: "cpu",
        checkpoint: report.checkpoint.display().to_string(),
    };
    let path = dir.join(format!("rank-{}.json", options.rank));
    let bytes = serde_json::to_vec_pretty(&evidence)
        .map_err(|err| FerrisResError::Shape(err.to_string()))?;
    fs::write(&path, bytes)?;
    Ok(())
}

fn validate(options: &TrainOptions, tokens: &[u32]) -> Result<()> {
    if options.seq_len == 0 || options.seq_len > options.config.max_seq {
        return Err(FerrisResError::Shape(format!(
            "seq_len {} must be in 1..={} (the model's max_seq)",
            options.seq_len, options.config.max_seq
        )));
    }
    if options.batch_size == 0 {
        return Err(FerrisResError::Shape("batch_size must be > 0".to_string()));
    }
    if options.world_size == 0 {
        return Err(FerrisResError::Shape("--world-size must be at least 1".to_string()));
    }
    if options.rank >= options.world_size {
        return Err(FerrisResError::Shape(format!(
            "--rank {} is not in 0..{}",
            options.rank, options.world_size
        )));
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
//   magic[8] = "FRTRAIN1", version u32, step u64,
//   world_size u64, rank u64                    (version 2 and later)
//   config: 8 x u64 (hidden, layers, heads, kv_heads, head_dim, intermediate,
//           vocab, max_seq) + rope_theta f32 + eps f32
//   block: num_blocks u64, layers_per_block u64, boundary count u64, boundaries
//   tensors: count u64, then name (len u64 + UTF-8), element count u64, f32s
//   adam: t u32, m (len u64 + f32s), v (len u64 + f32s)
//
// The GPU [`super::checkpointing::CheckpointStore`] is activation
// checkpointing for the wgpu model, so it does not fit this CPU trainer; the
// parameter layout below is the same stable one `F32Model::tensors` exposes.

fn save_checkpoint(
    path: &Path,
    model: &F32Model,
    adam: &Adam,
    step: u64,
    parallelism: RunParallelism,
) -> Result<()> {
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
    write_u64(&mut w, parallelism.world_size as u64)?;
    write_u64(&mut w, parallelism.rank as u64)?;
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

fn load_checkpoint(path: &Path) -> Result<(F32Model, Adam, u64, RunParallelism)> {
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
    if version != 1 && version != CHECKPOINT_VERSION {
        return Err(FerrisResError::Unsupported(format!(
            "checkpoint version {} is not supported (expected 1 or {})",
            version, CHECKPOINT_VERSION
        )));
    }
    let step = read_u64(&mut r)?;
    // A version 1 file predates parallelism and was necessarily one process.
    let parallelism = if version >= 2 {
        RunParallelism::new(read_u64(&mut r)? as usize, read_u64(&mut r)? as usize)?
    } else {
        RunParallelism::new(1, 0)?
    };
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
    Ok((model, adam, step, parallelism))
}

/// Read a checkpoint's header and parameter count without running training.
///
/// Used by the resume gate: it proves both per-rank files load and records the
/// world size the run used.
pub fn inspect_checkpoint(path: &Path) -> Result<CheckpointInfo> {
    let (model, _adam, step, parallelism) = load_checkpoint(path)?;
    Ok(CheckpointInfo { step, parallelism, param_count: model.param_count(), config: model.config })
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

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "ferrisres-train-loop-{}-{}",
                std::process::id(),
                name
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn tiny_config() -> F32Config {
        F32Config {
            hidden: 8,
            num_layers: 2,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate: 16,
            vocab: 16,
            max_seq: 16,
            rope_theta: 10_000.0,
            eps: 1e-6,
        }
    }

    fn tokens(count: usize) -> Vec<u32> {
        (0..count).map(|t| ((t % 4) + 1) as u32).collect()
    }

    fn options(output: PathBuf) -> TrainOptions {
        TrainOptions {
            config: tiny_config(),
            seq_len: 16,
            batch_size: 1,
            epochs: 1,
            learning_rate: 1e-2,
            seed: 7,
            checkpoint_every: 0,
            output,
            resume: None,
            log_every: 0,
            rank: 0,
            world_size: 1,
            exchange_dir: None,
            exchange_every: 5,
        }
    }

    /// One process must not change behaviour just because the parallel flags
    /// exist: same seed, same tokens, same losses, same output path.
    #[test]
    fn world_size_one_matches_the_unsharded_schedule() {
        let temp = TempDir::new("world-one");
        let plain = options(temp.path().join("plain.bin"));
        let plain_report = run(&plain, &tokens(33)).expect("plain run");

        let mut flagged = options(temp.path().join("flagged.bin"));
        flagged.exchange_dir = Some(temp.path().join("exchange"));
        let flagged_report = run(&flagged, &tokens(33)).expect("flagged run");

        assert_eq!(plain_report.steps_run, flagged_report.steps_run);
        assert_eq!(plain_report.total_step, flagged_report.total_step);
        assert_eq!(plain_report.initial_loss.to_bits(), flagged_report.initial_loss.to_bits());
        assert_eq!(plain_report.final_loss.to_bits(), flagged_report.final_loss.to_bits());
        // No sharding means no per-rank checkpoint suffix and no weight files.
        assert_eq!(flagged_report.checkpoint, temp.path().join("flagged.bin"));
        assert!(flagged_report.checkpoint.is_file());
        assert!(!temp.path().join("flagged.bin.rank-0").exists());
        let weights: Vec<_> = fs::read_dir(temp.path().join("exchange"))
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with("weights-"))
            .collect();
        assert!(weights.is_empty(), "a single rank must not exchange: {weights:?}");
        assert!(temp.path().join("exchange/rank-0.json").is_file());
    }

    #[test]
    fn two_rank_threads_exchange_weights_and_write_distinct_evidence_or_refuse() {
        let temp = TempDir::new("two-ranks");
        let exchange = temp.path().join("exchange");
        // seq_len 16 with 33 tokens gives two windows; world 2 gives one step
        // per rank and an exchange on that final step.
        let tokens = tokens(33);
        let dir = exchange.clone();
        let output = temp.path().to_path_buf();

        let results: Vec<Result<TrainReport>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|rank| {
                    let mut opts = options(output.join(format!("rank{rank}.bin")));
                    opts.rank = rank;
                    opts.world_size = 2;
                    opts.exchange_dir = Some(dir.clone());
                    let tokens = &tokens;
                    scope.spawn(move || run(&opts, tokens))
                })
                .collect();
            handles.into_iter().map(|handle| handle.join().expect("rank thread")).collect()
        });

        for (rank, result) in results.iter().enumerate() {
            let report = result.as_ref().unwrap_or_else(|err| panic!("rank {rank}: {err}"));
            assert_eq!(report.steps_run, 1);
            assert!(report.initial_loss.is_finite());
            assert!(report.final_loss.is_finite());
        }

        for rank in 0..2 {
            let bytes = fs::read(exchange.join(format!("rank-{rank}.json"))).unwrap();
            let evidence: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(evidence["rank"], serde_json::json!(rank));
            assert_eq!(evidence["world_size"], serde_json::json!(2));
            assert_eq!(evidence["adapter"], serde_json::json!("cpu"));
            assert!(evidence["loss"].as_f64().unwrap().is_finite());
            assert!(evidence["tokens_per_second"].as_f64().unwrap() > 0.0);
            let info = inspect_checkpoint(&output.join(format!("rank{rank}.bin.rank-{rank}")))
                .expect("per-rank checkpoint loads");
            assert_eq!(info.parallelism.world_size, 2);
            assert_eq!(info.parallelism.rank, rank);
        }
        assert!(exchange.join("weights-1-avg.bin").is_file());
    }
}
