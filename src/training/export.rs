//! Export an `FRTRAIN1` training checkpoint to an inference-ready model.
//!
//! `train` writes a flat f32 checkpoint ([`super::train_loop`]) whose tensor
//! names are the trainer's internal layout (`embed`, `layer0.q`, ...). The CPU
//! Block AttnRes loader reads the safetensors + JSON convention defined in
//! [`crate::model::checkpoint`] (`embed_tokens`, `layers.0.q_proj`, ...).
//!
//! This module is the bridge. It reads the checkpoint, maps every tensor by
//! name and shape, and writes:
//!
//! - `<output>.safetensors` — f32 tensors in the `load_model` convention
//! - `<output>.json` — [`BlockMoeResConfig`] in the `load_model` convention
//!
//! # Why f32 and not bf16
//!
//! [`crate::model::checkpoint::save_model`] writes bf16, which is the right
//! choice for shipping a large model. An export of a reference checkpoint must
//! be lossless instead: the round-trip test compares tensors exactly, so this
//! path writes `F32`. The naming and config conventions are the same.
//!
//! # Explicit differences from the trainer
//!
//! - `F32Model` has a single per-head Q/K norm of length `head_dim`; the CPU
//!   loader applies the first `head_dim` weights of each head, so the tensor is
//!   copied as-is (not broadcast and not dropped).
//! - `F32Model` has `layer_scalar` as a tensor; [`BlockMoeResConfig`] carries
//!   it per layer, so it is written to the config and not to the weights.
//! - The CPU loader requires `block_config.attn_res_proj`, which the reference
//!   trainer does not train and never reads. It is written as the identity so
//!   the loader has a valid tensor; it is counted as a synthetic tensor.
//! - The CPU loader hardcodes RMSNorm `eps = 1e-6`. A checkpoint with another
//!   `eps` cannot be represented, so export refuses with a named error instead
//!   of silently changing the maths.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::error::{FerrisResError, Result};
use crate::model::checkpoint::{BlockMoeResConfig, LayerConfig};
use crate::model::safetensors::{write_safetensors, SafeDtype, TensorToWrite};
use crate::training::f32_trainer::{F32BlockConfig, F32Config};

const CHECKPOINT_MAGIC: &[u8; 8] = b"FRTRAIN1";
const CHECKPOINT_VERSION: u32 = 1;
/// The CPU Block AttnRes loader constructs every `CpuRmsNorm` with this value.
const CPU_RMS_EPS: f32 = 1e-6;

/// A parsed `FRTRAIN1` checkpoint. The Adam state is deliberately not kept:
/// export is an inference operation.
#[derive(Debug, Clone)]
pub struct TrainedCheckpoint {
    pub step: u64,
    pub config: F32Config,
    pub block_config: F32BlockConfig,
    /// Tensors in the checkpoint's declared order, with flat f32 data.
    pub tensors: Vec<(String, Vec<f32>)>,
}

/// What one export wrote. Serialized as the CLI's JSON summary.
#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub checkpoint: PathBuf,
    pub safetensors_path: PathBuf,
    pub config_path: PathBuf,
    pub step: u64,
    /// Tensors written, including the synthetic `block_config.attn_res_proj`.
    pub tensor_count: usize,
    /// Scalar values written, including the synthetic identity.
    pub parameter_count: usize,
    /// Tensors read from the `FRTRAIN1` checkpoint.
    pub checkpoint_tensor_count: usize,
    /// Scalar values read from the `FRTRAIN1` checkpoint.
    pub checkpoint_parameter_count: usize,
    pub bytes_written: u64,
}

/// Read an `FRTRAIN1` checkpoint without touching the trainer.
///
/// The tensor order and lengths are derived from the config, then checked
/// against the file. The trailing Adam state is ignored.
pub fn read_checkpoint(path: &Path) -> Result<TrainedCheckpoint> {
    let file = File::open(path)
        .map_err(|e| export_error("checkpoint_open", format!("{}: {}", path.display(), e)))?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != CHECKPOINT_MAGIC {
        return Err(export_error(
            "checkpoint_magic",
            format!("{} is not a FerrisRes FRTRAIN1 checkpoint", path.display()),
        ));
    }
    let version = read_u32(&mut r)?;
    if version != CHECKPOINT_VERSION {
        return Err(export_error(
            "checkpoint_version",
            format!("version {} is not supported (expected {})", version, CHECKPOINT_VERSION),
        ));
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

    let layout = tensor_layout(&config);
    let tensor_count = read_u64(&mut r)? as usize;
    if tensor_count != layout.len() {
        return Err(export_error(
            "checkpoint_tensor_count",
            format!("checkpoint has {} tensors, config implies {}", tensor_count, layout.len()),
        ));
    }

    let mut tensors = Vec::with_capacity(tensor_count);
    for (expected_name, expected_len) in layout {
        let name_len = read_u64(&mut r)? as usize;
        let mut name = vec![0u8; name_len];
        r.read_exact(&mut name)?;
        let name = String::from_utf8(name)
            .map_err(|_| export_error("checkpoint_name", "tensor name is not UTF-8".into()))?;
        if name != expected_name {
            return Err(export_error(
                "checkpoint_tensor_name",
                format!("expected `{}`, found `{}`", expected_name, name),
            ));
        }
        let len = read_u64(&mut r)? as usize;
        if len != expected_len {
            return Err(export_error(
                "checkpoint_tensor_length",
                format!("`{}` has {} elements, config implies {}", name, len, expected_len),
            ));
        }
        let mut values = vec![0.0f32; len];
        for value in values.iter_mut() {
            *value = read_f32(&mut r)?;
        }
        tensors.push((name, values));
    }

    Ok(TrainedCheckpoint { step, config, block_config, tensors })
}

/// Convert an `FRTRAIN1` checkpoint into `<output>.safetensors` + `<output>.json`.
///
/// Every checkpoint tensor is either written under its `load_model` name or
/// carried by the config (`layer_scalar`). An unmapped tensor is an error, not
/// a warning.
pub fn export_checkpoint(checkpoint_path: &Path, output_prefix: &Path) -> Result<ExportReport> {
    let checkpoint = read_checkpoint(checkpoint_path)?;
    let cfg = &checkpoint.config;

    validate_mapping(&checkpoint)?;

    let mut by_name: HashMap<&str, &[f32]> = HashMap::with_capacity(checkpoint.tensors.len());
    for (name, values) in &checkpoint.tensors {
        if by_name.insert(name.as_str(), values.as_slice()).is_some() {
            return Err(export_error(
                "checkpoint_duplicate_tensor",
                format!("`{}` appears more than once", name),
            ));
        }
    }

    let take = |name: &str| -> Result<&[f32]> {
        by_name.get(name).copied().ok_or_else(|| {
            export_error("checkpoint_missing_tensor", format!("`{}` is missing", name))
        })
    };

    let mut consumed: HashSet<String> = HashSet::new();
    let mut tensors: Vec<TensorToWrite> = Vec::new();

    let mut push = |name: String, shape: Vec<usize>, data: &[f32]| -> Result<()> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(export_error(
                "tensor_shape_mismatch",
                format!(
                    "`{}` has {} elements but shape {:?} needs {}",
                    name,
                    data.len(),
                    shape,
                    expected
                ),
            ));
        }
        tensors.push(TensorToWrite { name, shape, dtype: SafeDtype::F32, data_f32: data.to_vec() });
        Ok(())
    };

    // Embedding, LM head and final norm.
    consumed.insert("embed".to_string());
    push("embed_tokens".into(), vec![cfg.vocab, cfg.hidden], take("embed")?)?;
    consumed.insert("lm_head".to_string());
    push("lm_head".into(), vec![cfg.hidden, cfg.vocab], take("lm_head")?)?;
    consumed.insert("final_norm".to_string());
    push("final_norm".into(), vec![cfg.hidden], take("final_norm")?)?;

    // Inter-block attention norm; the projection is not trained and not read.
    consumed.insert("attn_res_norm".to_string());
    push("block_config.attn_res_norm".into(), vec![cfg.hidden], take("attn_res_norm")?)?;
    let mut identity = vec![0.0f32; cfg.hidden * cfg.hidden];
    for i in 0..cfg.hidden {
        identity[i * cfg.hidden + i] = 1.0;
    }
    push("block_config.attn_res_proj".into(), vec![cfg.hidden, cfg.hidden], &identity)?;

    // Per-layer weights. Every linear weight is `[in_features, out_features]`
    // in the trainer and in `CpuLinear::from_weight`.
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    for layer in 0..cfg.num_layers {
        let pfx = format!("layers.{}", layer);
        let src = |suffix: &str| format!("layer{}.{}", layer, suffix);

        for (suffix, dst, shape) in [
            ("attn_norm", "attn_norm", vec![cfg.hidden]),
            ("q", "q_proj", vec![cfg.hidden, q_dim]),
            ("k", "k_proj", vec![cfg.hidden, kv_dim]),
            ("v", "v_proj", vec![cfg.hidden, kv_dim]),
            ("q_norm", "q_norm", vec![cfg.head_dim]),
            ("k_norm", "k_norm", vec![cfg.head_dim]),
            ("o", "out_proj", vec![q_dim, cfg.hidden]),
            ("post_attn_norm", "post_attn_norm", vec![cfg.hidden]),
            ("pre_ffn_norm", "pre_ffn_norm", vec![cfg.hidden]),
            ("gate", "ffn_gate", vec![cfg.hidden, cfg.intermediate]),
            ("up", "ffn_up", vec![cfg.hidden, cfg.intermediate]),
            ("down", "ffn_down", vec![cfg.intermediate, cfg.hidden]),
            ("post_ffn_norm", "post_ffn_norm", vec![cfg.hidden]),
        ] {
            let name = src(suffix);
            consumed.insert(name.clone());
            push(format!("{}.{}", pfx, dst), shape, take(&name)?)?;
        }

        // `layer_scalar` lives in the config, not in the weights.
        let scalar_name = src("layer_scalar");
        consumed.insert(scalar_name.clone());
        let scalar = take(&scalar_name)?;
        if scalar.len() != 1 {
            return Err(export_error(
                "layer_scalar_length",
                format!("`{}` must have one element, got {}", scalar_name, scalar.len()),
            ));
        }
    }

    // Every tensor that was read must be represented somewhere. `layer_scalar`
    // is represented by the config, which is checked above.
    if consumed.len() != by_name.len() {
        let missing: Vec<&str> =
            by_name.keys().copied().filter(|name| !consumed.contains(*name)).collect();
        return Err(export_error(
            "checkpoint_tensor_unmapped",
            format!("no export mapping for {:?}", missing),
        ));
    }

    let config = build_config(&checkpoint)?;

    let safetensors_path = with_suffix(output_prefix, ".safetensors");
    let config_path = with_suffix(output_prefix, ".json");
    if let Some(parent) = safetensors_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    write_safetensors(&safetensors_path, &tensors)?;
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| export_error("config_serialize", e.to_string()))?;
    fs::write(&config_path, config_json)
        .map_err(|e| export_error("config_write", format!("{}: {}", config_path.display(), e)))?;

    let bytes_written = fs::metadata(&safetensors_path)?.len() + fs::metadata(&config_path)?.len();
    let parameter_count = tensors.iter().map(|t| t.data_f32.len()).sum();
    let checkpoint_parameter_count = checkpoint.tensors.iter().map(|(_, t)| t.len()).sum();

    let report = ExportReport {
        checkpoint: checkpoint_path.to_path_buf(),
        safetensors_path,
        config_path,
        step: checkpoint.step,
        tensor_count: tensors.len(),
        parameter_count,
        checkpoint_tensor_count: checkpoint.tensors.len(),
        checkpoint_parameter_count,
        bytes_written,
    };

    tracing::info!(
        event = "checkpoint_exported",
        checkpoint = %report.checkpoint.display(),
        safetensors = %report.safetensors_path.display(),
        config = %report.config_path.display(),
        tensors = report.tensor_count,
        parameters = report.parameter_count,
        bytes = report.bytes_written,
        "training checkpoint exported for inference"
    );

    Ok(report)
}

fn validate_mapping(checkpoint: &TrainedCheckpoint) -> Result<()> {
    let cfg = &checkpoint.config;
    if cfg.eps != CPU_RMS_EPS {
        return Err(export_error(
            "eps_not_representable",
            format!(
                "F32Config.eps={} differs from the CPU loader's hardcoded {}",
                cfg.eps, CPU_RMS_EPS
            ),
        ));
    }
    if cfg.num_heads == 0 || cfg.num_kv_heads == 0 {
        return Err(export_error(
            "head_count_zero",
            "num_heads and num_kv_heads must be positive".into(),
        ));
    }
    if cfg.num_heads % cfg.num_kv_heads != 0 {
        return Err(export_error(
            "head_count_indivisible",
            format!(
                "num_heads ({}) is not a multiple of num_kv_heads ({})",
                cfg.num_heads, cfg.num_kv_heads
            ),
        ));
    }
    if cfg.head_dim % 2 != 0 {
        return Err(export_error(
            "head_dim_odd",
            format!("head_dim ({}) must be even for RoPE", cfg.head_dim),
        ));
    }
    if cfg.intermediate == 0 || cfg.hidden == 0 || cfg.vocab == 0 {
        return Err(export_error(
            "dimension_zero",
            "hidden, intermediate and vocab must be positive".into(),
        ));
    }
    Ok(())
}

fn build_config(checkpoint: &TrainedCheckpoint) -> Result<BlockMoeResConfig> {
    let cfg = &checkpoint.config;
    let by_name: HashMap<&str, &[f32]> = checkpoint
        .tensors
        .iter()
        .map(|(name, values)| (name.as_str(), values.as_slice()))
        .collect();

    let mut layer_configs = Vec::with_capacity(cfg.num_layers);
    for layer in 0..cfg.num_layers {
        let name = format!("layer{}.layer_scalar", layer);
        let values = by_name.get(name.as_str()).ok_or_else(|| {
            export_error("checkpoint_missing_tensor", format!("`{}` is missing", name))
        })?;
        if values.len() != 1 {
            return Err(export_error(
                "layer_scalar_length",
                format!("`{}` must have one element, got {}", name, values.len()),
            ));
        }
        layer_configs.push(LayerConfig {
            // The f32 reference trainer uses SwiGLU (`silu(gate) * up`).
            use_gelu: false,
            kv_shared: false,
            is_full_attention: cfg.head_dim != cfg.hidden / cfg.num_heads,
            rope_theta: cfg.rope_theta as f64,
            partial_rotary_factor: 1.0,
            layer_scalar: values[0],
            has_ple: false,
            has_moe: false,
            num_experts: 0,
            top_k: 0,
            inter_dim: cfg.intermediate,
        });
    }

    Ok(BlockMoeResConfig {
        hidden_dim: cfg.hidden,
        vocab_size: cfg.vocab,
        num_layers: cfg.num_layers,
        num_heads: cfg.num_heads,
        num_kv_heads: cfg.num_kv_heads,
        head_dim: cfg.head_dim,
        final_logit_softcapping: None,
        hidden_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        num_blocks: checkpoint.block_config.num_blocks,
        layers_per_block: checkpoint.block_config.layers_per_block,
        boundary_layers: checkpoint.block_config.boundary_layers.clone(),
        layer_configs,
    })
}

/// Names and flat lengths of every tensor the trainer writes, in its order.
fn tensor_layout(cfg: &F32Config) -> Vec<(String, usize)> {
    let h = cfg.hidden;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let inter = cfg.intermediate;

    let mut layout = Vec::with_capacity(4 + cfg.num_layers * 14);
    layout.push(("embed".to_string(), cfg.vocab * h));
    for layer in 0..cfg.num_layers {
        let n = |suffix: &str| format!("layer{}.{}", layer, suffix);
        for (suffix, len) in [
            ("attn_norm", h),
            ("q", h * q_dim),
            ("k", h * kv_dim),
            ("v", h * kv_dim),
            ("q_norm", cfg.head_dim),
            ("k_norm", cfg.head_dim),
            ("o", q_dim * h),
            ("post_attn_norm", h),
            ("pre_ffn_norm", h),
            ("gate", h * inter),
            ("up", h * inter),
            ("down", inter * h),
            ("post_ffn_norm", h),
            ("layer_scalar", 1),
        ] {
            layout.push((n(suffix), len));
        }
    }
    layout.push(("final_norm".to_string(), h));
    layout.push(("lm_head".to_string(), h * cfg.vocab));
    layout.push(("attn_res_norm".to_string(), h));
    layout
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = prefix.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

fn export_error(code: &str, detail: String) -> FerrisResError {
    FerrisResError::Unsupported(format!("export: {}: {}", code, detail))
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

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    r.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut bytes = [0u8; 4];
    r.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}
