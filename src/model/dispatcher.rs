//! Model architecture dispatcher — auto-detect BlockAttnRes vs Standard from weights.
//!
//! When loading model weights (safetensors or GGUF), this module auto-detects
//! the architecture and instantiates the correct model type. A unified
//! [`AnyModel`] enum wraps both [`BlockAttnResModel`] and
//! [`StandardTransformerModel`] so the TokenGenerator pipeline works with
//! either transparently.
//!
//! CLI usage: `--arch auto|block-attn-res|standard` to override detection.

use std::path::Path;
use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::error::{FerrisResError, Result};
use crate::inference::kv_cache::ModelKVCache;
use crate::model::config::BlockAttnResConfig;
use crate::model::gguf::{load_gguf, GgufFile};
use crate::model::model::BlockAttnResModel;
use crate::model::safetensors::{load_safetensors, LoadedWeights, ModelArchitecture};
use crate::model::standard_transformer::{StandardTransformerConfig, StandardTransformerModel};

// ---------------------------------------------------------------------------
// ArchitectureConfig — user-specified or auto-detected
// ---------------------------------------------------------------------------

/// User preference for architecture selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchitectureHint {
    /// Auto-detect from weight tensor names.
    Auto,
    /// Force BlockAttnRes O(n) mode.
    BlockAttnRes,
    /// Force standard O(n²) transformer.
    Standard,
}

impl ArchitectureHint {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "block-attn-res" | "block_attn_res" | "bar" => Ok(Self::BlockAttnRes),
            "standard" | "std" => Ok(Self::Standard),
            other => Err(FerrisResError::Shape(format!(
                "Unknown architecture hint '{}'. Use: auto, block-attn-res, standard", other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// ArchitectureDetector — inspect weights to determine model type
// ---------------------------------------------------------------------------

/// Detects model architecture from weight tensor names.
pub struct ArchitectureDetector;

impl ArchitectureDetector {
    /// Detect architecture from safetensors weights.
    pub fn detect_from_safetensors(weights: &LoadedWeights, hint: &ArchitectureHint) -> DetectedArchitecture {
        match hint {
            ArchitectureHint::BlockAttnRes => {
                return DetectedArchitecture::BlockAttnRes;
            }
            ArchitectureHint::Standard => {
                return DetectedArchitecture::Standard;
            }
            ArchitectureHint::Auto => {}
        }

        // Use the safetensors built-in detector
        let arch = weights.detect_architecture();
        match arch {
            ModelArchitecture::BlockAttnRes => DetectedArchitecture::BlockAttnRes,
            ModelArchitecture::Llama
            | ModelArchitecture::Mistral
            | ModelArchitecture::GptNeoX
            | ModelArchitecture::Standard => DetectedArchitecture::Standard,
            ModelArchitecture::Unknown => {
                // Heuristic: check tensor name patterns
                let names = weights.tensor_names();
                Self::detect_from_names(&names)
            }
        }
    }

    /// Detect architecture from GGUF file.
    pub fn detect_from_gguf(gguf: &GgufFile, hint: &ArchitectureHint) -> DetectedArchitecture {
        match hint {
            ArchitectureHint::BlockAttnRes => return DetectedArchitecture::BlockAttnRes,
            ArchitectureHint::Standard => return DetectedArchitecture::Standard,
            ArchitectureHint::Auto => {}
        }

        let arch = gguf.architecture();
        match arch {
            "block-attn-res" | "block_attn_res" => DetectedArchitecture::BlockAttnRes,
            "llama" | "mistral" | "gpt-neox" | "gpt_neox" | "falcon" | "mpt" | "gemma" => {
                DetectedArchitecture::Standard
            }
            _ => {
                // Fallback: GGUF files are standard transformers by default
                DetectedArchitecture::Standard
            }
        }
    }

    /// Detect architecture from tensor name patterns.
    fn detect_from_names(names: &[&str]) -> DetectedArchitecture {
        // BlockAttnRes uses: layers.N.intra_block_layers.M.xxx
        let has_block_attn = names.iter().any(|n| n.contains("intra_block"));
        if has_block_attn {
            return DetectedArchitecture::BlockAttnRes;
        }

        // Standard patterns: q_proj/k_proj/v_proj or attn_q/attn_k/attn_v
        let has_standard = names.iter().any(|n| {
            n.contains("q_proj") || n.contains("attn_q") ||
            n.contains("self_attn.q_proj") || n.contains("model.layers")
        });
        if has_standard {
            return DetectedArchitecture::Standard;
        }

        // Default to standard (most common)
        DetectedArchitecture::Standard
    }
}

/// The detected architecture type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectedArchitecture {
    BlockAttnRes,
    Standard,
}

// ---------------------------------------------------------------------------
// AnyModel — unified enum wrapping both model types
// ---------------------------------------------------------------------------

/// A unified model type that can be either BlockAttnRes or Standard.
///
/// This enum provides a common interface so the TokenGenerator pipeline
/// can work with either architecture transparently.
pub enum AnyModel {
    BlockAttnRes(BlockAttnResModel),
    Standard(StandardTransformerModel),
}

/// Configuration for model creation (architecture-agnostic).
#[derive(Debug, Clone)]
pub struct AnyModelConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub intermediate_dim: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

impl AnyModel {
    /// Create a BlockAttnRes variant.
    pub fn new_block_attn_res(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: BlockAttnResConfig,
        vocab_size: usize,
    ) -> Result<Self> {
        let model = BlockAttnResModel::new(device, queue, config, vocab_size)?;
        Ok(Self::BlockAttnRes(model))
    }

    /// Create a Standard transformer variant.
    pub fn new_standard(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: StandardTransformerConfig,
    ) -> Result<Self> {
        let model = StandardTransformerModel::new(device, queue, config)?;
        Ok(Self::Standard(model))
    }

    /// Auto-detect and create from a safetensors file.
    pub fn from_safetensors(
        path: &Path,
        device: Arc<Device>,
        queue: Arc<Queue>,
        hint: &ArchitectureHint,
    ) -> Result<Self> {
        let weights = load_safetensors(path)?;
        let detected = ArchitectureDetector::detect_from_safetensors(&weights, hint);

        match detected {
            DetectedArchitecture::BlockAttnRes => {
                let hidden_dim = weights.infer_hidden_dim().unwrap_or(512);
                let vocab_size = weights.infer_vocab_size().unwrap_or(32000);
                let config = BlockAttnResConfig::new(hidden_dim);
                Self::new_block_attn_res(device, queue, config, vocab_size)
            }
            DetectedArchitecture::Standard => {
                let config = infer_standard_config(&weights);
                Self::new_standard(device, queue, config)
            }
        }
    }

    /// Auto-detect and create from a GGUF file.
    pub fn from_gguf(
        path: &Path,
        device: Arc<Device>,
        queue: Arc<Queue>,
        hint: &ArchitectureHint,
    ) -> Result<Self> {
        let gguf = load_gguf(path)?;
        let detected = ArchitectureDetector::detect_from_gguf(&gguf, hint);

        match detected {
            DetectedArchitecture::BlockAttnRes => {
                let hidden_dim = gguf.infer_hidden_dim().unwrap_or(512);
                let vocab_size = gguf.infer_vocab_size().unwrap_or(32000);
                let config = BlockAttnResConfig::new(hidden_dim);
                Self::new_block_attn_res(device, queue, config, vocab_size)
            }
            DetectedArchitecture::Standard => {
                let config = infer_standard_config_gguf(&gguf);
                Self::new_standard(device, queue, config)
            }
        }
    }

    /// Auto-detect from file extension and load.
    pub fn from_path(
        path: &Path,
        device: Arc<Device>,
        queue: Arc<Queue>,
        hint: &ArchitectureHint,
    ) -> Result<Self> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext {
            "safetensors" => Self::from_safetensors(path, device, queue, hint),
            "gguf" => Self::from_gguf(path, device, queue, hint),
            _ => {
                // Try safetensors first, then GGUF
                if let Ok(m) = Self::from_safetensors(path, device.clone(), queue.clone(), hint) {
                    return Ok(m);
                }
                Self::from_gguf(path, device, queue, hint)
            }
        }
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        match self {
            Self::BlockAttnRes(m) => m.layers().len(),
            Self::Standard(m) => m.num_layers(),
        }
    }

    /// Hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        match self {
            Self::BlockAttnRes(m) => m.config().hidden_dim,
            Self::Standard(m) => m.config().hidden_dim,
        }
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        match self {
            Self::BlockAttnRes(m) => m.config().attention_heads,
            Self::Standard(m) => m.config().num_heads,
        }
    }

    /// Create a KV cache sized for this model.
    pub fn create_kv_cache(
        &self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        max_seq_len: u32,
    ) -> Result<ModelKVCache> {
        ModelKVCache::new(
            device,
            queue,
            self.num_layers() as u32,
            max_seq_len,
            self.num_heads() as u32,
            self.hidden_dim() as u32,
        )
    }

    /// Get the detected architecture type.
    pub fn architecture(&self) -> DetectedArchitecture {
        match self {
            Self::BlockAttnRes(_) => DetectedArchitecture::BlockAttnRes,
            Self::Standard(_) => DetectedArchitecture::Standard,
        }
    }

    /// Get the common config.
    pub fn any_config(&self) -> AnyModelConfig {
        match self {
            Self::BlockAttnRes(m) => {
                let c = m.config();
                AnyModelConfig {
                    hidden_dim: c.hidden_dim,
                    num_heads: c.attention_heads,
                    num_layers: c.num_layers,
                    intermediate_dim: c.intermediate_dim,
                    head_dim: c.hidden_dim / c.attention_heads,
                    vocab_size: 0, // BlockAttnResConfig doesn't track vocab_size
                }
            }
            Self::Standard(m) => {
                let c = m.config();
                AnyModelConfig {
                    hidden_dim: c.hidden_dim,
                    num_heads: c.num_heads,
                    num_layers: c.num_layers,
                    intermediate_dim: c.intermediate_dim,
                    head_dim: c.head_dim,
                    vocab_size: c.vocab_size,
                }
            }
        }
    }

    /// Forward decode for a single token through all layers (decode path).
    ///
    /// Returns the output hidden states after all layers.
    /// The caller handles embedding lookup and LM head projection.
    pub fn forward_decode_layer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &crate::compute::GpuBuffer,
        kv_cache: &mut crate::inference::kv_cache::ModelKVCache,
        layer_idx: usize,
        effective_pos: Option<u32>,
    ) -> Result<crate::compute::GpuBuffer> {
        match self {
            Self::BlockAttnRes(m) => {
                let layer = &m.layers()[layer_idx];
                let cache = kv_cache.layer(layer_idx);
                layer.forward_decode_token_direct(encoder, hidden_states, cache, effective_pos)
            }
            Self::Standard(m) => {
                let layer = &m.layers()[layer_idx];
                let cache = kv_cache.layer(layer_idx);
                layer.forward_decode_token_direct(encoder, hidden_states, cache, effective_pos)
            }
        }
    }

    /// Forward prefill through all layers.
    pub fn forward_prefill_layer(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &crate::compute::GpuBuffer,
        kv_cache: &mut crate::inference::kv_cache::ModelKVCache,
        layer_idx: usize,
        seq_len: u32,
    ) -> Result<crate::compute::GpuBuffer> {
        match self {
            Self::BlockAttnRes(m) => {
                let layer = &m.layers()[layer_idx];
                let cache = kv_cache.layer(layer_idx);
                layer.forward_prefill(encoder, hidden_states, cache, seq_len)
            }
            Self::Standard(m) => {
                let layer = &m.layers()[layer_idx];
                let cache = kv_cache.layer(layer_idx);
                layer.forward_prefill(encoder, hidden_states, cache, seq_len)
            }
        }
    }

    /// Access the underlying BlockAttnResModel if applicable.
    pub fn as_block_attn_res(&self) -> Option<&BlockAttnResModel> {
        match self {
            Self::BlockAttnRes(m) => Some(m),
            _ => None,
        }
    }

    /// Access the underlying StandardTransformerModel if applicable.
    pub fn as_standard(&self) -> Option<&StandardTransformerModel> {
        match self {
            Self::Standard(m) => Some(m),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper inference functions
// ---------------------------------------------------------------------------

fn infer_num_heads(weights: &LoadedWeights, hidden_dim: usize) -> usize {
    // Try to find num_heads from weight shape
    for name in &["q_proj.weight", "layers.0.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"] {
        if let Some(t) = weights.get(name) {
            if t.shape.len() == 2 {
                let out_dim = t.shape[0];
                if out_dim > 0 && hidden_dim > 0 {
                    return hidden_dim / (hidden_dim / out_dim.max(1)).max(1);
                }
            }
        }
    }
    // Default: assume head_dim = 128 (LLaMA/Mistral common)
    (hidden_dim / 128).max(1)
}

fn infer_num_heads_gguf(gguf: &GgufFile, hidden_dim: usize) -> usize {
    gguf.metadata_u32("llama.attention.head_count")
        .map(|n| n as usize)
        .unwrap_or_else(|| (hidden_dim / 128).max(1))
}

fn infer_standard_config(weights: &LoadedWeights) -> StandardTransformerConfig {
    let hidden_dim = weights.infer_hidden_dim().unwrap_or(512);
    let num_layers = weights.infer_num_layers();
    let vocab_size = weights.infer_vocab_size().unwrap_or(32000);
    let num_heads = infer_num_heads(weights, hidden_dim);
    StandardTransformerConfig::from_inferred(hidden_dim, num_heads, num_layers, vocab_size)
}

fn infer_standard_config_gguf(gguf: &GgufFile) -> StandardTransformerConfig {
    let hidden_dim = gguf.infer_hidden_dim().unwrap_or(512);
    let num_layers = gguf.infer_num_layers();
    let vocab_size = gguf.infer_vocab_size().unwrap_or(32000);
    let num_heads = infer_num_heads_gguf(gguf, hidden_dim);
    StandardTransformerConfig::from_inferred(hidden_dim, num_heads, num_layers, vocab_size)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_hint_from_str() {
        assert_eq!(ArchitectureHint::from_str("auto").unwrap(), ArchitectureHint::Auto);
        assert_eq!(ArchitectureHint::from_str("block-attn-res").unwrap(), ArchitectureHint::BlockAttnRes);
        assert_eq!(ArchitectureHint::from_str("block_attn_res").unwrap(), ArchitectureHint::BlockAttnRes);
        assert_eq!(ArchitectureHint::from_str("standard").unwrap(), ArchitectureHint::Standard);
        assert_eq!(ArchitectureHint::from_str("std").unwrap(), ArchitectureHint::Standard);
        assert!(ArchitectureHint::from_str("unknown").is_err());
    }

    #[test]
    fn test_detect_from_names_block_attn_res() {
        let names = vec![
            "layers.0.intra_block_layers.0.q_proj.weight",
            "layers.0.intra_block_layers.0.k_proj.weight",
            "embedding.weight",
        ];
        assert_eq!(
            ArchitectureDetector::detect_from_names(&names),
            DetectedArchitecture::BlockAttnRes
        );
    }

    #[test]
    fn test_detect_from_names_standard_q_proj() {
        let names = vec![
            "layers.0.q_proj.weight",
            "layers.0.k_proj.weight",
            "layers.0.v_proj.weight",
            "embedding.weight",
        ];
        assert_eq!(
            ArchitectureDetector::detect_from_names(&names),
            DetectedArchitecture::Standard
        );
    }

    #[test]
    fn test_detect_from_names_standard_attn_q() {
        let names = vec![
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "token_embd.weight",
        ];
        assert_eq!(
            ArchitectureDetector::detect_from_names(&names),
            DetectedArchitecture::Standard
        );
    }

    #[test]
    fn test_detect_from_names_standard_model_layers() {
        let names = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.embed_tokens.weight",
        ];
        assert_eq!(
            ArchitectureDetector::detect_from_names(&names),
            DetectedArchitecture::Standard
        );
    }

    #[test]
    fn test_detect_from_names_default_standard() {
        let names = vec!["some_tensor.weight", "other.weight"];
        assert_eq!(
            ArchitectureDetector::detect_from_names(&names),
            DetectedArchitecture::Standard
        );
    }

    #[test]
    fn test_hint_overrides_detection() {
        use crate::model::safetensors::LoadedTensor;
        let weights = LoadedWeights {
            tensors: {
                let mut m = std::collections::HashMap::new();
                m.insert("blk.0.attn_q.weight".into(), LoadedTensor {
                    name: "blk.0.attn_q.weight".into(),
                    shape: vec![512, 512],
                    dtype: "F32".into(),
                    data: vec![0.0f32; 512 * 512],
                });
                m
            },
            source_files: vec![],
        };
        // Auto: would detect Standard from names
        assert_eq!(
            ArchitectureDetector::detect_from_safetensors(&weights, &ArchitectureHint::Auto),
            DetectedArchitecture::Standard
        );
        // Override to BlockAttnRes
        assert_eq!(
            ArchitectureDetector::detect_from_safetensors(&weights, &ArchitectureHint::BlockAttnRes),
            DetectedArchitecture::BlockAttnRes
        );
        // Override to Standard
        assert_eq!(
            ArchitectureDetector::detect_from_safetensors(&weights, &ArchitectureHint::Standard),
            DetectedArchitecture::Standard
        );
    }

    #[test]
    fn test_any_model_config() {
        let config = StandardTransformerConfig::llama_7b();
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.num_heads, 32);
    }

    #[test]
    fn test_file_extension_dispatch() {
        let path = std::path::PathBuf::from("model.gguf");
        assert_eq!(
            path.extension().and_then(|e| e.to_str()).unwrap_or(""),
            "gguf"
        );
        let path = std::path::PathBuf::from("model.safetensors");
        assert_eq!(
            path.extension().and_then(|e| e.to_str()).unwrap_or(""),
            "safetensors"
        );
    }
}
