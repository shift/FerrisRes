//! Checkpoint serialization for Block-MoE-Res models.
//!
//! Format: standard safetensors with FerrisRes-specific tensor naming convention.
//!
//! Tensor naming:
//!   embed_tokens           [vocab_size, hidden_dim]
//!   lm_head                [hidden_dim, vocab_size]  (or tied to embed_tokens)
//!   final_norm             [hidden_dim]
//!   block_config.attn_res_proj   [hidden_dim, hidden_dim]
//!   block_config.attn_res_norm   [hidden_dim]
//!   layers.{i}.q_proj      [num_heads*head_dim, hidden_dim]
//!   layers.{i}.k_proj      [num_kv_heads*head_dim, hidden_dim]
//!   layers.{i}.v_proj      [num_kv_heads*head_dim, hidden_dim]
//!   layers.{i}.out_proj      [hidden_dim, num_heads*head_dim]
//!   layers.{i}.attn_norm   [hidden_dim]
//!   layers.{i}.post_attn_norm [q_dim]
//!   layers.{i}.ffn_norm / pre_ffn_norm [hidden_dim]
//!   layers.{i}.post_ffn_norm [hidden_dim]
//!   layers.{i}.q_norm      [num_heads, head_dim]
//!   layers.{i}.k_norm      [num_kv_heads, head_dim]
//!   layers.{i}.moe.router  [num_experts, hidden_dim]
//!   layers.{i}.moe.expert.{e}.gate  [inter_dim, hidden_dim]
//!   layers.{i}.moe.expert.{e}.up    [inter_dim, hidden_dim]
//!   layers.{i}.moe.expert.{e}.down  [hidden_dim, inter_dim]
//!   layers.{i}.ple_gate    [ple_dim, hidden_dim]
//!   layers.{i}.ple_proj    [hidden_dim, ple_dim]
//!   layers.{i}.ple_norm    [hidden_dim]
//!
//! Config is stored as a separate JSON file alongside the safetensors.

use std::path::Path;

use crate::error::{FerrisResError, Result};
use crate::model::cpu_block_attn_res::CpuBlockAttnResModel;
use crate::model::safetensors::{TensorToWrite, SafeDtype, write_safetensors};

/// Model configuration stored as JSON alongside the safetensors weights.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlockMoeResConfig {
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub final_logit_softcapping: Option<f32>,
    pub hidden_size_per_layer_input: usize,
    pub num_kv_shared_layers: usize,
    // Block config
    pub num_blocks: usize,
    pub layers_per_block: usize,
    pub boundary_layers: Vec<usize>,
    // Per-layer config (compressed — only stores deviations from defaults)
    pub layer_configs: Vec<LayerConfig>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerConfig {
    pub use_gelu: bool,
    pub kv_shared: bool,
    pub is_full_attention: bool,
    pub rope_theta: f64,
    pub partial_rotary_factor: f32,
    pub layer_scalar: f32,
    pub has_ple: bool,
    pub has_moe: bool,
    pub num_experts: usize,
    pub top_k: usize,
    pub inter_dim: usize,
}

/// Save a complete Block-MoE-Res model to disk.
///
/// Creates two files:
/// - `{path}.safetensors` — model weights
/// - `{path}.json` — model configuration
pub fn save_model(model: &CpuBlockAttnResModel, path: &str) -> Result<()> {
    let st_path = format!("{}.safetensors", path);
    let json_path = format!("{}.json", path);

    let mut tensors = Vec::new();

    // Embedding
    tensors.push(TensorToWrite {
        name: "embed_tokens".into(),
        shape: vec![model.vocab_size, model.hidden_dim],
        dtype: SafeDtype::BF16,
        data_f32: model.embed_tokens.clone(),
    });

    // LM head
    tensors.push(TensorToWrite {
        name: "lm_head".into(),
        shape: vec![model.hidden_dim, model.vocab_size],
        dtype: SafeDtype::BF16,
        data_f32: model.lm_head.clone(),
    });

    // Final norm
    tensors.push(TensorToWrite {
        name: "final_norm".into(),
        shape: vec![model.hidden_dim],
        dtype: SafeDtype::BF16,
        data_f32: model.final_norm.clone(),
    });

    // Block config
    tensors.push(TensorToWrite {
        name: "block_config.attn_res_proj".into(),
        shape: vec![model.hidden_dim, model.hidden_dim],
        dtype: SafeDtype::BF16,
        data_f32: model.block_config.attn_res_proj.clone(),
    });
    tensors.push(TensorToWrite {
        name: "block_config.attn_res_norm".into(),
        shape: vec![model.hidden_dim],
        dtype: SafeDtype::BF16,
        data_f32: model.block_config.attn_res_norm.clone(),
    });

    // Per-layer weights
    for (i, layer) in model.layers.iter().enumerate() {
        let pfx = format!("layers.{}", i);

        // Attention projections
        tensors.push(TensorToWrite {
            name: format!("{}.q_proj", pfx),
            shape: vec![layer.q_proj.out_features(), layer.q_proj.in_features()],
            dtype: SafeDtype::BF16,
            data_f32: layer.q_proj.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.k_proj", pfx),
            shape: vec![layer.k_proj.out_features(), layer.k_proj.in_features()],
            dtype: SafeDtype::BF16,
            data_f32: layer.k_proj.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.v_proj", pfx),
            shape: vec![layer.v_proj.out_features(), layer.v_proj.in_features()],
            dtype: SafeDtype::BF16,
            data_f32: layer.v_proj.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.out_proj", pfx),
            shape: vec![layer.out_proj.out_features(), layer.out_proj.in_features()],
            dtype: SafeDtype::BF16,
            data_f32: layer.out_proj.weight().to_vec(),
        });

        // Norms
        tensors.push(TensorToWrite {
            name: format!("{}.attn_norm", pfx),
            shape: vec![model.hidden_dim],
            dtype: SafeDtype::BF16,
            data_f32: layer.attn_norm.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.post_attn_norm", pfx),
            shape: vec![layer.post_attn_norm.weight().len()],
            dtype: SafeDtype::BF16,
            data_f32: layer.post_attn_norm.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.pre_ffn_norm", pfx),
            shape: vec![model.hidden_dim],
            dtype: SafeDtype::BF16,
            data_f32: layer.pre_ffn_norm.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.post_ffn_norm", pfx),
            shape: vec![model.hidden_dim],
            dtype: SafeDtype::BF16,
            data_f32: layer.post_ffn_norm.weight().to_vec(),
        });

        // Per-head norms
        tensors.push(TensorToWrite {
            name: format!("{}.q_norm", pfx),
            shape: vec![layer.num_heads, layer.head_dim],
            dtype: SafeDtype::BF16,
            data_f32: layer.q_norm.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.k_norm", pfx),
            shape: vec![layer.num_kv_heads, layer.head_dim],
            dtype: SafeDtype::BF16,
            data_f32: layer.k_norm.weight().to_vec(),
        });

        // MoE or dense FFN
        if let Some(ref moe) = layer.moe {
            // Router
            tensors.push(TensorToWrite {
                name: format!("{}.moe.router", pfx),
                shape: vec![moe.num_experts, model.hidden_dim],
                dtype: SafeDtype::BF16,
                data_f32: moe.gate_weights.clone(),
            });
            // Experts
            for (e, expert_gate) in moe.expert_gate.iter().enumerate() {
                tensors.push(TensorToWrite {
                    name: format!("{}.moe.expert.{}.gate", pfx, e),
                    shape: vec![moe.intermediate_dim, model.hidden_dim],
                    dtype: SafeDtype::BF16,
                    data_f32: expert_gate.clone(),
                });
            }
            for (e, expert_up) in moe.expert_up.iter().enumerate() {
                tensors.push(TensorToWrite {
                    name: format!("{}.moe.expert.{}.up", pfx, e),
                    shape: vec![moe.intermediate_dim, model.hidden_dim],
                    dtype: SafeDtype::BF16,
                    data_f32: expert_up.clone(),
                });
            }
            for (e, expert_down) in moe.expert_down.iter().enumerate() {
                tensors.push(TensorToWrite {
                    name: format!("{}.moe.expert.{}.down", pfx, e),
                    shape: vec![model.hidden_dim, moe.intermediate_dim],
                    dtype: SafeDtype::BF16,
                    data_f32: expert_down.clone(),
                });
            }
        } else {
            // Dense FFN
            if let Some(ref w) = layer.ffn_gate {
                tensors.push(TensorToWrite {
                    name: format!("{}.ffn_gate", pfx),
                    shape: vec![w.out_features(), w.in_features()],
                    dtype: SafeDtype::BF16,
                    data_f32: w.weight().to_vec(),
                });
            }
            if let Some(ref w) = layer.ffn_up {
                tensors.push(TensorToWrite {
                    name: format!("{}.ffn_up", pfx),
                    shape: vec![w.out_features(), w.in_features()],
                    dtype: SafeDtype::BF16,
                    data_f32: w.weight().to_vec(),
                });
            }
            if let Some(ref w) = layer.ffn_down {
                tensors.push(TensorToWrite {
                    name: format!("{}.ffn_down", pfx),
                    shape: vec![w.out_features(), w.in_features()],
                    dtype: SafeDtype::BF16,
                    data_f32: w.weight().to_vec(),
                });
            }
        }

        // PLE
        if let Some(ref gate) = layer.ple_input_gate {
            tensors.push(TensorToWrite {
                name: format!("{}.ple_gate", pfx),
                shape: vec![gate.out_features(), gate.in_features()],
                dtype: SafeDtype::BF16,
                data_f32: gate.weight().to_vec(),
            });
        }
        if let Some(ref proj) = layer.ple_projection {
            tensors.push(TensorToWrite {
                name: format!("{}.ple_proj", pfx),
                shape: vec![proj.out_features(), proj.in_features()],
                dtype: SafeDtype::BF16,
                data_f32: proj.weight().to_vec(),
            });
        }
        if let Some(ref norm) = layer.ple_post_norm {
            tensors.push(TensorToWrite {
                name: format!("{}.ple_norm", pfx),
                shape: vec![model.hidden_dim],
                dtype: SafeDtype::BF16,
                data_f32: norm.weight().to_vec(),
            });
        }
    }

    // PLE model-level weights
    if let Some(ref proj) = model.ple_model_projection {
        tensors.push(TensorToWrite {
            name: "ple_model_projection".into(),
            shape: vec![model.hidden_dim, model.hidden_size_per_layer_input * model.num_layers],
            dtype: SafeDtype::BF16,
            data_f32: proj.clone(),
        });
    }
    if let Some(ref norm) = model.ple_projection_norm {
        tensors.push(TensorToWrite {
            name: "ple_projection_norm".into(),
            shape: vec![model.hidden_size_per_layer_input],
            dtype: SafeDtype::BF16,
            data_f32: norm.clone(),
        });
    }

    // Write weights
    write_safetensors(Path::new(&st_path), &tensors)?;

    // Write config
    let config = serialize_config(model);
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| FerrisResError::Shape(format!("Failed to serialize config: {}", e)))?;
    std::fs::write(&json_path, config_json)
        .map_err(|e| FerrisResError::Shape(format!("Failed to write config: {}", e)))?;

    tracing::info!(
        event = "checkpoint_saved",
        path = %st_path,
        tensors = tensors.len(),
        "Block-MoE-Res model saved"
    );

    Ok(())
}

/// Load a complete Block-MoE-Res model from disk.
///
/// Reads:
/// - `{path}.safetensors` — model weights
/// - `{path}.json` — model configuration
pub fn load_model(path: &str) -> Result<CpuBlockAttnResModel> {
    use crate::model::safetensors::MmapedSafetensors;
    use crate::model::cpu_linear::{CpuLinear, CpuRmsNorm};
    use crate::model::cpu_moe::CpuMoELayer;
    use crate::model::cpu_block_attn_res::{CpuBlockAttnResLayer, BlockConfig};

    let st_path = format!("{}.safetensors", path);
    let json_path = format!("{}.json", path);

    // Read config
    let config_json = std::fs::read_to_string(&json_path)
        .map_err(|e| FerrisResError::Shape(format!("Failed to read config {}: {}", json_path, e)))?;
    let config: BlockMoeResConfig = serde_json::from_str(&config_json)
        .map_err(|e| FerrisResError::Shape(format!("Failed to parse config: {}", e)))?;

    // Mmap safetensors
    let mmap = MmapedSafetensors::open(Path::new(&st_path))
        .map_err(|e| FerrisResError::Shape(format!("Failed to open {}: {:?}", st_path, e)))?;

    // Helper: load tensor as Vec<f32>
    let load = |name: &str| -> Result<Vec<f32>> {
        mmap.get_tensor_f32(name)
            .map_err(|e| FerrisResError::Shape(format!("Tensor '{}' not found: {:?}", name, e)))
    };

    // Helper: try loading a tensor (returns None if not found)
    let try_load = |name: &str| -> Option<Vec<f32>> {
        mmap.get_tensor_f32(name).ok()
    };

    // Embedding
    let embed_tokens = load("embed_tokens")?;
    let lm_head = load("lm_head")?;
    let final_norm = load("final_norm")?;

    // Block config
    let attn_res_proj = load("block_config.attn_res_proj")?;
    let attn_res_norm = load("block_config.attn_res_norm")?;

    // PLE model-level
    let ple_model_projection = try_load("ple_model_projection");
    let ple_projection_norm = try_load("ple_projection_norm");

    // Per-layer
    let mut layers = Vec::with_capacity(config.num_layers);
    for (i, lc) in config.layer_configs.iter().enumerate() {
        let pfx = format!("layers.{}", i);

        let q_proj = CpuLinear::from_weight(load(&format!("{}.q_proj", pfx))?, config.hidden_dim, config.num_heads * config.head_dim);
        let k_proj = CpuLinear::from_weight(load(&format!("{}.k_proj", pfx))?, config.hidden_dim, config.num_kv_heads * config.head_dim);
        let v_proj = CpuLinear::from_weight(load(&format!("{}.v_proj", pfx))?, config.hidden_dim, config.num_kv_heads * config.head_dim);
        let out_proj = CpuLinear::from_weight(load(&format!("{}.out_proj", pfx))?, config.num_heads * config.head_dim, config.hidden_dim);

        let attn_norm = CpuRmsNorm::from_weight(load(&format!("{}.attn_norm", pfx))?, 1e-6);
        let post_attn_norm = CpuRmsNorm::from_weight(load(&format!("{}.post_attn_norm", pfx))?, 1e-6);
        let pre_ffn_norm = CpuRmsNorm::from_weight(load(&format!("{}.pre_ffn_norm", pfx))?, 1e-6);
        let post_ffn_norm = CpuRmsNorm::from_weight(load(&format!("{}.post_ffn_norm", pfx))?, 1e-6);

        let q_norm = CpuRmsNorm::from_weight(load(&format!("{}.q_norm", pfx))?, 1e-6);
        let k_norm = CpuRmsNorm::from_weight(load(&format!("{}.k_norm", pfx))?, 1e-6);
        // v_norm — may not be stored, create default
        let v_norm_data = mmap.get_tensor_f32(&format!("{}.v_norm", pfx))
            .unwrap_or_else(|_| vec![1.0f32; config.num_kv_heads * config.head_dim]);
        let v_norm = CpuRmsNorm::from_weight(v_norm_data, 1e-6);

        // MoE or dense FFN
        let moe = if lc.has_moe {
            let gate_weights = load(&format!("{}.moe.router", pfx))?;
            let mut expert_gate = Vec::with_capacity(lc.num_experts);
            let mut expert_up = Vec::with_capacity(lc.num_experts);
            let mut expert_down = Vec::with_capacity(lc.num_experts);
            for e in 0..lc.num_experts {
                expert_gate.push(load(&format!("{}.moe.expert.{}.gate", pfx, e))?);
                expert_up.push(load(&format!("{}.moe.expert.{}.up", pfx, e))?);
                expert_down.push(load(&format!("{}.moe.expert.{}.down", pfx, e))?);
            }
            Some(CpuMoELayer {
                gate_weights,
                expert_gate,
                expert_up,
                expert_down,
                num_experts: lc.num_experts,
                top_k: lc.top_k,
                intermediate_dim: lc.inter_dim,
                hidden_dim: config.hidden_dim,
                use_gelu: lc.use_gelu,
            })
        } else {
            None
        };

        let ffn_gate = if !lc.has_moe { try_load(&format!("{}.ffn_gate", pfx)).map(|w| CpuLinear::from_weight(w, config.hidden_dim, lc.inter_dim)) } else { None };
        let ffn_up = if !lc.has_moe { try_load(&format!("{}.ffn_up", pfx)).map(|w| CpuLinear::from_weight(w, config.hidden_dim, lc.inter_dim)) } else { None };
        let ffn_down = if !lc.has_moe { try_load(&format!("{}.ffn_down", pfx)).map(|w| CpuLinear::from_weight(w, lc.inter_dim, config.hidden_dim)) } else { None };

        // PLE
        let ple_input_gate = try_load(&format!("{}.ple_gate", pfx)).map(|w| CpuLinear::from_weight(w, config.hidden_dim, config.hidden_size_per_layer_input));
        let ple_projection = try_load(&format!("{}.ple_proj", pfx)).map(|w| CpuLinear::from_weight(w, config.hidden_size_per_layer_input, config.hidden_dim));
        let ple_post_norm = try_load(&format!("{}.ple_norm", pfx)).map(|w| CpuRmsNorm::from_weight(w, 1e-6));

        layers.push(CpuBlockAttnResLayer {
            layer_number: i,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_dim: lc.inter_dim,
            attn_norm, q_proj, k_proj, v_proj, out_proj,
            q_norm, k_norm, v_norm,
            post_attn_norm,
            pre_ffn_norm, ffn_gate, ffn_up, ffn_down, moe, post_ffn_norm,
            layer_scalar: lc.layer_scalar,
            ple_input_gate, ple_projection, ple_post_norm,
            rope_theta: lc.rope_theta,
            partial_rotary_factor: lc.partial_rotary_factor,
            use_gelu: lc.use_gelu,
            kv_shared: lc.kv_shared,
        });
    }

    let model = CpuBlockAttnResModel {
        embed_tokens,
        lm_head,
        final_norm,
        hidden_dim: config.hidden_dim,
        vocab_size: config.vocab_size,
        num_layers: config.num_layers,
        final_logit_softcapping: config.final_logit_softcapping,
        hidden_size_per_layer_input: config.hidden_size_per_layer_input,
        num_kv_shared_layers: config.num_kv_shared_layers,
        block_config: BlockConfig {
            num_blocks: config.num_blocks,
            layers_per_block: config.layers_per_block,
            boundary_layers: config.boundary_layers,
            attn_res_proj,
            attn_res_norm,
        },
        layers,
        ple_model_projection,
        ple_projection_norm,
        embed_tokens_per_layer: None,
        lora_manager: None,
    };

    tracing::info!(
        event = "checkpoint_loaded",
        path = %st_path,
        layers = model.num_layers,
        "Block-MoE-Res model loaded"
    );

    Ok(model)
}

/// Extract serializable config from model.
fn serialize_config(model: &CpuBlockAttnResModel) -> BlockMoeResConfig {
    let layer_configs: Vec<LayerConfig> = model.layers.iter().map(|layer| {
        LayerConfig {
            use_gelu: layer.use_gelu,
            kv_shared: layer.kv_shared,
            is_full_attention: layer.head_dim != model.hidden_dim / layer.num_heads,
            rope_theta: layer.rope_theta,
            partial_rotary_factor: layer.partial_rotary_factor,
            layer_scalar: layer.layer_scalar,
            has_ple: layer.ple_input_gate.is_some(),
            has_moe: layer.moe.is_some(),
            num_experts: layer.moe.as_ref().map(|m| m.num_experts).unwrap_or(0),
            top_k: layer.moe.as_ref().map(|m| m.top_k).unwrap_or(0),
            inter_dim: layer.moe.as_ref().map(|m| m.intermediate_dim)
                .or(layer.ffn_gate.as_ref().map(|g| g.out_features()))
                .unwrap_or(0),
        }
    }).collect();

    BlockMoeResConfig {
        hidden_dim: model.hidden_dim,
        vocab_size: model.vocab_size,
        num_layers: model.num_layers,
        num_heads: model.layers.first().map(|l| l.num_heads).unwrap_or(8),
        num_kv_heads: model.layers.first().map(|l| l.num_kv_heads).unwrap_or(1),
        head_dim: model.layers.first().map(|l| l.head_dim).unwrap_or(0),
        final_logit_softcapping: model.final_logit_softcapping,
        hidden_size_per_layer_input: model.hidden_size_per_layer_input,
        num_kv_shared_layers: model.num_kv_shared_layers,
        num_blocks: model.block_config.num_blocks,
        layers_per_block: model.block_config.layers_per_block,
        boundary_layers: model.block_config.boundary_layers.clone(),
        layer_configs,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_config_serde_roundtrip() {
        use super::*;
        let config = BlockMoeResConfig {
            hidden_dim: 1536,
            vocab_size: 262144,
            num_layers: 35,
            num_heads: 8,
            num_kv_heads: 1,
            head_dim: 256,
            final_logit_softcapping: Some(30.0),
            hidden_size_per_layer_input: 256,
            num_kv_shared_layers: 20,
            num_blocks: 7,
            layers_per_block: 5,
            boundary_layers: vec![4, 9, 14, 19, 24, 29, 34],
            layer_configs: vec![],
        };
        let json = serde_json::to_string(&config).unwrap();
        let recovered: BlockMoeResConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.hidden_dim, 1536);
        assert_eq!(recovered.boundary_layers.len(), 7);
    }
}
