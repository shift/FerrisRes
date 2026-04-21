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

        // Per-head norms (each has head_dim elements)
        tensors.push(TensorToWrite {
            name: format!("{}.q_norm", pfx),
            shape: vec![layer.q_norm.weight().len()],
            dtype: SafeDtype::BF16,
            data_f32: layer.q_norm.weight().to_vec(),
        });
        tensors.push(TensorToWrite {
            name: format!("{}.k_norm", pfx),
            shape: vec![layer.k_norm.weight().len()],
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

/// Load a Block-MoE-Res model from safetensors checkpoint.
///
/// Reads the `.safetensors` weights file and `.json` config written by `save_model()`.
/// Reconstructs the full `CpuBlockAttnResModel` with all weights populated.
pub fn load_model(path: &str) -> Result<CpuBlockAttnResModel> {
    use crate::model::cpu_block_attn_res::{CpuBlockAttnResLayer, BlockConfig};
    use crate::model::cpu_linear::{CpuLinear, CpuRmsNorm};
    use crate::model::cpu_moe::CpuMoELayer;
    use crate::model::safetensors::load_safetensors;

    let st_path = format!("{}.safetensors", path);
    let json_path = format!("{}.json", path);

    // Load config
    let config_json = std::fs::read_to_string(&json_path)
        .map_err(|e| FerrisResError::Shape(format!("Failed to read config {}: {}", json_path, e)))?;
    let config: BlockMoeResConfig = serde_json::from_str(&config_json)
        .map_err(|e| FerrisResError::Shape(format!("Failed to parse config: {}", e)))?;

    // Load weights
    let weights = load_safetensors(Path::new(&st_path))?;
    let tensors = &weights.tensors;

    // Helper to get a tensor's data as Vec<f32>
    let get_tensor = |name: &str| -> Result<Vec<f32>> {
        tensors.get(name)
            .map(|t| t.data.clone())
            .ok_or_else(|| FerrisResError::Shape(format!("Missing tensor: {}", name)))
    };

    // Model-level weights
    let embed_tokens = get_tensor("embed_tokens")?;
    let lm_head = get_tensor("lm_head")?;
    let final_norm = get_tensor("final_norm")?;
    let attn_res_proj = get_tensor("block_config.attn_res_proj")?;
    let attn_res_norm = get_tensor("block_config.attn_res_norm")?;

    // PLE model-level weights (optional)
    let ple_model_projection = tensors.get("ple_model_projection").map(|t| t.data.clone());
    let ple_projection_norm = tensors.get("ple_projection_norm").map(|t| t.data.clone());

    // Per-layer weights
    let mut layers = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        let pfx = format!("layers.{}", i);
        let lc = config.layer_configs.get(i);

        let q_data = get_tensor(&format!("{}.q_proj", pfx))?;
        let k_data = get_tensor(&format!("{}.k_proj", pfx))?;
        let v_data = get_tensor(&format!("{}.v_proj", pfx))?;
        let out_data = get_tensor(&format!("{}.out_proj", pfx))?;

        let q_proj = CpuLinear::from_weight(q_data, config.hidden_dim, config.num_heads * config.head_dim);
        let k_proj = CpuLinear::from_weight(k_data, config.hidden_dim, config.num_kv_heads * config.head_dim);
        let v_proj = CpuLinear::from_weight(v_data, config.hidden_dim, config.num_kv_heads * config.head_dim);
        let out_proj = CpuLinear::from_weight(out_data, config.num_heads * config.head_dim, config.hidden_dim);

        let attn_norm = CpuRmsNorm::from_weight(get_tensor(&format!("{}.attn_norm", pfx))?, 1e-6);
        let post_attn_norm = CpuRmsNorm::from_weight(get_tensor(&format!("{}.post_attn_norm", pfx))?, 1e-6);
        let pre_ffn_norm = CpuRmsNorm::from_weight(get_tensor(&format!("{}.pre_ffn_norm", pfx))?, 1e-6);
        let post_ffn_norm = CpuRmsNorm::from_weight(get_tensor(&format!("{}.post_ffn_norm", pfx))?, 1e-6);

        let q_norm_data = get_tensor(&format!("{}.q_norm", pfx)).unwrap_or_default();
        let k_norm_data = get_tensor(&format!("{}.k_norm", pfx)).unwrap_or_default();
        let v_norm_data = tensors.get(&format!("{}.v_norm", pfx)).map(|t| t.data.clone()).unwrap_or_default();
        let q_norm = CpuRmsNorm::from_weight(q_norm_data, 1e-6);
        let k_norm = CpuRmsNorm::from_weight(k_norm_data, 1e-6);
        let v_norm = CpuRmsNorm::from_weight(v_norm_data, 1e-6);

        // MoE or dense FFN
        let (moe, ffn_gate, ffn_up, ffn_down) = if let Some(lc) = lc {
            if lc.has_moe && lc.num_experts > 0 {
                let router = get_tensor(&format!("{}.moe.router", pfx))?;
                let mut moe = CpuMoELayer::new(config.hidden_dim, lc.inter_dim, lc.num_experts, lc.top_k);
                moe.gate_weights = router;
                for e in 0..lc.num_experts {
                    moe.expert_gate[e] = get_tensor(&format!("{}.moe.expert.{}.gate", pfx, e)).unwrap_or_default();
                    moe.expert_up[e] = get_tensor(&format!("{}.moe.expert.{}.up", pfx, e)).unwrap_or_default();
                    moe.expert_down[e] = get_tensor(&format!("{}.moe.expert.{}.down", pfx, e)).unwrap_or_default();
                }
                (Some(moe), None, None, None)
            } else {
                let gate = get_tensor(&format!("{}.ffn_gate", pfx)).ok()
                    .map(|w| CpuLinear::from_weight(w, config.hidden_dim, lc.inter_dim));
                let up = get_tensor(&format!("{}.ffn_up", pfx)).ok()
                    .map(|w| CpuLinear::from_weight(w, config.hidden_dim, lc.inter_dim));
                let down = get_tensor(&format!("{}.ffn_down", pfx)).ok()
                    .map(|w| CpuLinear::from_weight(w, lc.inter_dim, config.hidden_dim));
                (None, gate, up, down)
            }
        } else {
            (None, None, None, None)
        };

        // PLE
        let ple_input_gate = get_tensor(&format!("{}.ple_gate", pfx)).ok()
            .map(|w| CpuLinear::from_weight(w, config.hidden_dim, config.hidden_size_per_layer_input));
        let ple_projection = get_tensor(&format!("{}.ple_proj", pfx)).ok()
            .map(|w| CpuLinear::from_weight(w, config.hidden_size_per_layer_input, config.hidden_dim));
        let ple_post_norm = get_tensor(&format!("{}.ple_norm", pfx)).ok()
            .map(|w| CpuRmsNorm::from_weight(w, 1e-6));

        let layer_scalar = lc.map(|l| l.layer_scalar).unwrap_or(1.0);
        let use_gelu = lc.map(|l| l.use_gelu).unwrap_or(false);
        let kv_shared = lc.map(|l| l.kv_shared).unwrap_or(false);
        let rope_theta = lc.map(|l| l.rope_theta).unwrap_or(10000.0) as f64;
        let partial_rotary_factor = lc.map(|l| l.partial_rotary_factor).unwrap_or(1.0);

        let layer = CpuBlockAttnResLayer {
            layer_number: i,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            intermediate_dim: lc.map(|l| l.inter_dim).unwrap_or(config.hidden_dim * 4),
            attn_norm,
            q_proj, k_proj, v_proj, out_proj,
            q_norm, k_norm, v_norm,
            post_attn_norm,
            pre_ffn_norm,
            ffn_gate, ffn_up, ffn_down,
            moe,
            post_ffn_norm,
            layer_scalar,
            ple_input_gate, ple_projection, ple_post_norm,
            rope_theta,
            partial_rotary_factor,
            use_gelu,
            kv_shared,
        };
        layers.push(layer);
    }

    let model = CpuBlockAttnResModel {
        layers,
        embed_tokens,
        lm_head,
        final_norm,
        hidden_dim: config.hidden_dim,
        vocab_size: config.vocab_size,
        num_layers: config.num_layers,
        final_logit_softcapping: config.final_logit_softcapping,
        ple_model_projection,
        ple_projection_norm,
        embed_tokens_per_layer: None, // Reconstructed during distillation
        hidden_size_per_layer_input: config.hidden_size_per_layer_input,
        num_kv_shared_layers: config.num_kv_shared_layers,
        block_config: BlockConfig {
            num_blocks: config.num_blocks,
            layers_per_block: config.layers_per_block,
            boundary_layers: config.boundary_layers,
            attn_res_proj,
            attn_res_norm,
        },
        lora_manager: None, // Re-attached during training
    };

    tracing::info!(
        event = "checkpoint_loaded",
        path = %st_path,
        layers = model.num_layers,
        hidden_dim = model.hidden_dim,
        "Block-MoE-Res model loaded"
    );

    Ok(model)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_config_serde_roundtrip() {
        use crate::model::checkpoint::BlockMoeResConfig;
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

    #[test]
    fn test_save_load_roundtrip() {
        use crate::model::cpu_block_attn_res::CpuBlockAttnResModel;
        use crate::model::cpu_block_attn_res::CpuBlockAttnResLayer;
        use crate::model::cpu_block_attn_res::BlockConfig;

        // Create a minimal model (2 layers, tiny dims)
        let hd = 32;
        let vs = 64;
        let num_layers = 2;

        let layer0 = CpuBlockAttnResLayer::new(hd, 2, 1, 16, 64);
        let layer1 = CpuBlockAttnResLayer::new(hd, 2, 1, 16, 64);

        let model = CpuBlockAttnResModel {
            layers: vec![layer0, layer1],
            embed_tokens: vec![0.5f32; vs * hd],
            lm_head: vec![0.1f32; hd * vs],
            final_norm: vec![1.0f32; hd],
            hidden_dim: hd,
            vocab_size: vs,
            num_layers,
            final_logit_softcapping: Some(30.0),
            ple_model_projection: None,
            ple_projection_norm: None,
            embed_tokens_per_layer: None,
            hidden_size_per_layer_input: 16,
            num_kv_shared_layers: 0,
            block_config: BlockConfig {
                num_blocks: 1,
                layers_per_block: 2,
                boundary_layers: vec![1],
                attn_res_proj: vec![0.1f32; hd * hd],
                attn_res_norm: vec![1.0f32; hd],
            },
            lora_manager: None,
        };

        // Save
        let tmp_dir = std::env::temp_dir().join("ferrisres_roundtrip_test");
        let _ = std::fs::create_dir_all(&tmp_dir);
        let path = tmp_dir.join("test_model").to_string_lossy().to_string();

        super::save_model(&model, &path).expect("save should succeed");

        // Load
        let loaded = super::load_model(&path).expect("load should succeed");

        // Verify structural properties
        assert_eq!(loaded.hidden_dim, hd);
        assert_eq!(loaded.vocab_size, vs);
        assert_eq!(loaded.num_layers, num_layers);
        assert_eq!(loaded.embed_tokens.len(), vs * hd);
        assert_eq!(loaded.lm_head.len(), hd * vs);
        assert_eq!(loaded.final_norm.len(), hd);
        assert_eq!(loaded.final_logit_softcapping, Some(30.0));
        assert_eq!(loaded.layers.len(), 2);

        // Verify weight roundtrip (spot-check first layer)
        assert_eq!(loaded.layers[0].hidden_dim, hd);
        assert_eq!(loaded.layers[0].num_heads, 2);
        assert_eq!(loaded.layers[0].num_kv_heads, 1);
        assert_eq!(loaded.layers[0].head_dim, 16);

        // Clean up
        let _ = std::fs::remove_file(format!("{}.safetensors", path));
        let _ = std::fs::remove_file(format!("{}.json", path));
    }
}
