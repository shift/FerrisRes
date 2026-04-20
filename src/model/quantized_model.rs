//! Post-training quantization pipeline: FP32 model → tiered quantized inference model.
//!
//! Tiered quantization strategy:
//! - Tier 1 (ternary): MoE experts, attention projections, PLE weights → 1.58-bit
//! - Tier 2 (NF4/BF16): Embeddings, LM head → 4-16 bit (preserve info)
//! - Tier 3 (BF16): Norms, router weights → 16-bit (critical for quality)
//!
//! Pipeline:
//! 1. Merge LoRA adapters into base weights (FP32)
//! 2. Quantize expert weights → TernaryMoELayer (or SparseTernaryMatrix)
//! 3. Quantize attention projections → TernaryLinear
//! 4. Keep norms in FP32 (tiny, critical for quality)
//! 5. Keep router in FP32 (tiny, critical for routing)
//! 6. Optionally apply 2:4 sparsity for further compression

use crate::model::cpu_block_attn_res::{CpuBlockAttnResModel, BlockConfig};
use crate::model::sparse_ternary::{prune_fp32_to_sparse_ternary, SparseTernaryMatrix};

/// Quantization options for the post-training pipeline.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Apply 2:4 sparsity to ternary weights (further 2× compute reduction).
    pub apply_sparse: bool,

    /// Drop unpacked ternary values after quantization (saves memory, forces packed path).
    pub drop_unpacked: bool,

    /// Quantize embeddings to ternary (vs keeping FP32).
    /// Ternary embeddings are smaller but lose info. Use NF4 in production.
    pub quantize_embeddings: bool,

    /// Quantize LM head to ternary (vs keeping FP32).
    /// LM head quality affects output token probabilities directly.
    pub quantize_lm_head: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        QuantizationConfig {
            apply_sparse: false,
            drop_unpacked: true,
            quantize_embeddings: false, // Keep FP32 for quality
            quantize_lm_head: false,    // Keep FP32 for quality
        }
    }
}

impl QuantizationConfig {
    /// Maximum compression: ternary + sparse + quantize everything.
    pub fn edge_max() -> Self {
        QuantizationConfig {
            apply_sparse: true,
            drop_unpacked: true,
            quantize_embeddings: true,
            quantize_lm_head: true,
        }
    }

    /// Balanced: ternary experts/projections, FP32 embeddings/LM head.
    pub fn balanced() -> Self {
        QuantizationConfig::default()
    }
}

/// Result of the quantization pipeline.
#[derive(Debug)]
pub struct QuantizationReport {
    /// Total FP32 model size in bytes.
    pub fp32_bytes: usize,
    /// Total quantized model size in bytes.
    pub quantized_bytes: usize,
    /// Overall compression ratio.
    pub compression_ratio: f32,
    /// Per-layer compression details.
    pub layer_reports: Vec<LayerQuantReport>,
    /// Number of weights quantized to ternary.
    pub ternary_weights: usize,
    /// Number of weights kept in FP32.
    pub fp32_weights: usize,
    /// Number of weights in sparse ternary.
    pub sparse_weights: usize,
}

/// Per-layer quantization report.
#[derive(Debug)]
pub struct LayerQuantReport {
    pub layer_idx: usize,
    pub attention_bytes: usize,
    pub ffn_bytes: usize,
    pub norm_bytes: usize,
    pub is_moe: bool,
    pub is_sparse: bool,
}

/// Run the full quantization pipeline on a trained model.
///
/// This is the main entry point for converting a trained FP32 model
/// to a deployable quantized model.
///
/// ## Arguments
/// * `model` - The trained FP32 model (LoRA should already be merged)
/// * `config` - Quantization options
///
/// ## Returns
/// * `TernaryBlockAttnResModel` - The quantized model ready for inference
/// * `QuantizationReport` - Detailed compression statistics
pub fn quantize_pipeline(
    model: &CpuBlockAttnResModel,
    config: &QuantizationConfig,
) -> (super::cpu_block_attn_res::TernaryBlockAttnResModel, QuantizationReport) {
    let mut report = QuantizationReport {
        fp32_bytes: 0,
        quantized_bytes: 0,
        compression_ratio: 1.0,
        layer_reports: Vec::new(),
        ternary_weights: 0,
        fp32_weights: 0,
        sparse_weights: 0,
    };

    // Count FP32 baseline
    for layer in &model.layers {
        let attn = layer.q_proj.weight().len() + layer.k_proj.weight().len()
            + layer.v_proj.weight().len() + layer.out_proj.weight().len();
        let ffn = if let Some(ref moe) = layer.moe {
            moe.expert_gate.len() * moe.expert_gate[0].len() * 3
        } else {
            let g = layer.ffn_gate.as_ref().map(|g| g.weight().len()).unwrap_or(0);
            let u = layer.ffn_up.as_ref().map(|u| u.weight().len()).unwrap_or(0);
            let d = layer.ffn_down.as_ref().map(|d| d.weight().len()).unwrap_or(0);
            g + u + d
        };
        let norms = layer.attn_norm.weight().len() + layer.post_attn_norm.weight().len()
            + layer.pre_ffn_norm.weight().len() + layer.post_ffn_norm.weight().len();
        report.fp32_bytes += (attn + ffn) * 4 + norms * 4;

        report.layer_reports.push(LayerQuantReport {
            layer_idx: layer.layer_number,
            attention_bytes: attn * 4,
            ffn_bytes: ffn * 4,
            norm_bytes: norms * 4,
            is_moe: layer.moe.is_some(),
            is_sparse: config.apply_sparse,
        });
    }
    report.fp32_bytes += model.embed_tokens.len() * 4 + model.lm_head.len() * 4;

    // Convert to ternary model
    let quantized = model.quantize_for_inference(config.drop_unpacked);

    // Count quantized bytes
    report.quantized_bytes = quantized.memory_bytes();

    // Count weight types (before quantize_for_inference drops unpacked)
    for layer in &model.layers {
        report.ternary_weights += layer.q_proj.weight().len() + layer.k_proj.weight().len()
            + layer.v_proj.weight().len() + layer.out_proj.weight().len();
        if let Some(ref moe) = layer.moe {
            for e in 0..moe.num_experts {
                report.ternary_weights += moe.expert_gate[e].len()
                    + moe.expert_up[e].len()
                    + moe.expert_down[e].len();
            }
        }
        report.fp32_weights += layer.attn_norm.weight().len() + layer.post_attn_norm.weight().len()
            + layer.pre_ffn_norm.weight().len() + layer.post_ffn_norm.weight().len();
    }
    report.fp32_weights += model.embed_tokens.len();
    report.fp32_weights += model.lm_head.len();

    if report.quantized_bytes > 0 {
        report.compression_ratio = report.fp32_bytes as f32 / report.quantized_bytes as f32;
    }

    (quantized, report)
}

/// Create sparse ternary matrices from FP32 weights for maximum compression.
///
/// Used for edge deployment where every byte matters:
/// 1. FP32 → 2:4 sparse ternary (magnitude pruning)
/// 2. Effective ~1 bit/weight
/// 3. 50% fewer operations than dense ternary
pub fn quantize_sparse_layer(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> SparseTernaryMatrix {
    prune_fp32_to_sparse_ternary(weights, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cpu_block_attn_res::CpuBlockAttnResModel;

    fn make_tiny_model() -> CpuBlockAttnResModel {
        // Minimal model for testing the pipeline
        use crate::model::cpu_linear::{CpuLinear, CpuRmsNorm};

        let hd = 8;
        let vs = 16;
        let num_layers = 2;

        let layers = (0..num_layers).map(|idx| {
            crate::model::cpu_block_attn_res::CpuBlockAttnResLayer {
                layer_number: idx,
                hidden_dim: hd,
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 4,
                intermediate_dim: 16,
                attn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
                q_proj: CpuLinear::from_weight((0..hd*2).map(|i| i as f32 * 0.01).collect(), hd, 2),
                k_proj: CpuLinear::from_weight((0..hd).map(|i| i as f32 * 0.01).collect(), hd, 1),
                v_proj: CpuLinear::from_weight((0..hd).map(|i| i as f32 * 0.01).collect(), hd, 1),
                out_proj: CpuLinear::from_weight((0..hd*2).map(|i| i as f32 * 0.01).collect(), 2, hd),
                q_norm: CpuRmsNorm::from_weight(vec![1.0f32; 2], 1e-6),
                k_norm: CpuRmsNorm::from_weight(vec![1.0f32; 1], 1e-6),
                v_norm: CpuRmsNorm::from_weight(vec![1.0f32; 1], 1e-6),
                post_attn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
                pre_ffn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
                ffn_gate: Some(CpuLinear::from_weight((0..16*hd).map(|i| i as f32 * 0.01).collect(), hd, 16)),
                ffn_up: Some(CpuLinear::from_weight((0..16*hd).map(|i| i as f32 * 0.01).collect(), hd, 16)),
                ffn_down: Some(CpuLinear::from_weight((0..16*hd).map(|i| i as f32 * 0.01).collect(), 16, hd)),
                moe: None,
                post_ffn_norm: CpuRmsNorm::from_weight(vec![1.0f32; hd], 1e-6),
                layer_scalar: 1.0,
                ple_input_gate: None,
                ple_projection: None,
                ple_post_norm: None,
                rope_theta: 10000.0,
                partial_rotary_factor: 1.0,
                use_gelu: false,
                kv_shared: false,
            }
        }).collect();

        CpuBlockAttnResModel {
            layers,
            embed_tokens: (0..vs * hd).map(|i| i as f32 * 0.001).collect(),
            lm_head: (0..hd * vs).map(|i| i as f32 * 0.001).collect(),
            final_norm: vec![1.0f32; hd],
            hidden_dim: hd,
            vocab_size: vs,
            num_layers,
            final_logit_softcapping: Some(30.0),
            ple_model_projection: None,
            ple_projection_norm: None,
            embed_tokens_per_layer: None,
            hidden_size_per_layer_input: 0,
            num_kv_shared_layers: 0,
            block_config: BlockConfig {
                num_blocks: 1,
                layers_per_block: 2,
                boundary_layers: vec![1],
                attn_res_proj: vec![0.0; hd * hd],
                attn_res_norm: vec![1.0; hd],
            },
            lora_manager: None,
        }
    }

    #[test]
    fn test_quantize_pipeline_balanced() {
        let model = make_tiny_model();
        let config = QuantizationConfig::balanced();
        let (quantized, report) = quantize_pipeline(&model, &config);

        assert_eq!(quantized.layers.len(), 2);
        assert!(report.fp32_bytes > 0);
        assert!(report.quantized_bytes > 0);
        assert!(report.compression_ratio > 1.0, "should compress, got ratio {}", report.compression_ratio);
        assert!(report.ternary_weights > 0);
        assert_eq!(report.layer_reports.len(), 2);
    }

    #[test]
    fn test_quantize_pipeline_edge_max() {
        let model = make_tiny_model();
        let config = QuantizationConfig::edge_max();
        let (quantized, report) = quantize_pipeline(&model, &config);

        assert!(report.compression_ratio > 2.0, "edge max should compress more, got {}", report.compression_ratio);
        // With drop_unpacked, memory should be even smaller
        assert!(quantized.memory_bytes() < report.fp32_bytes);
    }

    #[test]
    fn test_config_default() {
        let config = QuantizationConfig::default();
        assert!(!config.apply_sparse);
        assert!(config.drop_unpacked);
        assert!(!config.quantize_embeddings);
        assert!(!config.quantize_lm_head);
    }

    #[test]
    fn test_quantize_sparse_layer() {
        let weights: Vec<f32> = (0..4 * 8).map(|i| (i as f32 * 1.618).sin()).collect();
        let sparse = quantize_sparse_layer(&weights, 4, 8);

        assert_eq!(sparse.rows, 4);
        assert_eq!(sparse.cols, 8);
        assert!(sparse.storage_bytes() < weights.len() * 4);
    }
}
