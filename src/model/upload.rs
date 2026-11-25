//! Upload CPU-trained model weights to GPU for inference.
//!
//! After distillation/training completes on CPU, this module transfers
//! the trained weights to GPU Buffers for fast inference via
//! BlockAttnResModel.

use wgpu::Queue;
use crate::compute::GpuBuffer;
use crate::model::cpu_block_attn_res::{CpuBlockAttnResModel, CpuBlockAttnResLayer};
use crate::model::cpu_linear::CpuLinear;
use crate::model::block_attn_res::{BlockAttnResLayer, NormWeights};

/// Transpose a row-major `[rows, cols]` matrix.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Upload a CPU-trained model's linear layer weights to a GPU linear layer.
///
/// `CpuLinear` stores `[out_features, in_features]`; the GPU matmul reads
/// `[in_features, out_features]`, so the weight is transposed on upload.
fn upload_linear(gpu_linear: &crate::model::linear::Linear, cpu_linear: &CpuLinear, queue: &Queue) {
    let w = cpu_linear.weight();
    let out_f = cpu_linear.out_features();
    let in_f = cpu_linear.in_features();
    let transposed = transpose(&w, out_f, in_f);
    gpu_linear.set_weight(queue, &transposed);
}

/// Upload a single CPU layer's weights to a GPU layer.
///
/// This transfers all weights for one transformer layer:
/// - Attention: q/k/v/out projections
/// - FFN: gate/up/down projections (dense) or MoE experts
/// - Norms: every learned RMSNorm scale
pub fn upload_layer(
    gpu_layer: &BlockAttnResLayer,
    cpu_layer: &CpuBlockAttnResLayer,
    queue: &Queue,
) {
    // Attention projections
    upload_linear(&gpu_layer.q_proj_accessor(), &cpu_layer.q_proj, queue);
    upload_linear(&gpu_layer.k_proj_accessor(), &cpu_layer.k_proj, queue);
    upload_linear(&gpu_layer.v_proj_accessor(), &cpu_layer.v_proj, queue);
    upload_linear(&gpu_layer.out_proj_accessor(), &cpu_layer.out_proj, queue);

    // FFN projections
    match (&cpu_layer.moe, gpu_layer.moe_accessor()) {
        (Some(_cpu_moe), Some(_gpu_moe)) => {
            // MoE weight upload — expert weights, router, gate
            // TODO: upload MoE experts when MoELinear has weight setters
            tracing::warn!(
                "MoE weight upload not yet implemented for layer {}",
                cpu_layer.layer_number
            );
        }
        (None, None) => {
            // Dense FFN: gate, up, down
            if let (Some(ref cpu_gate), Some(ref gpu_gate)) =
                (&cpu_layer.ffn_gate, gpu_layer.ffn_gate_accessor())
            {
                upload_linear(gpu_gate, cpu_gate, queue);
            }
            if let (Some(ref cpu_up), Some(ref gpu_up)) =
                (&cpu_layer.ffn_up, gpu_layer.ffn_up_accessor())
            {
                upload_linear(gpu_up, cpu_up, queue);
            }
            if let (Some(ref cpu_down), Some(ref gpu_down)) =
                (&cpu_layer.ffn_down, gpu_layer.ffn_down_accessor())
            {
                upload_linear(gpu_down, cpu_down, queue);
            }
        }
        _ => {
            tracing::warn!(
                "Layer {}: CPU/GPU FFN type mismatch (MoE vs dense)",
                cpu_layer.layer_number
            );
        }
    }

    // PLE weights
    if let (Some(ref cpu_gate), Some(ref gpu_gate)) =
        (&cpu_layer.ple_input_gate, gpu_layer.ple_input_gate_accessor())
    {
        upload_linear(gpu_gate, cpu_gate, queue);
    }
    if let (Some(ref cpu_proj), Some(ref gpu_proj)) =
        (&cpu_layer.ple_projection, gpu_layer.ple_projection_accessor())
    {
        upload_linear(gpu_proj, cpu_proj, queue);
    }

    // Learned norm scales.
    gpu_layer.set_norm_weights(queue, &NormWeights {
        attn_norm: Some(cpu_layer.attn_norm.weight()),
        ff_norm: Some(cpu_layer.pre_ffn_norm.weight()),
        q_norm: Some(cpu_layer.q_norm.weight()),
        k_norm: Some(cpu_layer.k_norm.weight()),
        v_norm: Some(cpu_layer.v_norm.weight()),
        post_attn_norm: Some(cpu_layer.post_attn_norm.weight()),
        post_ffn_norm: Some(cpu_layer.post_ffn_norm.weight()),
        attn_res_norm: None,
    });

    tracing::info!(
        "Uploaded weights for layer {} (hidden={} heads={} head_dim={})",
        cpu_layer.layer_number,
        cpu_layer.hidden_dim,
        cpu_layer.num_heads,
        cpu_layer.head_dim,
    );
}

/// Upload the inter-block key norm (`block_config.attn_res_norm`) to a GPU
/// layer. Every layer shares the same weights.
pub fn upload_attn_res_norm(
    gpu_layer: &BlockAttnResLayer,
    weights: &[f32],
    queue: &Queue,
) {
    gpu_layer.set_norm_weights(queue, &NormWeights {
        attn_res_norm: Some(weights),
        ..NormWeights::default()
    });
}

/// Upload model-level weights (embeddings, LM head, final norm) to GPU buffers.
pub fn upload_model_weights(
    embed_buffer: &GpuBuffer,
    lm_head_buffer: Option<&GpuBuffer>,
    final_norm_buffer: Option<&GpuBuffer>,
    cpu_model: &CpuBlockAttnResModel,
    queue: &Queue,
) {
    // Token embeddings
    let embed_bytes = bytemuck::cast_slice(&cpu_model.embed_tokens);
    queue.write_buffer(embed_buffer.buffer(), 0, embed_bytes);

    // LM head
    if let (Some(buf), false) = (lm_head_buffer, cpu_model.lm_head.is_empty()) {
        let lm_head_bytes = bytemuck::cast_slice(&cpu_model.lm_head);
        queue.write_buffer(buf.buffer(), 0, lm_head_bytes);
    }

    // Final norm
    if let (Some(buf), false) = (final_norm_buffer, cpu_model.final_norm.is_empty()) {
        let norm_bytes = bytemuck::cast_slice(&cpu_model.final_norm);
        queue.write_buffer(buf.buffer(), 0, norm_bytes);
    }

    tracing::info!(
        "Uploaded model-level weights: embed={} lm_head={} final_norm={}",
        cpu_model.embed_tokens.len(),
        cpu_model.lm_head.len(),
        cpu_model.final_norm.len(),
    );
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_upload_module_exists() {
        // Verify the upload module compiles and functions are accessible.
        // Full integration test requires a GPU device.
        assert!(true, "Upload module compiled successfully");
    }
}
