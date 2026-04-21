//! Proper layer-by-layer backward pass for Block-MoE-Res distillation.
//!
//! Stores activations during forward, computes per-layer gradients in reverse.
//! Memory: ~seq × hidden_dim × 2 per layer for norms + expert intermediates.

use crate::model::cpu_block_attn_res::{
    CpuBlockAttnResModel, LayerActivations, ExpertActivation,
};
use crate::model::gemma_mapper;
use crate::model::gemma_mapper::{apply_rope, apply_rope_gqa};

/// Output of the training forward pass — logits + everything needed for backward.
pub struct TrainForwardOutput {
    pub logits: Vec<f32>,
    pub routing_data: Vec<crate::model::cpu_moe::MoERoutingData>,
    pub activations: Vec<LayerActivations>,
    /// Post-final-norm hidden states [seq, hidden_dim] — input to lm_head
    pub final_hidden: Vec<f32>,
}

impl CpuBlockAttnResModel {
    /// Forward pass that stores all activations needed for proper backward.
    /// This is the ONLY forward that should be used during training.
    ///
    /// Key difference from `forward_with_routing`:
    /// - Stores pre-attention normed input (for Q/K/V/O LoRA grad)
    /// - Stores pre-FFN normed input (for expert LoRA grad)
    /// - Stores per-expert intermediate activations (gate, up, combined — for expert backward)
    /// - Stores post-attention raw output (before O-LoRA — for O projection LoRA grad)
    ///
    /// All stored inline during the actual forward — no extra computation.
    pub fn forward_train(&self, token_ids: &[u32]) -> TrainForwardOutput {
        let seq = token_ids.len();
        let hd = self.hidden_dim;
        let vs = self.vocab_size;

        // 1. Embedding
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.embed_tokens.len() {
                for d in 0..hd { hidden[t * hd + d] = self.embed_tokens[id * hd + d]; }
            }
        }
        let scale = (hd as f32).sqrt();
        for h in hidden.iter_mut() { *h *= scale; }

        // 2. PLE precompute
        let ple_dim = self.hidden_size_per_layer_input;
        let ple_precomputed = self.precompute_ple(&hidden, token_ids, seq, hd, ple_dim);

        // 3. Per-layer forward with activation storage
        let first_shared_layer = self.num_layers.saturating_sub(self.num_kv_shared_layers);
        let mut shared_kv: std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)> = std::collections::HashMap::new();
        let mut block_reps = Vec::new();
        let mut partial_sum = vec![0.0f32; hd];
        for t in 0..seq { for d in 0..hd { partial_sum[d] += hidden[t * hd + d]; } }
        for d in 0..hd { partial_sum[d] /= seq as f32; }
        block_reps.push(partial_sum.clone());

        let mut routing_data = Vec::new();
        let mut activations = Vec::with_capacity(self.num_layers);
        let lora_m = self.lora_manager.as_ref();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let ple_slice = ple_precomputed.as_ref().map(|pre| {
                let mut slice = vec![0.0f32; seq * ple_dim];
                for t in 0..seq {
                    let src = t * self.num_layers * ple_dim + layer_idx * ple_dim;
                    let dst = t * ple_dim;
                    for d in 0..ple_dim { slice[dst + d] = pre[src + d]; }
                }
                slice
            });

            let (kv, should_store) = if layer_idx >= first_shared_layer && first_shared_layer > 0 && layer.kv_shared {
                let kv_source = Self::kv_shared_source_layer(layer_idx, first_shared_layer, &self.layers);
                if let Some((sk, sv)) = shared_kv.get(&kv_source) {
                    (Some((sk.as_slice(), sv.as_slice())), false)
                } else { (None, false) }
            } else { (None, true) };

            let nh = layer.num_heads;
            let nkv = layer.num_kv_heads;
            let head_d = layer.head_dim;
            let q_dim = nh * head_d;
            let kv_dim_out = nkv * head_d;

            // === Attention ===
            let _residual_attn = hidden.clone();
            let pre_attn_normed = layer.attn_norm.forward(&hidden);

            // Q + LoRA
            let mut q = layer.q_proj.forward(&pre_attn_normed, seq);
            if let Some(ref lm) = lora_m {
                if let Some(lo) = lm.forward(layer_idx, "q_proj", &pre_attn_normed, seq) {
                    for (i, v) in lo.iter().enumerate() { q[i] += v; }
                }
            }
            q = gemma_mapper::per_head_rms_norm(&q, layer.q_norm.weight(), seq, nh, head_d);
            apply_rope(&mut q, seq, nh, head_d, 0, layer.rope_theta, layer.partial_rotary_factor);

            // K/V + LoRA
            let (k, v) = match kv {
                Some((sk, sv)) => (sk.to_vec(), sv.to_vec()),
                None => {
                    let mut k = layer.k_proj.forward(&pre_attn_normed, seq);
                    if let Some(ref lm) = lora_m {
                        if let Some(lo) = lm.forward(layer_idx, "k_proj", &pre_attn_normed, seq) {
                            for (i, l) in lo.iter().enumerate() { k[i] += l; }
                        }
                    }
                    let mut v_raw = layer.v_proj.forward(&pre_attn_normed, seq);
                    if let Some(ref lm) = lora_m {
                        if let Some(lo) = lm.forward(layer_idx, "v_proj", &pre_attn_normed, seq) {
                            for (i, l) in lo.iter().enumerate() { v_raw[i] += l; }
                        }
                    }
                    k = gemma_mapper::per_head_rms_norm(&k, layer.k_norm.weight(), seq, nkv, head_d);
                    let v = gemma_mapper::per_head_rms_norm_no_scale(&v_raw, seq, nkv, head_d);
                    apply_rope_gqa(&mut k, seq, nkv, head_d, 0, layer.rope_theta, layer.partial_rotary_factor);
                    (k, v)
                }
            };

            // Attention scores
            let mut attn_out = layer.cpu_attention(&q, &k, &v, seq, nh, nkv, head_d, q_dim, kv_dim_out);
            let post_attn_raw = attn_out.clone();
            if let Some(ref lm) = lora_m {
                if let Some(lo) = lm.forward(layer_idx, "o_proj", &attn_out, seq) {
                    for (i, l) in lo.iter().enumerate() { attn_out[i] += l; }
                }
            }
            let attn_out = layer.post_attn_norm.forward(&attn_out);
            for i in 0..hidden.len() { hidden[i] = _residual_attn[i] + attn_out[i]; }

            if should_store && first_shared_layer > 0 && !k.is_empty() {
                shared_kv.insert(layer_idx, (k, v));
            }

            // === FFN with activation + routing collection ===
            let _residual_ffn = hidden.clone();
            let pre_ffn_normed = layer.pre_ffn_norm.forward(&hidden);

            let (ffn_out, layer_routing, layer_expert_act) = if let Some(ref moe) = layer.moe {
                let ne = moe.num_experts;
                let tk = moe.top_k;
                let gate_logits = gemma_mapper::matmul(&pre_ffn_normed, &moe.gate_weights, seq, hd, ne);
                let (selected, weights, gate_probs) = moe.top_k_select_with_probs(&gate_logits, seq);

                let mut output = vec![0.0f32; seq * hd];
                let mut all_expert_act = Vec::with_capacity(seq);

                for t in 0..seq {
                    let mut token_expert_act = Vec::with_capacity(tk);
                    for k_idx in 0..tk {
                        let ei = selected[t * tk + k_idx];
                        let w = weights[t * tk + k_idx];
                        if w.abs() < 1e-8 {
                            token_expert_act.push(ExpertActivation {
                                expert_idx: ei, gated: vec![], upped: vec![], combined: vec![], input: vec![],
                            });
                            continue;
                        }
                        let token = &pre_ffn_normed[t * hd..(t + 1) * hd];
                        let (expert_out, gated, upped, combined) = moe.expert_forward_store_act(
                            ei, token, lora_m, layer_idx,
                        );
                        for (i, &v) in expert_out.iter().enumerate() {
                            output[t * hd + i] += w * v;
                        }
                        token_expert_act.push(ExpertActivation {
                            expert_idx: ei, gated, upped, combined, input: token.to_vec(),
                        });
                    }
                    all_expert_act.push(token_expert_act);
                }

                let routing = crate::model::cpu_moe::MoERoutingData {
                    layer_idx,
                    gate_logits,
                    gate_probs,
                    selected_experts: selected,
                    expert_weights: weights,
                    pre_ffn_input: pre_ffn_normed.clone(),
                    expert_outputs: vec![],
                };
                (output, Some(routing), Some(all_expert_act))
            } else {
                (layer.cpu_ffn(&pre_ffn_normed, seq), None, None)
            };

            let ffn_out = layer.post_ffn_norm.forward(&ffn_out);
            for i in 0..hidden.len() { hidden[i] = _residual_ffn[i] + ffn_out[i] * layer.layer_scalar; }

            // PLE injection
            if let Some(ref ple_s) = ple_slice {
                if let (Some(ref gate), Some(ref proj), Some(ref norm)) =
                    (&layer.ple_input_gate, &layer.ple_projection, &layer.ple_post_norm)
                {
                    let ple_dim = ple_s.len() / seq;
                    let gate_out = gate.forward(&hidden, seq);
                    let gate_gelu: Vec<f32> = gate_out.iter().map(|&x| gemma_mapper::gelu_tanh(x)).collect();
                    let mut gated = vec![0.0f32; seq * ple_dim];
                    for i in 0..seq * ple_dim { gated[i] = gate_gelu[i] * ple_s[i]; }
                    let proj_out = proj.forward(&gated, seq);
                    let ple_final = norm.forward(&proj_out);
                    for i in 0..hidden.len() { hidden[i] += ple_final[i]; }
                }
            }

            if let Some(rd) = layer_routing { routing_data.push(rd); }
            activations.push(LayerActivations {
                pre_attn_normed,
                post_attn_raw,
                pre_ffn_normed,
                expert_activations: layer_expert_act.unwrap_or_default(),
            });

            for t in 0..seq { for d in 0..hd { partial_sum[d] += hidden[t * hd + d]; } }
            if self.is_block_boundary(layer_idx) {
                for d in 0..hd { partial_sum[d] /= ((seq) * (self.block_config.layers_per_block)) as f32; }
                block_reps.push(partial_sum.clone());
                let inter_out = self.inter_block_attention(&hidden, &block_reps, seq);
                for t in 0..seq { for d in 0..hd { hidden[t * hd + d] += inter_out[d]; } }
                partial_sum = vec![0.0f32; hd];
            }
        }

        // 4. Final norm + LM head
        hidden = gemma_mapper::rms_norm(&hidden, &self.final_norm, hd, 1e-6);
        let final_hidden = hidden.clone();
        let mut logits = gemma_mapper::matmul(&hidden, &self.lm_head, seq, hd, vs);
        if let Some(cap) = self.final_logit_softcapping {
            for l in logits.iter_mut() { *l = (*l / cap).tanh() * cap; }
        }

        TrainForwardOutput { logits, routing_data, activations, final_hidden }
    }
}

/// Transpose a row-major flat matrix.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
