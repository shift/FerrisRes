//! Reverse-mode LoRA gradients for the blocked CPU reference model.
//!
//! [`CpuBlockAttnResModel::forward_train`] stores only the activations the
//! distillation loop reads (pre-attention and pre-FFN norms, the raw attention
//! concatenation, per-layer hidden states). This module reconstructs the rest
//! of each layer from those values and walks the graph in reverse, so a LoRA
//! adapter receives the gradient of the loss w.r.t. its module's output.
//!
//! The four terms the old surrogate skipped are all present here:
//!
//! * the final-norm Jacobian ([`rmsnorm_bwd`] on `final_norm`);
//! * the attention Jacobian, including the softmax and the value path
//!   ([`attention_bwd`]);
//! * per-layer `d_hidden`: the gradient is propagated through every layer
//!   instead of reusing the logit gradient for every projection;
//! * the inter-block suffix sum: a block summary is a causal prefix mean, so
//!   its gradient is summed back over every later position of every layer in
//!   the block ([`propagate_mean_grad`]).
//!
//! The block bookkeeping mirrors the f32 reference trainer; that trainer's
//! backward has finite-difference tests in `f32_trainer.rs`, which is why its
//! primitives are reused here rather than re-derived.
//!
//! # Scope
//!
//! [`CpuBlockAttnResModel::lora_output_gradients`] returns `None` for paths the
//! reference backward does not cover yet: a mixture-of-experts FFN, per-layer
//! embeddings (PLE), or KV sharing. Those configurations keep the previous
//! behaviour at the call site; they are not silently given an approximate
//! gradient from this function.

use crate::model::cpu_block_attn_res::{
    add_layer_prefix, block_index_of_layer, embedding_prefix, prefix_means, CpuBlockAttnResModel,
};
use crate::model::gemma_mapper::{
    apply_rope, apply_rope_gqa, gelu_tanh, per_head_rms_norm, per_head_rms_norm_no_scale, rms_norm,
};
use crate::training::backward::TrainForwardOutput;
use crate::training::f32_trainer::{
    attention_bwd, attention_fwd, inter_block_attention_bwd, inter_block_attention_fwd,
    propagate_mean_grad, rmsnorm_bwd, rmsnorm_no_scale_bwd,
};

/// The gradient of a scalar loss w.r.t. one LoRA adapter's module output.
///
/// `grad_output` is `[seq, out_features]`, the same tensor
/// [`crate::training::lora::LoraLayer::backward`] expects as its first
/// argument.
#[derive(Clone, Debug)]
pub struct LoraOutputGradient {
    pub layer_idx: usize,
    pub module: String,
    pub grad_output: Vec<f32>,
}

impl CpuBlockAttnResModel {
    /// Reverse-mode gradients for every LoRA adapter in a dense, blocked model.
    ///
    /// Returns `None` when the model uses a path this reference backward does
    /// not implement (MoE, PLE, KV sharing). The caller is expected to keep its
    /// previous gradient path in that case.
    pub fn lora_output_gradients(
        &self,
        token_ids: &[u32],
        fwd: &TrainForwardOutput,
        d_logits: &[f32],
    ) -> Option<Vec<LoraOutputGradient>> {
        if !self.supported_by_reference_backward() {
            return None;
        }
        if fwd.activations.len() != self.num_layers {
            return None;
        }

        let hd = self.hidden_dim;
        let seq = token_ids.len();
        if seq == 0 || fwd.final_hidden.len() != seq * hd {
            return None;
        }

        let boundaries = &self.block_config.boundary_layers;
        let has_blocks = !boundaries.is_empty();
        let lpb = self.block_config.layers_per_block.max(1);
        let num_entries = boundaries.len() + 1;
        let lora_m = self.lora_manager.as_ref();

        // ---- rebuild the causal block summaries -----------------------------
        //
        // `per_layer_hidden[l]` is layer l's output *before* any inter-block
        // residual, exactly the tensor `add_layer_prefix` consumed in the
        // forward pass, so the prefix sums are reproducible without a second
        // forward.
        let mut x_embed = vec![0.0f32; seq * hd];
        let sqrt_hd = (hd as f32).sqrt();
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.embed_tokens.len() {
                for d in 0..hd {
                    x_embed[t * hd + d] = self.embed_tokens[id * hd + d] * sqrt_hd;
                }
            }
        }

        let mut prefix_sums: Vec<Vec<f32>> = if has_blocks {
            vec![vec![0.0f32; seq * hd]; num_entries]
        } else {
            Vec::new()
        };
        if has_blocks {
            prefix_sums[0] = embedding_prefix(&x_embed, seq, hd);
            for (l, x_out) in fwd.per_layer_hidden.iter().enumerate() {
                let bi = block_index_of_layer(boundaries, l);
                add_layer_prefix(&mut prefix_sums[bi + 1], x_out, seq, hd);
            }
        }

        // The inter-block residual added after each boundary layer. It is part
        // of the next layer's input, so the layer input can be reconstructed.
        let mut inter_out: Vec<Option<Vec<f32>>> = vec![None; self.num_layers];
        if has_blocks {
            for l in 0..self.num_layers {
                if self.is_block_boundary(l) {
                    let bi = block_index_of_layer(boundaries, l);
                    let means = prefix_means(&prefix_sums, bi + 1, seq, hd, lpb);
                    let inter = inter_block_attention_fwd(
                        &fwd.per_layer_hidden[l],
                        &means,
                        &self.block_config.attn_res_norm,
                        seq,
                        hd,
                        1e-6,
                    );
                    inter_out[l] = Some(inter.output);
                }
            }
        }

        let mut x_in_all: Vec<Vec<f32>> = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            if l == 0 {
                x_in_all.push(x_embed.clone());
            } else {
                let mut x_in = fwd.per_layer_hidden[l - 1].clone();
                if let Some(inter) = &inter_out[l - 1] {
                    for i in 0..x_in.len() {
                        x_in[i] += inter[i];
                    }
                }
                x_in_all.push(x_in);
            }
        }
        let last = self.num_layers - 1;
        let mut x_final_in = fwd.per_layer_hidden[last].clone();
        if let Some(inter) = &inter_out[last] {
            for i in 0..x_final_in.len() {
                x_final_in[i] += inter[i];
            }
        }

        // ---- lm_head + final norm ------------------------------------------
        // The audit's loss is linear in the logits, but a capped model still
        // needs the tanh derivative: logits = tanh(z / cap) * cap.
        let mut d_logits_eff = d_logits.to_vec();
        if let Some(cap) = self.final_logit_softcapping {
            if cap != 0.0 {
                for i in 0..d_logits_eff.len() {
                    let t = fwd_logit_derivative(fwd, i, cap);
                    d_logits_eff[i] *= t;
                }
            }
        }

        let mut dh_final = vec![0.0f32; seq * hd];
        matmul_bwd_x_in_out(&d_logits_eff, &self.lm_head, &mut dh_final, seq, hd, self.vocab_size);

        let mut dx = vec![0.0f32; seq * hd];
        let mut dw_final = vec![0.0f32; hd];
        rmsnorm_bwd(
            &x_final_in,
            &self.final_norm,
            &dh_final,
            &mut dw_final,
            &mut dx,
            seq,
            hd,
            1e-6,
        );

        // ---- block-summary gradients ---------------------------------------
        let mut d_block_means: Vec<Vec<f32>> = if has_blocks {
            vec![vec![0.0f32; seq * hd]; num_entries]
        } else {
            Vec::new()
        };
        let mut d_layer_extra: Vec<Vec<f32>> = if has_blocks {
            vec![vec![0.0f32; seq * hd]; self.num_layers]
        } else {
            Vec::new()
        };

        let mut out: Vec<LoraOutputGradient> = Vec::new();
        // Only adapted modules need their output gradient returned; the
        // scratch values still feed the rest of the layer.
        let adapted = |layer: usize, module: &str| {
            lora_m.map_or(false, |lm| lm.get(layer, module).is_some())
        };

        for l in (0..self.num_layers).rev() {
            let layer = &self.layers[l];
            let act = &fwd.activations[l];
            let pre_attn = &act.pre_attn_normed;
            let attn_raw = &act.post_attn_raw;
            let pre_ffn = &act.pre_ffn_normed;
            let x_in = &x_in_all[l];
            let x_out = &fwd.per_layer_hidden[l];

            let nh = layer.num_heads;
            let nkv = layer.num_kv_heads;
            let head_d = layer.head_dim;
            let q_dim = nh * head_d;
            let kv_dim = nkv * head_d;
            let inter_dim = layer.intermediate_dim;

            // Inter-block attention at this boundary consumed `x_out` as the
            // query and its output was added to `dx`. Split that gradient.
            if has_blocks && self.is_block_boundary(l) {
                let bi = block_index_of_layer(boundaries, l);
                let n = bi + 2;
                let means = prefix_means(&prefix_sums, bi + 1, seq, hd, lpb);
                let normed_reps: Vec<Vec<f32>> = means
                    .iter()
                    .map(|rep| rms_norm(rep, &self.block_config.attn_res_norm, hd, 1e-6))
                    .collect();
                let inter = inter_block_attention_fwd(
                    x_out,
                    &means,
                    &self.block_config.attn_res_norm,
                    seq,
                    hd,
                    1e-6,
                );
                let mut d_query = vec![0.0f32; seq * hd];
                let mut dw_res = vec![0.0f32; hd];
                inter_block_attention_bwd(
                    x_out,
                    &means,
                    &normed_reps,
                    &inter.weights,
                    &dx,
                    &self.block_config.attn_res_norm,
                    &mut dw_res,
                    &mut d_query,
                    &mut d_block_means[..n],
                    seq,
                    hd,
                    1e-6,
                );
                for i in 0..seq * hd {
                    dx[i] += d_query[i];
                }

                // Every layer of this block contributed to the summary, and
                // the summary is a prefix mean: suffix-sum its gradient back.
                let mut d_block = vec![0.0f32; seq * hd];
                propagate_mean_grad(
                    &d_block_means[bi + 1],
                    seq,
                    hd,
                    lpb as f32,
                    &mut d_block,
                );
                let first = bi * lpb;
                for bl in first..=l {
                    for i in 0..seq * hd {
                        d_layer_extra[bl][i] += d_block[i];
                    }
                }
            }
            if has_blocks {
                for i in 0..seq * hd {
                    dx[i] += d_layer_extra[l][i];
                }
            }

            // ---- layer output -------------------------------------------------
            // x_out = (x_mid + post_ffn_normed) * layer_scalar
            let scalar = layer.layer_scalar;
            let d_pre_scalar: Vec<f32> = dx.iter().map(|&v| v * scalar).collect();

            // Rebuild the attention intermediates. `pre_attn_normed` and
            // `attn_raw` come from the forward pass, so only the projections,
            // norms and RoPE need replaying. This must run before `x_mid`,
            // because `x_mid` contains the post-attention branch.
            let mut q_raw = layer.q_proj.forward(pre_attn, seq);
            if let Some(lm) = lora_m {
                if let Some(lo) = lm.forward(l, "q_proj", pre_attn, seq) {
                    for (i, v) in lo.iter().enumerate() {
                        q_raw[i] += v;
                    }
                }
            }
            let mut q = per_head_rms_norm(&q_raw, layer.q_norm.weight(), seq, nh, head_d);
            apply_rope(
                &mut q,
                seq,
                nh,
                head_d,
                0,
                layer.rope_theta,
                layer.partial_rotary_factor,
            );

            let mut k_raw = layer.k_proj.forward(pre_attn, seq);
            if let Some(lm) = lora_m {
                if let Some(lo) = lm.forward(l, "k_proj", pre_attn, seq) {
                    for (i, v) in lo.iter().enumerate() {
                        k_raw[i] += v;
                    }
                }
            }
            let mut k = per_head_rms_norm(&k_raw, layer.k_norm.weight(), seq, nkv, head_d);
            apply_rope_gqa(
                &mut k,
                seq,
                nkv,
                head_d,
                0,
                layer.rope_theta,
                layer.partial_rotary_factor,
            );

            let mut v_raw = layer.v_proj.forward(pre_attn, seq);
            if let Some(lm) = lora_m {
                if let Some(lo) = lm.forward(l, "v_proj", pre_attn, seq) {
                    for (i, v) in lo.iter().enumerate() {
                        v_raw[i] += v;
                    }
                }
            }
            let v = per_head_rms_norm_no_scale(&v_raw, seq, nkv, head_d);

            let mut probs = vec![0.0f32; nh * seq * seq];
            let mut attn_concat = vec![0.0f32; seq * q_dim];
            attention_fwd(
                &q,
                &k,
                &v,
                seq,
                nh,
                nkv,
                head_d,
                &mut probs,
                &mut attn_concat,
            );

            let mut attn_out = layer.out_proj.forward(attn_raw, seq);
            if let Some(lm) = lora_m {
                if let Some(lo) = lm.forward(l, "o_proj", attn_raw, seq) {
                    for (i, v) in lo.iter().enumerate() {
                        attn_out[i] += v;
                    }
                }
            }
            let post_attn_normed = layer.post_attn_norm.forward(&attn_out);

            // x_mid = x_in + post_attn_normed (the attention residual).
            let mut x_mid = vec![0.0f32; seq * hd];
            for i in 0..seq * hd {
                x_mid[i] = x_in[i] + post_attn_normed[i];
            }

            // ---- FFN (dense) ---------------------------------------------------
            let ffn_gate = layer.ffn_gate.as_ref()?;
            let ffn_up = layer.ffn_up.as_ref()?;
            let ffn_down = layer.ffn_down.as_ref()?;

            let gate_pre = ffn_gate.forward(pre_ffn, seq);
            let up = ffn_up.forward(pre_ffn, seq);
            let activated: Vec<f32> = gate_pre
                .iter()
                .map(|&x| if layer.use_gelu { gelu_tanh(x) } else { silu(x) })
                .collect();
            let mut combined = vec![0.0f32; seq * inter_dim];
            for i in 0..combined.len() {
                combined[i] = activated[i] * up[i];
            }
            let ffn_out = ffn_down.forward(&combined, seq);

            let mut d_ffn_out = vec![0.0f32; seq * hd];
            let mut dw = vec![0.0f32; hd];
            rmsnorm_bwd(
                &ffn_out,
                layer.post_ffn_norm.weight(),
                &d_pre_scalar,
                &mut dw,
                &mut d_ffn_out,
                seq,
                hd,
                layer.post_ffn_norm.eps(),
            );

            let mut d_combined = vec![0.0f32; seq * inter_dim];
            linear_input_grad(
                &d_ffn_out,
                &ffn_down.weight(),
                &mut d_combined,
                seq,
                inter_dim,
                hd,
            );

            let mut d_gate_pre = vec![0.0f32; seq * inter_dim];
            let mut d_up = vec![0.0f32; seq * inter_dim];
            for i in 0..seq * inter_dim {
                let g = gate_pre[i];
                let grad_act = if layer.use_gelu {
                    gelu_tanh_grad(g)
                } else {
                    silu_grad(g)
                };
                d_gate_pre[i] = d_combined[i] * up[i] * grad_act;
                d_up[i] = d_combined[i] * activated[i];
            }

            let mut d_h_ffn = vec![0.0f32; seq * hd];
            linear_input_grad(&d_gate_pre, &ffn_gate.weight(), &mut d_h_ffn, seq, hd, inter_dim);
            let mut d_h_ffn_up = vec![0.0f32; seq * hd];
            linear_input_grad(&d_up, &ffn_up.weight(), &mut d_h_ffn_up, seq, hd, inter_dim);
            for i in 0..seq * hd {
                d_h_ffn[i] += d_h_ffn_up[i];
            }

            let mut d_x_mid_ffn = vec![0.0f32; seq * hd];
            let mut dw = vec![0.0f32; hd];
            rmsnorm_bwd(
                &x_mid,
                layer.pre_ffn_norm.weight(),
                &d_h_ffn,
                &mut dw,
                &mut d_x_mid_ffn,
                seq,
                hd,
                layer.pre_ffn_norm.eps(),
            );
            // x_out = (x_mid + post_ffn_normed) * scalar reaches `x_mid` twice:
            // directly and through `pre_ffn_norm`.
            let mut d_x_mid = d_pre_scalar.clone();
            for i in 0..seq * hd {
                d_x_mid[i] += d_x_mid_ffn[i];
            }

            // ---- attention backward -------------------------------------------
            // x_mid = x_in + post_attn_normed, so the attention branch receives
            // the whole `d_x_mid`.
            let d_post_attn = d_x_mid.clone();

            let mut d_attn_out = vec![0.0f32; seq * hd];
            let mut dw = vec![0.0f32; hd];
            rmsnorm_bwd(
                &attn_out,
                layer.post_attn_norm.weight(),
                &d_post_attn,
                &mut dw,
                &mut d_attn_out,
                seq,
                hd,
                layer.post_attn_norm.eps(),
            );
            if adapted(l, "o_proj") {
                out.push(LoraOutputGradient {
                    layer_idx: l,
                    module: "o_proj".to_string(),
                    grad_output: d_attn_out.clone(),
                });
            }

            let mut d_attn_concat = vec![0.0f32; seq * q_dim];
            linear_input_grad(
                &d_attn_out,
                &layer.out_proj.weight(),
                &mut d_attn_concat,
                seq,
                q_dim,
                hd,
            );

            let mut dq = vec![0.0f32; seq * q_dim];
            let mut dk = vec![0.0f32; seq * kv_dim];
            let mut dv = vec![0.0f32; seq * kv_dim];
            attention_bwd(
                &d_attn_concat,
                &probs,
                &q,
                &k,
                &v,
                seq,
                nh,
                nkv,
                head_d,
                &mut dq,
                &mut dk,
                &mut dv,
            );

            rope_backward(
                &mut dq,
                seq,
                nh,
                head_d,
                layer.rope_theta,
                layer.partial_rotary_factor,
            );
            rope_backward(
                &mut dk,
                seq,
                nkv,
                head_d,
                layer.rope_theta,
                layer.partial_rotary_factor,
            );

            let mut dq_proj = vec![0.0f32; seq * q_dim];
            let mut dw = vec![0.0f32; head_d];
            rmsnorm_bwd(
                &q_raw,
                layer.q_norm.weight(),
                &dq,
                &mut dw,
                &mut dq_proj,
                seq * nh,
                head_d,
                1e-6,
            );
            let mut dk_proj = vec![0.0f32; seq * kv_dim];
            let mut dw = vec![0.0f32; head_d];
            rmsnorm_bwd(
                &k_raw,
                layer.k_norm.weight(),
                &dk,
                &mut dw,
                &mut dk_proj,
                seq * nkv,
                head_d,
                1e-6,
            );
            let mut dv_raw = vec![0.0f32; seq * kv_dim];
            rmsnorm_no_scale_bwd(&v_raw, &dv, &mut dv_raw, seq * nkv, head_d, 1e-6);

            // The adapter is added to the projection output, so its upstream
            // gradient is exactly the gradient w.r.t. that sum.
            if adapted(l, "q_proj") {
                out.push(LoraOutputGradient {
                    layer_idx: l,
                    module: "q_proj".to_string(),
                    grad_output: dq_proj.clone(),
                });
            }
            if adapted(l, "k_proj") {
                out.push(LoraOutputGradient {
                    layer_idx: l,
                    module: "k_proj".to_string(),
                    grad_output: dk_proj.clone(),
                });
            }
            if adapted(l, "v_proj") {
                out.push(LoraOutputGradient {
                    layer_idx: l,
                    module: "v_proj".to_string(),
                    grad_output: dv_raw.clone(),
                });
            }

            let mut d_h_attn = vec![0.0f32; seq * hd];
            linear_input_grad(
                &dq_proj,
                &layer.q_proj.weight(),
                &mut d_h_attn,
                seq,
                hd,
                q_dim,
            );
            let mut d_h_attn_k = vec![0.0f32; seq * hd];
            linear_input_grad(
                &dk_proj,
                &layer.k_proj.weight(),
                &mut d_h_attn_k,
                seq,
                hd,
                kv_dim,
            );
            let mut d_h_attn_v = vec![0.0f32; seq * hd];
            linear_input_grad(
                &dv_raw,
                &layer.v_proj.weight(),
                &mut d_h_attn_v,
                seq,
                hd,
                kv_dim,
            );
            for i in 0..seq * hd {
                d_h_attn[i] += d_h_attn_k[i] + d_h_attn_v[i];
            }

            let mut d_x_in = vec![0.0f32; seq * hd];
            let mut dw = vec![0.0f32; hd];
            rmsnorm_bwd(
                x_in,
                layer.attn_norm.weight(),
                &d_h_attn,
                &mut dw,
                &mut d_x_in,
                seq,
                hd,
                layer.attn_norm.eps(),
            );

            for i in 0..seq * hd {
                dx[i] = d_x_mid[i] + d_x_in[i];
            }

            // Keep the reconstruction honest: `attn_concat` must equal the
            // forward's stored raw attention, which pins the attention replay.
            debug_assert_attn_matches(attn_raw, &attn_concat);
        }

        // Entry 0's summary is the initial embedding; its suffix-summed gradient
        // only reaches `embed_tokens`, so nothing here needs it for LoRA.

        Some(out)
    }

    /// True when every layer uses the dense FFN and no auxiliary path (MoE,
    /// PLE, KV sharing) changes the hidden state.
    fn supported_by_reference_backward(&self) -> bool {
        if self.hidden_size_per_layer_input != 0 || self.ple_model_projection.is_some() {
            return false;
        }
        if self.num_kv_shared_layers != 0 {
            return false;
        }
        self.layers.iter().all(|layer| {
            layer.moe.is_none()
                && layer.ffn_gate.is_some()
                && layer.ffn_up.is_some()
                && layer.ffn_down.is_some()
                && layer.ple_input_gate.is_none()
        })
    }
}

/// `dx[t, i] = sum_o dy[t, o] * w[o * in_dim + i]`.
///
/// [`crate::model::cpu_linear::CpuLinear`] stores weights as `[out, in]`
/// (see `ternary_matmul`), while `gemma_mapper::matmul` consumes `[in, out]`
/// for the raw `lm_head` matrix. Keeping the two helpers separate makes the
/// layout explicit at each call.
fn linear_input_grad(
    dy: &[f32],
    w_out_in: &[f32],
    dx: &mut [f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(dy.len(), rows * out_dim);
    debug_assert_eq!(w_out_in.len(), in_dim * out_dim);
    debug_assert_eq!(dx.len(), rows * in_dim);
    for t in 0..rows {
        let dyr = &dy[t * out_dim..(t + 1) * out_dim];
        let dxr = &mut dx[t * in_dim..(t + 1) * in_dim];
        for i in 0..in_dim {
            let mut acc = 0.0f32;
            for o in 0..out_dim {
                acc += dyr[o] * w_out_in[o * in_dim + i];
            }
            dxr[i] = acc;
        }
    }
}

/// `dx[t, i] = sum_o dy[t, o] * w[i * out_dim + o]` for an `[in, out]` matrix.
fn matmul_bwd_x_in_out(
    dy: &[f32],
    w_in_out: &[f32],
    dx: &mut [f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) {
    debug_assert_eq!(dy.len(), rows * out_dim);
    debug_assert_eq!(w_in_out.len(), in_dim * out_dim);
    debug_assert_eq!(dx.len(), rows * in_dim);
    for t in 0..rows {
        let dyr = &dy[t * out_dim..(t + 1) * out_dim];
        let dxr = &mut dx[t * in_dim..(t + 1) * in_dim];
        for i in 0..in_dim {
            let wr = &w_in_out[i * out_dim..(i + 1) * out_dim];
            let mut acc = 0.0f32;
            for o in 0..out_dim {
                acc += dyr[o] * wr[o];
            }
            dxr[i] = acc;
        }
    }
}

/// Transposed RoPE, matching [`apply_rope`] including the proportional factor.
fn rope_backward(
    dx: &mut [f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    theta: f64,
    partial_rotary_factor: f32,
) {
    let rotary_dims = (head_dim as f32 * partial_rotary_factor) as usize;
    let half = rotary_dims / 2;
    if half == 0 {
        return;
    }
    for t in 0..seq {
        let pos = t as f64;
        for h in 0..heads {
            for d in 0..half {
                let freq = 1.0 / theta.powf((2 * d) as f64 / head_dim as f64);
                let angle = pos * freq;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;

                let base = t * heads * head_dim + h * head_dim;
                let d0 = dx[base + d];
                let d1 = dx[base + d + half];
                dx[base + d] = d0 * cos_a + d1 * sin_a;
                dx[base + d + half] = -d0 * sin_a + d1 * cos_a;
            }
        }
    }
}

/// Derivative of [`gelu_tanh`].
fn gelu_tanh_grad(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    let tanh = inner.tanh();
    0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * c * (1.0 + 3.0 * 0.044715 * x * x)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

/// Derivative of the logit cap at flat index `i`.
fn fwd_logit_derivative(fwd: &TrainForwardOutput, i: usize, cap: f32) -> f32 {
    let t = fwd.logits[i] / cap;
    1.0 - t * t
}

/// Debug-only check that the replayed attention matches the stored
/// `post_attn_raw` tensor from the forward pass.
fn debug_assert_attn_matches(stored: &[f32], replayed: &[f32]) {
    debug_assert_eq!(stored.len(), replayed.len());
    if stored.len() != replayed.len() {
        return;
    }
    let mut max_diff = 0.0f32;
    for (a, b) in stored.iter().zip(replayed.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    debug_assert!(
        max_diff < 1e-3,
        "attention replay diverged from the forward pass: max diff {max_diff}"
    );
}
