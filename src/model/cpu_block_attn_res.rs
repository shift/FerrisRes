use crate::model::cpu_linear::{CpuLinear, CpuRmsNorm};
use crate::model::cpu_moe::CpuMoELayer;
use crate::model::gemma_mapper::{matmul, rms_norm, apply_rope, apply_rope_gqa, gelu_tanh};

/// CPU-only BlockAttnResLayer with full Gemma 4 architectural support.
///
/// This is the student model layer for distillation. It mirrors the GPU
/// BlockAttnResLayer but uses Vec<f32> weights and CPU computation.
/// All 10 Gemma 4 features are supported:
/// 1. Post-attention RMSNorm
/// 2. Post-FFN RMSNorm
/// 3. Per-head Q/K RMSNorm
/// 4. V norm
/// 5. Layer scalar (out_scale)
/// 6. PLE (Per-Layer Embeddings) gate + projection
/// 7. KV sharing (accept externally-computed K/V)
/// 8. Per-layer RoPE parameters
/// 9. Per-layer dimensions (head_dim, intermediate_dim)
/// 10. GELU activation (configurable SwiGLU vs GELU)
pub struct CpuBlockAttnResLayer {
    pub layer_number: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,

    // Attention
    pub attn_norm: CpuRmsNorm,          // Pre-attention RMSNorm
    pub q_proj: CpuLinear,
    pub k_proj: CpuLinear,
    pub v_proj: CpuLinear,
    pub out_proj: CpuLinear,
    pub q_norm: CpuRmsNorm,             // Per-head Q RMSNorm [num_heads * head_dim]
    pub k_norm: CpuRmsNorm,             // Per-head K RMSNorm [num_kv_heads * head_dim]
    pub v_norm: CpuRmsNorm,             // Per-head V RMSNorm [num_kv_heads * head_dim]
    pub post_attn_norm: CpuRmsNorm,     // Post-attention RMSNorm (feature 1)

    // FFN
    pub pre_ffn_norm: CpuRmsNorm,       // Pre-FFN RMSNorm
    pub ffn_gate: Option<CpuLinear>,    // Dense FFN gate projection
    pub ffn_up: Option<CpuLinear>,      // Dense FFN up projection
    pub ffn_down: Option<CpuLinear>,    // Dense FFN down projection
    pub moe: Option<CpuMoELayer>,       // MoE FFN (alternative to dense)
    pub post_ffn_norm: CpuRmsNorm,      // Post-FFN RMSNorm (feature 2)

    // Layer scalar (feature 5)
    pub layer_scalar: f32,

    // PLE (feature 6) — per-layer gate and projection
    pub ple_input_gate: Option<CpuLinear>,   // [hidden_dim → ple_dim]
    pub ple_projection: Option<CpuLinear>,   // [ple_dim → hidden_dim]
    pub ple_post_norm: Option<CpuRmsNorm>,   // [hidden_dim]

    // RoPE parameters (feature 8)
    pub rope_theta: f64,
    pub partial_rotary_factor: f32,

    // Activation type (feature 10)
    pub use_gelu: bool,

    // KV sharing (feature 7) — if true, this layer uses externally-provided K/V
    pub kv_shared: bool,
}

impl CpuBlockAttnResLayer {
    /// Create a new layer with all-zero weights.
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        Self {
            layer_number: 0,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,

            // Attention
            attn_norm: CpuRmsNorm::new(hidden_dim, 1e-6),
            q_proj: CpuLinear::new(hidden_dim, q_dim, false),
            k_proj: CpuLinear::new(hidden_dim, kv_dim, false),
            v_proj: CpuLinear::new(hidden_dim, kv_dim, false),
            out_proj: CpuLinear::new(q_dim, hidden_dim, false),
            q_norm: CpuRmsNorm::new(head_dim, 1e-6),       // Per-head Q RMSNorm [head_dim]
            k_norm: CpuRmsNorm::new(head_dim, 1e-6),       // Per-head K RMSNorm [head_dim]
            v_norm: CpuRmsNorm::new(head_dim, 1e-6),       // Per-head V RMSNorm [head_dim]
            post_attn_norm: CpuRmsNorm::new(hidden_dim, 1e-6),

            // FFN
            pre_ffn_norm: CpuRmsNorm::new(hidden_dim, 1e-6),
            ffn_gate: Some(CpuLinear::new(hidden_dim, intermediate_dim, false)),
            ffn_up: Some(CpuLinear::new(hidden_dim, intermediate_dim, false)),
            ffn_down: Some(CpuLinear::new(intermediate_dim, hidden_dim, false)),
            moe: None,
            post_ffn_norm: CpuRmsNorm::new(hidden_dim, 1e-6),

            layer_scalar: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,

            rope_theta: 10000.0,
            partial_rotary_factor: 1.0,
            use_gelu: true, // Gemma 4 default
            kv_shared: false,
        }
    }

    /// Full layer forward: attention + FFN + PLE + layer_scalar.
    ///
    /// `hidden`: [seq, hidden_dim] input hidden states (modified in-place)
    /// `ple_input`: optional pre-computed PLE input for this layer [seq, ple_dim]
    /// `shared_kv`: optional externally-provided K/V for KV-shared layers
    ///   - shared_k: [seq, kv_dim]  shared K states
    ///   - shared_v: [seq, kv_dim]  shared V states
    ///   Returns: (k_states, v_states) for KV sharing with later layers
    pub fn forward(
        &self,
        hidden: &mut Vec<f32>,
        _ple_input: Option<&[f32]>,
        shared_kv: Option<(&[f32], &[f32])>,
    ) -> (Vec<f32>, Vec<f32>) {
        let hd = self.hidden_dim;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let head_d = self.head_dim;
        let seq = hidden.len() / hd;
        let q_dim = nh * head_d;
        let kv_dim = nkv * head_d;

        // === Attention Block ===
        let residual = hidden.clone();

        // Pre-attention RMSNorm
        let normed = self.attn_norm.forward(hidden);

        // Q projection + per-head norm + RoPE
        let mut q = self.q_proj.forward(&normed, seq);
        q = crate::model::gemma_mapper::per_head_rms_norm(&q, self.q_norm.weight(), seq, nh, head_d);
        apply_rope(&mut q, seq, nh, head_d, 0, self.rope_theta, self.partial_rotary_factor);

        // K/V: either shared or computed
        let (k, v, should_share) = match shared_kv {
            Some((shared_k, shared_v)) => {
                // KV-shared layer — use externally provided K/V
                (shared_k.to_vec(), shared_v.to_vec(), false)
            }
            None => {
                // Normal layer — compute own K/V
                let mut k = self.k_proj.forward(&normed, seq);
                let v_raw = self.v_proj.forward(&normed, seq);

                // Per-head K/V norms
                k = crate::model::gemma_mapper::per_head_rms_norm(&k, self.k_norm.weight(), seq, nkv, head_d);
                let v = crate::model::gemma_mapper::per_head_rms_norm_no_scale(&v_raw, seq, nkv, head_d);

                // RoPE on K
                apply_rope_gqa(&mut k, seq, nkv, head_d, 0, self.rope_theta, self.partial_rotary_factor);

                (k, v, true)
            }
        };

        // GQA attention with scale=1.0 (after per-head norm)
        let attn_out = self.cpu_attention(&q, &k, &v, seq, nh, nkv, head_d, q_dim, kv_dim);

        // Post-attention norm (feature 1)
        let attn_out = self.post_attn_norm.forward(&attn_out);

        // Residual
        for i in 0..hidden.len() {
            hidden[i] = residual[i] + attn_out[i];
        }

        // Return K/V for sharing (empty if we used shared K/V)
        if should_share { (k, v) } else { (vec![], vec![]) }
    }

    /// Complete the FFN + PLE + layer_scalar after attention.
    /// Call after forward() to apply the second half of the transformer layer.
    pub fn forward_ffn(
        &self,
        hidden: &mut Vec<f32>,
        ple_input: Option<&[f32]>,
    ) {
        let hd = self.hidden_dim;
        let seq = hidden.len() / hd;

        // === FFN Block ===
        let residual2 = hidden.clone();

        // Pre-FFN RMSNorm
        let normed2 = self.pre_ffn_norm.forward(hidden);

        // FFN (feature 10: GELU vs SwiGLU)
        let ffn_out = if let Some(ref moe) = self.moe {
            moe.forward(&normed2, seq)
        } else {
            self.cpu_ffn(&normed2, seq)
        };

        // Post-FFN norm (feature 2)
        let ffn_out = self.post_ffn_norm.forward(&ffn_out);

        // Residual
        for i in 0..hidden.len() {
            hidden[i] = residual2[i] + ffn_out[i];
        }

        // PLE injection (feature 6)
        if let Some(ple_slice) = ple_input {
            if let (Some(ref gate), Some(ref proj), Some(ref norm)) =
                (&self.ple_input_gate, &self.ple_projection, &self.ple_post_norm)
            {
                let ple_dim = ple_slice.len() / seq;
                // Gate: hidden → ple_dim with GELU
                let gate_out = gate.forward(hidden, seq);
                let gate_gelu: Vec<f32> = gate_out.iter().map(|&x| gelu_tanh(x)).collect();

                // Gated input: gate * ple_slice (element-wise)
                let mut gated = vec![0.0f32; seq * ple_dim];
                for i in 0..seq * ple_dim {
                    gated[i] = gate_gelu[i] * ple_slice[i];
                }

                // Project back: [seq, ple_dim] → [seq, hd]
                let proj_out = proj.forward(&gated, seq);

                // Post norm and add residual
                let ple_final = norm.forward(&proj_out);
                for i in 0..hidden.len() {
                    hidden[i] += ple_final[i];
                }
            }
        }

        // Layer scalar (feature 5)
        if self.layer_scalar != 1.0 {
            for h in hidden.iter_mut() {
                *h *= self.layer_scalar;
            }
        }
    }

    /// Full forward: attention + FFN + PLE + layer_scalar in one call.
    pub fn forward_full(
        &self,
        hidden: &mut Vec<f32>,
        ple_input: Option<&[f32]>,
        shared_kv: Option<(&[f32], &[f32])>,
    ) -> (Vec<f32>, Vec<f32>) {
        let kv = self.forward(hidden, ple_input, shared_kv);
        self.forward_ffn(hidden, ple_input);
        kv
    }

    /// CPU causal self-attention with GQA.
    /// scale = 1.0 (after per-head RMSNorm, no 1/sqrt(d) needed).
    fn cpu_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0f32; // Per-head RMSNorm replaces 1/sqrt(d)
        let mut attn_out = vec![0.0f32; seq * q_dim];

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            for t in 0..seq {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * q_dim + h * head_dim + d]
                             * k[s * kv_dim + kv_h * head_dim + d];
                    }
                    scores[s] = dot * scale;
                    if scores[s] > max_score { max_score = scores[s]; }
                }
                let mut sum_exp = 0.0f32;
                for s in 0..=t {
                    scores[s] = (scores[s] - max_score).exp();
                    sum_exp += scores[s];
                }
                for s in 0..=t { scores[s] /= sum_exp; }
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        sum += scores[s] * v[s * kv_dim + kv_h * head_dim + d];
                    }
                    attn_out[t * q_dim + h * head_dim + d] = sum;
                }
            }
        }

        // Output projection
        self.out_proj.forward(&attn_out, seq)
    }

    /// CPU dense FFN with configurable activation (feature 10).
    fn cpu_ffn(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let hd = self.hidden_dim;
        let id = self.intermediate_dim;

        if let (Some(gate), Some(up), Some(down)) =
            (&self.ffn_gate, &self.ffn_up, &self.ffn_down)
        {
            let gated = gate.forward(input, seq);
            let gated: Vec<f32> = if self.use_gelu {
                gated.iter().map(|&x| gelu_tanh(x)).collect()
            } else {
                // SwiGLU: silu(x) = x / (1 + exp(-x))
                gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
            };
            let upped = up.forward(input, seq);
            let mut combined = vec![0.0; seq * id];
            for i in 0..combined.len() {
                combined[i] = gated[i] * upped[i];
            }
            down.forward(&combined, seq)
        } else {
            vec![0.0; seq * hd]
        }
    }
}

// ---------------------------------------------------------------------------
// CpuBlockAttnResModel — full model with PLE pre-computation and KV sharing
// ---------------------------------------------------------------------------

/// Full BlockAttnRes model that runs on CPU with all Gemma 4 features.
/// Manages layers, PLE pre-computation, KV sharing, embedding, and LM head.
pub struct CpuBlockAttnResModel {
    pub layers: Vec<CpuBlockAttnResLayer>,
    pub embed_tokens: Vec<f32>,             // [vocab_size, hidden_dim]
    pub lm_head: Vec<f32>,                  // [hidden_dim, vocab_size]
    pub final_norm: Vec<f32>,               // [hidden_dim]
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub final_logit_softcapping: Option<f32>,

    // PLE model-level weights
    pub ple_model_projection: Option<Vec<f32>>,  // [hidden_dim, ple_total]
    pub ple_projection_norm: Option<Vec<f32>>,   // [ple_dim]
    pub embed_tokens_per_layer: Option<Vec<f32>>, // [vocab_size, ple_total]
    pub hidden_size_per_layer_input: usize,       // ple_dim
    pub num_kv_shared_layers: usize,              // number of shared KV layers
}

impl CpuBlockAttnResModel {
    /// Full forward pass: token IDs → logits.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq = token_ids.len();
        let hd = self.hidden_dim;
        let vs = self.vocab_size;

        // 1. Embedding (Gemma scales by sqrt(hidden_dim))
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.embed_tokens.len() {
                for d in 0..hd {
                    hidden[t * hd + d] = self.embed_tokens[id * hd + d];
                }
            }
        }
        let scale = (hd as f32).sqrt();
        for h in hidden.iter_mut() { *h *= scale; }

        // 2. Pre-compute PLE inputs from initial hidden state
        let ple_dim = self.hidden_size_per_layer_input;
        let ple_precomputed = self.precompute_ple(&hidden, token_ids, seq, hd, ple_dim);

        // 3. Per-layer transformer with KV sharing
        let first_shared_layer = self.num_layers.saturating_sub(self.num_kv_shared_layers);
        let mut shared_kv: std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)> = std::collections::HashMap::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Get PLE slice for this layer
            let ple_slice = ple_precomputed.as_ref().map(|pre| {
                let mut slice = vec![0.0f32; seq * ple_dim];
                for t in 0..seq {
                    let src = t * self.num_layers * ple_dim + layer_idx * ple_dim;
                    let dst = t * ple_dim;
                    for d in 0..ple_dim {
                        slice[dst + d] = pre[src + d];
                    }
                }
                slice
            });

            // Determine KV sharing
            let (kv, should_store) = if layer_idx >= first_shared_layer && first_shared_layer > 0 && layer.kv_shared {
                // Shared KV layer — look up source
                let kv_source = Self::kv_shared_source_layer(
                    layer_idx, first_shared_layer, &self.layers,
                );
                if let Some((sk, sv)) = shared_kv.get(&kv_source) {
                    (Some((sk.as_slice(), sv.as_slice())), false)
                } else {
                    (None, false)
                }
            } else {
                (None, true)
            };

            let (k_states, v_states) = layer.forward_full(
                &mut hidden,
                ple_slice.as_ref().map(|s| s.as_slice()),
                kv,
            );

            // Store K/V for later shared layers
            if should_store && first_shared_layer > 0 && !k_states.is_empty() {
                shared_kv.insert(layer_idx, (k_states, v_states));
            }
        }

        // 4. Final norm + LM head
        hidden = rms_norm(&hidden, &self.final_norm, hd, 1e-6);
        let mut logits = matmul(&hidden, &self.lm_head, seq, hd, vs);

        // 5. Final logit softcapping
        if let Some(cap) = self.final_logit_softcapping {
            for l in logits.iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }

        logits
    }

    /// Forward pass returning per-layer hidden states for distillation.
    /// Returns Vec where [0] = post-embedding, [1] = post-layer0, etc.
    pub fn forward_with_hidden_states(&self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        let seq = token_ids.len();
        let hd = self.hidden_dim;
        let mut states = Vec::new();

        // Embedding
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.embed_tokens.len() {
                for d in 0..hd {
                    hidden[t * hd + d] = self.embed_tokens[id * hd + d];
                }
            }
        }
        let scale = (hd as f32).sqrt();
        for h in hidden.iter_mut() { *h *= scale; }
        states.push(hidden.clone());

        // PLE pre-compute
        let ple_dim = self.hidden_size_per_layer_input;
        let ple_precomputed = self.precompute_ple(&hidden, token_ids, seq, hd, ple_dim);

        // Layers
        let first_shared_layer = self.num_layers.saturating_sub(self.num_kv_shared_layers);
        let mut shared_kv: std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)> = std::collections::HashMap::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let ple_slice = ple_precomputed.as_ref().map(|pre| {
                let mut slice = vec![0.0f32; seq * ple_dim];
                for t in 0..seq {
                    let src = t * self.num_layers * ple_dim + layer_idx * ple_dim;
                    let dst = t * ple_dim;
                    for d in 0..ple_dim {
                        slice[dst + d] = pre[src + d];
                    }
                }
                slice
            });

            let (kv, should_store) = if layer_idx >= first_shared_layer && first_shared_layer > 0 && layer.kv_shared {
                let kv_source = Self::kv_shared_source_layer(layer_idx, first_shared_layer, &self.layers);
                if let Some((sk, sv)) = shared_kv.get(&kv_source) {
                    (Some((sk.as_slice(), sv.as_slice())), false)
                } else {
                    (None, false)
                }
            } else {
                (None, true)
            };

            let (k_states, v_states) = layer.forward_full(
                &mut hidden,
                ple_slice.as_ref().map(|s| s.as_slice()),
                kv,
            );

            if should_store && first_shared_layer > 0 && !k_states.is_empty() {
                shared_kv.insert(layer_idx, (k_states, v_states));
            }

            states.push(hidden.clone());
        }

        states
    }

    /// Pre-compute PLE inputs from initial hidden state (matches llama.cpp).
    fn precompute_ple(
        &self,
        hidden: &[f32],
        token_ids: &[u32],
        seq: usize,
        hd: usize,
        ple_dim: usize,
    ) -> Option<Vec<f32>> {
        if ple_dim == 0
            || self.ple_model_projection.is_none()
            || self.ple_projection_norm.is_none()
        {
            return None;
        }

        let model_proj = self.ple_model_projection.as_ref().unwrap();
        let proj_norm = self.ple_projection_norm.as_ref().unwrap();
        let ple_total = ple_dim * self.num_layers;

        // 1. Context projection scaled by 1/sqrt(hd)
        let model_proj_scale = (hd as f32).sqrt().recip();
        let ple_all = matmul(hidden, model_proj, seq, hd, ple_total);

        // Reshape and scale
        let mut ple_projected = vec![0.0f32; seq * self.num_layers * ple_dim];
        for t in 0..seq {
            for l in 0..self.num_layers {
                for d in 0..ple_dim {
                    let src = t * ple_total + l * ple_dim + d;
                    let dst = t * self.num_layers * ple_dim + l * ple_dim + d;
                    ple_projected[dst] = ple_all[src] * model_proj_scale;
                }
            }
        }

        // 2. Per-layer RMS norm
        let mut ple_normed = vec![0.0f32; ple_projected.len()];
        for t in 0..seq {
            for l in 0..self.num_layers {
                let base = t * self.num_layers * ple_dim + l * ple_dim;
                let slice = &ple_projected[base..base + ple_dim];
                let normed = rms_norm(slice, proj_norm, ple_dim, 1e-6);
                ple_normed[base..base + ple_dim].copy_from_slice(&normed);
            }
        }

        // 3. Add token embeddings scaled by sqrt(ple_dim)
        let tok_scale = (ple_dim as f32).sqrt();
        if let Some(ref token_embs) = self.embed_tokens_per_layer {
            for t in 0..seq {
                let tok_id = token_ids[t] as usize;
                for l in 0..self.num_layers {
                    let dst = t * self.num_layers * ple_dim + l * ple_dim;
                    let src = tok_id * ple_total + l * ple_dim;
                    for d in 0..ple_dim {
                        ple_normed[dst + d] += token_embs[src + d] * tok_scale;
                    }
                }
            }
        }

        // 4. 1/sqrt(2) scale
        let combined_scale = (2.0f32).sqrt().recip();
        for v in ple_normed.iter_mut() {
            *v *= combined_scale;
        }

        Some(ple_normed)
    }

    /// Find KV sharing source: last non-shared layer of same type.
    fn kv_shared_source_layer(
        layer_idx: usize,
        first_shared: usize,
        layers: &[CpuBlockAttnResLayer],
    ) -> usize {
        let my_rope = layers.get(layer_idx).map(|l| l.rope_theta).unwrap_or(10000.0);
        layers[..first_shared].iter().enumerate().rev()
            .find(|(_, l)| (l.rope_theta - my_rope).abs() < 1.0)
            .map(|(i, _)| i)
            .unwrap_or(first_shared.saturating_sub(1))
    }
}
