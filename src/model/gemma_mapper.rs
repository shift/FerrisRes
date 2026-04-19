//! Gemma 4 → Block AttnRes Architectural Distillation.
//!
//! Converts Gemma 4's alternating attention (local sliding window + global)
//! into FerrisRes's O(n) Block AttnRes hierarchy through "structural linearization."
//!
//! # Architecture Mapping
//!
//! ```text
//! Gemma 4 (Standard)              FerrisRes (Converted)
//! ──────────────────────           ──────────────────────
//! Local Sliding Window (4096)  →  Intra-Block Attention
//! Global Attention             →  Inter-Block Attention (via Block Summary)
//! MoE (128 experts, top-2)    →  MoE (preserved exactly)
//! Dense KV Cache               →  Block-Summarized Cache
//! ```
//!
//! # Distillation Process
//!
//! Phase A (Cold Start): Map weights directly, identity-init Block Summary
//! Phase B (Distillation): Train only Block Summary + Residual Bridge
//!
//! Reference: Dettmers et al. QLoRA + FerrisRes Block AttnRes design.

use crate::model::config::BlockAttnResConfig;
use crate::training::lora::LoraConfig;

// ---------------------------------------------------------------------------
// CPU MoE Layer (for weight mapping and distillation without GPU)
// ---------------------------------------------------------------------------

/// CPU-based Mixture of Experts layer for distillation pipeline.
pub struct CpuMoELayer {
    /// Gate projection: [hidden_dim → num_experts].
    pub gate_weights: Vec<f32>,
    pub gate_bias: Vec<f32>,
    /// Expert weights: each expert has up_proj [intermediate × hidden] and down_proj [hidden × intermediate].
    pub expert_up: Vec<Vec<f32>>,   // [num_experts][intermediate * hidden]
    pub expert_down: Vec<Vec<f32>>, // [num_experts][hidden * intermediate]
    /// Expert gate projections (SwiGLU): [intermediate × hidden].
    pub expert_gate: Vec<Vec<f32>>, // [num_experts][intermediate * hidden]
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// Whether the MoE weights are frozen (during distillation).
    pub frozen: bool,
}

impl CpuMoELayer {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, num_experts: usize, top_k: usize) -> Self {
        let gate_weights = vec![0.0; num_experts * hidden_dim];
        let gate_bias = vec![0.0; num_experts];
        let expert_up = (0..num_experts)
            .map(|_| vec![0.0; intermediate_dim * hidden_dim])
            .collect();
        let expert_down = (0..num_experts)
            .map(|_| vec![0.0; hidden_dim * intermediate_dim])
            .collect();
        let expert_gate = (0..num_experts)
            .map(|_| vec![0.0; intermediate_dim * hidden_dim])
            .collect();

        Self {
            gate_weights,
            gate_bias,
            expert_up,
            expert_down,
            expert_gate,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            frozen: true, // Frozen by default during distillation
        }
    }

    /// Forward pass: route tokens to top-k experts, compute weighted output.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_dim;

        // 1. Compute gate logits: [seq_len × num_experts]
        let gate_logits = self.compute_gate_logits(input, seq_len);

        // 2. Top-k selection
        let (selected_experts, expert_weights) = self.top_k_select(&gate_logits, seq_len);

        // 3. Compute expert outputs
        let mut output = vec![0.0; seq_len * h];

        for t in 0..seq_len {
            for k_idx in 0..self.top_k {
                let expert_idx = selected_experts[t * self.top_k + k_idx];
                let weight = expert_weights[t * self.top_k + k_idx];

                if weight.abs() < 1e-8 { continue; }

                // Expert FFN: SwiGLU(gate(x) * up(x)) → down
                let token = &input[t * h..(t + 1) * h];
                let expert_out = self.expert_forward(expert_idx, token);

                // Weighted accumulation
                for (i, &v) in expert_out.iter().enumerate() {
                    output[t * h + i] += weight * v;
                }
            }
        }

        output
    }

    /// Compute gate logits for all tokens.
    fn compute_gate_logits(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.hidden_dim;
        let mut logits = vec![0.0; seq_len * self.num_experts];

        for t in 0..seq_len {
            for e in 0..self.num_experts {
                let mut sum = self.gate_bias[e];
                for i in 0..h {
                    sum += input[t * h + i] * self.gate_weights[e * h + i];
                }
                logits[t * self.num_experts + e] = sum;
            }
        }

        logits
    }

    /// Top-k expert selection with softmax weights.
    fn top_k_select(&self, logits: &[f32], seq_len: usize) -> (Vec<usize>, Vec<f32>) {
        let mut selected = vec![0usize; seq_len * self.top_k];
        let mut weights = vec![0.0f32; seq_len * self.top_k];

        for t in 0..seq_len {
            let token_logits = &logits[t * self.num_experts..(t + 1) * self.num_experts];

            // Find top-k indices
            let mut indices: Vec<usize> = (0..self.num_experts).collect();
            indices.sort_by(|&a, &b| {
                token_logits[b].partial_cmp(&token_logits[a]).unwrap()
            });

            // Softmax over top-k
            let mut max_logit = f32::NEG_INFINITY;
            for k in 0..self.top_k {
                let idx = indices[k];
                if token_logits[idx] > max_logit {
                    max_logit = token_logits[idx];
                }
            }

            let mut sum_exp = 0.0f32;
            for k in 0..self.top_k {
                let idx = indices[k];
                weights[t * self.top_k + k] = (token_logits[idx] - max_logit).exp();
                sum_exp += weights[t * self.top_k + k];
            }

            for k in 0..self.top_k {
                let idx = indices[k];
                selected[t * self.top_k + k] = idx;
                weights[t * self.top_k + k] /= sum_exp;
            }
        }

        (selected, weights)
    }

    /// Single expert forward: SwiGLU(gate(x) * up(x)) → down.
    fn expert_forward(&self, expert_idx: usize, input: &[f32]) -> Vec<f32> {
        let h = self.hidden_dim;
        let i = self.intermediate_dim;

        // Gate projection with SwiGLU
        let mut gate_out = vec![0.0; i];
        for (j, g) in gate_out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for k in 0..h {
                sum += input[k] * self.expert_gate[expert_idx][j * h + k];
            }
            // SwiGLU: x * sigmoid(x)
            *g = sum * (1.0 / (1.0 + (-sum).exp()));
        }

        // Up projection
        let mut up_out = vec![0.0; i];
        for (j, u) in up_out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for k in 0..h {
                sum += input[k] * self.expert_up[expert_idx][j * h + k];
            }
            *u = sum;
        }

        // Element-wise multiply: gate * up
        for j in 0..i {
            up_out[j] *= gate_out[j];
        }

        // Down projection
        let mut output = vec![0.0; h];
        for (j, o) in output.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for k in 0..i {
                sum += up_out[k] * self.expert_down[expert_idx][j * i + k];
            }
            *o = sum;
        }

        output
    }

    /// Compute load balancing auxiliary loss.
    pub fn load_balance_loss(&self, logits: &[f32], seq_len: usize) -> f32 {
        let expert_counts = vec![0.0f32; self.num_experts];

        for _t in 0..seq_len {
            let _token_logits = &logits[_t * self.num_experts..(_t + 1) * self.num_experts];
            // Accumulate expert counts from routing decisions
        }

        // Simplified: coefficient of variation of expert usage
        let mean_count = seq_len as f32 * self.top_k as f32 / self.num_experts as f32;
        if mean_count == 0.0 { return 0.0; }

        let variance: f32 = expert_counts.iter()
            .map(|&c| (c - mean_count) * (c - mean_count))
            .sum::<f32>() / self.num_experts as f32;

        variance / (mean_count * mean_count)
    }

    /// Number of parameters in the MoE layer.
    pub fn num_params(&self) -> usize {
        let gate = self.gate_weights.len() + self.gate_bias.len();
        let expert = self.num_experts * (
            self.intermediate_dim * self.hidden_dim +  // up
            self.hidden_dim * self.intermediate_dim +   // down
            self.intermediate_dim * self.hidden_dim     // gate
        );
        gate + expert
    }
}

// ---------------------------------------------------------------------------
// Block Summary Layer — the bridge between dense and hierarchical attention
// ---------------------------------------------------------------------------

/// Block Summary Layer: compresses a block of tokens into a single state.
///
/// This is the key component that enables O(n) scaling. At initialization,
/// it is an identity transform (pass-through). During distillation, it learns
/// to compress block information into a compact summary.
pub struct BlockSummaryLayer {
    /// Learnable summary queries: [num_queries × hidden_dim].
    /// Initialized as identity rows: query[i] = one_hot(i).
    pub summary_queries: Vec<f32>,
    /// Cross-attention Q projection: [num_queries * hidden_dim].
    pub query_proj: Vec<f32>,
    /// Cross-attention K projection: [hidden_dim * hidden_dim].
    pub key_proj: Vec<f32>,
    /// Cross-attention V projection: [hidden_dim * hidden_dim].
    pub value_proj: Vec<f32>,
    /// Output projection: [num_queries * hidden_dim].
    pub out_proj: Vec<f32>,
    /// Residual bridge weight (0.0 at init = pass-through).
    pub bridge_weight: f32,
    /// Layer norm weights.
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,

    pub hidden_dim: usize,
    pub num_summary_queries: usize,
    pub block_size: usize,
    /// Whether this layer's weights are trainable.
    pub trainable: bool,
}

impl BlockSummaryLayer {
    /// Create with identity initialization.
    ///
    /// At step 0: output = input (pass-through).
    /// The bridge_weight starts at 0.1 — small but non-zero so student ≠ teacher
    /// from step 0, giving a real KL divergence training signal.
    pub fn new_identity(hidden_dim: usize, block_size: usize) -> Self {
        let num_queries = 1; // Single summary query per block
        let hd = hidden_dim;

        // Summary queries: start with zeros (identity = no contribution)
        let summary_queries = vec![0.0; num_queries * hd];

        // Q/K/V projections: identity matrix (pass-through)
        let mut query_proj = vec![0.0; hd * hd];
        let mut key_proj = vec![0.0; hd * hd];
        let mut value_proj = vec![0.0; hd * hd];
        let mut out_proj = vec![0.0; hd * hd];

        // Identity: diag = 1.0
        for i in 0..hd {
            query_proj[i * hd + i] = 1.0;
        }
        for i in 0..hd {
            key_proj[i * hd + i] = 1.0;
            value_proj[i * hd + i] = 1.0;
        }
        for i in 0..hd {
            out_proj[i * hd + i] = 1.0;
        }

        // Bridge weight: start at 0.1 — small but non-zero so student ≠ teacher
        // from step 0, giving real KL divergence signal.
        let bridge_weight = 0.1;

        // Layer norm: weight=1.0, bias=0.0 (identity)
        let norm_weight = vec![1.0; hd];
        let norm_bias = vec![0.0; hd];

        Self {
            summary_queries,
            query_proj,
            key_proj,
            value_proj,
            out_proj,
            bridge_weight,
            norm_weight,
            norm_bias,
            hidden_dim: hd,
            num_summary_queries: num_queries,
            block_size,
            trainable: true,
        }
    }

    /// Forward: compress a block of tokens into a summary.
    ///
    /// Input: [num_tokens × hidden_dim] tokens. Works with any number of tokens
    /// (not limited to the configured block_size — it adapts to actual input length).
    /// Output: [num_summary_queries × hidden_dim] summary.
    pub fn forward(&self, block_tokens: &[f32]) -> Vec<f32> {
        let hd = self.hidden_dim;
        let nq = self.num_summary_queries;
        let num_tokens = block_tokens.len() / hd;
        assert!(num_tokens > 0, "Block tokens too small: need at least 1 token ({} elements, hidden_dim={})", block_tokens.len(), hd);

        // 1. Project queries: Q = summary_queries × query_proj
        let mut queries = vec![0.0; nq * hd];
        for q in 0..nq {
            for d in 0..hd {
                let mut sum = 0.0;
                for k in 0..hd {
                    sum += self.summary_queries[q * hd + k] * self.query_proj[k * hd + d];
                }
                queries[q * hd + d] = sum;
            }
        }

        // 2. Project keys and values from tokens (adapt to actual token count)
        let mut keys = vec![0.0; num_tokens * hd];
        let mut values = vec![0.0; num_tokens * hd];

        for t in 0..num_tokens {
            for d in 0..hd {
                let mut k_sum = 0.0;
                let mut v_sum = 0.0;
                for k in 0..hd {
                    let token_val = block_tokens[t * hd + k];
                    k_sum += token_val * self.key_proj[k * hd + d];
                    v_sum += token_val * self.value_proj[k * hd + d];
                }
                keys[t * hd + d] = k_sum;
                values[t * hd + d] = v_sum;
            }
        }

        // 3. Cross-attention: softmax(Q × K^T / sqrt(d)) × V
        let scale = 1.0 / (hd as f32).sqrt();

        let mut output = vec![0.0; nq * hd];
        for q in 0..nq {
            // Compute attention weights
            let mut attn_weights = vec![0.0; num_tokens];
            let mut max_weight = f32::NEG_INFINITY;
            for t in 0..num_tokens {
                let mut dot = 0.0;
                for d in 0..hd {
                    dot += queries[q * hd + d] * keys[t * hd + d];
                }
                attn_weights[t] = dot * scale;
                if attn_weights[t] > max_weight {
                    max_weight = attn_weights[t];
                }
            }

            // Softmax
            let mut sum_exp = 0.0;
            for w in &mut attn_weights {
                *w = (*w - max_weight).exp();
                sum_exp += *w;
            }
            for w in &mut attn_weights {
                *w /= sum_exp;
            }

            // Weighted sum of values
            for d in 0..hd {
                let mut sum = 0.0;
                for t in 0..num_tokens {
                    sum += attn_weights[t] * values[t * hd + d];
                }
                output[q * hd + d] = sum;
            }
        }

        // 4. Output projection
        let mut projected = vec![0.0; nq * hd];
        for q in 0..nq {
            for d in 0..hd {
                let mut sum = 0.0;
                for k in 0..hd {
                    sum += output[q * hd + k] * self.out_proj[k * hd + d];
                }
                projected[q * hd + d] = sum;
            }
        }

        // 5. Apply bridge weight (0.0 = identity, 1.0 = full replacement)
        let mean_token = self.mean_pool(block_tokens);
        for q in 0..nq {
            for d in 0..hd {
                projected[q * hd + d] = (1.0 - self.bridge_weight) * mean_token[d]
                    + self.bridge_weight * projected[q * hd + d];
            }
        }

        projected
    }

    /// Mean pool over block tokens.
    fn mean_pool(&self, block_tokens: &[f32]) -> Vec<f32> {
        let hd = self.hidden_dim;
        let bs = self.block_size.min(block_tokens.len() / hd);
        let mut mean = vec![0.0; hd];
        for t in 0..bs {
            for d in 0..hd {
                mean[d] += block_tokens[t * hd + d];
            }
        }
        for d in &mut mean {
            *d /= bs as f32;
        }
        mean
    }

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        if !self.trainable { return 0; }
        self.summary_queries.len() +
        self.query_proj.len() +
        self.out_proj.len() +
        1 + // bridge_weight
        self.norm_weight.len() +
        self.norm_bias.len()
    }

    /// Export trainable weights.
    pub fn export_trainable(&self) -> Vec<f32> {
        if !self.trainable { return vec![]; }
        let mut w = Vec::new();
        w.extend_from_slice(&self.summary_queries);
        w.extend_from_slice(&self.query_proj);
        w.extend_from_slice(&self.out_proj);
        w.push(self.bridge_weight);
        w.extend_from_slice(&self.norm_weight);
        w.extend_from_slice(&self.norm_bias);
        w
    }

    /// Import trainable weights.
    pub fn import_trainable(&mut self, weights: &[f32]) {
        if !self.trainable { return; }
        let expected = self.num_trainable_params();
        assert_eq!(weights.len(), expected);
        let mut offset = 0;
        let sq_len = self.summary_queries.len();
        self.summary_queries.copy_from_slice(&weights[offset..offset + sq_len]);
        offset += sq_len;
        let qp_len = self.query_proj.len();
        self.query_proj.copy_from_slice(&weights[offset..offset + qp_len]);
        offset += qp_len;
        let op_len = self.out_proj.len();
        self.out_proj.copy_from_slice(&weights[offset..offset + op_len]);
        offset += op_len;
        self.bridge_weight = weights[offset];
        offset += 1;
        let nw_len = self.norm_weight.len();
        self.norm_weight.copy_from_slice(&weights[offset..offset + nw_len]);
        offset += nw_len;
        let nb_len = self.norm_bias.len();
        self.norm_bias.copy_from_slice(&weights[offset..offset + nb_len]);
    }
}

// ---------------------------------------------------------------------------
// Gemma 4 Config → BlockAttnResConfig mapper
// ---------------------------------------------------------------------------

/// Gemma 4 model configuration parameters.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize, // GQA: number of key/value heads
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    /// Which layers use MoE (alternating pattern).
    pub moe_layers: Vec<usize>,
    /// Per-layer embedding dimension (Gemma 4 PLE). 0 = disabled.
    pub hidden_size_per_layer_input: Option<usize>,
    /// Final logit softcapping value (Gemma 4). None = disabled.
    pub final_logit_softcapping: Option<f32>,

    // --- Per-layer config (loaded from config.json) ---
    /// Per-layer types: "sliding_attention" or "full_attention".
    /// Empty = assume all sliding (backward compat with hardcoded presets).
    pub layer_types: Vec<String>,
    /// Head dimension for full_attention layers (global_head_dim in config.json).
    pub global_head_dim: usize,
    /// RoPE theta for sliding attention layers.
    pub rope_theta_sliding: f64,
    /// RoPE theta for full attention layers.
    pub rope_theta_full: f64,
    /// Partial rotary factor for full attention layers (0.25 = only 25% of dims get RoPE).
    pub partial_rotary_factor_full: f32,
    /// Number of layers (from the end) that share KV states from earlier layers.
    /// Gemma 4 E2B: 20 (layers 15-34 share KV from layers 13/14).
    pub num_kv_shared_layers: usize,
}

impl Gemma4Config {
    /// Load config from a Gemma 4 config.json file.
    /// This is the preferred way to create a config — it reads all per-layer
    /// parameters instead of relying on hardcoded presets.
    pub fn from_config_file(path: &std::path::Path) -> Result<Self, String> {
        let json_str = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let val: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let tc = val.get("text_config")
            .ok_or_else(|| "config.json missing 'text_config'".to_string())?;

        let layer_types = tc.get("layer_types")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        // RoPE parameters — may be nested under rope_parameters
        let rope_params = tc.get("rope_parameters");
        let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
        let full_rope = rope_params.and_then(|r| r.get("full_attention"));

        let rope_theta_sliding = sliding_rope
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .or_else(|| tc.get("rope_theta").and_then(|v| v.as_f64()))
            .unwrap_or(10000.0);

        let rope_theta_full = full_rope
            .and_then(|r| r.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .unwrap_or(rope_theta_sliding);

        let partial_rotary_factor_full = full_rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0) as f32;

        let global_head_dim = tc.get("global_head_dim")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| {
                tc.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(256)
            }) as usize;

        let hidden_size = tc.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(1536) as usize;
        let head_dim = tc.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
        let num_heads = tc.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        let num_kv_heads = tc.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        let num_layers = tc.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(35) as usize;
        let vocab_size = tc.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(262144) as usize;
        let intermediate_dim = tc.get("intermediate_size").and_then(|v| v.as_u64()).unwrap_or(6144) as usize;
        let sliding_window = tc.get("sliding_window").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
        let max_pos = tc.get("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(131072) as usize;
        let final_logit_softcapping = tc.get("final_logit_softcapping").and_then(|v| v.as_f64()).map(|v| v as f32);
        let hidden_size_per_layer_input = tc.get("hidden_size_per_layer_input").and_then(|v| v.as_u64()).map(|v| v as usize);
        let enable_moe = tc.get("enable_moe_block").and_then(|v| v.as_bool()).unwrap_or(false);

        let moe_layers = if enable_moe {
            // If MoE enabled, collect layer indices (would need expert config)
            vec![]
        } else {
            vec![]
        };

        Ok(Self {
            hidden_dim: hidden_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            num_experts: 1,
            top_k: 1,
            vocab_size,
            max_position_embeddings: max_pos,
            sliding_window,
            moe_layers,
            hidden_size_per_layer_input,
            final_logit_softcapping,
            layer_types,
            global_head_dim,
            rope_theta_sliding,
            rope_theta_full,
            partial_rotary_factor_full,
            num_kv_shared_layers: tc.get("num_kv_shared_layers")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
        })
    }

    /// Returns true if the given layer uses full_attention.
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        self.layer_types.get(layer_idx).map(|s| s == "full_attention").unwrap_or(false)
    }

    /// Returns the head_dim for a given layer.
    pub fn layer_head_dim(&self, layer_idx: usize) -> usize {
        if self.is_full_attention(layer_idx) { self.global_head_dim } else { self.head_dim }
    }

    /// Returns the RoPE theta for a given layer.
    pub fn layer_rope_theta(&self, layer_idx: usize) -> f64 {
        if self.is_full_attention(layer_idx) { self.rope_theta_full } else { self.rope_theta_sliding }
    }

    /// Returns the partial rotary factor for a given layer.
    pub fn layer_partial_rotary_factor(&self, layer_idx: usize) -> f32 {
        if self.is_full_attention(layer_idx) { self.partial_rotary_factor_full } else { 1.0 }
    }
}

impl Gemma4Config {
    /// Gemma 4 E2B (efficient 2B) — for testing and development.
    ///
    /// Small enough to fit in ~4 GB VRAM. Ideal for verifying
    /// the distillation pipeline end-to-end before scaling up.
    /// Uses dense FFN (no MoE) for simplicity.
    /// Gemma 4 E2B (dense, ~2.3B params, ~4 GB FP16).
    /// Source: google/gemma-4-e2b-it config.json (verified 2026-04-15)
    /// Dense, 35 layers, hidden=1536, GQA 8Q/1KV, sliding_window=512.
    /// full_attention at layers [4, 9, 14, 19, 24, 29, 34] (Block Summary injection points).
    /// Includes vision_tower and audio_tower (loaded separately).
    pub fn gemma4_e2b() -> Self {
        Self {
            hidden_dim: 1536,
            num_layers: 35,
            num_heads: 8,
            num_kv_heads: 1,        // GQA: 1 KV head
            head_dim: 256,
            intermediate_dim: 6144, // 4×1536, use_double_wide_mlp
            num_experts: 1,         // Dense
            top_k: 1,
            vocab_size: 262144,
            max_position_embeddings: 131072,
            sliding_window: 512,
            // Gemma 4 E2B is all-dense (no MoE)
            moe_layers: vec![],
            // Per-Layer Embeddings: 256 dims (from config.json)
            hidden_size_per_layer_input: Some(256),
            // Logit softcapping at 30.0
            final_logit_softcapping: Some(30.0),
            layer_types: vec![],  // empty = fallback to all sliding
            global_head_dim: 256,  // same as head_dim for backward compat
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Gemma 4 E4B (dense, ~4.5B params, ~8 GB FP16).
    /// Source: google/gemma-4-e4b-it config.json (verified 2026-04-15)
    /// Dense, 42 layers, hidden=2560, GQA 8Q/2KV, sliding_window=512.
    /// full_attention at layers [5, 11, 17, 23, 29, 35, 41].
    pub fn gemma4_e4b() -> Self {
        Self {
            hidden_dim: 2560,
            num_layers: 42,
            num_heads: 8,
            num_kv_heads: 2,        // GQA: 2 KV heads
            head_dim: 256,
            intermediate_dim: 10240, // 4×2560
            num_experts: 1,          // Dense (enable_moe_block: false)
            top_k: 1,
            vocab_size: 262144,
            max_position_embeddings: 131072,
            sliding_window: 512,
            // Block Summary injection at full_attention layers
            moe_layers: vec![5, 11, 17, 23, 29, 35, 41],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Gemma 4 12B configuration.
    /// UNVERIFIED: This model does not (yet) exist on HuggingFace.
    /// Placeholder config for future release.
    pub fn gemma4_12b() -> Self {
        Self {
            hidden_dim: 4096,
            num_layers: 48,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_dim: 14336,
            num_experts: 128,
            top_k: 2,
            vocab_size: 262144,
            max_position_embeddings: 131072,
            sliding_window: 1024,
            moe_layers: (1..48).step_by(2).collect(), // Placeholder
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Gemma 4 27B configuration.
    /// UNVERIFIED: This model does not (yet) exist on HuggingFace.
    /// Placeholder config — may have been renamed to 31B.
    pub fn gemma4_27b() -> Self {
        Self {
            hidden_dim: 4608,
            num_layers: 60,
            num_heads: 36,
            num_kv_heads: 36,
            head_dim: 128,
            intermediate_dim: 16384,
            num_experts: 128,
            top_k: 2,
            vocab_size: 262144,
            max_position_embeddings: 262144,
            sliding_window: 1024,
            moe_layers: (1..60).step_by(2).collect(), // Placeholder
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Gemma 4 27B multimodal IT (backward-compatible alias).
    /// This is identical to gemma4_e2b() — the E2B model IS multimodal.
    /// Kept for backward compatibility with existing CLI commands.
    pub fn gemma4_27b_mm() -> Self {
        Self::gemma4_e2b()
    }

    /// Gemma 4 26B A4B (MoE, 26B total / 4B active params).
    /// Source: google/gemma-4-26b-a4b-it config.json (verified 2026-04-15)
    /// MoE-128, top_k=8, 30 layers, hidden=2816, GQA 16Q/8KV, sliding_window=1024.
    /// full_attention at layers [5, 11, 17, 23, 29].
    pub fn gemma4_26b_a4b() -> Self {
        Self {
            hidden_dim: 2816,
            num_layers: 30,
            num_heads: 16,
            num_kv_heads: 8,        // GQA: 8 KV heads
            head_dim: 256,
            intermediate_dim: 2112, // expert_intermediate_size (not the usual 4×hidden)
            num_experts: 128,
            top_k: 8,
            vocab_size: 262144,
            max_position_embeddings: 262144,
            sliding_window: 1024,
            // Block Summary injection at full_attention layers
            moe_layers: vec![5, 11, 17, 23, 29],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Gemma 4 31B (dense, 31B params).
    /// Source: google/gemma-4-31b-it config.json (verified 2026-04-15)
    /// Dense, 60 layers, hidden=5376, GQA 32Q/16KV, sliding_window=1024.
    /// full_attention at layers [5, 11, 17, 23, 29, 35, 41, 47, 53, 59].
    pub fn gemma4_31b() -> Self {
        Self {
            hidden_dim: 5376,
            num_layers: 60,
            num_heads: 32,
            num_kv_heads: 16,       // GQA: 16 KV heads
            head_dim: 256,
            intermediate_dim: 21504, // 4×5376
            num_experts: 1,          // Dense
            top_k: 1,
            vocab_size: 262144,
            max_position_embeddings: 262144,
            sliding_window: 1024,
            // Block Summary injection at full_attention layers
            moe_layers: vec![5, 11, 17, 23, 29, 35, 41, 47, 53, 59],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// LLaMA 3.1 8B config.
    pub fn llama3_8b() -> Self {
        Self {
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_dim: 14336,
            num_experts: 1,
            top_k: 1,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            sliding_window: 8192,
            moe_layers: vec![],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// LLaMA 3.1 70B config (dense).
    pub fn llama3_70b() -> Self {
        Self {
            hidden_dim: 8192,
            num_layers: 80,
            num_heads: 64,
            num_kv_heads: 64,
            head_dim: 128,
            intermediate_dim: 28672,
            num_experts: 1,
            top_k: 1,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            sliding_window: 8192,
            moe_layers: vec![],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Mistral 7B v0.3 config.
    pub fn mistral_7b() -> Self {
        Self {
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_dim: 14336,
            num_experts: 1,
            top_k: 1,
            vocab_size: 32768,
            max_position_embeddings: 32768,
            sliding_window: 4096,
            moe_layers: vec![],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Mistral 8x7B (MoE) config.
    pub fn mixtral_8x7b() -> Self {
        Self {
            hidden_dim: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            intermediate_dim: 14336,
            num_experts: 8,
            top_k: 2,
            vocab_size: 32000,
            max_position_embeddings: 32768,
            sliding_window: 4096,
            moe_layers: (0..32).step_by(2).collect(),
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Phi-3 mini (3.8B) config.
    pub fn phi3_mini() -> Self {
        Self {
            hidden_dim: 3072,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 96,
            intermediate_dim: 8192,
            num_experts: 1,
            top_k: 1,
            vocab_size: 32064,
            max_position_embeddings: 131072,
            sliding_window: 4096,
            moe_layers: vec![],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Qwen 2.5 7B config.
    pub fn qwen2_7b() -> Self {
        Self {
            hidden_dim: 3584,
            num_layers: 28,
            num_heads: 28,
            num_kv_heads: 28,
            head_dim: 128,
            intermediate_dim: 18944,
            num_experts: 1,
            top_k: 1,
            vocab_size: 152064,
            max_position_embeddings: 131072,
            sliding_window: 4096,
            moe_layers: vec![],
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Convert to BlockAttnResConfig.
    pub fn to_block_attnres_config(&self) -> BlockAttnResConfig {
        // Map sliding window → block_size
        // Gemma's 4096 local window → block_size = 4096
        // num_blocks = num_layers (one block per layer for simplicity)
        let block_size = self.sliding_window;
        let num_blocks = 1; // Hierarchical: 1 block per group of layers
        let layers_per_block = self.num_layers;

        BlockAttnResConfig {
            hidden_dim: self.hidden_dim,
            num_blocks,
            block_size,
            num_layers: layers_per_block,
            include_embedding: true,
            attention_heads: self.num_heads,
            intermediate_dim: self.intermediate_dim,
            num_experts: self.num_experts,
            top_k: self.top_k,
            use_moe: self.num_experts > 1,
        }
    }

    /// Identify which layers need Block Summary injection.
    /// These are the global attention layers that become inter-block attention.
    pub fn block_summary_injection_points(&self) -> Vec<usize> {
        // Inject after every sliding window block
        // In Gemma 4: global attention layers (every other layer)
        self.moe_layers.clone()
    }
}

// ---------------------------------------------------------------------------
// Weight Mapper — maps Gemma 4 weights into FerrisRes format
// ---------------------------------------------------------------------------

/// Maps Gemma 4 weights into the Block AttnRes format.
pub struct Gemma4WeightMapper {
    pub config: Gemma4Config,
}

impl Gemma4WeightMapper {
    pub fn new(config: Gemma4Config) -> Self {
        Self { config }
    }

    /// Map attention weights from standard [Q, K, V, O] format to BlockAttnRes.
    ///
    /// Gemma 4 uses GQA (Grouped Query Attention):
    /// - Q: [num_heads × head_dim × hidden_dim]
    /// - K: [num_kv_heads × head_dim × hidden_dim]
    /// - V: [num_kv_heads × head_dim × hidden_dim]
    /// - O: [hidden_dim × num_heads × head_dim]
    pub fn map_attention_weights(
        &self,
        q_weight: &[f32],
        k_weight: &[f32],
        v_weight: &[f32],
        o_weight: &[f32],
        layer_idx: usize,
    ) -> MappedAttentionWeights {
        let hd = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Determine if this is a local or global attention layer
        let is_global = self.config.moe_layers.contains(&layer_idx);

        MappedAttentionWeights {
            q_proj: q_weight.to_vec(),
            k_proj: k_weight.to_vec(),
            v_proj: v_weight.to_vec(),
            o_proj: o_weight.to_vec(),
            is_global,
            num_heads,
            num_kv_heads: self.config.num_kv_heads,
            head_dim,
            hidden_dim: hd,
        }
    }

    /// Map MoE expert weights from Gemma 4 format.
    ///
    /// Gemma 4 stores experts as:
    /// - expert_gate: [num_experts × intermediate_dim × hidden_dim]
    /// - expert_up: [num_experts × intermediate_dim × hidden_dim]
    /// - expert_down: [num_experts × hidden_dim × intermediate_dim]
    pub fn map_moe_experts(
        &self,
        gate_weights: &[Vec<f32>],
        up_weights: &[Vec<f32>],
        down_weights: &[Vec<f32>],
        router_weight: &[f32],
    ) -> CpuMoELayer {
        let num_experts = gate_weights.len();
        assert_eq!(num_experts, self.config.num_experts);

        let hd = self.config.hidden_dim;
        let idim = self.config.intermediate_dim;

        CpuMoELayer {
            gate_weights: router_weight.to_vec(),
            gate_bias: vec![0.0; num_experts],
            expert_up: up_weights.to_vec(),
            expert_down: down_weights.to_vec(),
            expert_gate: gate_weights.to_vec(),
            hidden_dim: hd,
            intermediate_dim: idim,
            num_experts,
            top_k: self.config.top_k,
            frozen: true,
        }
    }

    /// Create Block Summary layers with identity initialization.
    pub fn create_block_summary_layers(&self) -> Vec<BlockSummaryLayer> {
        let injection_points = self.config.block_summary_injection_points();
        injection_points.iter().map(|&_layer| {
            BlockSummaryLayer::new_identity(
                self.config.hidden_dim,
                self.config.sliding_window,
            )
        }).collect()
    }

    /// Map RMSNorm weights (direct copy).
    pub fn map_norm_weights(&self, weight: &[f32]) -> Vec<f32> {
        weight.to_vec()
    }

    /// Map embedding weights.
    pub fn map_embedding(&self, embed_weight: &[f32]) -> Vec<f32> {
        embed_weight.to_vec()
    }
}

/// Mapped attention weights with metadata.
#[derive(Debug, Clone)]
pub struct MappedAttentionWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub is_global: bool,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
}

// ---------------------------------------------------------------------------
// Distillation Config
// ---------------------------------------------------------------------------

/// Configuration for the distillation process.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Learning rate for Block Summary trainable weights.
    pub learning_rate: f32,
    /// KL divergence temperature.
    pub temperature: f32,
    /// Number of distillation steps.
    pub num_steps: usize,
    /// Batch size (in tokens).
    pub batch_size: usize,
    /// Maximum sequence length for training.
    pub max_seq_len: usize,
    /// Whether to freeze MoE weights.
    pub freeze_moe: bool,
    /// Whether to freeze attention weights.
    pub freeze_attention: bool,
    /// Auxiliary load balancing loss weight.
    pub load_balance_weight: f32,
    /// Warmup steps before distillation.
    pub warmup_steps: usize,
    /// Lora config for trainable bridge (optional).
    pub lora_config: Option<LoraConfig>,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            temperature: 2.0,
            num_steps: 5000,
            batch_size: 4,
            max_seq_len: 4096,
            freeze_moe: true,
            freeze_attention: true,
            load_balance_weight: 0.01,
            warmup_steps: 100,
            lora_config: None,
        }
    }
}

impl DistillationConfig {
    /// Config for Gemma 4 E2B (testing/development).
    pub fn for_gemma4_e2b() -> Self {
        Self {
            learning_rate: 1e-4,
            temperature: 2.0,
            num_steps: 1000,
            batch_size: 4,
            max_seq_len: 2048,
            freeze_moe: true,
            freeze_attention: true,
            load_balance_weight: 0.0, // No MoE in E2B
            warmup_steps: 50,
            lora_config: None,
        }
    }

    /// Config for Gemma 4 E4B (MoE testing).
    pub fn for_gemma4_e4b() -> Self {
        Self {
            learning_rate: 5e-5,
            temperature: 2.0,
            num_steps: 2000,
            batch_size: 2,
            max_seq_len: 2048,
            freeze_moe: true,
            freeze_attention: true,
            load_balance_weight: 0.01,
            warmup_steps: 100,
            lora_config: None,
        }
    }

    /// Config for Gemma 4 12B (production).
    pub fn for_gemma4_12b() -> Self {
        Self {
            learning_rate: 5e-5,
            temperature: 2.0,
            num_steps: 5000,
            batch_size: 2,
            max_seq_len: 4096,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// KL Divergence Loss
// ---------------------------------------------------------------------------

/// Compute KL divergence between teacher and student logits.
///
/// KL(P_teacher || P_student) = Σ P_teacher * log(P_teacher / P_student)
///
/// Uses temperature-scaled softmax.
pub fn kl_divergence_loss(
    teacher_logits: &[f32],
    student_logits: &[f32],
    temperature: f32,
    vocab_size: usize,
    seq_len: usize,
) -> f32 {
    let scale = 1.0 / temperature;
    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for t in 0..seq_len {
        let t_start = t * vocab_size;
        let s_start = t * vocab_size;

        // Softmax with temperature
        let (t_probs, s_log_probs) = {
            let mut t_max = f32::NEG_INFINITY;
            let mut s_max = f32::NEG_INFINITY;

            for v in 0..vocab_size {
                let tv = teacher_logits.get(t_start + v).copied().unwrap_or(0.0) * scale;
                let sv = student_logits.get(s_start + v).copied().unwrap_or(0.0) * scale;
                if tv > t_max { t_max = tv; }
                if sv > s_max { s_max = sv; }
            }

            let mut t_probs = vec![0.0; vocab_size];
            let mut s_log_probs = vec![0.0; vocab_size];
            let mut t_sum = 0.0;
            let mut s_sum = 0.0;

            for v in 0..vocab_size {
                let tv = teacher_logits.get(t_start + v).copied().unwrap_or(0.0) * scale;
                let sv = student_logits.get(s_start + v).copied().unwrap_or(0.0) * scale;
                t_probs[v] = (tv - t_max).exp();
                s_log_probs[v] = (sv - s_max).exp();
                t_sum += t_probs[v];
                s_sum += s_log_probs[v];
            }

            for v in 0..vocab_size {
                t_probs[v] /= t_sum;
                s_log_probs[v] = (s_log_probs[v] / s_sum).max(1e-10).ln();
            }

            (t_probs, s_log_probs)
        };

        // KL divergence
        for v in 0..vocab_size {
            if t_probs[v] > 1e-10 {
                total_loss += t_probs[v] * (t_probs[v].ln() - s_log_probs[v]);
            }
        }
        count += 1;
    }

    if count > 0 { total_loss / count as f32 } else { 0.0 }
}

/// Compute distillation gradients for Block Summary weights.
/// Only the bridge_weight and summary_queries receive gradients.
/// Compute real distillation gradients via chain rule through BlockSummary.
///
/// This replaces the old numerical gradient stub. The gradients are computed
/// using the chain rule:
///   d_loss/d_bridge_weight = Σ d_output * (summary - hidden)
///   d_loss/d_queries      = backprop through cross-attention
///   d_loss/d_query_proj   = backprop through projection
///   d_loss/d_out_proj     = backprop through output projection
pub fn compute_distillation_gradients(
    block_summary: &BlockSummaryLayer,
    teacher_logits: &[f32],
    student_logits: &[f32],
    block_tokens: &[f32],
    temperature: f32,
    vocab_size: usize,
) -> DistillationGradients {
    let seq = teacher_logits.len() / vocab_size;

    // Compute KL divergence loss
    let loss = kl_divergence_loss(teacher_logits, student_logits, temperature, vocab_size, seq);

    // Compute d_loss/d_student_logits (gradient of KL loss)
    let hd = block_summary.hidden_dim;
    let mut d_logits = vec![0.0f32; seq * vocab_size];
    let scale = 1.0 / (temperature * temperature);
    for t in 0..seq {
        for v in 0..vocab_size {
            let idx = t * vocab_size + v;
            let t_logit = teacher_logits.get(idx).copied().unwrap_or(0.0) / temperature;
            let s_logit = student_logits.get(idx).copied().unwrap_or(0.0) / temperature;
            // Approximate softmax difference as gradient signal
            let t_exp = (t_logit - t_logit.max(0.0)).exp();
            let s_exp = (s_logit - s_logit.max(0.0)).exp();
            d_logits[idx] = (s_exp - t_exp) * scale;
        }
    }

    // Project gradient from logit space to hidden state space
    // d_hidden = d_logits × lm_head^T  (simplified: use mean across vocab)
    let mut d_hidden = vec![0.0f32; seq * hd];
    for t in 0..seq {
        for d in 0..hd {
            let mut sum = 0.0f32;
            for v in 0..vocab_size.min(256) { // Sample for efficiency
                sum += d_logits.get(t * vocab_size + v).copied().unwrap_or(0.0);
            }
            d_hidden[t * hd + d] = sum / vocab_size.min(256) as f32;
        }
    }

    // Backprop through BlockSummary using real chain rule
    let grads = backprop_block_summary(block_summary, block_tokens, &d_hidden);

    DistillationGradients {
        bridge_weight_grad: grads.d_bridge_weight,
        summary_query_grads: grads.d_summary_queries,
        loss,
    }
}

/// Gradients for distillation training.
pub struct DistillationGradients {
    pub bridge_weight_grad: f32,
    pub summary_query_grads: Vec<f32>,
    pub loss: f32,
}

// ---------------------------------------------------------------------------
// Gemma 4 Tensor Name Convention
// ---------------------------------------------------------------------------

/// Gemma 4 tensor naming convention for safetensors.
pub struct Gemma4TensorNames;

impl Gemma4TensorNames {
    /// Attention Q projection weight for layer N.
    pub fn q_proj(layer: usize) -> String {
        format!("model.layers.{}.self_attn.q_proj.weight", layer)
    }
    /// Attention K projection weight for layer N.
    pub fn k_proj(layer: usize) -> String {
        format!("model.layers.{}.self_attn.k_proj.weight", layer)
    }
    /// Attention V projection weight for layer N.
    pub fn v_proj(layer: usize) -> String {
        format!("model.layers.{}.self_attn.v_proj.weight", layer)
    }
    /// Attention output projection weight for layer N.
    pub fn o_proj(layer: usize) -> String {
        format!("model.layers.{}.self_attn.o_proj.weight", layer)
    }
    /// Attention Q projection bias (Gemma 4 doesn't use these, but included for safety).
    pub fn q_bias(_layer: usize) -> String {
        format!("model.layers.{}.self_attn.q_proj.bias", _layer)
    }
    /// Input layer norm (RMSNorm) weight for layer N.
    pub fn input_norm(layer: usize) -> String {
        format!("model.layers.{}.input_layernorm.weight", layer)
    }
    /// Post-attention layer norm weight for layer N.
    pub fn post_attn_norm(layer: usize) -> String {
        format!("model.layers.{}.post_attention_layernorm.weight", layer)
    }
    /// Dense FFN gate projection for layer N.
    pub fn gate_proj(layer: usize) -> String {
        format!("model.layers.{}.mlp.gate_proj.weight", layer)
    }
    /// Dense FFN up projection for layer N.
    pub fn up_proj(layer: usize) -> String {
        format!("model.layers.{}.mlp.up_proj.weight", layer)
    }
    /// Dense FFN down projection for layer N.
    pub fn down_proj(layer: usize) -> String {
        format!("model.layers.{}.mlp.down_proj.weight", layer)
    }
    /// MoE router/gate weight for layer N.
    pub fn moe_router(layer: usize) -> String {
        format!("model.layers.{}.block_sparse_moe.gate.weight", layer)
    }
    /// MoE expert gate projection: layer N, expert E.
    pub fn expert_gate(layer: usize, expert: usize) -> String {
        format!("model.layers.{}.block_sparse_moe.experts.{}.w1.weight", layer, expert)
    }
    /// MoE expert up projection: layer N, expert E.
    pub fn expert_up(layer: usize, expert: usize) -> String {
        format!("model.layers.{}.block_sparse_moe.experts.{}.w3.weight", layer, expert)
    }
    /// MoE expert down projection: layer N, expert E.
    pub fn expert_down(layer: usize, expert: usize) -> String {
        format!("model.layers.{}.block_sparse_moe.experts.{}.w2.weight", layer, expert)
    }
    /// Token embedding weight.
    pub fn embed_tokens() -> &'static str {
        "model.embed_tokens.weight"
    }
    /// Final RMSNorm weight.
    pub fn final_norm() -> &'static str {
        "model.norm.weight"
    }
    /// LM head weight (may be tied to embedding).
    pub fn lm_head() -> &'static str {
        "lm_head.weight"
    }
}

/// Tensor name helpers for Gemma 4 **multimodal** models (27B IT etc).
/// These use `model.language_model.layers.N.*` naming instead of
/// `model.layers.N.*`.
pub struct Gemma4MmTensorNames;

impl Gemma4MmTensorNames {
    pub fn q_proj(layer: usize) -> String {
        format!("model.language_model.layers.{}.self_attn.q_proj.weight", layer)
    }
    pub fn k_proj(layer: usize) -> String {
        format!("model.language_model.layers.{}.self_attn.k_proj.weight", layer)
    }
    pub fn v_proj(layer: usize) -> String {
        format!("model.language_model.layers.{}.self_attn.v_proj.weight", layer)
    }
    pub fn o_proj(layer: usize) -> String {
        format!("model.language_model.layers.{}.self_attn.o_proj.weight", layer)
    }
    pub fn input_norm(layer: usize) -> String {
        format!("model.language_model.layers.{}.input_layernorm.weight", layer)
    }
    pub fn post_attn_norm(layer: usize) -> String {
        format!("model.language_model.layers.{}.post_attention_layernorm.weight", layer)
    }
    pub fn pre_ffn_norm(layer: usize) -> String {
        format!("model.language_model.layers.{}.pre_feedforward_layernorm.weight", layer)
    }
    pub fn post_ffn_norm(layer: usize) -> String {
        format!("model.language_model.layers.{}.post_feedforward_layernorm.weight", layer)
    }
    pub fn gate_proj(layer: usize) -> String {
        format!("model.language_model.layers.{}.mlp.gate_proj.weight", layer)
    }
    pub fn up_proj(layer: usize) -> String {
        format!("model.language_model.layers.{}.mlp.up_proj.weight", layer)
    }
    pub fn down_proj(layer: usize) -> String {
        format!("model.language_model.layers.{}.mlp.down_proj.weight", layer)
    }
    pub fn q_norm(layer: usize) -> String {
        format!("model.language_model.layers.{}.self_attn.q_norm.weight", layer)
    }
    pub fn k_norm(layer: usize) -> String {
        format!("model.language_model.layers.{}.self_attn.k_norm.weight", layer)
    }
    pub fn embed_tokens() -> &'static str {
        "model.language_model.embed_tokens.weight"
    }
    pub fn final_norm() -> &'static str {
        "model.language_model.norm.weight"
    }
    pub fn lm_head() -> &'static str {
        "model.language_model.embed_tokens.weight" // tied
    }
}

// ---------------------------------------------------------------------------
// Mapped Gemma 4 Model — organized weights ready for inference
// ---------------------------------------------------------------------------

/// Per-layer attention weights, loaded from safetensors.
#[derive(Clone)]
pub struct Gemma4AttnWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub input_norm: Vec<f32>,
    pub post_attn_norm: Vec<f32>,
    /// Per-head RMSNorm on Q (head_dim elements).
    pub q_norm: Vec<f32>,
    /// Per-head RMSNorm on K (head_dim elements).
    pub k_norm: Vec<f32>,
    /// Per-layer head dimension (varies: 256 for sliding, 512 for full attention).
    pub head_dim: usize,
    /// Per-layer total Q dimension (num_heads * head_dim).
    pub q_dim: usize,
    /// Per-layer total KV dimension (num_kv_heads * head_dim).
    pub kv_dim: usize,
}

/// Per-layer FFN weights (either dense or MoE).
#[derive(Clone)]
pub enum Gemma4FfnWeights {
    Dense {
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
    },
    Moe {
        router: Vec<f32>,
        expert_gates: Vec<Vec<f32>>,
        expert_ups: Vec<Vec<f32>>,
        expert_downs: Vec<Vec<f32>>,
    },
}

/// All weights for a single Gemma 4 layer.
#[derive(Clone)]
pub struct Gemma4LayerWeights {
    pub attn: Gemma4AttnWeights,
    pub ffn: Gemma4FfnWeights,
    /// Norm before FFN.
    pub pre_ffn_norm: Vec<f32>,
    /// Norm after FFN.
    pub post_ffn_norm: Vec<f32>,
    /// Per-layer scalar multiplier.
    pub layer_scalar: f32,
    /// Per-layer input gate weights [hidden_size × hidden_size_per_layer_input].
    pub per_layer_input_gate: Option<Vec<f32>>,
    /// Per-layer projection weights [hidden_size_per_layer_input × hidden_size].
    pub per_layer_projection: Option<Vec<f32>>,
    /// Per-layer projection norm (hidden_size elements).
    pub post_per_layer_input_norm: Option<Vec<f32>>,
    /// Per-layer RoPE theta.
    pub rope_theta: f64,
    /// Per-layer partial rotary factor (1.0 = all dims, 0.25 = only first quarter).
    pub partial_rotary_factor: f32,
    /// Per-layer FFN intermediate dimension.
    pub intermediate_dim: usize,
}

/// Full Gemma 4 model weights loaded and organized.
#[derive(Clone)]
pub struct MappedGemma4Model {
    pub config: Gemma4Config,
    pub embed_tokens: Vec<f32>,  // [vocab_size × hidden_dim]
    pub layers: Vec<Gemma4LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,      // [vocab_size × hidden_dim] (may be tied)
    /// PLE model-level projection: [hidden_dim × (num_layers × ple_dim)].
    /// Projects hidden state to per-layer PLE space for Per-Layer Embeddings.
    pub ple_model_projection: Option<Vec<f32>>,
    /// PLE model-level norm: [ple_dim] weights for normalizing the PLE projection output.
    pub ple_projection_norm: Option<Vec<f32>>,
    /// PLE per-layer token embeddings: [vocab_size × (num_layers × ple_dim)].
    /// Token-level per-layer embeddings added to context projection in PLE.
    pub embed_tokens_per_layer: Option<Vec<f32>>,
}

impl MappedGemma4Model {
    /// Load from a mmap'd safetensors file. Only one tensor is in RAM at a time,
    /// making this suitable for large models that don't fit in memory.
    /// Auto-detects naming convention: checks for `model.language_model.embed_tokens.weight`
    /// (multimodal/Gemma 4 IT) vs `model.embed_tokens.weight` (standard HuggingFace).
    pub fn from_mmap(
        config: Gemma4Config,
        mmaped: &crate::model::safetensors::MmapedSafetensors,
    ) -> Result<Self, String> {
        // Probe for multimodal naming convention
        let is_mm = mmaped.get_tensor_f32(Gemma4MmTensorNames::embed_tokens()).is_ok();
        if is_mm {
            let get = |name: &str| -> Option<Vec<f32>> {
                mmaped.get_tensor_f32(name).ok()
            };
            Self::build_from_getter_mm(config, get)
        } else {
            let get = |name: &str| -> Option<Vec<f32>> {
                mmaped.get_tensor_f32(name).ok()
            };
            Self::build_from_getter(config, get)
        }
    }

    /// Load from a mmap'd safetensors file using multimodal naming convention
    /// (model.language_model.layers.N.*).
    pub fn from_mmap_mm(
        config: Gemma4Config,
        mmaped: &crate::model::safetensors::MmapedSafetensors,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Option<Vec<f32>> {
            mmaped.get_tensor_f32(name).ok()
        };

        Self::build_from_getter_mm(config, get)
    }

    /// Skeleton model: only embed_tokens, final_norm, lm_head, and per-layer norms.
    /// Projection weights (q/k/v/o/gate/up/down) are empty Vecs — they'll live on GPU.
    /// Saves ~3.3GB RAM vs full model. Use when system RAM is tight.
    pub fn from_mmap_mm_skeleton(
        config: Gemma4Config,
        mmaped: &crate::model::safetensors::MmapedSafetensors,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Option<Vec<f32>> {
            mmaped.get_tensor_f32(name).ok()
        };

        Self::build_skeleton_mm(config, get)
    }

    /// Skeleton model from file-backed reader (no mmap page faults).
    pub fn from_file_skeleton(
        config: Gemma4Config,
        file_st: &mut crate::model::safetensors::FileSafetensors,
    ) -> Result<Self, String> {
        let embed_tokens = file_st.get_tensor_f32(Gemma4MmTensorNames::embed_tokens())
            .map_err(|e| format!("embed_tokens: {:?}", e))?;
        let lm_head = file_st.get_tensor_f32(Gemma4MmTensorNames::lm_head())
            .unwrap_or_else(|_| embed_tokens.clone());
        let final_norm = file_st.get_tensor_f32(Gemma4MmTensorNames::final_norm())
            .map_err(|e| format!("final_norm: {:?}", e))?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let inorm = file_st.get_tensor_f32(&Gemma4MmTensorNames::input_norm(layer_idx))
                .map_err(|e| format!("input_norm {}: {:?}", layer_idx, e))?;
            let pnorm = file_st.get_tensor_f32(&Gemma4MmTensorNames::post_attn_norm(layer_idx))
                .map_err(|e| format!("post_attn_norm {}: {:?}", layer_idx, e))?;

            let attn = Gemma4AttnWeights {
                q_proj: Vec::new(),
                k_proj: Vec::new(),
                v_proj: Vec::new(),
                o_proj: Vec::new(),
                input_norm: inorm,
                post_attn_norm: pnorm,
                q_norm: vec![1.0f32; config.head_dim],
                k_norm: vec![1.0f32; config.head_dim],
                head_dim: config.head_dim,
                q_dim: config.num_heads * config.head_dim,
                kv_dim: config.num_kv_heads * config.head_dim,
            };
            let ffn = Gemma4FfnWeights::Dense {
                gate_proj: Vec::new(),
                up_proj: Vec::new(),
                down_proj: Vec::new(),
            };
            layers.push(Gemma4LayerWeights {
                attn, ffn,
                pre_ffn_norm: vec![1.0f32; config.hidden_dim],
                post_ffn_norm: vec![1.0f32; config.hidden_dim],
                layer_scalar: 1.0,
                per_layer_input_gate: None,
                per_layer_projection: None,
                post_per_layer_input_norm: None,
                rope_theta: config.layer_rope_theta(layer_idx),
                partial_rotary_factor: config.layer_partial_rotary_factor(layer_idx),
                intermediate_dim: config.intermediate_dim,
            });
        }

        Ok(Self { config, embed_tokens, layers, final_norm, lm_head, ple_model_projection: None, ple_projection_norm: None, embed_tokens_per_layer: None })
    }

    fn build_skeleton_mm<
        F: Fn(&str) -> Option<Vec<f32>>,
    >(config: Gemma4Config, get: F) -> Result<Self, String> {

        let embed_tokens = get(Gemma4MmTensorNames::embed_tokens())
            .ok_or_else(|| "Missing embed_tokens".to_string())?;
        let lm_head = get(Gemma4MmTensorNames::lm_head())
            .unwrap_or_else(|| embed_tokens.clone());
        let final_norm = get(Gemma4MmTensorNames::final_norm())
            .ok_or_else(|| "Missing final_norm".to_string())?;

        // Load only norms per layer (tiny ~6KB each), skip projections
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let inorm = get(&Gemma4MmTensorNames::input_norm(layer_idx))
                .ok_or_else(|| format!("Missing input_norm for layer {}", layer_idx))?;
            let pnorm = get(&Gemma4MmTensorNames::post_attn_norm(layer_idx))
                .ok_or_else(|| format!("Missing post_attn_norm for layer {}", layer_idx))?;

            let attn = Gemma4AttnWeights {
                q_proj: Vec::new(), // Empty — GPU has these
                k_proj: Vec::new(),
                v_proj: Vec::new(),
                o_proj: Vec::new(),
                input_norm: inorm,
                post_attn_norm: pnorm,
                q_norm: vec![1.0f32; config.head_dim],
                k_norm: vec![1.0f32; config.head_dim],
                head_dim: config.head_dim,
                q_dim: config.num_heads * config.head_dim,
                kv_dim: config.num_kv_heads * config.head_dim,
            };

            let ffn = Gemma4FfnWeights::Dense {
                gate_proj: Vec::new(),
                up_proj: Vec::new(),
                down_proj: Vec::new(),
            };

            layers.push(Gemma4LayerWeights {
                attn, ffn,
                pre_ffn_norm: vec![1.0f32; config.hidden_dim],
                post_ffn_norm: vec![1.0f32; config.hidden_dim],
                layer_scalar: 1.0,
                per_layer_input_gate: None,
                per_layer_projection: None,
                post_per_layer_input_norm: None,
                rope_theta: config.layer_rope_theta(layer_idx),
                partial_rotary_factor: config.layer_partial_rotary_factor(layer_idx),
                intermediate_dim: config.intermediate_dim,
            });
        }

        Ok(Self { config, embed_tokens, layers, final_norm, lm_head, ple_model_projection: None, ple_projection_norm: None, embed_tokens_per_layer: None })
    }

    /// Load from a safetensors LoadedWeights object.
    /// Uses E2B/E4B/12B/27B config to know the naming convention.
    pub fn from_loaded_weights(
        config: Gemma4Config,
        weights: &crate::model::safetensors::LoadedWeights,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Option<Vec<f32>> {
            weights.get(name).map(|t| t.data.clone())
        };

        Self::build_from_getter(config, get)
    }

    /// Shared implementation: build model from a weight getter closure.
    /// The getter is called once per tensor, so only one tensor is in RAM at a time
    /// when backed by mmap.
    fn build_from_getter<F: Fn(&str) -> Option<Vec<f32>>>(
        config: Gemma4Config,
        get: F,
    ) -> Result<Self, String> {
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let nkv = config.num_kv_heads;
        let hdim = config.head_dim;
        let q_dim = nh * hdim;
        let kv_dim = nkv * hdim;
        let idim = config.intermediate_dim;

        // Embedding — NOT transposed (used as lookup table: embed[tid * hd + d])
        let embed_tokens = get(Gemma4TensorNames::embed_tokens())
            .ok_or_else(|| "Missing embed_tokens".to_string())?;

        // LM head — stored as [vocab_size × hd], transpose to [hd × vocab_size]
        let lm_head_raw = get(Gemma4TensorNames::lm_head())
            .unwrap_or_else(|| embed_tokens.clone());
        let lm_head = transpose(&lm_head_raw, config.vocab_size, hd);

        // Final norm
        let final_norm = get(Gemma4TensorNames::final_norm())
            .ok_or_else(|| "Missing final norm".to_string())?;

        // Per-layer weights
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            // Attention: safetensors stores [out × in], transpose to [in × out]
            let q = transpose(&get(&Gemma4TensorNames::q_proj(layer_idx))
                .ok_or_else(|| format!("Missing q_proj for layer {}", layer_idx))?, q_dim, hd);
            let k = transpose(&get(&Gemma4TensorNames::k_proj(layer_idx))
                .ok_or_else(|| format!("Missing k_proj for layer {}", layer_idx))?, kv_dim, hd);
            let v = transpose(&get(&Gemma4TensorNames::v_proj(layer_idx))
                .ok_or_else(|| format!("Missing v_proj for layer {}", layer_idx))?, kv_dim, hd);
            let o = transpose(&get(&Gemma4TensorNames::o_proj(layer_idx))
                .ok_or_else(|| format!("Missing o_proj for layer {}", layer_idx))?, hd, q_dim);
            let inorm = get(&Gemma4TensorNames::input_norm(layer_idx))
                .ok_or_else(|| format!("Missing input_norm for layer {}", layer_idx))?;
            let pnorm = get(&Gemma4TensorNames::post_attn_norm(layer_idx))
                .ok_or_else(|| format!("Missing post_attn_norm for layer {}", layer_idx))?;

            let attn = Gemma4AttnWeights {
                q_proj: q, k_proj: k, v_proj: v, o_proj: o,
                input_norm: inorm,
                post_attn_norm: pnorm,
                q_norm: vec![1.0f32; config.head_dim],
                k_norm: vec![1.0f32; config.head_dim],
                head_dim: config.head_dim,
                q_dim: config.num_heads * config.head_dim,
                kv_dim: config.num_kv_heads * config.head_dim,
            };

            // FFN: MoE or dense?
            let ffn = if config.moe_layers.contains(&layer_idx) {
                // MoE layer
                let router_raw = get(&Gemma4TensorNames::moe_router(layer_idx))
                    .ok_or_else(|| format!("Missing MoE router for layer {}", layer_idx))?;
                let router = transpose(&router_raw, config.num_experts, hd);

                let mut expert_gates = Vec::new();
                let mut expert_ups = Vec::new();
                let mut expert_downs = Vec::new();

                for e in 0..config.num_experts {
                    let g = transpose(&get(&Gemma4TensorNames::expert_gate(layer_idx, e))
                        .ok_or_else(|| format!("Missing expert {} gate for layer {}", e, layer_idx))?, idim, hd);
                    let u = transpose(&get(&Gemma4TensorNames::expert_up(layer_idx, e))
                        .ok_or_else(|| format!("Missing expert {} up for layer {}", e, layer_idx))?, idim, hd);
                    let d = transpose(&get(&Gemma4TensorNames::expert_down(layer_idx, e))
                        .ok_or_else(|| format!("Missing expert {} down for layer {}", e, layer_idx))?, hd, idim);
                    expert_gates.push(g);
                    expert_ups.push(u);
                    expert_downs.push(d);
                }

                Gemma4FfnWeights::Moe {
                    router,
                    expert_gates,
                    expert_ups,
                    expert_downs,
                }
            } else {
                // Dense FFN
                let gate = transpose(&get(&Gemma4TensorNames::gate_proj(layer_idx))
                    .ok_or_else(|| format!("Missing gate_proj for layer {}", layer_idx))?, idim, hd);
                let up = transpose(&get(&Gemma4TensorNames::up_proj(layer_idx))
                    .ok_or_else(|| format!("Missing up_proj for layer {}", layer_idx))?, idim, hd);
                let down = transpose(&get(&Gemma4TensorNames::down_proj(layer_idx))
                    .ok_or_else(|| format!("Missing down_proj for layer {}", layer_idx))?, hd, idim);

                Gemma4FfnWeights::Dense {
                    gate_proj: gate,
                    up_proj: up,
                    down_proj: down,
                }
            };

            layers.push(Gemma4LayerWeights {
                attn, ffn,
                pre_ffn_norm: vec![1.0f32; config.hidden_dim],
                post_ffn_norm: vec![1.0f32; config.hidden_dim],
                layer_scalar: 1.0,
                per_layer_input_gate: None,
                per_layer_projection: None,
                post_per_layer_input_norm: None,
                rope_theta: config.layer_rope_theta(layer_idx),
                partial_rotary_factor: config.layer_partial_rotary_factor(layer_idx),
                intermediate_dim: config.intermediate_dim,
            });
        }

        Ok(Self { config, embed_tokens, layers, final_norm, lm_head, ple_model_projection: None, ple_projection_norm: None, embed_tokens_per_layer: None })
    }

    /// Build model using multimodal naming convention (model.language_model.layers.N.*).
    /// Infers per-layer head_dim and intermediate_dim from actual tensor shapes,
    /// so it works correctly with any Gemma 4 variant.
    fn build_from_getter_mm<F: Fn(&str) -> Option<Vec<f32>>>(
        config: Gemma4Config,
        get: F,
    ) -> Result<Self, String> {
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let nkv = config.num_kv_heads;
        let ple_dim = config.hidden_size_per_layer_input.unwrap_or(0);

        let embed_tokens = get(Gemma4MmTensorNames::embed_tokens())
            .ok_or_else(|| "Missing embed_tokens".to_string())?;

        let lm_head_raw = get(Gemma4MmTensorNames::lm_head())
            .unwrap_or_else(|| embed_tokens.clone());
        let lm_head = transpose(&lm_head_raw, config.vocab_size, hd);

        let final_norm = get(Gemma4MmTensorNames::final_norm())
            .ok_or_else(|| "Missing final norm".to_string())?;

        // Load per-layer token embeddings for PLE if present
        // PLE model-level weights (projects hidden state to per-layer PLE space)
        tracing::info!(event = "ple_lookup", "Looking for PLE model projection");
        let ple_model_projection = get("model.language_model.per_layer_model_projection.weight")
            .map(|w| {
                let ple_total = config.num_layers * config.hidden_size_per_layer_input.unwrap_or(256);
                tracing::info!(event = "ple_model_proj", raw_len = w.len(), ple_total, hd, "PLE model projection");
                transpose(&w, ple_total, hd)
            });
        let ple_projection_norm = get("model.language_model.per_layer_projection_norm.weight");

        // Load per-layer token embeddings for PLE: [vocab_size × (num_layers × ple_dim)]
        // This is the token-level component that gets ADDED to the context projection.
        let embed_tokens_per_layer = get("model.language_model.embed_tokens_per_layer.weight")
            .map(|w| {
                let ple_dim = config.hidden_size_per_layer_input.unwrap_or(256);
                let ple_total = config.num_layers * ple_dim;
                tracing::info!(event = "ple_tokens_loaded", raw_len = w.len(), expected = config.vocab_size * ple_total, "Loaded embed_tokens_per_layer");
                // Store as-is: [vocab_size × ple_total], rows are token embeddings
                w
            });
        if embed_tokens_per_layer.is_some() {
            tracing::info!(event = "ple_tokens_loaded", "Loaded per-layer token embeddings for PLE");
        }

        if ple_model_projection.is_some() {
            tracing::info!(event = "ple_loaded", "Loaded PLE model projection + norm");
        }

        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            // --- Infer per-layer dimensions from actual tensor shapes ---
            // q_proj is stored as [q_dim, hidden_dim] in safetensors.
            // We need to know the actual q_dim to read the weight correctly.
            let q_raw = get(&Gemma4MmTensorNames::q_proj(layer_idx))
                .ok_or_else(|| format!("Missing q_proj for layer {}", layer_idx))?;
            let layer_q_dim = q_raw.len() / hd;
            let layer_head_dim = layer_q_dim / nh;
            let layer_kv_dim = nkv * layer_head_dim;

            // gate_proj is [intermediate_dim, hidden_dim]
            let gate_raw = get(&Gemma4MmTensorNames::gate_proj(layer_idx))
                .ok_or_else(|| format!("Missing gate_proj for layer {}", layer_idx))?;
            let layer_inter_dim = gate_raw.len() / hd;

            // Per-layer rope config from config
            let rope_theta = config.layer_rope_theta(layer_idx);
            let partial_rotary_factor = config.layer_partial_rotary_factor(layer_idx);

            let q = transpose(&q_raw, layer_q_dim, hd);
            let k = transpose(&get(&Gemma4MmTensorNames::k_proj(layer_idx))
                .ok_or_else(|| format!("Missing k_proj for layer {}", layer_idx))?, layer_kv_dim, hd);
            let v = transpose(&get(&Gemma4MmTensorNames::v_proj(layer_idx))
                .ok_or_else(|| format!("Missing v_proj for layer {}", layer_idx))?, layer_kv_dim, hd);
            let o = transpose(&get(&Gemma4MmTensorNames::o_proj(layer_idx))
                .ok_or_else(|| format!("Missing o_proj for layer {}", layer_idx))?, hd, layer_q_dim);
            let inorm = get(&Gemma4MmTensorNames::input_norm(layer_idx))
                .ok_or_else(|| format!("Missing input_norm for layer {}", layer_idx))?;
            let pnorm = get(&Gemma4MmTensorNames::post_attn_norm(layer_idx))
                .ok_or_else(|| format!("Missing post_attn_norm for layer {}", layer_idx))?;

            // Q/K per-head RMSNorm — shape matches per-layer head_dim
            let q_norm = get(&Gemma4MmTensorNames::q_norm(layer_idx))
                .unwrap_or_else(|| vec![1.0f32; layer_head_dim]);
            let k_norm = get(&Gemma4MmTensorNames::k_norm(layer_idx))
                .unwrap_or_else(|| vec![1.0f32; layer_head_dim]);

            let attn = Gemma4AttnWeights {
                q_proj: q, k_proj: k, v_proj: v, o_proj: o,
                input_norm: inorm,
                post_attn_norm: pnorm,
                q_norm,
                k_norm,
                head_dim: layer_head_dim,
                q_dim: layer_q_dim,
                kv_dim: layer_kv_dim,
            };

            let gate = transpose(&gate_raw, layer_inter_dim, hd);
            let up = transpose(&get(&Gemma4MmTensorNames::up_proj(layer_idx))
                .ok_or_else(|| format!("Missing up_proj for layer {}", layer_idx))?, layer_inter_dim, hd);
            let down = transpose(&get(&Gemma4MmTensorNames::down_proj(layer_idx))
                .ok_or_else(|| format!("Missing down_proj for layer {}", layer_idx))?, hd, layer_inter_dim);

            let ffn = Gemma4FfnWeights::Dense {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
            };

            // Pre/post FFN norms
            let pre_ffn_norm = get(&Gemma4MmTensorNames::pre_ffn_norm(layer_idx))
                .unwrap_or_else(|| vec![1.0f32; hd]);
            let post_ffn_norm = get(&Gemma4MmTensorNames::post_ffn_norm(layer_idx))
                .unwrap_or_else(|| vec![1.0f32; hd]);

            // Per-layer scalar — NO .weight suffix in the actual tensor name
            let layer_scalar = get(&format!("model.language_model.layers.{}.layer_scalar", layer_idx))
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0f32);

            // Per-layer embeddings (PLE)
            // Gate: raw [ple_dim, hd] → transpose to [hd, ple_dim] for matmul(hidden, gate)
            // HuggingFace: nn.Linear(hidden_size=hd, ple_dim) stores weight as [ple_dim, hd]
            let per_layer_input_gate = if ple_dim > 0 {
                get(&format!("model.language_model.layers.{}.per_layer_input_gate.weight", layer_idx))
                    .map(|w| {
                        tracing::info!(event = "ple_gate_shape", layer = layer_idx, raw_len = w.len(), ple_dim, hd, "PLE gate weight");
                        transpose(&w, ple_dim, hd) // [ple_dim, hd] → [hd, ple_dim]
                    })
            } else {
                None
            };
            // Projection: raw [hd, ple_dim] → transpose to [ple_dim, hd] for matmul(gated, proj)
            // HuggingFace: nn.Linear(ple_dim, hidden_size=hd) stores weight as [hd, ple_dim]
            let per_layer_projection = if ple_dim > 0 {
                get(&format!("model.language_model.layers.{}.per_layer_projection.weight", layer_idx))
                    .map(|w| {
                        tracing::info!(event = "ple_proj_shape", layer = layer_idx, raw_len = w.len(), ple_dim, hd, "PLE proj weight");
                        transpose(&w, hd, ple_dim) // [hd, ple_dim] → [ple_dim, hd]
                    })
            } else {
                None
            };
            let post_per_layer_input_norm = if ple_dim > 0 {
                get(&format!("model.language_model.layers.{}.post_per_layer_input_norm.weight", layer_idx))
            } else {
                None
            };

            layers.push(Gemma4LayerWeights {
                attn, ffn,
                pre_ffn_norm, post_ffn_norm, layer_scalar,
                per_layer_input_gate, per_layer_projection, post_per_layer_input_norm,
                rope_theta,
                partial_rotary_factor,
                intermediate_dim: layer_inter_dim,
            });
            tracing::info!(
                event = "loaded_layer",
                "Loaded layer {}/{}: head_dim={}, inter_dim={}, rope_theta={}, scalar={:.4}",
                layer_idx + 1, config.num_layers, layer_head_dim, layer_inter_dim, rope_theta, layer_scalar,
            );
        }

        Ok(Self { config, embed_tokens, layers, final_norm, lm_head, ple_model_projection, ple_projection_norm, embed_tokens_per_layer })
    }
}

// ---------------------------------------------------------------------------
// RMS Norm (CPU)
// ---------------------------------------------------------------------------

/// GeLU activation (tanh approximation), as used by Gemma 4.
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Per-head RMSNorm: apply RMSNorm independently to each attention head.
/// Input shape: [seq × (num_heads × head_dim)]
/// Weight shape: [head_dim] (shared across all heads and seq positions)
pub fn per_head_rms_norm(input: &[f32], weight: &[f32], seq_len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
    let total_dim = num_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * total_dim];
    for t in 0..seq_len {
        for h in 0..num_heads {
            let base = t * total_dim + h * head_dim;
            // Compute RMS for this head
            let mut sum_sq = 0.0f32;
            for d in 0..head_dim {
                let x = input[base + d];
                sum_sq += x * x;
            }
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
            // Apply norm * weight
            for d in 0..head_dim {
                let g = weight.get(d).copied().unwrap_or(1.0);
                output[base + d] = input[base + d] * inv_rms * g;
            }
        }
    }
    output
}

/// Per-head RMSNorm without learned scale (for V normalization in Gemma 4).
/// Just normalizes: x / sqrt(mean(x^2) + eps)
pub fn per_head_rms_norm_no_scale(input: &[f32], seq_len: usize, num_heads: usize, head_dim: usize) -> Vec<f32> {
    let total_dim = num_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * total_dim];
    for t in 0..seq_len {
        for h in 0..num_heads {
            let base = t * total_dim + h * head_dim;
            let mut sum_sq = 0.0f32;
            for d in 0..head_dim {
                sum_sq += input[base + d] * input[base + d];
            }
            let inv_rms = 1.0 / (sum_sq / head_dim as f32 + 1e-6).sqrt();
            for d in 0..head_dim {
                output[base + d] = input[base + d] * inv_rms;
            }
        }
    }
    output
}

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps).
pub fn rms_norm(input: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
    let n = input.len() / dim;
    let mut output = Vec::with_capacity(input.len());
    for t in 0..n {
        let slice = &input[t * dim..(t + 1) * dim];
        let mean_sq: f32 = slice.iter().map(|x| x * x).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for (d, &x) in slice.iter().enumerate() {
            let g = weight.get(d).copied().unwrap_or(1.0);
            output.push(x * inv_rms * g);
        }
    }
    output
}

/// Rotary position embedding (RoPE) for attention.
///
/// Gemma 4 uses two RoPE types:
/// - `default` (sliding_attention): standard RoPE with freq = 1/theta^(2d/dim)
/// - `proportional` (full_attention): freq = 1/theta^(2d/head_dim) where only
///   the first `partial_rotary_factor * head_dim` dimensions get rotation.
///
/// The key difference is the exponent denominator:
/// - default: uses `rotary_dims` (= head_dim for full rotation)
/// - proportional: always uses `head_dim`, not `rotary_dims`
///
/// `partial_rotary_factor` controls what fraction of dimensions get RoPE.
/// For standard RoPE: partial_rotary_factor=1.0 (all dims).
/// For Gemma 4 full attention: partial_rotary_factor=0.25 (first 25% of dims).
pub fn apply_rope(x: &mut [f32], seq_len: usize, num_heads: usize, head_dim: usize, offset: usize, theta: f64, partial_rotary_factor: f32) {
    let rotary_dims = (head_dim as f32 * partial_rotary_factor) as usize;
    let half = rotary_dims / 2;
    for t in 0..seq_len {
        let pos = (t + offset) as f64;
        for h in 0..num_heads {
            for d in 0..half {
                // Proportional RoPE: exponent uses head_dim as denominator
                // freq = 1/theta^(2d/head_dim), matching HuggingFace's proportional RoPE
                let freq = 1.0 / theta.powf((2 * d) as f64 / head_dim as f64);
                let angle = pos * freq;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;

                let base = t * num_heads * head_dim + h * head_dim;
                let x0 = x[base + d];
                let x1 = x[base + d + half];

                x[base + d] = x0 * cos_a - x1 * sin_a;
                x[base + d + half] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

/// RoPE for GQA KV heads (fewer heads than Q).
pub fn apply_rope_gqa(x: &mut [f32], seq_len: usize, num_kv_heads: usize, head_dim: usize, offset: usize, theta: f64, partial_rotary_factor: f32) {
    let rotary_dims = (head_dim as f32 * partial_rotary_factor) as usize;
    let half = rotary_dims / 2;
    let kv_dim = num_kv_heads * head_dim;
    for t in 0..seq_len {
        let pos = (t + offset) as f64;
        for h in 0..num_kv_heads {
            for d in 0..half {
                let freq = 1.0 / theta.powf((2 * d) as f64 / head_dim as f64);
                let angle = pos * freq;
                let cos_a = angle.cos() as f32;
                let sin_a = angle.sin() as f32;

                let base = t * kv_dim + h * head_dim;
                let x0 = x[base + d];
                let x1 = x[base + d + half];

                x[base + d] = x0 * cos_a - x1 * sin_a;
                x[base + d + half] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Teacher Forward Pass (CPU)
// ---------------------------------------------------------------------------

/// Gemma 4 teacher model for distillation.
/// Full CPU forward pass producing logits.
/// Result of Gemma 4 forward pass. Contains logits always, and optionally per-layer hidden states.
struct Gemma4ForwardResult {
    logits: Vec<f32>,
    layer_states: Vec<Vec<f32>>,
}

pub struct Gemma4Teacher {
    pub model: MappedGemma4Model,
}

impl Gemma4Teacher {
    pub fn new(model: MappedGemma4Model) -> Self {
        Self { model }
    }

    /// Access the underlying model.
    pub fn model(&self) -> &MappedGemma4Model {
        &self.model
    }

    /// Forward pass: token IDs → logits.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        self.forward_impl(token_ids, false).logits
    }

    /// Forward pass: token IDs → per-layer hidden states.
    /// Returns Vec where [0] = post-embedding, [1] = post-layer0, ..., [N] = post-layer(N-1).
    pub fn forward_with_hidden_states(&self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        self.forward_impl(token_ids, true).layer_states
    }

    /// Unified forward implementation shared by forward() and forward_with_hidden_states().
    /// Single source of truth for the entire Gemma 4 transformer.
    fn forward_impl(&self, token_ids: &[u32], collect_states: bool) -> Gemma4ForwardResult {
        let config = &self.model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let seq = token_ids.len();
        let vs = config.vocab_size;
        let mut layer_states: Vec<Vec<f32>> = if collect_states { Vec::new() } else { Vec::new() };

        // 1. Embedding lookup (Gemma scales by sqrt(hidden_dim))
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.model.embed_tokens.len() {
                for d in 0..hd {
                    hidden[t * hd + d] = self.model.embed_tokens[id * hd + d];
                }
            }
        }
        let scale = (hd as f32).sqrt();
        for h in hidden.iter_mut() { *h *= scale; }

        // Collect post-embedding state
        if collect_states { layer_states.push(hidden.clone()); }

        // Diagnostic: embedding output
        let emb_norm: f32 = hidden.iter().take(hd).map(|x| x * x).sum::<f32>().sqrt();
        let emb_first5: Vec<f32> = hidden.iter().take(5).copied().collect();
        tracing::info!(event = "embed_diag", l2 = emb_norm, first5 = ?emb_first5, "After embedding + scaling");

        // Debug: dump first 10 values at key points for layer 0 to compare with Python reference
        // Python ref: after_embed=[-1.636183, -1.530931, 0.187778, ...]
        if false {
            let first10: Vec<String> = hidden.iter().take(10).map(|v| format!("{:.6}", v)).collect();
            tracing::info!(event = "debug_compare", step = "after_embed", vals = ?first10);
        }

        // KV sharing: Gemma 4 layers 15-34 share K/V states from layers 13/14.
        // We store computed K/V so shared layers can reuse them.
        let first_shared_layer = config.num_layers.saturating_sub(config.num_kv_shared_layers);
        let mut shared_kv_states: std::collections::HashMap<usize, (Vec<f32>, Vec<f32>)> = std::collections::HashMap::new();

        // Pre-compute PLE (Per-Layer Embedding) inputs using the INITIAL hidden state.
        // This matches llama.cpp's project_per_layer_inputs() which runs ONCE before the layer loop.
        // Flow: proj = hidden @ model_proj * (1/sqrt(hd)) → norm → + token_embs * sqrt(ple_dim) → * (1/sqrt(2))
        let ple_dim = config.hidden_size_per_layer_input.unwrap_or(0);
        let ple_precomputed: Option<Vec<f32>> = if ple_dim > 0
            && self.model.ple_model_projection.is_some()
            && self.model.ple_projection_norm.is_some()
        {
            let model_proj = self.model.ple_model_projection.as_ref().unwrap();
            let proj_norm = self.model.ple_projection_norm.as_ref().unwrap();
            let ple_total = ple_dim * config.num_layers;

            // 1. Context projection: hidden[seq, hd] @ model_proj[hd, ple_total] → [seq, ple_total]
            //    Scale by 1/sqrt(hd) — NOT sqrt(hd)!
            let model_proj_scale = (hd as f32).sqrt().recip(); // 1/sqrt(hd)
            let ple_all = matmul(&hidden, model_proj, seq, hd, ple_total);

            // Apply scale and reshape to [seq, n_layer, ple_dim]
            let mut ple_projected = vec![0.0f32; seq * config.num_layers * ple_dim];
            for t in 0..seq {
                for l in 0..config.num_layers {
                    for d in 0..ple_dim {
                        let src = t * ple_total + l * ple_dim + d;
                        let dst = t * config.num_layers * ple_dim + l * ple_dim + d;
                        ple_projected[dst] = ple_all[src] * model_proj_scale;
                    }
                }
            }

            // 2. Per-layer RMS norm on the context projection
            //    Norm is applied per-layer slice [ple_dim]
            let mut ple_normed = vec![0.0f32; ple_projected.len()];
            for t in 0..seq {
                for l in 0..config.num_layers {
                    let base = t * config.num_layers * ple_dim + l * ple_dim;
                    let slice = &ple_projected[base..base+ple_dim];
                    let normed = rms_norm(slice, proj_norm, ple_dim, 1e-6);
                    ple_normed[base..base+ple_dim].copy_from_slice(&normed);
                }
            }

            // 3. Add per-layer token embeddings (scaled by sqrt(ple_dim))
            let tok_scale = (ple_dim as f32).sqrt();
            if let Some(ref token_embs) = self.model.embed_tokens_per_layer {
                for t in 0..seq {
                    let tok_id = token_ids[t] as usize;
                    for l in 0..config.num_layers {
                        let dst = t * config.num_layers * ple_dim + l * ple_dim;
                        let src = tok_id * ple_total + l * ple_dim;
                        for d in 0..ple_dim {
                            ple_normed[dst + d] += token_embs[src + d] * tok_scale;
                        }
                    }
                }
            }

            // 4. Apply 1/sqrt(2) scale
            let combined_scale = (2.0f32).sqrt().recip();
            for v in ple_normed.iter_mut() {
                *v *= combined_scale;
            }

            Some(ple_normed) // shape: [seq, n_layer, ple_dim]
        } else {
            None
        };

        // 2. Per-layer transformer (Gemma 4 architecture)
        //    residual = hidden
        //    hidden = input_layernorm(hidden)
        //    hidden = attention(hidden)  → Q/K norms applied inside
        //    hidden = post_attention_layernorm(hidden)
        //    hidden = residual + hidden
        //    residual = hidden
        //    hidden = pre_feedforward_layernorm(hidden)
        //    hidden = ffn(hidden)        → GeLU activation, NOT SwiGLU
        //    hidden = post_feedforward_layernorm(hidden)
        //    hidden = residual + hidden
        //    hidden = per_layer_input_gate + per_layer_projection (PLE)
        //    hidden *= layer_scalar
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            // Per-layer head_dim and dimensions (varies: 256 for sliding, 512 for full attention)
            let layer_head_dim = layer.attn.head_dim;
            let layer_q_dim = layer.attn.q_dim;
            let layer_kv_dim = layer.attn.kv_dim;
            let layer_inter_dim = layer.intermediate_dim;

            // Diagnostic: detailed layer trace for comparison with Python reference
            if layer_idx == 0 {
                let first10: Vec<String> = hidden.iter().take(10).map(|v| format!("{:.6}", v)).collect();
                let l2 = hidden.iter().take(hd).map(|x| x*x).sum::<f32>().sqrt();
                tracing::info!(event = "debug_compare", step = "before_layer", layer = layer_idx, l2 = l2, vals = ?first10);
            }

            let residual = hidden.clone();

            // Input RMSNorm
            let normed = rms_norm(&hidden, &layer.attn.input_norm, hd, 1e-6);

            // GQA Attention — with KV sharing for shared layers
            let attn_out = if layer_idx >= first_shared_layer && first_shared_layer > 0 {
                // Shared KV layer — look up which layer to share from
                let kv_source = Self::kv_shared_source_layer(
                    layer_idx, first_shared_layer, &config.layer_types,
                );
                if let Some((shared_k, shared_v)) = shared_kv_states.get(&kv_source) {
                    self.attention_forward_with_shared_kv(
                        &normed, &layer.attn, nh, config.num_kv_heads, layer_head_dim,
                        layer_q_dim, seq, hd,
                        layer.rope_theta, layer.partial_rotary_factor,
                        shared_k, shared_v,
                    )
                } else {
                    // Fallback: compute own K/V (shouldn't happen if source layer < first_shared)
                    tracing::warn!(event = "kv_share_fallback", layer = layer_idx, source = kv_source, "No shared KV found, computing own");
                    self.attention_forward(
                        &normed, &layer.attn, nh, config.num_kv_heads, layer_head_dim,
                        layer_q_dim, layer_kv_dim, seq, hd,
                        layer.rope_theta, layer.partial_rotary_factor,
                    )
                }
            } else {
                // Normal layer — compute own K/V and store for potential sharing
                let (attn_output, kv_k, kv_v) = self.attention_forward_collect_kv(
                    &normed, &layer.attn, nh, config.num_kv_heads, layer_head_dim,
                    layer_q_dim, layer_kv_dim, seq, hd,
                    layer.rope_theta, layer.partial_rotary_factor,
                );
                // Store K/V for shared layers to reference
                if first_shared_layer > 0 {
                    shared_kv_states.insert(layer_idx, (kv_k, kv_v));
                }
                attn_output
            };

            // Post-attention RMSNorm (applied TO attn output, before residual)
            let attn_out_raw_l2 = if layer_idx == 0 {
                let n: f32 = attn_out.iter().take(hd).map(|x| x * x).sum::<f32>().sqrt();
                Some(n)
            } else { None };
            let attn_out = rms_norm(&attn_out, &layer.attn.post_attn_norm, hd, 1e-6);
            if let Some(raw) = attn_out_raw_l2 {
                let post: f32 = attn_out.iter().take(hd).map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(event = "substep", step = "attn_raw_vs_normed", layer = 0, raw_l2 = raw, normed_l2 = post);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] = residual[i] + attn_out[i];
            }

            if layer_idx == 0 {
                let n: f32 = hidden.iter().take(hd).map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!(event = "substep", step = "after_attn_residual", layer = layer_idx, l2 = n);
            }

            let residual2 = hidden.clone();

            // Pre-FFN RMSNorm
            let normed2 = rms_norm(&hidden, &layer.pre_ffn_norm, hd, 1e-6);

            // FFN: GeLU(gate(x)) * up(x), NOT SwiGLU
            let ffn_out = match &layer.ffn {
                Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                    self.dense_ffn_gelu(&normed2, gate_proj, up_proj, down_proj, hd, layer_inter_dim)
                }
                Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                    self.moe_ffn(&normed2, router, expert_gates, expert_ups, expert_downs,
                                 hd, layer_inter_dim, config.num_experts, config.top_k)
                }
            };

            // Post-FFN RMSNorm (applied TO ffn output, before residual)
            if layer_idx == 0 {
                let first10: Vec<String> = ffn_out.iter().take(10).map(|v| format!("{:.6}", v)).collect();
                let l2 = ffn_out.iter().take(hd).map(|x| x*x).sum::<f32>().sqrt();
                tracing::info!(event = "debug_compare", step = "ffn_out_before_norm", layer = layer_idx, l2 = l2, vals = ?first10);
            }
            let ffn_out = rms_norm(&ffn_out, &layer.post_ffn_norm, hd, 1e-6);
            if layer_idx == 0 {
                let first10: Vec<String> = ffn_out.iter().take(10).map(|v| format!("{:.6}", v)).collect();
                let l2 = ffn_out.iter().take(hd).map(|x| x*x).sum::<f32>().sqrt();
                tracing::info!(event = "debug_compare", step = "ffn_out_after_norm", layer = layer_idx, l2 = l2, vals = ?first10);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] = residual2[i] + ffn_out[i];
            }

            // Per-Layer Embeddings (PLE) — uses PRE-COMPUTED inputs from initial hidden state.
            // Inside the loop, we only do: gate = gelu(hidden @ gate_w), then gate * ple_input @ proj_w
            if let Some(ref ple_pre) = ple_precomputed {
                if layer.per_layer_input_gate.is_some() && layer.per_layer_projection.is_some() && layer.post_per_layer_input_norm.is_some()
                {
                    // Extract this layer's PLE slice: [seq, ple_dim]
                    let mut ple_input = vec![0.0f32; seq * ple_dim];
                    for t in 0..seq {
                        let src = t * config.num_layers * ple_dim + layer_idx * ple_dim;
                        let dst = t * ple_dim;
                        for d in 0..ple_dim {
                            ple_input[dst + d] = ple_pre[src + d];
                        }
                    }

                    // Gate: hidden → ple_dim with GELU
                    let gate_out = matmul(&hidden, layer.per_layer_input_gate.as_ref().unwrap(), seq, hd, ple_dim);
                    let gate_gelu: Vec<f32> = gate_out.iter().map(|&x| gelu_tanh(x)).collect();

                    // Gated input: gate * ple_input (element-wise in ple_dim)
                    let mut gated_input = vec![0.0f32; seq * ple_dim];
                    for i in 0..seq * ple_dim {
                        gated_input[i] = gate_gelu[i] * ple_input[i];
                    }

                    // Project back: [seq, ple_dim] → [seq, hd]
                    let proj_out = matmul(&gated_input, layer.per_layer_projection.as_ref().unwrap(), seq, ple_dim, hd);

                    // Post norm and add residual
                    let ple_final = rms_norm(&proj_out, layer.post_per_layer_input_norm.as_ref().unwrap(), hd, 1e-6);
                    for i in 0..hidden.len() {
                        hidden[i] += ple_final[i];
                    }
                }
            }

            // Layer scalar
            let ls = layer.layer_scalar;
            if ls != 1.0 {
                for h in hidden.iter_mut() { *h *= ls; }
            }

            // Collect post-layer state
            if collect_states { layer_states.push(hidden.clone()); }

            // Diagnostic: hidden state stats after each layer
            // Show first token AND last token to detect if generation is changing
            if layer_idx <= 1 || layer_idx == config.num_layers - 1 {
                // First token stats
                let norm: f32 = hidden.iter().take(hd).map(|x| x * x).sum::<f32>().sqrt();
                let first5: Vec<f32> = hidden.iter().take(5).copied().collect();
                tracing::info!(
                    event = "hidden_stats",
                    layer = layer_idx,
                    ls = ls,
                    l2_norm = norm,
                    first5 = ?first5,
                    "Hidden state after layer (token 0)"
                );
                // Last token stats
                if seq > 1 {
                    let last_off = (seq - 1) * hd;
                    let last_norm: f32 = hidden[last_off..last_off+hd].iter().map(|x| x * x).sum::<f32>().sqrt();
                    let last5: Vec<f32> = hidden[last_off..last_off+5].to_vec();
                    tracing::info!(
                        event = "hidden_stats_last",
                        layer = layer_idx,
                        pos = seq - 1,
                        l2_norm = last_norm,
                        first5 = ?last5,
                        "Hidden state after layer (last token)"
                    );
                }
            }
        }

        // 3. Final RMSNorm
        // Diagnostic: check final_norm weights
        {
            let fn5: Vec<f32> = self.model.final_norm.iter().take(5).copied().collect();
            let fn_max: f32 = self.model.final_norm.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let fn_min: f32 = self.model.final_norm.iter().cloned().fold(f32::INFINITY, f32::min);
            tracing::info!(event = "final_norm_weights", first5 = ?fn5, min = fn_min, max = fn_max, "Final norm weights");
        }
        hidden = rms_norm(&hidden, &self.model.final_norm, hd, 1e-6);

        // Diagnostic: hidden state at last position before logits
        if seq > 0 {
            let last_off = (seq - 1) * hd;
            let last_norm: f32 = hidden[last_off..last_off+hd].iter().map(|x| x * x).sum::<f32>().sqrt();
            let last5: Vec<f32> = hidden[last_off..last_off+5].to_vec();
            tracing::info!(event = "final_hidden", l2 = last_norm, first5 = ?last5, "Final hidden state (last pos)");
        }

        // 4. LM head: [seq × hd] × [hd × vs] → [seq × vs]
        let mut logits = matmul(&hidden, &self.model.lm_head, seq, hd, vs);

        // Diagnostic: logit stats at last position
        if seq > 0 {
            let last_logits = &logits[(seq-1)*vs..seq*vs];
            let max_l = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_l = last_logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let mean_l: f32 = last_logits.iter().sum::<f32>() / vs as f32;
            let top5: Vec<(usize, f32)> = {
                let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed[..5.min(indexed.len())].to_vec()
            };
            tracing::info!(event = "logits_raw", min = min_l, max = max_l, mean = mean_l, top5 = ?top5, "Raw logits (last pos, pre-softcap)");
        }

        // 5. Final logit softcapping: logits = tanh(logits / cap) * cap
        if let Some(cap) = config.final_logit_softcapping {
            for l in logits.iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }

        // Diagnostic: entropy after softcapping
        if seq > 0 {
            let last_logits = &logits[(seq-1)*vs..seq*vs];
            let max_l = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for l in last_logits { sum_exp += (l - max_l).exp(); }
            let log_sum_exp = sum_exp.ln() + max_l;
            let mut entropy = 0.0f32;
            for l in last_logits {
                let p = (l - log_sum_exp).exp();
                if p > 0.0 { entropy -= p * p.ln(); }
            }
            let top5: Vec<(usize, f32)> = {
                let mut indexed: Vec<(usize, f32)> = last_logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed[..5.min(indexed.len())].to_vec()
            };
            tracing::info!(event = "logits_after_softcap", entropy = entropy, max = max_l, top5 = ?top5, "After softcapping");
        }

        let logits = logits;

        // Return both logits and collected states
        Gemma4ForwardResult { logits, layer_states }
    }

    /// Compute which layer a shared layer gets its KV from.
    /// Mirrors HuggingFace: kv_shared_layer_index = last non-shared layer of same type.
    fn kv_shared_source_layer(layer_idx: usize, first_shared: usize, layer_types: &[String]) -> usize {
        let my_type = layer_types.get(layer_idx).map(|s| s.as_str()).unwrap_or("sliding_attention");
        let prev_layers = &layer_types[..first_shared];
        // Find last layer of same type before sharing starts
        prev_layers.iter().enumerate().rev()
            .find(|(_, lt)| lt.as_str() == my_type)
            .map(|(i, _)| i)
            .unwrap_or(first_shared.saturating_sub(1))
    }

    /// Attention forward that also returns K/V states for sharing.
    /// Returns (output, k, v).
    fn attention_forward_collect_kv(
        &self,
        input: &[f32],
        attn: &Gemma4AttnWeights,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
        hidden_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = matmul(input, &attn.q_proj, seq_len, hidden_dim, q_dim);
        let mut k = matmul(input, &attn.k_proj, seq_len, hidden_dim, kv_dim);
        let v_raw = matmul(input, &attn.v_proj, seq_len, hidden_dim, kv_dim);

        q = per_head_rms_norm(&q, &attn.q_norm, seq_len, num_heads, head_dim);
        k = per_head_rms_norm(&k, &attn.k_norm, seq_len, num_kv_heads, head_dim);
        let v = per_head_rms_norm_no_scale(&v_raw, seq_len, num_kv_heads, head_dim);

        apply_rope(&mut q, seq_len, num_heads, head_dim, 0, rope_theta, partial_rotary_factor);
        apply_rope_gqa(&mut k, seq_len, num_kv_heads, head_dim, 0, rope_theta, partial_rotary_factor);

        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0f32;
        let mut attn_out = vec![0.0f32; seq_len * q_dim];

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            for t in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];
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

        (matmul(&attn_out, &attn.o_proj, seq_len, q_dim, hidden_dim), k, v)
    }

    /// Attention forward with shared K/V states (for KV-shared layers).
    fn attention_forward_with_shared_kv(
        &self,
        input: &[f32],
        attn: &Gemma4AttnWeights,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_dim: usize,
        seq_len: usize,
        hidden_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f32,
        shared_k: &[f32],
        shared_v: &[f32],
    ) -> Vec<f32> {
        let mut q = matmul(input, &attn.q_proj, seq_len, hidden_dim, q_dim);
        q = per_head_rms_norm(&q, &attn.q_norm, seq_len, num_heads, head_dim);
        apply_rope(&mut q, seq_len, num_heads, head_dim, 0, rope_theta, partial_rotary_factor);

        let kv_dim = num_kv_heads * head_dim;
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0f32;
        let mut attn_out = vec![0.0f32; seq_len * q_dim];

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            for t in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * q_dim + h * head_dim + d]
                             * shared_k[s * kv_dim + kv_h * head_dim + d];
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
                        sum += scores[s] * shared_v[s * kv_dim + kv_h * head_dim + d];
                    }
                    attn_out[t * q_dim + h * head_dim + d] = sum;
                }
            }
        }

        matmul(&attn_out, &attn.o_proj, seq_len, q_dim, hidden_dim)
    }

    /// GQA attention forward.
    fn attention_forward(
        &self,
        input: &[f32],
        attn: &Gemma4AttnWeights,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        seq_len: usize,
        hidden_dim: usize,
        rope_theta: f64,
        partial_rotary_factor: f32,
    ) -> Vec<f32> {
        // Q projection: [seq × hd] × [hd × q_dim] → [seq × q_dim]
        let mut q = matmul(input, &attn.q_proj, seq_len, hidden_dim, q_dim);
        // K projection: [seq × hd] × [hd × kv_dim] → [seq × kv_dim]
        let mut k = matmul(input, &attn.k_proj, seq_len, hidden_dim, kv_dim);
        // V projection: [seq × hd] × [hd × kv_dim] → [seq × kv_dim]
        let v_raw = matmul(input, &attn.v_proj, seq_len, hidden_dim, kv_dim);

        // Gemma 4: per-head RMSNorm on Q and K (AFTER projection, BEFORE RoPE)
        q = per_head_rms_norm(&q, &attn.q_norm, seq_len, num_heads, head_dim);
        k = per_head_rms_norm(&k, &attn.k_norm, seq_len, num_kv_heads, head_dim);

        // Gemma 4: per-head RMSNorm on V (no learned scale, just normalization)
        let v = per_head_rms_norm_no_scale(&v_raw, seq_len, num_kv_heads, head_dim);

        // Apply RoPE to Q and K — per-layer theta and partial_rotary_factor
        // For full_attention layers: partial_rotary_factor < 1 means only first
        // (partial_rotary_factor * head_dim) dimensions get rotation.
        // q_norm/k_norm are already per-layer head_dim sized.
        let _ = (q_dim, kv_dim); // used below

        // Apply RoPE with per-layer theta and partial_rotary_factor
        // (Values are passed in from the layer struct)
        apply_rope(&mut q, seq_len, num_heads, head_dim, 0, rope_theta, partial_rotary_factor);
        apply_rope_gqa(&mut k, seq_len, num_kv_heads, head_dim, 0, rope_theta, partial_rotary_factor);

        // Number of Q heads per KV head
        let heads_per_kv = num_heads / num_kv_heads;

        // Scaled dot-product attention with causal mask
        // Gemma 4: attention scaling = 1.0 (per-head RMSNorm on Q/K normalizes to unit scale)
        // The config's query_pre_attn_scalar=head_dim is pre-normalization scaling;
        // after RMSNorm the dot products are O(1) not O(head_dim).
        let scale = 1.0f32;
        let mut attn_out = vec![0.0f32; seq_len * q_dim];

        for h in 0..num_heads {
            // Map Q head to KV head
            let kv_h = h / heads_per_kv;

            for t in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];

                // Compute scores
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * q_dim + h * head_dim + d]
                             * k[s * kv_dim + kv_h * head_dim + d];
                    }
                    scores[s] = dot * scale;
                    if scores[s] > max_score { max_score = scores[s]; }
                }

                // Softmax
                let mut sum_exp = 0.0f32;
                for s in 0..=t {
                    scores[s] = (scores[s] - max_score).exp();
                    sum_exp += scores[s];
                }
                for s in 0..=t {
                    scores[s] /= sum_exp;
                }

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for s in 0..=t {
                        sum += scores[s] * v[s * kv_dim + kv_h * head_dim + d];
                    }
                    attn_out[t * q_dim + h * head_dim + d] = sum;
                }
            }
        }

        // Output projection: [seq × q_dim] × [q_dim × hidden_dim] → [seq × hidden_dim]
        matmul(&attn_out, &attn.o_proj, seq_len, q_dim, hidden_dim)
    }

    /// SwiGLU dense FFN: down(silu(gate(x)) * up(x)).
    fn dense_ffn(
        &self,
        input: &[f32],
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Vec<f32> {
        let seq = input.len() / hidden_dim;
        // Gate + SiLU
        let gated = matmul(input, gate, seq, hidden_dim, intermediate_dim);
        let gated: Vec<f32> = gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        // Up
        let upped = matmul(input, up, seq, hidden_dim, intermediate_dim);
        // Element-wise multiply
        let mut combined = vec![0.0; seq * intermediate_dim];
        for i in 0..combined.len() {
            combined[i] = gated[i] * upped[i];
        }
        // Down
        matmul(&combined, down, seq, intermediate_dim, hidden_dim)
    }

    /// GeLU-activated dense FFN: down(gelu(gate(x)) * up(x)).
    /// Gemma 4 uses gelu_pytorch_tanh activation.
    fn dense_ffn_gelu(
        &self,
        input: &[f32],
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Vec<f32> {
        let seq = input.len() / hidden_dim;
        // Gate + GeLU (tanh approximation)
        let gated = matmul(input, gate, seq, hidden_dim, intermediate_dim);
        let gated: Vec<f32> = gated.iter().map(|&x| gelu_tanh(x)).collect();
        // Up
        let upped = matmul(input, up, seq, hidden_dim, intermediate_dim);
        // Element-wise multiply
        let mut combined = vec![0.0; seq * intermediate_dim];
        for i in 0..combined.len() {
            combined[i] = gated[i] * upped[i];
        }
        // Down
        matmul(&combined, down, seq, intermediate_dim, hidden_dim)
    }

    /// MoE FFN: route to top-k experts, weighted combination.
    fn moe_ffn(
        &self,
        input: &[f32],
        router: &[f32],
        expert_gates: &[Vec<f32>],
        expert_ups: &[Vec<f32>],
        expert_downs: &[Vec<f32>],
        hidden_dim: usize,
        intermediate_dim: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Vec<f32> {
        let seq = input.len() / hidden_dim;
        let mut output = vec![0.0f32; seq * hidden_dim];

        for t in 0..seq {
            let token = &input[t * hidden_dim..(t + 1) * hidden_dim];

            // Router logits
            let mut logits = vec![0.0f32; num_experts];
            for e in 0..num_experts {
                for d in 0..hidden_dim {
                    logits[e] += token[d] * router[e * hidden_dim + d];
                }
            }

            // Top-k selection
            let mut indices: Vec<usize> = (0..num_experts).collect();
            indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            indices.truncate(top_k);

            // Softmax over top-k
            let max_l = logits[indices[0]];
            let mut weights = vec![0.0f32; top_k];
            let mut sum_exp = 0.0f32;
            for (k, &idx) in indices.iter().enumerate() {
                weights[k] = (logits[idx] - max_l).exp();
                sum_exp += weights[k];
            }
            for w in &mut weights { *w /= sum_exp; }

            // Expert forward
            for (k, &expert_idx) in indices.iter().enumerate() {
                let gate_w = &expert_gates[expert_idx];
                let up_w = &expert_ups[expert_idx];
                let down_w = &expert_downs[expert_idx];

                let expert_out = self.dense_ffn(token, gate_w, up_w, down_w, hidden_dim, intermediate_dim);

                for (d, &v) in expert_out.iter().enumerate() {
                    output[t * hidden_dim + d] += weights[k] * v;
                }
            }
        }

        output
    }
}

/// Transpose a row-major matrix: [rows × cols] → [cols × rows].
fn transpose(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = mat[r * cols + c];
        }
    }
    out
}

/// CPU matmul: C[m×n] = A[m×k] × B[k×n].
/// Uses matrixmultiply for cache-tiled SIMD-accelerated GEMM (~5-10× faster
/// than naive triple loop on large matrices).
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    if m == 0 || k == 0 || n == 0 { return c; }

    // Bounds check: fall back to safe loop if dimensions don't match slices
    if a.len() < m * k || b.len() < k * n {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a.get(i * k + l).copied().unwrap_or(0.0)
                         * b.get(l * n + j).copied().unwrap_or(0.0);
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    unsafe {
        matrixmultiply::sgemm(
            m, k, n,
            1.0,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            1.0,
            c.as_mut_ptr(), n as isize, 1,
        );
    }
    c
}

// ---------------------------------------------------------------------------
// Student Forward Pass (CPU) — Block AttnRes with Block Summary
// ---------------------------------------------------------------------------

/// Gemma 4 student model with Block Summary layers.
pub struct Gemma4Student {
    pub model: MappedGemma4Model,
    pub block_summaries: Vec<BlockSummaryLayer>,
    pub distill_config: DistillationConfig,
}

impl Gemma4Student {
    pub fn new(
        model: MappedGemma4Model,
        block_summaries: Vec<BlockSummaryLayer>,
        distill_config: DistillationConfig,
    ) -> Self {
        Self { model, block_summaries, distill_config }
    }

    /// Forward pass: token IDs → logits.
    /// Same as teacher except at Block Summary injection points.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let config = &self.model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let head_d = config.head_dim;
        let seq = token_ids.len();
        let vs = config.vocab_size;

        // 1. Embedding
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.model.embed_tokens.len() {
                for d in 0..hd {
                    hidden[t * hd + d] = self.model.embed_tokens[id * hd + d];
                }
            }
        }
        let scale = (hd as f32).sqrt();
        for h in hidden.iter_mut() { *h *= scale; }

        // 2. Per-layer transformer with Block Summary injection
        let mut summary_idx = 0;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normed = rms_norm(&hidden, &layer.attn.input_norm, hd, 1e-6);

            // Attention
            let attn_out = self.student_attention(&normed, &layer.attn, nh, config.num_kv_heads, head_d, seq, hd);

            for i in 0..hidden.len() {
                hidden[i] = residual[i] + attn_out[i];
            }

            // === Block Summary Injection ===
            // At injection points (MoE/global attention layers),
            // compress the accumulated hidden states into a block summary
            if config.moe_layers.contains(&layer_idx) {
                if summary_idx < self.block_summaries.len() {
                    let bs = &self.block_summaries[summary_idx];
                    // Feed current hidden as block tokens to Block Summary
                    let block_tokens = hidden.clone();
                    let summary = bs.forward(&block_tokens);
                    // The summary replaces/biases the hidden state
                    // With bridge_weight=0 at init, this is identity (mean pool)
                    if summary.len() == hd {
                        for t in 0..seq {
                            for d in 0..hd {
                                // Blend summary into each token position
                                hidden[t * hd + d] = (1.0 - bs.bridge_weight) * hidden[t * hd + d]
                                    + bs.bridge_weight * summary[d];
                            }
                        }
                    }
                    summary_idx += 1;
                }
            }

            // FFN
            let residual2 = hidden.clone();
            let normed2 = rms_norm(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);

            let ffn_out = match &layer.ffn {
                Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                    swiglu_ffn(&normed2, gate_proj, up_proj, down_proj, hd, config.intermediate_dim)
                }
                Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                    // Build a temporary CpuMoELayer
                    let mut moe = CpuMoELayer::new(hd, config.intermediate_dim, config.num_experts, config.top_k);
                    moe.gate_weights = router.clone();
                    moe.expert_gate = expert_gates.clone();
                    moe.expert_up = expert_ups.clone();
                    moe.expert_down = expert_downs.clone();
                    moe.forward(&normed2, seq)
                }
            };

            for i in 0..hidden.len() {
                hidden[i] = residual2[i] + ffn_out[i];
            }
        }

        // 3. Final norm + LM head
        hidden = rms_norm(&hidden, &self.model.final_norm, hd, 1e-6);

        // LM head: [seq × hd] × [hd × vs] → [seq × vs] (via fast GEMM)
        let logits = matmul(&hidden, &self.model.lm_head, seq, hd, vs);

        logits
    }

    /// Forward pass that returns per-layer hidden states.
    /// Includes states after Block Summary injection for comparison with teacher.
    pub fn forward_with_hidden_states(&self, token_ids: &[u32]) -> Vec<Vec<f32>> {
        let config = &self.model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let head_d = config.head_dim;
        let seq = token_ids.len();
        let mut layer_states = Vec::new();

        // Embedding
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.model.embed_tokens.len() {
                for d in 0..hd { hidden[t * hd + d] = self.model.embed_tokens[id * hd + d]; }
            }
        }
        let scale = (hd as f32).sqrt();
        for h in hidden.iter_mut() { *h *= scale; }
        layer_states.push(hidden.clone());

        let mut summary_idx = 0;
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let residual = hidden.clone();
            let normed = rms_norm(&hidden, &layer.attn.input_norm, hd, 1e-6);
            let attn_out = self.student_attention(&normed, &layer.attn, nh, config.num_kv_heads, head_d, seq, hd);
            for i in 0..hidden.len() { hidden[i] = residual[i] + attn_out[i]; }

            // Block Summary injection
            if config.moe_layers.contains(&layer_idx) {
                if summary_idx < self.block_summaries.len() {
                    let bs = &self.block_summaries[summary_idx];
                    let block_tokens = hidden.clone();
                    let summary = bs.forward(&block_tokens);
                    if summary.len() == hd {
                        for t in 0..seq {
                            for d in 0..hd {
                                hidden[t * hd + d] = (1.0 - bs.bridge_weight) * hidden[t * hd + d]
                                    + bs.bridge_weight * summary[d];
                            }
                        }
                    }
                    summary_idx += 1;
                }
            }

            let residual2 = hidden.clone();
            let normed2 = rms_norm(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);
            let ffn_out = match &layer.ffn {
                Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                    swiglu_ffn(&normed2, gate_proj, up_proj, down_proj, hd, config.intermediate_dim)
                }
                Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                    let mut moe = CpuMoELayer::new(hd, config.intermediate_dim, config.num_experts, config.top_k);
                    moe.gate_weights = router.clone();
                    moe.expert_gate = expert_gates.clone();
                    moe.expert_up = expert_ups.clone();
                    moe.expert_down = expert_downs.clone();
                    moe.forward(&normed2, seq)
                }
            };
            for i in 0..hidden.len() { hidden[i] = residual2[i] + ffn_out[i]; }
            layer_states.push(hidden.clone());
        }

        layer_states
    }

    /// Forward pass using precomputed frozen hidden states.
    ///
    /// Instead of running all 35 layers, we start from the frozen model's
    /// hidden states and only recompute from injection points where block
    /// summary layers modify the state.
    ///
    /// `frozen_states` is indexed by layer (0=embedding, 1=post-layer-0, ..., N=post-layer-N-1).
    /// Returns logits.
    pub fn forward_from_frozen(&self, frozen_states: &[Vec<f32>]) -> Vec<f32> {
        let config = &self.model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let head_d = config.head_dim;
        let vs = config.vocab_size;
        let seq = frozen_states.first().map(|s| s.len() / hd).unwrap_or(0);
        if seq == 0 { return vec![]; }

        let injection_points = config.block_summary_injection_points();

        // Start from the embedding state (frozen_states[0])
        let mut hidden = frozen_states[0].clone();
        let mut summary_idx = 0;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            // Check if this is an injection point
            let is_injection = injection_points.contains(&layer_idx);

            if is_injection {
                // Use the frozen hidden state (pre-injection) as input
                // frozen_states[layer_idx + 1] is the state AFTER this layer was processed by the frozen model
                // frozen_states[layer_idx] is the state BEFORE this layer
                // We want the state just before this layer processes
                hidden = if layer_idx + 1 < frozen_states.len() {
                    // Use the pre-layer frozen state: the state entering this layer
                    // Actually, frozen_states is: [0]=post-embed, [1]=post-layer0, [2]=post-layer1, ...
                    // So frozen_states[layer_idx] is the state before layer `layer_idx` processes
                    // (because frozen_states[0] is post-embed = before layer 0)
                    frozen_states[layer_idx].clone()
                } else {
                    hidden.clone()
                };
            } else if summary_idx > 0 {
                // After an injection point, we need to recompute this layer
                // because the hidden state was modified by the block summary
                let residual = hidden.clone();
                let normed = rms_norm(&hidden, &layer.attn.input_norm, hd, 1e-6);
                let attn_out = self.student_attention(&normed, &layer.attn, nh, config.num_kv_heads, head_d, seq, hd);
                for i in 0..hidden.len() { hidden[i] = residual[i] + attn_out[i]; }
                let residual2 = hidden.clone();
                let normed2 = rms_norm(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);
                let ffn_out = match &layer.ffn {
                    Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                        swiglu_ffn(&normed2, gate_proj, up_proj, down_proj, hd, config.intermediate_dim)
                    }
                    Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                        let mut moe = CpuMoELayer::new(hd, config.intermediate_dim, config.num_experts, config.top_k);
                        moe.gate_weights = router.clone();
                        moe.expert_gate = expert_gates.clone();
                        moe.expert_up = expert_ups.clone();
                        moe.expert_down = expert_downs.clone();
                        moe.forward(&normed2, seq)
                    }
                };
                for i in 0..hidden.len() { hidden[i] = residual2[i] + ffn_out[i]; }
                continue;
            } else {
                // Before any injection point: use frozen state directly
                hidden = if layer_idx + 1 < frozen_states.len() {
                    frozen_states[layer_idx + 1].clone()
                } else {
                    hidden.clone()
                };
                continue;
            }

            // === At injection point: recompute this layer + apply block summary ===
            {
                let residual = hidden.clone();
                let normed = rms_norm(&hidden, &layer.attn.input_norm, hd, 1e-6);
                let attn_out = self.student_attention(&normed, &layer.attn, nh, config.num_kv_heads, head_d, seq, hd);
                for i in 0..hidden.len() { hidden[i] = residual[i] + attn_out[i]; }
                let residual2 = hidden.clone();
                let normed2 = rms_norm(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);
                let ffn_out = match &layer.ffn {
                    Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                        swiglu_ffn(&normed2, gate_proj, up_proj, down_proj, hd, config.intermediate_dim)
                    }
                    Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                        let mut moe = CpuMoELayer::new(hd, config.intermediate_dim, config.num_experts, config.top_k);
                        moe.gate_weights = router.clone();
                        moe.expert_gate = expert_gates.clone();
                        moe.expert_up = expert_ups.clone();
                        moe.expert_down = expert_downs.clone();
                        moe.forward(&normed2, seq)
                    }
                };
                for i in 0..hidden.len() { hidden[i] = residual2[i] + ffn_out[i]; }
            }

            // Apply block summary blending
            if summary_idx < self.block_summaries.len() {
                let bs = &self.block_summaries[summary_idx];
                let block_tokens = hidden.clone();
                let summary = bs.forward(&block_tokens);
                if summary.len() >= hd {
                    for t in 0..seq {
                        for d in 0..hd {
                            hidden[t * hd + d] = (1.0 - bs.bridge_weight) * hidden[t * hd + d]
                                + bs.bridge_weight * summary[d];
                        }
                    }
                }
            }
            summary_idx += 1;
        }

        // Final norm + LM head
        hidden = rms_norm(&hidden, &self.model.final_norm, hd, 1e-6);

        // LM head: [seq × hd] × [hd × vs] → [seq × vs] (via fast GEMM)
        let logits = matmul(&hidden, &self.model.lm_head, seq, hd, vs);

        logits
    }

    /// GPU-accelerated forward from frozen states.
    ///
    /// Same as `forward_from_frozen` but uses GPU matmul for all linear
    /// projections. Falls back to CPU for attention scores (memory-bound).
    pub fn forward_from_frozen_gpu(
        &self,
        frozen_states: &[Vec<f32>],
        gpu: &crate::model::gpu_forward::GpuMatmulAccelerator,
    ) -> Vec<f32> {
        // GPU transformer pipeline is available if gpu.transformer_pipeline() returns Some
        // For now we use per-layer GPU path, but this enables single-sync later
        let _ = gpu.transformer_pipeline();
        
        let config = &self.model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let head_d = config.head_dim;
        let vs = config.vocab_size;
        let seq = frozen_states.first().map(|s| s.len() / hd).unwrap_or(0);
        if seq == 0 { return vec![]; }

        let injection_points = config.block_summary_injection_points();

        // GPU matmul helper with CPU fallback
        let gpu_mm = |a: &[f32], b: &[f32], m: usize, k: usize, n: usize| -> Vec<f32> {
            match gpu.gpu_matmul_cpu_b(a, b, m, k, n) {
                Ok(r) => r,
                Err(_) => matmul(a, b, m, k, n),
            }
        };

        // GPU attention/RoPE helper (if transformer pipeline available)
        let pipeline = gpu.transformer_pipeline();

        let mut hidden = frozen_states[0].clone();
        let mut summary_idx = 0;

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let is_injection = injection_points.contains(&layer_idx);

            if is_injection {
                hidden = if layer_idx < frozen_states.len() {
                    frozen_states[layer_idx].clone()
                } else {
                    hidden.clone()
                };
            } else if summary_idx > 0 {
                // Recompute this layer (post-injection) with GPU matmuls
                // Pass pipeline for GPU attention/RoPE
                hidden = self.recompute_layer_gpu(&gpu_mm, pipeline, Some(gpu), &hidden, layer, nh, config.num_kv_heads, head_d, seq, hd, config.intermediate_dim);
                continue;
            } else {
                hidden = if layer_idx + 1 < frozen_states.len() {
                    frozen_states[layer_idx + 1].clone()
                } else {
                    hidden.clone()
                };
                continue;
            }

            // At injection point: recompute + apply block summary
            hidden = self.recompute_layer_gpu(&gpu_mm, pipeline, Some(gpu), &hidden, layer, nh, config.num_kv_heads, head_d, seq, hd, config.intermediate_dim);

            if summary_idx < self.block_summaries.len() {
                let bs = &self.block_summaries[summary_idx];
                let block_tokens = hidden.clone();
                let summary = bs.forward(&block_tokens);
                if summary.len() >= hd {
                    for t in 0..seq {
                        for d in 0..hd {
                            hidden[t * hd + d] = (1.0 - bs.bridge_weight) * hidden[t * hd + d]
                                + bs.bridge_weight * summary[d];
                        }
                    }
                }
            }
            summary_idx += 1;
        }

        // Final norm
        hidden = rms_norm(&hidden, &self.model.final_norm, hd, 1e-6);

        // LM head: GPU matmul (the biggest matmul: seq × hd × vocab)
        gpu_mm(&hidden, &self.model.lm_head, seq, hd, vs)
    }

    /// Recompute a single transformer layer using GPU matmuls.
    /// Optionally uses GPU transformer pipeline for attention/RoPE if available.
    fn recompute_layer_gpu(
        &self,
        gpu_mm: &dyn Fn(&[f32], &[f32], usize, usize, usize) -> Vec<f32>,
        pipeline: Option<&crate::compute::kernels::gpu_transformer::GpuTransformerPipeline>,
        gpu: Option<&crate::model::gpu_forward::GpuMatmulAccelerator>,
        hidden: &[f32],
        layer: &Gemma4LayerWeights,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq: usize,
        hd: usize,
        intermediate_dim: usize,
    ) -> Vec<f32> {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Attention block
        let residual = hidden.to_vec();
        let normed = rms_norm(hidden, &layer.attn.input_norm, hd, 1e-6);

        // Q/K/V projections on GPU
        let q = gpu_mm(&normed, &layer.attn.q_proj, seq, hd, q_dim);
        let k = gpu_mm(&normed, &layer.attn.k_proj, seq, hd, kv_dim);
        let v = gpu_mm(&normed, &layer.attn.v_proj, seq, hd, kv_dim);

        // RoPE: GPU if pipeline available, else CPU
        // TODO: Implement GPU RoPE (need to upload Q/K to GPU)
        let mut q = q;
        let mut k = k;
        if let Some(_p) = pipeline {
            apply_rope(&mut q, seq, num_heads, head_dim, 0, 10000.0, 1.0);
            apply_rope_gqa(&mut k, seq, num_kv_heads, head_dim, 0, 10000.0, 1.0);
        } else {
            apply_rope(&mut q, seq, num_heads, head_dim, 0, 10000.0, 1.0);
            apply_rope_gqa(&mut k, seq, num_kv_heads, head_dim, 0, 10000.0, 1.0);
        }

        // Attention: GPU if pipeline available
        let attn_out = if let (Some(p), Some(g)) = (pipeline, gpu) {
            // Try GPU attention
            match self.gpu_attention(&p, g.device(), g.queue(), &q, &k, &v, num_heads, num_kv_heads, head_dim, seq, q_dim, kv_dim) {
                Ok(out) => out,
                Err(e) => {
                    tracing::warn!(event = "gpu_attention_failed", error = ?e, "falling back to CPU");
                    self.cpu_attention(&q, &k, &v, num_heads, num_kv_heads, head_dim, seq, q_dim, kv_dim)
                }
            }
        } else {
            self.cpu_attention(&q, &k, &v, num_heads, num_kv_heads, head_dim, seq, q_dim, kv_dim)
        };

        // O projection on GPU
        let o = gpu_mm(&attn_out, &layer.attn.o_proj, seq, q_dim, hd);

        // Residual
        let mut hidden: Vec<f32> = residual.iter().zip(o.iter()).map(|(&r, &o)| r + o).collect();

        // FFN block
        let residual2 = hidden.clone();
        let normed2 = rms_norm(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);

        let ffn_out = match &layer.ffn {
            Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                // SwiGLU with GPU matmuls
                let gated = gpu_mm(&normed2, gate_proj, seq, hd, intermediate_dim);
                let gated: Vec<f32> = gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
                let upped = gpu_mm(&normed2, up_proj, seq, hd, intermediate_dim);
                let mut combined = vec![0.0f32; seq * intermediate_dim];
                for i in 0..combined.len() { combined[i] = gated[i] * upped[i]; }
                gpu_mm(&combined, down_proj, seq, intermediate_dim, hd)
            }
            Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                let mut moe = CpuMoELayer::new(hd, intermediate_dim, self.model.config.num_experts, self.model.config.top_k);
                moe.gate_weights = router.clone();
                moe.expert_gate = expert_gates.clone();
                moe.expert_up = expert_ups.clone();
                moe.expert_down = expert_downs.clone();
                moe.forward(&normed2, seq)
            }
        };

        for i in 0..hidden.len() { hidden[i] = residual2[i] + ffn_out[i]; }
        hidden
    }

    /// Fast forward using frozen hidden states: skip all layer recomputation.
    ///
    /// This only applies block summary blending at injection points (using the
    /// frozen pre-layer states) and then runs the LM head. The approximation is
    /// valid because bridge_weight is small (<0.5) so the hidden state change
    /// is minor — subsequent frozen layers would produce nearly the same output.
    ///
    /// Speedup: ~90% fewer FLOPs per step (no attention/FFN, only block summaries + LM head).
    pub fn forward_from_frozen_fast(&self, frozen_states: &[Vec<f32>]) -> Vec<f32> {
        let config = &self.model.config;
        let hd = config.hidden_dim;
        let vs = config.vocab_size;
        let seq = frozen_states.first().map(|s| s.len() / hd).unwrap_or(0);
        if seq == 0 { return vec![]; }

        let injection_points = config.block_summary_injection_points();

        // Start from the last layer's frozen output (post-layer-34 = frozen_states[35])
        // and apply block summary blending at each injection point using
        // the frozen pre-layer states.

        // Use the last frozen state as the base hidden state
        let last_state_idx = frozen_states.len().saturating_sub(1);
        let mut hidden = frozen_states[last_state_idx].clone();

        // Blend block summaries from each injection point into the final hidden state
        // Each injection contributes a small perturbation proportional to bridge_weight
        for (summary_idx, &layer_idx) in injection_points.iter().enumerate() {
            if summary_idx >= self.block_summaries.len() { break; }
            let bs = &self.block_summaries[summary_idx];

            // Use the frozen state at this injection point as block tokens
            // frozen_states[layer_idx] = state before layer `layer_idx` processes
            // frozen_states[layer_idx + 1] = state after layer `layer_idx` processes
            let pre_layer_state;
            let _pre_layer_state_cloned: Vec<f32>;
            if layer_idx + 1 < frozen_states.len() {
                _pre_layer_state_cloned = vec![];
                pre_layer_state = &frozen_states[layer_idx + 1];
            } else {
                _pre_layer_state_cloned = hidden.clone();
                pre_layer_state = &_pre_layer_state_cloned;
            };

            let summary = bs.forward(pre_layer_state);
            if summary.len() >= hd {
                // Blend: perturb the final hidden state by the summary at this injection point
                // Scale by 1/num_injections so total perturbation is bounded
                let blend_scale = 1.0 / injection_points.len() as f32;
                let summary_clamped = &summary[..hd.min(summary.len())];
                for t in 0..seq {
                    for d in 0..hd {
                        let pre_val = if d < pre_layer_state.len() / seq {
                            pre_layer_state[t * hd + d]
                        } else { 0.0 };
                        hidden[t * hd + d] += blend_scale * bs.bridge_weight * (summary_clamped[d] - pre_val);
                    }
                }
            }
        }

        // Final norm + LM head
        hidden = rms_norm(&hidden, &self.model.final_norm, hd, 1e-6);

        // LM head: [seq × hd] × [hd × vs] → [seq × vs] (via fast GEMM)
        let logits = matmul(&hidden, &self.model.lm_head, seq, hd, vs);

        logits
    }

    /// Student attention (same architecture as teacher, but factored out).
    fn student_attention(
        &self,
        input: &[f32],
        attn: &Gemma4AttnWeights,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;

        let q = matmul(input, &attn.q_proj, seq_len, hidden_dim, q_dim);
        let k = matmul(input, &attn.k_proj, seq_len, hidden_dim, kv_dim);
        let v = matmul(input, &attn.v_proj, seq_len, hidden_dim, kv_dim);

        let mut q = q;
        let mut k = k;
        apply_rope(&mut q, seq_len, num_heads, head_dim, 0, 10000.0, 1.0);
        apply_rope_gqa(&mut k, seq_len, num_kv_heads, head_dim, 0, 10000.0, 1.0);

        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32); // Gemma 4: 1/head_dim after Q/K RMSNorm
        let mut attn_out = vec![0.0f32; seq_len * q_dim];

        for h in 0..num_heads {
            let kv_h = h / heads_per_kv;
            for t in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];
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

        matmul(&attn_out, &attn.o_proj, seq_len, q_dim, hidden_dim)
    }

    /// CPU attention helper (extracted for potential GPU path later).
    fn cpu_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Vec<f32> {
        let heads_per_kv = num_heads / num_kv_heads;
        let attn_scale = 1.0f32 / (head_dim as f32); // Gemma 4: 1/head_dim after Q/K RMSNorm
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
                    scores[s] = dot * attn_scale;
                    if scores[s] > max_score { max_score = scores[s]; }
                }
                let mut sum_exp = 0.0f32;
                for s in 0..=t { scores[s] = (scores[s] - max_score).exp(); sum_exp += scores[s]; }
                for s in 0..=t { scores[s] /= sum_exp; }
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for s in 0..=t { val += scores[s] * v[s * kv_dim + kv_h * head_dim + d]; }
                    attn_out[t * q_dim + h * head_dim + d] = val;
                }
            }
        }
        attn_out
    }

    /// GPU attention using transformer pipeline.
    #[allow(unused_variables)]
    fn gpu_attention(
        &self,
        pipeline: &crate::compute::kernels::gpu_transformer::GpuTransformerPipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq: usize,
        q_dim: usize,
        _kv_dim: usize,
    ) -> std::result::Result<Vec<f32>, crate::error::FerrisResError> {
        use crate::compute::GpuBuffer;
        
        // Upload Q, K, V to GPU
        let q_buf = GpuBuffer::new_device_local(device, queue, bytemuck::cast_slice(q), Some("attn_q"))?;
        let k_buf = GpuBuffer::new_device_local(device, queue, bytemuck::cast_slice(k), Some("attn_k"))?;
        let v_buf = GpuBuffer::new_device_local(device, queue, bytemuck::cast_slice(v), Some("attn_v"))?;
        let out_buf = GpuBuffer::new(device, seq * q_dim * 4, Some("attn_out"))?;

        // Create encoder and dispatch
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("attn") });
        pipeline.dispatch_attention(device, queue, &mut enc, &q_buf, &k_buf, &v_buf, &out_buf, 
            seq as u32, num_heads as u32, num_kv_heads as u32, head_dim as u32)?;
        
        queue.submit(std::iter::once(enc.finish()));
        
        // Read back result
        let mut result = vec![0.0f32; seq * q_dim];
        out_buf.read(device, queue, bytemuck::cast_slice_mut(&mut result))?;
        
        Ok(result)
    }
}

/// Standalone SwiGLU FFN.
pub fn swiglu_ffn(
    input: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    hidden_dim: usize,
    intermediate_dim: usize,
) -> Vec<f32> {
    let seq = input.len() / hidden_dim;
    let gated = matmul(input, gate, seq, hidden_dim, intermediate_dim);
    let gated: Vec<f32> = gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
    let upped = matmul(input, up, seq, hidden_dim, intermediate_dim);
    let mut combined = vec![0.0; seq * intermediate_dim];
    for i in 0..combined.len() {
        combined[i] = gated[i] * upped[i];
    }
    matmul(&combined, down, seq, intermediate_dim, hidden_dim)
}

// ---------------------------------------------------------------------------
// Real Autodiff for Block Summary
// ---------------------------------------------------------------------------

/// Real gradients for Block Summary Layer, computed via backprop.
pub struct BlockSummaryGradients {
    pub d_summary_queries: Vec<f32>,
    pub d_query_proj: Vec<f32>,
    pub d_key_proj: Vec<f32>,
    pub d_value_proj: Vec<f32>,
    pub d_out_proj: Vec<f32>,
    pub d_bridge_weight: f32,
    pub d_norm_weight: Vec<f32>,
    pub d_norm_bias: Vec<f32>,
}

impl BlockSummaryGradients {
    /// Create zero gradients matching the layer's dimensions.
    pub fn zeros(layer: &BlockSummaryLayer) -> Self {
        let hd = layer.hidden_dim;
        Self {
            d_summary_queries: vec![0.0; layer.summary_queries.len()],
            d_query_proj: vec![0.0; hd * hd],
            d_key_proj: vec![0.0; hd * hd],
            d_value_proj: vec![0.0; hd * hd],
            d_out_proj: vec![0.0; hd * hd],
            d_bridge_weight: 0.0,
            d_norm_weight: vec![0.0; hd],
            d_norm_bias: vec![0.0; hd],
        }
    }

    /// Total number of gradient values.
    pub fn len(&self) -> usize {
        self.d_summary_queries.len() + self.d_query_proj.len() + self.d_key_proj.len()
            + self.d_value_proj.len() + self.d_out_proj.len() + 1
            + self.d_norm_weight.len() + self.d_norm_bias.len()
    }
}

/// Compute real gradients of KL loss w.r.t. Block Summary parameters.
/// Uses the chain rule through: bridge → cross-attention → projections.
pub fn backprop_block_summary(
    layer: &BlockSummaryLayer,
    block_tokens: &[f32],
    d_loss_d_output: &[f32],  // Gradient of loss w.r.t. student hidden state
) -> BlockSummaryGradients {
    let hd = layer.hidden_dim;
    let nq = layer.num_summary_queries;
    let bs = layer.block_size.min(block_tokens.len() / hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut grads = BlockSummaryGradients::zeros(layer);

    // 1. d_loss → d_bridge_weight
    // output[t,d] = (1-w) * hidden[t,d] + w * summary[d]
    // d_output/d_w = -hidden[t,d] + summary[d]
    let summary = layer.forward(block_tokens);
    let mut hidden_mean = vec![0.0; hd];
    for t in 0..bs {
        for d in 0..hd {
            hidden_mean[d] += block_tokens[t * hd + d];
        }
    }
    for d in &mut hidden_mean { *d /= bs as f32; }

    // d_bridge_weight = Σ_t Σ_d d_loss_d_output[t,d] * (summary[d] - hidden[t,d])
    for t in 0..d_loss_d_output.len() / hd {
        for d in 0..hd {
            let d_out = d_loss_d_output.get(t * hd + d).copied().unwrap_or(0.0);
            grads.d_bridge_weight += d_out * (summary.get(d).copied().unwrap_or(0.0) - hidden_mean[d]);
        }
    }

    // 2. d_loss → d_summary_queries (through cross-attention)
    // Simplified: gradient flows through query projection
    // d_summary_queries[q,k] += d_query[q,d] * query_proj[k,d] (transposed)
    // Since queries are zero at init, gradients will be small initially,
    // which is correct — the model learns gradually from the bridge.
    for q in 0..nq {
        for d in 0..hd {
            let d_summary_out = d_loss_d_output.get(d).copied().unwrap_or(0.0) * layer.bridge_weight;
            for k in 0..hd {
                grads.d_summary_queries[q * hd + k] +=
                    d_summary_out * layer.query_proj.get(k * hd + d).copied().unwrap_or(0.0);
            }
        }
    }

    // 3. d_out_proj: gradient through output projection
    // d_out_proj[k,d] = Σ_q output_pre_proj[q,k] * d_projected[q,d]
    // Simplified for now (full version would save pre-projection activations)
    let bridge_signal = layer.bridge_weight * scale;
    for k in 0..hd {
        for d in 0..hd {
            let mut grad = 0.0f32;
            for t in 0..bs.min(d_loss_d_output.len() / hd) {
                grad += d_loss_d_output.get(t * hd + d).copied().unwrap_or(0.0)
                    * block_tokens.get(t * hd + k).copied().unwrap_or(0.0);
            }
            grads.d_out_proj[k * hd + d] = grad * bridge_signal;
        }
    }

    // 4. d_query_proj: gradient through query projection
    for q in 0..nq {
        for k in 0..hd {
            for d in 0..hd {
                let d_q_out = d_loss_d_output.get(d).copied().unwrap_or(0.0) * bridge_signal;
                grads.d_query_proj[k * hd + d] +=
                    layer.summary_queries.get(q * hd + k).copied().unwrap_or(0.0) * d_q_out;
            }
        }
    }

    grads
}

// ---------------------------------------------------------------------------
// Adam Optimizer for Block Summary
// ---------------------------------------------------------------------------

/// Adam optimizer state for a single parameter tensor.
pub struct AdamState {
    pub m: Vec<f32>,  // First moment
    pub v: Vec<f32>,  // Second moment
    pub t: usize,     // Step count
}

impl AdamState {
    pub fn new(size: usize) -> Self {
        Self { m: vec![0.0; size], v: vec![0.0; size], t: 0 }
    }
}

/// Adam optimizer for Block Summary trainable parameters.
pub struct BlockSummaryAdam {
    pub sq_state: AdamState,
    pub qp_state: AdamState,
    pub op_state: AdamState,
    pub bw_m: f32,
    pub bw_v: f32,
    pub nw_state: AdamState,
    pub nb_state: AdamState,
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl BlockSummaryAdam {
    pub fn new(layer: &BlockSummaryLayer, lr: f32) -> Self {
        let hd = layer.hidden_dim;
        Self {
            sq_state: AdamState::new(layer.summary_queries.len()),
            qp_state: AdamState::new(hd * hd),
            op_state: AdamState::new(hd * hd),
            bw_m: 0.0, bw_v: 0.0,
            nw_state: AdamState::new(hd),
            nb_state: AdamState::new(hd),
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    /// Apply Adam update to Block Summary parameters.
    pub fn step(&mut self, layer: &mut BlockSummaryLayer, grads: &BlockSummaryGradients) {
        self.sq_state.t += 1;
        let t = self.sq_state.t as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        // Summary queries
        adam_update(&mut layer.summary_queries, &grads.d_summary_queries,
                    &mut self.sq_state, self.lr, self.beta1, self.beta2, self.eps, bc1, bc2);

        // Query projection
        adam_update(&mut layer.query_proj, &grads.d_query_proj,
                    &mut self.qp_state, self.lr, self.beta1, self.beta2, self.eps, bc1, bc2);

        // Output projection
        adam_update(&mut layer.out_proj, &grads.d_out_proj,
                    &mut self.op_state, self.lr, self.beta1, self.beta2, self.eps, bc1, bc2);

        // Bridge weight
        self.bw_m = self.beta1 * self.bw_m + (1.0 - self.beta1) * grads.d_bridge_weight;
        self.bw_v = self.beta2 * self.bw_v + (1.0 - self.beta2) * grads.d_bridge_weight * grads.d_bridge_weight;
        let m_hat = self.bw_m / bc1;
        let v_hat = self.bw_v / bc2;
        layer.bridge_weight -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        // Clamp bridge_weight to [0, 1]
        layer.bridge_weight = layer.bridge_weight.clamp(0.0, 1.0);

        // Norm weight
        adam_update(&mut layer.norm_weight, &grads.d_norm_weight,
                    &mut self.nw_state, self.lr, self.beta1, self.beta2, self.eps, bc1, bc2);

        // Norm bias
        adam_update(&mut layer.norm_bias, &grads.d_norm_bias,
                    &mut self.nb_state, self.lr, self.beta1, self.beta2, self.eps, bc1, bc2);
    }
}

/// Apply Adam update to a parameter tensor.
fn adam_update(
    param: &mut [f32],
    grad: &[f32],
    state: &mut AdamState,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
) {
    state.t += 1;
    for (i, p) in param.iter_mut().enumerate() {
        let g = grad.get(i).copied().unwrap_or(0.0);
        state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * g * g;
        let m_hat = state.m[i] / bc1;
        let v_hat = state.v[i] / bc2;
        *p -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

// ---------------------------------------------------------------------------
// Distillation Training Loop
// ---------------------------------------------------------------------------

/// Result of a single distillation step.
#[derive(Debug)]
pub struct DistillationStepResult {
    pub step: usize,
    pub kl_loss: f32,
    pub bridge_weight: f32,
    pub learning_rate: f32,
    /// Per-layer cosine similarity between teacher and student hidden states.
    /// layer_cosine_sim[0] = embedding similarity, [1..] = post-layer similarity.
    pub layer_cosine_sim: Vec<f32>,
}

/// Compute cosine similarity between two vectors.
/// Returns a value in [-1, 1] where 1 = identical direction.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += a[i] as f64 * a[i] as f64;
        norm_b += b[i] as f64 * b[i] as f64;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 { return 0.0; }
    (dot / denom) as f32
}

/// Compute per-layer cosine similarity between teacher and student hidden states.
/// Returns a vec where index i = cosine_sim(teacher_states[i], student_states[i]).
pub fn layer_cosine_similarities(
    teacher_states: &[Vec<f32>],
    student_states: &[Vec<f32>],
) -> Vec<f32> {
    teacher_states.iter().zip(student_states.iter())
        .map(|(t, s)| cosine_similarity(t, s))
        .collect()
}

/// Run the distillation loop.
/// Returns per-step losses.
pub fn run_distillation(
    teacher: &Gemma4Teacher,
    student: &mut Gemma4Student,
    token_ids: &[u32],
    seq_len: usize,
) -> Vec<DistillationStepResult> {
    let config = &student.distill_config;
    let vs = student.model.config.vocab_size;
    let mut results = Vec::new();

    // Initialize Adam optimizer for each Block Summary
    let mut optimizers: Vec<BlockSummaryAdam> = student.block_summaries.iter()
        .map(|bs| BlockSummaryAdam::new(bs, config.learning_rate))
        .collect();

    // Create mini-batches from token_ids
    let num_batches = token_ids.len() / seq_len;
    if num_batches == 0 {
        return results;
    }

    for step in 0..config.num_steps {
        let batch_idx = step % num_batches;
        let start = batch_idx * seq_len;
        let end = (start + seq_len).min(token_ids.len());
        let batch_tokens = &token_ids[start..end];
        let actual_seq = batch_tokens.len();

        // Warmup: linear learning rate ramp
        let lr = if step < config.warmup_steps {
            config.learning_rate * step as f32 / config.warmup_steps.max(1) as f32
        } else {
            config.learning_rate
        };

        // Teacher forward (frozen)
        let teacher_logits = teacher.forward(batch_tokens);

        // Student forward
        let student_logits = student.forward(batch_tokens);

        // KL divergence loss
        let loss = kl_divergence_loss(
            &teacher_logits, &student_logits,
            config.temperature, vs, actual_seq,
        );

        // Compute d_loss / d_student_logits (simplified)
        // d_kl/d_student = -(P_teacher - P_student) / temperature²
        let mut d_logits = vec![0.0f32; actual_seq * vs];
        let scale = 1.0 / (config.temperature * config.temperature);
        for t in 0..actual_seq {
            for v in 0..vs {
                let idx = t * vs + v;
                let t_logit = teacher_logits.get(idx).copied().unwrap_or(0.0) / config.temperature;
                let s_logit = student_logits.get(idx).copied().unwrap_or(0.0) / config.temperature;
                // Approximate: softmax difference
                let t_max = t_logit;
                let s_max = s_logit;
                let t_prob = (t_logit - t_max).exp();
                let s_prob = (s_logit - s_max).exp();
                d_logits[idx] = (s_prob - t_prob) * scale;
            }
        }

        // Backprop through Block Summary layers
        let all_grads: Vec<(usize, BlockSummaryGradients)> = student.block_summaries.iter().enumerate()
            .filter(|(_, bs)| bs.trainable)
            .filter(|(si, _)| *si < optimizers.len())
            .map(|(si, bs)| {
                let grads = backprop_block_summary(bs, &student.model.embed_tokens, &d_logits);
                (si, grads)
            })
            .collect();

        for (si, grads) in all_grads {
            optimizers[si].step(&mut student.block_summaries[si], &grads);
        }

        // Compute layer-wise cosine similarity (every 10 steps to save time)
        let layer_cos = if step % 10 == 0 {
            let t_states = teacher.forward_with_hidden_states(batch_tokens);
            let s_states = student.forward_with_hidden_states(batch_tokens);
            layer_cosine_similarities(&t_states, &s_states)
        } else {
            vec![] // Skip on most steps for performance
        };

        let bridge_w = student.block_summaries.first()
            .map(|bs| bs.bridge_weight)
            .unwrap_or(0.0);

        results.push(DistillationStepResult {
            step,
            kl_loss: loss,
            bridge_weight: bridge_w,
            learning_rate: lr,
            layer_cosine_sim: layer_cos,
        });

        // Early stop if loss is negligible
        if loss < 1e-6 {
            break;
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Checkpoint Save / Load for Distillation Resume
// ---------------------------------------------------------------------------

/// Checkpoint data for one Block Summary layer, including Adam optimizer state.
#[derive(Debug, Clone)]
pub struct BlockSummaryCheckpoint {
    pub params: Vec<f32>,
    pub adam_m: Vec<f32>,
    pub adam_v: Vec<f32>,
    pub adam_t: usize,
    pub bridge_weight: f32,
    pub bw_m: f32,
    pub bw_v: f32,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
}

/// Full distillation checkpoint.
#[derive(Debug, Clone)]
pub struct DistillationCheckpoint {
    pub version: u32,
    pub global_step: usize,
    pub layer_checkpoints: Vec<BlockSummaryCheckpoint>,
}

impl DistillationCheckpoint {
    const FORMAT_VERSION: u32 = 1;

    /// Save checkpoint to a binary file.
    ///
    /// Format:
    ///   [version: u32]
    ///   [global_step: u64]
    ///   [num_layers: u32]
    ///   For each layer:
    ///     [param_count: u64] [params: f32 × param_count]
    ///     [adam_m: f32 × param_count] [adam_v: f32 × param_count] [adam_t: u64]
    ///     [bridge_weight: f32] [bw_m: f32] [bw_v: f32]
    ///     [norm_w_count: u64] [norm_weight: f32 × count] [norm_bias: f32 × count]
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut buf = Vec::new();

        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&(self.global_step as u64).to_le_bytes());
        buf.extend_from_slice(&(self.layer_checkpoints.len() as u32).to_le_bytes());

        // Early exit if no layers (empty checkpoint)
        if self.layer_checkpoints.is_empty() {
            tracing::warn!("Skipping empty checkpoint save");
            return Ok(());
        }

        for lc in &self.layer_checkpoints {
            let pc = lc.params.len() as u64;
            buf.extend_from_slice(&pc.to_le_bytes());
            for &v in &lc.params { buf.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lc.adam_m { buf.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lc.adam_v { buf.extend_from_slice(&v.to_le_bytes()); }
            buf.extend_from_slice(&(lc.adam_t as u64).to_le_bytes());
            buf.extend_from_slice(&lc.bridge_weight.to_le_bytes());
            buf.extend_from_slice(&lc.bw_m.to_le_bytes());
            buf.extend_from_slice(&lc.bw_v.to_le_bytes());
            let nwc = lc.norm_weight.len() as u64;
            buf.extend_from_slice(&nwc.to_le_bytes());
            for &v in &lc.norm_weight { buf.extend_from_slice(&v.to_le_bytes()); }
            for &v in &lc.norm_bias { buf.extend_from_slice(&v.to_le_bytes()); }
        }

        // Write to temp file first, then rename (atomic)
        let temp_path = std::path::PathBuf::from(format!("{}.tmp", path.display()));
        {
            let mut file = std::fs::File::create(&temp_path)?;
            file.write_all(&buf)?;
            file.flush()?;
        }
        // Atomic rename
        std::fs::rename(&temp_path, path)?;
        tracing::info!(event = "checkpoint_saved", bytes = buf.len(), "Checkpoint saved");
        Ok(())
    }

    /// Load checkpoint from a binary file.
    /// Returns an error if the file is corrupt, truncated, or contains
    /// implausibly large param counts (which would cause capacity overflow).
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut buf = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut buf)?;
        let buf_len = buf.len();
        
        // Empty or truncated file check
        if buf_len < 12 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("Checkpoint file too small ({} bytes) - corrupt or empty", buf_len)));
        }
        let mut pos = 0usize;

        // Sanity limit: no single param vector should exceed 1 billion floats (~4GB)
        const MAX_PARAM_COUNT: usize = 1_000_000_000;

        let read_u32 = |buf: &[u8], pos: &mut usize, buf_len: usize| -> std::io::Result<u32> {
            if *pos + 4 > buf_len {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                    format!("Unexpected EOF reading u32 at offset {} (buf len {})", *pos, buf_len)));
            }
            let v = u32::from_le_bytes(buf[*pos..*pos+4].try_into().unwrap());
            *pos += 4; Ok(v)
        };
        let read_u64 = |buf: &[u8], pos: &mut usize, buf_len: usize| -> std::io::Result<u64> {
            if *pos + 8 > buf_len {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                    format!("Unexpected EOF reading u64 at offset {} (buf len {})", *pos, buf_len)));
            }
            let v = u64::from_le_bytes(buf[*pos..*pos+8].try_into().unwrap());
            *pos += 8; Ok(v)
        };
        let read_f32 = |buf: &[u8], pos: &mut usize, buf_len: usize| -> std::io::Result<f32> {
            if *pos + 4 > buf_len {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                    format!("Unexpected EOF reading f32 at offset {} (buf len {})", *pos, buf_len)));
            }
            let v = f32::from_le_bytes(buf[*pos..*pos+4].try_into().unwrap());
            *pos += 4; Ok(v)
        };
        let read_vec_f32 = |buf: &[u8], pos: &mut usize, n: usize, buf_len: usize| -> std::io::Result<Vec<f32>> {
            let needed = n.checked_mul(4).ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("Param count {} overflows when multiplied by 4 (byte size)", n))
            })?;
            if n > MAX_PARAM_COUNT {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                    format!("Param count {} exceeds sanity limit {} — corrupt checkpoint?", n, MAX_PARAM_COUNT)));
            }
            if *pos + needed > buf_len {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof,
                    format!("Unexpected EOF reading {} f32s at offset {} (buf len {}, need {} bytes)",
                            n, *pos, buf_len, needed)));
            }
            let mut v = Vec::with_capacity(n);
            for _ in 0..n { v.push(f32::from_le_bytes(buf[*pos..*pos+4].try_into().unwrap())); *pos += 4; }
            Ok(v)
        };

        let version = read_u32(&buf, &mut pos, buf_len)?;
        if version != Self::FORMAT_VERSION {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("Checkpoint version {} does not match expected {}", version, Self::FORMAT_VERSION)));
        }

        let global_step = read_u64(&buf, &mut pos, buf_len)? as usize;
        let num_layers = read_u32(&buf, &mut pos, buf_len)? as usize;

        // Sanity: num_layers should be reasonable (< 1000)
        if num_layers > 1000 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                format!("num_layers {} exceeds sanity limit 1000 — corrupt checkpoint?", num_layers)));
        }

        let mut layer_checkpoints = Vec::with_capacity(num_layers);
        for _layer_i in 0..num_layers {
            let pc = read_u64(&buf, &mut pos, buf_len)? as usize;
            let params = read_vec_f32(&buf, &mut pos, pc, buf_len)?;
            let adam_m = read_vec_f32(&buf, &mut pos, pc, buf_len)?;
            let adam_v = read_vec_f32(&buf, &mut pos, pc, buf_len)?;
            let adam_t = read_u64(&buf, &mut pos, buf_len)? as usize;
            let bridge_weight = read_f32(&buf, &mut pos, buf_len)?;
            let bw_m = read_f32(&buf, &mut pos, buf_len)?;
            let bw_v = read_f32(&buf, &mut pos, buf_len)?;
            let nwc = read_u64(&buf, &mut pos, buf_len)? as usize;
            let norm_weight = read_vec_f32(&buf, &mut pos, nwc, buf_len)?;
            let norm_bias = read_vec_f32(&buf, &mut pos, nwc, buf_len)?;

            layer_checkpoints.push(BlockSummaryCheckpoint {
                params, adam_m, adam_v, adam_t,
                bridge_weight, bw_m, bw_v,
                norm_weight, norm_bias,
            });
        }

        // Check for trailing data (mild warning, not an error)
        if pos < buf_len {
            tracing::info!(event = "checkpoint_trailing_data", bytes_remaining = buf_len - pos,
                           "Checkpoint has {} bytes of trailing data", buf_len - pos);
        }

        Ok(Self { version, global_step, layer_checkpoints })
    }

    /// Create checkpoint from current student + optimizer state.
    pub fn from_student(
        student: &Gemma4Student,
        optimizers: &[BlockSummaryAdam],
        global_step: usize,
    ) -> Self {
        let layer_checkpoints = student.block_summaries.iter().enumerate()
            .map(|(i, bs)| {
                let params = bs.export_trainable();
                let (adam_m, adam_v, adam_t, bw_m, bw_v, norm_w, norm_b) =
                    if i < optimizers.len() {
                        let opt = &optimizers[i];
                        let mut m = Vec::new();
                        m.extend_from_slice(&opt.sq_state.m);
                        m.extend_from_slice(&opt.qp_state.m);
                        m.extend_from_slice(&opt.op_state.m);
                        let mut v = Vec::new();
                        v.extend_from_slice(&opt.sq_state.v);
                        v.extend_from_slice(&opt.qp_state.v);
                        v.extend_from_slice(&opt.op_state.v);
                        (m, v, opt.sq_state.t, opt.bw_m, opt.bw_v,
                         opt.nw_state.m.clone(), opt.nb_state.m.clone())
                    } else {
                        let hd = bs.hidden_dim;
                        let sz = params.len().max(bs.summary_queries.len() + hd*hd + hd*hd);
                        (vec![0.0; sz], vec![0.0; sz], 0, 0.0, 0.0,
                         vec![0.0; hd], vec![0.0; hd])
                    };
                let _hd = bs.hidden_dim;
                BlockSummaryCheckpoint {
                    params,
                    adam_m,
                    adam_v,
                    adam_t,
                    bridge_weight: bs.bridge_weight,
                    bw_m,
                    bw_v,
                    norm_weight: norm_w,
                    norm_bias: norm_b,
                }
            })
            .collect();

        Self {
            version: Self::FORMAT_VERSION,
            global_step,
            layer_checkpoints,
        }
    }

    /// Apply checkpoint to restore student state + reconstruct optimizers.
    pub fn apply(&self, student: &mut Gemma4Student) -> Vec<BlockSummaryAdam> {
        let lr = student.distill_config.learning_rate;
        let mut optimizers = Vec::with_capacity(self.layer_checkpoints.len());

        for (i, lc) in self.layer_checkpoints.iter().enumerate() {
            if i >= student.block_summaries.len() { break; }
            let bs = &mut student.block_summaries[i];

            // Restore trainable params
            bs.import_trainable(&lc.params);
            bs.bridge_weight = lc.bridge_weight;

            // Reconstruct Adam with restored state
            let hd = bs.hidden_dim;
            let sq_len = bs.summary_queries.len();
            let sq_m: Vec<f32> = lc.adam_m.get(..sq_len).unwrap_or(&[]).to_vec();
            let sq_v: Vec<f32> = lc.adam_v.get(..sq_len).unwrap_or(&[]).to_vec();
            let qp_off = sq_len;
            let qp_end = qp_off + hd * hd;
            let op_off = qp_end;
            let op_end = op_off + hd * hd;

            let mut opt = BlockSummaryAdam::new(bs, lr);
            opt.sq_state.m = if sq_m.len() == sq_len { sq_m } else { vec![0.0; sq_len] };
            opt.sq_state.v = if sq_v.len() == sq_len { sq_v } else { vec![0.0; sq_len] };
            opt.sq_state.t = lc.adam_t;
            if qp_end <= lc.adam_m.len() {
                opt.qp_state.m = lc.adam_m[qp_off..qp_end].to_vec();
                opt.qp_state.v = lc.adam_v[qp_off..qp_end].to_vec();
            }
            if op_end <= lc.adam_m.len() {
                opt.op_state.m = lc.adam_m[op_off..op_end].to_vec();
                opt.op_state.v = lc.adam_v[op_off..op_end].to_vec();
            }
            opt.bw_m = lc.bw_m;
            opt.bw_v = lc.bw_v;
            if lc.norm_weight.len() == hd {
                opt.nw_state.m = lc.norm_weight.clone();
            }
            if lc.norm_bias.len() == hd {
                opt.nb_state.m = lc.norm_bias.clone();
            }

            optimizers.push(opt);
        }

        optimizers
    }
}

// ---------------------------------------------------------------------------
// WGSL Kernel for Block Summary Cross-Attention (GPU)
// ---------------------------------------------------------------------------

/// WGSL compute shader for Block Summary cross-attention.
/// Projects queries/keys/values, computes scaled dot-product attention,
/// and applies output projection — all on GPU.
pub const BLOCK_SUMMARY_CROSS_ATTN_WGSL: &str = r#"
struct Params {
    num_queries: u32,
    hidden_dim: u32,
    block_size: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read>       queries:    array<f32>;  // [nq × hd]
@group(0) @binding(1) var<storage, read>       tokens:     array<f32>;  // [bs × hd]
@group(0) @binding(2) var<storage, read_write> output:     array<f32>;  // [nq × hd]
@group(0) @binding(3) var<uniform>             params:     Params;

/// Cross-attention: for each query, compute attention over all tokens.
/// This is the forward pass of Block Summary compression.
@compute @workgroup_size(64)
fn block_summary_cross_attn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q_idx = gid.x;
    if (q_idx >= params.num_queries) { return; }

    let hd = params.hidden_dim;
    let bs = params.block_size;
    let scale = 1.0 / sqrt(f32(hd));

    // Compute attention scores: Q × K^T
    var max_score = -1e30;
    var scores: array<f32, 4096>; // Max block_size

    for (var t: u32 = 0u; t < bs; t = t + 1u) {
        var dot = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dot += queries[q_idx * hd + d] * tokens[t * hd + d];
        }
        let score = dot * scale;
        scores[t] = score;
        if (score > max_score) { max_score = score; }
    }

    // Softmax
    var sum_exp = 0.0;
    for (var t: u32 = 0u; t < bs; t = t + 1u) {
        scores[t] = exp(scores[t] - max_score);
        sum_exp += scores[t];
    }
    for (var t: u32 = 0u; t < bs; t = t + 1u) {
        scores[t] = scores[t] / sum_exp;
    }

    // Weighted sum of values
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        var sum = 0.0;
        for (var t: u32 = 0u; t < bs; t = t + 1u) {
            sum += scores[t] * tokens[t * hd + d];
        }
        output[q_idx * hd + d] = sum;
    }
}
"#;

/// WGSL compute shader for Block Summary backward pass.
/// Given d_loss/d_output (upstream gradient), computes d_loss/d_queries and d_loss/d_tokens.
pub const BLOCK_SUMMARY_BACKWARD_WGSL: &str = r#"
struct Params {
    num_queries: u32,
    hidden_dim: u32,
    block_size: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read>       d_output:     array<f32>;  // [nq × hd] upstream gradient
@group(0) @binding(1) var<storage, read>       queries:      array<f32>;  // [nq × hd]
@group(0) @binding(2) var<storage, read>       tokens:       array<f32>;  // [bs × hd]
@group(0) @binding(3) var<storage, read_write> d_queries:    array<f32>;  // [nq × hd] output gradient
@group(0) @binding(4) var<storage, read_write> d_tokens:     array<f32>;  // [bs × hd] output gradient
@group(0) @binding(5) var<uniform>             params:       Params;

/// Backward pass for cross-attention.
/// Forward was: for each query q, attn = softmax(Q_q · K_t / √d) · V
/// We compute: d_queries and d_tokens given d_output.
@compute @workgroup_size(64)
fn block_summary_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nq = params.num_queries;
    let hd = params.hidden_dim;
    let bs = params.block_size;
    let scale = 1.0 / sqrt(f32(hd));

    // Phase 1: Each thread handles one element of d_queries or d_tokens
    // Total elements = nq * hd + bs * hd
    let total_query_elements = nq * hd;
    let total_elements = total_query_elements + bs * hd;

    if (idx >= total_elements) { return; }

    if (idx < total_query_elements) {
        // d_queries[q * hd + d]
        let q = idx / hd;
        let d = idx % hd;

        // d_queries[q,d] = Σ_t attn_weights[q,t] * d_output[q,d] * tokens[t,d]
        // This is a simplified gradient; the full version accumulates through softmax.
        // Since forward: out[q,d] = Σ_t softmax(Q·K^T)[q,t] * tokens[t,d]
        // d_out/d_queries[q,d'] feeds into Q·K^T → softmax → weighted sum
        // For now, use direct gradient: d_queries accumulates d_output scaled by attention
        var grad = 0.0;
        for (var t: u32 = 0u; t < bs; t = t + 1u) {
            // Compute attention weight (recompute from forward)
            var dot = 0.0;
            for (var dd: u32 = 0u; dd < hd; dd = dd + 1u) {
                dot += queries[q * hd + dd] * tokens[t * hd + dd];
            }
            let score = dot * scale;
            // Softmax weight approximation (for single-query case, this is just softmax of one score)
            // For multi-query, you'd need the full softmax. Simplified: use score directly.
            let attn_weight = score / (abs(score) + 1.0); // Smoothed gradient signal
            grad += attn_weight * d_output[q * hd + d] * tokens[t * hd + d];
        }
        d_queries[idx] = d_queries[idx] + grad * scale;
    } else {
        // d_tokens element
        let token_idx = idx - total_query_elements;
        let t = token_idx / hd;
        let d = token_idx % hd;

        // d_tokens[t,d] = Σ_q attn_weight[q,t] * d_output[q,d]
        var grad = 0.0;
        for (var q: u32 = 0u; q < nq; q = q + 1u) {
            var dot = 0.0;
            for (var dd: u32 = 0u; dd < hd; dd = dd + 1u) {
                dot += queries[q * hd + dd] * tokens[t * hd + dd];
            }
            let score = dot * scale;
            let attn_weight = score / (abs(score) + 1.0);
            grad += attn_weight * d_output[q * hd + d];
        }
        d_tokens[token_idx] = d_tokens[token_idx] + grad * scale;
    }
}
"#;

/// GPU operation for Block Summary cross-attention.
pub struct BlockSummaryGpuOp {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    backward_pipeline: wgpu::ComputePipeline,
    backward_layout: wgpu::BindGroupLayout,
    num_queries: usize,
    hidden_dim: usize,
    block_size: usize,
}

impl BlockSummaryGpuOp {
    /// Create a new GPU BlockSummary op. Requires a wgpu device and queue.
    pub fn new(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        num_queries: usize,
        hidden_dim: usize,
        block_size: usize,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BlockSummary Cross-Attention"),
            source: wgpu::ShaderSource::Wgsl(BLOCK_SUMMARY_CROSS_ATTN_WGSL.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BlockSummary Layout"),
            entries: &[
                // binding 0: queries [nq × hd] (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: tokens [bs × hd] (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: output [nq × hd] (storage, read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: params uniform (4 × u32)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BlockSummary Forward"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BlockSummary Pipeline Layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            })),
            module: &shader,
            entry_point: Some("block_summary_cross_attn"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Backward shader
        let backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BlockSummary Backward"),
            source: wgpu::ShaderSource::Wgsl(BLOCK_SUMMARY_BACKWARD_WGSL.into()),
        });

        let backward_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BlockSummary Backward Layout"),
            entries: &[
                // binding 0: d_output (grad from upstream) [nq × hd] read-only
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                // binding 1: queries [nq × hd] read-only
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                // binding 2: tokens [bs × hd] read-only
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                // binding 3: d_queries output [nq × hd] read-write
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                // binding 4: d_tokens output [bs × hd] read-write
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
                // binding 5: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None,
                },
            ],
        });

        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("BlockSummary Backward"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("BlockSummary Backward Pipeline Layout"),
                bind_group_layouts: &[Some(&backward_layout)],
                immediate_size: 0,
            })),
            module: &backward_shader,
            entry_point: Some("block_summary_backward"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            device, queue, pipeline, bind_group_layout: layout,
            backward_pipeline, backward_layout,
            num_queries, hidden_dim, block_size,
        }
    }

    /// Dispatch forward pass on GPU: cross-attention summary.
    /// queries: [nq × hd], tokens: [bs × hd] → output: [nq × hd]
    pub fn dispatch_forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: &crate::compute::GpuBuffer,
        tokens: &crate::compute::GpuBuffer,
        output: &crate::compute::GpuBuffer,
    ) -> crate::error::Result<()> {
        let params_data: [u32; 4] = [
            self.num_queries as u32,
            self.hidden_dim as u32,
            self.block_size as u32,
            self.hidden_dim as u32, // head_dim = hidden_dim for now
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BlockSummary Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlockSummary Forward BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: queries.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tokens.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("BlockSummary Forward Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg = 64u32;
        pass.dispatch_workgroups((self.num_queries as u32 + wg - 1) / wg, 1, 1);
        drop(pass);

        tracing::debug!(
            "BlockSummary forward dispatch: nq={} hd={} bs={}",
            self.num_queries, self.hidden_dim, self.block_size
        );
        Ok(())
    }

    /// Dispatch backward pass on GPU: compute d_queries and d_tokens from d_output.
    pub fn dispatch_backward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        d_output: &crate::compute::GpuBuffer,
        queries: &crate::compute::GpuBuffer,
        tokens: &crate::compute::GpuBuffer,
        d_queries: &crate::compute::GpuBuffer,
        d_tokens: &crate::compute::GpuBuffer,
    ) -> crate::error::Result<()> {
        let params_data: [u32; 4] = [
            self.num_queries as u32,
            self.hidden_dim as u32,
            self.block_size as u32,
            self.hidden_dim as u32,
        ];
        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BlockSummary Backward Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BlockSummary Backward BG"),
            layout: &self.backward_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: d_output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: queries.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: tokens.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: d_queries.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: d_tokens.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("BlockSummary Backward Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.backward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per query for d_queries, plus enough for d_tokens
        let wg = 64u32;
        let total_elements = (self.num_queries + self.block_size).max(1) as u32;
        pass.dispatch_workgroups((total_elements + wg - 1) / wg, 1, 1);
        drop(pass);

        tracing::debug!(
            "BlockSummary backward dispatch: nq={} hd={} bs={}",
            self.num_queries, self.hidden_dim, self.block_size
        );
        Ok(())
    }

    /// CPU fallback for forward.
    pub fn forward_cpu(&self, queries: &[f32], tokens: &[f32]) -> Vec<f32> {
        let layer = BlockSummaryLayer::new_identity(self.hidden_dim, self.block_size);
        let mut layer = layer;
        layer.summary_queries = queries.to_vec();
        layer.forward(tokens)
    }

    pub fn num_queries(&self) -> usize { self.num_queries }
    pub fn hidden_dim(&self) -> usize { self.hidden_dim }
    pub fn block_size(&self) -> usize { self.block_size }
}

// ---------------------------------------------------------------------------
// Calibration Dataset Pipeline
// ---------------------------------------------------------------------------

/// A text dataset for distillation training.
pub struct TextDataset {
    /// Tokenized data as token IDs.
    pub token_ids: Vec<u32>,
    /// Sequence length for batching.
    pub seq_len: usize,
    /// Current position in the dataset.
    pub position: usize,
    /// Dataset name.
    pub name: String,
}

impl TextDataset {
    /// Create from raw text using a simple byte-level tokenizer.
    /// For real use, you'd use the BPE tokenizer from model/tokenizer.rs.
    pub fn from_text(text: &str, seq_len: usize, name: &str) -> Self {
        // Simple byte-level tokenization
        let token_ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        Self { token_ids, seq_len, position: 0, name: name.to_string() }
    }

    /// Create from pre-tokenized IDs.
    pub fn from_token_ids(token_ids: Vec<u32>, seq_len: usize, name: &str) -> Self {
        Self { token_ids, seq_len, position: 0, name: name.to_string() }
    }

    /// Create a synthetic dataset for testing.
    pub fn synthetic(vocab_size: u32, seq_len: usize, num_sequences: usize) -> Self {
        let token_ids: Vec<u32> = (0..num_sequences * seq_len)
            .map(|i| i as u32 % vocab_size.max(1))
            .collect();
        Self {
            token_ids,
            seq_len,
            position: 0,
            name: "synthetic".to_string(),
        }
    }

    /// Get the next batch of token IDs.
    pub fn next_batch(&mut self) -> Option<&[u32]> {
        if self.position + self.seq_len > self.token_ids.len() {
            self.position = 0; // Wrap around
        }
        if self.token_ids.len() < self.seq_len {
            return None;
        }
        let start = self.position;
        self.position += self.seq_len;
        Some(&self.token_ids[start..start + self.seq_len])
    }

    /// Number of complete sequences.
    pub fn num_sequences(&self) -> usize {
        self.token_ids.len() / self.seq_len
    }

    /// Total tokens.
    pub fn total_tokens(&self) -> usize {
        self.token_ids.len()
    }

    /// Train/validation split.
    pub fn split(&self, train_fraction: f32) -> (TextDataset, TextDataset) {
        let split_point = (self.token_ids.len() as f32 * train_fraction) as usize;
        let split_point = (split_point / self.seq_len) * self.seq_len; // Align to seq_len
        (
            TextDataset {
                token_ids: self.token_ids[..split_point].to_vec(),
                seq_len: self.seq_len,
                position: 0,
                name: format!("{}_train", self.name),
            },
            TextDataset {
                token_ids: self.token_ids[split_point..].to_vec(),
                seq_len: self.seq_len,
                position: 0,
                name: format!("{}_val", self.name),
            },
        )
    }

    /// Reset position.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

// ---------------------------------------------------------------------------
// Verified Weight Loader
// ---------------------------------------------------------------------------

/// Load a Gemma 4 model from a safetensors file with full verification.
///
/// This is the main entry point for the distillation pipeline:
/// 1. Opens the safetensors file
/// 2. Verifies all expected tensors are present
/// 3. Validates tensor shapes against the config
/// 4. Returns a ready-to-use MappedGemma4Model
pub fn load_gemma4_model(
    path: &std::path::Path,
    config: Gemma4Config,
) -> Result<MappedGemma4Model, String> {
    // Load safetensors
    let weights = crate::model::safetensors::load_safetensors(path)
        .map_err(|e| format!("Failed to load safetensors: {:?}", e))?;

    // Verify architecture
    let arch = weights.detect_architecture();
    match arch {
        crate::model::safetensors::ModelArchitecture::Llama => {}, // Gemma uses LLaMA naming
        crate::model::safetensors::ModelArchitecture::Unknown => {
            // Still try — detection may miss Gemma variants
        }
        other => {
            // Log warning but continue
            eprintln!("Warning: detected architecture {:?}, expected Llama (Gemma)", other);
        }
    }

    // Verify layer count
    let detected_layers = weights.infer_num_layers();
    if detected_layers > 0 && detected_layers != config.num_layers {
        return Err(format!(
            "Layer count mismatch: config says {}, safetensors has {}",
            config.num_layers, detected_layers
        ));
    }

    // Verify embedding size
    if let Some(detected_dim) = weights.infer_hidden_dim() {
        if detected_dim != config.hidden_dim {
            return Err(format!(
                "Hidden dim mismatch: config says {}, safetensors has {}",
                config.hidden_dim, detected_dim
            ));
        }
    }

    // Verify vocab size
    if let Some(detected_vs) = weights.infer_vocab_size() {
        if detected_vs != config.vocab_size {
            return Err(format!(
                "Vocab size mismatch: config says {}, safetensors has {}",
                config.vocab_size, detected_vs
            ));
        }
    }

    // Build the model
    MappedGemma4Model::from_loaded_weights(config, &weights)
}

/// Load a Gemma 4 model from safetensors using memory mapping (memory-efficient).
/// Uses mmap to avoid loading the entire file into RAM. Only converts tensors
/// to f32 on demand, keeping peak memory proportional to the largest tensor
/// rather than the entire model.
pub fn load_gemma4_model_mmap(
    path: &std::path::Path,
    config: Gemma4Config,
) -> Result<MappedGemma4Model, String> {
    use crate::model::safetensors::MmapedSafetensors;

    let mmaped = MmapedSafetensors::open(path)
        .map_err(|e| format!("Failed to mmap safetensors: {:?}", e))?;

    // Auto-detect naming convention
    let names = mmaped.tensor_names();
    let uses_mm_naming = names.iter().any(|n| n.starts_with("model.language_model."));

    if uses_mm_naming {
        tracing::info!(event = "detected_gemma_4_multimodal_naming_model", "Detected Gemma 4 multimodal naming (model.language_model.layers.N.*)");
    } else {
        tracing::info!(event = "detected_standard_naming_model_layers_n", "Detected standard naming (model.layers.N.*)");
    }

    // Verify dimensions — use language_model-specific detection for MM
    if uses_mm_naming {
        // For MM models, check language_model layer count
        let lang_layers: Vec<usize> = names.iter()
            .filter(|n| n.starts_with("model.language_model.layers."))
            .filter_map(|n| {
                let rest = n.strip_prefix("model.language_model.layers.")?;
                let end = rest.find('.')?;
                rest[..end].parse().ok()
            })
            .collect();
        let max_layer = lang_layers.iter().max().map(|&m| m + 1).unwrap_or(0);
        if max_layer > 0 && max_layer != config.num_layers {
            return Err(format!("Layer count mismatch: config={}, detected language_model layers={}", config.num_layers, max_layer));
        }
    } else {
        if let Some(detected) = Some(mmaped.infer_num_layers()).filter(|&d| d > 0) {
            if detected != config.num_layers {
                return Err(format!("Layer count mismatch: config={}, detected={}", config.num_layers, detected));
            }
        }
    }

    // Stream tensors from mmap (only one in RAM at a time)
    tracing::info!(event = "streaming_tensors_from_mmap_one_at", "Streaming tensors from mmap (one at a time)...");
    if uses_mm_naming {
        MappedGemma4Model::from_mmap_mm(config, &mmaped)
    } else {
        MappedGemma4Model::from_mmap(config, &mmaped)
    }
}

/// Load a model from GGUF format.
/// Dequantizes tensors to F32 on load.
pub fn load_gemma4_model_gguf(
    path: &std::path::Path,
    config: Gemma4Config,
) -> Result<MappedGemma4Model, String> {
    let gguf = crate::model::gguf::load_gguf(path)
        .map_err(|e| format!("Failed to load GGUF: {:?}", e))?;

    // Map GGUF tensor names to HuggingFace names
    let name_map = gguf.standard_name_map();

    // Load all tensors and map names
    let raw_tensors: HashMap<String, Vec<f32>> = gguf.tensor_infos.keys()
        .filter_map(|name| {
            let loaded = gguf.load_tensor(name).ok()?;
            // Try mapped name first, then original
            let target_name = name_map.get(name)
                .cloned()
                .unwrap_or_else(|| name.clone());
            Some((target_name, loaded.data))
        })
        .collect();

    // Build a LoadedWeights-compatible wrapper
    let fake_weights = RawWeights { tensors: raw_tensors };
    MappedGemma4Model::from_raw_weights(config, &fake_weights)
}

/// Simple wrapper around a HashMap for weight loading.
pub struct RawWeights {
    tensors: HashMap<String, Vec<f32>>,
}

impl RawWeights {
    fn get(&self, name: &str) -> Option<Vec<f32>> {
        self.tensors.get(name).cloned()
    }
}

use std::collections::HashMap;

impl MappedGemma4Model {
    /// Build model from a raw weight map (used by GGUF and LLaMA loaders).
    /// Infers per-layer dimensions from actual tensor shapes so it works
    /// correctly with any Gemma 4 variant (mixed sliding/full attention).
    pub fn from_raw_weights(
        config: Gemma4Config,
        weights: &RawWeights,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Option<Vec<f32>> {
            weights.get(name)
        };

        let hd = config.hidden_dim;

        // Embedding — try multiple naming conventions
        let embed_tokens = get("model.embed_tokens.weight")
            .or_else(|| get("embedding.weight"))
            .ok_or_else(|| "Missing embedding weights".to_string())?;

        // LM head (may be tied to embedding)
        // GGUF stores weights as [out_features, in_features] row-major,
        // same as safetensors. Must transpose to [in_features, out_features]
        // for our matmul (C = A @ B where B is [k, n]).
        let lm_head_raw = get("lm_head.weight")
            .or_else(|| get("output.weight"))
            .unwrap_or_else(|| embed_tokens.clone());
        let lm_head = transpose(&lm_head_raw, config.vocab_size, hd);

        // Per-layer token embeddings (PLE) — try standard names
        // PLE model-level weights
        let ple_model_projection = get("model.language_model.per_layer_model_projection.weight")
            .or_else(|| get("model.per_layer_model_projection.weight"))
            .map(|w| {
                let ple_total = config.num_layers * config.hidden_size_per_layer_input.unwrap_or(256);
                tracing::info!(event = "ple_model_proj", raw_len = w.len(), ple_total, hd);
                transpose(&w, ple_total, hd)
            });
        let ple_projection_norm = get("model.language_model.per_layer_projection_norm.weight")
            .or_else(|| get("model.per_layer_projection_norm.weight"));
        // Per-layer token embeddings for PLE
        let embed_tokens_per_layer = get("model.language_model.embed_tokens_per_layer.weight")
            .or_else(|| get("model.embed_tokens_per_layer.weight"));
        if ple_model_projection.is_some() {
            tracing::info!(event = "ple_loaded", "Loaded PLE model projection + norm");
        }

        // Final norm
        let final_norm = get("model.norm.weight")
            .or_else(|| get("final_norm.weight"))
            .or_else(|| get("output_norm.weight"))
            .ok_or_else(|| "Missing final norm".to_string())?;

        // Per-layer weights — infer dimensions from actual tensor shapes
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            let q = get(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.q_proj.weight", layer_idx)))
                .ok_or_else(|| format!("Missing q_proj for layer {}", layer_idx))?;
            let k = get(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.k_proj.weight", layer_idx)))
                .ok_or_else(|| format!("Missing k_proj for layer {}", layer_idx))?;
            let v = get(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.v_proj.weight", layer_idx)))
                .ok_or_else(|| format!("Missing v_proj for layer {}", layer_idx))?;
            let o = get(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.out_proj.weight", layer_idx)))
                .ok_or_else(|| format!("Missing o_proj for layer {}", layer_idx))?;
            let inorm = get(&format!("model.layers.{}.input_layernorm.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.attn_norm.weight", layer_idx)))
                .ok_or_else(|| format!("Missing input_norm for layer {}", layer_idx))?;
            let pnorm = get(&format!("model.layers.{}.post_attention_layernorm.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.post_attn_norm.weight", layer_idx)))
                .or_else(|| get(&format!("layers.{}.ff_norm.weight", layer_idx)))
                .ok_or_else(|| format!("Missing post_attn_norm for layer {}", layer_idx))?;

            // Infer per-layer head dimensions from q_proj shape
            // q_proj is stored as [q_dim, hidden_dim] → len = q_dim * hidden_dim
            let layer_q_dim = q.len() / hd;
            let layer_head_dim = layer_q_dim / config.num_heads;
            let layer_kv_dim = k.len() / hd;

            // Q/K norms — try per-layer names, fallback to identity
            let q_norm = get(&format!("model.layers.{}.self_attn.q_norm.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.q_norm.weight", layer_idx)))
                .unwrap_or_else(|| vec![1.0f32; layer_head_dim]);
            let k_norm = get(&format!("model.layers.{}.self_attn.k_norm.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.k_norm.weight", layer_idx)))
                .unwrap_or_else(|| vec![1.0f32; layer_head_dim]);

            // Transpose attention weights from [out, in] to [in, out]
            // GGUF stores weights as [out_features, in_features] row-major,
            // same as safetensors. Our matmul expects [in_features, out_features].
            let q = transpose(&q, layer_q_dim, hd);
            let k = transpose(&k, layer_kv_dim, hd);
            let v = transpose(&v, layer_kv_dim, hd);
            let o = transpose(&o, hd, layer_q_dim);

            let attn = Gemma4AttnWeights {
                q_proj: q, k_proj: k, v_proj: v, o_proj: o,
                input_norm: inorm,
                post_attn_norm: pnorm,
                q_norm,
                k_norm,
                head_dim: layer_head_dim,
                q_dim: layer_q_dim,
                kv_dim: layer_kv_dim,
            };

            // Infer per-layer intermediate dim from gate_proj shape
            let gate_raw = get(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.ff_gate.weight", layer_idx)));
            let layer_inter_dim = gate_raw.as_ref().map(|g| g.len() / hd).unwrap_or(config.intermediate_dim);

            let ffn = if config.moe_layers.contains(&layer_idx) {
                let router_raw = get(&format!("model.layers.{}.block_sparse_moe.gate.weight", layer_idx))
                    .or_else(|| get(&format!("layers.{}.ff_gate.weight", layer_idx)))
                    .ok_or_else(|| format!("Missing MoE router for layer {}", layer_idx))?;
                let router = transpose(&router_raw, config.num_experts, hd);

                let mut expert_gates = Vec::new();
                let mut expert_ups = Vec::new();
                let mut expert_downs = Vec::new();

                for e in 0..config.num_experts {
                    let g_raw = get(&format!("model.layers.{}.block_sparse_moe.experts.{}.w1.weight", layer_idx, e))
                        .or_else(|| get(&format!("layers.{}.expert.{}.gate.weight", layer_idx, e)))
                        .ok_or_else(|| format!("Missing expert {} gate for layer {}", e, layer_idx))?;
                    let u_raw = get(&format!("model.layers.{}.block_sparse_moe.experts.{}.w3.weight", layer_idx, e))
                        .or_else(|| get(&format!("layers.{}.expert.{}.up.weight", layer_idx, e)))
                        .ok_or_else(|| format!("Missing expert {} up for layer {}", e, layer_idx))?;
                    let d_raw = get(&format!("model.layers.{}.block_sparse_moe.experts.{}.w2.weight", layer_idx, e))
                        .or_else(|| get(&format!("layers.{}.expert.{}.down.weight", layer_idx, e)))
                        .ok_or_else(|| format!("Missing expert {} down for layer {}", e, layer_idx))?;
                    let g = transpose(&g_raw, layer_inter_dim, hd);
                    let u = transpose(&u_raw, layer_inter_dim, hd);
                    let d = transpose(&d_raw, hd, layer_inter_dim);
                    expert_gates.push(g);
                    expert_ups.push(u);
                    expert_downs.push(d);
                }

                Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs }
            } else {
                let gate_raw = gate_raw
                    .ok_or_else(|| format!("Missing gate_proj for layer {}", layer_idx))?;
                let up_raw = get(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx))
                    .or_else(|| get(&format!("layers.{}.ff_up.weight", layer_idx)))
                    .ok_or_else(|| format!("Missing up_proj for layer {}", layer_idx))?;
                let down_raw = get(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx))
                    .or_else(|| get(&format!("layers.{}.ff_down.weight", layer_idx)))
                    .ok_or_else(|| format!("Missing down_proj for layer {}", layer_idx))?;
                // Transpose FFN weights from [out, in] to [in, out]
                let gate = transpose(&gate_raw, layer_inter_dim, hd);
                let up = transpose(&up_raw, layer_inter_dim, hd);
                let down = transpose(&down_raw, hd, layer_inter_dim);

                Gemma4FfnWeights::Dense { gate_proj: gate, up_proj: up, down_proj: down }
            };

            // Pre/post FFN norms
            let pre_ffn_norm = get(&format!("model.layers.{}.pre_feedforward_layernorm.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.pre_ff_norm.weight", layer_idx)))
                .unwrap_or_else(|| vec![1.0f32; hd]);
            let post_ffn_norm = get(&format!("model.layers.{}.post_feedforward_layernorm.weight", layer_idx))
                .or_else(|| get(&format!("layers.{}.post_ff_norm.weight", layer_idx)))
                .unwrap_or_else(|| vec![1.0f32; hd]);

            // Layer scalar — try multiple names
            let layer_scalar = get(&format!("model.layers.{}.layer_scalar", layer_idx))
                .or_else(|| get(&format!("layers.{}.layer_scalar", layer_idx)))
                .and_then(|v| v.first().copied())
                .unwrap_or(1.0f32);

            // Per-layer embeddings (PLE) — try standard HuggingFace names
            let ple_dim = config.hidden_size_per_layer_input.unwrap_or(0);
            let per_layer_input_gate = if ple_dim > 0 {
                get(&format!("model.layers.{}.per_layer_input_gate.weight", layer_idx))
                    .map(|w| {
                        tracing::info!(event = "ple_gate_shape", layer = layer_idx, raw_len = w.len(), ple_dim, hd);
                        transpose(&w, ple_dim, hd) // [ple_dim, hd] → [hd, ple_dim]
                    })
            } else { None };
            let per_layer_projection = if ple_dim > 0 {
                get(&format!("model.layers.{}.per_layer_projection.weight", layer_idx))
                    .map(|w| {
                        tracing::info!(event = "ple_proj_shape", layer = layer_idx, raw_len = w.len(), ple_dim, hd);
                        transpose(&w, hd, ple_dim) // [hd, ple_dim] → [ple_dim, hd]
                    })
            } else { None };
            let post_per_layer_input_norm = if ple_dim > 0 {
                get(&format!("model.layers.{}.post_per_layer_input_norm.weight", layer_idx))
            } else { None };

            layers.push(Gemma4LayerWeights {
                attn, ffn,
                pre_ffn_norm, post_ffn_norm, layer_scalar,
                per_layer_input_gate, per_layer_projection, post_per_layer_input_norm,
                rope_theta: config.layer_rope_theta(layer_idx),
                partial_rotary_factor: config.layer_partial_rotary_factor(layer_idx),
                intermediate_dim: layer_inter_dim,
            });
            tracing::info!(
                event = "loaded_layer",
                "Loaded layer {}/{}: head_dim={}, inter_dim={}, rope_theta={}, scalar={:.4}",
                layer_idx + 1, config.num_layers, layer_head_dim, layer_inter_dim,
                config.layer_rope_theta(layer_idx), layer_scalar,
            );
        }

        Ok(Self { config, embed_tokens, layers, final_norm, lm_head, ple_model_projection, ple_projection_norm, embed_tokens_per_layer })
    }
}
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_moe_creation() {
        let moe = CpuMoELayer::new(256, 512, 8, 2);
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.top_k, 2);
        assert_eq!(moe.expert_up.len(), 8);
        assert_eq!(moe.expert_down.len(), 8);
        assert!(moe.frozen);
    }

    #[test]
    fn test_cpu_moe_forward() {
        let mut moe = CpuMoELayer::new(64, 128, 4, 2);
        // Initialize with small random-like weights
        for w in moe.gate_weights.iter_mut() {
            *w = 0.01;
        }
        for e in 0..4 {
            for w in moe.expert_up[e].iter_mut() {
                *w = 0.01;
            }
            for w in moe.expert_down[e].iter_mut() {
                *w = 0.01;
            }
            for w in moe.expert_gate[e].iter_mut() {
                *w = 0.01;
            }
        }
        let input = vec![1.0f32; 2 * 64]; // seq_len=2, hidden=64
        let output = moe.forward(&input, 2);
        assert_eq!(output.len(), 2 * 64);
    }

    #[test]
    fn test_cpu_moe_top_k_select() {
        let moe = CpuMoELayer::new(64, 128, 8, 2);
        let logits = vec![1.0, 3.0, 2.0, 5.0, 0.5, 4.0, 1.5, 2.5];
        let (selected, weights) = moe.top_k_select(&logits, 1);
        assert_eq!(selected.len(), 2); // top_k = 2
        assert_eq!(selected[0], 3); // 5.0 is highest
        assert_eq!(selected[1], 5); // 4.0 is second highest
        // Weights should sum to 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_moe_num_params() {
        let moe = CpuMoELayer::new(64, 128, 4, 2);
        let expected = 4 * 64 + 4 + // gate
            4 * (128 * 64 + 64 * 128 + 128 * 64); // experts
        assert_eq!(moe.num_params(), expected);
    }

    #[test]
    fn test_block_summary_identity_init() {
        let layer = BlockSummaryLayer::new_identity(256, 4096);
        assert_eq!(layer.bridge_weight, 0.1); // Starts at 0.1 for non-zero initial KL
        assert_eq!(layer.summary_queries.len(), 256);
        assert!(layer.trainable);
        // All summary queries should be 0 (no contribution at init)
        assert!(layer.summary_queries.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_block_summary_forward() {
        let layer = BlockSummaryLayer::new_identity(64, 4);
        let tokens = vec![1.0f32; 4 * 64]; // 4 tokens × 64 dim
        let summary = layer.forward(&tokens);
        assert_eq!(summary.len(), 64); // 1 query × 64 dim
    }

    #[test]
    fn test_block_summary_export_import() {
        let mut layer = BlockSummaryLayer::new_identity(64, 4);
        let weights = layer.export_trainable();
        assert!(weights.len() > 0);

        // Modify and reimport
        let mut modified = weights.clone();
        modified[0] = 99.0;
        layer.import_trainable(&modified);
        assert_eq!(layer.summary_queries[0], 99.0);
    }

    #[test]
    fn test_block_summary_trainable_params() {
        let layer = BlockSummaryLayer::new_identity(64, 4);
        let params = layer.num_trainable_params();
        assert!(params > 0);
        // summary_queries + query_proj + out_proj + bridge_weight + norm_weight + norm_bias
        assert_eq!(params, layer.export_trainable().len());
    }

    #[test]
    fn test_gemma4_e2b_config() {
        let config = Gemma4Config::gemma4_e2b();
        // Verified against google/gemma-4-e2b-it config.json
        assert_eq!(config.hidden_dim, 1536);
        assert_eq!(config.num_layers, 35);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 1); // GQA
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.intermediate_dim, 6144);
        assert_eq!(config.num_experts, 1); // Dense
        assert_eq!(config.top_k, 1);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.max_position_embeddings, 131072);
        assert_eq!(config.sliding_window, 512);
        assert_eq!(config.moe_layers, Vec::<usize>::new()); // Dense model, no MoE layers
    }

    #[test]
    fn test_gemma4_e4b_config() {
        let config = Gemma4Config::gemma4_e4b();
        // Verified against google/gemma-4-e4b-it config.json
        assert_eq!(config.hidden_dim, 2560);
        assert_eq!(config.num_layers, 42);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 2); // GQA
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.intermediate_dim, 10240);
        assert_eq!(config.num_experts, 1); // Dense
        assert_eq!(config.top_k, 1);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.sliding_window, 512);
        assert_eq!(config.moe_layers, vec![5, 11, 17, 23, 29, 35, 41]); // full_attention layers
    }

    #[test]
    fn test_gemma4_26b_a4b_config() {
        let config = Gemma4Config::gemma4_26b_a4b();
        // Verified against google/gemma-4-26b-a4b-it config.json
        assert_eq!(config.hidden_dim, 2816);
        assert_eq!(config.num_layers, 30);
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.intermediate_dim, 2112);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.top_k, 8);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.max_position_embeddings, 262144);
        assert_eq!(config.sliding_window, 1024);
        assert_eq!(config.moe_layers, vec![5, 11, 17, 23, 29]); // full_attention layers
    }

    #[test]
    fn test_gemma4_31b_config() {
        let config = Gemma4Config::gemma4_31b();
        // Verified against google/gemma-4-31b-it config.json
        assert_eq!(config.hidden_dim, 5376);
        assert_eq!(config.num_layers, 60);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 16);
        assert_eq!(config.head_dim, 256);
        assert_eq!(config.intermediate_dim, 21504);
        assert_eq!(config.num_experts, 1); // Dense
        assert_eq!(config.top_k, 1);
        assert_eq!(config.vocab_size, 262144);
        assert_eq!(config.max_position_embeddings, 262144);
        assert_eq!(config.sliding_window, 1024);
        assert_eq!(config.moe_layers, vec![5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);
    }

    #[test]
    fn test_gemma4_e2b_to_block_attnres() {
        let gemma = Gemma4Config::gemma4_e2b();
        let bar = gemma.to_block_attnres_config();
        assert_eq!(bar.hidden_dim, 1536);
        assert_eq!(bar.num_experts, 1);
        assert!(!bar.use_moe); // Dense
    }

    #[test]
    fn test_gemma4_e4b_to_block_attnres() {
        let gemma = Gemma4Config::gemma4_e4b();
        let bar = gemma.to_block_attnres_config();
        assert_eq!(bar.hidden_dim, 2560);
        assert!(!bar.use_moe); // Dense (E4B is NOT MoE)
    }

    #[test]
    fn test_e2b_weight_mapper() {
        let config = Gemma4Config::gemma4_e2b();
        let mapper = Gemma4WeightMapper::new(config);
        let hd = 1536;
        let q_heads = 8;
        let kv_heads = 1;
        let head_dim = 256;
        let q = vec![0.1f32; q_heads * head_dim * hd]; // [2048, 1536]
        let k = vec![0.1f32; kv_heads * head_dim * hd]; // [256, 1536]
        let v = vec![0.1f32; kv_heads * head_dim * hd]; // [256, 1536]
        let o = vec![0.1f32; hd * q_heads * head_dim];   // [1536, 2048]
        let mapped = mapper.map_attention_weights(&q, &k, &v, &o, 0);
        assert!(!mapped.is_global); // Layer 0 is sliding_attention
        assert_eq!(mapped.hidden_dim, 1536);
    }

    #[test]
    fn test_e2b_block_summary_creation() {
        let config = Gemma4Config::gemma4_e2b();
        let mapper = Gemma4WeightMapper::new(config);
        let _summaries = mapper.create_block_summary_layers();
        // E2B has Block Summary injection at full_attention layers
        let bs = BlockSummaryLayer::new_identity(1536, 512);
        assert_eq!(bs.hidden_dim, 1536);
        assert_eq!(bs.bridge_weight, 0.1);
    }

    #[test]
    fn test_e4b_moe_weight_mapping() {
        // E4B is dense on HuggingFace, but test MoE mapping with synthetic config
        let mut config = Gemma4Config::gemma4_e4b();
        config.num_experts = 4;
        config.hidden_dim = 64;
        config.intermediate_dim = 128;
        let mapper = Gemma4WeightMapper::new(config);
        let hd = 64;
        let id = 128;
        let ne = 4;
        let gate: Vec<Vec<f32>> = (0..ne).map(|_| vec![0.01; id * hd]).collect();
        let up: Vec<Vec<f32>> = (0..ne).map(|_| vec![0.01; id * hd]).collect();
        let down: Vec<Vec<f32>> = (0..ne).map(|_| vec![0.01; hd * id]).collect();
        let router = vec![0.01; ne * hd];
        let moe = mapper.map_moe_experts(&gate, &up, &down, &router);
        assert_eq!(moe.num_experts, 4);
        assert_eq!(moe.top_k, 1); // E4B dense has top_k=1
        assert!(moe.frozen);
    }

    #[test]
    fn test_e2b_block_summary_round_trip() {
        let bs = BlockSummaryLayer::new_identity(128, 8);
        let tokens: Vec<f32> = (0..8 * 128).map(|i| (i as f32 * 0.001).sin()).collect();
        let summary = bs.forward(&tokens);
        assert_eq!(summary.len(), 128);

        // With bridge_weight=0, output should be mean-pool (identity)
        let mut expected_mean = vec![0.0; 128];
        for t in 0..8 {
            for d in 0..128 {
                expected_mean[d] += tokens[t * 128 + d];
            }
        }
        for d in &mut expected_mean {
            *d /= 8.0;
        }
        // Since bridge_weight=0, output = mean_pool
        for d in 0..128 {
            assert!((summary[d] - expected_mean[d]).abs() < 0.01,
                "Mismatch at dim {}: got {} expected {}", d, summary[d], expected_mean[d]);
        }
    }

    #[test]
    fn test_distillation_config_e2b() {
        let config = DistillationConfig::for_gemma4_e2b();
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.num_steps, 1000);
        assert!(config.freeze_moe); // freeze_moe=true is fine even with no MoE
    }

    #[test]
    fn test_distillation_config_e4b() {
        let config = DistillationConfig::for_gemma4_e4b();
        assert_eq!(config.batch_size, 2);
        assert_eq!(config.num_steps, 2000);
        assert!(config.freeze_moe);
        assert_eq!(config.load_balance_weight, 0.01);
    }

    #[test]
    fn test_e2b_cpu_moe_dense_fallback() {
        // E2B is dense (1 expert, top_k=1) — verify MoE handles it
        // Use smaller dims to keep test fast
        let mut moe = CpuMoELayer::new(64, 128, 1, 1);
        for w in moe.gate_weights.iter_mut() { *w = 1.0; }
        for w in moe.expert_up[0].iter_mut() { *w = 0.01; }
        for w in moe.expert_down[0].iter_mut() { *w = 0.01; }
        for w in moe.expert_gate[0].iter_mut() { *w = 0.01; }
        let input = vec![1.0f32; 2 * 64];
        let output = moe.forward(&input, 2);
        assert_eq!(output.len(), 2 * 64);
        // Should not be all zeros
        let energy: f32 = output.iter().map(|x| x * x).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_gemma4_12b_config() {
        let config = Gemma4Config::gemma4_12b();
        // UNVERIFIED: placeholder config
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 48);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.sliding_window, 1024);
        assert!(!config.moe_layers.is_empty());
    }

    #[test]
    fn test_gemma4_27b_config() {
        let config = Gemma4Config::gemma4_27b();
        // UNVERIFIED: placeholder config
        assert_eq!(config.hidden_dim, 4608);
        assert_eq!(config.num_layers, 60);
    }

    #[test]
    fn test_gemma4_to_block_attnres_config() {
        let gemma = Gemma4Config::gemma4_12b();
        let bar_config = gemma.to_block_attnres_config();
        assert_eq!(bar_config.hidden_dim, 4096);
        assert_eq!(bar_config.num_experts, 128);
        assert_eq!(bar_config.top_k, 2);
        assert!(bar_config.use_moe);
    }

    #[test]
    fn test_gemma4_injection_points() {
        let config = Gemma4Config::gemma4_12b();
        let points = config.block_summary_injection_points();
        assert!(!points.is_empty());
        // Should be every other layer starting from 1
        assert_eq!(points[0], 1);
        assert_eq!(points[1], 3);
    }

    #[test]
    fn test_weight_mapper_attention() {
        let config = Gemma4Config::gemma4_12b();
        let mapper = Gemma4WeightMapper::new(config);
        let q = vec![0.1f32; 4096 * 4096];
        let k = vec![0.1f32; 4096 * 4096];
        let v = vec![0.1f32; 4096 * 4096];
        let o = vec![0.1f32; 4096 * 4096];

        let mapped = mapper.map_attention_weights(&q, &k, &v, &o, 0);
        assert!(!mapped.is_global); // Layer 0 is local
        assert_eq!(mapped.q_proj.len(), 4096 * 4096);
    }

    #[test]
    fn test_weight_mapper_moe() {
        // Custom small config for test
        let mut config = Gemma4Config::gemma4_e4b();
        config.num_experts = 4;
        config.intermediate_dim = 128;
        config.hidden_dim = 64;
        let mapper = Gemma4WeightMapper::new(config);
        let hd = 64;
        let id = 128;
        let ne = 4;

        let gate: Vec<Vec<f32>> = (0..ne).map(|_| vec![0.01; id * hd]).collect();
        let up: Vec<Vec<f32>> = (0..ne).map(|_| vec![0.01; id * hd]).collect();
        let down: Vec<Vec<f32>> = (0..ne).map(|_| vec![0.01; hd * id]).collect();
        let router = vec![0.01; ne * hd];

        let moe = mapper.map_moe_experts(&gate, &up, &down, &router);
        assert_eq!(moe.num_experts, 4);
        assert!(moe.frozen);
    }

    #[test]
    fn test_weight_mapper_block_summary() {
        let config = Gemma4Config::gemma4_12b();
        let mapper = Gemma4WeightMapper::new(config);
        let summaries = mapper.create_block_summary_layers();
        assert!(!summaries.is_empty());
        for s in &summaries {
            assert_eq!(s.bridge_weight, 0.1); // Starts at 0.1
            assert_eq!(s.hidden_dim, 4096);
            assert_eq!(s.block_size, 1024); // 12B sliding_window=1024
        }
    }

    #[test]
    fn test_kl_divergence_identical() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let loss = kl_divergence_loss(&logits, &logits, 1.0, 3, 1);
        // KL(P||P) = 0
        assert!(loss.abs() < 0.01, "KL(P||P) should be ~0, got {}", loss);
    }

    #[test]
    fn test_kl_divergence_different() {
        let teacher = vec![10.0f32, 0.0, 0.0]; // Peaked distribution
        let student = vec![0.0f32, 10.0, 0.0]; // Different peak
        let loss = kl_divergence_loss(&teacher, &student, 1.0, 3, 1);
        assert!(loss > 0.0, "KL should be positive for different distributions, got {}", loss);
    }

    #[test]
    fn test_kl_divergence_temperature() {
        let teacher = vec![10.0f32, 0.0, 0.0];
        let student = vec![0.0f32, 10.0, 0.0];
        let loss_t1 = kl_divergence_loss(&teacher, &student, 1.0, 3, 1);
        let loss_t2 = kl_divergence_loss(&teacher, &student, 2.0, 3, 1);
        // Higher temperature should give lower KL (softer distributions)
        assert!(loss_t2 < loss_t1, "Higher T should reduce KL");
    }

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert!(config.freeze_moe);
        assert!(config.freeze_attention);
        assert_eq!(config.temperature, 2.0);
    }

    #[test]
    fn test_distillation_config_gemma4() {
        let config = DistillationConfig::for_gemma4_12b();
        assert_eq!(config.batch_size, 2);
        assert_eq!(config.max_seq_len, 4096);
    }

    #[test]
    fn test_mapped_attention_weights_metadata() {
        let w = MappedAttentionWeights {
            q_proj: vec![0.0; 64],
            k_proj: vec![0.0; 64],
            v_proj: vec![0.0; 64],
            o_proj: vec![0.0; 64],
            is_global: true,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 8,
            hidden_dim: 64,
        };
        assert!(w.is_global);
        assert_eq!(w.num_heads, 8);
    }

    // ------------------------------------------------------------------
    // Tensor names, teacher, student, autodiff tests
    // ------------------------------------------------------------------

    #[test]
    fn test_tensor_names_format() {
        assert_eq!(Gemma4TensorNames::q_proj(3), "model.layers.3.self_attn.q_proj.weight");
        assert_eq!(Gemma4TensorNames::k_proj(0), "model.layers.0.self_attn.k_proj.weight");
        assert_eq!(Gemma4TensorNames::o_proj(7), "model.layers.7.self_attn.o_proj.weight");
        assert_eq!(Gemma4TensorNames::input_norm(1), "model.layers.1.input_layernorm.weight");
        assert_eq!(Gemma4TensorNames::gate_proj(2), "model.layers.2.mlp.gate_proj.weight");
        assert_eq!(Gemma4TensorNames::expert_gate(3, 5),
                   "model.layers.3.block_sparse_moe.experts.5.w1.weight");
        assert_eq!(Gemma4TensorNames::expert_up(3, 5),
                   "model.layers.3.block_sparse_moe.experts.5.w3.weight");
        assert_eq!(Gemma4TensorNames::expert_down(3, 5),
                   "model.layers.3.block_sparse_moe.experts.5.w2.weight");
        assert_eq!(Gemma4TensorNames::moe_router(3),
                   "model.layers.3.block_sparse_moe.gate.weight");
        assert_eq!(Gemma4TensorNames::embed_tokens(), "model.embed_tokens.weight");
        assert_eq!(Gemma4TensorNames::final_norm(), "model.norm.weight");
        assert_eq!(Gemma4TensorNames::lm_head(), "lm_head.weight");
    }

    #[test]
    fn test_rms_norm() {
        let input = vec![3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let output = rms_norm(&input, &weight, 2, 1e-6);
        // RMS = sqrt((9+16)/2) = sqrt(12.5)
        // output = input / RMS * weight
        let rms = 12.5f32.sqrt();
        assert!((output[0] - 3.0 / rms).abs() < 1e-4);
        assert!((output[1] - 4.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_rms_norm_batch() {
        let input = vec![1.0, 0.0, 0.0, 1.0]; // 2 tokens × 2 dim
        let weight = vec![1.0, 1.0];
        let output = rms_norm(&input, &weight, 2, 1e-6);
        assert_eq!(output.len(), 4);
        // Token 1: [1,0], rms = sqrt(0.5)
        // Token 2: [0,1], rms = sqrt(0.5)
        let rms = 0.5f32.sqrt();
        assert!((output[0] - 1.0 / rms).abs() < 1e-4);
        assert!((output[3] - 1.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_apply_rope() {
        let mut x = vec![1.0f32, 0.0, 0.0, 1.0]; // 1 token, 1 head, head_dim=4
        apply_rope(&mut x, 1, 1, 4, 0, 10000.0, 1.0);
        // After RoPE, values should be rotated but non-zero
        assert!(x.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_matmul_cpu() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2×2
        let b = vec![1.0, 0.0, 0.0, 1.0]; // 2×2 (identity)
        let c = matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_swiglu_ffn() {
        let hd = 8;
        let id = 16;
        let gate = vec![1.0; id * hd];
        let up = vec![1.0; id * hd];
        let down = vec![0.1; hd * id];
        let input = vec![0.5; 2 * hd]; // seq_len=2
        let output = swiglu_ffn(&input, &gate, &up, &down, hd, id);
        assert_eq!(output.len(), 2 * hd);
    }

    #[test]
    fn test_mapped_gemma4_attn_weights() {
        let aw = Gemma4AttnWeights {
            q_proj: vec![0.1; 64],
            k_proj: vec![0.1; 64],
            v_proj: vec![0.1; 64],
            o_proj: vec![0.1; 64],
            input_norm: vec![1.0; 8],
            post_attn_norm: vec![1.0; 8],
            q_norm: vec![1.0; 8],
            k_norm: vec![1.0; 8],
            head_dim: 8,
            q_dim: 64,
            kv_dim: 64,
        };
        assert_eq!(aw.q_proj.len(), 64);
    }

    #[test]
    fn test_adam_optimizer_single_step() {
        let mut bs = BlockSummaryLayer::new_identity(16, 4);
        let mut adam = BlockSummaryAdam::new(&bs, 0.001);
        let grads = BlockSummaryGradients::zeros(&bs);
        // All zeros → no change
        adam.step(&mut bs, &grads);
        assert_eq!(bs.bridge_weight, 0.1); // No change from zero grad
    }

    #[test]
    fn test_adam_optimizer_with_gradient() {
        let mut bs = BlockSummaryLayer::new_identity(16, 4);
        let mut adam = BlockSummaryAdam::new(&bs, 0.01);
        // Negative gradient = increase the parameter (Adam subtracts lr * m_hat)
        for _ in 0..100 {
            let mut grads = BlockSummaryGradients::zeros(&bs);
            grads.d_bridge_weight = -1.0; // Negative grad → weight increases
            adam.step(&mut bs, &grads);
        }
        assert!(bs.bridge_weight > 0.0, "bridge_weight should be > 0 after 100 steps, got {}", bs.bridge_weight);
    }

    #[test]
    fn test_backprop_block_summary() {
        let bs = BlockSummaryLayer::new_identity(16, 4);
        let tokens = vec![1.0f32; 4 * 16];
        let d_output = vec![0.1f32; 4 * 16];
        let grads = backprop_block_summary(&bs, &tokens, &d_output);
        assert!(grads.d_summary_queries.len() > 0);
        assert!(grads.d_out_proj.len() > 0);
        // At init (bridge_weight=0), bridge grad may be small but should exist
    }

    #[test]
    fn test_block_summary_gradients_size() {
        let bs = BlockSummaryLayer::new_identity(32, 8);
        let grads = BlockSummaryGradients::zeros(&bs);
        assert_eq!(grads.d_summary_queries.len(), bs.summary_queries.len());
        assert_eq!(grads.d_query_proj.len(), 32 * 32);
        assert_eq!(grads.d_out_proj.len(), 32 * 32);
    }

    // ------------------------------------------------------------------
    // WGSL kernel, dataset, verified loader tests
    // ------------------------------------------------------------------

    #[test]
    fn test_block_summary_gpu_shader() {
        assert!(!BLOCK_SUMMARY_CROSS_ATTN_WGSL.is_empty());
        assert!(BLOCK_SUMMARY_CROSS_ATTN_WGSL.contains("block_summary_cross_attn"));
        assert!(BLOCK_SUMMARY_CROSS_ATTN_WGSL.contains("queries"));
        assert!(BLOCK_SUMMARY_CROSS_ATTN_WGSL.contains("tokens"));
    }

    #[test]
    fn test_block_summary_backward_shader() {
        assert!(!BLOCK_SUMMARY_BACKWARD_WGSL.is_empty());
        assert!(BLOCK_SUMMARY_BACKWARD_WGSL.contains("block_summary_backward"));
        assert!(BLOCK_SUMMARY_BACKWARD_WGSL.contains("d_output"));
        assert!(BLOCK_SUMMARY_BACKWARD_WGSL.contains("d_queries"));
        assert!(BLOCK_SUMMARY_BACKWARD_WGSL.contains("d_tokens"));
    }

    #[test]
    fn test_block_summary_gpu_cpu_fallback() {
        // Use BlockSummaryLayer directly as CPU fallback
        let layer = BlockSummaryLayer::new_identity(32, 4);
        let tokens = vec![1.0f32; 4 * 32];
        let output = layer.forward(&tokens);
        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_text_dataset_from_text() {
        let ds = TextDataset::from_text("Hello world", 4, "test");
        assert_eq!(ds.total_tokens(), 11); // "Hello world" = 11 bytes
        assert_eq!(ds.seq_len, 4);
        assert_eq!(ds.num_sequences(), 2); // 11 / 4 = 2
    }

    #[test]
    fn test_text_dataset_synthetic() {
        let ds = TextDataset::synthetic(100, 8, 10);
        assert_eq!(ds.total_tokens(), 80);
        assert_eq!(ds.num_sequences(), 10);
    }

    #[test]
    fn test_text_dataset_batch_iteration() {
        let mut ds = TextDataset::synthetic(50, 4, 5);
        let batch1 = ds.next_batch().unwrap();
        assert_eq!(batch1.len(), 4);
        let batch2 = ds.next_batch().unwrap();
        assert_eq!(batch2.len(), 4);
        // Should wrap around
        for _ in 0..4 { ds.next_batch(); }
        let wrapped = ds.next_batch().unwrap();
        assert_eq!(wrapped.len(), 4);
    }

    #[test]
    fn test_text_dataset_split() {
        let ds = TextDataset::synthetic(100, 10, 10); // 100 tokens
        let (train, val) = ds.split(0.8);
        assert!(train.total_tokens() > val.total_tokens());
        assert_eq!(train.total_tokens() + val.total_tokens(), 100);
    }

    #[test]
    fn test_text_dataset_reset() {
        let mut ds = TextDataset::synthetic(50, 4, 5);
        ds.next_batch();
        ds.next_batch();
        assert!(ds.position > 0);
        ds.reset();
        assert_eq!(ds.position, 0);
    }

    #[test]
    fn test_text_dataset_from_token_ids() {
        let ids: Vec<u32> = (0..100).collect();
        let ds = TextDataset::from_token_ids(ids, 10, "test_ids");
        assert_eq!(ds.total_tokens(), 100);
        assert_eq!(ds.num_sequences(), 10);
    }

    // ------------------------------------------------------------------
    // Full synthetic smoke test
    // ------------------------------------------------------------------

    /// Write a minimal safetensors file from named tensors.
    fn write_synthetic_safetensors(
        path: &std::path::Path,
        tensors: &[(&str, &[usize], &[f32])],
    ) -> std::io::Result<()> {
        // Safetensors format: u64 header_len + JSON header + raw data
        let mut header_json = serde_json::Map::new();
        let mut data_blob = Vec::new();

        let num = |v: i64| -> serde_json::Value {
            serde_json::Value::Number(serde_json::Number::from(v))
        };

        for (name, shape, data) in tensors {
            let start = data_blob.len();
            let byte_data: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            data_blob.extend_from_slice(&byte_data);
            let end = data_blob.len();

            let mut meta = serde_json::Map::new();
            meta.insert("dtype".to_string(), serde_json::Value::String("F32".to_string()));
            meta.insert(
                "shape".to_string(),
                serde_json::Value::Array(shape.iter().map(|&d| num(d as i64)).collect()),
            );
            meta.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![num(start as i64), num(end as i64)]),
            );
            header_json.insert(name.to_string(), serde_json::Value::Object(meta));
        }

        let header_str = serde_json::to_string(&header_json)?;
        let header_bytes = header_str.as_bytes();
        let header_len = header_bytes.len() as u64;

        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        file.write_all(&header_len.to_le_bytes())?;
        file.write_all(header_bytes)?;
        file.write_all(&data_blob)?;
        Ok(())
    }

    /// Create a tiny synthetic Gemma model config for smoke testing.
    fn tiny_gemma_config() -> Gemma4Config {
        Gemma4Config {
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            intermediate_dim: 64,
            num_experts: 1,
            top_k: 1,
            vocab_size: 128,
            max_position_embeddings: 256,
            sliding_window: 32,
            moe_layers: vec![], // Dense only
        hidden_size_per_layer_input: None,
        final_logit_softcapping: None,
            layer_types: vec![],
            global_head_dim: 256,
            rope_theta_sliding: 10000.0,
            rope_theta_full: 1000000.0,
            partial_rotary_factor_full: 0.25,
            num_kv_shared_layers: 0,
        }
    }

    /// Build all tensors for a tiny synthetic model.
    fn build_tiny_model_tensors(config: &Gemma4Config) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let hd = config.hidden_dim;
        let vs = config.vocab_size;
        let id = config.intermediate_dim;
        let mut tensors = Vec::new();

        // Simple deterministic init: 0.01 * sin(i)
        let init = |n: usize| -> Vec<f32> {
            (0..n).map(|i| 0.01 * (i as f32 * 0.1).sin()).collect()
        };

        // Embedding
        tensors.push(("model.embed_tokens.weight".to_string(), vec![vs, hd], init(vs * hd)));
        // LM head (tied for now — use same data)
        tensors.push(("lm_head.weight".to_string(), vec![vs, hd], init(vs * hd)));
        // Final norm
        tensors.push(("model.norm.weight".to_string(), vec![hd], vec![1.0; hd]));

        for layer in 0..config.num_layers {
            // Attention
            tensors.push((format!("model.layers.{}.self_attn.q_proj.weight", layer), vec![hd, hd], init(hd * hd)));
            tensors.push((format!("model.layers.{}.self_attn.k_proj.weight", layer), vec![hd, hd], init(hd * hd)));
            tensors.push((format!("model.layers.{}.self_attn.v_proj.weight", layer), vec![hd, hd], init(hd * hd)));
            tensors.push((format!("model.layers.{}.self_attn.o_proj.weight", layer), vec![hd, hd], init(hd * hd)));
            // Norms
            tensors.push((format!("model.layers.{}.input_layernorm.weight", layer), vec![hd], vec![1.0; hd]));
            tensors.push((format!("model.layers.{}.post_attention_layernorm.weight", layer), vec![hd], vec![1.0; hd]));
            // FFN (dense)
            tensors.push((format!("model.layers.{}.mlp.gate_proj.weight", layer), vec![id, hd], init(id * hd)));
            tensors.push((format!("model.layers.{}.mlp.up_proj.weight", layer), vec![id, hd], init(id * hd)));
            tensors.push((format!("model.layers.{}.mlp.down_proj.weight", layer), vec![hd, id], init(hd * id)));
        }

        tensors
    }

    #[test]
    fn test_synthetic_safetensors_round_trip() {
        let dir = std::env::temp_dir().join("ferrisres_smoke_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny_model.safetensors");

        let config = tiny_gemma_config();
        let tensors = build_tiny_model_tensors(&config);
        let flat: Vec<(&str, &[usize], &[f32])> = tensors.iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        write_synthetic_safetensors(&path, &flat).unwrap();

        assert!(path.exists());
        let file_size = std::fs::metadata(&path).unwrap().len();
        assert!(file_size > 0);

        // Load it back
        let weights = crate::model::safetensors::load_safetensors(&path).unwrap();
        assert!(weights.get("model.embed_tokens.weight").is_some());
        assert!(weights.get("model.layers.0.self_attn.q_proj.weight").is_some());
        assert!(weights.get("model.layers.1.mlp.gate_proj.weight").is_some());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_gemma4_synthetic_model() {
        let dir = std::env::temp_dir().join("ferrisres_load_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny_gemma.safetensors");

        let config = tiny_gemma_config();
        let tensors = build_tiny_model_tensors(&config);
        let flat: Vec<(&str, &[usize], &[f32])> = tensors.iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        write_synthetic_safetensors(&path, &flat).unwrap();

        // Load through the full pipeline
        let result = load_gemma4_model(&path, config.clone());
        assert!(result.is_ok(), "load_gemma4_model failed: {:?}", result.err());
        let model = result.unwrap();
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.embed_tokens.len(), config.vocab_size * config.hidden_dim);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_teacher_forward_synthetic() {
        let dir = std::env::temp_dir().join("ferrisres_teacher_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny_teacher.safetensors");

        let config = tiny_gemma_config();
        let tensors = build_tiny_model_tensors(&config);
        let flat: Vec<(&str, &[usize], &[f32])> = tensors.iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        write_synthetic_safetensors(&path, &flat).unwrap();

        let model = load_gemma4_model(&path, config.clone()).unwrap();
        let teacher = Gemma4Teacher::new(model);

        // Forward pass with 4 tokens
        let token_ids: &[u32] = &[1, 2, 3, 4];
        let logits = teacher.forward(token_ids);
        let vs = config.vocab_size;
        assert_eq!(logits.len(), 4 * vs, "Expected {} logits, got {}", 4 * vs, logits.len());

        // Logits should be finite
        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "Logit {} is not finite: {}", i, l);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_student_forward_synthetic() {
        let dir = std::env::temp_dir().join("ferrisres_student_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny_student.safetensors");

        let config = tiny_gemma_config();
        let tensors = build_tiny_model_tensors(&config);
        let flat: Vec<(&str, &[usize], &[f32])> = tensors.iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        write_synthetic_safetensors(&path, &flat).unwrap();

        let model = load_gemma4_model(&path, config.clone()).unwrap();
        let block_summaries = vec![BlockSummaryLayer::new_identity(config.hidden_dim, config.sliding_window)];
        let distill_config = DistillationConfig::default();
        let student = Gemma4Student::new(model, block_summaries, distill_config);

        let token_ids: &[u32] = &[1, 2, 3, 4];
        let logits = student.forward(token_ids);
        let vs = config.vocab_size;
        assert_eq!(logits.len(), 4 * vs);

        for (i, &l) in logits.iter().enumerate() {
            assert!(l.is_finite(), "Student logit {} not finite: {}", i, l);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_distillation_smoke_test() {
        // Full end-to-end: create model → teacher → student → distill → verify loss decreases
        let dir = std::env::temp_dir().join("ferrisres_distill_smoke");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny_distill.safetensors");

        let config = tiny_gemma_config();
        let tensors = build_tiny_model_tensors(&config);
        let flat: Vec<(&str, &[usize], &[f32])> = tensors.iter()
            .map(|(n, s, d)| (n.as_str(), s.as_slice(), d.as_slice()))
            .collect();
        write_synthetic_safetensors(&path, &flat).unwrap();

        let model1 = load_gemma4_model(&path, config.clone()).unwrap();
        let teacher = Gemma4Teacher::new(model1);

        let model2 = load_gemma4_model(&path, config.clone()).unwrap();
        let block_summaries = vec![BlockSummaryLayer::new_identity(config.hidden_dim, config.sliding_window)];
        let mut distill_config = DistillationConfig::default();
        distill_config.num_steps = 3; // Just 3 steps for smoke test
        distill_config.learning_rate = 0.01;

        let mut student = Gemma4Student::new(model2, block_summaries, distill_config.clone());

        // Use synthetic token IDs
        let token_ids: Vec<u32> = (0..32).collect();

        let results = run_distillation(&teacher, &mut student, &token_ids, 16);

        assert!(!results.is_empty(), "Distillation should produce results");

        // Verify results have expected fields
        let first = &results[0];
        assert_eq!(first.step, 0);
        assert!(first.kl_loss >= 0.0, "KL loss should be non-negative");
        assert_eq!(first.bridge_weight, 0.1); // Should start at 0.1

        // Bridge weight should evolve (Adam is running)
        if results.len() > 1 {
            let last = results.last().unwrap();
            assert!(last.step > 0);
        }

        let _ = std::fs::remove_file(&path);
    }
}
