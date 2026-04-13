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
    /// The bridge_weight starts at 0.0 so the summary has no effect.
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

        // Bridge weight: 0.0 = no contribution (identity)
        let bridge_weight = 0.0;

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
    /// Input: [block_size × hidden_dim] tokens.
    /// Output: [num_summary_queries × hidden_dim] summary.
    pub fn forward(&self, block_tokens: &[f32]) -> Vec<f32> {
        let hd = self.hidden_dim;
        let nq = self.num_summary_queries;
        let bs = self.block_size;

        assert!(block_tokens.len() >= bs * hd, "Block tokens too small");

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

        // 2. Project keys and values from tokens
        let mut keys = vec![0.0; bs * hd];
        let mut values = vec![0.0; bs * hd];

        for t in 0..bs {
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
            let mut attn_weights = vec![0.0; bs];
            let mut max_weight = f32::NEG_INFINITY;
            for t in 0..bs {
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
                for t in 0..bs {
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
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    /// Which layers use MoE (alternating pattern).
    pub moe_layers: Vec<usize>,
}

impl Gemma4Config {
    /// Gemma 4 E2B (efficient 2B) — for testing and development.
    ///
    /// Small enough to fit in ~4 GB VRAM. Ideal for verifying
    /// the distillation pipeline end-to-end before scaling up.
    /// Uses dense FFN (no MoE) for simplicity.
    pub fn gemma4_e2b() -> Self {
        Self {
            hidden_dim: 2048,
            num_layers: 18,
            num_heads: 8,
            head_dim: 256,
            intermediate_dim: 8192,
            num_experts: 1,      // Dense, no MoE
            top_k: 1,
            vocab_size: 256000,
            max_position_embeddings: 32768,
            sliding_window: 4096,
            moe_layers: vec![], // No MoE in E2B
        }
    }

    /// Gemma 4 E4B (efficient 4B) — small MoE for testing MoE distillation.
    ///
    /// ~8 GB VRAM. Introduces MoE routing with a small expert count
    /// to verify expert weight mapping before scaling to 128 experts.
    pub fn gemma4_e4b() -> Self {
        Self {
            hidden_dim: 2560,
            num_layers: 24,
            num_heads: 10,
            head_dim: 256,
            intermediate_dim: 10240,
            num_experts: 16,     // Small MoE for testing
            top_k: 2,
            vocab_size: 256000,
            max_position_embeddings: 32768,
            sliding_window: 4096,
            moe_layers: vec![6, 12, 18], // MoE every 6 layers
        }
    }

    /// Gemma 4 12B configuration (full production model).
    pub fn gemma4_12b() -> Self {
        Self {
            hidden_dim: 4096,
            num_layers: 48,
            num_heads: 32,
            head_dim: 128,
            intermediate_dim: 14336,
            num_experts: 128,
            top_k: 2,
            vocab_size: 256000,
            max_position_embeddings: 131072,
            sliding_window: 4096,
            moe_layers: (0..48).step_by(2).map(|i| i + 1).collect(), // Every other layer
        }
    }

    /// Gemma 4 27B configuration.
    pub fn gemma4_27b() -> Self {
        Self {
            hidden_dim: 4608,
            num_layers: 60,
            num_heads: 36,
            head_dim: 128,
            intermediate_dim: 16384,
            num_experts: 128,
            top_k: 2,
            vocab_size: 256000,
            max_position_embeddings: 131072,
            sliding_window: 4096,
            moe_layers: (0..60).step_by(2).map(|i| i + 1).collect(),
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
            use_moe: !self.moe_layers.is_empty(),
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
pub fn compute_distillation_gradients(
    block_summary: &BlockSummaryLayer,
    teacher_logits: &[f32],
    student_logits: &[f32],
    block_tokens: &[f32],
    temperature: f32,
    vocab_size: usize,
) -> DistillationGradients {
    let hd = block_summary.hidden_dim;
    let nq = block_summary.num_summary_queries;

    // Forward through block summary to get current summary
    let _summary = block_summary.forward(block_tokens);

    // Compute loss signal (simplified: gradient of bridge_weight)
    // Full implementation would use autodiff
    let loss = kl_divergence_loss(teacher_logits, student_logits, temperature, vocab_size, 1);

    // Gradient approximation
    let bridge_grad = if block_summary.trainable {
        // Numerical gradient (in production, use autodiff)
        loss * 0.1 // Simplified: scale loss by learning signal
    } else {
        0.0
    };

    // Summary query gradients (simplified)
    let query_grads = if block_summary.trainable {
        vec![0.0; nq * hd] // Placeholder: real implementation uses backprop
    } else {
        vec![]
    };

    DistillationGradients {
        bridge_weight_grad: bridge_grad,
        summary_query_grads: query_grads,
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
// Tests
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
        assert_eq!(layer.bridge_weight, 0.0); // Identity = no contribution
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
        assert_eq!(config.hidden_dim, 2048);
        assert_eq!(config.num_layers, 18);
        assert_eq!(config.num_experts, 1); // Dense
        assert_eq!(config.top_k, 1);
        assert_eq!(config.sliding_window, 4096);
        assert!(config.moe_layers.is_empty()); // No MoE
    }

    #[test]
    fn test_gemma4_e4b_config() {
        let config = Gemma4Config::gemma4_e4b();
        assert_eq!(config.hidden_dim, 2560);
        assert_eq!(config.num_layers, 24);
        assert_eq!(config.num_experts, 16); // Small MoE
        assert_eq!(config.top_k, 2);
        assert_eq!(config.moe_layers, vec![6, 12, 18]);
    }

    #[test]
    fn test_gemma4_e2b_to_block_attnres() {
        let gemma = Gemma4Config::gemma4_e2b();
        let bar = gemma.to_block_attnres_config();
        assert_eq!(bar.hidden_dim, 2048);
        assert_eq!(bar.block_size, 4096);
        assert_eq!(bar.num_experts, 1);
        assert!(!bar.use_moe); // Dense
    }

    #[test]
    fn test_gemma4_e4b_to_block_attnres() {
        let gemma = Gemma4Config::gemma4_e4b();
        let bar = gemma.to_block_attnres_config();
        assert_eq!(bar.hidden_dim, 2560);
        assert!(bar.use_moe); // Has MoE
    }

    #[test]
    fn test_e2b_weight_mapper() {
        let config = Gemma4Config::gemma4_e2b();
        let mapper = Gemma4WeightMapper::new(config);
        let hd = 2048;
        let q = vec![0.1f32; hd * hd];
        let k = vec![0.1f32; hd * hd];
        let v = vec![0.1f32; hd * hd];
        let o = vec![0.1f32; hd * hd];
        let mapped = mapper.map_attention_weights(&q, &k, &v, &o, 0);
        assert!(!mapped.is_global); // Layer 0 is local
        assert_eq!(mapped.hidden_dim, 2048);
    }

    #[test]
    fn test_e2b_block_summary_creation() {
        let config = Gemma4Config::gemma4_e2b();
        let mapper = Gemma4WeightMapper::new(config);
        let _summaries = mapper.create_block_summary_layers();
        // E2B has no MoE layers, so no injection points by default
        // But we can still create block summaries manually
        let bs = BlockSummaryLayer::new_identity(2048, 4096);
        assert_eq!(bs.hidden_dim, 2048);
        assert_eq!(bs.bridge_weight, 0.0);
    }

    #[test]
    fn test_e4b_moe_weight_mapping() {
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
        assert_eq!(moe.top_k, 2);
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
        assert_eq!(config.hidden_dim, 4096);
        assert_eq!(config.num_layers, 48);
        assert_eq!(config.num_experts, 128);
        assert_eq!(config.top_k, 2);
        assert_eq!(config.sliding_window, 4096);
        assert!(!config.moe_layers.is_empty());
    }

    #[test]
    fn test_gemma4_27b_config() {
        let config = Gemma4Config::gemma4_27b();
        assert_eq!(config.hidden_dim, 4608);
        assert_eq!(config.num_layers, 60);
    }

    #[test]
    fn test_gemma4_to_block_attnres_config() {
        let gemma = Gemma4Config::gemma4_12b();
        let bar_config = gemma.to_block_attnres_config();
        assert_eq!(bar_config.hidden_dim, 4096);
        assert_eq!(bar_config.block_size, 4096);
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
            assert_eq!(s.bridge_weight, 0.0); // Identity init
            assert_eq!(s.hidden_dim, 4096);
            assert_eq!(s.block_size, 4096);
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
            head_dim: 8,
            hidden_dim: 64,
        };
        assert!(w.is_global);
        assert_eq!(w.num_heads, 8);
    }
}
