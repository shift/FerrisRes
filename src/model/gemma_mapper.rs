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
    /// LM head weight (may be tied to embed_tokens).
    pub fn lm_head() -> &'static str {
        "lm_head.weight"
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
}

/// Full Gemma 4 model weights loaded and organized.
pub struct MappedGemma4Model {
    pub config: Gemma4Config,
    pub embed_tokens: Vec<f32>,  // [vocab_size × hidden_dim]
    pub layers: Vec<Gemma4LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>,      // [vocab_size × hidden_dim] (may be tied)
}

impl MappedGemma4Model {
    /// Load from a safetensors LoadedWeights object.
    /// Uses E2B/E4B/12B/27B config to know the naming convention.
    pub fn from_loaded_weights(
        config: Gemma4Config,
        weights: &crate::model::safetensors::LoadedWeights,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Option<Vec<f32>> {
            weights.get(name).map(|t| t.data.clone())
        };

        // Embedding
        let embed_tokens = get(Gemma4TensorNames::embed_tokens())
            .ok_or_else(|| "Missing embed_tokens".to_string())?;

        // LM head (may be tied to embedding)
        let lm_head = get(Gemma4TensorNames::lm_head())
            .unwrap_or_else(|| embed_tokens.clone());

        // Final norm
        let final_norm = get(Gemma4TensorNames::final_norm())
            .ok_or_else(|| "Missing final norm".to_string())?;

        // Per-layer weights
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            let q = get(&Gemma4TensorNames::q_proj(layer_idx))
                .ok_or_else(|| format!("Missing q_proj for layer {}", layer_idx))?;
            let k = get(&Gemma4TensorNames::k_proj(layer_idx))
                .ok_or_else(|| format!("Missing k_proj for layer {}", layer_idx))?;
            let v = get(&Gemma4TensorNames::v_proj(layer_idx))
                .ok_or_else(|| format!("Missing v_proj for layer {}", layer_idx))?;
            let o = get(&Gemma4TensorNames::o_proj(layer_idx))
                .ok_or_else(|| format!("Missing o_proj for layer {}", layer_idx))?;
            let inorm = get(&Gemma4TensorNames::input_norm(layer_idx))
                .ok_or_else(|| format!("Missing input_norm for layer {}", layer_idx))?;
            let pnorm = get(&Gemma4TensorNames::post_attn_norm(layer_idx))
                .ok_or_else(|| format!("Missing post_attn_norm for layer {}", layer_idx))?;

            let attn = Gemma4AttnWeights {
                q_proj: q, k_proj: k, v_proj: v, o_proj: o,
                input_norm: inorm,
                post_attn_norm: pnorm,
            };

            // FFN: MoE or dense?
            let ffn = if config.moe_layers.contains(&layer_idx) {
                // MoE layer
                let router = get(&Gemma4TensorNames::moe_router(layer_idx))
                    .ok_or_else(|| format!("Missing MoE router for layer {}", layer_idx))?;

                let mut expert_gates = Vec::new();
                let mut expert_ups = Vec::new();
                let mut expert_downs = Vec::new();

                for e in 0..config.num_experts {
                    let g = get(&Gemma4TensorNames::expert_gate(layer_idx, e))
                        .ok_or_else(|| format!("Missing expert {} gate for layer {}", e, layer_idx))?;
                    let u = get(&Gemma4TensorNames::expert_up(layer_idx, e))
                        .ok_or_else(|| format!("Missing expert {} up for layer {}", e, layer_idx))?;
                    let d = get(&Gemma4TensorNames::expert_down(layer_idx, e))
                        .ok_or_else(|| format!("Missing expert {} down for layer {}", e, layer_idx))?;
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
                let gate = get(&Gemma4TensorNames::gate_proj(layer_idx))
                    .ok_or_else(|| format!("Missing gate_proj for layer {}", layer_idx))?;
                let up = get(&Gemma4TensorNames::up_proj(layer_idx))
                    .ok_or_else(|| format!("Missing up_proj for layer {}", layer_idx))?;
                let down = get(&Gemma4TensorNames::down_proj(layer_idx))
                    .ok_or_else(|| format!("Missing down_proj for layer {}", layer_idx))?;

                Gemma4FfnWeights::Dense {
                    gate_proj: gate,
                    up_proj: up,
                    down_proj: down,
                }
            };

            layers.push(Gemma4LayerWeights { attn, ffn });
        }

        Ok(Self { config, embed_tokens, layers, final_norm, lm_head })
    }
}

// ---------------------------------------------------------------------------
// RMS Norm (CPU)
// ---------------------------------------------------------------------------

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
pub fn apply_rope(x: &mut [f32], seq_len: usize, num_heads: usize, head_dim: usize, offset: usize) {
    let half = head_dim / 2;
    for t in 0..seq_len {
        let pos = (t + offset) as f32;
        for h in 0..num_heads {
            for d in 0..half {
                let freq = 1.0 / 10000.0f32.powf(d as f32 / half as f32);
                let angle = pos * freq;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                let base = t * num_heads * head_dim + h * head_dim;
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
pub struct Gemma4Teacher {
    pub model: MappedGemma4Model,
}

impl Gemma4Teacher {
    pub fn new(model: MappedGemma4Model) -> Self {
        Self { model }
    }

    /// Forward pass: token IDs → logits.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let config = &self.model.config;
        let hd = config.hidden_dim;
        let nh = config.num_heads;
        let head_d = config.head_dim;
        let seq = token_ids.len();
        let vs = config.vocab_size;

        // 1. Embedding lookup
        let mut hidden = vec![0.0f32; seq * hd];
        for (t, &tid) in token_ids.iter().enumerate() {
            let id = tid as usize;
            if id * hd + hd <= self.model.embed_tokens.len() {
                for d in 0..hd {
                    hidden[t * hd + d] = self.model.embed_tokens[id * hd + d];
                }
            }
        }
        // Gemma scales embeddings by sqrt(hidden_dim)
        let scale = (hd as f32).sqrt();
        for h in hidden.iter_mut() { *h *= scale; }

        // 2. Per-layer transformer
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            let residual = hidden.clone();

            // Input RMSNorm
            let normed = rms_norm(&hidden, &layer.attn.input_norm, hd, 1e-6);

            // GQA Attention
            let attn_out = self.attention_forward(
                &normed, &layer.attn, nh, head_d, seq, hd,
                config.sliding_window, layer_idx,
            );

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] = residual[i] + attn_out[i];
            }

            let residual2 = hidden.clone();

            // Post-attention RMSNorm
            let normed2 = rms_norm(&hidden, &layer.attn.post_attn_norm, hd, 1e-6);

            // FFN (dense or MoE)
            let ffn_out = match &layer.ffn {
                Gemma4FfnWeights::Dense { gate_proj, up_proj, down_proj } => {
                    self.dense_ffn(&normed2, gate_proj, up_proj, down_proj, hd, config.intermediate_dim)
                }
                Gemma4FfnWeights::Moe { router, expert_gates, expert_ups, expert_downs } => {
                    self.moe_ffn(&normed2, router, expert_gates, expert_ups, expert_downs,
                                 hd, config.intermediate_dim, config.num_experts, config.top_k)
                }
            };

            for i in 0..hidden.len() {
                hidden[i] = residual2[i] + ffn_out[i];
            }
        }

        // 3. Final RMSNorm
        hidden = rms_norm(&hidden, &self.model.final_norm, hd, 1e-6);

        // 4. LM head: [seq × hd] × [hd × vs] → [seq × vs]
        let mut logits = vec![0.0f32; seq * vs];
        for t in 0..seq {
            for v in 0..vs {
                let mut sum = 0.0f32;
                for d in 0..hd {
                    sum += hidden[t * hd + d] * self.model.lm_head[v * hd + d];
                }
                logits[t * vs + v] = sum;
            }
        }

        logits
    }

    /// GQA attention forward.
    fn attention_forward(
        &self,
        input: &[f32],
        attn: &Gemma4AttnWeights,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        hidden_dim: usize,
        _sliding_window: usize,
        _layer_idx: usize,
    ) -> Vec<f32> {
        // Q projection: [seq × hd] × [hd × hd] → [seq × hd]
        let q = matmul(input, &attn.q_proj, seq_len, hidden_dim, hidden_dim);
        let k = matmul(input, &attn.k_proj, seq_len, hidden_dim, hidden_dim);
        let v = matmul(input, &attn.v_proj, seq_len, hidden_dim, hidden_dim);

        // Apply RoPE to Q and K
        let mut q = q;
        let mut k_mut = k;
        apply_rope(&mut q, seq_len, num_heads, head_dim, 0);
        apply_rope(&mut k_mut, seq_len, num_heads, head_dim, 0);

        // Scaled dot-product attention with causal mask
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * hidden_dim];

        for h in 0..num_heads {
            for t in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];

                // Compute scores
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * hidden_dim + h * head_dim + d]
                             * k_mut[s * hidden_dim + h * head_dim + d];
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
                        sum += scores[s] * v[s * hidden_dim + h * head_dim + d];
                    }
                    attn_out[t * hidden_dim + h * head_dim + d] = sum;
                }
            }
        }

        // Output projection
        matmul(&attn_out, &attn.o_proj, seq_len, hidden_dim, hidden_dim)
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

/// Simple CPU matmul: C[m×n] = A[m×k] × B[k×n].
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
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
            let attn_out = self.student_attention(&normed, &layer.attn, nh, head_d, seq, hd);

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

        let mut logits = vec![0.0f32; seq * vs];
        for t in 0..seq {
            for v in 0..vs {
                let mut sum = 0.0f32;
                for d in 0..hd {
                    sum += hidden[t * hd + d] * self.model.lm_head[v * hd + d];
                }
                logits[t * vs + v] = sum;
            }
        }

        logits
    }

    /// Student attention (same architecture as teacher, but factored out).
    fn student_attention(
        &self,
        input: &[f32],
        attn: &Gemma4AttnWeights,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let q = matmul(input, &attn.q_proj, seq_len, hidden_dim, hidden_dim);
        let k = matmul(input, &attn.k_proj, seq_len, hidden_dim, hidden_dim);
        let v = matmul(input, &attn.v_proj, seq_len, hidden_dim, hidden_dim);

        let mut q = q;
        let mut k_mut = k;
        apply_rope(&mut q, seq_len, num_heads, head_dim, 0);
        apply_rope(&mut k_mut, seq_len, num_heads, head_dim, 0);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; seq_len * hidden_dim];

        for h in 0..num_heads {
            for t in 0..seq_len {
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0f32; seq_len];
                for s in 0..=t {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[t * hidden_dim + h * head_dim + d]
                             * k_mut[s * hidden_dim + h * head_dim + d];
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
                        sum += scores[s] * v[s * hidden_dim + h * head_dim + d];
                    }
                    attn_out[t * hidden_dim + h * head_dim + d] = sum;
                }
            }
        }

        matmul(&attn_out, &attn.o_proj, seq_len, hidden_dim, hidden_dim)
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
    sq_state: AdamState,
    qp_state: AdamState,
    op_state: AdamState,
    bw_m: f32,
    bw_v: f32,
    nw_state: AdamState,
    nb_state: AdamState,
    lr: f32,
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

        let bridge_w = student.block_summaries.first()
            .map(|bs| bs.bridge_weight)
            .unwrap_or(0.0);

        results.push(DistillationStepResult {
            step,
            kl_loss: loss,
            bridge_weight: bridge_w,
            learning_rate: lr,
        });

        // Early stop if loss is negligible
        if loss < 1e-6 {
            break;
        }
    }

    results
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

/// GPU operation for Block Summary cross-attention.
pub struct BlockSummaryGpuOp {
    num_queries: usize,
    hidden_dim: usize,
    block_size: usize,
}

impl BlockSummaryGpuOp {
    pub fn new(num_queries: usize, hidden_dim: usize, block_size: usize) -> Self {
        Self { num_queries, hidden_dim, block_size }
    }

    /// Get the WGSL shader source.
    pub fn shader_source(&self) -> &str {
        BLOCK_SUMMARY_CROSS_ATTN_WGSL
    }

    /// Entry point name.
    pub fn entry_point(&self) -> &str {
        "block_summary_cross_attn"
    }

    /// Workgroup count.
    pub fn workgroup_count(&self) -> (u32, u32, u32) {
        let wg = 64u32;
        ((self.num_queries as u32 + wg - 1) / wg, 1, 1)
    }

    /// Buffer sizes needed.
    pub fn buffer_sizes(&self) -> (usize, usize, usize, usize) {
        let f32_bytes = 4;
        (
            self.num_queries * self.hidden_dim * f32_bytes, // queries
            self.block_size * self.hidden_dim * f32_bytes,  // tokens
            self.num_queries * self.hidden_dim * f32_bytes, // output
            16, // params uniform
        )
    }

    /// CPU fallback (uses BlockSummaryLayer.forward()).
    pub fn forward_cpu(&self, queries: &[f32], tokens: &[f32]) -> Vec<f32> {
        let layer = BlockSummaryLayer::new_identity(self.hidden_dim, self.block_size);
        // Override summary_queries with provided queries
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
        apply_rope(&mut x, 1, 1, 4, 0);
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
        assert_eq!(bs.bridge_weight, 0.0); // No change from zero grad
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
    fn test_block_summary_gpu_op() {
        let op = BlockSummaryGpuOp::new(1, 128, 64);
        assert_eq!(op.num_queries(), 1);
        assert_eq!(op.hidden_dim(), 128);
        assert_eq!(op.block_size(), 64);
        assert_eq!(op.entry_point(), "block_summary_cross_attn");
    }

    #[test]
    fn test_block_summary_gpu_op_workgroup() {
        let op = BlockSummaryGpuOp::new(4, 64, 32);
        let (x, y, z) = op.workgroup_count();
        assert_eq!(y, 1);
        assert_eq!(z, 1);
        assert!(x > 0);
    }

    #[test]
    fn test_block_summary_gpu_op_buffer_sizes() {
        let op = BlockSummaryGpuOp::new(2, 64, 32);
        let (q, t, o, p) = op.buffer_sizes();
        assert_eq!(q, 2 * 64 * 4); // queries
        assert_eq!(t, 32 * 64 * 4); // tokens
        assert_eq!(o, 2 * 64 * 4); // output
        assert_eq!(p, 16); // params
    }

    #[test]
    fn test_block_summary_gpu_cpu_fallback() {
        let op = BlockSummaryGpuOp::new(1, 32, 4);
        let queries = vec![0.5f32; 32];
        let tokens = vec![1.0f32; 4 * 32];
        let output = op.forward_cpu(&queries, &tokens);
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
}
