//! CPU-side token generation for CpuBlockAttnResModel.
//!
//! Provides prefill (batch), incremental decode, and streaming generation
//! using a CPU KV cache with optional TurboQuant + recurrent block summaries.

use std::collections::HashMap;
use crate::error::Result;
use crate::inference::logit_processors::{LogitProcessor, LogitProcessorConfig};
use crate::model::cpu_block_attn_res::CpuBlockAttnResModel;
use crate::model::gemma_mapper::{apply_rope, apply_rope_gqa, gelu_tanh, rms_norm, matmul, per_head_rms_norm, per_head_rms_norm_no_scale};

/// Per-layer CPU KV cache: stores K and V as contiguous f32 buffers.
///
/// Layout: [max_seq_len, kv_dim] where kv_dim = num_kv_heads * head_dim.
/// Supports append, random access, and reset.
pub struct CpuLayerKVCache {
    key_cache: Vec<f32>,     // [max_seq_len, kv_dim]
    value_cache: Vec<f32>,   // [max_seq_len, kv_dim]
    current_len: usize,
    max_seq_len: usize,
    kv_dim: usize,           // num_kv_heads * head_dim
}

impl CpuLayerKVCache {
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        Self {
            key_cache: vec![0.0f32; max_seq_len * kv_dim],
            value_cache: vec![0.0f32; max_seq_len * kv_dim],
            current_len: 0,
            max_seq_len,
            kv_dim,
        }
    }

    /// Append K/V for `seq_len` tokens. K and V are [seq_len, kv_dim].
    pub fn append(&mut self, k: &[f32], v: &[f32], seq_len: usize) {
        let pos = self.current_len;
        assert!(pos + seq_len <= self.max_seq_len, "KV cache overflow");
        for t in 0..seq_len {
            let dst_offset = (pos + t) * self.kv_dim;
            let src_offset = t * self.kv_dim;
            self.key_cache[dst_offset..dst_offset + self.kv_dim]
                .copy_from_slice(&k[src_offset..src_offset + self.kv_dim]);
            self.value_cache[dst_offset..dst_offset + self.kv_dim]
                .copy_from_slice(&v[src_offset..src_offset + self.kv_dim]);
        }
        self.current_len += seq_len;
    }

    /// Get cached K: [current_len, kv_dim]
    pub fn keys(&self) -> &[f32] {
        &self.key_cache[..self.current_len * self.kv_dim]
    }

    /// Get cached V: [current_len, kv_dim]
    pub fn values(&self) -> &[f32] {
        &self.value_cache[..self.current_len * self.kv_dim]
    }

    /// Get number of cached positions.
    pub fn len(&self) -> usize {
        self.current_len
    }

    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// CPU model KV cache: one CpuLayerKVCache per layer.
pub struct CpuModelKVCache {
    layers: Vec<CpuLayerKVCache>,
}

impl CpuModelKVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| CpuLayerKVCache::new(max_seq_len, num_kv_heads, head_dim))
                .collect(),
        }
    }

    pub fn layer(&mut self, idx: usize) -> &mut CpuLayerKVCache {
        &mut self.layers[idx]
    }

    pub fn layer_ref(&self, idx: usize) -> &CpuLayerKVCache {
        &self.layers[idx]
    }

    pub fn reset_all(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Configuration for CPU token generation.
#[derive(Clone, Debug)]
pub struct CpuGenerateConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub eos_token: Option<u32>,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_window: usize,
    /// Maximum KV cache length. Defaults to 4096 if not set.
    pub max_kv_len: usize,
}

impl Default for CpuGenerateConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            max_tokens: 128,
            eos_token: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 0,
            max_kv_len: 4096,
        }
    }
}

impl CpuGenerateConfig {
    pub fn to_logit_config(&self) -> LogitProcessorConfig {
        LogitProcessorConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            repetition_window: self.repetition_window,
        }
    }
}

/// CPU token generator for CpuBlockAttnResModel.
///
/// Provides prefill → decode loop with:
/// - Per-layer KV caching
/// - KV sharing (layers 15-34 share K/V from layers computed earlier)
/// - Inter-block attention at block boundaries
/// - Optional recurrent block summaries for unlimited context
pub struct CpuTokenGenerator {
    model: CpuBlockAttnResModel,
    kv_cache: CpuModelKVCache,
    /// Block representations accumulated during generation.
    block_reps: Vec<Vec<f32>>,
    /// Partial sum accumulator for current block.
    partial_sum: Vec<f32>,
    /// Tokens in current block (for mean-pooling).
    partial_count: usize,
    /// Shared KV: maps source_layer_idx → (K, V) computed during prefill/decode.
    shared_kv: HashMap<usize, (Vec<f32>, Vec<f32>)>,
}

impl CpuTokenGenerator {
    pub fn new(model: CpuBlockAttnResModel, max_kv_len: usize) -> Self {
        let num_layers = model.num_layers;
        let (num_kv_heads, head_dim) = if let Some(layer) = model.layers.first() {
            (layer.num_kv_heads, layer.head_dim)
        } else {
            (1, model.hidden_dim)
        };

        let kv_cache = CpuModelKVCache::new(num_layers, max_kv_len, num_kv_heads, head_dim);

        Self {
            model,
            kv_cache,
            block_reps: Vec::new(),
            partial_sum: Vec::new(),
            partial_count: 0,
            shared_kv: HashMap::new(),
        }
    }

    /// Access the underlying model.
    pub fn model(&self) -> &CpuBlockAttnResModel {
        &self.model
    }

    /// Access the KV cache.
    pub fn kv_cache(&self) -> &CpuModelKVCache {
        &self.kv_cache
    }

    /// Generate tokens from a prompt.
    pub fn generate(&mut self, prompt_tokens: &[u32], config: &CpuGenerateConfig) -> Result<Vec<u32>> {
        if prompt_tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Reset state
        self.kv_cache.reset_all();
        self.block_reps.clear();
        self.shared_kv.clear();
        self.partial_sum = vec![0.0f32; self.model.hidden_dim];
        self.partial_count = 0;

        let mut logit_processor = LogitProcessor::new(config.to_logit_config());
        logit_processor.record_prompt(prompt_tokens);

        // --- Prefill ---
        let mut next_token = self.prefill(prompt_tokens, &mut logit_processor)?;
        logit_processor.record_token(next_token);

        let mut output_tokens = vec![next_token];

        if config.eos_token == Some(next_token) || output_tokens.len() >= config.max_tokens {
            return Ok(output_tokens);
        }

        // --- Decode loop ---
        for _step in 0..config.max_tokens.saturating_sub(1) {
            let token = self.decode_token(next_token, &mut logit_processor)?;
            logit_processor.record_token(token);
            output_tokens.push(token);

            if config.eos_token == Some(token) {
                break;
            }
            next_token = token;
        }

        Ok(output_tokens)
    }

    /// Prefill: process all prompt tokens in batch, fill KV cache.
    /// Returns the first sampled token.
    fn prefill(
        &mut self,
        token_ids: &[u32],
        logit_processor: &mut LogitProcessor,
    ) -> Result<u32> {
        let seq = token_ids.len();
        let hd = self.model.hidden_dim;

        // 1. Embedding with Gemma scaling
        let mut hidden = self.embed_tokens(token_ids);

        // 2. Initialize block reps with mean-pooled embedding
        for t in 0..seq {
            for d in 0..hd {
                self.partial_sum[d] += hidden[t * hd + d];
            }
        }
        self.partial_count = seq;

        // 3. Pre-compute PLE inputs
        let ple_dim = self.model.hidden_size_per_layer_input;
        let ple_precomputed = self.model.precompute_ple(&hidden, token_ids, seq, hd, ple_dim);

        // 4. Per-layer forward with KV caching
        let first_shared_layer = self.model.num_layers.saturating_sub(self.model.num_kv_shared_layers);

        for layer_idx in 0..self.model.num_layers {
            // Get PLE slice for this layer
            let ple_slice = ple_precomputed.as_ref().map(|pre| {
                let mut slice = vec![0.0f32; seq * ple_dim];
                for t in 0..seq {
                    let src = t * self.model.num_layers * ple_dim + layer_idx * ple_dim;
                    let dst = t * ple_dim;
                    for d in 0..ple_dim {
                        slice[dst + d] = pre[src + d];
                    }
                }
                slice
            });

            // Determine KV sharing
            let layer_kv_shared = self.model.layers[layer_idx].kv_shared;
            let is_shared = layer_idx >= first_shared_layer && first_shared_layer > 0 && layer_kv_shared;

            if is_shared {
                // Shared layer: look up K/V from source
                let source = CpuBlockAttnResModel::kv_shared_source_layer(
                    layer_idx, first_shared_layer, &self.model.layers,
                );
                if let Some((sk, sv)) = self.shared_kv.get(&source) {
                    // Forward using shared K/V — no new K/V to store
                    let _kv = (Some(sk.as_slice()), Some(sv.as_slice()));
                    self.model.layers[layer_idx].forward_full(
                        &mut hidden,
                        ple_slice.as_ref().map(|s| s.as_slice()),
                        Some((sk.as_slice(), sv.as_slice())),
                    );
                    // Copy shared K/V into this layer's cache
                    self.kv_cache.layer(layer_idx).append(sk, sv, seq);
                }
            } else {
                // Normal layer: compute and cache K/V
                let (k, v) = self.model.layers[layer_idx].forward_full(
                    &mut hidden,
                    ple_slice.as_ref().map(|s| s.as_slice()),
                    None,
                );

                // Cache K/V
                if !k.is_empty() {
                    self.kv_cache.layer(layer_idx).append(&k, &v, seq);

                    // Store for KV sharing by later layers
                    if first_shared_layer > 0 {
                        self.shared_kv.insert(layer_idx, (k, v));
                    }
                }
            }

            // Accumulate into partial_sum for block representation
            for t in 0..seq {
                for d in 0..hd {
                    self.partial_sum[d] += hidden[t * hd + d];
                }
            }
            self.partial_count += seq;

            // Block boundary: inter-block attention
            if self.model.is_block_boundary(layer_idx) {
                self.finalize_block_and_inter_attn(&mut hidden, seq);
            }
        }

        // 5. Final norm + LM head
        let logits = self.compute_logits(&hidden, seq);

        // Sample first token
        let mut logits_vec = logits;
        let idx = logit_processor.process_and_sample(&mut logits_vec);
        Ok(idx as u32)
    }

    /// Decode a single token using cached K/V.
    fn decode_token(
        &mut self,
        token_id: u32,
        logit_processor: &mut LogitProcessor,
    ) -> Result<u32> {
        let hd = self.model.hidden_dim;
        let pos = self.kv_cache.layer_ref(0).len();

        // 1. Embed single token
        let mut hidden = self.embed_tokens(&[token_id]);

        // 2. PLE for single token
        let ple_dim = self.model.hidden_size_per_layer_input;
        let ple_precomputed = self.model.precompute_ple(&hidden, &[token_id], 1, hd, ple_dim);

        // 3. Per-layer decode — extract layer params to avoid borrow conflicts
        let first_shared_layer = self.model.num_layers.saturating_sub(self.model.num_kv_shared_layers);
        let num_layers = self.model.num_layers;

        // Collect per-layer info needed for decode (avoids borrowing model.layers while mutating self)
        let layer_info: Vec<(bool, bool)> = (0..num_layers)
            .map(|idx| {
                let kv_shared = idx >= first_shared_layer && first_shared_layer > 0
                    && self.model.layers[idx].kv_shared;
                let is_boundary = self.model.is_block_boundary(idx);
                (kv_shared, is_boundary)
            })
            .collect();

        for layer_idx in 0..num_layers {
            let (kv_shared, _is_boundary) = layer_info[layer_idx];

            let ple_slice = ple_precomputed.as_ref().map(|pre| {
                let mut slice = vec![0.0f32; ple_dim];
                let src = layer_idx * ple_dim;
                slice.copy_from_slice(&pre[src..src + ple_dim]);
                slice
            });

            hidden = self.decode_layer_forward(
                layer_idx, &hidden, ple_slice.as_deref(), pos, kv_shared,
            );

            // Block boundary inter-block attention during decode
            if layer_info[layer_idx].1 {
                self.finalize_block_and_inter_attn(&mut hidden, 1);
            }
        }

        // 4. Final norm + LM head
        let logits = self.compute_logits(&hidden, 1);
        let mut logits_vec = logits;
        let idx = logit_processor.process_and_sample(&mut logits_vec);
        Ok(idx as u32)
    }

    /// Single-layer decode forward: compute attention with cached K/V.
    ///
    /// This is the core decode hot path. For seq=1:
    /// - Compute Q, new K, new V for the single token
    /// - Append new K, V to cache
    /// - Compute attention against full cached K, V
    /// - Apply FFN, PLE, layer scalar
    fn decode_layer_forward(
        &mut self,
        layer_idx: usize,
        hidden: &[f32],     // [1, hd]
        ple_input: Option<&[f32]>,
        pos: usize,
        kv_shared: bool,
    ) -> Vec<f32> {
        let hd = self.model.layers[layer_idx].hidden_dim;
        let nh = self.model.layers[layer_idx].num_heads;
        let nkv = self.model.layers[layer_idx].num_kv_heads;
        let head_d = self.model.layers[layer_idx].head_dim;
        let q_dim = nh * head_d;
        let kv_dim = nkv * head_d;
        let rope_theta = self.model.layers[layer_idx].rope_theta;
        let partial_rotary_factor = self.model.layers[layer_idx].partial_rotary_factor;
        let layer_scalar = self.model.layers[layer_idx].layer_scalar;

        // === Attention ===
        let residual = hidden.to_vec();

        // Pre-attention norm
        let normed = self.model.layers[layer_idx].attn_norm.forward(hidden);

        // Q projection + per-head norm + RoPE
        let mut q = self.model.layers[layer_idx].q_proj.forward(&normed, 1);
        q = per_head_rms_norm(&q, self.model.layers[layer_idx].q_norm.weight(), 1, nh, head_d);
        apply_rope(&mut q, 1, nh, head_d, pos, rope_theta, partial_rotary_factor);

        // K/V: compute new K/V for this token
        let (new_k, new_v) = if kv_shared {
            // Shared layer: get K/V from shared source cache
            let first_shared = self.model.num_layers.saturating_sub(self.model.num_kv_shared_layers);
            let source = CpuBlockAttnResModel::kv_shared_source_layer(
                layer_idx, first_shared, &self.model.layers,
            );
            if let Some((sk, sv)) = self.shared_kv.get(&source) {
                let k_slice = sk[pos * kv_dim..(pos + 1) * kv_dim].to_vec();
                let v_slice = sv[pos * kv_dim..(pos + 1) * kv_dim].to_vec();
                (k_slice, v_slice)
            } else {
                (vec![0.0f32; kv_dim], vec![0.0f32; kv_dim])
            }
        } else {
            let mut k = self.model.layers[layer_idx].k_proj.forward(&normed, 1);
            let v_raw = self.model.layers[layer_idx].v_proj.forward(&normed, 1);
            k = per_head_rms_norm(&k, self.model.layers[layer_idx].k_norm.weight(), 1, nkv, head_d);
            let v = per_head_rms_norm_no_scale(&v_raw, 1, nkv, head_d);
            apply_rope_gqa(&mut k, 1, nkv, head_d, pos, rope_theta, partial_rotary_factor);
            (k, v)
        };

        // Append new K/V to cache
        self.kv_cache.layer(layer_idx).append(&new_k, &new_v, 1);

        // Store non-shared K/V for future KV sharing
        if !kv_shared {
            let first_shared = self.model.num_layers.saturating_sub(self.model.num_kv_shared_layers);
            if first_shared > 0 {
                if let Some((existing_k, existing_v)) = self.shared_kv.get_mut(&layer_idx) {
                    existing_k.extend_from_slice(&new_k);
                    existing_v.extend_from_slice(&new_v);
                } else {
                    self.shared_kv.insert(layer_idx, (new_k.clone(), new_v.clone()));
                }
            }
        }

        // Get all cached K/V for attention
        let cache = self.kv_cache.layer_ref(layer_idx);
        let all_k = cache.keys().to_vec();
        let all_v = cache.values().to_vec();
        let total_len = cache.len();
        let _total_len = total_len;

        // Attention: q [1, q_dim], k [total_len, kv_dim], v [total_len, kv_dim]
        let attn_out = self.model.layers[layer_idx].cpu_attention(
            &q, &all_k, &all_v, 1, nh, nkv, head_d, q_dim, kv_dim,
        );

        // Post-attention norm
        let attn_out = self.model.layers[layer_idx].post_attn_norm.forward(&attn_out);

        // Residual
        let mut hidden: Vec<f32> = residual.iter().zip(attn_out.iter()).map(|(&r, &a)| r + a).collect();

        // === FFN ===
        let residual2 = hidden.clone();
        let normed2 = self.model.layers[layer_idx].pre_ffn_norm.forward(&hidden);

        let ffn_out = if let Some(ref moe) = self.model.layers[layer_idx].moe {
            moe.forward(&normed2, 1)
        } else {
            self.model.layers[layer_idx].cpu_ffn(&normed2, 1)
        };
        let ffn_out = self.model.layers[layer_idx].post_ffn_norm.forward(&ffn_out);

        for i in 0..hd {
            hidden[i] = residual2[i] + ffn_out[i];
        }

        // PLE injection
        if let Some(ple_slice) = ple_input {
            if let (Some(ref gate), Some(ref proj), Some(ref norm)) =
                (&self.model.layers[layer_idx].ple_input_gate,
                 &self.model.layers[layer_idx].ple_projection,
                 &self.model.layers[layer_idx].ple_post_norm)
            {
                let gate_out = gate.forward(&hidden, 1);
                let gate_gelu: Vec<f32> = gate_out.iter().map(|&x| gelu_tanh(x)).collect();
                let mut gated = vec![0.0f32; ple_slice.len()];
                for i in 0..ple_slice.len() {
                    gated[i] = gate_gelu[i] * ple_slice[i];
                }
                let proj_out = proj.forward(&gated, 1);
                let ple_final = norm.forward(&proj_out);
                for i in 0..hd {
                    hidden[i] += ple_final[i];
                }
            }
        }

        // Layer scalar
        if layer_scalar != 1.0 {
            for h in hidden.iter_mut() {
                *h *= layer_scalar;
            }
        }

        hidden
    }

    /// Finalize a block and apply inter-block attention.
    fn finalize_block_and_inter_attn(&mut self, hidden: &mut Vec<f32>, seq: usize) {
        let hd = self.model.hidden_dim;

        // Finalize block rep: mean pool the accumulated partial sum
        if self.partial_count > 0 {
            for d in 0..hd {
                self.partial_sum[d] /= self.partial_count as f32;
            }
        }
        self.block_reps.push(self.partial_sum.clone());
        self.partial_sum = vec![0.0f32; hd];
        self.partial_count = 0;

        // Inter-block attention
        let inter_out = self.model.inter_block_attention(hidden, &self.block_reps, seq);
        for t in 0..seq {
            for d in 0..hd {
                hidden[t * hd + d] += inter_out[d];
            }
        }
    }

    /// Embed tokens with Gemma scaling.
    fn embed_tokens(&self, token_ids: &[u32]) -> Vec<f32> {
        let seq = token_ids.len();
        let hd = self.model.hidden_dim;
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
        for h in hidden.iter_mut() {
            *h *= scale;
        }
        hidden
    }

    /// Compute logits from hidden states: final norm + LM head + softcapping.
    fn compute_logits(&self, hidden: &[f32], seq: usize) -> Vec<f32> {
        let hd = self.model.hidden_dim;
        let vs = self.model.vocab_size;

        let normed = rms_norm(hidden, &self.model.final_norm, hd, 1e-6);
        let mut logits = matmul(&normed, &self.model.lm_head, seq, hd, vs);

        if let Some(cap) = self.model.final_logit_softcapping {
            for l in logits.iter_mut() {
                *l = (*l / cap).tanh() * cap;
            }
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_layer_kv_cache_append_and_read() {
        let mut cache = CpuLayerKVCache::new(16, 2, 4);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        // Append 3 tokens: K and V each [3, 8]
        let k: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let v: Vec<f32> = (24..48).map(|i| i as f32).collect();
        cache.append(&k, &v, 3);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.keys().len(), 3 * 8);
        assert_eq!(cache.values().len(), 3 * 8);

        // Check first position
        assert!((cache.keys()[0] - 0.0).abs() < 1e-6);
        assert!((cache.keys()[7] - 7.0).abs() < 1e-6);
        assert!((cache.values()[0] - 24.0).abs() < 1e-6);

        // Append 2 more
        let k2 = vec![1.0f32; 16];
        let v2 = vec![2.0f32; 16];
        cache.append(&k2, &v2, 2);
        assert_eq!(cache.len(), 5);

        // Position 3 should be k2's first element
        assert!((cache.keys()[3 * 8] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_layer_kv_cache_reset() {
        let mut cache = CpuLayerKVCache::new(16, 1, 4);
        cache.append(&[1.0; 4], &[2.0; 4], 1);
        assert_eq!(cache.len(), 1);
        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cpu_model_kv_cache() {
        let mut cache = CpuModelKVCache::new(4, 16, 2, 8);
        assert_eq!(cache.num_layers(), 4);

        cache.layer(0).append(&[1.0; 16], &[2.0; 16], 1);
        cache.layer(1).append(&[3.0; 16], &[4.0; 16], 1);
        assert_eq!(cache.layer_ref(0).len(), 1);
        assert_eq!(cache.layer_ref(1).len(), 1);

        cache.reset_all();
        assert_eq!(cache.layer_ref(0).len(), 0);
        assert_eq!(cache.layer_ref(1).len(), 0);
    }

    #[test]
    fn test_cpu_generate_config_default() {
        let config = CpuGenerateConfig::default();
        assert!((config.temperature - 1.0).abs() < 0.01);
        assert_eq!(config.max_tokens, 128);
        assert_eq!(config.max_kv_len, 4096);
    }

    #[test]
    fn test_cpu_generate_config_to_logit_config() {
        let config = CpuGenerateConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.2,
            frequency_penalty: 0.5,
            presence_penalty: 0.3,
            repetition_window: 64,
            ..Default::default()
        };
        let lc = config.to_logit_config();
        assert!((lc.temperature - 0.7).abs() < 0.01);
        assert_eq!(lc.top_k, 50);
    }

    #[test]
    #[should_panic(expected = "KV cache overflow")]
    fn test_cpu_kv_cache_overflow() {
        let mut cache = CpuLayerKVCache::new(4, 1, 4);
        cache.append(&[0.0; 20], &[0.0; 20], 5); // 5 > max 4
    }

    #[test]
    fn test_cpu_layer_kv_cache_sequential_append() {
        let mut cache = CpuLayerKVCache::new(10, 1, 2);
        // Append 3, then 2, then 1
        cache.append(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[0.1; 6], 3);
        cache.append(&[7.0, 8.0, 9.0, 10.0], &[0.2; 4], 2);
        cache.append(&[11.0, 12.0], &[0.3; 2], 1);
        assert_eq!(cache.len(), 6);

        // Verify positions
        let keys = cache.keys();
        assert!((keys[0] - 1.0).abs() < 1e-6);
        assert!((keys[1] - 2.0).abs() < 1e-6);
        assert!((keys[2] - 3.0).abs() < 1e-6);
        assert!((keys[3] - 4.0).abs() < 1e-6);
        assert!((keys[4] - 5.0).abs() < 1e-6);
        assert!((keys[5] - 6.0).abs() < 1e-6);
        assert!((keys[6] - 7.0).abs() < 1e-6);
        assert!((keys[7] - 8.0).abs() < 1e-6);
        assert!((keys[8] - 9.0).abs() < 1e-6);
        assert!((keys[9] - 10.0).abs() < 1e-6);
        assert!((keys[10] - 11.0).abs() < 1e-6);
        assert!((keys[11] - 12.0).abs() < 1e-6);
    }
}
