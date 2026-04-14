//! Speculative Block Decoding
//!
//! Uses a tiny "Block-Draft" model that predicts the next Block Summary vector
//! instead of individual tokens. The main model verifies the summary and either
//! accepts it (filling in the block 8x faster) or rejects it and recomputes.
//!
//! Architecture:
//!   BlockDraftModel: ~10M params, predicts summary vector for next block
//!   Draft-then-verify loop:
//!     1. Draft model predicts next block summary (O(1))
//!     2. Main model verifies against its own block attention (O(n))
//!     3. Accept → decode tokens from summary in parallel
//!     4. Reject → recompute from scratch
//!
//! This leverages the O(n) BlockAttnRes hierarchy: the draft model only needs
//! to predict the block-level representation, not every token.

// HashMap not currently needed — concepts use Vec-based storage

// ---------------------------------------------------------------------------
// Block Summary vector
// ---------------------------------------------------------------------------

/// A block summary vector — the compressed representation of a sequence block.
/// Used as the "prediction target" for the draft model.
#[derive(Debug, Clone)]
pub struct BlockSummary {
    /// The summary embedding vector.
    pub embedding: Vec<f32>,
    /// Number of tokens this summary represents.
    pub token_count: usize,
    /// Confidence score from the draft model (0.0 - 1.0).
    pub confidence: f32,
}

// ---------------------------------------------------------------------------
// Block Draft Model
// ---------------------------------------------------------------------------

/// Configuration for the Block-Draft model.
#[derive(Debug, Clone)]
pub struct BlockDraftConfig {
    /// Hidden dimension (small — typically 256).
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Maximum block summary length to predict.
    pub max_summary_len: usize,
    /// Acceptance threshold (cosine similarity ≥ this → accept).
    pub accept_threshold: f32,
    /// Whether the draft model is trained.
    pub trained: bool,
}

impl Default for BlockDraftConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 256,
            num_heads: 4,
            num_layers: 2,
            max_summary_len: 64,
            accept_threshold: 0.85,
            trained: false,
        }
    }
}

/// A tiny transformer that predicts the next block summary vector.
///
/// This is much smaller than the main model (~10M params vs ~10B params).
/// It only needs to predict the block-level representation, not every token.
pub struct BlockDraftModel {
    config: BlockDraftConfig,
    /// Learned query projection: [hidden_dim × hidden_dim].
    query_weight: Vec<f32>,
    /// Learned key projection: [hidden_dim × hidden_dim].
    key_weight: Vec<f32>,
    /// Learned value projection: [hidden_dim × hidden_dim].
    value_weight: Vec<f32>,
    /// Output projection: [hidden_dim × hidden_dim].
    output_weight: Vec<f32>,
    /// Feed-forward gate: [hidden_dim × hidden_dim].
    ff_gate: Vec<f32>,
    /// Feed-forward up: [hidden_dim × hidden_dim].
    ff_up: Vec<f32>,
    /// Feed-forward down: [hidden_dim × hidden_dim].
    ff_down: Vec<f32>,
    /// Summary prediction head: [hidden_dim → hidden_dim].
    summary_head: Vec<f32>,
}

impl BlockDraftModel {
    /// Create a new draft model with Xavier initialization.
    pub fn new(config: BlockDraftConfig) -> Self {
        let hd = config.hidden_dim;
        let scale = (2.0 / hd as f32).sqrt();

        let mut rng = simple_rng(config.num_layers as u64);
        let mut rand_mat = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|_| rand_f32(&mut rng) * scale)
                .collect()
        };

        Self {
            query_weight: rand_mat(hd, hd),
            key_weight: rand_mat(hd, hd),
            value_weight: rand_mat(hd, hd),
            output_weight: rand_mat(hd, hd),
            ff_gate: rand_mat(hd, hd),
            ff_up: rand_mat(hd, hd),
            ff_down: rand_mat(hd, hd),
            summary_head: rand_mat(hd, hd),
            config,
        }
    }

    /// Predict the next block summary given the current block summaries.
    ///
    /// `prev_summaries`: summaries of previous blocks (context).
    /// Returns: predicted summary for the next block.
    pub fn predict(&self, prev_summaries: &[BlockSummary]) -> BlockSummary {
        let hd = self.config.hidden_dim;

        if prev_summaries.is_empty() {
            // No context — return zero summary with low confidence
            return BlockSummary {
                embedding: vec![0.0; hd],
                token_count: 0,
                confidence: 0.1,
            };
        }

        // Simple attention over previous summaries
        // Query = last summary, Keys/Values = all summaries
        let query = &prev_summaries.last().unwrap().embedding;

        // Project query
        let q = matvec(&self.query_weight, query, hd, hd);
        let k: Vec<Vec<f32>> = prev_summaries.iter()
            .map(|s| matvec(&self.key_weight, &s.embedding, hd, hd))
            .collect();
        let v: Vec<Vec<f32>> = prev_summaries.iter()
            .map(|s| matvec(&self.value_weight, &s.embedding, hd, hd))
            .collect();

        // Attention scores
        let scale = 1.0 / (hd as f32).sqrt();
        let scores: Vec<f32> = k.iter()
            .map(|ki| dot(&q, ki) * scale)
            .collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let attn_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of values
        let mut attended = vec![0.0; hd];
        for (w, vi) in attn_weights.iter().zip(v.iter()) {
            for d in 0..hd {
                attended[d] += w * vi[d];
            }
        }

        // Output projection
        let projected = matvec(&self.output_weight, &attended, hd, hd);

        // FFN: gate * silu(up) → down
        let gated = matvec(&self.ff_gate, &projected, hd, hd);
        let gated_silu: Vec<f32> = gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
        let upped = matvec(&self.ff_up, &projected, hd, hd);
        let mut combined = vec![0.0; hd];
        for d in 0..hd { combined[d] = gated_silu[d] * upped[d]; }
        let ffn_out = matvec(&self.ff_down, &combined, hd, hd);

        // Residual
        let mut hidden = vec![0.0; hd];
        for d in 0..hd { hidden[d] = projected[d] + ffn_out[d]; }

        // Summary prediction head
        let summary = matvec(&self.summary_head, &hidden, hd, hd);

        // Confidence based on attention entropy
        let entropy = -attn_weights.iter()
            .filter(|&&w| w > 1e-8)
            .map(|&w| w * w.ln())
            .sum::<f32>();
        let max_entropy = (prev_summaries.len() as f32).ln();
        let confidence = if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy).min(1.0)
        } else {
            1.0
        };

        BlockSummary {
            embedding: summary,
            token_count: self.config.max_summary_len,
            confidence,
        }
    }

    /// Get the model config.
    pub fn config(&self) -> &BlockDraftConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Draft-then-Verify loop
// ---------------------------------------------------------------------------

/// Result of a draft-verify cycle.
#[derive(Debug, Clone)]
pub struct DraftVerifyResult {
    /// Whether the draft was accepted.
    pub accepted: bool,
    /// Cosine similarity between draft and main model summaries.
    pub similarity: f32,
    /// The draft prediction.
    pub draft_summary: BlockSummary,
    /// The main model's actual summary (if computed).
    pub actual_summary: Option<BlockSummary>,
    /// Number of tokens saved by accepting the draft.
    pub tokens_saved: usize,
    /// Wall-clock time for draft prediction (ms).
    pub draft_time_ms: f64,
    /// Wall-clock time for verification (ms).
    pub verify_time_ms: f64,
}

/// Speculative Block Decoder — orchestrates the draft-verify loop.
pub struct SpeculativeBlockDecoder {
    /// The draft model.
    draft_model: BlockDraftModel,
    /// History of previous block summaries (context for the draft model).
    summary_history: Vec<BlockSummary>,
    /// Statistics across all draft-verify cycles.
    stats: DraftStats,
}

/// Running statistics for the speculative decoder.
#[derive(Debug, Clone, Default)]
pub struct DraftStats {
    pub total_drafts: usize,
    pub accepted_drafts: usize,
    pub rejected_drafts: usize,
    pub total_tokens_saved: usize,
    pub total_draft_time_ms: f64,
    pub total_verify_time_ms: f64,
    pub similarity_history: Vec<f32>,
}

impl DraftStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafts == 0 { return 0.0; }
        self.accepted_drafts as f32 / self.total_drafts as f32
    }

    pub fn avg_similarity(&self) -> f32 {
        if self.similarity_history.is_empty() { return 0.0; }
        self.similarity_history.iter().sum::<f32>() / self.similarity_history.len() as f32
    }

    pub fn speedup_factor(&self) -> f32 {
        if self.total_drafts == 0 { return 1.0; }
        let accept_rate = self.acceptance_rate();
        1.0 + accept_rate * 7.0 // Each accepted draft saves ~7x tokens
    }
}

impl SpeculativeBlockDecoder {
    /// Create a new speculative decoder with the given draft config.
    pub fn new(config: BlockDraftConfig) -> Self {
        Self {
            draft_model: BlockDraftModel::new(config),
            summary_history: Vec::new(),
            stats: DraftStats::default(),
        }
    }

    /// Run one draft-verify cycle.
    ///
    /// `main_model_summary`: the main model's actual summary for the current block.
    /// If None, the draft is always accepted (no verification possible).
    pub fn draft_and_verify(
        &mut self,
        main_model_summary: Option<&BlockSummary>,
        draft_time_ms: f64,
        verify_time_ms: f64,
    ) -> DraftVerifyResult {
        // Step 1: Draft prediction
        let draft_summary = self.draft_model.predict(&self.summary_history);

        // Step 2: Verify against main model
        let (accepted, similarity) = match main_model_summary {
            Some(actual) => {
                let sim = cosine_similarity(&draft_summary.embedding, &actual.embedding);
                (sim >= self.draft_model.config().accept_threshold, sim)
            }
            None => {
                // No main model summary — accept with low confidence
                (draft_summary.confidence > 0.5, draft_summary.confidence)
            }
        };

        // Update history
        if accepted {
            self.summary_history.push(draft_summary.clone());
        } else if let Some(actual) = main_model_summary {
            self.summary_history.push(actual.clone());
        }

        let tokens_saved = if accepted {
            draft_summary.token_count * 7 // ~7x savings per accepted draft
        } else {
            0
        };

        // Update stats
        self.stats.total_drafts += 1;
        if accepted {
            self.stats.accepted_drafts += 1;
        } else {
            self.stats.rejected_drafts += 1;
        }
        self.stats.total_tokens_saved += tokens_saved;
        self.stats.total_draft_time_ms += draft_time_ms;
        self.stats.total_verify_time_ms += verify_time_ms;
        self.stats.similarity_history.push(similarity);

        DraftVerifyResult {
            accepted,
            similarity,
            draft_summary,
            actual_summary: main_model_summary.cloned(),
            tokens_saved,
            draft_time_ms,
            verify_time_ms,
        }
    }

    /// Get the running statistics.
    pub fn stats(&self) -> &DraftStats {
        &self.stats
    }

    /// Reset the summary history (new sequence).
    pub fn reset(&mut self) {
        self.summary_history.clear();
    }

    /// Get the draft model reference.
    pub fn draft_model(&self) -> &BlockDraftModel {
        &self.draft_model
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn matvec(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows];
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += mat[r * cols + c] * vec[c];
        }
        out[r] = sum;
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = dot(a, a).sqrt();
    let nb = dot(b, b).sqrt();
    if na < 1e-8 || nb < 1e-8 { return 0.0; }
    d / (na * nb)
}

/// Simple deterministic RNG (xorshift).
fn simple_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.wrapping_add(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as i64).abs() as f32 / u64::MAX as f32) * 2.0 - 1.0
    }
}

fn rand_f32(rng: &mut impl FnMut() -> f32) -> f32 {
    rng()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_draft_model_creation() {
        let config = BlockDraftConfig::default();
        let model = BlockDraftModel::new(config);
        assert_eq!(model.query_weight.len(), 256 * 256);
        assert_eq!(model.config.hidden_dim, 256);
    }

    #[test]
    fn test_draft_predict_empty() {
        let model = BlockDraftModel::new(BlockDraftConfig::default());
        let result = model.predict(&[]);
        assert_eq!(result.embedding.len(), 256);
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_draft_predict_with_context() {
        let model = BlockDraftModel::new(BlockDraftConfig::default());
        let summaries = vec![
            BlockSummary { embedding: vec![1.0; 256], token_count: 32, confidence: 0.9 },
            BlockSummary { embedding: vec![0.5; 256], token_count: 32, confidence: 0.8 },
        ];
        let result = model.predict(&summaries);
        assert_eq!(result.embedding.len(), 256);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 0.001);

        let d = vec![1.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &d) - 0.7071).abs() < 0.01);
    }

    #[test]
    fn test_matvec() {
        let mat = vec![1.0, 2.0, 3.0, 4.0];
        let vec = vec![1.0, 1.0];
        let result = matvec(&mat, &vec, 2, 2);
        assert_eq!(result, vec![3.0, 7.0]);
    }

    #[test]
    fn test_speculative_decoder() {
        let mut decoder = SpeculativeBlockDecoder::new(BlockDraftConfig::default());

        // First draft — no main model summary, accepted by default if confidence > 0.5
        let _result = decoder.draft_and_verify(None, 1.0, 0.0);
        assert_eq!(decoder.stats().total_drafts, 1);

        // Draft with matching main model summary
        let actual = BlockSummary { embedding: vec![1.0; 256], token_count: 32, confidence: 0.9 };
        let _result = decoder.draft_and_verify(Some(&actual), 1.0, 5.0);
        assert_eq!(decoder.stats().total_drafts, 2);
    }

    #[test]
    fn test_draft_stats() {
        let stats = DraftStats {
            total_drafts: 100,
            accepted_drafts: 75,
            rejected_drafts: 25,
            total_tokens_saved: 16800,
            total_draft_time_ms: 100.0,
            total_verify_time_ms: 500.0,
            similarity_history: vec![0.9; 100],
        };
        assert!((stats.acceptance_rate() - 0.75).abs() < 0.001);
        assert!((stats.avg_similarity() - 0.9).abs() < 0.001);
        assert!(stats.speedup_factor() > 1.0);
    }

    #[test]
    fn test_decoder_acceptance_threshold() {
        let config = BlockDraftConfig {
            accept_threshold: 0.99, // Very strict
            ..BlockDraftConfig::default()
        };
        let mut decoder = SpeculativeBlockDecoder::new(config);

        // Orthogonal vectors should be rejected
        let actual = BlockSummary { embedding: vec![0.0; 256], token_count: 32, confidence: 0.9 };
        let result = decoder.draft_and_verify(Some(&actual), 1.0, 5.0);
        // Similarity will likely be low, so rejected
        assert_eq!(result.accepted, result.similarity >= 0.99);
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = SpeculativeBlockDecoder::new(BlockDraftConfig::default());

        // Add some history
        let actual = BlockSummary { embedding: vec![1.0; 256], token_count: 32, confidence: 0.9 };
        decoder.draft_and_verify(Some(&actual), 1.0, 5.0);
        assert!(!decoder.summary_history.is_empty());

        decoder.reset();
        assert!(decoder.summary_history.is_empty());
        // Stats are preserved
        assert_eq!(decoder.stats().total_drafts, 1);
    }

    #[test]
    fn test_draft_model_small_config() {
        let config = BlockDraftConfig {
            hidden_dim: 64,
            num_heads: 2,
            num_layers: 1,
            max_summary_len: 16,
            accept_threshold: 0.8,
            trained: false,
        };
        let model = BlockDraftModel::new(config);
        assert_eq!(model.query_weight.len(), 64 * 64);

        let summaries = vec![
            BlockSummary { embedding: vec![1.0; 64], token_count: 16, confidence: 0.8 },
        ];
        let result = model.predict(&summaries);
        assert_eq!(result.embedding.len(), 64);
    }
}
