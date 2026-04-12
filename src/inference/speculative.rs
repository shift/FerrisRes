//! Speculative decoding — draft-then-verify for faster inference.
//!
//! Uses a small draft model (or n-gram predictor) to propose K candidate tokens,
//! then verifies them in a single batched forward pass through the main model.
//! Accepted tokens are committed; on rejection, the first mismatched token is
//! replaced with the main model's prediction and decoding continues normally.
//!
//! Target: 2-3× speedup for predictable text (code, structured data, etc.)
//!
//! Reference: Leviathan et al., "Fast Inference from Transformers via
//! Speculative Decoding" (2023)

use std::collections::HashMap;

use crate::inference::logit_processors::{LogitProcessor, LogitProcessorConfig};

// ---------------------------------------------------------------------------
// Draft model trait
// ---------------------------------------------------------------------------

/// A draft model that proposes candidate tokens for speculative decoding.
pub trait DraftModel: Send + Sync {
    /// Propose K candidate tokens given the current context.
    /// Returns a vector of token IDs and their draft probabilities.
    fn propose(&mut self, last_token: u32, k: usize) -> Vec<DraftToken>;

    /// Reset the draft model state (e.g., at start of generation).
    fn reset(&mut self);

    /// Record an accepted token (to update n-gram state etc).
    fn record_token(&mut self, token: u32);
}

/// A single draft token proposal.
#[derive(Debug, Clone)]
pub struct DraftToken {
    pub token_id: u32,
    pub draft_prob: f32,
}

// ---------------------------------------------------------------------------
// N-gram draft model
// ---------------------------------------------------------------------------

/// A simple n-gram based draft model that predicts tokens from recent history.
///
/// This doesn't require a separate neural network — it uses statistical
/// patterns in the token sequence to predict likely continuations.
/// Much cheaper than a neural draft model while still providing speedups
/// for repetitive/structured text.
pub struct NGramDraftModel {
    /// N-gram context length (e.g., 3 → use last 2 tokens to predict next).
    n: usize,
    /// History of recent tokens.
    history: Vec<u32>,
    /// N-gram frequency table: (n-1 tokens) → {token → count}.
    ngram_table: HashMap<Vec<u32>, HashMap<u32, usize>>,
    /// Maximum table entries (for memory control).
    max_entries: usize,
}

impl NGramDraftModel {
    pub fn new(n: usize) -> Self {
        Self {
            n: n.max(2),
            history: Vec::with_capacity(256),
            ngram_table: HashMap::new(),
            max_entries: 100_000,
        }
    }

    /// Create a 3-gram model with default settings.
    pub fn trigram() -> Self {
        Self::new(3)
    }

    /// Create a 4-gram model.
    pub fn fourgram() -> Self {
        Self::new(4)
    }

    /// Train the n-gram model from a corpus.
    pub fn train(&mut self, tokens: &[u32]) {
        if tokens.len() < self.n {
            return;
        }
        for i in 0..=(tokens.len() - self.n) {
            let context = tokens[i..i + self.n - 1].to_vec();
            let next = tokens[i + self.n - 1];
            let entry = self.ngram_table.entry(context).or_default();
            *entry.entry(next).or_insert(0) += 1;

            if self.ngram_table.len() > self.max_entries {
                // Evict least-used entries (simple LRU approximation)
                if let Some(key) = self.ngram_table.keys().next().cloned() {
                    self.ngram_table.remove(&key);
                }
            }
        }
    }

    /// Get top-k predictions given context.
    fn top_k(&self, context: &[u32], k: usize) -> Vec<DraftToken> {
        if let Some(nexts) = self.ngram_table.get(context) {
            let mut pairs: Vec<(u32, usize)> = nexts.iter().map(|(&t, &c)| (t, c)).collect();
            pairs.sort_by(|a, b| b.1.cmp(&a.1));
            let total: usize = pairs.iter().map(|(_, c)| *c).sum();
            pairs.into_iter()
                .take(k)
                .map(|(token_id, count)| DraftToken {
                    token_id,
                    draft_prob: count as f32 / total.max(1) as f32,
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Number of n-gram entries in the table.
    pub fn table_size(&self) -> usize {
        self.ngram_table.len()
    }
}

impl DraftModel for NGramDraftModel {
    fn propose(&mut self, _last_token: u32, k: usize) -> Vec<DraftToken> {
        if self.history.len() >= self.n - 1 {
            let start = self.history.len() - (self.n - 1);
            let context = &self.history[start..];
            let proposals = self.top_k(context, k);
            if !proposals.is_empty() {
                return proposals;
            }
        }
        // Fallback: try shorter context
        if self.history.len() >= 1 {
            let context = &self.history[self.history.len() - 1..];
            let proposals = self.top_k(context, k);
            if !proposals.is_empty() {
                return proposals;
            }
        }
        Vec::new()
    }

    fn reset(&mut self) {
        self.history.clear();
    }

    fn record_token(&mut self, token: u32) {
        self.history.push(token);
        if self.history.len() > 512 {
            self.history.drain(0..256);
        }
    }
}

// ---------------------------------------------------------------------------
// Speculative decoding engine
// ---------------------------------------------------------------------------

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to propose per step.
    pub draft_length: usize,
    /// Temperature for verification sampling.
    pub temperature: f32,
    /// Maximum number of speculation rounds before falling back.
    pub max_rounds: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_length: 5,
            temperature: 1.0,
            max_rounds: 100,
        }
    }
}

impl SpeculativeConfig {
    pub fn new(draft_length: usize) -> Self {
        Self {
            draft_length,
            ..Default::default()
        }
    }
}

/// Result of a single speculative decoding step.
#[derive(Debug)]
pub struct SpeculativeResult {
    /// Accepted token IDs (all verified by main model).
    pub accepted_tokens: Vec<u32>,
    /// Whether the speculation was fully accepted (all K tokens matched).
    pub fully_accepted: bool,
    /// Number of draft tokens proposed.
    pub draft_count: usize,
    /// Acceptance rate (accepted / proposed).
    pub acceptance_rate: f32,
}

/// The speculative decoding engine.
///
/// Orchestrates draft-then-verify: the draft model proposes tokens,
/// the main model verifies them in a single batched pass, and accepted
/// tokens are committed to the output.
pub struct SpeculativeEngine {
    config: SpeculativeConfig,
    draft_model: Box<dyn DraftModel>,
    /// Logit processor for the main model's verification.
    #[allow(dead_code)]
    logit_processor: LogitProcessor,
}

impl SpeculativeEngine {
    pub fn new(config: SpeculativeConfig, draft_model: Box<dyn DraftModel>) -> Self {
        let logit_config = LogitProcessorConfig {
            temperature: config.temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 0,
        };
        Self {
            config,
            draft_model,
            logit_processor: LogitProcessor::new(logit_config),
        }
    }

    /// Create an engine with an n-gram draft model.
    pub fn with_ngram(config: SpeculativeConfig, n: usize) -> Self {
        Self::new(config, Box::new(NGramDraftModel::new(n)))
    }

    /// Train the n-gram draft model from a corpus.
    /// Only works if the draft model is an NGramDraftModel.
    pub fn train_draft(&mut self, tokens: &[u32]) {
        // We can't downcast Box<dyn DraftModel> easily, so we just
        // call record_token for each token to build up context
        for &token in tokens {
            self.draft_model.record_token(token);
        }
    }

    /// Reset state for a new generation.
    pub fn reset(&mut self) {
        self.draft_model.reset();
    }

    /// Propose K draft tokens.
    pub fn propose(&mut self, last_token: u32) -> Vec<DraftToken> {
        let k = self.config.draft_length;
        let proposals = self.draft_model.propose(last_token, k);
        // If draft model couldn't produce enough, pad with the last token
        self.draft_model.record_token(last_token);
        proposals
    }

    /// Verify draft tokens against main model logits.
    ///
    /// Given the draft proposals and the main model's logits for each position,
    /// accept the longest matching prefix using rejection sampling.
    ///
    /// Returns accepted tokens and whether to continue speculation.
    pub fn verify(
        &mut self,
        draft_tokens: &[DraftToken],
        main_logits: &[Vec<f32>],
        vocab_size: usize,
    ) -> SpeculativeResult {
        let draft_count = draft_tokens.len();
        if draft_count == 0 || main_logits.is_empty() {
            return SpeculativeResult {
                accepted_tokens: Vec::new(),
                fully_accepted: false,
                draft_count: 0,
                acceptance_rate: 0.0,
            };
        }

        let mut accepted = Vec::new();
        let mut all_accepted = true;

        for (i, draft) in draft_tokens.iter().enumerate() {
            if i >= main_logits.len() {
                break;
            }
            let logits = &main_logits[i];

            // Get main model's probability for the draft token
            let main_prob = softmax_prob(logits, draft.token_id as usize, vocab_size);

            // Rejection sampling: accept with probability min(1, p_main / p_draft)
            let acceptance_prob = if draft.draft_prob > 0.0 {
                (main_prob / draft.draft_prob).min(1.0)
            } else {
                1.0
            };

            // Deterministic verification for simplicity: accept if main model
            // also ranks this token in top-1 or with sufficient probability
            let main_top1 = argmax(logits);
            if main_top1 == draft.token_id as usize {
                // Perfect match — accept
                accepted.push(draft.token_id);
                self.draft_model.record_token(draft.token_id);
            } else if acceptance_prob > 0.5 {
                // Probabilistic acceptance
                accepted.push(draft.token_id);
                self.draft_model.record_token(draft.token_id);
            } else {
                // Rejection — use main model's top-1 instead
                accepted.push(main_top1 as u32);
                self.draft_model.record_token(main_top1 as u32);
                all_accepted = false;
                break;
            }
        }

        // If all draft tokens were accepted, add one more from main model's
        // logits at the last position
        if all_accepted && !main_logits.is_empty() {
            let last_logits = &main_logits[main_logits.len() - 1];
            let bonus_token = argmax(last_logits) as u32;
            accepted.push(bonus_token);
            self.draft_model.record_token(bonus_token);
        }

        let acceptance_rate = if draft_count > 0 {
            let accepted_count = accepted.len().min(draft_count);
            accepted_count as f32 / draft_count as f32
        } else {
            0.0
        };

        SpeculativeResult {
            accepted_tokens: accepted,
            fully_accepted: all_accepted,
            draft_count,
            acceptance_rate,
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn argmax(logits: &[f32]) -> usize {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn softmax_prob(logits: &[f32], token_idx: usize, _vocab_size: usize) -> f32 {
    if token_idx >= logits.len() {
        return 0.0;
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&v| (v - max_val).exp()).sum();
    if exp_sum == 0.0 {
        return 0.0;
    }
    (logits[token_idx] - max_val).exp() / exp_sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_draft_model_basic() {
        let mut model = NGramDraftModel::new(3);
        model.train(&[1, 2, 3, 1, 2, 4, 1, 2, 3]);

        // After [1, 2], should predict 3 or 4
        model.history = vec![1, 2];
        let proposals = model.propose(2, 5);
        assert!(!proposals.is_empty());
        // 3 appears twice after [1,2], 4 appears once
        assert_eq!(proposals[0].token_id, 3);
    }

    #[test]
    fn test_ngram_draft_model_empty() {
        let model = NGramDraftModel::new(3);
        assert_eq!(model.table_size(), 0);
    }

    #[test]
    fn test_ngram_train_and_query() {
        let mut model = NGramDraftModel::trigram();
        model.train(&[10, 20, 30, 10, 20, 30, 10, 20, 40]);
        model.history = vec![10, 20];

        let proposals = model.propose(20, 3);
        assert!(proposals.len() >= 1);
        // 30 appears 2× after [10,20], 40 appears 1× → 30 is top
        assert_eq!(proposals[0].token_id, 30);
    }

    #[test]
    fn test_ngram_fourgram() {
        let mut model = NGramDraftModel::fourgram();
        model.train(&[1, 2, 3, 4, 1, 2, 3, 5]);
        model.history = vec![1, 2, 3];

        let proposals = model.propose(3, 3);
        assert!(!proposals.is_empty());
    }

    #[test]
    fn test_ngram_record_and_evict() {
        let mut model = NGramDraftModel::new(2);
        model.history = vec![1];
        model.record_token(2);
        model.record_token(3);
        assert_eq!(&model.history, &[1, 2, 3]);
    }

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.draft_length, 5);
        assert_eq!(config.max_rounds, 100);
    }

    #[test]
    fn test_speculative_verify_all_accepted() {
        let mut engine = SpeculativeEngine::with_ngram(
            SpeculativeConfig::new(3),
            3,
        );

        let draft = vec![
            DraftToken { token_id: 10, draft_prob: 0.8 },
            DraftToken { token_id: 20, draft_prob: 0.7 },
            DraftToken { token_id: 30, draft_prob: 0.6 },
        ];

        // Main logits that agree with draft tokens (make them argmax)
        let mut logits1 = vec![0.0f32; 100];
        logits1[10] = 10.0;
        let mut logits2 = vec![0.0f32; 100];
        logits2[20] = 10.0;
        let mut logits3 = vec![0.0f32; 100];
        logits3[30] = 10.0;

        let main_logits = vec![logits1, logits2, logits3];
        let result = engine.verify(&draft, &main_logits, 100);

        // All 3 draft tokens accepted + 1 bonus = 4
        assert!(result.accepted_tokens.len() >= 3);
        assert!(result.fully_accepted);
        assert_eq!(result.draft_count, 3);
    }

    #[test]
    fn test_speculative_verify_rejection() {
        let mut engine = SpeculativeEngine::with_ngram(
            SpeculativeConfig::new(3),
            3,
        );

        let draft = vec![
            DraftToken { token_id: 10, draft_prob: 0.9 },
            DraftToken { token_id: 20, draft_prob: 0.8 },
        ];

        // First logits agree, second disagrees
        let mut logits1 = vec![0.0f32; 100];
        logits1[10] = 10.0;
        let mut logits2 = vec![0.0f32; 100];
        logits2[99] = 10.0; // Main model wants token 99, not 20

        let main_logits = vec![logits1, logits2];
        let result = engine.verify(&draft, &main_logits, 100);

        assert!(!result.fully_accepted);
        assert_eq!(result.accepted_tokens[0], 10); // first accepted
        assert_eq!(result.accepted_tokens[1], 99); // second rejected → main's choice
    }

    #[test]
    fn test_speculative_verify_empty() {
        let mut engine = SpeculativeEngine::with_ngram(
            SpeculativeConfig::new(3),
            3,
        );
        let result = engine.verify(&[], &[], 100);
        assert!(result.accepted_tokens.is_empty());
        assert_eq!(result.acceptance_rate, 0.0);
    }

    #[test]
    fn test_softmax_prob() {
        let logits = vec![1.0, 2.0, 3.0];
        let p0 = softmax_prob(&logits, 0, 3);
        let p1 = softmax_prob(&logits, 1, 3);
        let p2 = softmax_prob(&logits, 2, 3);
        assert!(p2 > p1);
        assert!(p1 > p0);
        let total = p0 + p1 + p2;
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
    }
}
