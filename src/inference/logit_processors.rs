//! Logit processor pipeline for controlled text generation.
//!
//! Implements a composable chain of logit transformations:
//!   RepetitionPenalty → Temperature → TopK → TopP → Sample
//!
//! Based on research task 01702fef: temperature scaling, top-k truncation,
//! top-p (nucleus) filtering, and frequency/presence repetition penalty.

use std::collections::HashMap;

/// Configuration for the full logit processor pipeline.
#[derive(Debug, Clone)]
pub struct LogitProcessorConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// Number of recent tokens to consider for repetition penalty.
    /// 0 means use the full history.
    pub repetition_window: usize,
}

impl Default for LogitProcessorConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,        // 0 = disabled
            top_p: 1.0,      // 1.0 = disabled
            repetition_penalty: 1.0, // 1.0 = disabled
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 0,
        }
    }
}

impl LogitProcessorConfig {
    /// Greedy decoding: temperature=0, no sampling randomness.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Creative sampling: higher temperature, moderate top-p.
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 64,
        }
    }

    /// Precise sampling: low temperature, tight top-p.
    pub fn precise() -> Self {
        Self {
            temperature: 0.3,
            top_k: 10,
            top_p: 0.9,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_window: 0,
        }
    }
}

/// Tracks token history for repetition/frequency/presence penalties.
#[derive(Debug, Clone, Default)]
pub struct TokenHistory {
    /// Maps token_id -> (count, last_position).
    token_counts: HashMap<u32, (usize, usize)>,
    /// Ordered history of generated tokens.
    tokens: Vec<u32>,
    /// Current position in the sequence.
    position: usize,
}

impl TokenHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a generated token.
    pub fn record(&mut self, token_id: u32) {
        self.position += 1;
        self.tokens.push(token_id);
        let entry = self.token_counts.entry(token_id).or_insert((0, 0));
        entry.0 += 1;
        entry.1 = self.position;
    }

    /// Record multiple tokens (e.g., from a prompt).
    pub fn record_prompt(&mut self, tokens: &[u32]) {
        for &token_id in tokens {
            self.record(token_id);
        }
    }

    /// Get the frequency count for a token within the optional window.
    pub fn frequency(&self, token_id: u32, window: usize) -> usize {
        if let Some(&(count, _)) = self.token_counts.get(&token_id) {
            if window == 0 {
                return count;
            }
            // Count only within the recent window
            let start = if self.tokens.len() > window {
                self.tokens.len() - window
            } else {
                0
            };
            self.tokens[start..].iter().filter(|&&t| t == token_id).count()
        } else {
            0
        }
    }

    /// Check if a token appeared in the recent window.
    pub fn present(&self, token_id: u32, window: usize) -> bool {
        self.frequency(token_id, window) > 0
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.token_counts.clear();
        self.tokens.clear();
        self.position = 0;
    }

    /// Get the full token history.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// Composable logit processor pipeline.
///
/// Applies processors in order:
/// 1. Repetition penalty (multiplicative)
/// 2. Frequency/presence penalty (additive)
/// 3. Temperature scaling
/// 4. Top-K filtering
/// 5. Top-P (nucleus) filtering
/// 6. Sampling (argmax if temperature=0, else weighted random)
pub struct LogitProcessor {
    config: LogitProcessorConfig,
    history: TokenHistory,
}

impl LogitProcessor {
    pub fn new(config: LogitProcessorConfig) -> Self {
        Self {
            config,
            history: TokenHistory::new(),
        }
    }

    /// Process logits and sample a token.
    pub fn process_and_sample(&mut self, logits: &mut [f32]) -> usize {
        self.apply_penalties(logits);
        self.apply_temperature(logits);
        self.apply_top_k(logits);
        self.apply_top_p(logits);
        self.sample(logits)
    }

    /// Apply repetition, frequency, and presence penalties.
    fn apply_penalties(&self, logits: &mut [f32]) {
        let window = self.config.repetition_window;
        let rep_penalty = self.config.repetition_penalty;
        let freq_penalty = self.config.frequency_penalty;
        let pres_penalty = self.config.presence_penalty;

        if rep_penalty == 1.0 && freq_penalty == 0.0 && pres_penalty == 0.0 {
            return; // No penalties configured
        }

        for (token_id, logit) in logits.iter_mut().enumerate() {
            let token_id = token_id as u32;
            let freq = self.history.frequency(token_id, window);
            let present = freq > 0;

            // Multiplicative repetition penalty
            if present && rep_penalty != 1.0 {
                if *logit > 0.0 {
                    *logit /= rep_penalty;
                } else {
                    *logit *= rep_penalty;
                }
            }

            // Additive frequency penalty (penalize proportional to frequency)
            if freq > 0 && freq_penalty != 0.0 {
                *logit -= freq_penalty * freq as f32;
            }

            // Additive presence penalty (flat penalty if token appeared at all)
            if present && pres_penalty != 0.0 {
                *logit -= pres_penalty;
            }
        }
    }

    /// Apply temperature scaling to logits.
    fn apply_temperature(&self, logits: &mut [f32]) {
        if self.config.temperature <= 0.0 {
            return; // Greedy: don't scale, will argmax later
        }
        if self.config.temperature == 1.0 {
            return; // No-op
        }
        for logit in logits.iter_mut() {
            *logit /= self.config.temperature;
        }
    }

    /// Apply top-k filtering: keep only the k highest logits, set rest to -inf.
    fn apply_top_k(&self, logits: &mut [f32]) {
        if self.config.top_k == 0 || self.config.top_k >= logits.len() {
            return; // Disabled or no-op
        }

        let k = self.config.top_k.min(logits.len());
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        // Partial sort: find the k-th largest value
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build a set of top-k indices
        let mut top_k_set = vec![false; logits.len()];
        for (idx, _) in &indexed[..k] {
            top_k_set[*idx] = true;
        }

        // Set non-top-k logits to -inf
        for (i, logit) in logits.iter_mut().enumerate() {
            if !top_k_set[i] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Apply top-p (nucleus) filtering: keep smallest set of tokens
    /// whose cumulative probability >= p, set rest to -inf.
    fn apply_top_p(&self, logits: &mut [f32]) {
        if self.config.top_p >= 1.0 {
            return; // Disabled
        }

        // Compute softmax to get probabilities
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            return;
        }

        // Sort indices by probability descending
        let mut indexed: Vec<(usize, f32)> = exps.iter().enumerate()
            .map(|(i, &e)| (i, e / sum))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find cutoff
        let mut cumulative = 0.0f32;
        let mut cutoff = indexed.len();
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative >= self.config.top_p {
                cutoff = i + 1;
                break;
            }
        }

        // Build kept set
        let mut kept = vec![false; logits.len()];
        for &(idx, _) in &indexed[..cutoff] {
            kept[idx] = true;
        }

        // Filter
        for (i, logit) in logits.iter_mut().enumerate() {
            if !kept[i] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Sample from processed logits.
    fn sample(&self, logits: &[f32]) -> usize {
        // Temperature=0 → greedy (argmax)
        if self.config.temperature == 0.0 {
            return argmax(logits);
        }

        // Weighted random sampling from softmax distribution
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            return argmax(logits);
        }

        let mut rng = rand::thread_rng();
        let mut r: f32 = rand::Rng::gen_range(&mut rng, 0.0..1.0);
        for (i, &e) in exps.iter().enumerate() {
            let prob = e / sum;
            r -= prob;
            if r <= 0.0 {
                return i;
            }
        }
        exps.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Record a generated token in history.
    pub fn record_token(&mut self, token_id: u32) {
        self.history.record(token_id);
    }

    /// Record prompt tokens in history.
    pub fn record_prompt(&mut self, tokens: &[u32]) {
        self.history.record_prompt(tokens);
    }

    /// Reset history.
    pub fn reset(&mut self) {
        self.history.reset();
    }

    /// Get the current config.
    pub fn config(&self) -> &LogitProcessorConfig {
        &self.config
    }

    /// Get the token history.
    pub fn history(&self) -> &TokenHistory {
        &self.history
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scaling() {
        let config = LogitProcessorConfig {
            temperature: 0.5,
            ..Default::default()
        };
        let processor = LogitProcessor::new(config);
        let mut logits = vec![1.0, 2.0, 3.0];
        processor.apply_temperature(&mut logits);
        assert_eq!(logits, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_temperature_zero_is_greedy() {
        let mut processor = LogitProcessor::new(LogitProcessorConfig::greedy());
        let logits = vec![0.1, 0.5, 0.9, 0.3];
        let result = processor.process_and_sample(&mut logits.clone());
        assert_eq!(result, 2); // index of 0.9
    }

    #[test]
    fn test_top_k_filters() {
        let config = LogitProcessorConfig {
            top_k: 2,
            ..Default::default()
        };
        let processor = LogitProcessor::new(config);
        let mut logits = vec![1.0, 5.0, 3.0, 0.5];
        processor.apply_top_k(&mut logits);
        // Only indices 1 (5.0) and 2 (3.0) should survive
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        assert!(!logits[1].is_infinite());
        assert!(!logits[2].is_infinite());
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    }

    #[test]
    fn test_top_p_filters() {
        let config = LogitProcessorConfig {
            top_p: 0.5,
            ..Default::default()
        };
        let processor = LogitProcessor::new(config);
        // 90% probability on index 0
        let mut logits = vec![10.0, 0.1, 0.1, 0.1];
        processor.apply_top_p(&mut logits);
        // Index 0 should survive (it alone exceeds 0.5)
        assert!(!logits[0].is_infinite());
    }

    #[test]
    fn test_repetition_penalty() {
        let config = LogitProcessorConfig {
            repetition_penalty: 2.0,
            ..Default::default()
        };
        let mut processor = LogitProcessor::new(config);
        processor.record_token(0); // Token 0 appeared once

        let mut logits = vec![1.0, 1.0, 1.0];
        processor.apply_penalties(&mut logits);
        // Token 0 should be penalized (positive logit / 2.0)
        assert!(logits[0] < logits[1]);
        assert_eq!(logits[1], 1.0); // Unseen tokens unchanged
    }

    #[test]
    fn test_frequency_penalty() {
        let config = LogitProcessorConfig {
            frequency_penalty: 0.5,
            ..Default::default()
        };
        let mut processor = LogitProcessor::new(config);
        processor.record_token(0);
        processor.record_token(0);
        processor.record_token(0); // Token 0 appeared 3 times

        let mut logits = vec![2.0, 2.0, 2.0];
        processor.apply_penalties(&mut logits);
        // Token 0: 2.0 - 0.5 * 3 = 0.5
        assert!((logits[0] - 0.5).abs() < 0.001);
        assert_eq!(logits[1], 2.0); // Unseen tokens unchanged
    }

    #[test]
    fn test_presence_penalty() {
        let config = LogitProcessorConfig {
            presence_penalty: 1.0,
            ..Default::default()
        };
        let mut processor = LogitProcessor::new(config);
        processor.record_token(1); // Token 1 is present

        let mut logits = vec![2.0, 2.0, 2.0];
        processor.apply_penalties(&mut logits);
        assert_eq!(logits[0], 2.0); // Not present, unchanged
        assert!((logits[1] - 1.0).abs() < 0.001); // Present, penalized
    }

    #[test]
    fn test_repetition_window() {
        let config = LogitProcessorConfig {
            repetition_penalty: 2.0,
            repetition_window: 2,
            ..Default::default()
        };
        let mut processor = LogitProcessor::new(config);
        processor.record_token(0);
        processor.record_token(1);
        processor.record_token(2);
        processor.record_token(3);
        // Window is 2, so only tokens 2 and 3 should be penalized
        let mut logits = vec![1.0, 1.0, 1.0, 1.0];
        processor.apply_penalties(&mut logits);
        // Token 0 and 1 are outside window → not penalized
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 1.0);
        // Token 2 and 3 are in window → penalized
        assert!(logits[2] < 1.0);
        assert!(logits[3] < 1.0);
    }

    #[test]
    fn test_full_pipeline_greedy() {
        let mut processor = LogitProcessor::new(LogitProcessorConfig::greedy());
        let logits = vec![0.1, 0.2, 0.9, 0.3];
        let result = processor.process_and_sample(&mut logits.clone());
        assert_eq!(result, 2);
    }

    #[test]
    fn test_token_history() {
        let mut history = TokenHistory::new();
        history.record_prompt(&[1, 2, 3, 2, 1]);
        assert_eq!(history.frequency(1, 0), 2);
        assert_eq!(history.frequency(2, 0), 2);
        assert_eq!(history.frequency(3, 0), 1);
        assert_eq!(history.frequency(4, 0), 0);
        assert!(history.present(1, 0));
        assert!(!history.present(4, 0));
    }

    #[test]
    fn test_token_history_window() {
        let mut history = TokenHistory::new();
        history.record_prompt(&[1, 2, 3, 4, 5]);
        // Window of 2: only tokens 4 and 5 visible
        assert_eq!(history.frequency(1, 2), 0);
        assert_eq!(history.frequency(5, 2), 1);
    }

    #[test]
    fn test_processor_records_after_sample() {
        let mut processor = LogitProcessor::new(LogitProcessorConfig::greedy());
        let logits = vec![1.0, 5.0, 2.0];
        let tok = processor.process_and_sample(&mut logits.clone());
        processor.record_token(tok as u32);
        assert_eq!(processor.history().frequency(1, 0), 1);
    }
}
