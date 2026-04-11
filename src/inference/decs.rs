//! DECS (Dynamic Early Conscious Stopping) — Reasoning Token Optimization.
//!
//! Identifies and reduces redundant reasoning tokens during chain-of-thought
//! generation. Uses token-level reward signals to detect when a reasoning chain
//! has converged, enabling early stopping without quality degradation.
//!
//! Based on ICLR 2026 research: 50%+ reasoning token reduction across 7 benchmarks.
//! Key idea: track "progress" of reasoning chain, stop when progress plateaus.

/// Token-level reward tracking for a single reasoning chain.
#[derive(Debug, Clone)]
pub struct TokenReward {
    /// Token position in the reasoning chain.
    pub position: usize,
    /// Estimated reward/value of this token's contribution.
    pub reward: f32,
    /// Running average reward up to this position.
    pub cumulative_avg: f32,
    /// Whether this token is classified as redundant.
    pub is_redundant: bool,
}

/// Configuration for DECS reasoning optimization.
#[derive(Debug, Clone)]
pub struct DecsConfig {
    /// Window size for detecting reward plateaus.
    pub plateau_window: usize,
    /// Minimum reward improvement to consider progress.
    pub min_improvement: f32,
    /// Number of consecutive plateau tokens before early stop.
    pub plateau_patience: usize,
    /// Maximum reasoning tokens before forced stop.
    pub max_reasoning_tokens: usize,
    /// Redundancy threshold: tokens with reward below this are flagged.
    pub redundancy_threshold: f32,
    /// Whether to actually stop generation early.
    pub early_stopping: bool,
    /// Smoothing factor for running reward average.
    pub smoothing: f32,
}

impl Default for DecsConfig {
    fn default() -> Self {
        Self {
            plateau_window: 8,
            min_improvement: 0.01,
            plateau_patience: 3,
            max_reasoning_tokens: 1024,
            redundancy_threshold: 0.05,
            early_stopping: true,
            smoothing: 0.9,
        }
    }
}

impl DecsConfig {
    /// Aggressive config for maximum token reduction.
    pub fn aggressive() -> Self {
        Self {
            plateau_window: 4,
            min_improvement: 0.02,
            plateau_patience: 2,
            redundancy_threshold: 0.1,
            ..Default::default()
        }
    }

    /// Conservative config that preserves more reasoning.
    pub fn conservative() -> Self {
        Self {
            plateau_window: 16,
            min_improvement: 0.005,
            plateau_patience: 5,
            redundancy_threshold: 0.02,
            ..Default::default()
        }
    }
}

/// Reasoning chain state for DECS tracking.
#[derive(Debug, Clone)]
pub struct ReasoningChain {
    tokens: Vec<u32>,
    rewards: Vec<f32>,
    cumulative_avg: f32,
    plateau_count: usize,
    stopped: bool,
    stop_reason: Option<StopReason>,
}

/// Reason for chain termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Reward plateaued for patience tokens.
    PlateauDetected,
    /// Maximum token limit reached.
    MaxTokens,
    /// Natural end-of-sequence.
    EndOfSequence,
}

impl ReasoningChain {
    /// Create a new reasoning chain.
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            rewards: Vec::new(),
            cumulative_avg: 0.0,
            plateau_count: 0,
            stopped: false,
            stop_reason: None,
        }
    }

    /// Get the token IDs in this chain.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Get the number of tokens in the chain.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if chain is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Check if chain has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Get stop reason.
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    /// Get reasoning efficiency metrics.
    pub fn metrics(&self) -> ReasoningMetrics {
        let total = self.tokens.len();
        let redundant = self.rewards.iter().filter(|&&r| r < 0.05).count();
        let productive = total - redundant;

        ReasoningMetrics {
            total_tokens: total,
            productive_tokens: productive,
            redundant_tokens: redundant,
            efficiency: if total > 0 { productive as f32 / total as f32 } else { 1.0 },
            avg_reward: if total > 0 { self.rewards.iter().sum::<f32>() / total as f32 } else { 0.0 },
            final_avg_reward: self.cumulative_avg,
            stop_reason: self.stop_reason,
        }
    }
}

impl Default for ReasoningChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics about a reasoning chain.
#[derive(Debug, Clone)]
pub struct ReasoningMetrics {
    pub total_tokens: usize,
    pub productive_tokens: usize,
    pub redundant_tokens: usize,
    pub efficiency: f32,
    pub avg_reward: f32,
    pub final_avg_reward: f32,
    pub stop_reason: Option<StopReason>,
}

impl std::fmt::Display for ReasoningMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tokens={}/{} efficiency={:.1}% avg_reward={:.3} stop={:?}",
            self.productive_tokens,
            self.total_tokens,
            self.efficiency * 100.0,
            self.avg_reward,
            self.stop_reason,
        )
    }
}

/// DECS reasoning optimizer.
pub struct DecsOptimizer {
    config: DecsConfig,
}

impl DecsOptimizer {
    /// Create a new DECS optimizer.
    pub fn new(config: DecsConfig) -> Self {
        Self { config }
    }

    /// Create with default config.
    pub fn default_optimizer() -> Self {
        Self::new(DecsConfig::default())
    }

    /// Evaluate a single token and update the reasoning chain.
    ///
    /// `reward` should be an estimate of this token's contribution to reasoning
    /// quality (e.g., from a reward model, log-probability delta, or entropy change).
    ///
    /// Returns `true` if generation should continue, `false` if early stopping is triggered.
    pub fn evaluate_token(&self, chain: &mut ReasoningChain, token_id: u32, reward: f32) -> bool {
        if chain.stopped {
            return false;
        }

        chain.tokens.push(token_id);
        chain.rewards.push(reward);

        // Update running average with exponential smoothing
        let _n = chain.tokens.len() as f32;
        chain.cumulative_avg = self.config.smoothing * chain.cumulative_avg
            + (1.0 - self.config.smoothing) * reward;

        // Check for max tokens
        if chain.tokens.len() >= self.config.max_reasoning_tokens {
            chain.stopped = true;
            chain.stop_reason = Some(StopReason::MaxTokens);
            return false;
        }

        // Check for plateau
        if chain.tokens.len() >= self.config.plateau_window {
            let recent_avg = self.recent_average(chain, self.config.plateau_window);
            let older_avg = self.recent_average(
                chain,
                self.config.plateau_window * 2,
            );

            let improvement = recent_avg - older_avg;
            if improvement < self.config.min_improvement {
                chain.plateau_count += 1;
                if chain.plateau_count >= self.config.plateau_patience && self.config.early_stopping {
                    chain.stopped = true;
                    chain.stop_reason = Some(StopReason::PlateauDetected);
                    return false;
                }
            } else {
                chain.plateau_count = 0;
            }
        }

        true
    }

    /// Compute a token-level reward estimate from log-probabilities.
    ///
    /// Higher log-prob → higher reward (model is confident = likely productive).
    /// Lower log-prob → lower reward (model is uncertain = potentially redundant).
    pub fn logprob_to_reward(&self, logprob: f32) -> f32 {
        // Normalize: typical logprobs range from -20 to 0
        // Map to 0.0 to 1.0 range
        (logprob + 20.0).max(0.0) / 20.0
    }

    /// Compute reward from entropy change.
    /// Decreasing entropy (model converging) = productive.
    /// Stable/increasing entropy = redundant.
    pub fn entropy_delta_to_reward(&self, prev_entropy: f32, curr_entropy: f32) -> f32 {
        let delta = prev_entropy - curr_entropy;
        // Positive delta = entropy decreased = productive
        // Clamp to [0, 1] range
        (delta * 2.0 + 0.5).clamp(0.0, 1.0)
    }

    /// Compute the average reward over the last n tokens.
    fn recent_average(&self, chain: &ReasoningChain, n: usize) -> f32 {
        let start = chain.rewards.len().saturating_sub(n);
        let slice = &chain.rewards[start..];
        if slice.is_empty() {
            return 0.0;
        }
        slice.iter().sum::<f32>() / slice.len() as f32
    }

    /// Analyze a completed reasoning chain and identify redundant tokens.
    pub fn analyze_chain(&self, chain: &ReasoningChain) -> Vec<TokenReward> {
        let mut rewards = Vec::with_capacity(chain.tokens.len());
        let mut running_avg = 0.0f32;

        for (i, &reward) in chain.rewards.iter().enumerate() {
            running_avg = self.config.smoothing * running_avg + (1.0 - self.config.smoothing) * reward;
            rewards.push(TokenReward {
                position: i,
                reward,
                cumulative_avg: running_avg,
                is_redundant: reward < self.config.redundancy_threshold,
            });
        }

        rewards
    }

    /// Filter a reasoning chain to remove redundant tokens.
    /// Returns a pruned list of token IDs with estimated redundancy removed.
    pub fn prune_redundant(&self, chain: &ReasoningChain) -> Vec<u32> {
        let analysis = self.analyze_chain(chain);
        analysis.iter()
            .zip(chain.tokens.iter())
            .filter(|(reward, _)| !reward.is_redundant)
            .map(|(_, &token)| token)
            .collect()
    }

    /// Get the config.
    pub fn config(&self) -> &DecsConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reasoning_chain_new() {
        let chain = ReasoningChain::new();
        assert!(chain.is_empty());
        assert!(!chain.is_stopped());
    }

    #[test]
    fn test_evaluate_token_continues() {
        let optimizer = DecsOptimizer::default_optimizer();
        let mut chain = ReasoningChain::new();

        // High-reward tokens should keep going
        for i in 0..5 {
            let cont = optimizer.evaluate_token(&mut chain, i, 0.8);
            assert!(cont, "Should continue at token {}", i);
        }
        assert_eq!(chain.len(), 5);
    }

    #[test]
    fn test_plateau_detection() {
        let config = DecsConfig {
            plateau_window: 4,
            plateau_patience: 2,
            min_improvement: 0.01,
            max_reasoning_tokens: 100,
            early_stopping: true,
            ..Default::default()
        };
        let optimizer = DecsOptimizer::new(config);
        let mut chain = ReasoningChain::new();

        // Start with improving rewards
        for i in 0..4 {
            optimizer.evaluate_token(&mut chain, i, 0.1 + i as f32 * 0.1);
        }

        // Then plateau
        for i in 0..20 {
            let cont = optimizer.evaluate_token(&mut chain, 4 + i, 0.5);
            if !cont {
                break;
            }
        }

        assert!(chain.is_stopped());
        assert_eq!(chain.stop_reason(), Some(StopReason::PlateauDetected));
    }

    #[test]
    fn test_max_tokens_stop() {
        let config = DecsConfig {
            max_reasoning_tokens: 5,
            ..Default::default()
        };
        let optimizer = DecsOptimizer::new(config);
        let mut chain = ReasoningChain::new();

        for i in 0..10 {
            if !optimizer.evaluate_token(&mut chain, i, 0.9) {
                break;
            }
        }

        assert!(chain.is_stopped());
        assert_eq!(chain.stop_reason(), Some(StopReason::MaxTokens));
        assert_eq!(chain.len(), 5);
    }

    #[test]
    fn test_logprob_to_reward() {
        let optimizer = DecsOptimizer::default_optimizer();
        // High logprob → high reward
        let high = optimizer.logprob_to_reward(-0.1);
        assert!(high > 0.9);
        // Low logprob → low reward
        let low = optimizer.logprob_to_reward(-15.0);
        assert!(low < 0.3);
    }

    #[test]
    fn test_entropy_delta_to_reward() {
        let optimizer = DecsOptimizer::default_optimizer();
        // Decreasing entropy → productive
        let productive = optimizer.entropy_delta_to_reward(3.0, 2.0);
        assert!(productive > 0.5);
        // Increasing entropy → less productive
        let unproductive = optimizer.entropy_delta_to_reward(2.0, 3.0);
        assert!(unproductive < 0.5);
    }

    #[test]
    fn test_analyze_chain() {
        let optimizer = DecsOptimizer::new(DecsConfig {
            redundancy_threshold: 0.1,
            ..Default::default()
        });
        let mut chain = ReasoningChain::new();
        optimizer.evaluate_token(&mut chain, 0, 0.8);
        optimizer.evaluate_token(&mut chain, 1, 0.05); // Redundant
        optimizer.evaluate_token(&mut chain, 2, 0.7);

        let analysis = optimizer.analyze_chain(&chain);
        assert_eq!(analysis.len(), 3);
        assert!(!analysis[0].is_redundant);
        assert!(analysis[1].is_redundant);
        assert!(!analysis[2].is_redundant);
    }

    #[test]
    fn test_prune_redundant() {
        let optimizer = DecsOptimizer::new(DecsConfig {
            redundancy_threshold: 0.1,
            ..Default::default()
        });
        let mut chain = ReasoningChain::new();
        optimizer.evaluate_token(&mut chain, 10, 0.8);
        optimizer.evaluate_token(&mut chain, 11, 0.05); // Redundant
        optimizer.evaluate_token(&mut chain, 12, 0.7);

        let pruned = optimizer.prune_redundant(&chain);
        assert_eq!(pruned, vec![10, 12]);
    }

    #[test]
    fn test_metrics() {
        let optimizer = DecsOptimizer::new(DecsConfig {
            max_reasoning_tokens: 5,
            ..Default::default()
        });
        let mut chain = ReasoningChain::new();
        for i in 0..5 {
            optimizer.evaluate_token(&mut chain, i, 0.5 + i as f32 * 0.1);
        }

        let metrics = chain.metrics();
        assert_eq!(metrics.total_tokens, 5);
        assert!(metrics.avg_reward > 0.5);
    }

    #[test]
    fn test_metrics_display() {
        let mut chain = ReasoningChain::new();
        chain.tokens.push(0);
        chain.rewards.push(0.5);
        let metrics = chain.metrics();
        let display = format!("{}", metrics);
        assert!(display.contains("tokens="));
    }

    #[test]
    fn test_no_early_stopping() {
        let config = DecsConfig {
            early_stopping: false,
            plateau_window: 4,
            plateau_patience: 2,
            max_reasoning_tokens: 100,
            ..Default::default()
        };
        let optimizer = DecsOptimizer::new(config);
        let mut chain = ReasoningChain::new();

        // All same reward = plateau, but early stopping disabled
        for i in 0..20 {
            if !optimizer.evaluate_token(&mut chain, i, 0.5) {
                break;
            }
        }

        // Should NOT have stopped (early stopping disabled)
        assert!(!chain.is_stopped());
    }
}
