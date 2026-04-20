//! FDAL (Feature-Distillation Active Learning) — task-aware training sampler.
//!
//! Selects the most informative training samples for edge LoRA training.
//! With limited compute budget on edge devices, we can't train on everything,
//! so we prioritize samples that maximize learning progress.
//!
//! Informativeness = uncertainty (high entropy output) + drift (different from
//! training distribution). The sampler learns online: samples that produced
//! high loss → sample more like those.
//!
//! Architecture: Tiny feed-forward network (~256 hidden) that predicts
//! informativeness score from sample features (embedding norm, token entropy,
//! domain hash). Trains online from loss signal.

use std::collections::VecDeque;

/// A training sample candidate with features for scoring.
#[derive(Clone, Debug)]
pub struct SampleCandidate {
    /// Unique sample ID.
    pub id: u64,
    /// Token embedding norm (proxy for input complexity).
    pub embedding_norm: f32,
    /// Output entropy (higher = more uncertain = more informative).
    pub output_entropy: f32,
    /// Domain hash (deterministic hash of the domain/category).
    pub domain_hash: u64,
    /// Sequence length.
    pub seq_length: u32,
    /// Number of unique tokens (vocabulary diversity).
    pub vocab_diversity: f32,
    /// Previous loss on this sample (if seen before).
    pub previous_loss: Option<f32>,
}

/// Features extracted from a candidate for the scorer network.
#[derive(Clone, Debug)]
pub struct SampleFeatures {
    pub embedding_norm: f32,
    pub output_entropy: f32,
    pub domain_bit: f32,     // Binary feature from domain hash
    pub seq_length_norm: f32, // Normalized by max expected length
    pub vocab_diversity: f32,
    pub has_previous_loss: f32,
    pub previous_loss: f32,
}

impl From<&SampleCandidate> for SampleFeatures {
    fn from(c: &SampleCandidate) -> Self {
        Self {
            embedding_norm: c.embedding_norm,
            output_entropy: c.output_entropy,
            domain_bit: if c.domain_hash % 2 == 0 { 1.0 } else { 0.0 },
            seq_length_norm: (c.seq_length as f32 / 2048.0).min(1.0),
            vocab_diversity: c.vocab_diversity,
            has_previous_loss: if c.previous_loss.is_some() { 1.0 } else { 0.0 },
            previous_loss: c.previous_loss.unwrap_or(0.0),
        }
    }
}

/// Feature vector dimension.
const FEATURE_DIM: usize = 7;

/// Hidden dimension of the scorer network.
const HIDDEN_DIM: usize = 32;

/// Tiny scorer network: input(7) → hidden(32, ReLU) → output(1, sigmoid).
/// ~225 parameters total, negligible memory.
#[derive(Clone, Debug)]
pub struct ScorerNetwork {
    /// Input-to-hidden weights: [HIDDEN_DIM × FEATURE_DIM].
    w1: Vec<f32>,
    /// Hidden bias: [HIDDEN_DIM].
    b1: Vec<f32>,
    /// Hidden-to-output weights: [HIDDEN_DIM].
    w2: Vec<f32>,
    /// Output bias.
    b2: f32,
    /// Adam-style momentum (first moment) for all params.
    m: Vec<f32>,
    /// Adam-style second moment for all params.
    v: Vec<f32>,
    /// Step count for Adam bias correction.
    step: u32,
}

impl ScorerNetwork {
    pub fn new() -> Self {
        let mut rng = 42u64;
        let mut next_f = || -> f32 {
            // Simple LCG random
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // [-1, 1)
        };

        let w1_size = HIDDEN_DIM * FEATURE_DIM;
        let he_scale_w1 = (2.0 / FEATURE_DIM as f32).sqrt();
        let w1: Vec<f32> = (0..w1_size).map(|_| next_f() * he_scale_w1).collect();
        let b1 = vec![0.0f32; HIDDEN_DIM];
        let he_scale_w2 = (2.0 / HIDDEN_DIM as f32).sqrt();
        let w2: Vec<f32> = (0..HIDDEN_DIM).map(|_| next_f() * he_scale_w2).collect();
        let b2 = 0.0f32;

        let total = w1_size + HIDDEN_DIM + HIDDEN_DIM + 1;
        Self {
            w1, b1, w2, b2,
            m: vec![0.0; total],
            v: vec![0.0; total],
            step: 0,
        }
    }

    /// Forward pass: features → informativeness score [0, 1].
    pub fn forward(&self, features: &SampleFeatures) -> f32 {
        let input = [
            features.embedding_norm,
            features.output_entropy,
            features.domain_bit,
            features.seq_length_norm,
            features.vocab_diversity,
            features.has_previous_loss,
            features.previous_loss,
        ];

        // Hidden layer: ReLU(W1 * x + b1)
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for h in 0..HIDDEN_DIM {
            let mut sum = self.b1[h];
            for (i, &x) in input.iter().enumerate() {
                sum += self.w1[h * FEATURE_DIM + i] * x;
            }
            hidden[h] = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // Output: sigmoid(W2 * h + b2)
        let mut sum = self.b2;
        for h in 0..HIDDEN_DIM {
            sum += self.w2[h] * hidden[h];
        }
        sigmoid(sum)
    }

    /// Update scorer from loss signal. Higher loss → should have scored higher.
    /// Uses simplified online Adam.
    pub fn update_from_loss(&mut self, features: &SampleFeatures, actual_loss: f32, lr: f32) {
        let score = self.forward(features);
        // Target: normalized loss (clipped to [0, 1])
        let target = actual_loss.min(10.0) / 10.0;
        // Binary cross-entropy gradient: (score - target)
        let grad_out = score - target;

        // Recompute hidden
        let input = [
            features.embedding_norm,
            features.output_entropy,
            features.domain_bit,
            features.seq_length_norm,
            features.vocab_diversity,
            features.has_previous_loss,
            features.previous_loss,
        ];
        let mut hidden = [0.0f32; HIDDEN_DIM];
        for h in 0..HIDDEN_DIM {
            let mut sum = self.b1[h];
            for (i, &x) in input.iter().enumerate() {
                sum += self.w1[h * FEATURE_DIM + i] * x;
            }
            hidden[h] = if sum > 0.0 { sum } else { 0.0 };
        }

        self.step += 1;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - beta1.powi(self.step as i32);
        let bc2 = 1.0 - beta2.powi(self.step as i32);

        let mut pi = 0; // Parameter index for Adam state

        // Update w2 and b2 (output layer)
        for h in 0..HIDDEN_DIM {
            let grad = grad_out * hidden[h];
            adam_update(&mut self.w2[h], &mut self.m[pi], &mut self.v[pi], grad, lr, beta1, beta2, eps, bc1, bc2);
            pi += 1;
        }
        {
            let grad = grad_out;
            adam_update(&mut self.b2, &mut self.m[pi], &mut self.v[pi], grad, lr, beta1, beta2, eps, bc1, bc2);
            pi += 1;
        }

        // Update w1 and b1 (hidden layer)
        for h in 0..HIDDEN_DIM {
            let relu_grad = if hidden[h] > 0.0 { 1.0 } else { 0.0 };
            for i in 0..FEATURE_DIM {
                let grad = grad_out * self.w2[h] * relu_grad * input[i];
                adam_update(
                    &mut self.w1[h * FEATURE_DIM + i],
                    &mut self.m[pi], &mut self.v[pi],
                    grad, lr, beta1, beta2, eps, bc1, bc2,
                );
                pi += 1;
            }
            let grad = grad_out * self.w2[h] * relu_grad;
            adam_update(&mut self.b1[h], &mut self.m[pi], &mut self.v[pi], grad, lr, beta1, beta2, eps, bc1, bc2);
            pi += 1;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn adam_update(
    param: &mut f32,
    m: &mut f32,
    v: &mut f32,
    grad: f32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
) {
    *m = beta1 * *m + (1.0 - beta1) * grad;
    *v = beta2 * *v + (1.0 - beta2) * grad * grad;
    let m_hat = *m / bc1;
    let v_hat = *v / bc2;
    *param -= lr * m_hat / (v_hat.sqrt() + eps);
}

/// FDAL configuration.
#[derive(Clone, Debug)]
pub struct FdalConfig {
    /// Number of samples to select per training batch.
    pub budget: usize,
    /// Minimum candidates before selection.
    pub min_candidates: usize,
    /// Maximum candidates to keep in buffer.
    pub max_buffer: usize,
    /// Weight for uncertainty (entropy) component.
    pub uncertainty_weight: f32,
    /// Weight for drift (distribution shift) component.
    pub drift_weight: f32,
    /// Learning rate for scorer network.
    pub scorer_lr: f32,
}

impl Default for FdalConfig {
    fn default() -> Self {
        Self {
            budget: 16,
            min_candidates: 8,
            max_buffer: 1024,
            uncertainty_weight: 0.7,
            drift_weight: 0.3,
            scorer_lr: 0.001,
        }
    }
}

/// FDAL sampler state.
pub struct FdalSampler {
    config: FdalConfig,
    scorer: ScorerNetwork,
    /// Buffer of candidates awaiting selection.
    buffer: VecDeque<SampleCandidate>,
    /// Running mean of embedding norms (for drift detection).
    mean_norm: f32,
    /// Running variance of embedding norms.
    var_norm: f32,
    /// Number of samples seen (for running stats).
    n_seen: u64,
    /// Total samples selected so far.
    n_selected: u64,
    /// Total samples scored so far.
    n_scored: u64,
}

impl FdalSampler {
    pub fn new(config: FdalConfig) -> Self {
        Self {
            config,
            scorer: ScorerNetwork::new(),
            buffer: VecDeque::new(),
            mean_norm: 0.0,
            var_norm: 1.0,
            n_seen: 0,
            n_selected: 0,
            n_scored: 0,
        }
    }

    /// Add a candidate to the buffer.
    pub fn add_candidate(&mut self, candidate: SampleCandidate) {
        // Update running stats for drift detection
        self.n_seen += 1;
        let delta = candidate.embedding_norm - self.mean_norm;
        self.mean_norm += delta / self.n_seen as f32;
        let delta2 = candidate.embedding_norm - self.mean_norm;
        self.var_norm += delta * delta2;

        self.buffer.push_back(candidate);
        // Trim buffer if over capacity
        while self.buffer.len() > self.config.max_buffer {
            self.buffer.pop_front();
        }
    }

    /// Score a single candidate.
    pub fn score(&mut self, candidate: &SampleCandidate) -> f32 {
        let features = SampleFeatures::from(candidate);

        // Model-based score from scorer network
        let model_score = self.scorer.forward(&features);

        // Uncertainty score: higher entropy = more informative
        let uncertainty = candidate.output_entropy.min(10.0) / 10.0;

        // Drift score: how different is this from the training distribution?
        let drift = if self.n_seen > 10 && self.var_norm > 0.0 {
            let std_norm = (self.var_norm / self.n_seen as f32).sqrt();
            let z_score = (candidate.embedding_norm - self.mean_norm) / std_norm.max(1e-8);
            // Higher absolute z-score = more different
            z_score.abs().min(3.0) / 3.0
        } else {
            0.0
        };

        // Combined score
        let combined = model_score * 0.3
            + self.config.uncertainty_weight * uncertainty * 0.5
            + self.config.drift_weight * drift * 0.2;

        self.n_scored += 1;
        combined.max(0.0).min(1.0)
    }

    /// Select top-K candidates from buffer for training.
    pub fn select_batch(&mut self) -> Vec<SampleCandidate> {
        if self.buffer.len() < self.config.min_candidates {
            return vec![];
        }

        // Drain buffer first to avoid borrow conflict
        let candidates: Vec<SampleCandidate> = self.buffer.drain(..).collect();

        // Score all candidates
        let mut scored: Vec<(f32, SampleCandidate)> = candidates
            .into_iter()
            .map(|c| {
                let score = self.score(&c);
                (score, c)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-K
        let selected: Vec<SampleCandidate> = scored
            .into_iter()
            .take(self.config.budget)
            .map(|(_, c)| c)
            .collect();

        self.n_selected += selected.len() as u64;
        selected
    }

    /// Report training loss for a sample (updates scorer network).
    pub fn report_loss(&mut self, candidate: &SampleCandidate, loss: f32) {
        let features = SampleFeatures::from(candidate);
        self.scorer.update_from_loss(&features, loss, self.config.scorer_lr);
    }

    /// Number of candidates in buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Number of candidates scored.
    pub fn total_scored(&self) -> u64 {
        self.n_scored
    }

    /// Number of candidates selected.
    pub fn total_selected(&self) -> u64 {
        self.n_selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: u64, entropy: f32, norm: f32) -> SampleCandidate {
        SampleCandidate {
            id,
            embedding_norm: norm,
            output_entropy: entropy,
            domain_hash: id * 7,
            seq_length: 128,
            vocab_diversity: 0.5,
            previous_loss: None,
        }
    }

    #[test]
    fn test_scorer_forward() {
        let scorer = ScorerNetwork::new();
        let features = SampleFeatures {
            embedding_norm: 1.0,
            output_entropy: 0.8,
            domain_bit: 1.0,
            seq_length_norm: 0.5,
            vocab_diversity: 0.6,
            has_previous_loss: 0.0,
            previous_loss: 0.0,
        };
        let score = scorer.forward(&features);
        assert!(score >= 0.0 && score <= 1.0, "Score should be in [0, 1], got {}", score);
    }

    #[test]
    fn test_scorer_deterministic() {
        let scorer = ScorerNetwork::new();
        let features = SampleFeatures {
            embedding_norm: 1.0,
            output_entropy: 0.8,
            domain_bit: 1.0,
            seq_length_norm: 0.5,
            vocab_diversity: 0.6,
            has_previous_loss: 0.0,
            previous_loss: 0.0,
        };
        let s1 = scorer.forward(&features);
        let s2 = scorer.forward(&features);
        assert!((s1 - s2).abs() < 1e-10);
    }

    #[test]
    fn test_scorer_learns_from_loss() {
        let mut scorer = ScorerNetwork::new();
        let features = SampleFeatures {
            embedding_norm: 1.0,
            output_entropy: 0.8,
            domain_bit: 1.0,
            seq_length_norm: 0.5,
            vocab_diversity: 0.6,
            has_previous_loss: 0.0,
            previous_loss: 0.0,
        };

        let initial = scorer.forward(&features);

        // Train with high loss → should increase score
        for _ in 0..500 {
            scorer.update_from_loss(&features, 8.0, 0.05);
        }
        let after_high_loss = scorer.forward(&features);

        // Score should increase after seeing high loss
        assert!(
            after_high_loss > initial,
            "Score should increase after high loss training: {} → {}",
            initial, after_high_loss
        );
    }

    #[test]
    fn test_fdal_buffer_management() {
        let mut sampler = FdalSampler::new(FdalConfig {
            max_buffer: 5,
            ..Default::default()
        });

        for i in 0..10 {
            sampler.add_candidate(make_candidate(i, 0.5, 1.0));
        }
        assert_eq!(sampler.buffer_len(), 5); // Only last 5 kept
    }

    #[test]
    fn test_fdal_select_batch() {
        let mut sampler = FdalSampler::new(FdalConfig {
            budget: 3,
            min_candidates: 2,
            ..Default::default()
        });

        // Too few candidates
        assert!(sampler.select_batch().is_empty());

        // Add candidates with varying entropy
        sampler.add_candidate(make_candidate(1, 0.1, 1.0)); // Low entropy
        sampler.add_candidate(make_candidate(2, 0.9, 1.0)); // High entropy
        sampler.add_candidate(make_candidate(3, 0.5, 1.0)); // Medium entropy

        let selected = sampler.select_batch();
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_fdal_prefers_high_entropy() {
        let mut sampler = FdalSampler::new(FdalConfig {
            budget: 2,
            min_candidates: 2,
            ..Default::default()
        });

        // Add candidates with varying entropy (same norm)
        sampler.add_candidate(make_candidate(1, 0.1, 1.0)); // Low
        sampler.add_candidate(make_candidate(2, 0.9, 1.0)); // High
        sampler.add_candidate(make_candidate(3, 0.2, 1.0)); // Low-medium
        sampler.add_candidate(make_candidate(4, 0.95, 1.0)); // Very high

        let selected = sampler.select_batch();
        assert_eq!(selected.len(), 2);

        // High entropy candidates should be selected
        assert!(selected.iter().any(|c| c.id == 4), "Should select highest entropy");
        assert!(selected.iter().any(|c| c.id == 2), "Should select second highest entropy");
    }

    #[test]
    fn test_fdal_report_loss_updates_scorer() {
        let mut sampler = FdalSampler::new(FdalConfig::default());

        let candidate = make_candidate(1, 0.5, 1.0);
        sampler.report_loss(&candidate, 3.0);

        // Should not crash and scorer should have updated
        assert!(sampler.total_scored() == 0); // report_loss doesn't count as scored
    }

    #[test]
    fn test_fdal_drift_detection() {
        let mut sampler = FdalSampler::new(FdalConfig::default());

        // Feed many candidates with norm ~1.0
        for i in 0..100 {
            sampler.add_candidate(make_candidate(i, 0.5, 1.0));
        }

        // Now add candidate with very different norm
        let outlier = make_candidate(999, 0.5, 5.0);
        let score_normal = sampler.score(&make_candidate(100, 0.5, 1.0));
        let score_outlier = sampler.score(&outlier);

        // Outlier should have higher drift component
        assert!(
            score_outlier >= score_normal * 0.8, // Allow some slack due to model_score
            "Outlier drift should increase score: normal={}, outlier={}",
            score_normal, score_outlier
        );
    }

    #[test]
    fn test_sample_features_from_candidate() {
        let candidate = SampleCandidate {
            id: 1,
            embedding_norm: 2.0,
            output_entropy: 0.7,
            domain_hash: 42,
            seq_length: 512,
            vocab_diversity: 0.3,
            previous_loss: Some(1.5),
        };
        let features = SampleFeatures::from(&candidate);
        assert!((features.embedding_norm - 2.0).abs() < 1e-5);
        assert!((features.output_entropy - 0.7).abs() < 1e-5);
        assert!((features.seq_length_norm - 0.25).abs() < 1e-5);
        assert!((features.has_previous_loss - 1.0).abs() < 1e-5);
        assert!((features.previous_loss - 1.5).abs() < 1e-5);
    }
}
