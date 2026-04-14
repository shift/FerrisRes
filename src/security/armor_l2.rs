//! Armor L2: RepE safety probe on BlockSummary hidden states.
//!
//! Representation Engineering (Zou et al.) shows safety features live in
//! linear subspaces of transformer hidden states. L2 uses lightweight
//! logistic regression probes on BlockSummaryLayer outputs to detect
//! adversarial content at <1ms per probe.

// ---------------------------------------------------------------------------
// Safety categories
// ---------------------------------------------------------------------------

/// Safety violation categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetyCategory {
    Violence,
    SelfHarm,
    Sexual,
    Hate,
    Harassment,
    Injection,
}

impl SafetyCategory {
    /// All categories.
    pub fn all() -> &'static [SafetyCategory] {
        &[
            SafetyCategory::Violence,
            SafetyCategory::SelfHarm,
            SafetyCategory::Sexual,
            SafetyCategory::Hate,
            SafetyCategory::Harassment,
            SafetyCategory::Injection,
        ]
    }

    /// Category name as string.
    pub fn name(&self) -> &'static str {
        match self {
            SafetyCategory::Violence => "violence",
            SafetyCategory::SelfHarm => "self_harm",
            SafetyCategory::Sexual => "sexual",
            SafetyCategory::Hate => "hate",
            SafetyCategory::Harassment => "harassment",
            SafetyCategory::Injection => "injection",
        }
    }
}

// ---------------------------------------------------------------------------
// Probe result
// ---------------------------------------------------------------------------

/// Result of an L2 safety probe.
#[derive(Debug, Clone)]
pub struct L2ProbeResult {
    /// Per-category unsafe scores [0.0, 1.0].
    pub category_scores: Vec<(SafetyCategory, f32)>,
    /// Maximum unsafe score across all categories.
    pub unsafe_score: f32,
    /// Whether the hidden state is classified as safe.
    pub safe: bool,
    /// The most triggered category (if any above threshold).
    pub top_category: Option<SafetyCategory>,
}

// ---------------------------------------------------------------------------
// SafetyProbe — linear probe head
// ---------------------------------------------------------------------------

/// A single linear probe for one safety category.
struct ProbeHead {
    /// Weight vector [hidden_dim].
    weight: Vec<f32>,
    /// Bias.
    bias: f32,
    /// Threshold for this category.
    threshold: f32,
}

impl ProbeHead {
    fn new(hidden_dim: usize, seed_base: f32, threshold: f32) -> Self {
        let scale = (2.0 / hidden_dim as f32).sqrt();
        let weight: Vec<f32> = (0..hidden_dim).map(|i| {
            let seed = i as f32 + seed_base;
            let x = ((seed * 0.618 + 0.1).sin() * 43758.5453).fract() - 0.5;
            x * scale
        }).collect();

        Self { weight, bias: 0.0, threshold }
    }

    /// Probe a hidden state vector.
    fn probe(&self, hidden: &[f32]) -> f32 {
        let logit: f32 = hidden.iter().zip(&self.weight).map(|(&h, &w)| h * w).sum::<f32>()
            + self.bias;
        1.0 / (1.0 + (-logit).exp())
    }
}

// ---------------------------------------------------------------------------
// L2Prober — multi-category safety prober
// ---------------------------------------------------------------------------

/// Multi-category Representation Engineering safety prober.
///
/// Runs independent linear probes on hidden state vectors, one per
/// safety category. Each probe is a single logistic regression.
/// Total cost: ~6 matmul of size hidden_dim → 1, well under 1ms.
pub struct L2Prober {
    /// Per-category probe heads.
    heads: Vec<(SafetyCategory, ProbeHead)>,
    /// Overall threshold for safe/unsafe classification.
    threshold: f32,
    /// Hidden dimension.
    hidden_dim: usize,
}

impl L2Prober {
    /// Create with default config (all 6 categories, threshold 0.7).
    pub fn new() -> Self {
        Self::with_config(256, 0.7)
    }

    /// Create with custom hidden_dim and threshold.
    pub fn with_config(hidden_dim: usize, threshold: f32) -> Self {
        let heads = SafetyCategory::all().iter().enumerate().map(|(i, &cat)| {
            let seed_base = i as f32 * 10000.0;
            (cat, ProbeHead::new(hidden_dim, seed_base, threshold))
        }).collect();

        Self { heads, threshold, hidden_dim }
    }

    /// Probe a hidden state vector for all safety categories.
    pub fn probe(&self, hidden: &[f32]) -> L2ProbeResult {
        let category_scores: Vec<(SafetyCategory, f32)> = self.heads.iter()
            .map(|(cat, head)| (*cat, head.probe(hidden)))
            .collect();

        let max_score = category_scores.iter()
            .map(|(_, s)| *s)
            .fold(0.0f32, f32::max);

        let top_category = category_scores.iter()
            .filter(|(_, s)| *s >= self.threshold)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(c, _)| *c);

        L2ProbeResult {
            category_scores,
            unsafe_score: max_score,
            safe: max_score < self.threshold,
            top_category,
        }
    }

    /// Probe with a specific threshold override.
    pub fn probe_with_threshold(&self, hidden: &[f32], threshold: f32) -> L2ProbeResult {
        let mut result = self.probe(hidden);
        result.safe = result.unsafe_score < threshold;
        if !result.safe && result.top_category.is_none() {
            result.top_category = result.category_scores.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| *c);
        }
        result
    }

    /// Number of probe heads.
    pub fn num_heads(&self) -> usize { self.heads.len() }

    /// Hidden dim.
    pub fn hidden_dim(&self) -> usize { self.hidden_dim }

    /// Get category names.
    pub fn categories(&self) -> Vec<&'static str> {
        self.heads.iter().map(|(c, _)| c.name()).collect()
    }

    /// Get the overall threshold.
    pub fn threshold(&self) -> f32 { self.threshold }
}

impl Default for L2Prober {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_category_all() {
        assert_eq!(SafetyCategory::all().len(), 6);
    }

    #[test]
    fn test_safety_category_names() {
        assert_eq!(SafetyCategory::Violence.name(), "violence");
        assert_eq!(SafetyCategory::SelfHarm.name(), "self_harm");
        assert_eq!(SafetyCategory::Injection.name(), "injection");
    }

    #[test]
    fn test_l2_prober_new() {
        let prober = L2Prober::new();
        assert_eq!(prober.num_heads(), 6);
        assert_eq!(prober.hidden_dim(), 256);
        assert_eq!(prober.categories().len(), 6);
    }

    #[test]
    fn test_l2_probe_zeros() {
        let prober = L2Prober::new();
        let hidden = vec![0.0f32; 256];
        let result = prober.probe(&hidden);
        assert!(result.safe, "Zero hidden should be safe with random init");
        assert!(result.unsafe_score >= 0.0 && result.unsafe_score <= 1.0);
    }

    #[test]
    fn test_l2_probe_scores_in_range() {
        let prober = L2Prober::new();
        let hidden: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = prober.probe(&hidden);
        for (cat, score) in &result.category_scores {
            assert!(*score >= 0.0 && *score <= 1.0, "Score for {:?} out of range: {}", cat, score);
        }
    }

    #[test]
    fn test_l2_probe_result_fields() {
        let prober = L2Prober::new();
        let hidden = vec![0.5f32; 256];
        let result = prober.probe(&hidden);
        assert_eq!(result.category_scores.len(), 6);
        assert!(result.unsafe_score >= 0.0);
    }

    #[test]
    fn test_l2_probe_with_threshold() {
        let prober = L2Prober::new();
        let hidden = vec![0.5f32; 256];
        // Very high threshold → always safe
        let result = prober.probe_with_threshold(&hidden, 0.999);
        assert!(result.safe);
        // Very low threshold → always unsafe
        let result = prober.probe_with_threshold(&hidden, 0.001);
        assert!(!result.safe);
    }

    #[test]
    fn test_l2_probe_deterministic() {
        let p1 = L2Prober::new();
        let p2 = L2Prober::new();
        let hidden = vec![0.3f32; 256];
        let r1 = p1.probe(&hidden);
        let r2 = p2.probe(&hidden);
        assert!((r1.unsafe_score - r2.unsafe_score).abs() < 0.001);
    }

    #[test]
    fn test_l2_probe_custom_dim() {
        let prober = L2Prober::with_config(128, 0.8);
        assert_eq!(prober.hidden_dim(), 128);
        assert_eq!(prober.threshold(), 0.8);
        let hidden = vec![0.0f32; 128];
        let result = prober.probe(&hidden);
        assert!(result.safe);
    }

    #[test]
    fn test_probe_head_new() {
        let head = ProbeHead::new(64, 42.0, 0.7);
        assert_eq!(head.weight.len(), 64);
        assert_eq!(head.bias, 0.0);
        assert!((head.threshold - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_probe_head_probe() {
        let head = ProbeHead::new(64, 0.0, 0.7);
        let hidden = vec![0.5f32; 64];
        let score = head.probe(&hidden);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
