//! Wire EntropyPredictor into BlockAttnResModel for adaptive block boundaries.
//!
//! Instead of fixed block_size iteration (e.g., 5 layers per block), the
//! EntropyPredictor uses output entropy to decide when to start a new block.
//! High entropy (uncertainty) triggers a block boundary, creating more blocks
//! where the model is confused and fewer where it's confident.
//!
//! Usage:
//!   1. Create AdaptiveBlockIterator from model config
//!   2. After each layer, call observe() with the layer's output
//!   3. is_boundary() returns true when entropy suggests starting a new block

use crate::model::config::{AdaptivePatchingConfig, EntropyPredictor};

/// Adaptive block boundary iterator.
///
/// Wraps EntropyPredictor to work with the layer-by-layer iteration in
/// CpuBlockAttnResModel::forward(). Instead of checking is_block_boundary()
/// at fixed intervals, queries entropy to decide dynamically.
pub struct AdaptiveBlockIterator {
    predictor: EntropyPredictor,
    /// Minimum layers between boundaries (prevents too many tiny blocks).
    min_layers: usize,
    /// Maximum layers before forcing a boundary (prevents too large blocks).
    max_layers: usize,
    /// Current layer index.
    current_layer: usize,
    /// Last boundary layer index.
    last_boundary: usize,
    /// Whether adaptive patching is enabled.
    enabled: bool,
    /// Pre-computed fixed boundaries (fallback when disabled).
    fixed_boundaries: Vec<usize>,
}

impl AdaptiveBlockIterator {
    /// Create from model parameters.
    ///
    /// If `use_adaptive` is false, uses fixed boundaries from num_layers/block_size.
    pub fn new(
        num_layers: usize,
        block_size: usize,
        use_adaptive: bool,
        adaptive_config: Option<AdaptivePatchingConfig>,
    ) -> Self {
        let config = adaptive_config.unwrap_or_default();
        let min_layers = config.min_patch_size.max(2);
        let max_layers = config.max_patch_size.min(block_size * 2);

        // Pre-compute fixed boundaries
        let fixed_boundaries: Vec<usize> = (0..num_layers)
            .step_by(block_size)
            .skip(1) // Skip layer 0
            .chain(std::iter::once(num_layers)) // Always end at last layer
            .map(|i| i.min(num_layers))
            .filter(|&i| i > 0)
            .collect();

        Self {
            predictor: EntropyPredictor::new(config),
            min_layers,
            max_layers,
            current_layer: 0,
            last_boundary: 0,
            enabled: use_adaptive,
            fixed_boundaries,
        }
    }

    /// Observe layer output and check if this is a block boundary.
    ///
    /// Call after each layer's forward pass.
    /// `hidden` is the layer output [seq, hidden_dim].
    /// Returns true if a block boundary should be created here.
    pub fn observe(&mut self, hidden: &[f32], seq: usize, hidden_dim: usize) -> bool {
        self.current_layer += 1;

        if !self.enabled {
            // Fixed boundaries
            return self.fixed_boundaries.contains(&self.current_layer);
        }

        // Compute entropy proxy from hidden state statistics
        let entropy = hidden_entropy_proxy(hidden, seq, hidden_dim);

        // Feed to predictor (using a pseudo-probability distribution)
        // We convert the proxy entropy to a simple 2-element distribution
        let p_confident = 1.0 / (1.0 + entropy.exp());
        let probs = vec![p_confident, 1.0 - p_confident];
        self.predictor.predict_boundary(&probs, self.current_layer);

        let layers_since_last = self.current_layer - self.last_boundary;

        // Check adaptive boundary conditions
        let entropy_boundary = layers_since_last >= self.min_layers
            && self.predictor.boundaries().last().copied().unwrap_or(0) == self.current_layer;

        // Force boundary at max_layers
        let forced_boundary = layers_since_last >= self.max_layers;

        // Force boundary at the last layer
        let final_boundary = self.fixed_boundaries.last().copied() == Some(self.current_layer);

        if entropy_boundary || forced_boundary || final_boundary {
            self.last_boundary = self.current_layer;
            true
        } else {
            false
        }
    }

    /// Whether adaptive mode is enabled.
    pub fn is_adaptive(&self) -> bool {
        self.enabled
    }

    /// Reset for a new sequence.
    pub fn reset(&mut self) {
        self.predictor.reset();
        self.current_layer = 0;
        self.last_boundary = 0;
    }

    /// Get the current block boundaries (for logging/debugging).
    pub fn boundaries(&self) -> &[usize] {
        self.predictor.boundaries()
    }
}

/// Compute entropy proxy from hidden state statistics.
///
/// Uses the variance of the hidden state as a proxy for entropy:
/// - Low variance = model is confident (low entropy)
/// - High variance = model is uncertain (high entropy)
fn hidden_entropy_proxy(hidden: &[f32], seq: usize, hidden_dim: usize) -> f32 {
    if hidden.is_empty() || seq == 0 || hidden_dim == 0 {
        return 0.0;
    }

    // Compute mean
    let n = (seq * hidden_dim).min(hidden.len());
    let mean: f32 = hidden[..n].iter().sum::<f32>() / n as f32;

    // Compute variance
    let variance: f32 = hidden[..n]
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>()
        / n as f32;

    // Log transform to get entropy-like scale
    (1.0 + variance).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_boundaries() {
        let mut iter = AdaptiveBlockIterator::new(10, 5, false, None);
        assert!(!iter.is_adaptive());

        // Fixed boundaries at layer 5 and 10
        assert!(!iter.observe(&[0.0f32; 64], 1, 64)); // layer 1
        assert!(!iter.observe(&[0.0f32; 64], 1, 64)); // layer 2
        assert!(!iter.observe(&[0.0f32; 64], 1, 64)); // layer 3
        assert!(!iter.observe(&[0.0f32; 64], 1, 64)); // layer 4
        assert!(iter.observe(&[0.0f32; 64], 1, 64));  // layer 5 = boundary
    }

    #[test]
    fn test_adaptive_boundaries_respect_min() {
        let config = AdaptivePatchingConfig {
            enabled: true,
            min_patch_size: 5,
            max_patch_size: 10,
            entropy_threshold: 0.5,
            smoothing_factor: 0.5,
        };
        let mut iter = AdaptiveBlockIterator::new(35, 5, true, Some(config));

        // High entropy at every layer, but min_patch_size = 5
        let high_entropy_hidden = vec![10.0f32; 64]; // High variance = high entropy
        for layer in 1..5 {
            assert!(!iter.observe(&high_entropy_hidden, 1, 64),
                "Layer {} should not be boundary (min_patch_size=5)", layer);
        }
    }

    #[test]
    fn test_adaptive_forces_boundary_at_max() {
        let config = AdaptivePatchingConfig {
            enabled: true,
            min_patch_size: 2,
            max_patch_size: 5,
            entropy_threshold: 100.0, // Very high, so entropy never triggers
            smoothing_factor: 0.5,
        };
        let mut iter = AdaptiveBlockIterator::new(35, 5, true, Some(config));

        let low_entropy_hidden = vec![0.1f32; 64]; // Low variance
        for _layer in 1..5 {
            let _ = iter.observe(&low_entropy_hidden, 1, 64);
        }
        // Layer 5 should force a boundary (max_patch_size = 5)
        assert!(iter.observe(&low_entropy_hidden, 1, 64),
            "Should force boundary at max_patch_size");
    }

    #[test]
    fn test_entropy_proxy_low() {
        let hidden = vec![1.0f32; 64]; // Zero variance
        let entropy = hidden_entropy_proxy(&hidden, 1, 64);
        assert!(entropy < 0.1, "Low variance should give low entropy proxy, got {}", entropy);
    }

    #[test]
    fn test_entropy_proxy_high() {
        let mut hidden = vec![0.0f32; 64];
        for i in 0..64 {
            hidden[i] = (i as f32 - 32.0) * 10.0; // High variance
        }
        let entropy = hidden_entropy_proxy(&hidden, 1, 64);
        assert!(entropy > 1.0, "High variance should give high entropy proxy, got {}", entropy);
    }

    #[test]
    fn test_entropy_proxy_empty() {
        let entropy = hidden_entropy_proxy(&[], 0, 0);
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_reset() {
        let mut iter = AdaptiveBlockIterator::new(35, 5, true, None);
        iter.observe(&[0.0f32; 64], 1, 64);
        iter.observe(&[0.0f32; 64], 1, 64);
        assert_eq!(iter.current_layer, 2);
        iter.reset();
        assert_eq!(iter.current_layer, 0);
    }
}
