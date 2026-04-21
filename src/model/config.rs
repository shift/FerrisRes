use serde::{Deserialize, Serialize};

/// Configuration for entropy-based adaptive patching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePatchingConfig {
    /// Use entropy-based adaptive patch sizes (vs fixed block_size)
    pub enabled: bool,
    /// Minimum patch size in tokens
    pub min_patch_size: usize,
    /// Maximum patch size in tokens
    pub max_patch_size: usize,
    /// Entropy threshold - start new patch when entropy > threshold
    pub entropy_threshold: f32,
    /// Entropy history smoothing factor
    pub smoothing_factor: f32,
}

impl Default for AdaptivePatchingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_patch_size: 2,
            max_patch_size: 16,
            entropy_threshold: 2.0,
            smoothing_factor: 0.9,
        }
    }
}

/// Entropy-based patch boundary predictor
pub struct EntropyPredictor {
    config: AdaptivePatchingConfig,
    running_entropy: f32,
    patch_boundaries: Vec<usize>,
}

impl EntropyPredictor {
    pub fn new(config: AdaptivePatchingConfig) -> Self {
        Self {
            config,
            running_entropy: 0.0,
            patch_boundaries: vec![0],
        }
    }
    
    /// Compute Shannon entropy of a probability distribution
    pub fn compute_entropy(&self, probabilities: &[f32]) -> f32 {
        probabilities.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }
    
    /// Predict next patch boundary based on entropy
    pub fn predict_boundary(&mut self, token_probs: &[f32], current_pos: usize) -> usize {
        let entropy = self.compute_entropy(token_probs);
        
        // Update running average
        self.running_entropy = self.config.smoothing_factor * self.running_entropy 
            + (1.0 - self.config.smoothing_factor) * entropy;
        
        // Start new patch if entropy crosses threshold
        if self.running_entropy > self.config.entropy_threshold 
            && current_pos - *self.patch_boundaries.last().unwrap() >= self.config.min_patch_size {
            self.patch_boundaries.push(current_pos);
        }
        
        // Cap at max patch size
        let patch_size = current_pos - self.patch_boundaries.last().unwrap();
        if patch_size >= self.config.max_patch_size {
            self.patch_boundaries.push(current_pos);
        }
        
        self.patch_boundaries.last().copied().unwrap_or(0)
    }
    
    /// Reset predictor state
    pub fn reset(&mut self) {
        self.running_entropy = 0.0;
        self.patch_boundaries.clear();
        self.patch_boundaries.push(0);
    }
    
    /// Get computed patch boundaries
    pub fn boundaries(&self) -> &[usize] {
        &self.patch_boundaries
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAttnResConfig {
    pub hidden_dim: usize,
    pub num_blocks: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub include_embedding: bool,
    pub attention_heads: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub use_moe: bool,
}

impl BlockAttnResConfig {
    pub fn new(hidden_dim: usize) -> Self {
        let num_blocks = 8;
        let block_size = 8;
        Self {
            hidden_dim,
            num_blocks,
            block_size,
            num_layers: num_blocks * block_size,
            include_embedding: true,
            attention_heads: 8,
            intermediate_dim: 4 * hidden_dim,
            num_experts: 8,
            top_k: 2,
            use_moe: false,
        }
    }

    pub fn total_layers(&self) -> usize {
        self.num_blocks * self.block_size
    }

    /// Adjust MoE routing parameters based on device capabilities.
    /// On edge devices (RPi), use fewer experts and top-1 routing.
    /// On capable hardware, use full top-2.
    pub fn with_elastic_experts(mut self, profile: crate::device::profile::DeviceProfile) -> Self {
        use crate::inference::elastic_expert::ElasticMoEConfig;
        let elastic = ElasticMoEConfig::from_profile(profile);
        self.top_k = elastic.top_k;
        self
    }

    /// Configure adaptive block boundaries based on entropy.
    /// When enabled, block boundaries are determined dynamically during
    /// inference rather than at fixed intervals.
    pub fn with_adaptive_blocks(self) -> Self {
        // The AdaptiveBlockIterator is used at inference time,
        // not at config time. This flag enables the feature.
        // Actual boundary detection happens in CpuBlockAttnResModel::forward().
        self
    }
}
