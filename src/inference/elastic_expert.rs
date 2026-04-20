//! Elastic expert activation — dynamic expert count based on DeviceProfile.
//!
//! On edge devices, activating fewer experts saves memory and compute.
//! On high-end devices, activating all experts gives maximum quality.
//!
//! DeviceProfile routing:
//! - Integrated (RPi): top-1 from 2 candidates (minimum compute)
//! - LowEnd (mobile): top-2 from 4 candidates (default)
//! - MidRange (desktop): top-2 from 4 candidates (full)
//! - HighEnd (discrete GPU): top-2 from 4 candidates with prefetch (full + fast)

use crate::device::profile::DeviceProfile;

/// Elastic MoE configuration derived from DeviceProfile.
#[derive(Clone, Debug)]
pub struct ElasticMoEConfig {
    /// Number of candidate experts to consider.
    pub num_candidates: usize,
    /// Number of experts to activate per token (top-k).
    pub top_k: usize,
    /// Whether to use predictive prefetching for expert weights.
    pub use_prefetch: bool,
    /// Whether to enable buddy substitution for cache misses.
    pub use_buddy: bool,
}

impl ElasticMoEConfig {
    /// Derive elastic config from device profile.
    pub fn from_profile(profile: DeviceProfile) -> Self {
        match profile {
            DeviceProfile::Integrated => Self {
                num_candidates: 2,
                top_k: 1,
                use_prefetch: false,
                use_buddy: false,
            },
            DeviceProfile::LowEnd => Self {
                num_candidates: 4,
                top_k: 2,
                use_prefetch: false,
                use_buddy: false,
            },
            DeviceProfile::MidRange => Self {
                num_candidates: 4,
                top_k: 2,
                use_prefetch: true,
                use_buddy: false,
            },
            DeviceProfile::HighEnd => Self {
                num_candidates: 4,
                top_k: 2,
                use_prefetch: true,
                use_buddy: true,
            },
        }
    }

    /// Default (full) configuration.
    pub fn full() -> Self {
        Self {
            num_candidates: 4,
            top_k: 2,
            use_prefetch: false,
            use_buddy: false,
        }
    }

    /// Minimum configuration for constrained devices.
    pub fn minimal() -> Self {
        Self {
            num_candidates: 2,
            top_k: 1,
            use_prefetch: false,
            use_buddy: false,
        }
    }
}

/// Select top-k experts from router logits with elastic configuration.
///
/// Unlike CpuMoELayer::top_k_select which uses fixed top_k and num_experts,
/// this uses the elastic config to dynamically adjust how many experts to consider.
///
/// Returns (selected_expert_indices, weights) both of length `config.top_k`.
pub fn elastic_top_k_select(
    gate_logits: &[f32],
    num_experts: usize,
    config: &ElasticMoEConfig,
) -> (Vec<usize>, Vec<f32>) {
    let candidates = config.num_candidates.min(num_experts);
    let top_k = config.top_k.min(candidates);

    // Find top-candidates from all experts
    let mut indexed: Vec<(usize, f32)> = gate_logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(idx, _)| idx < &candidates)
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut selected = vec![0usize; top_k];
    let mut weights = vec![0.0f32; top_k];

    // Softmax over top-k logits
    let max_logit = indexed.first().map(|&(_, v)| v).unwrap_or(0.0);
    let mut sum_exp = 0.0f32;
    for k in 0..top_k {
        if k < indexed.len() {
            let exp_val = (indexed[k].1 - max_logit).exp();
            sum_exp += exp_val;
            selected[k] = indexed[k].0;
            weights[k] = exp_val;
        }
    }
    if sum_exp > 0.0 {
        for w in &mut weights {
            *w /= sum_exp;
        }
    }

    (selected, weights)
}

/// Select top-k experts for a batch of tokens.
///
/// gate_logits: [seq_len, num_experts]
/// Returns: (selected[seq_len * top_k], weights[seq_len * top_k])
pub fn elastic_top_k_batch(
    gate_logits: &[f32],
    seq_len: usize,
    num_experts: usize,
    config: &ElasticMoEConfig,
) -> (Vec<usize>, Vec<f32>) {
    let top_k = config.top_k.min(config.num_candidates.min(num_experts));
    let mut selected = vec![0usize; seq_len * top_k];
    let mut weights = vec![0.0f32; seq_len * top_k];

    for t in 0..seq_len {
        let logits = &gate_logits[t * num_experts..(t + 1) * num_experts];
        let (s, w) = elastic_top_k_select(logits, num_experts, config);
        selected[t * top_k..(t + 1) * top_k].copy_from_slice(&s);
        weights[t * top_k..(t + 1) * top_k].copy_from_slice(&w);
    }

    (selected, weights)
}

/// Compute theoretical FLOPs savings from elastic config vs full activation.
pub fn compute_savings(config: &ElasticMoEConfig, full_config: &ElasticMoEConfig) -> f32 {
    let full_compute = full_config.top_k as f32;
    let elastic_compute = config.top_k as f32;
    if full_compute > 0.0 {
        1.0 - elastic_compute / full_compute
    } else {
        0.0
    }
}

/// Memory savings from reduced expert loading.
pub fn compute_memory_savings(config: &ElasticMoEConfig, full_config: &ElasticMoEConfig) -> f32 {
    let full_experts = full_config.top_k as f32;
    let elastic_experts = config.top_k as f32;
    if full_experts > 0.0 {
        1.0 - elastic_experts / full_experts
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_integrated_profile() -> DeviceProfile {
        DeviceProfile::Integrated
    }

    fn make_highend_profile() -> DeviceProfile {
        DeviceProfile::HighEnd
    }

    #[test]
    fn test_elastic_config_integrated() {
        let profile = make_integrated_profile();
        let config = ElasticMoEConfig::from_profile(profile);
        assert_eq!(config.num_candidates, 2);
        assert_eq!(config.top_k, 1);
        assert!(!config.use_prefetch);
        assert!(!config.use_buddy);
    }

    #[test]
    fn test_elastic_config_lowend() {
        let config = ElasticMoEConfig::from_profile(DeviceProfile::LowEnd);
        assert_eq!(config.num_candidates, 4);
        assert_eq!(config.top_k, 2);
        assert!(!config.use_prefetch);
    }

    #[test]
    fn test_elastic_config_highend() {
        let profile = make_highend_profile();
        let config = ElasticMoEConfig::from_profile(profile);
        assert_eq!(config.num_candidates, 4);
        assert_eq!(config.top_k, 2);
        assert!(config.use_prefetch);
        assert!(config.use_buddy);
    }

    #[test]
    fn test_elastic_top_k_single() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.8];
        let config = ElasticMoEConfig::minimal(); // top-1 from 2
        let (selected, weights) = elastic_top_k_select(&logits, 4, &config);

        assert_eq!(selected.len(), 1);
        assert_eq!(weights.len(), 1);
        // Should pick index 1 (0.5 > 0.1 among first 2 candidates)
        assert_eq!(selected[0], 1);
        assert!((weights[0] - 1.0).abs() < 1e-5); // Only 1 expert, softmax = 1.0
    }

    #[test]
    fn test_elastic_top_k_full() {
        let logits = vec![0.1f32, 0.5, 0.3, 0.8];
        let config = ElasticMoEConfig::full(); // top-2 from 4
        let (selected, weights) = elastic_top_k_select(&logits, 4, &config);

        assert_eq!(selected.len(), 2);
        // Top-2 from [0.1, 0.5, 0.3, 0.8] → indices 3 (0.8) and 1 (0.5)
        assert_eq!(selected[0], 3);
        assert_eq!(selected[1], 1);
        // Weights should sum to 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_elastic_top_k_batch() {
        // 3 tokens, 4 experts
        let logits = vec![
            0.1f32, 0.5, 0.3, 0.8,  // token 0
            0.9, 0.2, 0.1, 0.3,     // token 1
            0.4, 0.4, 0.4, 0.4,     // token 2 (tie)
        ];
        let config = ElasticMoEConfig::full();
        let (selected, weights) = elastic_top_k_batch(&logits, 3, 4, &config);

        assert_eq!(selected.len(), 6); // 3 tokens × 2 experts
        assert_eq!(weights.len(), 6);
        // Each pair should sum to ~1.0
        for t in 0..3 {
            let sum: f32 = weights[t * 2..t * 2 + 2].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "Token {} weights don't sum to 1.0: {}", t, sum);
        }
    }

    #[test]
    fn test_compute_savings() {
        let full = ElasticMoEConfig::full();
        let minimal = ElasticMoEConfig::minimal();
        let savings = compute_savings(&minimal, &full);
        assert!((savings - 0.5).abs() < 1e-5, "Expected 50% savings, got {}", savings);
    }

    #[test]
    fn test_compute_memory_savings() {
        let full = ElasticMoEConfig::full();
        let minimal = ElasticMoEConfig::minimal();
        let savings = compute_memory_savings(&minimal, &full);
        assert!((savings - 0.5).abs() < 1e-5);
    }
}
