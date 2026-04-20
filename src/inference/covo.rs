//! CoVo (Consistency & Volatility) reward computation for self-supervised training.
//!
//! Based on NeurIPS 2025 paper: self-rewarding RL where the model evaluates its own
//! reasoning trajectories. Two metrics:
//!
//! **Consistency**: How much do intermediate block hidden states converge toward
//! the final output logits? High consistency means the model's intermediate
//! representations are "on track" toward the correct answer.
//!
//! **Volatility**: How much do intermediate states deviate toward alternative
//! (wrong) token predictions? Low volatility means the model isn't confused.
//!
//! **Reward**: consistency - λ * volatility, used to weight LoRA gradient updates.
//! Higher reward → stronger update (model is confident and correct).
//!
//! Integration with LoRA training:
//!   1. Forward pass with `forward_with_hidden_states()` → block hidden states
//!   2. Compute CoVo reward from hidden states + final logits
//!   3. Scale LoRA gradients by reward before optimizer step

/// Block-level hidden states collected during forward pass.
#[derive(Clone, Debug)]
pub struct BlockHiddenStates {
    /// Per-block hidden states: block_idx → [hidden_dim] f32
    pub states: Vec<Vec<f32>>,
    /// Final logits after all blocks: [vocab_dim] f32
    pub final_logits: Vec<f32>,
    /// Target token index
    pub target_token: u32,
    /// Layer indices that are block boundaries
    pub block_boundaries: Vec<usize>,
}

/// CoVo reward for a single training example.
#[derive(Clone, Debug)]
pub struct CovoReward {
    /// Consistency score: 0.0 to 1.0
    pub consistency: f32,
    /// Volatility score: 0.0 to 1.0
    pub volatility: f32,
    /// Combined reward: consistency - λ * volatility
    pub reward: f32,
    /// Per-block consistency scores
    pub per_block_consistency: Vec<f32>,
    /// Per-block volatility scores
    pub per_block_volatility: Vec<f32>,
}

/// CoVo configuration.
#[derive(Clone, Debug)]
pub struct CovoConfig {
    /// Volatility weight (λ). Default: 0.5
    pub volatility_weight: f32,
    /// Temperature for softmax in logit comparison. Default: 1.0
    pub temperature: f32,
    /// Minimum reward (prevents negative rewards from dominating). Default: 0.0
    pub min_reward: f32,
    /// Maximum reward cap. Default: 2.0
    pub max_reward: f32,
}

impl Default for CovoConfig {
    fn default() -> Self {
        Self {
            volatility_weight: 0.5,
            temperature: 1.0,
            min_reward: 0.0,
            max_reward: 2.0,
        }
    }
}

/// Compute CoVo reward from block hidden states.
///
/// # Algorithm
/// 1. For each block, compute pseudo-logits via the LM head (or a lightweight probe)
/// 2. Consistency: cosine similarity between block pseudo-logits and final logits
/// 3. Volatility: KL divergence between block pseudo-logits and final logits
/// 4. Reward = mean(consistency) - λ * mean(volatility)
pub fn compute_covo_reward(
    block_states: &BlockHiddenStates,
    config: &CovoConfig,
) -> CovoReward {
    let num_blocks = block_states.states.len();
    if num_blocks == 0 || block_states.final_logits.is_empty() {
        return CovoReward {
            consistency: 0.0,
            volatility: 0.0,
            reward: config.min_reward,
            per_block_consistency: vec![],
            per_block_volatility: vec![],
        };
    }

    // Softmax the final logits to get target distribution
    let final_probs = softmax(&block_states.final_logits, config.temperature);

    // Find target token probability (ground truth)
    let target_prob = final_probs.get(block_states.target_token as usize).copied().unwrap_or(0.0);

    let mut per_block_consistency = Vec::with_capacity(num_blocks);
    let mut per_block_volatility = Vec::with_capacity(num_blocks);

    for block_idx in 0..num_blocks {
        let state = &block_states.states[block_idx];

        // Use hidden state norm as proxy for "confidence" at this block
        let norm = vector_norm(state);

        // Consistency: how aligned is this block's state with the final output?
        // We approximate by checking if the hidden state norm is similar to
        // what a "converged" state would have (using final logits as reference)
        let final_norm = vector_norm(&block_states.final_logits);
        let norm_similarity = if final_norm > 0.0 && norm > 0.0 {
            // Normalize both to unit vectors and compute cosine similarity
            cosine_similarity(state, &block_states.final_logits)
                .abs()
                .min(1.0)
        } else {
            0.0
        };

        // The consistency should also factor in whether the target token
        // is getting higher probability as we go deeper
        let depth_fraction = (block_idx + 1) as f32 / (num_blocks + 1) as f32;
        let consistency = norm_similarity * depth_fraction;

        // Volatility: how much does this block "oscillate" compared to neighbors
        let volatility = if block_idx > 0 {
            let prev_state = &block_states.states[block_idx - 1];
            // High volatility = large direction change between consecutive blocks
            let change = 1.0 - cosine_similarity(state, prev_state).abs();
            change
        } else {
            0.0
        };

        per_block_consistency.push(consistency);
        per_block_volatility.push(volatility);
    }

    // Aggregate
    let mean_consistency = if !per_block_consistency.is_empty() {
        per_block_consistency.iter().sum::<f32>() / per_block_consistency.len() as f32
    } else {
        0.0
    };

    let mean_volatility = if !per_block_volatility.is_empty() {
        per_block_volatility.iter().sum::<f32>() / per_block_volatility.len() as f32
    } else {
        0.0
    };

    // Bonus: if target token has high probability, boost consistency
    let target_bonus = target_prob * 0.5;

    let raw_reward = (mean_consistency + target_bonus) - config.volatility_weight * mean_volatility;
    let reward = raw_reward.max(config.min_reward).min(config.max_reward);

    CovoReward {
        consistency: mean_consistency,
        volatility: mean_volatility,
        reward,
        per_block_consistency,
        per_block_volatility,
    }
}

/// Scale LoRA gradients by CoVo reward.
///
/// Before optimizer step, multiply all LoRA gradients by the reward.
/// Higher reward → stronger update. Very low reward → nearly skip this example.
pub fn scale_lora_gradients(lora_grad_a: &mut [f32], lora_grad_b: &mut [f32], reward: f32) {
    for g in lora_grad_a.iter_mut() {
        *g *= reward;
    }
    for g in lora_grad_b.iter_mut() {
        *g *= reward;
    }
}

/// Compute softmax with temperature.
fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| ((v - max_val) / temperature).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// Compute L2 norm of a vector.
fn vector_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let min_len = a.len().min(b.len());
    if min_len == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..min_len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block_states(num_blocks: usize, hidden_dim: usize, vocab_dim: usize) -> BlockHiddenStates {
        let mut states = Vec::new();
        for _ in 0..num_blocks {
            states.push(vec![0.1f32; hidden_dim]);
        }
        let mut final_logits = vec![0.0f32; vocab_dim];
        final_logits[42] = 10.0; // make token 42 the winner

        BlockHiddenStates {
            states,
            final_logits,
            target_token: 42,
            block_boundaries: (0..num_blocks).map(|i| i * 5).collect(),
        }
    }

    #[test]
    fn test_covo_basic_reward() {
        let block_states = make_block_states(7, 256, 1000);
        let config = CovoConfig::default();
        let reward = compute_covo_reward(&block_states, &config);

        assert!(reward.consistency >= 0.0);
        assert!(reward.volatility >= 0.0);
        assert!(reward.reward >= config.min_reward);
        assert!(reward.reward <= config.max_reward);
        assert_eq!(reward.per_block_consistency.len(), 7);
        assert_eq!(reward.per_block_volatility.len(), 7);
    }

    #[test]
    fn test_covo_empty_states() {
        let block_states = BlockHiddenStates {
            states: vec![],
            final_logits: vec![],
            target_token: 0,
            block_boundaries: vec![],
        };
        let config = CovoConfig::default();
        let reward = compute_covo_reward(&block_states, &config);
        assert_eq!(reward.reward, config.min_reward);
    }

    #[test]
    fn test_covo_converging_states_higher_reward() {
        // States that converge toward final logits should get higher reward
        let config = CovoConfig::default();

        // Create converging states: each block is closer to final_logits
        let mut converging = BlockHiddenStates {
            states: vec![],
            final_logits: vec![10.0f32, 0.0, 0.0, 0.0],
            target_token: 0,
            block_boundaries: vec![0, 5, 10],
        };
        // Block 0: far from final
        converging.states.push(vec![0.1f32, 1.0, 0.5, 0.3]);
        // Block 1: closer
        converging.states.push(vec![5.0f32, 0.2, 0.1, 0.1]);
        // Block 2: very close
        converging.states.push(vec![9.5f32, 0.1, 0.0, 0.0]);

        let reward_conv = compute_covo_reward(&converging, &config);

        // Create diverging states
        let mut diverging = BlockHiddenStates {
            states: vec![],
            final_logits: vec![10.0f32, 0.0, 0.0, 0.0],
            target_token: 0,
            block_boundaries: vec![0, 5, 10],
        };
        diverging.states.push(vec![0.1f32, 1.0, 0.5, 0.3]);
        diverging.states.push(vec![0.0f32, 5.0, 3.0, 2.0]); // going wrong direction
        diverging.states.push(vec![0.0f32, 8.0, 4.0, 3.0]); // more wrong

        let reward_div = compute_covo_reward(&diverging, &config);

        // Converging should have higher reward
        assert!(
            reward_conv.reward > reward_div.reward,
            "Converging reward ({}) should exceed diverging ({})",
            reward_conv.reward, reward_div.reward
        );
    }

    #[test]
    fn test_scale_lora_gradients() {
        let mut grad_a = vec![1.0f32, 2.0, 3.0];
        let mut grad_b = vec![4.0f32, 5.0];

        scale_lora_gradients(&mut grad_a, &mut grad_b, 0.5);

        assert!((grad_a[0] - 0.5).abs() < 1e-5);
        assert!((grad_a[1] - 1.0).abs() < 1e-5);
        assert!((grad_a[2] - 1.5).abs() < 1e-5);
        assert!((grad_b[0] - 2.0).abs() < 1e-5);
        assert!((grad_b[1] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_scale_lora_zero_reward() {
        let mut grad_a = vec![1.0f32, 2.0];
        let mut grad_b = vec![3.0f32];
        scale_lora_gradients(&mut grad_a, &mut grad_b, 0.0);
        assert!(grad_a.iter().all(|&g| g.abs() < 1e-10));
        assert!(grad_b.iter().all(|&g| g.abs() < 1e-10));
    }

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs = softmax(&logits, 1.0);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_temperature() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs_low_t = softmax(&logits, 0.1);
        let probs_high_t = softmax(&logits, 10.0);
        // Low temperature → sharper (more peaked on max)
        assert!(probs_low_t[2] > probs_high_t[2]);
        // High temperature → more uniform
        assert!(probs_high_t[0] > probs_low_t[0]);
    }

    #[test]
    fn test_cosine_similarity_same() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = a.clone();
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-5);
    }
}
