//! Positional uncertainty tracking for long-range degradation detection.
//!
//! Tracks output entropy at specific sequence positions during inference.
//! When entropy spikes at certain positions (indicating confusion), generates
//! "Context Practice Goals" — training targets focused on those positions.
//!
//! Integration with IntrinsicMotivation:
//!   1. During forward pass, record per-position entropy
//!   2. Identify positions with high/degrading uncertainty
//!   3. Generate practice goals: "practice sequences of length N at position P"
//!   4. Feed goals into FDAL sampler for targeted training

use std::collections::HashMap;

/// Per-position entropy observation.
#[derive(Clone, Debug)]
pub struct PositionObservation {
    /// Absolute position in sequence.
    pub position: usize,
    /// Output entropy at this position.
    pub entropy: f32,
    /// Which sequence this was from (for tracking).
    pub sequence_id: u64,
    /// Timestamp of observation.
    pub timestamp: u64,
}

/// Aggregated statistics for a position range.
#[derive(Clone, Debug)]
pub struct PositionStats {
    /// Center position.
    pub center: usize,
    /// Number of observations.
    pub count: u32,
    /// Mean entropy at this position.
    pub mean_entropy: f32,
    /// Maximum entropy observed.
    pub max_entropy: f32,
    /// Minimum entropy observed.
    pub min_entropy: f32,
    /// Trend: positive = entropy increasing (degrading), negative = improving.
    pub trend: f32,
}

/// A generated practice goal targeting high-uncertainty positions.
#[derive(Clone, Debug)]
pub struct PracticeGoal {
    /// Goal description.
    pub description: String,
    /// Target position range [start, end].
    pub position_range: (usize, usize),
    /// Current mean entropy in this range.
    pub current_entropy: f32,
    /// Baseline entropy (expected "good" level).
    pub baseline_entropy: f32,
    /// Degradation factor: current / baseline.
    pub degradation: f32,
    /// Priority for training (higher = more urgent).
    pub priority: f32,
}

/// Configuration for positional uncertainty tracking.
#[derive(Clone, Debug)]
pub struct PositionalUncertaintyConfig {
    /// Bin size for aggregating position observations.
    pub bin_size: usize,
    /// Maximum sequence length to track.
    pub max_seq_length: usize,
    /// Entropy threshold above which a position is considered "high uncertainty".
    pub high_entropy_threshold: f32,
    /// Degradation threshold (current/baseline) to trigger a practice goal.
    pub degradation_threshold: f32,
    /// Minimum observations before generating statistics.
    pub min_observations: u32,
    /// Maximum number of practice goals to maintain.
    pub max_goals: usize,
}

impl Default for PositionalUncertaintyConfig {
    fn default() -> Self {
        Self {
            bin_size: 64,
            max_seq_length: 131072, // 128k
            high_entropy_threshold: 3.0,
            degradation_threshold: 1.5,
            min_observations: 5,
            max_goals: 10,
        }
    }
}

/// Positional uncertainty tracker.
pub struct PositionalUncertaintyTracker {
    config: PositionalUncertaintyConfig,
    /// Per-bin observations: bin_idx → list of entropy values.
    bins: HashMap<usize, Vec<f32>>,
    /// Per-bin first 5 observations (baseline).
    baselines: HashMap<usize, f32>,
    /// Generated practice goals.
    goals: Vec<PracticeGoal>,
    /// Total observations recorded.
    total_observations: u64,
    /// Next sequence ID.
    next_sequence_id: u64,
}

impl PositionalUncertaintyTracker {
    pub fn new(config: PositionalUncertaintyConfig) -> Self {
        Self {
            config,
            bins: HashMap::new(),
            baselines: HashMap::new(),
            goals: Vec::new(),
            total_observations: 0,
            next_sequence_id: 0,
        }
    }

    /// Record per-position entropy observations from a sequence.
    /// `entropies`: one f32 per position in the sequence.
    pub fn observe_sequence(&mut self, entropies: &[f32]) -> u64 {
        let seq_id = self.next_sequence_id;
        self.next_sequence_id += 1;

        for (pos, &entropy) in entropies.iter().enumerate() {
            if pos >= self.config.max_seq_length {
                break;
            }
            let bin = pos / self.config.bin_size;
            self.bins.entry(bin).or_default().push(entropy);
            self.total_observations += 1;

            // Update baseline (mean of first N observations)
            let observations = self.bins.get(&bin).unwrap();
            if observations.len() == self.config.min_observations as usize {
                let baseline: f32 = observations.iter().sum::<f32>() / observations.len() as f32;
                self.baselines.insert(bin, baseline);
            }
        }

        seq_id
    }

    /// Get aggregated statistics for a position bin.
    pub fn get_stats(&self, bin: usize) -> Option<PositionStats> {
        let observations = self.bins.get(&bin)?;
        if observations.len() < self.config.min_observations as usize {
            return None;
        }

        let count = observations.len() as u32;
        let mean_entropy = observations.iter().sum::<f32>() / count as f32;
        let max_entropy = observations.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_entropy = observations.iter().copied().fold(f32::INFINITY, f32::min);

        // Compute trend: compare last third to first third
        let n = observations.len();
        let third = n / 3;
        let first_third_mean: f32 = observations[..third]
            .iter()
            .sum::<f32>() / third.max(1) as f32;
        let last_third_mean: f32 = observations[n - third.max(1)..]
            .iter()
            .sum::<f32>() / third.max(1) as f32;
        let trend = last_third_mean - first_third_mean;

        Some(PositionStats {
            center: bin * self.config.bin_size + self.config.bin_size / 2,
            count,
            mean_entropy,
            max_entropy,
            min_entropy,
            trend,
        })
    }

    /// Generate practice goals for positions with high uncertainty or degradation.
    pub fn generate_goals(&mut self) -> &[PracticeGoal] {
        self.goals.clear();

        let mut candidates: Vec<PracticeGoal> = Vec::new();

        for (&bin, observations) in &self.bins {
            if observations.len() < self.config.min_observations as usize {
                continue;
            }

            let mean_entropy: f32 = observations.iter().sum::<f32>() / observations.len() as f32;
            let baseline = self.baselines.get(&bin).copied().unwrap_or(mean_entropy);

            let degradation = if baseline > 0.0 {
                mean_entropy / baseline
            } else {
                1.0
            };

            let is_high_entropy = mean_entropy > self.config.high_entropy_threshold;
            let is_degraded = degradation > self.config.degradation_threshold;

            if is_high_entropy || is_degraded {
                let pos_start = bin * self.config.bin_size;
                let pos_end = pos_start + self.config.bin_size;

                // Compute trend for this bin
                let n = observations.len();
                let third = n / 3;
                let first_mean: f32 = observations[..third.max(1)]
                    .iter().sum::<f32>() / third.max(1) as f32;
                let last_mean: f32 = observations[n - third.max(1)..]
                    .iter().sum::<f32>() / third.max(1) as f32;
                let _trend = last_mean - first_mean;

                let priority = if is_degraded {
                    degradation * 2.0 // Degrading positions are higher priority
                } else {
                    mean_entropy / self.config.high_entropy_threshold
                };

                candidates.push(PracticeGoal {
                    description: format!(
                        "Practice long-range context at positions {}-{} (entropy={:.2}, degradation={:.1}x)",
                        pos_start, pos_end, mean_entropy, degradation
                    ),
                    position_range: (pos_start, pos_end),
                    current_entropy: mean_entropy,
                    baseline_entropy: baseline,
                    degradation,
                    priority,
                });
            }
        }

        // Sort by priority (highest first), take top-N
        candidates.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
        self.goals = candidates.into_iter().take(self.config.max_goals).collect();

        &self.goals
    }

    /// Number of bins with observations.
    pub fn bin_count(&self) -> usize {
        self.bins.len()
    }

    /// Total observations recorded.
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Current practice goals.
    pub fn goals(&self) -> &[PracticeGoal] {
        &self.goals
    }

    /// Reset all tracking state.
    pub fn reset(&mut self) {
        self.bins.clear();
        self.baselines.clear();
        self.goals.clear();
        self.total_observations = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_short_sequence() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            ..Default::default()
        });

        let entropies = vec![1.0f32, 2.0, 3.0, 4.0]; // 4 positions, 1 bin
        let id = tracker.observe_sequence(&entropies);
        assert_eq!(id, 0);
        assert_eq!(tracker.bin_count(), 1);
        assert_eq!(tracker.total_observations(), 4);
    }

    #[test]
    fn test_observe_multiple_sequences() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            min_observations: 3,
            ..Default::default()
        });

        // 3 sequences, each contributing to the same bin
        for _ in 0..3 {
            tracker.observe_sequence(&[1.0f32, 2.0, 3.0, 4.0]);
        }

        let stats = tracker.get_stats(0).unwrap();
        assert_eq!(stats.count, 3 * 4);
        assert!((stats.mean_entropy - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_stats_insufficient_observations() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            min_observations: 10,
            ..Default::default()
        });

        tracker.observe_sequence(&[1.0f32, 2.0, 3.0, 4.0]);
        assert!(tracker.get_stats(0).is_none());
    }

    #[test]
    fn test_generate_goals_no_degradation() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            min_observations: 3,
            high_entropy_threshold: 5.0,
            degradation_threshold: 2.0,
            ..Default::default()
        });

        // Low entropy, stable observations
        for _ in 0..10 {
            tracker.observe_sequence(&[1.0f32, 1.0, 1.0, 1.0]);
        }

        let goals = tracker.generate_goals();
        assert!(goals.is_empty(), "Should have no goals when entropy is low and stable");
    }

    #[test]
    fn test_generate_goals_high_entropy() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            min_observations: 3,
            high_entropy_threshold: 2.0,
            degradation_threshold: 10.0, // High threshold so only entropy triggers
            ..Default::default()
        });

        // High entropy observations
        for _ in 0..10 {
            tracker.observe_sequence(&[5.0f32, 6.0, 7.0, 8.0]);
        }

        let goals = tracker.generate_goals();
        assert!(!goals.is_empty(), "Should generate goals for high entropy");
        assert!(goals[0].current_entropy > 2.0);
    }

    #[test]
    fn test_generate_goals_degradation() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            min_observations: 3,
            high_entropy_threshold: 100.0, // Very high so only degradation triggers
            degradation_threshold: 1.5,
            ..Default::default()
        });

        // First: low entropy (establishes baseline)
        for _ in 0..5 {
            tracker.observe_sequence(&[1.0f32, 1.0, 1.0, 1.0]);
        }

        // Then: high entropy (causes degradation)
        for _ in 0..5 {
            tracker.observe_sequence(&[5.0f32, 5.0, 5.0, 5.0]);
        }

        let goals = tracker.generate_goals();
        assert!(!goals.is_empty(), "Should detect degradation");
        assert!(goals[0].degradation > 1.5);
    }

    #[test]
    fn test_multi_bin_tracking() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            ..Default::default()
        });

        // 12 positions = 3 bins (0-3, 4-7, 8-11)
        let entropies = vec![1.0f32; 12];
        tracker.observe_sequence(&entropies);
        assert_eq!(tracker.bin_count(), 3);
    }

    #[test]
    fn test_reset() {
        let mut tracker = PositionalUncertaintyTracker::new(PositionalUncertaintyConfig {
            bin_size: 4,
            ..Default::default()
        });

        tracker.observe_sequence(&[1.0f32, 2.0, 3.0, 4.0]);
        assert!(tracker.total_observations() > 0);

        tracker.reset();
        assert_eq!(tracker.bin_count(), 0);
        assert_eq!(tracker.total_observations(), 0);
    }
}
