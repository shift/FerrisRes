//! BuddyMoE: Expert redundancy for prefetch miss resilience.
//!
//! Based on arXiv:2511.10054 (BuddyMoE): When predictive prefetch misses
//! (PreScope predicted wrong experts), BuddyMoE substitutes a "buddy" expert
//! that IS resident in memory. This eliminates synchronous I/O stalls on
//! edge devices with slow storage (USB SSD, SD card).
//!
//! Key insight: In top-k MoE routing (k=2), experts within the same layer
//! develop correlated activation patterns. Expert A and expert B are "buddies"
//! if they frequently co-activate on similar inputs. When the prefetch
//! correctly loaded expert A but missed expert B, using A's output twice
//! (A + A instead of A + B) loses only ~2-5% accuracy.
//!
//! Calibration:
//!   1. Run N tokens through the model, record expert activations per layer
//!   2. Compute co-activation matrix: coact[i][j] = times expert_i and expert_j both activated
//!   3. For each expert, its buddy is the expert with highest co-activation
//!   4. Store buddy map: layer → expert → buddy_expert
//!
//! Runtime:
//!   - When PreScope prefetch misses expert B but got expert A
//!   - Look up A's buddy for this layer
//!   - If buddy(A) is already loaded, substitute B with buddy(A)
//!   - If buddy(A) is also not loaded, fall back to synchronous load

/// Buddy map for a single layer: expert_idx → buddy_expert_idx.
pub type LayerBuddyMap = Vec<usize>;

/// BuddyMoE configuration and buddy maps for all layers.
#[derive(Clone, Debug)]
pub struct BuddyMoE {
    /// Per-layer buddy maps: layer → expert → buddy.
    buddy_maps: Vec<LayerBuddyMap>,

    /// Number of experts per layer.
    num_experts: usize,

    /// Whether buddy substitution is enabled.
    enabled: bool,

    /// Statistics.
    stats: BuddyStats,
}

#[derive(Clone, Debug, Default)]
pub struct BuddyStats {
    /// Number of buddy substitutions performed.
    pub substitutions: u64,
    /// Number of times no buddy was available (fell back to sync load).
    pub fallback_loads: u64,
    /// Total tokens processed.
    pub tokens_processed: u64,
}

impl BuddyMoE {
    /// Create a new BuddyMoE system with no buddy maps.
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        Self {
            buddy_maps: vec![vec![]; num_layers],
            num_experts,
            enabled: true,
            stats: BuddyStats::default(),
        }
    }

    /// Enable or disable buddy substitution.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether buddy substitution is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the buddy for a given expert in a layer.
    ///
    /// Returns None if no buddy map is calibrated for this layer,
    /// or if the buddy is the same expert (no valid buddy found).
    pub fn get_buddy(&self, layer: usize, expert: usize) -> Option<usize> {
        if !self.enabled {
            return None;
        }
        let map = self.buddy_maps.get(layer)?;
        let buddy = *map.get(expert)?;
        if buddy == expert {
            None // No valid buddy (expert's best buddy is itself)
        } else {
            Some(buddy)
        }
    }

    /// Attempt buddy substitution for missing experts.
    ///
    /// Given a list of experts that were requested but are NOT loaded,
    /// and a set of experts that ARE loaded, returns a substitution map:
    ///   missing_expert → substitute_expert (either buddy or None)
    ///
    /// # Arguments
    /// * `layer` — layer index
    /// * `missing` — expert indices that are NOT loaded
    /// * `loaded` — expert indices that ARE loaded
    ///
    /// # Returns
    /// Vector of (missing_expert, Option<substitute>) pairs.
    pub fn substitute(
        &mut self,
        layer: usize,
        missing: &[usize],
        loaded: &[usize],
    ) -> Vec<(usize, Option<usize>)> {
        if !self.enabled {
            return missing.iter().map(|&e| (e, None)).collect();
        }

        let loaded_set: std::collections::HashSet<usize> = loaded.iter().copied().collect();

        missing.iter().map(|&miss_expert| {
            // Try buddy substitution
            if let Some(buddy) = self.get_buddy(layer, miss_expert) {
                if loaded_set.contains(&buddy) {
                    self.stats.substitutions += 1;
                    return (miss_expert, Some(buddy));
                }
            }

            // No buddy available — need sync load
            self.stats.fallback_loads += 1;
            (miss_expert, None)
        }).collect()
    }

    /// Set buddy map for a layer (from calibration).
    pub fn set_buddy_map(&mut self, layer: usize, map: LayerBuddyMap) {
        assert_eq!(map.len(), self.num_experts, "buddy map must have num_experts entries");
        self.buddy_maps[layer] = map;
    }

    /// Whether buddy maps are calibrated for all layers.
    pub fn is_calibrated(&self) -> bool {
        self.buddy_maps.iter().all(|m| !m.is_empty())
    }

    /// Get statistics.
    pub fn stats(&self) -> &BuddyStats {
        &self.stats
    }
}

/// Calibrate buddy maps from expert activation records.
///
/// # Arguments
/// * `activations` — Per-layer activation counts: layer → expert → count
/// * `co_activations` — Per-layer co-activation matrix: layer → (i,j) → count
///   (symmetric, only upper triangle needed)
/// * `num_experts` — Number of experts per layer
///
/// # Returns
/// Vec of buddy maps (one per layer).
pub fn calibrate_buddy_maps(
    num_layers: usize,
    num_experts: usize,
    co_activations: &[Vec<f32>], // [layer][i * num_experts + j]
) -> Vec<LayerBuddyMap> {
    let mut maps = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let coact = &co_activations[layer];
        let mut buddy_map = vec![0usize; num_experts];

        for expert in 0..num_experts {
            let mut best_buddy = expert; // default to self
            let mut best_score = 0.0f32;

            for other in 0..num_experts {
                if other == expert {
                    continue;
                }
                let score = coact[expert * num_experts + other];
                if score > best_score {
                    best_score = score;
                    best_buddy = other;
                }
            }

            buddy_map[expert] = best_buddy;
        }

        maps.push(buddy_map);
    }

    maps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buddy_moe_basic() {
        let mut buddy = BuddyMoE::new(2, 4);
        // Set buddy map for layer 0: expert 0→2, 1→3, 2→0, 3→1
        buddy.set_buddy_map(0, vec![2, 3, 0, 1]);

        assert_eq!(buddy.get_buddy(0, 0), Some(2));
        assert_eq!(buddy.get_buddy(0, 1), Some(3));
        assert_eq!(buddy.get_buddy(0, 2), Some(0));
        assert_eq!(buddy.get_buddy(0, 3), Some(1));
    }

    #[test]
    fn test_buddy_moe_no_map() {
        let buddy = BuddyMoE::new(2, 4);
        // No buddy map set for layer 0
        assert_eq!(buddy.get_buddy(0, 0), None);
    }

    #[test]
    fn test_buddy_moe_self_buddy() {
        let mut buddy = BuddyMoE::new(1, 4);
        // Expert 0's buddy is itself (no co-activation with others)
        buddy.set_buddy_map(0, vec![0, 2, 1, 1]);

        assert_eq!(buddy.get_buddy(0, 0), None); // self = no buddy
        assert_eq!(buddy.get_buddy(0, 1), Some(2));
    }

    #[test]
    fn test_buddy_substitute() {
        let mut buddy = BuddyMoE::new(1, 4);
        buddy.set_buddy_map(0, vec![2, 3, 0, 1]);

        // Missing experts [1, 3], loaded experts [0, 2]
        let result = buddy.substitute(0, &[1, 3], &[0, 2]);

        // Expert 1's buddy is 3 (not loaded) → no substitute
        // Expert 3's buddy is 1 (not loaded) → no substitute
        assert_eq!(result[0], (1, None)); // buddy 3 not loaded
        assert_eq!(result[1], (3, None)); // buddy 1 not loaded
        assert_eq!(buddy.stats().fallback_loads, 2);
    }

    #[test]
    fn test_buddy_substitute_hit() {
        let mut buddy = BuddyMoE::new(1, 4);
        buddy.set_buddy_map(0, vec![2, 3, 0, 1]);

        // Missing experts [0], loaded experts [1, 2]
        let result = buddy.substitute(0, &[0], &[1, 2]);

        // Expert 0's buddy is 2 (loaded!) → substitute
        assert_eq!(result[0], (0, Some(2)));
        assert_eq!(buddy.stats().substitutions, 1);
    }

    #[test]
    fn test_buddy_disabled() {
        let mut buddy = BuddyMoE::new(1, 4);
        buddy.set_buddy_map(0, vec![2, 3, 0, 1]);
        buddy.set_enabled(false);

        assert_eq!(buddy.get_buddy(0, 0), None);
        assert!(buddy.substitute(0, &[0], &[2]).iter().all(|(_, s)| s.is_none()));
    }

    #[test]
    fn test_calibrate_buddy_maps() {
        let num_experts = 4;
        // Co-activation matrix for 1 layer:
        // Expert 0 co-activates most with expert 2
        // Expert 1 co-activates most with expert 3
        let coact = vec![
            // [i * 4 + j] layout
            vec![
                0.0, 0.1, 0.9, 0.2,  // expert 0: best buddy = 2 (0.9)
                0.1, 0.0, 0.3, 0.8,  // expert 1: best buddy = 3 (0.8)
                0.9, 0.3, 0.0, 0.1,  // expert 2: best buddy = 0 (0.9)
                0.2, 0.8, 0.1, 0.0,  // expert 3: best buddy = 1 (0.8)
            ],
        ];

        let maps = calibrate_buddy_maps(1, num_experts, &coact);
        assert_eq!(maps[0][0], 2);
        assert_eq!(maps[0][1], 3);
        assert_eq!(maps[0][2], 0);
        assert_eq!(maps[0][3], 1);
    }
}
