//! Partial backpropagation for memory-efficient fine-tuning.
//!
//! Allows selective gradient computation for specific layers while freezing
//! the rest, enabling fine-tuning of large models on memory-constrained GPUs.
//!
//! Features:
//! - Layer freeze: mark layers as frozen (skip gradient + activation storage)
//! - Selective backward: only compute gradients for unfrozen layers
//! - LoRA integration: only train adapter weights while base model is frozen
//! - Memory savings: skip activation storage for frozen layers

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// LayerId — identifies a layer in the model
// ---------------------------------------------------------------------------

/// Unique identifier for a model layer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LayerId {
    /// Layer group (e.g., "encoder", "decoder").
    pub group: String,
    /// Layer index within the group.
    pub index: usize,
}

impl LayerId {
    pub fn new(group: &str, index: usize) -> Self {
        Self { group: group.to_string(), index }
    }

    /// Parse from string like "encoder.3" or "layer-3".
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.rsplitn(2, '.').collect();
        if parts.len() == 2 {
            if let Ok(idx) = parts[0].parse::<usize>() {
                return Some(Self { group: parts[1].to_string(), index: idx });
            }
        }
        // Try "layer-N" format
        if let Some(n) = s.strip_prefix("layer-") {
            if let Ok(idx) = n.parse::<usize>() {
                return Some(Self { group: "default".to_string(), index: idx });
            }
        }
        None
    }
}

impl std::fmt::Display for LayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.group, self.index)
    }
}

// ---------------------------------------------------------------------------
// FreezePolicy — determines which layers to freeze
// ---------------------------------------------------------------------------

/// Policy for which layers to freeze during training.
#[derive(Debug, Clone)]
pub enum FreezePolicy {
    /// Freeze all layers except the last N.
    LastNUnfrozen { n: usize },
    /// Freeze specific layer indices.
    FrozenIndices { indices: HashSet<usize> },
    /// Freeze a range of layers [start, end).
    FrozenRange { start: usize, end: usize },
    /// Custom per-group freeze decisions.
    Custom { frozen: HashMap<String, HashSet<usize>> },
    /// Freeze all (inference only).
    AllFrozen,
    /// None frozen (full training).
    NoneFrozen,
}

impl FreezePolicy {
    /// Parse CLI-style freeze specification: "0-10" means freeze layers 0 through 9.
    pub fn from_range_spec(spec: &str) -> Option<Self> {
        let parts: Vec<&str> = spec.split('-').collect();
        if parts.len() == 2 {
            let start = parts[0].parse::<usize>().ok()?;
            let end = parts[1].parse::<usize>().ok()?;
            Some(Self::FrozenRange { start, end })
        } else {
            None
        }
    }

    /// Whether a layer should be frozen.
    pub fn is_frozen(&self, layer: &LayerId) -> bool {
        match self {
            FreezePolicy::AllFrozen => true,
            FreezePolicy::NoneFrozen => false,
            FreezePolicy::LastNUnfrozen { n: _ } => {
                // This requires knowing total layers, so use index-based check
                // In practice, the caller provides the total
                false // Simplified; use is_frozen_with_total instead
            }
            FreezePolicy::FrozenIndices { indices } => indices.contains(&layer.index),
            FreezePolicy::FrozenRange { start, end } => {
                layer.index >= *start && layer.index < *end
            }
            FreezePolicy::Custom { frozen } => {
                frozen.get(&layer.group).map_or(false, |indices| indices.contains(&layer.index))
            }
        }
    }

    /// Whether a layer should be frozen, given total layer count.
    pub fn is_frozen_with_total(&self, layer: &LayerId, total_layers: usize) -> bool {
        match self {
            FreezePolicy::LastNUnfrozen { n } => {
                layer.index < total_layers.saturating_sub(*n)
            }
            _ => self.is_frozen(layer),
        }
    }
}

// ---------------------------------------------------------------------------
// PartialBackpropConfig — full configuration
// ---------------------------------------------------------------------------

/// Configuration for partial backpropagation.
#[derive(Debug, Clone)]
pub struct PartialBackpropConfig {
    /// Freeze policy.
    pub policy: FreezePolicy,
    /// Whether to skip activation storage for frozen layers.
    pub skip_frozen_activations: bool,
    /// Whether LoRA is enabled (only train adapters).
    pub lora_enabled: bool,
    /// Total number of layers in the model.
    pub total_layers: usize,
}

impl PartialBackpropConfig {
    pub fn new(policy: FreezePolicy, total_layers: usize) -> Self {
        Self {
            policy,
            skip_frozen_activations: true,
            lora_enabled: false,
            total_layers,
        }
    }

    /// Full training (nothing frozen).
    pub fn full_training(total_layers: usize) -> Self {
        Self::new(FreezePolicy::NoneFrozen, total_layers)
    }

    /// Freeze all (inference mode).
    pub fn inference(total_layers: usize) -> Self {
        Self::new(FreezePolicy::AllFrozen, total_layers)
    }

    /// Freeze bottom N layers, train top layers.
    pub fn freeze_bottom(n: usize, total_layers: usize) -> Self {
        Self::new(FreezePolicy::FrozenRange { start: 0, end: n }, total_layers)
    }

    /// Only last N layers trainable.
    pub fn train_last_n(n: usize, total_layers: usize) -> Self {
        Self::new(FreezePolicy::LastNUnfrozen { n }, total_layers)
    }

    /// With LoRA enabled.
    pub fn with_lora(mut self) -> Self {
        self.lora_enabled = true;
        self
    }

    /// Whether a layer is frozen.
    pub fn is_frozen(&self, layer: &LayerId) -> bool {
        self.policy.is_frozen_with_total(layer, self.total_layers)
    }

    /// Whether to compute gradients for a layer.
    pub fn compute_gradients(&self, layer: &LayerId) -> bool {
        if self.lora_enabled {
            // With LoRA, base weights are always frozen, only adapters train
            // But we still need forward pass through all layers
            !self.is_frozen(layer)
        } else {
            !self.is_frozen(layer)
        }
    }

    /// Whether to store activations for a layer (needed for backward pass).
    pub fn store_activations(&self, layer: &LayerId) -> bool {
        if self.skip_frozen_activations && self.is_frozen(layer) {
            false
        } else {
            true
        }
    }

    /// List of frozen layer indices.
    pub fn frozen_layers(&self) -> Vec<usize> {
        (0..self.total_layers)
            .filter(|&i| self.is_frozen(&LayerId::new("default", i)))
            .collect()
    }

    /// List of trainable layer indices.
    pub fn trainable_layers(&self) -> Vec<usize> {
        (0..self.total_layers)
            .filter(|&i| !self.is_frozen(&LayerId::new("default", i)))
            .collect()
    }

    /// Fraction of layers that are frozen.
    pub fn freeze_fraction(&self) -> f32 {
        if self.total_layers == 0 { return 0.0; }
        self.frozen_layers().len() as f32 / self.total_layers as f32
    }

    /// Estimated memory savings ratio vs full training.
    pub fn memory_savings(&self) -> f32 {
        if self.skip_frozen_activations {
            self.freeze_fraction()
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// LayerTrainingState — tracks per-layer state during training
// ---------------------------------------------------------------------------

/// Training state for a single layer.
#[derive(Debug, Clone, PartialEq)]
pub enum LayerTrainingState {
    /// Layer is frozen — no gradient computation.
    Frozen,
    /// Layer is trainable — full gradient computation.
    Trainable,
    /// Layer has LoRA adapters — only adapter gradients.
    LoraOnly,
}

/// Tracks training state for all layers.
pub struct LayerStateTracker {
    config: PartialBackpropConfig,
    states: Vec<LayerTrainingState>,
}

impl LayerStateTracker {
    pub fn new(config: PartialBackpropConfig) -> Self {
        let states: Vec<LayerTrainingState> = (0..config.total_layers)
            .map(|i| {
                let layer = LayerId::new("default", i);
                if config.lora_enabled {
                    LayerTrainingState::LoraOnly
                } else if config.is_frozen(&layer) {
                    LayerTrainingState::Frozen
                } else {
                    LayerTrainingState::Trainable
                }
            })
            .collect();
        Self { config, states }
    }

    /// Get the training state for a layer.
    pub fn state(&self, index: usize) -> &LayerTrainingState {
        self.states.get(index).unwrap_or(&LayerTrainingState::Frozen)
    }

    /// Whether to skip activation checkpoint for this layer.
    pub fn skip_checkpoint(&self, index: usize) -> bool {
        matches!(self.state(index), LayerTrainingState::Frozen)
            && self.config.skip_frozen_activations
    }

    /// Whether to compute gradients for this layer.
    pub fn needs_gradients(&self, index: usize) -> bool {
        !matches!(self.state(index), LayerTrainingState::Frozen)
    }

    /// Number of frozen layers.
    pub fn frozen_count(&self) -> usize {
        self.states.iter().filter(|s| matches!(s, LayerTrainingState::Frozen)).count()
    }

    /// Number of trainable layers.
    pub fn trainable_count(&self) -> usize {
        self.states.iter().filter(|s| !matches!(s, LayerTrainingState::Frozen)).count()
    }

    /// Get the config.
    pub fn config(&self) -> &PartialBackpropConfig {
        &self.config
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.states.len()
    }

    /// Dynamically unfreeze a layer (for gradual unfreezing).
    pub fn unfreeze(&mut self, index: usize) {
        if index < self.states.len() {
            if matches!(self.states[index], LayerTrainingState::Frozen) {
                self.states[index] = if self.config.lora_enabled {
                    LayerTrainingState::LoraOnly
                } else {
                    LayerTrainingState::Trainable
                };
            }
        }
    }

    /// Freeze a layer.
    pub fn freeze(&mut self, index: usize) {
        if index < self.states.len() {
            self.states[index] = LayerTrainingState::Frozen;
        }
    }

    /// Apply gradual unfreezing: unfreeze the next N layers from the top.
    pub fn unfreeze_next_n(&mut self, n: usize) {
        let mut unfrozen = 0;
        for i in (0..self.states.len()).rev() {
            if unfrozen >= n { break; }
            if matches!(self.states[i], LayerTrainingState::Frozen) {
                self.unfreeze(i);
                unfrozen += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SelectiveBackward — simulates the backward pass selection
// ---------------------------------------------------------------------------

/// Simulated backward pass that only processes unfrozen layers.
pub struct SelectiveBackward {
    tracker: LayerStateTracker,
}

impl SelectiveBackward {
    pub fn new(tracker: LayerStateTracker) -> Self {
        Self { tracker }
    }

    /// Run selective backward: only compute gradients for trainable layers.
    /// Returns the indices of layers where gradients were computed.
    pub fn backward(&self) -> Vec<usize> {
        (0..self.tracker.num_layers())
            .filter(|&i| self.tracker.needs_gradients(i))
            .collect()
    }

    /// Run selective backward in reverse order (typical for backprop).
    pub fn backward_reverse(&self) -> Vec<usize> {
        let mut layers = self.backward();
        layers.reverse();
        layers
    }

    /// Get the tracker.
    pub fn tracker(&self) -> &LayerStateTracker {
        &self.tracker
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_id() {
        let id = LayerId::new("encoder", 3);
        assert_eq!(id.group, "encoder");
        assert_eq!(id.index, 3);
        assert_eq!(id.to_string(), "encoder.3");
    }

    #[test]
    fn test_layer_id_parse() {
        let id = LayerId::parse("encoder.3").unwrap();
        assert_eq!(id.group, "encoder");
        assert_eq!(id.index, 3);

        let id2 = LayerId::parse("layer-5").unwrap();
        assert_eq!(id2.group, "default");
        assert_eq!(id2.index, 5);

        assert!(LayerId::parse("invalid").is_none());
    }

    #[test]
    fn test_freeze_policy_none() {
        let policy = FreezePolicy::NoneFrozen;
        assert!(!policy.is_frozen(&LayerId::new("default", 0)));
        assert!(!policy.is_frozen(&LayerId::new("default", 99)));
    }

    #[test]
    fn test_freeze_policy_all() {
        let policy = FreezePolicy::AllFrozen;
        assert!(policy.is_frozen(&LayerId::new("default", 0)));
    }

    #[test]
    fn test_freeze_policy_range() {
        let policy = FreezePolicy::FrozenRange { start: 0, end: 10 };
        assert!(policy.is_frozen(&LayerId::new("default", 5)));
        assert!(!policy.is_frozen(&LayerId::new("default", 10)));
        assert!(!policy.is_frozen(&LayerId::new("default", 15)));
    }

    #[test]
    fn test_freeze_policy_last_n() {
        let policy = FreezePolicy::LastNUnfrozen { n: 2 };
        assert!(policy.is_frozen_with_total(&LayerId::new("default", 5), 12));
        assert!(!policy.is_frozen_with_total(&LayerId::new("default", 11), 12));
    }

    #[test]
    fn test_freeze_policy_custom() {
        let mut frozen = HashMap::new();
        frozen.insert("encoder".to_string(), {
            let mut s = HashSet::new();
            s.insert(0);
            s.insert(1);
            s
        });
        let policy = FreezePolicy::Custom { frozen };
        assert!(policy.is_frozen(&LayerId::new("encoder", 0)));
        assert!(!policy.is_frozen(&LayerId::new("encoder", 2)));
        assert!(!policy.is_frozen(&LayerId::new("decoder", 0)));
    }

    #[test]
    fn test_freeze_from_range_spec() {
        let policy = FreezePolicy::from_range_spec("0-10").unwrap();
        match policy {
            FreezePolicy::FrozenRange { start, end } => {
                assert_eq!(start, 0);
                assert_eq!(end, 10);
            }
            _ => panic!("Expected FrozenRange"),
        }
    }

    #[test]
    fn test_partial_config_full_training() {
        let config = PartialBackpropConfig::full_training(12);
        assert!(config.trainable_layers().len() == 12);
        assert!(config.frozen_layers().is_empty());
        assert!((config.freeze_fraction() - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_partial_config_inference() {
        let config = PartialBackpropConfig::inference(12);
        assert!(config.frozen_layers().len() == 12);
        assert!(config.trainable_layers().is_empty());
    }

    #[test]
    fn test_partial_config_freeze_bottom() {
        let config = PartialBackpropConfig::freeze_bottom(8, 12);
        assert_eq!(config.frozen_layers(), vec![0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(config.trainable_layers(), vec![8, 9, 10, 11]);
        assert!((config.freeze_fraction() - 8.0 / 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_partial_config_train_last_n() {
        let config = PartialBackpropConfig::train_last_n(2, 12);
        assert_eq!(config.trainable_layers(), vec![10, 11]);
        assert_eq!(config.frozen_layers().len(), 10);
    }

    #[test]
    fn test_partial_config_lora() {
        let config = PartialBackpropConfig::freeze_bottom(10, 12).with_lora();
        assert!(config.lora_enabled);
    }

    #[test]
    fn test_partial_config_store_activations() {
        let config = PartialBackpropConfig::freeze_bottom(8, 12);
        // Frozen layers should not store activations
        assert!(!config.store_activations(&LayerId::new("default", 3)));
        // Trainable layers should
        assert!(config.store_activations(&LayerId::new("default", 9)));
    }

    #[test]
    fn test_memory_savings() {
        let config = PartialBackpropConfig::freeze_bottom(8, 12);
        let savings = config.memory_savings();
        assert!((savings - 8.0 / 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_state_tracker() {
        let config = PartialBackpropConfig::freeze_bottom(8, 12);
        let tracker = LayerStateTracker::new(config);
        assert_eq!(tracker.frozen_count(), 8);
        assert_eq!(tracker.trainable_count(), 4);
        assert_eq!(*tracker.state(3), LayerTrainingState::Frozen);
        assert_eq!(*tracker.state(9), LayerTrainingState::Trainable);
        assert!(!tracker.needs_gradients(3));
        assert!(tracker.needs_gradients(9));
    }

    #[test]
    fn test_state_tracker_lora() {
        let config = PartialBackpropConfig::freeze_bottom(8, 12).with_lora();
        let tracker = LayerStateTracker::new(config);
        assert_eq!(*tracker.state(3), LayerTrainingState::LoraOnly);
        assert_eq!(*tracker.state(9), LayerTrainingState::LoraOnly);
        assert!(tracker.needs_gradients(3)); // LoRA still needs forward
    }

    #[test]
    fn test_state_tracker_unfreeze() {
        let config = PartialBackpropConfig::freeze_bottom(10, 12);
        let mut tracker = LayerStateTracker::new(config);
        assert_eq!(tracker.frozen_count(), 10);

        tracker.unfreeze(8);
        assert_eq!(tracker.frozen_count(), 9);
        assert_eq!(*tracker.state(8), LayerTrainingState::Trainable);
    }

    #[test]
    fn test_state_tracker_freeze() {
        let config = PartialBackpropConfig::full_training(12);
        let mut tracker = LayerStateTracker::new(config);
        assert_eq!(tracker.frozen_count(), 0);

        tracker.freeze(5);
        assert_eq!(tracker.frozen_count(), 1);
    }

    #[test]
    fn test_gradual_unfreezing() {
        let config = PartialBackpropConfig::new(FreezePolicy::AllFrozen, 12);
        let mut tracker = LayerStateTracker::new(config);

        assert_eq!(tracker.frozen_count(), 12);

        // Unfreeze top 2
        tracker.unfreeze_next_n(2);
        assert_eq!(tracker.frozen_count(), 10);
        assert_eq!(*tracker.state(11), LayerTrainingState::Trainable);
        assert_eq!(*tracker.state(10), LayerTrainingState::Trainable);

        // Unfreeze 2 more
        tracker.unfreeze_next_n(2);
        assert_eq!(tracker.frozen_count(), 8);
    }

    #[test]
    fn test_selective_backward() {
        let config = PartialBackpropConfig::freeze_bottom(8, 12);
        let tracker = LayerStateTracker::new(config);
        let backward = SelectiveBackward::new(tracker);

        let layers = backward.backward();
        assert_eq!(layers, vec![8, 9, 10, 11]);

        let rev = backward.backward_reverse();
        assert_eq!(rev, vec![11, 10, 9, 8]);
    }
}
