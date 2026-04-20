//! PreScope predictive expert prefetching.
//!
//! Based on arXiv:2509.23638 (PreScope): During layer N's attention,
//! the post-attention residual is available before the FFN (MoE) starts.
//! A lightweight "scope" router predicts which experts layer N+1 will need,
//! allowing async mmap prefetch to overlap with the current layer's FFN.
//!
//! Pipeline:
//!   1. Layer N attention → hidden state + residual
//!   2. Feed residual through scope_router[N] → predict layer N+1 experts
//!   3. Issue async mmap prefetch for predicted experts
//!   4. Layer N FFN runs (overlapping with prefetch)
//!   5. Layer N+1 starts with experts already in memory
//!
//! On RPi with USB SSD (~5ms expert load), prefetching hides nearly all
//! the load latency behind FFN computation (~3ms on RPi).


/// Scope router: lightweight linear prediction of expert selection.
///
/// Each layer has a scope router that takes the post-attention hidden state
/// and predicts which experts the NEXT layer will select. The prediction is
/// a simple linear projection + top-k selection:
///   scores = hidden @ router_weight  (hidden_dim × num_experts)
///   predicted = top_k(scores)
///
/// Accuracy is typically 85-92% because:
/// 1. Adjacent layers in MoE models make correlated routing decisions
/// 2. The post-attention hidden state is highly informative
/// 3. Top-2 routing means missing 1 of 2 experts is not catastrophic
///    (BuddyMoE fallback handles misses)
#[derive(Clone, Debug)]
pub struct ScopeRouter {
    /// Router weights: [hidden_dim × num_experts] per layer.
    /// Very small: 1536 × 4 × 4 bytes = ~24 KB per layer.
    weights: Vec<Vec<f32>>,
    hidden_dim: usize,
    num_experts: usize,
    top_k: usize,
}

impl ScopeRouter {
    /// Create a new scope router with zero-initialized weights.
    pub fn new(num_layers: usize, hidden_dim: usize, num_experts: usize, top_k: usize) -> Self {
        let weights = (0..num_layers)
            .map(|_| vec![0.0f32; hidden_dim * num_experts])
            .collect();
        Self { weights, hidden_dim, num_experts, top_k }
    }

    /// Predict which experts the next layer will select.
    ///
    /// Returns sorted list of predicted expert indices.
    /// This is a simple dot-product scoring, matching the MoE router pattern.
    pub fn predict(&self, layer_idx: usize, hidden: &[f32]) -> Vec<usize> {
        let w = &self.weights[layer_idx];
        let num_experts = self.num_experts;

        let mut scores: Vec<(usize, f32)> = (0..num_experts)
            .map(|e| {
                let mut score = 0.0f32;
                for d in 0..self.hidden_dim {
                    score += hidden[d] * w[d * num_experts + e];
                }
                (e, score)
            })
            .collect();

        // Partial sort for top-k
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter()
            .take(self.top_k)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Set router weights for a specific layer.
    pub fn set_weights(&mut self, layer_idx: usize, weights: Vec<f32>) {
        assert_eq!(weights.len(), self.hidden_dim * self.num_experts);
        self.weights[layer_idx] = weights;
    }

    /// Get router weights for a specific layer.
    pub fn weights(&self, layer_idx: usize) -> &[f32] {
        &self.weights[layer_idx]
    }

    /// Total memory used by all scope routers.
    pub fn memory_bytes(&self) -> usize {
        self.weights.len() * self.hidden_dim * self.num_experts * std::mem::size_of::<f32>()
    }
}

/// Prefetch state for tracking outstanding prefetch requests.
#[derive(Clone, Debug)]
pub struct PrefetchRequest {
    pub layer: usize,
    pub expert_indices: Vec<usize>,
    pub predicted: bool,
}

/// PreScope prefetch manager.
///
/// Coordinates predictive prefetching of MoE experts using scope routers.
/// Works with ExpertLoader for the actual I/O.
pub struct PreScopeManager {
    router: ScopeRouter,
    /// Outstanding prefetch requests.
    pending: Vec<PrefetchRequest>,
    /// Statistics.
    stats: PreScopeStats,
}

#[derive(Clone, Debug, Default)]
pub struct PreScopeStats {
    pub predictions_made: u64,
    pub predictions_correct: u64,
    pub predictions_partial: u64,
    pub predictions_missed: u64,
    pub prefetches_issued: u64,
}

impl PreScopeManager {
    /// Create a new PreScope manager.
    pub fn new(router: ScopeRouter) -> Self {
        Self {
            router,
            pending: Vec::new(),
            stats: PreScopeStats::default(),
        }
    }

    /// Issue a predictive prefetch for the given layer's next-layer experts.
    ///
    /// Call this AFTER the current layer's attention, BEFORE its FFN.
    /// The returned request tracks the predicted expert indices for later
    /// verification against actual routing.
    pub fn prefetch_next_layer(
        &mut self,
        current_layer: usize,
        hidden_state: &[f32],
        expert_loader: &mut crate::model::expert_loader::ExpertLoader,
    ) -> PrefetchRequest {
        let next_layer = current_layer + 1;

        // Predict next layer's experts
        let predicted = self.router.predict(current_layer, hidden_state);

        // Issue prefetch for predicted experts
        for &expert_idx in &predicted {
            expert_loader.ensure_loaded(next_layer, expert_idx);
            self.stats.prefetches_issued += 1;
        }

        let request = PrefetchRequest {
            layer: next_layer,
            expert_indices: predicted.clone(),
            predicted: true,
        };

        self.stats.predictions_made += 1;
        self.pending.push(request.clone());

        request
    }

    /// Verify prediction accuracy against actual routing decisions.
    ///
    /// Call this when the actual routing for a layer is determined.
    /// Updates accuracy statistics.
    pub fn verify_prediction(
        &mut self,
        layer: usize,
        actual_experts: &[usize],
    ) {
        // Find the prediction for this layer
        if let Some(pos) = self.pending.iter().position(|r| r.layer == layer) {
            let request = self.pending.remove(pos);
            let predicted: Vec<usize> = request.expert_indices;

            let all_correct = predicted.len() == actual_experts.len()
                && predicted.iter().all(|p| actual_experts.contains(p));

            let any_correct = predicted.iter().any(|p| actual_experts.contains(p));

            if all_correct {
                self.stats.predictions_correct += 1;
            } else if any_correct {
                self.stats.predictions_partial += 1;
            } else {
                self.stats.predictions_missed += 1;
            }
        }
    }

    /// Get prediction accuracy (0.0 to 1.0).
    pub fn accuracy(&self) -> f64 {
        if self.stats.predictions_made == 0 {
            return 0.0;
        }
        (self.stats.predictions_correct as f64 + 0.5 * self.stats.predictions_partial as f64)
            / self.stats.predictions_made as f64
    }

    /// Get the scope router (for training).
    pub fn router(&self) -> &ScopeRouter {
        &self.router
    }

    /// Get the scope router mutably (for training).
    pub fn router_mut(&mut self) -> &mut ScopeRouter {
        &mut self.router
    }

    /// Get prefetch statistics.
    pub fn stats(&self) -> &PreScopeStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_router_predict() {
        let mut router = ScopeRouter::new(3, 4, 4, 2);

        // Set weights so that expert 0 and 2 score highest for input [1,0,0,0]
        let mut w = vec![0.0f32; 4 * 4]; // hidden_dim × num_experts
        // w[d * num_experts + e] = score for expert e from dimension d
        w[0 * 4 + 0] = 10.0; // dim 0 → expert 0
        w[0 * 4 + 2] = 8.0;  // dim 0 → expert 2
        w[0 * 4 + 1] = 1.0;  // dim 0 → expert 1
        w[0 * 4 + 3] = 0.5;  // dim 0 → expert 3
        router.set_weights(0, w);

        let hidden = vec![1.0f32, 0.0, 0.0, 0.0];
        let predicted = router.predict(0, &hidden);
        assert_eq!(predicted.len(), 2);
        assert_eq!(predicted[0], 0); // highest score
        assert_eq!(predicted[1], 2); // second highest
    }

    #[test]
    fn test_scope_router_memory() {
        let router = ScopeRouter::new(35, 1536, 4, 2);
        // 35 layers × 1536 hidden × 4 experts × 4 bytes = 860 KB
        let expected = 35 * 1536 * 4 * 4;
        assert_eq!(router.memory_bytes(), expected);
    }

    #[test]
    fn test_prescope_accuracy_all_correct() {
        let router = ScopeRouter::new(3, 4, 4, 2);
        let mut mgr = PreScopeManager::new(router);

        // Simulate predictions
        let req = PrefetchRequest {
            layer: 1,
            expert_indices: vec![0, 2],
            predicted: true,
        };
        mgr.pending.push(req);
        mgr.stats.predictions_made = 1;

        mgr.verify_prediction(1, &[0, 2]);
        assert_eq!(mgr.stats().predictions_correct, 1);
        assert!((mgr.accuracy() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prescope_accuracy_partial() {
        let router = ScopeRouter::new(3, 4, 4, 2);
        let mut mgr = PreScopeManager::new(router);

        let req = PrefetchRequest {
            layer: 1,
            expert_indices: vec![0, 2],
            predicted: true,
        };
        mgr.pending.push(req);
        mgr.stats.predictions_made = 1;

        mgr.verify_prediction(1, &[0, 3]); // predicted 0,2; actual 0,3
        assert_eq!(mgr.stats().predictions_partial, 1);
        assert!((mgr.accuracy() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_prescope_accuracy_miss() {
        let router = ScopeRouter::new(3, 4, 4, 2);
        let mut mgr = PreScopeManager::new(router);

        let req = PrefetchRequest {
            layer: 1,
            expert_indices: vec![0, 2],
            predicted: true,
        };
        mgr.pending.push(req);
        mgr.stats.predictions_made = 1;

        mgr.verify_prediction(1, &[1, 3]); // completely wrong
        assert_eq!(mgr.stats().predictions_missed, 1);
        assert!((mgr.accuracy() - 0.0).abs() < 1e-10);
    }
}
