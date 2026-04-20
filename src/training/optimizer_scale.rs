//! SCALE optimizer (Stochastic Column-normAlized Last-layer momEntum).
//!
//! Svoboda et al. 2025.
//! State per [m×n] matrix: n values (column norms) + optional momentum on output layers.
//! Total state for FerrisRes all-trainable: ~12.1 MB — fits Raspberry Pi.
//!
//! Algorithm:
//!   1. c_j = sqrt(Σ_i g_ij²)           — column norms of gradient
//!   2. g̃_ij = g_ij / c_j               — column-normalized gradient
//!   3. W -= lr · g̃                      — update (no momentum)
//!   4. For output layers only:           — momentum on LoRA B (output adapter)
//!      m = β₁·m + g̃, W -= lr · m

use std::collections::{HashMap, HashSet};

use super::optimizer::WeightOptimizer;

/// Per-matrix optimizer state.
struct ScaleMatrixState {
    rows: usize,
    cols: usize,
    /// Workspace: column norms [cols], recomputed each step.
    column_norms: Vec<f32>,
    /// Momentum [rows * cols], only allocated for output layers.
    momentum: Option<Vec<f32>>,
}

/// SCALE optimizer backend.
pub struct ScaleOptimizer {
    learning_rate: f32,
    beta1: f32,
    epsilon: f32,
    timestep: u32,
    matrices: HashMap<String, ScaleMatrixState>,
    output_layers: HashSet<String>,
}

impl ScaleOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            epsilon: 1e-8,
            timestep: 0,
            matrices: HashMap::new(),
            output_layers: HashSet::new(),
        }
    }
}

impl WeightOptimizer for ScaleOptimizer {
    fn register_matrix(&mut self, name: &str, rows: usize, cols: usize) {
        tracing::debug!(
            event = "scale_register_matrix",
            name, rows, cols,
            "SCALE: registering matrix"
        );
        self.matrices.insert(name.to_string(), ScaleMatrixState {
            rows,
            cols,
            column_norms: vec![0.0; cols],
            momentum: None,
        });
    }

    fn mark_output_layer(&mut self, name: &str) {
        self.output_layers.insert(name.to_string());
        // Allocate momentum buffer if matrix already registered
        if let Some(state) = self.matrices.get_mut(name) {
            if state.momentum.is_none() {
                state.momentum = Some(vec![0.0; state.rows * state.cols]);
            }
        }
    }

    fn step(&mut self, name: &str, gradient: &[f32], weights: &mut [f32]) {
        let state = self.matrices.get_mut(name).unwrap_or_else(|| {
            panic!("SCALE: matrix '{}' not registered", name)
        });

        let rows = state.rows;
        let cols = state.cols;
        assert_eq!(gradient.len(), rows * cols, "SCALE: gradient size mismatch for '{}'", name);
        assert_eq!(weights.len(), rows * cols, "SCALE: weight size mismatch for '{}'", name);

        self.timestep += 1;

        // 1. Compute column norms: c_j = sqrt(Σ_i g_ij² + ε)
        for j in 0..cols {
            let mut sum_sq = 0.0f32;
            for i in 0..rows {
                let g = gradient[i * cols + j];
                sum_sq += g * g;
            }
            state.column_norms[j] = (sum_sq + self.epsilon).sqrt();
        }

        // 2-3. Column-normalized gradient update: W -= lr * g / c
        if state.momentum.is_none() {
            // No momentum: direct update
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    let g_tilde = gradient[idx] / state.column_norms[j];
                    weights[idx] -= self.learning_rate * g_tilde;
                }
            }
        } else {
            // 4. Output layer: apply momentum
            let momentum = state.momentum.as_mut().unwrap();
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    let g_tilde = gradient[idx] / state.column_norms[j];
                    momentum[idx] = self.beta1 * momentum[idx] + (1.0 - self.beta1) * g_tilde;
                    weights[idx] -= self.learning_rate * momentum[idx];
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        // SCALE doesn't accumulate gradients between steps (no grad buffer).
        // This is a no-op — the gradient is consumed immediately in step().
    }

    fn state_bytes(&self) -> usize {
        self.matrices.values().map(|s| {
            let base = s.cols * std::mem::size_of::<f32>(); // column_norms
            let mom = s.momentum.as_ref().map_or(0, |m| m.len() * std::mem::size_of::<f32>());
            base + mom
        }).sum()
    }

    fn num_registered(&self) -> usize {
        self.matrices.len()
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn timestep(&self) -> u32 {
        self.timestep
    }

    fn name(&self) -> &'static str {
        "SCALE"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_basic_update() {
        let mut opt = ScaleOptimizer::new(0.01);
        opt.register_matrix("test", 2, 3);

        // Gradient = [[1, 2, 3], [4, 5, 6]]
        let grad = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut weights = vec![0.0; 6];

        opt.step("test", &grad, &mut weights);

        // Column norms: c = [sqrt(1+16), sqrt(4+25), sqrt(9+36)] = [sqrt(17), sqrt(29), sqrt(45)]
        // g_tilde row 0: [1/sqrt(17), 2/sqrt(29), 3/sqrt(45)]
        // g_tilde row 1: [4/sqrt(17), 5/sqrt(29), 6/sqrt(45)]
        // W -= 0.01 * g_tilde
        let c0 = (17.0f32 + 1e-8).sqrt();
        let c1 = (29.0f32 + 1e-8).sqrt();
        let c2 = (45.0f32 + 1e-8).sqrt();

        let expected = [
            -0.01 * 1.0 / c0, -0.01 * 2.0 / c1, -0.01 * 3.0 / c2,
            -0.01 * 4.0 / c0, -0.01 * 5.0 / c1, -0.01 * 6.0 / c2,
        ];

        for (i, (&w, &e)) in weights.iter().zip(expected.iter()).enumerate() {
            assert!((w - e).abs() < 1e-6, "Mismatch at index {}: got {}, expected {}", i, w, e);
        }
    }

    #[test]
    fn test_scale_output_layer_momentum() {
        let mut opt = ScaleOptimizer::new(0.01);
        opt.register_matrix("lora_b", 4, 3);
        opt.mark_output_layer("lora_b");

        let grad = vec![1.0; 12];
        let mut weights = vec![0.0; 12];

        // First step: momentum = (1-β₁)*g̃ = 0.1*g̃
        opt.step("lora_b", &grad, &mut weights);
        assert!(weights.iter().any(|&w| w != 0.0), "Weights should be updated");

        // Verify momentum is being applied (second step should be larger due to momentum)
        let w_after_1 = weights[0];
        opt.step("lora_b", &grad, &mut weights);
        let w_after_2 = weights[0];
        // Second update is larger because momentum has accumulated
        assert!(w_after_2.abs() > w_after_1.abs(), "Momentum should amplify updates");
    }

    #[test]
    fn test_scale_state_bytes() {
        let mut opt = ScaleOptimizer::new(0.01);
        opt.register_matrix("test", 4, 3);
        // State: column_norms [3] = 12 bytes
        assert_eq!(opt.state_bytes(), 3 * 4);

        opt.mark_output_layer("test");
        // State: column_norms [3] + momentum [12] = 60 bytes
        assert_eq!(opt.state_bytes(), (3 + 12) * 4);
    }

    #[test]
    fn test_scale_name() {
        let opt = ScaleOptimizer::new(0.01);
        assert_eq!(opt.name(), "SCALE");
    }
}
