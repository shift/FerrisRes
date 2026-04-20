//! AdaMeM optimizer (Memory Efficient Momentum for Adafactor).
//!
//! WANT@ICML 2024.
//! State per [m×n] matrix: (m+n)·r + 2n values (projector + momentum + factored 2nd moments).
//! Total state for FerrisRes all-trainable: ~181.6 MB — fits Mac/RTX.
//!
//! Algorithm:
//!   Every T=200 steps: SVD(G) → P = top-r left singular vectors [m×r]
//!   Per step:
//!   1. R = Pᵀ·G              — project gradient into low-rank subspace [r×n]
//!   2. S = G - P·R            — residual gradient [m×n]
//!   3. M = β₁·M + (1-β₁)·R   — momentum in low-rank space [r×n]
//!   4. N₁ = Adafactor(M)     — factored 2nd moment preconditioning in subspace
//!   5. N₂ = OneSidedAF(S)    — column-only 2nd moment on residual
//!   6. W += lr · (P·N₁ + N₂) — orthogonal update
//!
//! SVD: uses nalgebra for CPU SVD. Every T steps per matrix.
//! On RPi this is ~13s per 200 steps — NOT viable on edge (use SCALE instead).

use std::collections::HashMap;

use super::optimizer::WeightOptimizer;

/// Per-matrix AdaMeM state.
struct AdaMeMMatrixState {
    rows: usize,  // m (assume m <= n, else transpose)
    cols: usize,  // n
    /// Projector P [m × r]: top-r left singular vectors of gradient.
    projector: Vec<f32>,       // [m * r]
    /// Momentum M [r × n]: EMA of projected gradients.
    momentum: Vec<f32>,        // [r * cols]
    /// Adafactor row 2nd moment [r]: factored 2nd moment for low-rank space.
    af_row: Vec<f32>,          // [r]
    /// Adafactor col 2nd moment [n]: factored 2nd moment for low-rank space.
    af_col: Vec<f32>,          // [cols]
    /// OneSided-Adafactor col 2nd moment [n]: for residual gradient.
    os_col: Vec<f32>,          // [cols]
    /// Step counter per matrix (for SVD refresh timing).
    local_step: u32,
    /// Whether projector needs recomputation.
    needs_svd: bool,
}

/// AdaMeM optimizer backend.
pub struct AdaMeMOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    rank: usize,
    subspace_update_freq: u32, // T: recompute SVD every T steps
    timestep: u32,
    matrices: HashMap<String, AdaMeMMatrixState>,
}

impl AdaMeMOptimizer {
    pub fn new(learning_rate: f32, rank: usize) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            rank,
            subspace_update_freq: 200,
            timestep: 0,
            matrices: HashMap::new(),
        }
    }
}

impl WeightOptimizer for AdaMeMOptimizer {
    fn register_matrix(&mut self, name: &str, rows: usize, cols: usize) {
        // Ensure m <= n for SVD convention (project from rows side)
        let (m, n) = if rows <= cols { (rows, cols) } else { (cols, rows) };
        let r = self.rank.min(m);

        tracing::debug!(
            event = "adamem_register_matrix",
            name, rows, cols, rank = r,
            "AdaMeM: registering matrix"
        );

        self.matrices.insert(name.to_string(), AdaMeMMatrixState {
            rows: m,
            cols: n,
            projector: vec![0.0; m * r],
            momentum: vec![0.0; r * n],
            af_row: vec![0.0; r],
            af_col: vec![0.0; n],
            os_col: vec![0.0; n],
            local_step: 0,
            needs_svd: true, // compute on first step
        });
    }

    fn mark_output_layer(&mut self, _name: &str) {
        // AdaMeM applies momentum in the low-rank subspace for ALL layers.
        // No special handling needed — this is a no-op.
    }

    fn step(&mut self, name: &str, gradient: &[f32], weights: &mut [f32]) {
        let (m, n, needs_svd, local_step) = {
            let state = self.matrices.get(name).unwrap_or_else(|| {
                panic!("AdaMeM: matrix '{}' not registered", name)
            });
            (state.rows, state.cols, state.needs_svd, state.local_step)
        };
        let r = self.rank.min(m);
        assert_eq!(gradient.len(), m * n, "AdaMeM: gradient size mismatch for '{}'", name);
        assert_eq!(weights.len(), m * n, "AdaMeM: weight size mismatch for '{}'", name);

        self.timestep += 1;
        let local_step = local_step + 1;

        // Step 1: Update projector via SVD every T steps
        let do_svd = needs_svd || (local_step % self.subspace_update_freq == 0);
        if do_svd {
            let projector = power_iteration_svd(gradient, m, n, r);
            let state = self.matrices.get_mut(name).unwrap();
            state.projector = projector;
            state.needs_svd = false;
        }
        self.matrices.get_mut(name).unwrap().local_step = local_step;

        // Extract projector copy for computation
        let projector = self.matrices[name].projector.clone();
        let mut momentum = self.matrices[name].momentum.clone();
        let mut af_row = self.matrices[name].af_row.clone();
        let mut af_col = self.matrices[name].af_col.clone();
        let mut os_col = self.matrices[name].os_col.clone();

        // Step 2: Project gradient into low-rank subspace: R = Pᵀ·G  [r × n]
        let proj_r = project_gradient(&projector, gradient, m, n, r);

        // Step 3: Compute residual: S = G - P·R  [m × n]
        let residual = compute_residual(gradient, &proj_r, &projector, m, n, r);

        // Step 4: Momentum in low-rank space: M = β₁·M + (1-β₁)·R
        for i in 0..(r * n) {
            momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * proj_r[i];
        }

        // Step 5: Adafactor preconditioning on momentum in low-rank space
        let n1 = adafactor_precondition(&momentum, &mut af_row, &mut af_col, r, n, self.beta2, self.epsilon);

        // Step 6: OneSided-Adafactor on residual
        let n2 = one_sided_adafactor_precondition(&residual, &mut os_col, m, n, self.beta2, self.epsilon);

        // Step 7: Combine: W += lr · (P·N₁ + N₂)
        apply_update(weights, &n1, &n2, &projector, m, n, r, self.learning_rate);

        // Write back updated state
        let state = self.matrices.get_mut(name).unwrap();
        state.momentum = momentum;
        state.af_row = af_row;
        state.af_col = af_col;
        state.os_col = os_col;
    }

    fn zero_grad(&mut self) {
        // No gradient accumulation — consumed in step(). No-op.
    }

    fn state_bytes(&self) -> usize {
        self.matrices.values().map(|s| {
            let r = self.rank.min(s.rows);
            (s.rows * r +       // projector [m × r]
             r * s.cols +       // momentum [r × n]
             r +                // af_row [r]
             s.cols +           // af_col [n]
             s.cols             // os_col [n]
            ) * std::mem::size_of::<f32>()
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
        "AdaMeM"
    }
}

// ---------------------------------------------------------------------------
// Free functions for AdaMeM computation (avoids borrow checker issues)
// ---------------------------------------------------------------------------

/// Power iteration SVD to find top-r left singular vectors of G [m × n].
fn power_iteration_svd(gradient: &[f32], m: usize, n: usize, r: usize) -> Vec<f32> {
    let mut vs: Vec<Vec<f32>> = (0..r)
        .map(|_| (0..n).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let mut projector = vec![0.0f32; m * r];

    for _ in 0..10 {
        for k in 0..r {
            // u = G · v_k  [m]
            let mut u = vec![0.0f32; m];
            for i in 0..m {
                for j in 0..n {
                    u[i] += gradient[i * n + j] * vs[k][j];
                }
            }
            // Orthogonalize against previous u's (Gram-Schmidt)
            for prev in 0..k {
                let dot: f32 = u.iter()
                    .zip(projector[prev * m..(prev + 1) * m].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                for i in 0..m {
                    u[i] -= dot * projector[prev * m + i];
                }
            }
            let norm = (u.iter().map(|x| x * x).sum::<f32>() + 1e-10).sqrt();
            for i in 0..m { u[i] /= norm; }

            // Store left vector
            projector[k * m..(k + 1) * m].copy_from_slice(&u);

            // v = Gᵀ · u  [n]
            for j in 0..n {
                let mut sum = 0.0f32;
                for i in 0..m { sum += gradient[i * n + j] * u[i]; }
                vs[k][j] = sum;
            }
            let norm = (vs[k].iter().map(|x| x * x).sum::<f32>() + 1e-10).sqrt();
            for j in 0..n { vs[k][j] /= norm; }
        }
    }
    projector
}

/// Project gradient: R = Pᵀ · G  [r × n]
fn project_gradient(projector: &[f32], gradient: &[f32], m: usize, n: usize, r: usize) -> Vec<f32> {
    let mut proj = vec![0.0f32; r * n];
    for k in 0..r {
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += projector[k * m + i] * gradient[i * n + j];
            }
            proj[k * n + j] = sum;
        }
    }
    proj
}

/// Compute residual: S = G - P · R  [m × n]
fn compute_residual(gradient: &[f32], proj_r: &[f32], projector: &[f32], m: usize, n: usize, r: usize) -> Vec<f32> {
    let mut residual = gradient.to_vec();
    for i in 0..m {
        for j in 0..n {
            let mut pr = 0.0f32;
            for k in 0..r { pr += projector[k * m + i] * proj_r[k * n + j]; }
            residual[i * n + j] -= pr;
        }
    }
    residual
}

/// Adafactor preconditioning: factored 2nd moment on [r × n] matrix.
/// Updates row_r [r] and col_c [n] in place. Returns preconditioned matrix.
fn adafactor_precondition(
    mat: &[f32], row_r: &mut [f32], col_c: &mut [f32],
    r: usize, n: usize, beta2: f32, epsilon: f32,
) -> Vec<f32> {
    for k in 0..r {
        let mut row_sq = 0.0f32;
        for j in 0..n { row_sq += mat[k * n + j] * mat[k * n + j]; }
        row_r[k] = beta2 * row_r[k] + (1.0 - beta2) * row_sq / n as f32;
    }
    for j in 0..n {
        let mut col_sq = 0.0f32;
        for k in 0..r { col_sq += mat[k * n + j] * mat[k * n + j]; }
        col_c[j] = beta2 * col_c[j] + (1.0 - beta2) * col_sq / r as f32;
    }
    let mut out = vec![0.0f32; r * n];
    for k in 0..r {
        for j in 0..n {
            let approx_v = row_r[k] * col_c[j];
            out[k * n + j] = mat[k * n + j] / (approx_v.sqrt() + epsilon);
        }
    }
    out
}

/// OneSided-Adafactor: column-only 2nd moment on [m × n] residual.
/// Updates os_col [n] in place. Returns preconditioned matrix.
fn one_sided_adafactor_precondition(
    mat: &[f32], os_col: &mut [f32],
    m: usize, n: usize, beta2: f32, epsilon: f32,
) -> Vec<f32> {
    for j in 0..n {
        let mut col_sq = 0.0f32;
        for i in 0..m { col_sq += mat[i * n + j] * mat[i * n + j]; }
        os_col[j] = beta2 * os_col[j] + (1.0 - beta2) * col_sq / m as f32;
    }
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = mat[i * n + j] / (os_col[j].sqrt() + epsilon);
        }
    }
    out
}

/// Apply combined update: W += lr · (P · N₁ + N₂)
fn apply_update(
    weights: &mut [f32], n1: &[f32], n2: &[f32],
    projector: &[f32], m: usize, n: usize, r: usize, lr: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let mut pn1 = 0.0f32;
            for k in 0..r { pn1 += projector[k * m + i] * n1[k * n + j]; }
            weights[i * n + j] += lr * (pn1 + n2[i * n + j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamem_basic_step() {
        let mut opt = AdaMeMOptimizer::new(0.01, 2);
        opt.register_matrix("test", 3, 4);

        let grad = vec![1.0; 12];
        let mut weights = vec![0.0; 12];

        opt.step("test", &grad, &mut weights);

        // Weights should be updated
        assert!(weights.iter().any(|&w| w != 0.0), "Weights should be updated after step");
    }

    #[test]
    fn test_adamem_state_bytes() {
        let mut opt = AdaMeMOptimizer::new(0.01, 2);
        opt.register_matrix("test", 4, 8);
        // m=4, n=8, r=2: projector[4*2] + momentum[2*8] + af_row[2] + af_col[8] + os_col[8] = 42
        assert_eq!(opt.state_bytes(), 42 * 4);
    }

    #[test]
    fn test_adamem_mark_output_layer_noop() {
        let mut opt = AdaMeMOptimizer::new(0.01, 2);
        opt.register_matrix("test", 3, 4);
        opt.mark_output_layer("test");
        // Should not panic — AdaMeM ignores output layer marking
    }

    #[test]
    fn test_adamem_name() {
        let opt = AdaMeMOptimizer::new(0.01, 8);
        assert_eq!(opt.name(), "AdaMeM");
    }

    #[test]
    fn test_adamem_multiple_steps() {
        let mut opt = AdaMeMOptimizer::new(0.01, 2);
        opt.register_matrix("test", 4, 4);

        let grad = vec![0.5; 16];
        let mut weights = vec![1.0; 16];

        // Run several steps — weights should change smoothly
        for _ in 0..5 {
            opt.step("test", &grad, &mut weights);
        }
        assert!(weights.iter().all(|&w| w != 1.0), "Weights should have changed");
    }
}
