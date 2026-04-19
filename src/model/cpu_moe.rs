use crate::model::gemma_mapper::matmul;

/// CPU-only Mixture of Experts layer. Stores all weights as `Vec<f32>`.
/// Used by CpuBlockAttnResLayer for MoE FFN.
///
/// Each expert has a gate_proj, up_proj, and down_proj.
/// The router computes softmax logits, selects top-k experts per token,
/// and produces a weighted sum of expert outputs.
///
/// Activation: supports both SwiGLU (silu-based) and GeLU (tanh approximation).
pub struct CpuMoELayer {
    /// Router weights: [hidden_dim, num_experts].
    pub gate_weights: Vec<f32>,
    /// Expert gate projections (for activation): [num_experts][intermediate * hidden].
    pub expert_gate: Vec<Vec<f32>>,
    /// Expert up projections: [num_experts][intermediate * hidden].
    pub expert_up: Vec<Vec<f32>>,
    /// Expert down projections: [num_experts][hidden * intermediate].
    pub expert_down: Vec<Vec<f32>>,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    /// Activation type: true = GeLU (tanh approx), false = SwiGLU (silu).
    pub use_gelu: bool,
}

impl CpuMoELayer {
    pub fn new(hidden_dim: usize, intermediate_dim: usize, num_experts: usize, top_k: usize) -> Self {
        let gate_weights = vec![0.0; num_experts * hidden_dim];
        let expert_gate = (0..num_experts)
            .map(|_| vec![0.0; intermediate_dim * hidden_dim])
            .collect();
        let expert_up = (0..num_experts)
            .map(|_| vec![0.0; intermediate_dim * hidden_dim])
            .collect();
        let expert_down = (0..num_experts)
            .map(|_| vec![0.0; hidden_dim * intermediate_dim])
            .collect();

        Self {
            gate_weights,
            expert_gate,
            expert_up,
            expert_down,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            use_gelu: false, // Default SwiGLU
        }
    }

    /// Forward pass: route tokens to top-k experts, compute weighted output.
    /// Input: [seq, hidden_dim], Output: [seq, hidden_dim].
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let h = self.hidden_dim;

        // 1. Compute gate logits: [seq, num_experts]
        let gate_logits = matmul(input, &self.gate_weights, seq, h, self.num_experts);

        // 2. Top-k selection with softmax weighting
        let (selected_experts, expert_weights) = self.top_k_select(&gate_logits, seq);

        // 3. Compute expert outputs and accumulate
        let mut output = vec![0.0; seq * h];

        for t in 0..seq {
            for k_idx in 0..self.top_k {
                let expert_idx = selected_experts[t * self.top_k + k_idx];
                let weight = expert_weights[t * self.top_k + k_idx];

                if weight.abs() < 1e-8 { continue; }

                let token = &input[t * h..(t + 1) * h];
                let expert_out = self.expert_forward(expert_idx, token);

                for (i, &v) in expert_out.iter().enumerate() {
                    output[t * h + i] += weight * v;
                }
            }
        }

        output
    }

    /// Top-k expert selection with softmax normalization.
    /// Returns (selected_expert_indices[seq * top_k], weights[seq * top_k]).
    fn top_k_select(&self, gate_logits: &[f32], seq: usize) -> (Vec<usize>, Vec<f32>) {
        let ne = self.num_experts;
        let tk = self.top_k;
        let mut selected = vec![0usize; seq * tk];
        let mut weights = vec![0.0f32; seq * tk];

        for t in 0..seq {
            let logits = &gate_logits[t * ne..(t + 1) * ne];

            // Find top-k
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Softmax over top-k logits for weight normalization
            let max_logit = indexed[0].1;
            let mut sum_exp = 0.0f32;
            for k in 0..tk.min(ne) {
                let exp_val = (indexed[k].1 - max_logit).exp();
                sum_exp += exp_val;
                selected[t * tk + k] = indexed[k].0;
                weights[t * tk + k] = exp_val;
            }
            if sum_exp > 0.0 {
                for k in 0..tk.min(ne) {
                    weights[t * tk + k] /= sum_exp;
                }
            }
        }

        (selected, weights)
    }

    /// Single expert forward: activation(gate(x)) * up(x) → down.
    fn expert_forward(&self, expert_idx: usize, token: &[f32]) -> Vec<f32> {
        let h = self.hidden_dim;
        let id = self.intermediate_dim;
        let gate = &self.expert_gate[expert_idx]; // [id * h] stored as [h, id] for matmul
        let up = &self.expert_up[expert_idx];     // [id * h] stored as [h, id] for matmul
        let down = &self.expert_down[expert_idx]; // [h * id] stored as [id, h] for matmul

        // gate projection: [1, h] @ [h, id] = [1, id]
        let gated = matmul(token, gate, 1, h, id);
        // Apply activation
        let gated: Vec<f32> = if self.use_gelu {
            gated.iter().map(|&x| crate::model::gemma_mapper::gelu_tanh(x)).collect()
        } else {
            // SwiGLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
            gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
        };

        // up projection: [1, h] @ [h, id] = [1, id]
        let upped = matmul(token, up, 1, h, id);

        // Element-wise multiply
        let mut combined = vec![0.0; id];
        for i in 0..id {
            combined[i] = gated[i] * upped[i];
        }

        // down projection: [1, id] @ [id, h] = [1, h]
        matmul(&combined, down, 1, id, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_moe_basic() {
        let hd = 4;
        let id = 8;
        let ne = 2;
        let tk = 1;

        let mut moe = CpuMoELayer::new(hd, id, ne, tk);
        // Set router to always pick expert 0
        moe.gate_weights[0] = 10.0; // Expert 0 gets high score for token[0]
        // Make expert 0 identity-like: gate=eye, up=eye, down=eye
        for i in 0..hd.min(id) {
            moe.expert_gate[0][i * id + i] = 1.0;
            moe.expert_up[0][i * id + i] = 1.0;
            moe.expert_down[0][i * hd + i] = 1.0;
        }

        let input = vec![1.0, 2.0, 3.0, 4.0]; // seq=1, hd=4
        let output = moe.forward(&input, 1);
        assert_eq!(output.len(), 4);
        // With silu(1)=0.73, *up(1), then down: should produce nonzero output
        assert!(output.iter().any(|&x| x.abs() > 0.01));
    }
}
