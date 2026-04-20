//! Ternary MoE layer — quantized Mixture-of-Experts using {-1, 0, +1} expert weights.
//!
//! Inference-only version of `CpuMoELayer`. Expert weights are ternary-quantized
//! for 16× memory reduction. Router stays in FP32 (tiny, critical for routing quality).
//!
//! Conversion: `CpuMoELayer` → `TernaryMoELayer::from_cpu_moe()`
//! Memory: 4 experts × 3 matrices × [1536×12288] = 216MB FP32 → ~13.5MB ternary

use crate::model::cpu_moe::CpuMoELayer;
use crate::model::ternary::{pack_ternary, quantize_ternary, ternary_matmul, ternary_matmul_decode};

/// A single ternary-quantized weight matrix: (packed 2-bit data, absmean scale, unpacked values).
#[derive(Debug, Clone)]
pub struct TernaryWeight {
    /// Packed 2-bit ternary weights.
    pub packed: Vec<u8>,
    /// Absmean scale factor.
    pub scale: f32,
    /// Unpacked ternary values (kept for fast decode path).
    pub ternary: Vec<i8>,
    /// Original number of values.
    #[allow(dead_code)]
    pub packed_len: usize,
}

impl TernaryWeight {
    /// Quantize an FP32 weight matrix to ternary.
    pub fn from_fp32(weights: &[f32]) -> Self {
        let (ternary, scale) = quantize_ternary(weights);
        let packed = pack_ternary(&ternary);
        TernaryWeight {
            packed,
            scale,
            ternary,
            packed_len: weights.len(),
        }
    }

    /// Forward: `output = scale * ternary @ input`.
    pub fn matmul(&self, input: &[f32], out_rows: usize, in_cols: usize, seq: usize) -> Vec<f32> {
        ternary_matmul(&self.ternary, self.scale, input, out_rows, in_cols, seq)
    }

    /// Decode forward (seq=1).
    pub fn matmul_decode(&self, input: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
        ternary_matmul_decode(&self.ternary, self.scale, input, out_rows, in_cols)
    }

    /// Drop unpacked values to save memory.
    pub fn drop_unpacked(&mut self) {
        self.ternary = Vec::new();
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.ternary.len()
    }
}

/// Quantized MoE layer with ternary expert weights.
///
/// Each expert has 3 ternary weight matrices (gate, up, down) for SwiGLU/GeLU.
/// The router is kept in FP32 — it's tiny (hidden_dim × num_experts) and
/// routing quality is critical.
#[derive(Debug, Clone)]
pub struct TernaryMoELayer {
    /// Router weights: [hidden_dim × num_experts] in FP32.
    /// Kept unquantized because routing accuracy dominates MoE quality.
    pub gate_weights: Vec<f32>,

    /// Ternary expert weights: [num_experts][gate, up, down].
    pub expert_gate: Vec<TernaryWeight>,
    pub expert_up: Vec<TernaryWeight>,
    pub expert_down: Vec<TernaryWeight>,

    /// Hidden dimension (input/output).
    pub hidden_dim: usize,

    /// Intermediate dimension (FFN expansion).
    pub intermediate_dim: usize,

    /// Number of experts.
    pub num_experts: usize,

    /// Top-k experts to activate per token.
    pub top_k: usize,

    /// Activation function: true = GeLU (tanh approx), false = SwiGLU.
    pub use_gelu: bool,
}

impl TernaryMoELayer {
    /// Create from an FP32 CpuMoELayer via absmean quantization.
    ///
    /// Quantizes each expert's gate/up/down weight matrices to ternary.
    /// Router weights are copied as-is (FP32).
    pub fn from_cpu_moe(moe: &CpuMoELayer) -> Self {
        let expert_gate: Vec<TernaryWeight> = moe
            .expert_gate
            .iter()
            .map(|w| TernaryWeight::from_fp32(w))
            .collect();

        let expert_up: Vec<TernaryWeight> = moe
            .expert_up
            .iter()
            .map(|w| TernaryWeight::from_fp32(w))
            .collect();

        let expert_down: Vec<TernaryWeight> = moe
            .expert_down
            .iter()
            .map(|w| TernaryWeight::from_fp32(w))
            .collect();

        TernaryMoELayer {
            gate_weights: moe.gate_weights.clone(),
            expert_gate,
            expert_up,
            expert_down,
            hidden_dim: moe.hidden_dim,
            intermediate_dim: moe.intermediate_dim,
            num_experts: moe.num_experts,
            top_k: moe.top_k,
            use_gelu: moe.use_gelu,
        }
    }

    /// Forward pass: router → top-k selection → ternary expert matmul → weighted sum.
    ///
    /// - `input`: `[seq * hidden_dim]` FP32 activations
    /// - Returns: `[seq * hidden_dim]` FP32 output
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; seq * self.hidden_dim];

        for s in 0..seq {
            let input_row = &input[s * self.hidden_dim..(s + 1) * self.hidden_dim];
            let output_row = &mut output[s * self.hidden_dim..(s + 1) * self.hidden_dim];

            // 1. Router: gate_weights @ input → expert scores
            let mut scores = vec![0.0f32; self.num_experts];
            for e in 0..self.num_experts {
                let mut sum = 0.0f32;
                for j in 0..self.hidden_dim {
                    sum += self.gate_weights[e * self.hidden_dim + j] * input_row[j];
                }
                scores[e] = sum;
            }

            // 2. Softmax over scores
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s_val in scores.iter_mut() {
                *s_val = (*s_val - max_score).exp();
                sum_exp += *s_val;
            }
            for s_val in scores.iter_mut() {
                *s_val /= sum_exp;
            }

            // 3. Top-k selection
            let mut indexed: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k = indexed[..self.top_k.min(indexed.len())].to_vec();

            // 4. Compute weighted expert outputs
            let mut weight_sum = 0.0f32;
            for &(_expert_idx, score) in &top_k {
                weight_sum += score;
            }

            for &(expert_idx, score) in &top_k {
                let gate_out = self.expert_gate[expert_idx].matmul_decode(
                    input_row, self.intermediate_dim, self.hidden_dim,
                );
                let up_out = self.expert_up[expert_idx].matmul_decode(
                    input_row, self.intermediate_dim, self.hidden_dim,
                );

                // Activation: GeLU or SwiGLU
                let mut intermediate = vec![0.0f32; self.intermediate_dim];
                for i in 0..self.intermediate_dim {
                    if self.use_gelu {
                        // GeLU tanh approximation
                        let x = gate_out[i];
                        let tanh_arg = 0.7978845608 * (x + 0.044715 * x * x * x);
                        let gelu = 0.5 * x * (1.0 + tanh_arg.tanh());
                        intermediate[i] = gelu;
                    } else {
                        // SwiGLU: gate * sigmoid(gate) * up
                        let gate_val = gate_out[i];
                        let silu = gate_val / (1.0 + (-gate_val).exp());
                        intermediate[i] = silu * up_out[i];
                    }
                }

                // Down projection: intermediate → hidden_dim
                let down_out = self.expert_down[expert_idx].matmul_decode(
                    &intermediate, self.hidden_dim, self.intermediate_dim,
                );

                // Weighted sum into output
                let weight = if weight_sum > 0.0 { score / weight_sum } else { score };
                for j in 0..self.hidden_dim {
                    output_row[j] += weight * down_out[j];
                }
            }
        }

        output
    }

    /// Total memory usage in bytes (all expert weights + router).
    pub fn memory_bytes(&self) -> usize {
        let router = self.gate_weights.len() * 4;
        let experts: usize = self.expert_gate.iter().map(|w| w.memory_bytes()).sum::<usize>()
            + self.expert_up.iter().map(|w| w.memory_bytes()).sum::<usize>()
            + self.expert_down.iter().map(|w| w.memory_bytes()).sum::<usize>();
        router + experts
    }

    /// Equivalent FP32 memory for comparison.
    pub fn fp32_equivalent_bytes(&self) -> usize {
        let router = self.gate_weights.len() * 4;
        let experts = self.num_experts * 3 * self.hidden_dim * self.intermediate_dim * 4;
        router + experts
    }

    /// Compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f32 {
        let fp32 = self.fp32_equivalent_bytes() as f32;
        let actual = self.memory_bytes() as f32;
        if actual > 0.0 { fp32 / actual } else { 1.0 }
    }

    /// Drop unpacked ternary values from all experts to save memory.
    pub fn drop_all_unpacked(&mut self) {
        for w in &mut self.expert_gate {
            w.drop_unpacked();
        }
        for w in &mut self.expert_up {
            w.drop_unpacked();
        }
        for w in &mut self.expert_down {
            w.drop_unpacked();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cpu_moe::CpuMoELayer;

    fn make_test_moe() -> CpuMoELayer {
        let hidden_dim = 8;
        let intermediate_dim = 16;
        let num_experts = 2;
        let top_k = 2;

        let gate_weights: Vec<f32> = (0..hidden_dim * num_experts)
            .map(|i| i as f32 * 0.1 - 0.4)
            .collect();

        let make_expert_weights = |base: f32, size: usize| -> Vec<f32> {
            (0..size).map(|i| ((i as f32 + base) * 0.05).sin()).collect()
        };

        let expert_gate: Vec<Vec<f32>> = (0..num_experts)
            .map(|e| make_expert_weights(e as f32 * 0.1, intermediate_dim * hidden_dim))
            .collect();
        let expert_up: Vec<Vec<f32>> = (0..num_experts)
            .map(|e| make_expert_weights(e as f32 * 0.1 + 0.5, intermediate_dim * hidden_dim))
            .collect();
        let expert_down: Vec<Vec<f32>> = (0..num_experts)
            .map(|e| make_expert_weights(e as f32 * 0.1 + 1.0, hidden_dim * intermediate_dim))
            .collect();

        CpuMoELayer {
            gate_weights,
            expert_gate,
            expert_up,
            expert_down,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k,
            use_gelu: false,
        }
    }

    #[test]
    fn test_from_cpu_moe() {
        let moe = make_test_moe();
        let ternary = TernaryMoELayer::from_cpu_moe(&moe);

        assert_eq!(ternary.hidden_dim, 8);
        assert_eq!(ternary.intermediate_dim, 16);
        assert_eq!(ternary.num_experts, 2);
        assert_eq!(ternary.top_k, 2);
        assert_eq!(ternary.expert_gate.len(), 2);
        assert_eq!(ternary.expert_up.len(), 2);
        assert_eq!(ternary.expert_down.len(), 2);
    }

    #[test]
    fn test_forward_output_shape() {
        let moe = make_test_moe();
        let ternary = TernaryMoELayer::from_cpu_moe(&moe);

        let input = vec![0.1f32; 8];
        let output = ternary.forward(&input, 1);

        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_forward_batch() {
        let moe = make_test_moe();
        let ternary = TernaryMoELayer::from_cpu_moe(&moe);

        let input = vec![0.1f32; 8 * 3]; // batch of 3
        let output = ternary.forward(&input, 3);

        assert_eq!(output.len(), 8 * 3);
    }

    #[test]
    fn test_forward_not_all_zeros() {
        let moe = make_test_moe();
        let ternary = TernaryMoELayer::from_cpu_moe(&moe);

        let input: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let output = ternary.forward(&input, 1);

        let has_nonzero = output.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "output should have non-zero values");
    }

    #[test]
    fn test_compression_ratio() {
        let moe = make_test_moe();
        let ternary = TernaryMoELayer::from_cpu_moe(&moe);

        let ratio = ternary.compression_ratio();
        // With packed + unpacked stored, ratio should be > 2x
        assert!(ratio > 2.0, "compression ratio should be > 2x, got {}", ratio);
    }

    #[test]
    fn test_drop_unpacked() {
        let moe = make_test_moe();
        let mut ternary = TernaryMoELayer::from_cpu_moe(&moe);

        let before = ternary.memory_bytes();
        ternary.drop_all_unpacked();
        let after = ternary.memory_bytes();

        assert!(after < before, "memory should decrease after dropping unpacked");

        let ratio = ternary.compression_ratio();
        assert!(ratio > 8.0, "packed-only ratio should be > 8x, got {}", ratio);
    }

    #[test]
    fn test_gelu_vs_swiglu() {
        let hidden_dim = 4;
        let intermediate_dim = 8;
        let num_experts = 2;

        let gate_weights: Vec<f32> = vec![0.1; hidden_dim * num_experts];
        let make_weights = |base: f32, size: usize| -> Vec<f32> {
            (0..size).map(|i| ((i as f32 + base) * 0.1).sin()).collect()
        };

        let expert_gate = vec![make_weights(0.0, intermediate_dim * hidden_dim), make_weights(1.0, intermediate_dim * hidden_dim)];
        let expert_up = vec![make_weights(0.5, intermediate_dim * hidden_dim), make_weights(1.5, intermediate_dim * hidden_dim)];
        let expert_down = vec![make_weights(0.0, hidden_dim * intermediate_dim), make_weights(1.0, hidden_dim * intermediate_dim)];

        let moe_gelu = CpuMoELayer {
            gate_weights: gate_weights.clone(),
            expert_gate: expert_gate.clone(),
            expert_up: expert_up.clone(),
            expert_down: expert_down.clone(),
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k: 1,
            use_gelu: true,
        };

        let moe_swiglu = CpuMoELayer {
            gate_weights,
            expert_gate,
            expert_up,
            expert_down,
            hidden_dim,
            intermediate_dim,
            num_experts,
            top_k: 1,
            use_gelu: false,
        };

        let ternary_gelu = TernaryMoELayer::from_cpu_moe(&moe_gelu);
        let ternary_swiglu = TernaryMoELayer::from_cpu_moe(&moe_swiglu);

        let input = vec![0.5f32; hidden_dim];
        let out_gelu = ternary_gelu.forward(&input, 1);
        let out_swiglu = ternary_swiglu.forward(&input, 1);

        // GeLU and SwiGLU should produce different outputs
        let any_different = out_gelu.iter().zip(out_swiglu.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_different, "GeLU and SwiGLU should produce different outputs");
    }
}
