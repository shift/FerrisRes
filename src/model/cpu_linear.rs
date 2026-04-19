use crate::model::gemma_mapper::matmul;

/// CPU-only linear layer. Stores weights as `Vec<f32>`, computes with CPU matmul.
/// Used by CpuBlockAttnResLayer and weight mapping from Gemma 4.
/// No wgpu dependency — works on any platform.
pub struct CpuLinear {
    /// Weight matrix stored as `[out_features, in_features]` (row-major).
    /// matmul computes: output = input @ weight^T  (nn.Linear convention)
    /// But we store as `[in_features, out_features]` for direct matmul:
    /// `output = input @ weight` where weight is `[in_features, out_features]`.
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl CpuLinear {
    /// Create a zero-initialized linear layer.
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        Self {
            weight: vec![0.0f32; in_features * out_features],
            bias: if use_bias { Some(vec![0.0f32; out_features]) } else { None },
            in_features,
            out_features,
        }
    }

    /// Create from an existing weight vector.
    /// Weight shape: `[in_features, out_features]` (for direct matmul).
    pub fn from_weight(weight: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        assert_eq!(weight.len(), in_features * out_features,
            "CpuLinear weight len {} != in_features({}) * out_features({})",
            weight.len(), in_features, out_features);
        Self {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Create from weight + bias.
    pub fn from_weight_bias(weight: Vec<f32>, bias: Option<Vec<f32>>, in_features: usize, out_features: usize) -> Self {
        assert_eq!(weight.len(), in_features * out_features,
            "CpuLinear weight len {} != in_features({}) * out_features({})",
            weight.len(), in_features, out_features);
        if let Some(ref b) = bias {
            assert_eq!(b.len(), out_features,
                "CpuLinear bias len {} != out_features({})", b.len(), out_features);
        }
        Self { weight, bias, in_features, out_features }
    }

    /// Forward pass: input `[seq, in_features]` → output `[seq, out_features]`.
    /// Uses the same matmul as gemma_mapper (matrixmultiply with SIMD).
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let mut output = matmul(input, &self.weight, seq, self.in_features, self.out_features);
        if let Some(ref bias) = self.bias {
            for t in 0..seq {
                for j in 0..self.out_features {
                    output[t * self.out_features + j] += bias[j];
                }
            }
        }
        output
    }

    /// Access the weight matrix.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Access the weight matrix mutably (for weight initialization).
    pub fn weight_mut(&mut self) -> &mut Vec<f32> {
        &mut self.weight
    }

    /// Access bias if present.
    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }
}

/// CPU-only RMS normalization. Stores weights as `Vec<f32>`.
/// Used by CpuBlockAttnResLayer alongside CpuLinear.
pub struct CpuRmsNorm {
    /// Learnable scale weights `[hidden_dim]`.
    weight: Vec<f32>,
    eps: f32,
    hidden_dim: usize,
}

impl CpuRmsNorm {
    pub fn new(hidden_dim: usize, eps: f32) -> Self {
        Self {
            weight: vec![1.0f32; hidden_dim],
            eps,
            hidden_dim,
        }
    }

    pub fn from_weight(weight: Vec<f32>, eps: f32) -> Self {
        let hidden_dim = weight.len();
        Self { weight, eps, hidden_dim }
    }

    /// Forward: normalized(x) = x / sqrt(mean(x^2) + eps) * weight
    /// Input shape: [seq * hidden_dim], output: [seq * hidden_dim]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        crate::model::gemma_mapper::rms_norm(input, &self.weight, self.hidden_dim, self.eps)
    }

    pub fn weight(&self) -> &[f32] { &self.weight }
    pub fn weight_mut(&mut self) -> &mut Vec<f32> { &mut self.weight }
    pub fn hidden_dim(&self) -> usize { self.hidden_dim }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_linear_forward() {
        // 2x3 @ 3x2 = 2x2
        let input = vec![1.0, 0.0, 0.0,  0.0, 1.0, 0.0]; // 2x3 identity-ish
        let weight = vec![1.0, 0.0,  0.0, 1.0,  0.0, 0.0]; // 3x2
        let linear = CpuLinear::from_weight(weight, 3, 2);
        let output = linear.forward(&input, 2);
        assert_eq!(output.len(), 4);
        assert!((output[0] - 1.0).abs() < 1e-5); // [1,0] @ [1;0; 0;1; 0;0] = [1,0]
        assert!((output[1] - 0.0).abs() < 1e-5);
        assert!((output[2] - 0.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_linear_with_bias() {
        let weight = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let bias = Some(vec![10.0, 20.0]);
        let linear = CpuLinear::from_weight_bias(weight, bias, 2, 2);
        let input = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let output = linear.forward(&input, 2);
        assert!((output[0] - 11.0).abs() < 1e-5);
        assert!((output[3] - 21.0).abs() < 1e-5);
    }
}
