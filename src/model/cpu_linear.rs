use crate::model::ternary::{quantize_ternary, pack_ternary, ternary_matmul};

/// Linear layer with ternary base weights {-1, 0, +1}.
///
/// All base weights are stored as ternary (1 byte/value) with an absmean scale factor.
/// This is the default storage — FerrisRes is a ternary architecture.
/// Memory: 1536×1536 = 9 MB FP32 → ~1.1 MB ternary (8× reduction).
///
/// For training: LoRA adapters (FP32) run on top of the dequantized base output.
/// The base weights are frozen in ternary — only LoRA A/B are updated.
pub struct CpuLinear {
    /// Ternary values {-1, 0, +1}: [out_features × in_features].
    /// Stored as i8 (1 byte each) vs f32 (4 bytes) — 4× less memory.
    /// Packed format (2 bits/value = 4 values/byte) available via `packed`.
    ternary: Vec<i8>,
    /// Packed 2-bit ternary weights (4 values/byte). Used for single-token decode.
    packed: Vec<u8>,
    /// Absmean scale factor: α = mean(|W|) / sqrt(2/π).
    /// Dequantized weight = α * ternary_value.
    scale: f32,
    /// Optional bias vector [out_features].
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl CpuLinear {
    /// Create a zero-initialized linear layer (all ternary values = 0).
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let ternary = vec![0i8; in_features * out_features];
        let packed = pack_ternary(&ternary);
        Self {
            ternary,
            packed,
            scale: 1.0,
            bias: if use_bias { Some(vec![0.0f32; out_features]) } else { None },
            in_features,
            out_features,
        }
    }

    /// Create from FP32 weights — immediately quantized to ternary.
    /// This is the primary constructor. FP32 weights are consumed and quantized.
    /// Weight layout: [in_features × out_features] for direct matmul.
    pub fn from_weight(weight: Vec<f32>, in_features: usize, out_features: usize) -> Self {
        assert_eq!(weight.len(), in_features * out_features,
            "CpuLinear weight len {} != in_features({}) * out_features({})",
            weight.len(), in_features, out_features);
        let (ternary, scale) = quantize_ternary(&weight);
        let packed = pack_ternary(&ternary);
        Self {
            ternary,
            packed,
            scale,
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
            assert_eq!(b.len(), out_features);
        }
        let (ternary, scale) = quantize_ternary(&weight);
        let packed = pack_ternary(&ternary);
        Self { ternary, packed, scale, bias, in_features, out_features }
    }

    /// Create from already-quantized ternary values.
    pub fn from_ternary(ternary: Vec<i8>, scale: f32, in_features: usize, out_features: usize) -> Self {
        let packed = pack_ternary(&ternary);
        Self { ternary, packed, scale, bias: None, in_features, out_features }
    }

    /// Forward pass: input [seq × in_features] → output [seq × out_features].
    /// Uses ternary matmul (add/subtract only — no multiplications on weights).
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let mut output = ternary_matmul(
            &self.ternary, self.scale, input,
            self.out_features, self.in_features, seq,
        );
        if let Some(ref bias) = self.bias {
            for t in 0..seq {
                for j in 0..self.out_features {
                    output[t * self.out_features + j] += bias[j];
                }
            }
        }
        output
    }

    /// Parallel forward using rayon — ~4-8x faster on multi-core.
    /// Use for large matrices during prefill or batch processing.
    pub fn forward_parallel(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let mut output = crate::model::ternary::ternary_matmul_parallel(
            &self.ternary, self.scale, input,
            self.out_features, self.in_features, seq,
        );
        if let Some(ref bias) = self.bias {
            for t in 0..seq {
                for j in 0..self.out_features {
                    output[t * self.out_features + j] += bias[j];
                }
            }
        }
        output
    }

    /// Packed parallel forward — 4× less memory bandwidth than unpacked.
    /// Fastest path for decode (seq=1) on memory-bound systems.
    pub fn forward_packed_parallel(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let mut output = crate::model::ternary::ternary_matmul_packed_parallel(
            &self.packed, self.scale, input,
            self.out_features, self.in_features, seq,
        );
        if let Some(ref bias) = self.bias {
            for j in 0..self.out_features {
                output[j] += bias[j];
            }
        }
        output
    }

    /// Dequantize all weights to FP32 (for export, checkpoint saving, etc).
    pub fn weight(&self) -> Vec<f32> {
        self.ternary.iter().map(|&v| v as f32 * self.scale).collect()
    }

    /// Access ternary values directly.
    pub fn ternary_values(&self) -> &[i8] { &self.ternary }
    pub fn scale(&self) -> f32 { self.scale }
    pub fn packed(&self) -> &[u8] { &self.packed }

    /// Access bias if present.
    pub fn bias(&self) -> Option<&[f32]> { self.bias.as_deref() }

    pub fn in_features(&self) -> usize { self.in_features }
    pub fn out_features(&self) -> usize { self.out_features }

    /// Memory usage in bytes (ternary + packed + scale).
    pub fn memory_bytes(&self) -> usize {
        self.ternary.len() + self.packed.len() + 4 + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }

    /// What the FP32 equivalent would be.
    pub fn fp32_equivalent_bytes(&self) -> usize {
        self.in_features * self.out_features * 4
    }
}

/// CPU-only RMS normalization. Stores weights as `Vec<f32>`.
/// Norms are tiny (a few KB per layer) and critical for quality — kept in FP32.
pub struct CpuRmsNorm {
    weight: Vec<f32>,
    eps: f32,
    hidden_dim: usize,
}

impl CpuRmsNorm {
    pub fn new(hidden_dim: usize, eps: f32) -> Self {
        Self { weight: vec![1.0f32; hidden_dim], eps, hidden_dim }
    }

    pub fn from_weight(weight: Vec<f32>, eps: f32) -> Self {
        let hidden_dim = weight.len();
        Self { weight, eps, hidden_dim }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        crate::model::gemma_mapper::rms_norm(input, &self.weight, self.hidden_dim, self.eps)
    }

    /// RMS norm for a single token (avoids unnecessary allocation).
    pub fn forward_single(&self, input: &[f32]) -> Vec<f32> {
        let dim = self.hidden_dim;
        let mean_sq: f32 = input[..dim].iter().map(|x| x * x).sum::<f32>() / dim as f32;
        let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
        input[..dim].iter().enumerate().map(|(d, &x)| {
            x * inv_rms * self.weight.get(d).copied().unwrap_or(1.0)
        }).collect()
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
        let weight = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // 3×2 identity-ish
        let linear = CpuLinear::from_weight(weight, 3, 2);
        let input = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2×3
        let output = linear.forward(&input, 2);
        assert_eq!(output.len(), 4);
        // Ternary of [1,0,0,1,0,0] with scale ≈ mean/0.798
        // Should produce finite output
        for &v in &output { assert!(v.is_finite()); }
    }

    #[test]
    fn test_memory_reduction() {
        let weight: Vec<f32> = (0..1536*1536).map(|i| (i as f32 * 0.001).sin()).collect();
        let linear = CpuLinear::from_weight(weight, 1536, 1536);
        let fp32 = linear.fp32_equivalent_bytes();
        let actual = linear.memory_bytes();
        assert!(actual < fp32 / 3, "ternary {} should be < fp32 {}/3", actual, fp32);
    }

    #[test]
    fn test_from_ternary() {
        let ternary = vec![-1i8, 0, 1, -1, 1, 0];
        let linear = CpuLinear::from_ternary(ternary, 0.5, 3, 2);
        let input = vec![1.0, 0.0, -1.0]; // 1×3
        let output = linear.forward(&input, 1);
        assert_eq!(output.len(), 2);
        // Row 0: [-1,0,1] · [1,0,-1] = -1*1 + 0*0 + 1*(-1) = -2, × scale 0.5 = -1.0
        assert!((output[0] - (-1.0)).abs() < 1e-5, "got {}", output[0]);
    }
}
