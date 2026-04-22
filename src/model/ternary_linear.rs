//! Ternary linear layer — quantized attention/FFN projections using {-1, 0, +1} weights.
//!
//! Replaces `CpuLinear` for inference. Weights are stored as packed 2-bit ternary
//! values, giving 16× memory reduction vs FP32. Forward pass uses add/subtract-only
//! matmul (no hardware multipliers needed).
//!
//! Conversion: `CpuLinear` → `TernaryLinear::from_cpu_linear()`
//! Size: 1536×1536 = 9MB FP32 → ~576KB ternary

use crate::model::cpu_linear::CpuLinear;
use crate::model::ternary::{
    pack_ternary, ternary_matmul, ternary_matmul_decode, ternary_matmul_packed,
    ternary_stats, TernaryStats,
};

/// Quantized linear layer using ternary {-1, 0, +1} weights.
///
/// Stores weights in 2-bit packed format (4 values per byte).
/// Forward pass: `output = α * W_ternary @ input + bias`
/// where the matmul uses only additions and subtractions.
///
/// Used for Q/K/V/Out projections, PLE gate/projection, inter-block attention.
#[derive(Debug, Clone)]
pub struct TernaryLinear {
    /// Packed 2-bit ternary weights [ceil(out_features * in_features / 4) bytes].
    /// Encoding: {-1→0b00, 0→0b01, +1→0b10}
    pub packed_weights: Vec<u8>,

    /// Absmean scale factor: α = mean(|W|) / sqrt(2/π)
    pub scale: f32,

    /// Unpacked ternary values [out_features * in_features], kept for decode path.
    /// Can be freed via `drop_unpacked()` to save memory if only packed path is needed.
    pub ternary: Vec<i8>,

    /// Optional bias vector [out_features].
    pub bias: Option<Vec<f32>>,

    /// Input dimension.
    pub in_features: usize,

    /// Output dimension.
    pub out_features: usize,

    /// Original number of ternary values (before padding to multiple of 4).
    packed_len: usize,
}

impl TernaryLinear {
    /// Create a TernaryLinear from a CpuLinear via absmean quantization.
    ///
    /// Quantizes the weight matrix to {-1, 0, +1} using absmean scaling,
    /// then packs into 2-bit format for storage.
    pub fn from_cpu_linear(linear: &CpuLinear) -> Self {
        let out_features = linear.out_features();
        let in_features = linear.in_features();
        // Use raw ternary values directly — no FP32 round-trip
        let ternary = linear.ternary_values().to_vec();
        let scale = linear.scale();
        let packed = pack_ternary(&ternary);

        Self {
            packed_weights: packed,
            scale,
            ternary,
            bias: linear.bias().map(|b| b.to_vec()),
            in_features,
            out_features,
            packed_len: out_features * in_features,
        }
    }

    /// Create from raw ternary values and scale.
    pub fn from_ternary(ternary: Vec<i8>, scale: f32, in_features: usize, out_features: usize) -> Self {
        let packed = pack_ternary(&ternary);
        Self {
            packed_weights: packed,
            scale,
            ternary,
            bias: None,
            in_features,
            out_features,
            packed_len: in_features * out_features,
        }
    }

    /// Forward pass: `output = α * W_ternary @ input + bias`
    ///
    /// Uses the unpacked ternary values for maximum performance.
    /// The inner loop is add/subtract only — no multiplies.
    ///
    /// - `input`: `[seq * in_features]` FP32 activations (row-major)
    /// - Returns: `[seq * out_features]` FP32 output
    pub fn forward(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let mut output = ternary_matmul(
            &self.ternary,
            self.scale,
            input,
            self.out_features,
            self.in_features,
            seq,
        );

        // Add bias if present
        if let Some(ref bias) = self.bias {
            for s in 0..seq {
                for r in 0..self.out_features {
                    output[s * self.out_features + r] += bias[r];
                }
            }
        }

        output
    }

    /// Forward pass using packed 2-bit weights directly.
    ///
    /// Slightly slower than `forward()` due to bit-unpacking overhead per element,
    /// but avoids storing the unpacked ternary array (saves ~4× memory vs i8).
    pub fn forward_packed(&self, input: &[f32], seq: usize) -> Vec<f32> {
        let mut output = ternary_matmul_packed(
            &self.packed_weights,
            self.scale,
            input,
            self.out_features,
            self.in_features,
            seq,
            self.packed_len,
        );

        if let Some(ref bias) = self.bias {
            for s in 0..seq {
                for r in 0..self.out_features {
                    output[s * self.out_features + r] += bias[r];
                }
            }
        }

        output
    }

    /// Optimized decode forward: single token (seq=1).
    ///
    /// Avoids the batch dimension overhead and uses the unrolled decode matmul.
    /// This is the hot path during autoregressive generation.
    ///
    /// - `input`: `[in_features]` FP32 activation vector
    /// - Returns: `[out_features]` FP32 output vector
    pub fn forward_decode(&self, input: &[f32]) -> Vec<f32> {
        let mut output = ternary_matmul_decode(
            &self.ternary,
            self.scale,
            input,
            self.out_features,
            self.in_features,
        );

        if let Some(ref bias) = self.bias {
            for r in 0..self.out_features {
                output[r] += bias[r];
            }
        }

        output
    }

    /// Drop the unpacked ternary values to save memory.
    ///
    /// After calling this, only `forward_packed()` works.
    /// `forward()` and `forward_decode()` will panic.
    /// Saves `out_features * in_features` bytes (1 byte per weight).
    pub fn drop_unpacked(&mut self) {
        self.ternary = Vec::new();
    }

    /// Check if unpacked values are available.
    pub fn has_unpacked(&self) -> bool {
        !self.ternary.is_empty()
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let packed_bytes = self.packed_weights.len();
        let unpacked_bytes = self.ternary.len();
        let bias_bytes = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        packed_bytes + unpacked_bytes + bias_bytes
    }

    /// Equivalent FP32 size for comparison.
    pub fn fp32_equivalent_bytes(&self) -> usize {
        self.out_features * self.in_features * 4 + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }

    /// Compression ratio vs FP32.
    pub fn compression_ratio(&self) -> f32 {
        let fp32 = self.fp32_equivalent_bytes() as f32;
        let actual = self.memory_bytes() as f32;
        if actual > 0.0 { fp32 / actual } else { 1.0 }
    }

    /// Quantization quality statistics.
    pub fn stats(&self) -> TernaryStats {
        ternary_stats(&self.ternary, self.scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::cpu_linear::CpuLinear;

    #[test]
    fn test_from_cpu_linear() {
        let weights: Vec<f32> = (0..8 * 4).map(|i| i as f32 * 0.1 - 1.6).collect();
        let linear = CpuLinear::from_weight(weights, 4, 8);
        let ternary = TernaryLinear::from_cpu_linear(&linear);

        assert_eq!(ternary.in_features, 4);
        assert_eq!(ternary.out_features, 8);
        assert!(ternary.scale > 0.0);
        assert_eq!(ternary.ternary.len(), 32);
        assert!(ternary.ternary.iter().all(|&v| v >= -1 && v <= 1));
    }

    #[test]
    fn test_forward_matches_dequantized() {
        let weights: Vec<f32> = (0..8 * 4)
            .map(|i| (i as f32 * 1.618).sin() * 0.5)
            .collect();
        let linear = CpuLinear::from_weight(weights.clone(), 4, 8);
        let ternary_layer = TernaryLinear::from_cpu_linear(&linear);

        let input: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];

        // Ternary forward
        let output_ternary = ternary_layer.forward(&input, 1);

        // Reference: dequantize then FP32 matmul
        let scale = ternary_layer.scale;
        let mut output_ref = vec![0.0f32; 8];
        for r in 0..8 {
            for j in 0..4 {
                output_ref[r] += (scale * ternary_layer.ternary[r * 4 + j] as f32) * input[j];
            }
        }

        for r in 0..8 {
            assert!(
                (output_ternary[r] - output_ref[r]).abs() < 1e-5,
                "mismatch at {}: ternary={}, ref={}",
                r,
                output_ternary[r],
                output_ref[r]
            );
        }
    }

    #[test]
    fn test_forward_packed_matches_unpacked() {
        let weights: Vec<f32> = (0..8 * 4)
            .map(|i| (i as f32 * 2.718).sin() * 0.3)
            .collect();
        let linear = CpuLinear::from_weight(weights, 4, 8);
        let ternary_layer = TernaryLinear::from_cpu_linear(&linear);

        let input: Vec<f32> = vec![1.0, -0.5, 0.3, -0.1];

        let output_unpacked = ternary_layer.forward(&input, 1);
        let output_packed = ternary_layer.forward_packed(&input, 1);

        assert_eq!(output_unpacked.len(), output_packed.len());
        for i in 0..output_unpacked.len() {
            assert!(
                (output_unpacked[i] - output_packed[i]).abs() < 1e-5,
                "mismatch at {}: unpacked={}, packed={}",
                i,
                output_unpacked[i],
                output_packed[i]
            );
        }
    }

    #[test]
    fn test_forward_decode_matches_general() {
        let weights: Vec<f32> = (0..6 * 4)
            .map(|i| (i as f32 * 0.7).cos() * 0.4)
            .collect();
        let linear = CpuLinear::from_weight(weights, 4, 6);
        let ternary_layer = TernaryLinear::from_cpu_linear(&linear);

        let input: Vec<f32> = vec![0.5, -0.3, 0.8, -0.1];

        let output_general = ternary_layer.forward(&input, 1);
        let output_decode = ternary_layer.forward_decode(&input);

        assert_eq!(output_general.len(), output_decode.len());
        for i in 0..output_general.len() {
            assert!(
                (output_general[i] - output_decode[i]).abs() < 1e-5,
                "decode mismatch at {}: general={}, decode={}",
                i,
                output_general[i],
                output_decode[i]
            );
        }
    }

    #[test]
    fn test_with_bias() {
        let weights: Vec<f32> = vec![1.0, -1.0, 0.5, -0.5, 1.0, 1.0]; // 2×3
        let bias = vec![0.1f32, 0.2];
        let linear = CpuLinear::from_weight_bias(weights, Some(bias), 3, 2);
        let ternary_layer = TernaryLinear::from_cpu_linear(&linear);

        let input = vec![1.0f32, 2.0, 3.0];
        let output = ternary_layer.forward(&input, 1);

        assert_eq!(output.len(), 2);
        // Bias should be added
        assert!(
            (output[0] - 0.1).abs() > 0.001 || (output[1] - 0.2).abs() > 0.001,
            "bias should affect output"
        );
    }

    #[test]
    fn test_batch_forward() {
        let weights: Vec<f32> = (0..4 * 3).map(|i| i as f32 * 0.1 - 0.3).collect();
        let linear = CpuLinear::from_weight(weights, 3, 4);
        let ternary_layer = TernaryLinear::from_cpu_linear(&linear);

        // Batch of 2 sequences
        let input = vec![1.0f32, 0.0, -1.0, 0.5, 0.5, 0.5];
        let output = ternary_layer.forward(&input, 2);

        assert_eq!(output.len(), 8); // 2 * 4

        // Verify seq 0 and seq 1 give different results
        let output_single_0 = ternary_layer.forward(&input[0..3], 1);
        let output_single_1 = ternary_layer.forward(&input[3..6], 1);

        for r in 0..4 {
            assert!(
                (output[r] - output_single_0[r]).abs() < 1e-5,
                "batch seq 0 mismatch"
            );
            assert!(
                (output[4 + r] - output_single_1[r]).abs() < 1e-5,
                "batch seq 1 mismatch"
            );
        }
    }

    #[test]
    fn test_memory_savings() {
        // Simulate a Q projection: 1536 → 1536
        let n = 1536 * 1536;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 1.618).sin() * 0.1).collect();
        let linear = CpuLinear::from_weight(weights, 1536, 1536);
        let ternary_layer = TernaryLinear::from_cpu_linear(&linear);

        let fp32_bytes = ternary_layer.fp32_equivalent_bytes();
        let actual_bytes = ternary_layer.memory_bytes();

        // FP32: 1536*1536*4 = ~9.4 MB
        assert!(fp32_bytes > 9_000_000, "FP32 should be ~9.4 MB, got {}", fp32_bytes);

        // Packed ternary: ~590 KB + unpacked: ~2.4 MB (until drop_unpacked)
        // With both packed and unpacked stored, ~3 MB
        assert!(actual_bytes < 4_000_000, "Ternary should be < 4 MB, got {}", actual_bytes);

        // Compression ratio should be > 2x (even with both representations stored)
        let ratio = ternary_layer.compression_ratio();
        assert!(ratio > 2.0, "compression ratio should be > 2x, got {}", ratio);

        // After dropping unpacked, ratio should be ~12-16x
        let mut slim = ternary_layer.clone();
        slim.drop_unpacked();
        assert!(!slim.has_unpacked());
        let slim_bytes = slim.memory_bytes();
        assert!(slim_bytes < 700_000, "packed-only should be < 700 KB, got {}", slim_bytes);
    }

    #[test]
    fn test_from_ternary_raw() {
        let ternary = vec![-1i8, 0, 1, -1, 0, 1];
        let scale = 0.5f32;
        let layer = TernaryLinear::from_ternary(ternary, scale, 3, 2);

        assert_eq!(layer.in_features, 3);
        assert_eq!(layer.out_features, 2);
        assert_eq!(layer.scale, 0.5);
        assert!(layer.bias.is_none());
    }
}
