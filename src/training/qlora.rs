//! QLoRA: Quantized-weight Low-Rank Adaptation.
//!
//! Implements training on quantized base weights (INT4) with LoRA adapters
//! in higher precision (F16/BF16). Based on Dettmers et al. 2023.
//!
//! How QLoRA works:
//! 1. Base model weights are stored in 4-bit (NF4 format)
//! 2. Forward pass: dequantize weights to F16/BF16, add LoRA adaptation
//! 3. Backward pass: only compute gradients for LoRA parameters
//! 4. Base weights stay frozen in 4-bit — massive memory savings
//!
//! Memory savings: a 7B model in FP16 = 14 GB, in NF4+QLoRA = ~3.5 GB + adapter overhead.

use crate::training::lora::{LoraConfig, LoraLayer};

// ---------------------------------------------------------------------------
// NF4 quantization — NormalFloat 4-bit (from QLoRA paper)
// ---------------------------------------------------------------------------

/// NF4 quantization: 4-bit normal float quantiles.
/// Maps f32 values to 16 quantile levels optimized for normally-distributed weights.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1850, -0.0929, 0.0000,
     0.0593,  0.1463,  0.2629,  0.3839,  0.5046,  0.6589,  0.8399, 1.0000,
];

/// Quantize an f32 value to NF4 (returns 4-bit index 0..15).
pub fn quantize_nf4(value: f32, scale: f32, zero_point: f32) -> u8 {
    let normalized = value / scale + zero_point;
    let mut best_idx = 0u8;
    let mut best_dist = f32::MAX;
    for (i, &level) in NF4_LEVELS.iter().enumerate() {
        let dist = (normalized - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Dequantize an NF4 4-bit index back to f32.
pub fn dequantize_nf4(index: u8, scale: f32, zero_point: f32) -> f32 {
    let level = NF4_LEVELS[index as usize];
    (level - zero_point) * scale
}

/// Quantize a block of f32 values to NF4 with block-wise scale.
/// Returns packed bytes (2 values per byte) and per-block scales.
pub fn quantize_nf4_block(data: &[f32], block_size: usize) -> (Vec<u8>, Vec<f32>) {
    let n_blocks = (data.len() + block_size - 1) / block_size;
    let mut packed = Vec::with_capacity((data.len() + 1) / 2);
    let mut scales = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(data.len());

        // Compute block scale and zero point
        let mut max_abs = 0.0f32;
        for &v in &data[start..end] {
            let abs = v.abs();
            if abs > max_abs { max_abs = abs; }
        }
        let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
        scales.push(scale);

        // Quantize pairs into bytes
        let mut i = start;
        while i + 1 < end {
            let lo = quantize_nf4(data[i], scale, 0.0);
            let hi = quantize_nf4(data[i + 1], scale, 0.0);
            packed.push(lo | (hi << 4));
            i += 2;
        }
        if i < end {
            let lo = quantize_nf4(data[i], scale, 0.0);
            packed.push(lo);
        }
    }

    (packed, scales)
}

/// Dequantize NF4 packed data back to f32.
pub fn dequantize_nf4_block(
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    total_elements: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(total_elements);
    let mut byte_idx = 0;

    for &scale in scales {
        let remaining = total_elements - out.len();
        let block_len = remaining.min(block_size);
        let mut i = 0;
        while i + 1 < block_len {
            let byte = packed.get(byte_idx).copied().unwrap_or(0);
            let lo = byte & 0x0F;
            let hi = (byte >> 4) & 0x0F;
            out.push(dequantize_nf4(lo, scale, 0.0));
            out.push(dequantize_nf4(hi, scale, 0.0));
            byte_idx += 1;
            i += 2;
        }
        if i < block_len {
            let byte = packed.get(byte_idx).copied().unwrap_or(0);
            let lo = byte & 0x0F;
            out.push(dequantize_nf4(lo, scale, 0.0));
            byte_idx += 1;
        }
    }

    out.truncate(total_elements);
    out
}

// ---------------------------------------------------------------------------
// Double quantization — quantize the scales themselves
// ---------------------------------------------------------------------------

/// Double quantization: quantize the NF4 block scales to INT8 with a second-level scale.
pub struct DoubleQuantizedScales {
    /// INT8 quantized scales.
    pub quantized_scales: Vec<i8>,
    /// Second-level scale factor.
    pub second_scale: f32,
    /// Second-level zero point.
    pub second_zero: f32,
}

impl DoubleQuantizedScales {
    /// Quantize f32 scales to INT8 with a global scale.
    pub fn quantize(scales: &[f32]) -> Self {
        let mut max_abs = 0.0f32;
        for &s in scales {
            let abs = s.abs();
            if abs > max_abs { max_abs = abs; }
        }
        let second_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let quantized: Vec<i8> = scales.iter().map(|&s| {
            let q = (s / second_scale).round() as i32;
            q.clamp(-128, 127) as i8
        }).collect();

        Self {
            quantized_scales: quantized,
            second_scale,
            second_zero: 0.0,
        }
    }

    /// Dequantize INT8 scales back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        self.quantized_scales.iter().map(|&q| q as f32 * self.second_scale).collect()
    }
}

// ---------------------------------------------------------------------------
// QuantizedLoraLayer — QLoRA layer combining NF4 base + LoRA adapter
// ---------------------------------------------------------------------------

/// A QLoRA layer: NF4 quantized base weights with LoRA adapter in F16.
pub struct QuantizedLoraLayer {
    /// NF4 packed weights (2 values per byte).
    packed_weights: Vec<u8>,
    /// Per-block scales for NF4.
    block_scales: Vec<f32>,
    /// Block size for NF4 quantization.
    block_size: usize,
    /// Original tensor dimensions.
    out_features: usize,
    in_features: usize,
    /// LoRA adapter in higher precision.
    lora: LoraLayer,
    /// Whether the adapter is merged into dequantized weights.
    merged: bool,
}

impl QuantizedLoraLayer {
    /// Create a new QLoRA layer from F32 weights.
    pub fn new(
        weights: &[f32],
        out_features: usize,
        in_features: usize,
        block_size: usize,
        lora_config: &LoraConfig,
    ) -> Self {
        let (packed, scales) = quantize_nf4_block(weights, block_size);
        let lora = LoraLayer::new(in_features, out_features, lora_config);
        Self {
            packed_weights: packed,
            block_scales: scales,
            block_size,
            out_features,
            in_features,
            lora,
            merged: false,
        }
    }

    /// Forward pass: dequantize NF4 → F16 + LoRA adaptation → matmul.
    ///
    /// 1. Dequantize base weights from NF4 to F32
    /// 2. Merge LoRA adapter (if not already merged)
    /// 3. Compute output = input × adapted_weights
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let _total = self.out_features * self.in_features;
        let dequant = self.dequantize_weights();

        // Compute base matmul: output[seq_len × out_features] = input[seq_len × in_features] × weights[in_features × out_features]
        let mut output = vec![0.0f32; seq_len * self.out_features];
        for s in 0..seq_len {
            for o in 0..self.out_features {
                let mut sum = 0.0f32;
                for i in 0..self.in_features {
                    sum += input[s * self.in_features + i] * dequant[o * self.in_features + i];
                }
                output[s * self.out_features + o] = sum;
            }
        }

        // Add LoRA adaptation
        if !self.merged {
            let lora_out = self.lora.forward(input, seq_len);
            for (i, v) in lora_out.iter().enumerate() {
                if i < output.len() {
                    output[i] += v;
                }
            }
        }

        output
    }

    /// Dequantize NF4 weights back to F32.
    pub fn dequantize_weights(&self) -> Vec<f32> {
        dequantize_nf4_block(
            &self.packed_weights,
            &self.block_scales,
            self.block_size,
            self.out_features * self.in_features,
        )
    }

    /// Merge LoRA adapter into dequantized weights (for deployment).
    pub fn merge(&mut self) {
        if self.merged { return; }
        let mut dequant = self.dequantize_weights();
        self.lora.merge_into(&mut dequant);
        // Re-quantize (note: this loses some precision — for deployment only)
        let (packed, scales) = quantize_nf4_block(&dequant, self.block_size);
        self.packed_weights = packed;
        self.block_scales = scales;
        self.merged = true;
    }

    /// Unmerge LoRA from weights.
    pub fn unmerge(&mut self) {
        if !self.merged { return; }
        let mut dequant = self.dequantize_weights();
        self.lora.unmerge_from(&mut dequant);
        let (packed, scales) = quantize_nf4_block(&dequant, self.block_size);
        self.packed_weights = packed;
        self.block_scales = scales;
        self.merged = false;
    }

    /// Get LoRA gradients (for optimizer step).
    pub fn lora_gradients(&mut self) -> (&mut [f32], &mut [f32]) {
        self.lora.gradients()
    }

    /// Zero LoRA gradients.
    pub fn zero_lora_grad(&mut self) {
        self.lora.zero_grad();
    }

    /// Memory savings vs FP16.
    pub fn memory_savings(&self) -> f32 {
        // FP16: 2 bytes per weight
        // NF4: 0.5 bytes per weight + scale overhead
        let total_weights = self.out_features * self.in_features;
        let fp16_bytes = total_weights * 2;
        let nf4_bytes = (total_weights + 1) / 2; // packed
        let scale_bytes = self.block_scales.len() * 4;
        let lora_bytes = self.lora.num_params() * 4;
        let total_qlora = nf4_bytes + scale_bytes + lora_bytes;
        fp16_bytes as f32 / total_qlora as f32
    }

    /// Total parameters (base + LoRA).
    pub fn total_params(&self) -> usize {
        self.out_features * self.in_features + self.lora.num_params()
    }

    /// Trainable parameters (LoRA only).
    pub fn trainable_params(&self) -> usize {
        self.lora.num_params()
    }

    /// Trainable fraction.
    pub fn trainable_fraction(&self) -> f32 {
        self.trainable_params() as f32 / self.total_params() as f32
    }

    /// Packed weights size in bytes.
    pub fn packed_size_bytes(&self) -> usize {
        self.packed_weights.len()
    }

    /// Whether adapter is merged.
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora.rank()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::lora::LoraConfig;

    #[test]
    fn test_nf4_round_trip() {
        // Quantize then dequantize — should be close
        let values: Vec<f32> = (-10..=10).map(|i| i as f32 / 10.0).collect();
        for &v in &values {
            let idx = quantize_nf4(v, 1.0, 0.0);
            let recovered = dequantize_nf4(idx, 1.0, 0.0);
            let err = (v - recovered).abs();
            assert!(err < 0.3, "NF4 error too large: {} → {} (err={})", v, recovered, err);
        }
    }

    #[test]
    fn test_nf4_symmetry() {
        // NF4 is symmetric around 0
        let pos = dequantize_nf4(quantize_nf4(0.5, 1.0, 0.0), 1.0, 0.0);
        let neg = dequantize_nf4(quantize_nf4(-0.5, 1.0, 0.0), 1.0, 0.0);
        assert!((pos + neg).abs() < 0.3); // Roughly symmetric
    }

    #[test]
    fn test_nf4_block_round_trip() {
        let data: Vec<f32> = (0..256).map(|i| ((i as f32 - 128.0) / 128.0).sin()).collect();
        let (packed, scales) = quantize_nf4_block(&data, 64);
        let recovered = dequantize_nf4_block(&packed, &scales, 64, 256);

        assert_eq!(recovered.len(), 256);
        // Check reasonable accuracy
        let mut total_err = 0.0f32;
        for (orig, &rec) in data.iter().zip(recovered.iter()) {
            total_err += (orig - rec).abs();
        }
        let mean_err = total_err / data.len() as f32;
        assert!(mean_err < 0.1, "Mean NF4 error: {}", mean_err);
    }

    #[test]
    fn test_nf4_block_sizes() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();

        for bs in [32, 64, 128] {
            let (packed, scales) = quantize_nf4_block(&data, bs);
            let recovered = dequantize_nf4_block(&packed, &scales, bs, 100);
            assert_eq!(recovered.len(), 100);
        }
    }

    #[test]
    fn test_double_quantization() {
        let scales: Vec<f32> = vec![0.5, 1.0, 1.5, 2.0, 0.1];
        let dq = DoubleQuantizedScales::quantize(&scales);
        let recovered = dq.dequantize();

        assert_eq!(recovered.len(), 5);
        // INT8 quantization of scales should be fairly accurate
        for (orig, &rec) in scales.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            assert!(err < 0.05, "Double quant error: {} vs {} (err={})", orig, rec, err);
        }
    }

    #[test]
    fn test_qlora_layer_forward() {
        let lora_config = LoraConfig::new(4);
        let weights: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let layer = QuantizedLoraLayer::new(&weights, 8, 8, 32, &lora_config);

        let input = vec![1.0f32; 4 * 8]; // seq_len=4, in_features=8
        let output = layer.forward(&input, 4);
        assert_eq!(output.len(), 4 * 8); // seq_len × out_features
    }

    #[test]
    fn test_qlora_layer_memory_savings() {
        let lora_config = LoraConfig::new(4);
        let weights = vec![0.5f32; 1024 * 1024]; // 1M weights
        let layer = QuantizedLoraLayer::new(&weights, 1024, 1024, 64, &lora_config);

        let savings = layer.memory_savings();
        assert!(savings > 2.0, "QLoRA should save >2x memory, got {}x", savings);
    }

    #[test]
    fn test_qlora_layer_trainable_fraction() {
        let lora_config = LoraConfig::new(8);
        let weights = vec![0.5f32; 4096]; // 64×64
        let layer = QuantizedLoraLayer::new(&weights, 64, 64, 32, &lora_config);

        let frac = layer.trainable_fraction();
        // rank=8, 64×8×2 = 1024 params out of 4096+1024=5120
        assert!(frac < 0.3, "Trainable fraction should be <30%, got {}%", frac * 100.0);
        assert!(frac > 0.0);
    }

    #[test]
    fn test_qlora_merge_unmerge() {
        let lora_config = LoraConfig::new(4);
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mut layer = QuantizedLoraLayer::new(&weights, 8, 8, 32, &lora_config);

        assert!(!layer.is_merged());
        layer.merge();
        assert!(layer.is_merged());
        layer.unmerge();
        assert!(!layer.is_merged());
    }

    #[test]
    fn test_qlora_packed_size() {
        let lora_config = LoraConfig::new(4);
        let weights = vec![0.5f32; 256];
        let layer = QuantizedLoraLayer::new(&weights, 16, 16, 64, &lora_config);

        // 256 f32 values → 128 bytes packed (2 per byte)
        assert_eq!(layer.packed_size_bytes(), 128);
    }

    #[test]
    fn test_qlora_gradients() {
        let lora_config = LoraConfig::new(4);
        let weights = vec![0.5f32; 128];
        let mut layer = QuantizedLoraLayer::new(&weights, 16, 8, 32, &lora_config);

        let (grad_a, grad_b) = layer.lora_gradients();
        assert_eq!(grad_a.len(), 8 * 4); // in_features × rank
        assert_eq!(grad_b.len(), 16 * 4); // out_features × rank

        layer.zero_lora_grad();
        let (grad_a, grad_b) = layer.lora_gradients();
        assert!(grad_a.iter().all(|&g| g == 0.0));
        assert!(grad_b.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn test_qlora_vs_fp16_size() {
        let weights = vec![0.5f32; 4096];
        let lora_config = LoraConfig::new(4);
        let layer = QuantizedLoraLayer::new(&weights, 64, 64, 64, &lora_config);

        let fp16_bytes = 4096 * 2; // 8192 bytes
        let qlora_bytes = layer.packed_size_bytes()
            + layer.block_scales.len() * 4
            + layer.trainable_params() * 4;

        assert!(qlora_bytes < fp16_bytes, "QLoRA should use less memory than FP16");
    }
}
