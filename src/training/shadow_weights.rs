//! Shadow weight infrastructure for STE (Straight-Through Estimator) training.
//!
//! Implements the "Master Weight" pattern:
//!   - Forward pass: ternary base weights {-1, 0, +1} with absmean scale
//!   - Backward pass: high-precision shadow weights (BF16 or FP32)
//!   - Gradients/optimizer: always FP32 for accumulation stability
//!
//! Memory layout (2-expert MoE, 2.66B params):
//!   - Ternary base:  ~2.1 GB (always resident)
//!   - BF16 shadows:  ~8.5 GB (training only, dropped after)
//!   - FP32 gradients: ~8.5 GB (transient per backward pass)
//!   - FP32 optimizer: ~8.5 GB (persistent during training)
//!   Total training:  ~27.6 GB — fits on 32 GB systems
//!
//! After training: drop shadows + optimizer → 2.1 GB deployment model.

use half::bf16;

/// Abstraction over shadow weight precision.
///
/// Enables switching between BF16 (production, 50% memory savings)
/// and FP32 (debugging, no precision loss) without changing training code.
pub trait ShadowPrecision: Copy + Default + Send + Sync + 'static {
    /// Convert from FP32 (used for gradient accumulation).
    fn from_f32(val: f32) -> Self;
    /// Convert back to FP32 (for gradient computation).
    fn to_f32(self) -> f32;
    /// Size in bytes per value.
    const SIZE_BYTES: usize;
}

impl ShadowPrecision for f32 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        val
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    const SIZE_BYTES: usize = 4;
}

impl ShadowPrecision for bf16 {
    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        bf16::from_f32(val)
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    const SIZE_BYTES: usize = 2;
}

/// BitLinear: ternary base weights with optional shadow weights for STE training.
///
/// This is the future replacement for `CpuLinear` in STE mode:
///   - Inference: only `ternary` + `scale` are used (shadow = None)
///   - Training (STE): shadow weights track the continuous-space version
///     of the weights, quantized to ternary for forward pass
///   - Training (LoRA): shadow = None, LoRA adapters handle the delta
///
/// The quantization step is:
///   ```text
///   forward:  W_ternary = round(W_shadow / α) ∈ {-1, 0, +1}
///   backward: ∂L/∂W_shadow ≈ ∂L/∂W_ternary  (STE identity)
///   update:   W_shadow -= lr × ∂L/∂W_shadow
///   ```
#[derive(Debug)]
pub struct BitLinear<P: ShadowPrecision = bf16> {
    /// Ternary base weights {-1, 0, +1}: [out_features × in_features].
    /// Always present — this is the "deployed" model.
    pub ternary: Vec<i8>,

    /// Packed 2-bit ternary (4 values/byte) for decode-optimized forward.
    pub packed: Vec<u8>,

    /// Absmean scale factor: α = mean(|W|) / sqrt(2/π).
    pub scale: f32,

    /// Shadow weights in high precision: [out_features × in_features].
    /// Only present during STE training. None for inference or LoRA-only training.
    /// BF16 = 50% memory savings vs FP32, same dynamic range.
    pub shadow: Option<Vec<P>>,

    /// Optional bias [out_features].
    pub bias: Option<Vec<f32>>,

    pub in_features: usize,
    pub out_features: usize,
}

impl<P: ShadowPrecision> BitLinear<P> {
    /// Create a zero-initialized BitLinear (all ternary = 0, no shadow).
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let ternary = vec![0i8; in_features * out_features];
        let packed = crate::model::ternary::pack_ternary(&ternary);
        Self {
            ternary,
            packed,
            scale: 1.0,
            shadow: None,
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Create from FP32 weights — quantizes to ternary, optionally stores shadow.
    pub fn from_fp32(weights: &[f32], in_features: usize, out_features: usize, keep_shadow: bool) -> Self {
        let (ternary, scale) = crate::model::ternary::quantize_ternary(weights);
        let packed = crate::model::ternary::pack_ternary(&ternary);
        let shadow = if keep_shadow {
            Some(weights.iter().map(|&w| P::from_f32(w)).collect())
        } else {
            None
        };
        Self {
            ternary,
            packed,
            scale,
            shadow,
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Initialize shadow weights from current ternary (dequantized).
    /// Used when starting STE from an existing ternary model.
    pub fn init_shadow_from_ternary(&mut self) {
        if self.shadow.is_some() {
            return; // Already has shadow
        }
        let shadow: Vec<P> = self
            .ternary
            .iter()
            .map(|&t| P::from_f32(t as f32 * self.scale))
            .collect();
        self.shadow = Some(shadow);
    }

    /// Drop shadow weights — convert to inference-only mode.
    /// Saves 50% (BF16) or 75% (vs FP32 shadow) memory.
    pub fn drop_shadow(&mut self) {
        self.shadow = None;
    }

    /// Re-quantize shadow weights to ternary (STE forward step).
    /// Call after each optimizer step to keep ternary in sync.
    pub fn requantize(&mut self) {
        if let Some(ref shadow) = self.shadow {
            let fp32: Vec<f32> = shadow.iter().map(|&s| s.to_f32()).collect();
            let (new_ternary, new_scale) = crate::model::ternary::quantize_ternary(&fp32);
            self.ternary = new_ternary;
            self.scale = new_scale;
            self.packed = crate::model::ternary::pack_ternary(&self.ternary);
        }
    }

    /// Memory usage in bytes (excluding shadow weights).
    pub fn base_memory_bytes(&self) -> usize {
        self.ternary.len() + self.packed.len() + std::mem::size_of::<f32>() // scale
            + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }

    /// Memory usage of shadow weights only.
    pub fn shadow_memory_bytes(&self) -> usize {
        self.shadow.as_ref().map_or(0, |s| s.len() * P::SIZE_BYTES)
    }

    /// Total memory usage in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.base_memory_bytes() + self.shadow_memory_bytes()
    }

    /// Whether this layer is in STE training mode (has shadow weights).
    pub fn is_ste(&self) -> bool {
        self.shadow.is_some()
    }

    /// === Forward pass methods ===

    /// Unified forward: routes to STE or inference path based on shadow presence.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        match &self.shadow {
            Some(s) => self.forward_ste(input, s, seq_len),
            None => self.forward_ternary(input, seq_len),
        }
    }

    /// STE forward: quantize shadow → ternary on-the-fly, then ternary matmul.
    ///
    /// This is the "virtual" forward pass — shadow weights are quantized to
    /// {-1, 0, +1} using the current scale α, but the real weights (shadows)
    /// are updated by gradients from this pass via STE.
    ///
    /// Scale α is NOT a learnable parameter. It's recomputed from shadow weights:
    ///   α = mean(|W_shadow|)
    /// This is the BitNet b1.58 standard — the quantization grid adapts
    /// as weights evolve, preventing drift.
    fn forward_ste(&self, input: &[f32], shadow: &[P], seq_len: usize) -> Vec<f32> {
        // On-the-fly quantization: shadow → ternary using current α
        let current_alpha = {
            let sum: f32 = shadow.iter().map(|&s| s.to_f32().abs()).sum();
            sum / shadow.len() as f32
        };

        let on_the_fly_ternary: Vec<i8> = shadow
            .iter()
            .map(|&s| {
                let v = s.to_f32();
                if current_alpha < 1e-10 {
                    0
                } else {
                    let normalized = v / current_alpha;
                    if normalized > 0.5 { 1 } else if normalized < -0.5 { -1 } else { 0 }
                }
            })
            .collect();

        crate::model::ternary::ternary_matmul(
            &on_the_fly_ternary,
            current_alpha,
            input,
            self.out_features,
            self.in_features,
            seq_len,
        )
    }

    /// Inference/LoRA forward: use stored ternary weights directly.
    fn forward_ternary(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        crate::model::ternary::ternary_matmul(
            &self.ternary,
            self.scale,
            input,
            self.out_features,
            self.in_features,
            seq_len,
        )
    }

    /// Recompute scale α from shadow weights (running average).
    ///
    /// Call this AFTER each optimizer step to keep the quantization grid
    /// aligned with the current weight distribution.
    ///
    /// α = mean(|W_shadow|)
    ///
    /// This is NOT a learnable parameter — it's derived from the shadow weights.
    /// As weights evolve, α adapts, preventing the quantization grid from
    /// becoming misaligned (the "drift" problem).
    pub fn recompute_scale(&mut self) {
        if let Some(ref shadow) = self.shadow {
            let sum: f32 = shadow.iter().map(|&s| s.to_f32().abs()).sum();
            self.scale = sum / shadow.len() as f32;
        }
    }

    /// Inject noise into shadow weights to escape the "zero-point dead zone".
    ///
    /// With BF16 shadow weights, gradients can be too small to move a weight
    /// across the quantization threshold (e.g., 0.49α → 0.51α). The weight
    /// gets "stuck" at 0 and never recovers.
    ///
    /// Solution: add small Gaussian noise to shadow weights near the
    /// quantization boundaries. This "kicks" stuck weights into an active
    /// state with some probability.
    ///
    /// Parameters:
    /// - `noise_scale`: magnitude relative to α. Typical: 0.01–0.05.
    /// - `boundary_zone`: how close to a boundary (in units of α) to apply
    ///   noise. Typical: 0.3 (applies to weights within 0.3α of a boundary).
    pub fn inject_boundary_noise(&mut self, noise_scale: f32, boundary_zone: f32) {
        if let Some(ref mut shadow) = self.shadow {
            let alpha = self.scale;
            if alpha < 1e-10 { return; }

            for (i, s) in shadow.iter_mut().enumerate() {
                let v = s.to_f32();
                let normalized = v / alpha;

                // Distance to nearest boundary (0.0 or ±0.5 in normalized space)
                let dist_to_zero = normalized.abs();
                let dist_to_half = (normalized.abs() - 0.5).abs();
                let min_dist = dist_to_zero.min(dist_to_half);

                if min_dist < boundary_zone {
                    // Weight is near a quantization boundary — inject noise
                    // Simple deterministic noise from index (no rand crate needed)
                    let noise_seed = ((i as u64).wrapping_mul(0x517cc1b727220a95)) >> 33;
                    let noise = ((noise_seed as f32 / u32::MAX as f32) - 0.5) * 2.0 * noise_scale * alpha;
                    *s = P::from_f32(v + noise);
                }
            }
        }
    }

    /// Apply stochastic rounding to shadow weights after an optimizer step.
    ///
    /// Prevents the "frozen weight" problem with BF16 where tiny gradient
    /// updates round to zero. For each weight, if the update would be lost
    /// to rounding, we randomly round up with probability equal to the
    /// fractional part.
    ///
    /// Only meaningful when P = bf16. No-op for f32.
    pub fn stochastic_round_update(&mut self, gradients: &[f32], lr: f32) {
        if let Some(ref mut shadow) = self.shadow {
            for (i, g) in gradients.iter().enumerate() {
                let old = shadow[i].to_f32();
                let update = lr * g;

                // Exact new value in FP32
                let exact_new = old - update;

                // Convert to target precision and back
                let rounded = P::from_f32(exact_new);
                let rounded_f32 = rounded.to_f32();

                // If rounding lost information, apply stochastic rounding
                if rounded_f32 != exact_new && P::SIZE_BYTES < 4 {
                    // Use simple xorshift PRNG (deterministic per weight)
                    let frac = (exact_new - rounded_f32).abs();
                    let threshold = frac / (rounded_f32 - exact_new).abs().max(1e-10);
                    // Pseudo-random: use index as seed (deterministic but sufficient)
                    let rand_val = ((i as u64).wrapping_mul(6364136223846793005) + 1442695040888963407) as u32;
                    let rand_f = (rand_val >> 8) as f32 / (1u32 << 24) as f32;
                    if rand_f < threshold {
                        // Round up instead
                        let step = if exact_new > rounded_f32 { 1.0 } else { -1.0 };
                        shadow[i] = P::from_f32(rounded_f32 + step * P::from_f32(1.0).to_f32().max(1e-7));
                    } else {
                        shadow[i] = rounded;
                    }
                } else {
                    shadow[i] = rounded;
                }
            }
        }
    }
}

impl BitLinear<bf16> {
    /// Convenience: create BitLinear with BF16 shadow weights from FP32.
    /// This is the standard production path for STE training.
    pub fn from_fp32_bf16_shadow(weights: &[f32], in_features: usize, out_features: usize) -> Self {
        Self::from_fp32(weights, in_features, out_features, true)
    }
}

impl BitLinear<f32> {
    /// Convenience: create BitLinear with FP32 shadow weights (debug mode).
    pub fn from_fp32_debug(weights: &[f32], in_features: usize, out_features: usize) -> Self {
        Self::from_fp32(weights, in_features, out_features, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        let val = 0.5f32;
        let bf = bf16::from_f32(val);
        let back = bf.to_f32();
        assert!((val - back).abs() < 0.01, "BF16 roundtrip should be close");
    }

    #[test]
    fn test_bitlinear_from_fp32_no_shadow() {
        let w: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 4, 4, false);
        assert!(bl.shadow.is_none());
        assert_eq!(bl.ternary.len(), 16);
        assert!(bl.scale > 0.0);
    }

    #[test]
    fn test_bitlinear_from_fp32_with_shadow() {
        let w: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 4, 4, true);
        assert!(bl.shadow.is_some());
        let shadow = bl.shadow.unwrap();
        assert_eq!(shadow.len(), 16);
        // Shadow should preserve approximate values
        assert!((shadow[8].to_f32() - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_requantize_preserves_ternary() {
        let w: Vec<f32> = vec![5.0, -3.0, 0.1, -0.1, 2.0, -2.0, 0.0, 1.0];
        let mut bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 2, 4, true);

        // Re-quantize (should produce valid ternary)
        bl.requantize();

        // Large values should preserve sign
        assert_eq!(bl.ternary[0], 1);  // 5.0 → +1
        assert_eq!(bl.ternary[1], -1); // -3.0 → -1

        // All values must be valid ternary
        for (i, &t) in bl.ternary.iter().enumerate() {
            assert!(t == -1 || t == 0 || t == 1,
                "ternary[{}] = {} must be in {{-1, 0, +1}}", i, t);
        }
    }

    #[test]
    fn test_init_shadow_from_ternary() {
        let mut bl: BitLinear<bf16> = BitLinear::new(4, 4);
        assert!(bl.shadow.is_none());
        bl.init_shadow_from_ternary();
        assert!(bl.shadow.is_some());
        // All ternary are 0, so shadow should be ~0
        let shadow = bl.shadow.unwrap();
        for s in shadow {
            assert!(s.to_f32().abs() < 1e-6);
        }
    }

    #[test]
    fn test_drop_shadow() {
        let w: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 2, 4, true);
        assert!(bl.shadow.is_some());
        bl.drop_shadow();
        assert!(bl.shadow.is_none());
    }

    #[test]
    fn test_memory_savings() {
        let n = 1536 * 1536; // Typical attention projection
        let w = vec![1.0f32; n];

        let bl_bf16: BitLinear<bf16> = BitLinear::from_fp32(&w, 1536, 1536, true);
        let bl_f32: BitLinear<f32> = BitLinear::from_fp32(&w, 1536, 1536, true);

        let bf16_shadow = bl_bf16.shadow_memory_bytes();
        let f32_shadow = bl_f32.shadow_memory_bytes();

        assert_eq!(bf16_shadow, n * 2, "BF16 shadow: 2 bytes per value");
        assert_eq!(f32_shadow, n * 4, "FP32 shadow: 4 bytes per value");
        assert!(
            bf16_shadow * 2 == f32_shadow,
            "BF16 should be exactly 50% of FP32"
        );
    }

    #[test]
    fn test_stochastic_rounding_doesnt_crash() {
        let w: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let mut bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 4, 4, true);
        let grads = vec![0.001f32; 16]; // Tiny gradients
        bl.stochastic_round_update(&grads, 1e-4);
        // Should not panic, shadow should still be valid
        assert!(bl.shadow.is_some());
    }

    #[test]
    fn test_forward_inference_path() {
        // Inference mode (no shadow): uses stored ternary directly
        let w: Vec<f32> = vec![5.0, -3.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
        let bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 2, 4, false);
        let input = vec![1.0, 2.0];
        let output = bl.forward(&input, 1);
        assert_eq!(output.len(), 4, "Output should be [out_features]");
        // Should produce non-zero output
        assert!(output.iter().any(|&x| x.abs() > 0.01), "Output should be non-zero");
    }

    #[test]
    fn test_forward_ste_path() {
        // STE mode (with shadow): on-the-fly quantization from shadow
        let w: Vec<f32> = vec![5.0, -3.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
        let bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 2, 4, true);
        let input = vec![1.0, 2.0];
        let output = bl.forward(&input, 1);
        assert_eq!(output.len(), 4);
        assert!(output.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_recompute_scale_from_shadow() {
        let w: Vec<f32> = vec![2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0];
        let mut bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 2, 4, true);

        // Initial scale from quantization
        let initial_scale = bl.scale;

        // Modify shadow weights (simulate optimizer step)
        if let Some(ref mut shadow) = bl.shadow {
            for s in shadow.iter_mut() {
                *s = bf16::from_f32(s.to_f32() * 2.0); // Double all weights
            }
        }

        // Recompute scale
        bl.recompute_scale();

        // Scale should have increased (doubled weights → doubled mean)
        assert!(
            bl.scale > initial_scale * 1.5,
            "Scale should increase after doubling weights: was {}, now {}",
            initial_scale, bl.scale
        );
    }

    #[test]
    fn test_boundary_noise_injection() {
        let w: Vec<f32> = vec![0.1, -0.1, 0.1, -0.1]; // Near zero (boundary)
        let mut bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 2, 2, true);

        let shadow_before: Vec<f32> = bl.shadow.as_ref().unwrap().iter().map(|s| s.to_f32()).collect();
        bl.inject_boundary_noise(0.05, 0.3);
        let shadow_after: Vec<f32> = bl.shadow.as_ref().unwrap().iter().map(|s| s.to_f32()).collect();

        // At least some weights should have been perturbed (they're near boundaries)
        let changed = shadow_before.iter().zip(shadow_after.iter())
            .filter(|(&a, &b)| (a - b).abs() > 1e-10)
            .count();
        assert!(changed > 0, "Boundary noise should perturb some weights near zero");
    }

    #[test]
    fn test_ste_requantize_cycle() {
        // Full STE cycle: init → shadow update → recompute scale → requantize
        let w: Vec<f32> = vec![5.0, -3.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
        let mut bl: BitLinear<bf16> = BitLinear::from_fp32(&w, 2, 4, true);

        // Simulate: update shadow weights (gradient step)
        if let Some(ref mut shadow) = bl.shadow {
            for s in shadow.iter_mut() {
                *s = bf16::from_f32(s.to_f32() + 0.1); // Small update
            }
        }

        // Recompute scale from updated shadows
        bl.recompute_scale();

        // Re-quantize ternary from updated shadows
        bl.requantize();

        // Ternary should still be valid {-1, 0, +1}
        for &t in &bl.ternary {
            assert!(t == -1 || t == 0 || t == 1, "Ternary must be in {{-1, 0, +1}}, got {}", t);
        }
    }
}
