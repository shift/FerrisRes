//! STE-enabled MoE layer with per-block synchronized scale (α).
//!
//! Key architectural decision: all experts in a MoE layer share the same α.
//! This ensures the softmax-weighted expert combination is meaningful —
//! without synchronized α, an expert with larger scale would dominate the
//! weighted sum regardless of routing probability.
//!
//! Layout per layer (4 experts, hidden=1536, inter=12288):
//!   - Ternary base:    4 × 3 × (12288×1536) × 1 byte = ~226 MB
//!   - BF16 shadows:    4 × 3 × (12288×1536) × 2 bytes = ~452 MB
//!   - Router (FP32):   1536 × 4 × 4 bytes = ~24 KB
//!   Total per layer:   ~678 MB training, ~226 MB inference

use half::bf16;
use crate::model::cpu_moe::TernaryExpert;
use crate::model::ternary::ternary_matmul;

/// Scale synchronization strategy for a MoE layer's experts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleSync {
    /// All experts in the layer share the same α (recommended).
    /// Computed as mean of all expert shadow weights' absolute values.
    /// Ensures fair softmax-weighted expert combination.
    PerBlock,
    /// Each expert has its own α. More flexible but can cause unfair routing —
    /// experts with larger α dominate the weighted output.
    PerExpert,
}

/// STE-enabled MoE layer with synchronized scale and BF16 shadow weights.
///
/// This is the future MoE implementation for STE training. Current training
/// uses `CpuMoELayer` with LoRA adapters. This struct will replace it when
/// STE is fully implemented.
///
/// The per-block synchronized α prevents routing bias:
///   - If expert 0 has α=2.0 and expert 1 has α=0.5, expert 0's output
///     is 4× larger regardless of routing probability → unfair.
///   - With synchronized α, both experts use the same quantization grid.
///   - Expert specialization comes from the ternary pattern, not scale.
#[derive(Debug)]
pub struct BitMoELayer {
    /// Router weights: [hidden_dim × num_experts]. FP32 — trainable.
    pub gate_weights: Vec<f32>,

    /// Expert gate projections (ternary base + optional BF16 shadow).
    pub expert_gate: Vec<BitExpert>,
    /// Expert up projections (ternary base + optional BF16 shadow).
    pub expert_up: Vec<BitExpert>,
    /// Expert down projections (ternary base + optional BF16 shadow).
    pub expert_down: Vec<BitExpert>,

    /// PLE per-layer projections (included in block-wide α for synchronized
    /// quantization grid). PLE + experts share the same scale because
    /// Gemma 4 E2B's "Effective 2B" architecture adapts representation
    /// intensity to layer depth — separating α would break this.
    pub ple_input_gate: Option<BitExpert>,
    pub ple_projection: Option<BitExpert>,

    /// Per-block synchronized scale factor.
    /// All experts AND PLE projections in this layer share the same α.
    /// Recomputed from shadow weights each training step.
    pub alpha: f32,

    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub use_gelu: bool,

    /// How scale is synchronized across experts.
    pub scale_sync: ScaleSync,
}

/// A single expert with ternary base + optional BF16 shadow weights.
///
/// In inference mode: only `ternary` and `scale` are used.
/// In STE training: `shadow` tracks the continuous-space weights.
#[derive(Debug, Clone)]
pub struct BitExpert {
    /// Ternary base weights {-1, 0, +1}: [rows × cols].
    pub ternary: Vec<i8>,
    /// Packed 2-bit weights for decode-optimized forward.
    pub packed: Vec<u8>,
    /// Absmean scale factor.
    pub scale: f32,
    /// BF16 shadow weights for STE training. None in inference.
    pub shadow: Option<Vec<bf16>>,
    /// Matrix dimensions.
    pub rows: usize,
    pub cols: usize,
}

impl BitExpert {
    /// Create from existing TernaryExpert (no shadow weights).
    pub fn from_ternary_expert(te: &TernaryExpert) -> Self {
        Self {
            ternary: te.values.clone(),
            packed: te.packed.clone(),
            scale: te.scale,
            shadow: None,
            rows: te.rows,
            cols: te.cols,
        }
    }

    /// Create from FP32 weights, with optional shadow.
    pub fn from_fp32(weights: &[f32], rows: usize, cols: usize, keep_shadow: bool) -> Self {
        let (ternary, scale) = crate::model::ternary::quantize_ternary(weights);
        let packed = crate::model::ternary::pack_ternary(&ternary);
        let shadow = if keep_shadow {
            Some(weights.iter().map(|&w| bf16::from_f32(w)).collect())
        } else {
            None
        };
        Self { ternary, packed, scale, shadow, rows, cols }
    }

    /// Forward using stored ternary (inference/LoRA mode).
    pub fn forward_ternary(&self, input: &[f32], seq: usize) -> Vec<f32> {
        ternary_matmul(&self.ternary, self.scale, input, self.rows, self.cols, seq)
    }

    /// Forward using on-the-fly quantization from shadow (STE mode).
    /// Uses the layer-level α, not the expert's own scale.
    pub fn forward_ste(&self, input: &[f32], seq: usize, layer_alpha: f32) -> Vec<f32> {
        if let Some(ref shadow) = self.shadow {
            // On-the-fly quantize shadow → ternary using layer α
            let q: Vec<i8> = shadow.iter().map(|&s| {
                let v = s.to_f32();
                if layer_alpha < 1e-10 { 0 }
                else {
                    let n = v / layer_alpha;
                    if n > 0.5 { 1 } else if n < -0.5 { -1 } else { 0 }
                }
            }).collect();
            ternary_matmul(&q, layer_alpha, input, self.rows, self.cols, seq)
        } else {
            self.forward_ternary(input, seq)
        }
    }

    /// Memory usage (bytes) for ternary base only.
    pub fn base_memory(&self) -> usize {
        self.ternary.len() + self.packed.len()
    }

    /// Memory usage (bytes) for shadow weights only.
    pub fn shadow_memory(&self) -> usize {
        self.shadow.as_ref().map_or(0, |s| s.len() * 2) // bf16 = 2 bytes
    }
}

impl BitMoELayer {
    /// Create from existing CpuMoELayer (inference mode, no shadows).
    pub fn from_cpu_moe(moe: &crate::model::cpu_moe::CpuMoELayer) -> Self {
        Self {
            gate_weights: moe.gate_weights.clone(),
            expert_gate: moe.expert_gate.iter().map(BitExpert::from_ternary_expert).collect(),
            expert_up: moe.expert_up.iter().map(BitExpert::from_ternary_expert).collect(),
            expert_down: moe.expert_down.iter().map(BitExpert::from_ternary_expert).collect(),
            ple_input_gate: None,  // PLE comes from CpuBlockAttnResLayer, not CpuMoELayer
            ple_projection: None,
            alpha: 1.0, // Will be computed from shadows when STE starts
            hidden_dim: moe.hidden_dim,
            intermediate_dim: moe.intermediate_dim,
            num_experts: moe.num_experts,
            top_k: moe.top_k,
            use_gelu: moe.use_gelu,
            scale_sync: ScaleSync::PerBlock,
        }
    }

    /// Initialize shadow weights for all experts from current ternary.
    /// Call once before starting STE training.
    pub fn init_shadows_from_ternary(&mut self) {
        let experts: Vec<&mut Vec<BitExpert>> = vec![
            &mut self.expert_gate,
            &mut self.expert_up,
            &mut self.expert_down,
        ];
        for expert_list in experts {
            for expert in expert_list.iter_mut() {
                if expert.shadow.is_some() { continue; }
                let shadow: Vec<bf16> = expert.ternary.iter()
                    .map(|&t| bf16::from_f32(t as f32 * expert.scale))
                    .collect();
                expert.shadow = Some(shadow);
            }
        }
        // Also init PLE projection shadows (part of block-wide α)
        for ple in [&mut self.ple_input_gate, &mut self.ple_projection].iter_mut() {
            if let Some(ref mut expert) = ple {
                if expert.shadow.is_none() {
                    let shadow: Vec<bf16> = expert.ternary.iter()
                        .map(|&t| bf16::from_f32(t as f32 * expert.scale))
                        .collect();
                    expert.shadow = Some(shadow);
                }
            }
        }
        self.recompute_alpha();
    }

    /// Recompute the per-block synchronized α from all expert shadow weights.
    ///
    /// α = mean(|W_shadow|) across ALL experts in this layer.
    /// This ensures the quantization grid is consistent — expert specialization
    /// comes from the ternary pattern, not scale differences.
    pub fn recompute_alpha(&mut self) {
        let all_experts: Vec<&Vec<BitExpert>> = vec![&self.expert_gate, &self.expert_up, &self.expert_down];
        let mut total_abs: f64 = 0.0;
        let mut count: usize = 0;

        for expert_list in &all_experts {
            for expert in expert_list.iter() {
                if let Some(ref shadow) = expert.shadow {
                    for s in shadow.iter() {
                        total_abs += s.to_f32().abs() as f64;
                        count += 1;
                    }
                }
            }
        }

        // Include PLE projections in block-wide α reduction.
        // Gemma 4 E2B's PLE adapts representation intensity to layer depth —
        // separating α would break the "Effective 2B" scaling behavior.
        for ple in [&self.ple_input_gate, &self.ple_projection].iter().filter_map(|&x| x.as_ref()) {
            if let Some(ref shadow) = ple.shadow {
                for s in shadow.iter() {
                    total_abs += s.to_f32().abs() as f64;
                    count += 1;
                }
            }
        }

        if count > 0 {
            self.alpha = (total_abs / count as f64) as f32;
        }
    }

    /// Re-quantize all experts from their shadow weights.
    /// Call after each optimizer step to keep ternary in sync.
    ///
    /// With PerBlock sync: all experts use the same α.
    /// With PerExpert sync: each expert uses its own α.
    pub fn requantize_all(&mut self) {
        match self.scale_sync {
            ScaleSync::PerBlock => {
                // Use synchronized α for all experts
                let alpha = self.alpha;
                for expert in self.expert_gate.iter_mut().chain(self.expert_up.iter_mut()).chain(self.expert_down.iter_mut()) {
                    if let Some(ref shadow) = expert.shadow {
                        expert.ternary = shadow.iter().map(|&s| {
                            let v = s.to_f32();
                            if alpha < 1e-10 { 0 }
                            else {
                                let n = v / alpha;
                                if n > 0.5 { 1 } else if n < -0.5 { -1 } else { 0 }
                            }
                        }).collect();
                        expert.scale = alpha;
                        expert.packed = crate::model::ternary::pack_ternary(&expert.ternary);
                    }
                }
            }
            ScaleSync::PerExpert => {
                // Each expert recomputes its own α
                for expert in self.expert_gate.iter_mut().chain(self.expert_up.iter_mut()).chain(self.expert_down.iter_mut()) {
                    if let Some(ref shadow) = expert.shadow {
                        let sum: f32 = shadow.iter().map(|s| s.to_f32().abs()).sum();
                        let local_alpha = sum / shadow.len() as f32;
                        expert.ternary = shadow.iter().map(|&s| {
                            let v = s.to_f32();
                            if local_alpha < 1e-10 { 0 }
                            else {
                                let n = v / local_alpha;
                                if n > 0.5 { 1 } else if n < -0.5 { -1 } else { 0 }
                            }
                        }).collect();
                        expert.scale = local_alpha;
                        expert.packed = crate::model::ternary::pack_ternary(&expert.ternary);
                    }
                }
            }
        }
    }

    /// Drop all shadow weights — convert to inference-only mode.
    pub fn drop_shadows(&mut self) {
        for expert in self.expert_gate.iter_mut().chain(self.expert_up.iter_mut()).chain(self.expert_down.iter_mut()) {
            expert.shadow = None;
        }
    }

    /// Total shadow memory across all experts (bytes).
    pub fn total_shadow_memory(&self) -> usize {
        let experts: Vec<&Vec<BitExpert>> = vec![&self.expert_gate, &self.expert_up, &self.expert_down];
        experts.iter().flat_map(|e| e.iter()).map(|e| e.shadow_memory()).sum()
    }

    /// Total base memory across all experts (bytes).
    pub fn total_base_memory(&self) -> usize {
        let experts: Vec<&Vec<BitExpert>> = vec![&self.expert_gate, &self.expert_up, &self.expert_down];
        experts.iter().flat_map(|e| e.iter()).map(|e| e.base_memory()).sum()
    }

    /// Whether this layer has shadow weights (STE mode).
    pub fn is_ste(&self) -> bool {
        self.expert_gate.iter().any(|e| e.shadow.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_bit_moe() -> BitMoELayer {
        let hd = 4;
        let id = 8;
        let ne = 2;

        let gate_weights = vec![0.1f32; hd * ne];
        let make_weights = |base: f32, size: usize| -> Vec<f32> {
            (0..size).map(|i| ((i as f32 + base) * 0.1).sin()).collect()
        };

        BitMoELayer {
            gate_weights,
            expert_gate: (0..ne).map(|e| {
                let w = make_weights(e as f32, id * hd);
                BitExpert::from_fp32(&w, id, hd, false)
            }).collect(),
            expert_up: (0..ne).map(|e| {
                let w = make_weights(e as f32 + 0.5, id * hd);
                BitExpert::from_fp32(&w, id, hd, false)
            }).collect(),
            expert_down: (0..ne).map(|e| {
                let w = make_weights(e as f32 + 1.0, hd * id);
                BitExpert::from_fp32(&w, hd, id, false)
            }).collect(),
            ple_input_gate: None,
            ple_projection: None,
            alpha: 1.0,
            hidden_dim: hd,
            intermediate_dim: id,
            num_experts: ne,
            top_k: 2,
            use_gelu: false,
            scale_sync: ScaleSync::PerBlock,
        }
    }

    #[test]
    fn test_bit_moe_from_cpu_moe() {
        let cpu_moe = crate::model::cpu_moe::CpuMoELayer::new(4, 8, 2, 2);
        let bit_moe = BitMoELayer::from_cpu_moe(&cpu_moe);
        assert_eq!(bit_moe.num_experts, 2);
        assert_eq!(bit_moe.hidden_dim, 4);
        assert!(!bit_moe.is_ste());
    }

    #[test]
    fn test_init_shadows_sets_ste_mode() {
        let mut moe = make_test_bit_moe();
        assert!(!moe.is_ste());
        moe.init_shadows_from_ternary();
        assert!(moe.is_ste());
    }

    #[test]
    fn test_synchronized_alpha_consistent() {
        let mut moe = make_test_bit_moe();
        moe.init_shadows_from_ternary();

        // All experts should use the same α
        let alpha = moe.alpha;
        assert!(alpha > 0.0, "Alpha should be positive after init");

        // Expert scales should be set to layer alpha after requantize
        moe.requantize_all();
        for e in &moe.expert_gate {
            assert!((e.scale - alpha).abs() < 1e-6,
                "Expert scale {} should match layer alpha {}", e.scale, alpha);
        }
    }

    #[test]
    fn test_per_expert_alpha_unequal() {
        let mut moe = make_test_bit_moe();
        moe.scale_sync = ScaleSync::PerExpert;
        moe.init_shadows_from_ternary();

        // Skew one expert's shadows
        for s in moe.expert_gate[0].shadow.as_mut().unwrap().iter_mut() {
            *s = bf16::from_f32(s.to_f32() * 3.0);
        }

        moe.requantize_all();

        // With per-expert sync, scales should differ
        let s0 = moe.expert_gate[0].scale;
        let s1 = moe.expert_gate[1].scale;
        assert!(s0 > s1, "Expert 0 scale ({}) should be > expert 1 ({})", s0, s1);
    }

    #[test]
    fn test_drop_shadows_frees_memory() {
        let mut moe = make_test_bit_moe();
        moe.init_shadows_from_ternary();
        assert!(moe.total_shadow_memory() > 0);
        moe.drop_shadows();
        assert_eq!(moe.total_shadow_memory(), 0);
        assert!(!moe.is_ste());
    }

    #[test]
    fn test_expert_forward_ternary() {
        // Expert: rows=2, cols=4 → matmul expects input [seq × cols] = [1 × 4]
        let w = vec![5.0f32, -3.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
        let expert = BitExpert::from_fp32(&w, 2, 4, false);
        let input = vec![1.0, 2.0, 0.5, -0.5]; // cols=4
        let output = expert.forward_ternary(&input, 1);
        assert_eq!(output.len(), 2); // rows=2
        assert!(output.iter().any(|&x| x.abs() > 0.01));
    }

    #[test]
    fn test_expert_forward_ste_matches_ternary() {
        // When shadow ≈ dequantized ternary, STE forward should produce
        // similar results to ternary forward
        let w = vec![5.0f32, -3.0, 2.0, -1.0, 0.5, -0.5, 3.0, -2.0];
        let expert = BitExpert::from_fp32(&w, 2, 4, true);
        let input = vec![1.0, 2.0, 0.5, -0.5]; // cols=4

        let out_ternary = expert.forward_ternary(&input, 1);
        let out_ste = expert.forward_ste(&input, 1, expert.scale);

        // Should be close (not exact due to bf16 round-trip)
        for (a, b) in out_ternary.iter().zip(out_ste.iter()) {
            assert!((a - b).abs() < 0.1, "STE and ternary outputs should be close: {} vs {}", a, b);
        }
    }
}
