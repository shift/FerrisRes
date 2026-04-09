/// Property-based tests for GPU-independent kernel logic in FerrisRes.
///
/// These tests exercise pure-Rust reference implementations of the math
/// embedded in WGSL compute shaders. No GPU dispatch is required.
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Reference implementations (mirroring WGSL shader logic in pure Rust)
// ---------------------------------------------------------------------------

/// RoPE frequency for a given (position, pair_index, head_dim).
/// Mirrors the WGSL in src/compute/kernels/rope.rs:
///   let freq = 1.0 / pow(10000.0, f32(2*pair_idx) / f32(head_dim));
///   let theta = f32(pos) * freq;
fn rope_theta(pos: u32, pair_idx: u32, head_dim: u32) -> f32 {
    assert!(head_dim > 0, "head_dim must be > 0");
    let base: f32 = 10000.0;
    let freq = 1.0_f32 / base.powf((2 * pair_idx) as f32 / head_dim as f32);
    pos as f32 * freq
}

/// CPU reference softmax over a slice.
/// Mirrors the numerically-stable WGSL in src/compute/kernels/softmax.rs.
fn softmax_ref(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty());
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// CPU reference RMSNorm (no learnable weight, as in the WGSL).
/// y_i = x_i * inverseSqrt(mean(x^2) + eps)
fn rmsnorm_ref(x: &[f32], eps: f32) -> Vec<f32> {
    assert!(!x.is_empty());
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
    let inv_rms = 1.0_f32 / (mean_sq + eps).sqrt();
    x.iter().map(|&v| v * inv_rms).collect()
}

/// CPU reference MoE top-k selection count.
/// Mirrors `let max_k = min(params.top_k, params.num_experts)` in moe.rs.
fn moe_top_k_count(top_k: usize, num_experts: usize) -> usize {
    top_k.min(num_experts)
}

/// CPU reference workgroup count formula used across all kernels.
/// (n + wg_size - 1) / wg_size  — i.e., ceiling division
fn workgroup_count(n: u32, wg_size: u32) -> u32 {
    (n + wg_size - 1) / wg_size
}

/// CPU reference causal-mask predicate.
/// A position (row, col) in a square seq_len×seq_len attention matrix is
/// masked (set to -1e9) when col > (row % seq_len).
fn causal_mask_is_masked(row: usize, col: usize, seq_len: usize) -> bool {
    col > (row % seq_len)
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

proptest! {
    // ------------------------------------------------------------------
    // 1. RoPE theta is always finite and non-NaN
    // ------------------------------------------------------------------
    #[test]
    fn prop_rope_theta_finite(
        pos       in 0u32..8192u32,
        pair_idx  in 0u32..64u32,
        head_dim  in 2u32..128u32,
    ) {
        // pair_idx must be < head_dim/2 for a meaningful rotation; clamp it.
        let pair_idx = pair_idx % (head_dim / 2).max(1);
        let theta = rope_theta(pos, pair_idx, head_dim);
        prop_assert!(theta.is_finite(), "theta={} for pos={} pair_idx={} head_dim={}", theta, pos, pair_idx, head_dim);
        prop_assert!(!theta.is_nan());
    }

    // ------------------------------------------------------------------
    // 2. RoPE rotation preserves vector length (cos²+sin²=1 identity)
    //    For any theta, (x0*cos-x1*sin)²+(x0*sin+x1*cos)² == x0²+x1²
    // ------------------------------------------------------------------
    #[test]
    fn prop_rope_rotation_preserves_norm(
        pos       in 0u32..4096u32,
        pair_idx  in 0u32..32u32,
        head_dim  in 2u32..64u32,
        x0        in -1e3_f32..1e3_f32,
        x1        in -1e3_f32..1e3_f32,
    ) {
        let pair_idx = pair_idx % (head_dim / 2).max(1);
        let theta = rope_theta(pos, pair_idx, head_dim);
        let (cos_t, sin_t) = (theta.cos(), theta.sin());

        let y0 = x0 * cos_t - x1 * sin_t;
        let y1 = x0 * sin_t + x1 * cos_t;

        let in_norm  = x0 * x0 + x1 * x1;
        let out_norm = y0 * y0 + y1 * y1;
        let rel_err  = (in_norm - out_norm).abs() / (in_norm.abs() + 1e-6);
        prop_assert!(rel_err < 1e-4,
            "norm not preserved: in={} out={} rel_err={}", in_norm, out_norm, rel_err);
    }

    // ------------------------------------------------------------------
    // 3. Softmax outputs always sum to 1.0 (within f32 epsilon)
    // ------------------------------------------------------------------
    #[test]
    fn prop_softmax_sums_to_one(
        logits in prop::collection::vec(-100.0f32..100.0f32, 1..256),
    ) {
        // Skip inputs that contain NaN/Inf — those are not valid model outputs.
        prop_assume!(logits.iter().all(|x| x.is_finite()));
        let probs = softmax_ref(&logits);
        let sum: f32 = probs.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-4,
            "softmax sum = {} (expected 1.0), len={}", sum, logits.len());
    }

    // ------------------------------------------------------------------
    // 4. Softmax outputs are all in [0, 1]
    // ------------------------------------------------------------------
    #[test]
    fn prop_softmax_values_in_range(
        logits in prop::collection::vec(-50.0f32..50.0f32, 1..128),
    ) {
        prop_assume!(logits.iter().all(|x| x.is_finite()));
        let probs = softmax_ref(&logits);
        for &p in &probs {
            prop_assert!(p >= 0.0 && p <= 1.0 + 1e-6,
                "softmax value out of [0,1]: {}", p);
        }
    }

    // ------------------------------------------------------------------
    // 5. RMSNorm: output RMS ≈ 1.0 for any non-zero input
    // ------------------------------------------------------------------
    #[test]
    fn prop_rmsnorm_output_rms_near_one(
        x in prop::collection::vec(-100.0f32..100.0f32, 1..256),
    ) {
        prop_assume!(x.iter().all(|v| v.is_finite()));
        // Require at least one non-zero element so RMS is well-defined.
        prop_assume!(x.iter().any(|&v| v.abs() > 1e-6));

        let y = rmsnorm_ref(&x, 1e-5);
        let rms_out = (y.iter().map(|&v| v * v).sum::<f32>() / y.len() as f32).sqrt();
        // RMS of output should be ≈ 1; allow generous tolerance due to eps term.
        prop_assert!((rms_out - 1.0).abs() < 0.1,
            "RMSNorm output RMS = {} (expected ~1.0)", rms_out);
    }

    // ------------------------------------------------------------------
    // 6. MoE top-k count is always min(k, num_experts)
    // ------------------------------------------------------------------
    #[test]
    fn prop_moe_top_k_count_correct(
        num_experts in 1usize..128usize,
        top_k       in 1usize..128usize,
    ) {
        let selected = moe_top_k_count(top_k, num_experts);
        let expected = top_k.min(num_experts);
        prop_assert_eq!(selected, expected,
            "expected min({},{})={} got {}", top_k, num_experts, expected, selected);
        prop_assert!(selected <= num_experts,
            "selected {} > num_experts {}", selected, num_experts);
        prop_assert!(selected <= top_k,
            "selected {} > top_k {}", selected, top_k);
    }

    // ------------------------------------------------------------------
    // 7. Workgroup count formula: never zero for n > 0, always covers n
    // ------------------------------------------------------------------
    #[test]
    fn prop_workgroup_count_covers_all_elements(
        n       in 1u32..65536u32,
        wg_size in 1u32..512u32,
    ) {
        let count = workgroup_count(n, wg_size);
        prop_assert!(count > 0, "workgroup_count({},{}) == 0", n, wg_size);
        prop_assert!(count * wg_size >= n,
            "workgroup_count={} * wg_size={} = {} < n={}", count, wg_size, count * wg_size, n);
        // Tightness: (count-1)*wg_size < n  (no unnecessary extra workgroup)
        if count > 1 {
            prop_assert!((count - 1) * wg_size < n,
                "too many workgroups: ({}-1)*{}={} >= n={}", count, wg_size, (count-1)*wg_size, n);
        }
    }

    // ------------------------------------------------------------------
    // 8. Causal mask: upper-triangle positions are always masked
    // ------------------------------------------------------------------
    #[test]
    fn prop_causal_mask_upper_triangle_masked(
        seq_len in 1usize..64usize,
        row     in 0usize..64usize,
        col     in 0usize..64usize,
    ) {
        let row = row % seq_len;
        let col = col % seq_len;
        let masked = causal_mask_is_masked(row, col, seq_len);
        // Upper triangle: col > row => masked
        if col > row {
            prop_assert!(masked,
                "row={} col={} seq_len={} should be masked", row, col, seq_len);
        }
        // Diagonal and lower triangle: col <= row => not masked
        if col <= row {
            prop_assert!(!masked,
                "row={} col={} seq_len={} should NOT be masked", row, col, seq_len);
        }
    }
}
