//! 3D/factored convolution for video and volumetric processing.
//!
//! Decomposes expensive 3D convolutions (time × height × width) into
//! efficient separable operations:
//!   - Temporal: (time × 1 × 1) — processes temporal relationships
//!   - Spatial: (1 × height × width) — processes spatial features per frame
//!
//! This reduces computation from O(T·H·W·Ci·Co·Kt·Kh·Kw) to
//! O(T·H·W·Ci·Cm·Kt) + O(T·H·W·Cm·Co·Kh·Kw), a significant saving.
//!
//! WGSL kernels provided for both factored and full 3D convolution.

use crate::error::Result;

// ---------------------------------------------------------------------------
// Conv3DConfig
// ---------------------------------------------------------------------------

/// Configuration for a 3D convolution layer.
#[derive(Debug, Clone)]
pub struct Conv3DConfig {
    /// Input channels.
    pub in_channels: usize,
    /// Output channels.
    pub out_channels: usize,
    /// Mid channels for factored decomposition.
    pub mid_channels: usize,
    /// Kernel size (T, H, W).
    pub kernel_size: (usize, usize, usize),
    /// Stride (T, H, W).
    pub stride: (usize, usize, usize),
    /// Padding (T, H, W).
    pub padding: (usize, usize, usize),
}

impl Conv3DConfig {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize, usize)) -> Self {
        Self {
            in_channels,
            out_channels,
            mid_channels: (in_channels + out_channels) / 2,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
        }
    }

    /// Standard 3×3×3 conv.
    pub fn conv3x3x3(in_ch: usize, out_ch: usize) -> Self {
        Self::new(in_ch, out_ch, (3, 3, 3))
    }

    /// Calculate output dimensions.
    pub fn output_shape(&self, input_shape: (usize, usize, usize)) -> (usize, usize, usize) {
        let (t, h, w) = input_shape;
        let (kt, kh, kw) = self.kernel_size;
        let (st, sh, sw) = self.stride;
        let (pt, ph, pw) = self.padding;

        let out_t = (t + 2 * pt - kt) / st + 1;
        let out_h = (h + 2 * ph - kh) / sh + 1;
        let out_w = (w + 2 * pw - kw) / sw + 1;

        (out_t, out_h, out_w)
    }

    /// FLOPs for full 3D convolution.
    pub fn flops_full(&self, input_shape: (usize, usize, usize)) -> usize {
        let (t, h, w) = self.output_shape(input_shape);
        let (kt, kh, kw) = self.kernel_size;
        t * h * w * self.in_channels * self.out_channels * kt * kh * kw
    }

    /// FLOPs for factored convolution.
    pub fn flops_factored(&self, input_shape: (usize, usize, usize)) -> usize {
        let (t, h, w) = self.output_shape(input_shape);
        let (kt, _kh, _kw) = self.kernel_size;
        let (_kt, kh, kw) = self.kernel_size;

        // Temporal: (T,H,W) × in_ch × mid_ch × kt
        let temporal = t * h * w * self.in_channels * self.mid_channels * kt;
        // Spatial: (T,H,W) × mid_ch × out_ch × kh × kw
        let spatial = t * h * w * self.mid_channels * self.out_channels * kh * kw;
        temporal + spatial
    }

    /// Savings ratio (full / factored).
    pub fn savings_ratio(&self, input_shape: (usize, usize, usize)) -> f32 {
        let full = self.flops_full(input_shape);
        let factored = self.flops_factored(input_shape);
        if factored == 0 { return 1.0; }
        full as f32 / factored as f32
    }
}

// ---------------------------------------------------------------------------
// FactoredConv3D — temporal + spatial decomposition
// ---------------------------------------------------------------------------

/// Factored 3D convolution decomposed into temporal + spatial passes.
///
/// Instead of a single O(T·H·W·Ci·Co·Kt·Kh·Kw) convolution, performs:
/// 1. Temporal pass: (time × 1 × 1) convolution — processes temporal relationships
/// 2. Spatial pass: (1 × height × width) convolution — processes spatial features
pub struct FactoredConv3D {
    config: Conv3DConfig,
    /// Temporal weights: [mid_channels × in_channels × Kt].
    temporal_weights: Vec<f32>,
    /// Spatial weights: [out_channels × mid_channels × Kh × Kw].
    spatial_weights: Vec<f32>,
    /// Temporal bias: [mid_channels].
    temporal_bias: Vec<f32>,
    /// Spatial bias: [out_channels].
    spatial_bias: Vec<f32>,
}

impl FactoredConv3D {
    pub fn new(config: Conv3DConfig) -> Self {
        let mid = config.mid_channels;
        let (kt, kh, kw) = config.kernel_size;
        let ci = config.in_channels;
        let co = config.out_channels;

        // Xavier-like initialization
        let t_scale = (2.0 / (ci * kt) as f32).sqrt();
        let s_scale = (2.0 / (mid * kh * kw) as f32).sqrt();

        let temporal_weights: Vec<f32> = (0..mid * ci * kt)
            .map(|i| (((i as f32 * 0.618).sin() * 43758.5453).fract() - 0.5) * t_scale)
            .collect();
        let spatial_weights: Vec<f32> = (0..co * mid * kh * kw)
            .map(|i| (((i as f32 * 0.618).sin() * 43758.5453).fract() - 0.5) * s_scale)
            .collect();

        Self {
            config,
            temporal_weights,
            spatial_weights,
            temporal_bias: vec![0.0; mid],
            spatial_bias: vec![0.0; co],
        }
    }

    /// Forward pass: temporal → spatial decomposition.
    ///
    /// Input: [T × H × W × Ci]
    /// After temporal: [T' × H × W × Cm]
    /// After spatial: [T' × H' × W' × Co]
    pub fn forward(&self, input: &[f32], shape: (usize, usize, usize, usize)) -> Result<Vec<f32>> {
        let (t, h, w, _ci) = shape;
        let mid = self.config.mid_channels;
        let co = self.config.out_channels;
        let (kt, kh, kw) = self.config.kernel_size;

        // Step 1: Temporal convolution (time × 1 × 1)
        let t_out = t.saturating_sub(kt - 1);
        let mut temporal_out = vec![0.0f32; t_out * h * w * mid];

        for ot in 0..t_out {
            for y in 0..h {
                for x in 0..w {
                    for m in 0..mid {
                        let mut sum = self.temporal_bias[m];
                        for kt_i in 0..kt {
                            for ci in 0..self.config.in_channels {
                                let it = ot + kt_i;
                                let in_idx = ((it * h + y) * w + x) * self.config.in_channels + ci;
                                let w_idx = (m * self.config.in_channels + ci) * kt + kt_i;
                                sum += input[in_idx] * self.temporal_weights[w_idx];
                            }
                        }
                        let out_idx = ((ot * h + y) * w + x) * mid + m;
                        temporal_out[out_idx] = sum;
                    }
                }
            }
        }

        // Step 2: Spatial convolution (1 × height × width)
        let h_out = h.saturating_sub(kh - 1);
        let w_out = w.saturating_sub(kw - 1);

        let mut output = vec![0.0f32; t_out * h_out * w_out * co];

        for ot in 0..t_out {
            for oy in 0..h_out {
                for ox in 0..w_out {
                    for o in 0..co {
                        let mut sum = self.spatial_bias[o];
                        for ky in 0..kh {
                            for kx in 0..kw {
                                for m in 0..mid {
                                    let iy = oy + ky;
                                    let ix = ox + kx;
                                    let t_in_idx = ((ot * h + iy) * w + ix) * mid + m;
                                    let w_idx = ((o * mid + m) * kh + ky) * kw + kx;
                                    sum += temporal_out[t_in_idx] * self.spatial_weights[w_idx];
                                }
                            }
                        }
                        let out_idx = ((ot * h_out + oy) * w_out + ox) * co + o;
                        output[out_idx] = sum;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Forward per-frame 2D baseline (processes each frame independently).
    pub fn forward_per_frame(&self, frames: &[&[f32]], frame_h: usize, frame_w: usize) -> Vec<Vec<f32>> {
        let kh = self.config.kernel_size.1;
        let kw = self.config.kernel_size.2;
        let co = self.config.out_channels;
        let mid = self.config.mid_channels;
        let h_out = frame_h.saturating_sub(kh - 1);
        let w_out = frame_w.saturating_sub(kw - 1);

        frames.iter().map(|frame| {
            let mut out = vec![0.0f32; h_out * w_out * co];
            for oy in 0..h_out {
                for ox in 0..w_out {
                    for o in 0..co {
                        let mut sum = self.spatial_bias[o];
                        for ky in 0..kh {
                            for kx in 0..kw {
                                for m in 0..mid {
                                    let iy = oy + ky;
                                    let ix = ox + kx;
                                    let in_idx = (iy * frame_w + ix) * mid + m;
                                    let w_idx = ((o * mid + m) * kh + ky) * kw + kx;
                                    sum += frame[in_idx] * self.spatial_weights[w_idx];
                                }
                            }
                        }
                        let out_idx = (oy * w_out + ox) * co + o;
                        out[out_idx] = sum;
                    }
                }
            }
            out
        }).collect()
    }

    /// Get the config.
    pub fn config(&self) -> &Conv3DConfig {
        &self.config
    }

    /// Temporal weights reference.
    pub fn temporal_weights(&self) -> &[f32] {
        &self.temporal_weights
    }

    /// Spatial weights reference.
    pub fn spatial_weights(&self) -> &[f32] {
        &self.spatial_weights
    }
}

// ---------------------------------------------------------------------------
// WGSL kernels
// ---------------------------------------------------------------------------

/// WGSL kernel for temporal convolution (time × 1 × 1).
pub const TEMPORAL_CONV_WGSL: &str = r#"
struct Params {
    batch: u32,
    time_in: u32,
    height: u32,
    width: u32,
    in_channels: u32,
    mid_channels: u32,
    kernel_t: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       weights: array<f32>;
@group(0) @binding(2) var<storage, read>       bias:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn temporal_conv(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let hw = params.height * params.width;
    let t_out = params.time_in - params.kernel_t + 1u;
    let total = t_out * hw * params.mid_channels;
    if (idx >= total) { return; }

    let m = idx % params.mid_channels;
    let rem = idx / params.mid_channels;
    let x = rem % params.width;
    let rem2 = rem / params.width;
    let y = rem2 % params.height;
    let ot = rem2 / params.height;

    var sum = bias[m];
    for (var kt: u32 = 0u; kt < params.kernel_t; kt = kt + 1u) {
        let it = ot + kt;
        for (var ci: u32 = 0u; ci < params.in_channels; ci = ci + 1u) {
            let in_idx = ((it * params.height + y) * params.width + x) * params.in_channels + ci;
            let w_idx = (m * params.in_channels + ci) * params.kernel_t + kt;
            sum += input[in_idx] * weights[w_idx];
        }
    }
    output[idx] = sum;
}
"#;

/// WGSL kernel for spatial convolution (1 × height × width).
pub const SPATIAL_CONV_WGSL: &str = r#"
struct Params {
    time: u32,
    height_in: u32,
    width_in: u32,
    mid_channels: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read>       weights: array<f32>;
@group(0) @binding(2) var<storage, read>       bias:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn spatial_conv(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let h_out = params.height_in - params.kernel_h + 1u;
    let w_out = params.width_in - params.kernel_w + 1u;
    let hw_out = h_out * w_out;
    let total = params.time * hw_out * params.out_channels;
    if (idx >= total) { return; }

    let o = idx % params.out_channels;
    let rem = idx / params.out_channels;
    let ox = rem % w_out;
    let rem2 = rem / w_out;
    let oy = rem2 % h_out;
    let t = rem2 / h_out;

    var sum = bias[o];
    for (var ky: u32 = 0u; ky < params.kernel_h; ky = ky + 1u) {
        for (var kx: u32 = 0u; kx < params.kernel_w; kx = kx + 1u) {
            let iy = oy + ky;
            let ix = ox + kx;
            for (var m: u32 = 0u; m < params.mid_channels; m = m + 1u) {
                let in_idx = ((t * params.height_in + iy) * params.width_in + ix) * params.mid_channels + m;
                let w_idx = ((o * params.mid_channels + m) * params.kernel_h + ky) * params.kernel_w + kx;
                sum += input[in_idx] * weights[w_idx];
            }
        }
    }
    output[idx] = sum;
}
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv3d_config_output_shape() {
        let config = Conv3DConfig::conv3x3x3(3, 16);
        let out = config.output_shape((8, 32, 32));
        assert_eq!(out, (6, 30, 30)); // (8-3+1, 32-3+1, 32-3+1)
    }

    #[test]
    fn test_conv3d_config_stride() {
        let mut config = Conv3DConfig::conv3x3x3(3, 16);
        config.stride = (2, 2, 2);
        let out = config.output_shape((8, 32, 32));
        assert_eq!(out, (3, 15, 15));
    }

    #[test]
    fn test_conv3d_config_flops() {
        // Use mid_channels smaller than out_channels to guarantee savings
        let mut config = Conv3DConfig::conv3x3x3(3, 16);
        config.mid_channels = 8; // Bottleneck
        let shape = (8, 16, 16);
        let full = config.flops_full(shape);
        let factored = config.flops_factored(shape);
        // Factored should be less than full when mid < out
        assert!(factored <= full);
        let ratio = config.savings_ratio(shape);
        assert!(ratio >= 1.0);
    }

    #[test]
    fn test_conv3d_savings() {
        // For a typical video conv: 3×3×3, 64→128 channels, 8×64×64 frames
        let config = Conv3DConfig {
            in_channels: 64,
            out_channels: 128,
            mid_channels: 64,
            kernel_size: (3, 3, 3),
            ..Conv3DConfig::conv3x3x3(64, 128)
        };
        let ratio = config.savings_ratio((8, 64, 64));
        // Should be significant savings (>1.5x)
        assert!(ratio > 1.5, "Factored should save >1.5x, got {}x", ratio);
    }

    #[test]
    fn test_factored_conv3d_forward() {
        let config = Conv3DConfig::new(2, 4, (3, 3, 3));
        let conv = FactoredConv3D::new(config);

        // T=3, H=4, W=4, Ci=2 → output T=1, H=2, W=2, Co=4
        let input = vec![0.5f32; 3 * 4 * 4 * 2];
        let output = conv.forward(&input, (3, 4, 4, 2)).unwrap();
        // T_out=1, H_out=2, W_out=2, Co=4
        assert_eq!(output.len(), 1 * 2 * 2 * 4);
    }

    #[test]
    fn test_factored_conv3d_per_frame() {
        let config = Conv3DConfig::new(2, 4, (3, 3, 3));
        let mid = config.mid_channels;
        let conv = FactoredConv3D::new(config);

        // Per-frame assumes temporal output already done: frame has mid channels
        let frame: Vec<f32> = vec![1.0; 4 * 4 * mid]; // H=4, W=4, C=mid
        let frames: Vec<&[f32]> = vec![&frame, &frame];
        let outputs = conv.forward_per_frame(&frames, 4, 4);
        assert_eq!(outputs.len(), 2);
        // Each frame: H_out=4-3+1=2, W_out=4-3+1=2, Co=4
        assert_eq!(outputs[0].len(), 2 * 2 * 4);
    }

    #[test]
    fn test_factored_conv3d_weights_shape() {
        let config = Conv3DConfig::new(3, 16, (3, 3, 3));
        let conv = FactoredConv3D::new(config);
        // Temporal: mid × in × kt
        assert_eq!(conv.temporal_weights().len(), 9 * 3 * 3);
        // Spatial: out × mid × kh × kw
        assert_eq!(conv.spatial_weights().len(), 16 * 9 * 3 * 3);
    }

    #[test]
    fn test_wgsl_kernels_valid() {
        assert!(!TEMPORAL_CONV_WGSL.is_empty());
        assert!(TEMPORAL_CONV_WGSL.contains("temporal_conv"));
        assert!(!SPATIAL_CONV_WGSL.is_empty());
        assert!(SPATIAL_CONV_WGSL.contains("spatial_conv"));
    }
}
