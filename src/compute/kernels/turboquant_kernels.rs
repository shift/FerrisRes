//! TurboQuant WGSL compute kernels for GPU-accelerated quantization.
//!
//! This module provides WGSL shaders for:
//! - Rotation: Matrix-vector multiply with rotation matrix R
//! - Quantization: Assign coordinates to nearest centroids (1D k-means)
//! - Dequantization: Reconstruct from centroid indices
//! - QJL: Quantized Johnson-Lindenstrauss projection
//!
//! # Usage
//! ```rust
//! use ferrisres::compute::turboquant_kernels::TURBOQUANT_WGSL;
//! ```

/// The complete TurboQuant WGSL shader source code
pub const TURBOQUANT_WGSL: &str = r#"
// ============================================================================
// TurboQuant WGSL Kernels
// ============================================================================

// ----------------------------------------------------------------
// TurboQuant Configuration Uniform
// ----------------------------------------------------------------
struct TurboQuantConfig {
    hidden_dim: u32,
    num_centroids: u32,
    bit_width: u32,
    enable_qjl: u32,
}

// ----------------------------------------------------------------
// Rotation Kernel: y = R @ x
// Applies random rotation matrix to input vector
// ----------------------------------------------------------------
@compute @workgroup_size(64)
fn kernel_rotation(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @storage Buffer input: array<f32>,
    @storage Buffer rotation: array<f32>,
    @storage Buffer output: array<f32>,
    @uniform config: TurboQuantConfig,
) {
    let hidden_dim = config.hidden_dim;
    let idx = gid.x;
    
    if idx >= hidden_dim {
        return;
    }
    
    var sum = 0.0;
    for (var j = 0u; j < hidden_dim; j++) {
        sum += rotation[idx * hidden_dim + j] * input[j];
    }
    output[idx] = sum;
}

// ----------------------------------------------------------------
// Quantization Kernel: Assign to nearest centroid
// Input: rotated coordinates y
// Output: centroid indices
// ----------------------------------------------------------------
@compute @workgroup_size(64)
fn kernel_quantize(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @storage Buffer input: array<f32>,
    @storage Buffer centroids: array<f32>,
    @storage_write output: array<u32>,
    @uniform config: TurboQuantConfig,
) {
    let idx = gid.x;
    let num_centroids = config.num_centroids;
    let hidden_dim = config.hidden_dim;
    
    if idx >= hidden_dim {
        return;
    }
    
    let value = input[idx];
    
    // Find nearest centroid
    var min_dist = 1e30;
    var best_centroid = 0u;
    
    for (var i = 0u; i < num_centroids; i++) {
        let c = centroids[i];
        let dist = (value - c) * (value - c);
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
        }
    }
    
    output[idx] = best_centroid;
}

// ----------------------------------------------------------------
// Dequantization Kernel: Reconstruct from centroid indices
// Input: centroid indices
// Output: reconstructed coordinates
// ----------------------------------------------------------------
@compute @workgroup_size(64)
fn kernel_dequantize(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @storage Buffer indices: array<u32>,
    @storage Buffer centroids: array<f32>,
    @storage_write output: array<f32>,
    @uniform config: TurboQuantConfig,
) {
    let idx = gid.x;
    let hidden_dim = config.hidden_dim;
    
    if idx >= hidden_dim {
        return;
    }
    
    let centroid_idx = indices[idx];
    output[idx] = centroids[centroid_idx];
}

// ----------------------------------------------------------------
// QJL Projection Kernel: q = sign(S @ r)
// Applied to residual for unbiased inner products
// ----------------------------------------------------------------
@compute @workgroup_size(64)
fn kernel_qjl_project(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @storage Buffer residual: array<f32>,
    @storage Buffer projection: array<f32>,
    @storage_write output: array<i32>,
    @uniform config: TurboQuantConfig,
) {
    let hidden_dim = config.hidden_dim;
    let idx = gid.x;
    
    if idx >= hidden_dim {
        return;
    }
    
    // Compute projection: sum_j S[idx][j] * residual[j]
    var sum = 0.0;
    for (var j = 0u; j < hidden_dim; j++) {
        sum += projection[idx * hidden_dim + j] * residual[j];
    }
    
    // Sign function (-1 for negative, +1 for positive)
    output[idx] = select(1i, -1i, sum < 0.0);
}

// ----------------------------------------------------------------
// QJL Reconstruction: x_qjl = (sqrt(pi/2)/d) * ||r||_2 * S^T @ q
// ----------------------------------------------------------------
@compute @workgroup_size(64)
fn kernel_qjl_reconstruct(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @storage Buffer qjl_bits: array<i32>,
    @storage Buffer projection: array<f32>,
    @storage_write output: array<f32>,
    @uniform residual_norm: f32,
    @uniform config: TurboQuantConfig,
) {
    let hidden_dim = config.hidden_dim;
    let idx = gid.x;
    
    if idx >= hidden_dim {
        return;
    }
    
    var sum = 0.0;
    for (var j = 0u; j < hidden_dim; j++) {
        sum += projection[j * hidden_dim + idx] * f32(qjl_bits[j]);
    }
    
    let scale = (1.41421356 / f32(hidden_dim)) * residual_norm;
    output[idx] = scale * sum;
}

// ----------------------------------------------------------------
// Inverse Rotation Kernel: x = R^T @ y
// ----------------------------------------------------------------
@compute @workgroup_size(64)
fn kernel_inverse_rotation(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @storage Buffer input: array<f32>,
    @storage Buffer rotation: array<f32>,
    @storage Buffer output: array<f32>,
    @uniform config: TurboQuantConfig,
) {
    let hidden_dim = config.hidden_dim;
    let idx = gid.x;
    
    if idx >= hidden_dim {
        return;
    }
    
    var sum = 0.0;
    for (var j = 0u; j < hidden_dim; j++) {
        sum += rotation[j * hidden_dim + idx] * input[j];
    }
    output[idx] = sum;
}
"#;

/// Errors for TurboQuant kernels
#[derive(Debug)]
pub struct TurboQuantKernelError {
    message: String,
}

impl std::fmt::Display for TurboQuantKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TurboQuantKernelError: {}", self.message)
    }
}

impl std::error::Error for TurboQuantKernelError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wgsl_source_exists() {
        // Verify the WGSL source is present
        assert!(TURBOQUANT_WGSL.len() > 2000);
        assert!(TURBOQUANT_WGSL.contains("kernel_rotation"));
        assert!(TURBOQUANT_WGSL.contains("kernel_quantize"));
        assert!(TURBOQUANT_WGSL.contains("kernel_dequantize"));
        assert!(TURBOQUANT_WGSL.contains("kernel_qjl_project"));
    }
    
    #[test]
    fn test_kernel_names() {
        // Verify all expected kernels are defined
        let expected = [
            "kernel_rotation",
            "kernel_quantize", 
            "kernel_dequantize",
            "kernel_qjl_project",
            "kernel_qjl_reconstruct",
            "kernel_inverse_rotation",
        ];
        
        for name in expected {
            assert!(
                TURBOQUANT_WGSL.contains(name),
                "Missing kernel: {}",
                name
            );
        }
    }
}