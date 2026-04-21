use wgpu::{
    BindGroupLayout, BufferDescriptor, BufferUsages, ComputePipeline, Device,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};
use crate::compute::GpuBuffer;
use crate::error::{FerrisResError, Result};

pub const RMSNORM_WGSL: &str = r#"
struct Params {
    hidden_dim: u32,
    rows: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let col = lid.x;
    let hidden_dim = params.hidden_dim;

    var partial = 0.0;
    for (var c = col; c < hidden_dim; c = c + 256u) {
        let v = input[row * hidden_dim + c];
        partial = partial + v * v;
    }

    wg_data[col] = partial;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (col < stride) {
            let a = wg_data[col];
            let b = wg_data[col + stride];
            wg_data[col] = a + b;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    var inv_rms = 1.0;
    if (col == 0u) {
        let ss = wg_data[0u];
        inv_rms = inverseSqrt(ss / f32(hidden_dim) + 1e-5);
        wg_data[0u] = inv_rms;
    }
    workgroupBarrier();
    inv_rms = wg_data[0u];

    for (var c = col; c < hidden_dim; c = c + 256u) {
        let v = input[row * hidden_dim + c];
        output[row * hidden_dim + c] = v * inv_rms;
    }
}
"#;

/// RMSNorm backward: dL/dx = (dL/dy - mean(dL/dy * y_hat) * y_hat) / rms
///
/// Where y_hat = x * inv_rms(x), rms = sqrt(mean(x²) + eps)
///
/// Inputs: grad_output [rows, hidden], input_x [rows, hidden]
/// Output: grad_input [rows, hidden]
pub const RMSNORM_BACKWARD_WGSL: &str = r#"
struct Params {
    hidden_dim: u32,
    rows: u32,
}

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> input_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> wg_sum_sq: array<f32, 256>;
var<workgroup> wg_sum_dy_y: array<f32, 256>;
var<workgroup> wg_inv_rms: f32;
var<workgroup> wg_mean_dy_y: f32;

@compute @workgroup_size(256)
fn rmsnorm_backward_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    let col = lid.x;
    let hidden_dim = params.hidden_dim;

    // Phase 1: Compute sum(x²) for RMS, and sum(dy * y_hat) for the backward
    var partial_sq = 0.0;
    var partial_dy_y = 0.0;
    for (var c = col; c < hidden_dim; c = c + 256u) {
        let idx = row * hidden_dim + c;
        let x_val = input_x[idx];
        partial_sq = partial_sq + x_val * x_val;
        // y_hat = x * inv_rms, but we don't know inv_rms yet.
        // We compute sum(dy * x) and multiply by inv_rms later.
        partial_dy_y = partial_dy_y + grad_output[idx] * x_val;
    }

    wg_sum_sq[col] = partial_sq;
    wg_sum_dy_y[col] = partial_dy_y;
    workgroupBarrier();

    // Phase 2: Reduce sum_sq and sum_dy_y
    var stride = 128u;
    while (stride > 0u) {
        if (col < stride) {
            wg_sum_sq[col] = wg_sum_sq[col] + wg_sum_sq[col + stride];
            wg_sum_dy_y[col] = wg_sum_dy_y[col] + wg_sum_dy_y[col + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Phase 3: Compute inv_rms and mean(dy * y_hat)
    if (col == 0u) {
        let mean_sq = wg_sum_sq[0u] / f32(hidden_dim);
        let rms = sqrt(mean_sq + 1e-5);
        wg_inv_rms = 1.0 / rms;
        // mean(dy * y_hat) = mean(dy * x * inv_rms) = inv_rms * mean(dy * x)
        wg_mean_dy_y = wg_sum_dy_y[0u] * wg_inv_rms / f32(hidden_dim);
    }
    workgroupBarrier();

    let inv_rms = wg_inv_rms;
    let mean_dy_y = wg_mean_dy_y;

    // Phase 4: Compute grad_input
    // dL/dx = inv_rms * (dL/dy - mean(dy * y_hat) * x * inv_rms)
    for (var c = col; c < hidden_dim; c = c + 256u) {
        let idx = row * hidden_dim + c;
        let dy = grad_output[idx];
        let x_val = input_x[idx];
        grad_input[idx] = inv_rms * (dy - mean_dy_y * x_val * inv_rms);
    }
}
"#;

pub struct RmsNormOp {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    backward_pipeline: ComputePipeline,
    backward_bind_group_layout: BindGroupLayout,
}

impl RmsNormOp {
    pub fn new(device: &Device) -> Result<Self> {
        tracing::info!(event = "creating_rmsnormop_compute_pipeline", "Creating RmsNormOp compute pipeline");

        // === Forward pipeline ===
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("RmsNorm Shader"),
            source: ShaderSource::Wgsl(RMSNORM_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RmsNorm Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("RmsNorm Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RmsNorm Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("rmsnorm_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // === Backward pipeline ===
        let backward_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RmsNorm Backward BGL"),
            entries: &[
                // binding 0: grad_output (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: input_x (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: grad_input (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let backward_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("RmsNorm Backward Shader"),
            source: ShaderSource::Wgsl(RMSNORM_BACKWARD_WGSL.into()),
        });

        let backward_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("RmsNorm Backward Pipeline Layout"),
            bind_group_layouts: &[Some(&backward_bgl)],
            immediate_size: 0,
        });

        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RmsNorm Backward Pipeline"),
            layout: Some(&backward_layout),
            module: &backward_shader,
            entry_point: Some("rmsnorm_backward_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            backward_pipeline,
            backward_bind_group_layout: backward_bgl,
        })
    }

    pub fn dispatch(
        &self,
        device: &Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: u32,
        hidden_dim: u32,
    ) -> Result<()> {
        tracing::debug!(
            "RmsNormOp::dispatch rows={} hidden_dim={}",
            rows,
            hidden_dim
        );

        if hidden_dim > 256 {
            return Err(FerrisResError::Unsupported(format!(
                "RmsNorm hidden_dim {} exceeds max workgroup size 256",
                hidden_dim
            )));
        }

        let params_data: [u32; 2] = [hidden_dim, rows];

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("RmsNorm Params"),
            size: 8,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RmsNorm Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RmsNorm Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows, 1, 1);

        Ok(())
    }

    /// Backward pass for RMSNorm.
    ///
    /// Given y = x / sqrt(mean(x²) + eps), computes dL/dx:
    ///   dL/dx = inv_rms * (dL/dy - mean(dL/dy * y_hat) * y_hat)
    /// where y_hat = x * inv_rms.
    ///
    /// # Arguments
    /// * `grad_output` — gradient w.r.t. normalized output [rows, hidden_dim]
    /// * `input_x` — original input before normalization [rows, hidden_dim]
    /// * `grad_input` — output gradient w.r.t. input [rows, hidden_dim]
    pub fn dispatch_backward(
        &self,
        device: &Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        grad_output: &GpuBuffer,
        input_x: &GpuBuffer,
        grad_input: &GpuBuffer,
        rows: u32,
        hidden_dim: u32,
    ) -> Result<()> {
        if hidden_dim > 256 {
            return Err(FerrisResError::Unsupported(format!(
                "RmsNorm backward hidden_dim {} exceeds max workgroup size 256",
                hidden_dim
            )));
        }

        let params_data: [u32; 2] = [hidden_dim, rows];
        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("RmsNorm Backward Params"),
            size: 8,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RmsNorm Backward Bind Group"),
            layout: &self.backward_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RmsNorm Backward Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.backward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows, 1, 1);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    /// Reference CPU implementation of RMSNorm backward.
    /// dL/dx = inv_rms * (dL/dy - mean(dL/dy * y_hat) * y_hat)
    /// where y_hat = x * inv_rms, inv_rms = 1/sqrt(mean(x²) + eps)
    fn rmsnorm_backward_ref(
        grad_output: &[f32],
        input_x: &[f32],
        rows: usize,
        hidden_dim: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut grad_input = vec![0.0f32; rows * hidden_dim];
        for r in 0..rows {
            let base = r * hidden_dim;
            let x_row = &input_x[base..base + hidden_dim];
            let dy_row = &grad_output[base..base + hidden_dim];

            // Compute inv_rms
            let sum_sq: f32 = x_row.iter().map(|&x| x * x).sum();
            let rms = (sum_sq / hidden_dim as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;

            // Compute y_hat = x * inv_rms
            let y_hat: Vec<f32> = x_row.iter().map(|&x| x * inv_rms).collect();

            // Compute mean(dy * y_hat)
            let mean_dy_y: f32 = dy_row.iter().zip(y_hat.iter()).map(|(&dy, &yh)| dy * yh).sum::<f32>() / hidden_dim as f32;

            // dL/dx = inv_rms * (dy - mean(dy * y_hat) * y_hat)
            for (i, (&dy, &yh)) in dy_row.iter().zip(y_hat.iter()).enumerate() {
                grad_input[base + i] = inv_rms * (dy - mean_dy_y * yh);
            }
        }
        grad_input
    }

    #[test]
    fn test_rmsnorm_backward_reference_unit() {
        // Single row, identity: if grad_output = y_hat, then dL/dx should be 0
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let hd = x.len();
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let inv_rms = 1.0 / (sum_sq / hd as f32 + 1e-5).sqrt();
        let y_hat: Vec<f32> = x.iter().map(|&v| v * inv_rms).collect();

        // When dy = y_hat, mean(dy * y_hat) = mean(y_hat²) = 1 (by definition of rms_norm)
        // So dL/dx = inv_rms * (y_hat - 1 * y_hat) = 0
        let grad = rmsnorm_backward_ref(&y_hat, &x, 1, hd, 1e-5);
        for &g in &grad {
            assert!(g.abs() < 1e-5, "Expected ~0, got {}", g);
        }
    }

    #[test]
    fn test_rmsnorm_backward_reference_uniform() {
        // x = all ones, dy = all ones
        let x = vec![1.0f32; 8];
        let dy = vec![1.0f32; 8];
        let hd = 8;

        let grad = rmsnorm_backward_ref(&dy, &x, 1, hd, 1e-6);

        // x = [1,1,...,1], inv_rms = 1/sqrt(1 + eps) ≈ 1
        // y_hat = [1,...,1], mean(dy * y_hat) = 1
        // dL/dx = 1 * (1 - 1*1) = 0 for all elements
        for &g in &grad {
            assert!(g.abs() < 1e-4, "Expected ~0 for uniform case, got {}", g);
        }
    }

    #[test]
    fn test_rmsnorm_backward_reference_nontrivial() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let dy = vec![1.0f32, 0.0, 0.0, 0.0];
        let hd = 4;

        let grad = rmsnorm_backward_ref(&dy, &x, 1, hd, 1e-5);

        // Verify finite
        for &g in &grad {
            assert!(g.is_finite(), "Got non-finite gradient: {}", g);
        }

        // Verify gradient sums to a reasonable value (not all zero)
        let grad_sum: f32 = grad.iter().sum();
        assert!(grad_sum.abs() > 0.1, "Expected non-trivial gradient, sum = {}", grad_sum);
    }

    #[test]
    fn test_rmsnorm_backward_reference_multirow() {
        let x = vec![1.0f32, 2.0, 1.0, -1.0, 0.5, 0.5];
        let dy = vec![1.0f32, 0.0, 0.5, 0.5, 1.0, -1.0];
        let rows = 2;
        let hd = 3;

        let grad = rmsnorm_backward_ref(&dy, &x, rows, hd, 1e-5);

        // Each row is independent — verify both have finite gradients
        for &g in &grad {
            assert!(g.is_finite());
        }
        assert_eq!(grad.len(), 6);
    }
}
