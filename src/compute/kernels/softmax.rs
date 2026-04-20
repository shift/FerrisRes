use std::sync::Arc;
use wgpu::{Device, Queue, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

const SOFTMAX_WGSL: &str = r#"
struct Params {
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_max: array<f32, WG_SIZE>;
var<workgroup> shared_sum: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn softmax_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local_id: u32) {
    let row_idx = wid.x;
    let cols = params.cols;
    let row_offset = row_idx * cols;

    var local_max: f32 = -3.4028235e+38;

    for (var i = local_id; i < cols; i += WG_SIZE) {
        let val = input[row_offset + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    shared_max[local_id] = local_max;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            let a = shared_max[local_id];
            let b = shared_max[local_id + stride];
            if (b > a) {
                shared_max[local_id] = b;
            }
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let row_max = shared_max[0];

    var local_sum: f32 = 0.0;

    for (var i = local_id; i < cols; i += WG_SIZE) {
        let val = exp(input[row_offset + i] - row_max);
        output[row_offset + i] = val;
        local_sum = local_sum + val;
    }

    shared_sum[local_id] = local_sum;
    workgroupBarrier();

    stride = WG_SIZE / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            let a = shared_sum[local_id];
            let b = shared_sum[local_id + stride];
            shared_sum[local_id] = a + b;
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let row_sum = shared_sum[0];

    for (var i = local_id; i < cols; i += WG_SIZE) {
        output[row_offset + i] = output[row_offset + i] / row_sum;
    }
}
"#;

/// Softmax backward: dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
/// Uses workgroup reduction for the dot product term.
const SOFTMAX_BACKWARD_WGSL: &str = r#"
struct Params {
    cols: u32,
}

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> softmax_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_dot: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn softmax_backward_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) local_id: u32,
) {
    let row_idx = wid.x;
    let cols = params.cols;
    let row_offset = row_idx * cols;

    // Phase 1: Compute partial sum of (dL/ds_j * s_j)
    var partial_dot = 0.0;
    for (var i = local_id; i < cols; i += WG_SIZE) {
        let idx = row_offset + i;
        partial_dot = partial_dot + grad_output[idx] * softmax_output[idx];
    }
    shared_dot[local_id] = partial_dot;
    workgroupBarrier();

    // Phase 2: Tree reduction
    var stride = WG_SIZE / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_dot[local_id] = shared_dot[local_id] + shared_dot[local_id + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    let dot_product = shared_dot[0];

    // Phase 3: dL/dx_i = s_i * (dL/ds_i - dot_product)
    for (var i = local_id; i < cols; i += WG_SIZE) {
        let idx = row_offset + i;
        let s = softmax_output[idx];
        grad_input[idx] = s * (grad_output[idx] - dot_product);
    }
}
"#;

pub struct SoftmaxOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    backward_pipeline: wgpu::ComputePipeline,
    backward_bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl SoftmaxOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Softmax Shader"),
            source: wgpu::ShaderSource::Wgsl(SOFTMAX_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Softmax Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Softmax Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Softmax Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("softmax_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!(event = "softmaxop_pipeline_created", "SoftmaxOp pipeline created");

        // === Backward pipeline ===
        let backward_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Softmax Backward BGL"),
            entries: &[
                // binding 0: grad_output (read)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: softmax_output (read)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: grad_input (write)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: params (uniform)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Softmax Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(SOFTMAX_BACKWARD_WGSL.into()),
        });

        let backward_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Softmax Backward Pipeline Layout"),
            bind_group_layouts: &[Some(&backward_bgl)],
            immediate_size: 0,
        });

        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Softmax Backward Pipeline"),
            layout: Some(&backward_layout),
            module: &backward_shader,
            entry_point: Some("softmax_backward_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
            backward_pipeline,
            backward_bind_group_layout: backward_bgl,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }

        let params_data: [u32; 1] = [cols];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Softmax Params"),
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Softmax Bind Group"),
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
            label: Some("Softmax Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows, 1, 1);

        drop(pass);

        tracing::debug!(event = "softmaxop_dispatched_rows_cols", "SoftmaxOp dispatched: rows={} cols={}", rows, cols);

        Ok(())
    }

    /// Backward pass for softmax.
    ///
    /// dL/dx_i = s_i * (dL/ds_i - Σ_j(dL/ds_j * s_j))
    ///
    /// # Arguments
    /// * `grad_output` — gradient w.r.t. softmax output [rows, cols]
    /// * `softmax_output` — the forward softmax output [rows, cols]
    /// * `grad_input` — output gradient w.r.t. pre-softmax logits [rows, cols]
    pub fn dispatch_backward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        grad_output: &GpuBuffer,
        softmax_output: &GpuBuffer,
        grad_input: &GpuBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        if cols == 0 || rows == 0 {
            return Ok(());
        }

        let params_data: [u32; 1] = [cols];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Softmax Backward Params"),
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Softmax Backward Bind Group"),
            layout: &self.backward_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: softmax_output.buffer().as_entire_binding(),
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
            label: Some("Softmax Backward Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.backward_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows, 1, 1);

        drop(pass);

        tracing::debug!(
            "Softmax backward dispatched: rows={} cols={}",
            rows, cols
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    /// Reference CPU softmax backward.
    /// dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
    fn softmax_backward_ref(
        grad_output: &[f32],
        softmax_out: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let mut grad_input = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            let dy = &grad_output[base..base + cols];
            let s = &softmax_out[base..base + cols];

            // sum_j(dy_j * s_j)
            let dot: f32 = dy.iter().zip(s.iter()).map(|(&d, &sv)| d * sv).sum();

            // dL/dx_i = s_i * (dy_i - dot)
            for (i, (&d, &sv)) in dy.iter().zip(s.iter()).enumerate() {
                grad_input[base + i] = sv * (d - dot);
            }
        }
        grad_input
    }

    #[test]
    fn test_softmax_backward_reference_identity() {
        // If softmax output is [1, 0, 0, ...] and grad_output = [1, 0, 0, ...],
        // then dot = 1*1 = 1, and dL/dx_0 = 1*(1-1) = 0, dL/dx_i = 0*(0-1) = 0
        let s = vec![1.0f32, 0.0, 0.0, 0.0];
        let dy = vec![1.0f32, 0.0, 0.0, 0.0];
        let grad = softmax_backward_ref(&dy, &s, 1, 4);
        for &g in &grad {
            assert!(g.abs() < 1e-6, "Expected 0, got {}", g);
        }
    }

    #[test]
    fn test_softmax_backward_reference_uniform() {
        // Uniform softmax: s = [0.25, 0.25, 0.25, 0.25]
        // If dy = [1, 0, 0, 0], dot = 0.25
        // dL/dx = [0.25*(1-0.25), 0.25*(0-0.25), ...] = [0.1875, -0.0625, ...]
        let s = vec![0.25f32; 4];
        let dy = vec![1.0f32, 0.0, 0.0, 0.0];
        let grad = softmax_backward_ref(&dy, &s, 1, 4);

        assert!((grad[0] - 0.1875).abs() < 1e-5, "Expected 0.1875, got {}", grad[0]);
        assert!((grad[1] - (-0.0625)).abs() < 1e-5, "Expected -0.0625, got {}", grad[1]);
        assert!((grad[2] - (-0.0625)).abs() < 1e-5);
        assert!((grad[3] - (-0.0625)).abs() < 1e-5);

        // Gradient should sum to 0 (softmax is shift-invariant)
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-5, "Gradient sum should be 0, got {}", sum);
    }

    #[test]
    fn test_softmax_backward_reference_sum_to_zero() {
        // General property: sum of softmax gradients = 0
        let s = vec![0.1f32, 0.2, 0.3, 0.4];
        let dy = vec![1.0f32, 2.0, 3.0, 4.0];
        let grad = softmax_backward_ref(&dy, &s, 1, 4);
        let sum: f32 = grad.iter().sum();
        assert!(sum.abs() < 1e-4, "Gradient sum should be ~0, got {}", sum);
    }

    #[test]
    fn test_softmax_backward_reference_multirow() {
        let s = vec![0.5f32, 0.5, 0.3, 0.7];
        let dy = vec![1.0f32, 0.0, 0.0, 1.0];
        let grad = softmax_backward_ref(&dy, &s, 2, 2);

        // Row 0: dot = 1*0.5 + 0*0.5 = 0.5
        // dL/dx = [0.5*(1-0.5), 0.5*(0-0.5)] = [0.25, -0.25]
        assert!((grad[0] - 0.25).abs() < 1e-5);
        assert!((grad[1] - (-0.25)).abs() < 1e-5);

        // Row 1: dot = 0*0.3 + 1*0.7 = 0.7
        // dL/dx = [0.3*(0-0.7), 0.7*(1-0.7)] = [-0.21, 0.21]
        assert!((grad[2] - (-0.21)).abs() < 1e-5);
        assert!((grad[3] - 0.21).abs() < 1e-5);
    }
}
