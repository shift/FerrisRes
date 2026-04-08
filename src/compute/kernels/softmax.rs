use std::sync::Arc;
use wgpu::{Device, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
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

pub struct SoftmaxOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl SoftmaxOp {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
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

        tracing::debug!("SoftmaxOp pipeline created");

        Ok(Self {
            pipeline,
            bind_group_layout,
            device: Arc::clone(device),
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
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

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

        tracing::debug!("SoftmaxOp dispatched: rows={} cols={}", rows, cols);

        Ok(())
    }
}
