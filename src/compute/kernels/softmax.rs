use std::sync::Arc;
use wgpu::Device;
use crate::compute::GpuBuffer;
use crate::error::Result;

const SOFTMAX_WGSL: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<private> p: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_max: array<f32, WG_SIZE>;
var<workgroup> shared_sum: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn softmax_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row_idx = gid.x;
    if (row_idx >= p.rows) {
        return;
    }

    let cols = p.cols;
    let row_offset = row_idx * cols;
    let local_id = local_invocation_index;

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
            let other = shared_max[local_id + stride];
            if (other > shared_max[local_id]) {
                shared_max[local_id] = other;
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
        local_sum += val;
    }

    shared_sum[local_id] = local_sum;
    workgroupBarrier();

    stride = WG_SIZE / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Softmax Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 8,
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
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Softmax Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[
            rows.to_le_bytes(),
            cols.to_le_bytes(),
        ].concat());

        pass.dispatch_workgroups(rows, 1, 1);

        drop(pass);

        tracing::debug!("SoftmaxOp dispatched: rows={} cols={}", rows, cols);

        Ok(())
    }
}
