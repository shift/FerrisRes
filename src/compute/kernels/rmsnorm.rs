use wgpu::{
    BindGroupLayout, BufferDescriptor, BufferUsages, ComputePipeline, Device,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};
use crate::compute::GpuBuffer;
use crate::error::{FerrisResError, Result};

const RMSNORM_WGSL: &str = r#"
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

pub struct RmsNormOp {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl RmsNormOp {
    pub fn new(device: &Device) -> Result<Self> {
        tracing::info!("Creating RmsNormOp compute pipeline");

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

        Ok(Self {
            pipeline,
            bind_group_layout,
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
}
