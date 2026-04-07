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

var<workgroup> shared_sum_sq: array<f32, 1024>;

@compute @workgroup_size(1024)
fn rmsnorm_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let col = lid.x;
    let hidden_dim = params.hidden_dim;
    let warp_size = 32u;

    var sum_sq = 0.0;
    if (col < hidden_dim) {
        let val = input[row * hidden_dim + col];
        sum_sq = val * val;
    }

    shared_sum_sq[col] = sum_sq;
    workgroupBarrier();

    var lane = col % warp_size;
    var wid = col / warp_size;

    for (var offset = warp_size / 2u; offset > 0u; offset = offset / 2u) {
        if (lane < offset) {
            shared_sum_sq[wid * warp_size + lane] += shared_sum_sq[wid * warp_size + lane + offset];
        }
        workgroupBarrier();
    }

    if (wid == 0u) {
        var warp_sum = 0.0;
        let num_warps = (hidden_dim + warp_size - 1u) / warp_size;
        for (var i = 0u; i < num_warps; i++) {
            warp_sum += shared_sum_sq[i * warp_size];
        }
        let inv_rms = inverseSqrt(warp_sum / f32(hidden_dim) + 1e-5);
        shared_sum_sq[0u] = inv_rms;
    }
    workgroupBarrier();

    if (col < hidden_dim) {
        let val = input[row * hidden_dim + col];
        output[row * hidden_dim + col] = val * shared_sum_sq[0u];
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

        if hidden_dim > 1024 {
            return Err(FerrisResError::Unsupported(format!(
                "RmsNorm hidden_dim {} exceeds max workgroup size 1024",
                hidden_dim
            )));
        }

        let params_data: [u32; 2] = [hidden_dim, rows];

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("RmsNorm Params"),
            size: 8,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

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
