use wgpu::{
    BindGroupLayout, BufferDescriptor, BufferUsages, ComputePipeline, Device,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};
use crate::compute::GpuBuffer;
use crate::error::Result;

pub const CAUSAL_MASK_WGSL: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn causal_mask_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let total = params.rows * params.cols;
    if (flat_idx >= total) {
        return;
    }

    let rows = params.rows;
    let cols = params.cols;
    let row = flat_idx / cols;
    let col = flat_idx % cols;
    let local_row = row % cols;

    let v = scores[flat_idx];
    if (col > local_row) {
        output[flat_idx] = -1e9;
    } else {
        output[flat_idx] = v;
    }
}
"#;

pub struct CausalMaskOp {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl CausalMaskOp {
    pub fn new(device: &Device) -> Result<Self> {
        tracing::info!(event = "creating_causalmaskop_compute_pipeline", "Creating CausalMaskOp compute pipeline");

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("CausalMask Shader"),
            source: ShaderSource::Wgsl(CAUSAL_MASK_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CausalMask Bind Group Layout"),
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
            label: Some("CausalMask Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CausalMask Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("causal_mask_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    pub fn dispatch_causal_mask(
        &self,
        device: &Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scores: &GpuBuffer,
        output: &GpuBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        tracing::debug!(event = "causalmaskop_dispatch_causal_mask_rows_cols", "CausalMaskOp::dispatch_causal_mask rows={} cols={}", rows, cols);

        let params_data: [u32; 4] = [rows, cols, 0, 0];

        let params_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("CausalMask Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params_data));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CausalMask Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scores.buffer().as_entire_binding(),
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

        let total_elements = rows * cols;
        let wg_size = 256u32;
        let workgroup_count = (total_elements + wg_size - 1) / wg_size;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("CausalMask Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        Ok(())
    }
}
