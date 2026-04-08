use std::sync::Arc;
use wgpu::{Device, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

const SHADER: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 16 * 16>;
var<workgroup> tile_b: array<f32, 16 * 16>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    let local_row = lid.x;
    let local_col = lid.y;
    let tile_size = 16u;

    var acc: f32 = 0.0;
    let num_tiles = (params.K + tile_size - 1u) / tile_size;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * tile_size + local_col;
        let b_row = t * tile_size + local_row;

        let a_valid = row < params.M && a_col < params.K;
        let b_valid = b_row < params.K && col < params.N;

        if (a_valid) {
            tile_a[local_row * tile_size + local_col] = a[row * params.K + a_col];
        } else {
            tile_a[local_row * tile_size + local_col] = 0.0;
        }

        if (b_valid) {
            tile_b[local_row * tile_size + local_col] = b[b_row * params.N + col];
        } else {
            tile_b[local_row * tile_size + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < tile_size; i = i + 1u) {
            acc = acc + tile_a[local_row * tile_size + i] * tile_b[i * tile_size + local_col];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        c[row * params.N + col] = acc;
    }
}
"#;

pub struct MatMulOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl MatMulOp {
    pub fn new(device: &Arc<Device>) -> Self {
        tracing::debug!("Creating MatMulOp compute pipeline");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Tiled Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Bind Group Layout"),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Tiled Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("matmul"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!("MatMulOp pipeline created successfully");
        Self {
            pipeline,
            bind_group_layout,
            device: Arc::clone(device),
        }
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        let tile_size = 16u32;
        let workgroup_count_x = (m + tile_size - 1) / tile_size;
        let workgroup_count_y = (n + tile_size - 1) / tile_size;

        tracing::debug!(
            "MatMulOp dispatch: M={} K={} N={} workgroups=({},{},1)",
            m, k, n, workgroup_count_x, workgroup_count_y
        );

        let params_data: [u32; 3] = [m, k, n];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("MatMul Params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);

        drop(pass);

        Ok(())
    }
}
