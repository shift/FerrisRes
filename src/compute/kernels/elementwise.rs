use std::sync::Arc;
use wgpu::Device;
use crate::compute::GpuBuffer;
use crate::error::Result;

const ELEMENTWISE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

var<private> stride: u32;

@compute @workgroup_size(256)
fn add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let a_val = a[idx];
    let b_val = select(b[idx], b[idx % stride], stride > 0u);
    c[idx] = a_val + b_val;
}

var<private> scale_val: f32;

@compute @workgroup_size(256)
fn scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    c[idx] = a[idx] * scale_val;
}

@compute @workgroup_size(256)
fn copy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    c[idx] = a[idx];
}
"#;

pub struct ElementWiseOp {
    add_pipeline: wgpu::ComputePipeline,
    add_bind_group_layout: wgpu::BindGroupLayout,
    scale_pipeline: wgpu::ComputePipeline,
    scale_bind_group_layout: wgpu::BindGroupLayout,
    copy_pipeline: wgpu::ComputePipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl ElementWiseOp {
    pub fn new(device: &Arc<Device>) -> Self {
        tracing::debug!("Creating ElementWiseOp compute pipelines");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ElementWise Shader"),
            source: wgpu::ShaderSource::Wgsl(ELEMENTWISE_WGSL.into()),
        });

        let read_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let read_entry_b = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let rw_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let add_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ElementWise Add Bind Group Layout"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone()],
        });

        let add_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise Add Pipeline Layout"),
            bind_group_layouts: &[Some(&add_bind_group_layout)],
            immediate_size: 4,
        });

        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElementWise Add Pipeline"),
            layout: Some(&add_pipeline_layout),
            module: &shader,
            entry_point: Some("add_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let scale_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ElementWise Scale Bind Group Layout"),
            entries: &[read_entry.clone(), rw_entry.clone()],
        });

        let scale_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise Scale Pipeline Layout"),
            bind_group_layouts: &[Some(&scale_bind_group_layout)],
            immediate_size: 4,
        });

        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElementWise Scale Pipeline"),
            layout: Some(&scale_pipeline_layout),
            module: &shader,
            entry_point: Some("scale_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let copy_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ElementWise Copy Bind Group Layout"),
            entries: &[read_entry.clone(), rw_entry.clone()],
        });

        let copy_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise Copy Pipeline Layout"),
            bind_group_layouts: &[Some(&copy_bind_group_layout)],
            immediate_size: 0,
        });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElementWise Copy Pipeline"),
            layout: Some(&copy_pipeline_layout),
            module: &shader,
            entry_point: Some("copy_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!("ElementWiseOp pipelines created successfully");

        Self {
            add_pipeline,
            add_bind_group_layout,
            scale_pipeline,
            scale_bind_group_layout,
            copy_pipeline,
            copy_bind_group_layout,
            device: Arc::clone(device),
        }
    }

    pub fn dispatch_add(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        self.dispatch_add_with_stride(encoder, a, b, c, numel, 0)
    }

    pub fn dispatch_add_with_stride(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        numel: u32,
        stride: u32,
    ) -> Result<()> {
        let workgroup_count = (numel + 255) / 256;

        tracing::debug!(
            "ElementWiseOp::dispatch_add numel={} stride={} workgroups={}",
            numel, stride, workgroup_count
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ElementWise Add Bind Group"),
            layout: &self.add_bind_group_layout,
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
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ElementWise Add Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.add_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &stride.to_le_bytes());
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }

    pub fn dispatch_scale(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        c: &GpuBuffer,
        scale: f32,
        numel: u32,
    ) -> Result<()> {
        let workgroup_count = (numel + 255) / 256;

        tracing::debug!(
            "ElementWiseOp::dispatch_scale numel={} scale={} workgroups={}",
            numel, scale, workgroup_count
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ElementWise Scale Bind Group"),
            layout: &self.scale_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buffer().as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ElementWise Scale Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.scale_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &scale.to_le_bytes());
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }

    pub fn dispatch_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        c: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let workgroup_count = (numel + 255) / 256;

        tracing::debug!(
            "ElementWiseOp::dispatch_copy numel={} workgroups={}",
            numel, workgroup_count
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ElementWise Copy Bind Group"),
            layout: &self.copy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.buffer().as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ElementWise Copy Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.copy_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }
}
