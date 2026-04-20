use std::sync::Arc;
use wgpu::{Device, Queue, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

const ELEMENTWISE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> ew_a: array<f32>;
@group(0) @binding(1) var<storage, read> ew_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> ew_c: array<f32>;
@group(0) @binding(3) var<uniform> ew_param: u32;

@compute @workgroup_size(256)
fn add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let a_val = ew_a[idx];
    let b_val = select(ew_b[idx], ew_b[idx % ew_param], ew_param > 0u);
    ew_c[idx] = a_val + b_val;
}

@compute @workgroup_size(256)
fn scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let s = bitcast<f32>(ew_param);
    ew_c[idx] = ew_a[idx] * s;
}

@compute @workgroup_size(256)
fn copy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    ew_c[idx] = ew_a[idx];
}

@compute @workgroup_size(256)
fn relu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let val = ew_a[idx];
    ew_c[idx] = select(0.0, val, val > 0.0);
}

@compute @workgroup_size(256)
fn gelu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let val = ew_a[idx];
    // GELU tanh approximation
    let sqrt_2_over_pi = 0.7978845608028654;
    let coeff = 0.044715;
    let inner = sqrt_2_over_pi * (val + coeff * val * val * val);
    ew_c[idx] = 0.5 * val * (1.0 + tanh(inner));
}

@compute @workgroup_size(256)
fn mul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    ew_c[idx] = ew_a[idx] * ew_b[idx];
}
"#;

pub struct ElementWiseOp {
    add_pipeline: wgpu::ComputePipeline,
    add_bind_group_layout: wgpu::BindGroupLayout,
    scale_pipeline: wgpu::ComputePipeline,
    scale_bind_group_layout: wgpu::BindGroupLayout,
    copy_pipeline: wgpu::ComputePipeline,
    copy_bind_group_layout: wgpu::BindGroupLayout,
    relu_pipeline: wgpu::ComputePipeline,
    relu_bind_group_layout: wgpu::BindGroupLayout,
    gelu_pipeline: wgpu::ComputePipeline,
    gelu_bind_group_layout: wgpu::BindGroupLayout,
    mul_pipeline: wgpu::ComputePipeline,
    mul_bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl ElementWiseOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Self {
        tracing::debug!(event = "creating_elementwiseop_compute_pipelines", "Creating ElementWiseOp compute pipelines");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ElementWise Shader"),
            source: wgpu::ShaderSource::Wgsl(ELEMENTWISE_WGSL.into()),
        });

        let read_entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let read_entry_b = BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let rw_entry = BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let uniform_entry = BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let add_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ElementWise Add Bind Group Layout"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone(), uniform_entry.clone()],
        });

        let add_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise Add Pipeline Layout"),
            bind_group_layouts: &[Some(&add_bind_group_layout)],
            immediate_size: 0,
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
            entries: &[read_entry.clone(), rw_entry.clone(), uniform_entry.clone()],
        });

        let scale_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise Scale Pipeline Layout"),
            bind_group_layouts: &[Some(&scale_bind_group_layout)],
            immediate_size: 0,
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

        let relu_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ElementWise ReLU Bind Group Layout"),
            entries: &[read_entry.clone(), rw_entry.clone()],
        });

        let relu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise ReLU Pipeline Layout"),
            bind_group_layouts: &[Some(&relu_bind_group_layout)],
            immediate_size: 0,
        });

        let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElementWise ReLU Pipeline"),
            layout: Some(&relu_pipeline_layout),
            module: &shader,
            entry_point: Some("relu_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // GELU pipeline uses same BGL as ReLU (a + c bindings)
        let gelu_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ElementWise GELU BGL"),
            entries: &[read_entry.clone(), rw_entry.clone()],
        });

        let gelu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise GELU Pipeline Layout"),
            bind_group_layouts: &[Some(&gelu_bind_group_layout)],
            immediate_size: 0,
        });

        let gelu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElementWise GELU Pipeline"),
            layout: Some(&gelu_pipeline_layout),
            module: &shader,
            entry_point: Some("gelu_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Multiply pipeline uses a + b + c bindings (same as add)
        let mul_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ElementWise Mul BGL"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone()],
        });

        let mul_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ElementWise Mul Pipeline Layout"),
            bind_group_layouts: &[Some(&mul_bind_group_layout)],
            immediate_size: 0,
        });

        let mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ElementWise Mul Pipeline"),
            layout: Some(&mul_pipeline_layout),
            module: &shader,
            entry_point: Some("mul_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!(event = "elementwiseop_pipelines_created_successfully", "ElementWiseOp pipelines created successfully");

        Self {
            add_pipeline,
            add_bind_group_layout,
            scale_pipeline,
            scale_bind_group_layout,
            copy_pipeline,
            copy_bind_group_layout,
            relu_pipeline,
            relu_bind_group_layout,
            gelu_pipeline,
            gelu_bind_group_layout,
            mul_pipeline,
            mul_bind_group_layout,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
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
        let same_buf = std::ptr::eq(a.buffer() as *const _, c.buffer() as *const _);
        if same_buf {
            let tmp = GpuBuffer::new(
                &self.device,
                c.size(),
                Some("ew_add_tmp"),
            )?;
            self.dispatch_add_impl(encoder, a, b, &tmp, numel)?;
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ElementWise Add Copy Pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.copy_pipeline);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ElementWise Add Copy BG"),
                layout: &self.copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tmp.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: c.buffer().as_entire_binding() },
                ],
            });
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((numel + 255) / 256, 1, 1);
            drop(cp);
            return Ok(());
        }
        self.dispatch_add_impl(encoder, a, b, c, numel)
    }

    fn dispatch_add_impl(
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

        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("ElementWise Add Params"),
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, &stride.to_le_bytes());

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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ElementWise Add Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.add_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
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
        let same_buf = std::ptr::eq(a.buffer() as *const _, c.buffer() as *const _);
        if same_buf {
            let tmp = GpuBuffer::new(
                &self.device,
                c.size(),
                Some("ew_scale_tmp"),
            )?;
            self.dispatch_scale_impl(encoder, a, &tmp, scale, numel)?;
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ElementWise Scale Copy Pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.copy_pipeline);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ElementWise Scale Copy BG"),
                layout: &self.copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tmp.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: c.buffer().as_entire_binding() },
                ],
            });
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups((numel + 255) / 256, 1, 1);
            drop(cp);
            return Ok(());
        }
        self.dispatch_scale_impl(encoder, a, c, scale, numel)
    }

    fn dispatch_scale_impl(
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

        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("ElementWise Scale Params"),
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, &scale.to_le_bytes());

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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ElementWise Scale Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.scale_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
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

    pub fn dispatch_relu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        c: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let workgroup_count = (numel + 255) / 256;

        tracing::debug!(
            "ElementWiseOp::dispatch_relu numel={} workgroups={}",
            numel, workgroup_count
        );

        // wgpu forbids binding the same buffer as both read-only and read-write
        // in the same dispatch scope. When in-place (a == c), use a temporary.
        let same_buf = std::ptr::eq(a.buffer() as *const _, c.buffer() as *const _);
        if same_buf {
            let tmp = GpuBuffer::new(
                &self.device,
                c.size(),
                Some("ew_relu_tmp"),
            )?;
            self.dispatch_relu_impl(encoder, a, &tmp, numel)?;
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ElementWise ReLU Copy Pass"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&self.copy_pipeline);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ElementWise ReLU Copy BG"),
                layout: &self.copy_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: tmp.buffer().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: c.buffer().as_entire_binding() },
                ],
            });
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(workgroup_count, 1, 1);
            drop(cp);
            return Ok(());
        }
        self.dispatch_relu_impl(encoder, a, c, numel)
    }

    fn dispatch_relu_impl(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        c: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let workgroup_count = (numel + 255) / 256;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ElementWise ReLU Bind Group"),
            layout: &self.relu_bind_group_layout,
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
            label: Some("ElementWise ReLU Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.relu_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }

    /// GELU activation (tanh approximation): c = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a³)))
    pub fn dispatch_gelu(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        c: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let workgroup_count = (numel + 255) / 256;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ElementWise GELU Bind Group"),
            layout: &self.gelu_bind_group_layout,
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
            label: Some("ElementWise GELU Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.gelu_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }

    /// Element-wise multiply: c = a * b
    pub fn dispatch_mul(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let workgroup_count = (numel + 255) / 256;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ElementWise Mul Bind Group"),
            layout: &self.mul_bind_group_layout,
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
            label: Some("ElementWise Mul Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.mul_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }
}
