//! Portable, immutable parameter bindings for GPU training operations.
use wgpu::util::DeviceExt;

pub(crate) struct KernelParams {
    pub(crate) layout: wgpu::BindGroupLayout,
}

impl KernelParams {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Training parameter layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        Self { layout }
    }

    /// Each dispatch owns its initialized buffer, so later parameter updates
    /// cannot change commands recorded earlier in the same encoder.
    pub(crate) fn bind(&self, device: &wgpu::Device, pass: &mut wgpu::ComputePass<'_>, bytes: &[u8]) {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Training parameters"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Training parameter bind group"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        pass.set_bind_group(1, &group, &[]);
    }
}
