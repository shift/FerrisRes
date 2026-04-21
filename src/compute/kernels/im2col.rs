use std::sync::Arc;
use wgpu::{Device, Queue, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

pub const IM2COL_WGSL: &str = r#"
struct Params {
    height: u32,
    width: u32,
    channels: u32,
    patch_size: u32,
}

@group(0) @binding(0) var<storage, read> input_img: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_patches: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn im2col_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let patch_elems = params.patch_size * params.patch_size * params.channels;
    let patch_idx = flat_idx / patch_elems;
    let elem_idx = flat_idx % patch_elems;

    let patches_per_row = params.width / params.patch_size;
    let patch_row = patch_idx / patches_per_row;
    let patch_col = patch_idx % patches_per_row;

    let c = elem_idx % params.channels;
    let local_xy = elem_idx / params.channels;
    let local_y = local_xy / params.patch_size;
    let local_x = local_xy % params.patch_size;

    let img_y = patch_row * params.patch_size + local_y;
    let img_x = patch_col * params.patch_size + local_x;

    let input_idx = ((img_y * params.width + img_x) * params.channels) + c;
    output_patches[flat_idx] = input_img[input_idx];
}
"#;

pub struct Im2ColOp {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl Im2ColOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Self {
        tracing::debug!(event = "creating_im2colop_compute_pipeline", "Creating Im2ColOp compute pipeline");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Im2Col Shader"),
            source: wgpu::ShaderSource::Wgsl(IM2COL_WGSL.into()),
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

        let rw_entry = BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let uniform_entry = BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Im2Col Bind Group Layout"),
            entries: &[read_entry, rw_entry, uniform_entry],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Im2Col Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Im2Col Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("im2col_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!(event = "im2colop_pipeline_created_successfully", "Im2ColOp pipeline created successfully");

        Self {
            pipeline,
            bind_group_layout,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        }
    }

    pub fn dispatch_im2col(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        height: u32,
        width: u32,
        channels: u32,
        patch_size: u32,
    ) -> Result<()> {
        let num_patches = (height / patch_size) * (width / patch_size);
        let patch_elems = patch_size * patch_size * channels;
        let total_elems = num_patches * patch_elems;
        let workgroup_count = (total_elems + 255) / 256;

        tracing::debug!(
            "Im2ColOp::dispatch_im2col h={} w={} c={} ps={} patches={} total={} wg={}",
            height, width, channels, patch_size, num_patches, total_elems, workgroup_count
        );

        let params_data: [u8; 16] = [
            height.to_le_bytes(),
            width.to_le_bytes(),
            channels.to_le_bytes(),
            patch_size.to_le_bytes(),
        ].concat().try_into().unwrap();

        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Im2Col Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buffer, 0, &params_data);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Im2Col Bind Group"),
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
            label: Some("Im2Col Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }
}
