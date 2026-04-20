use std::sync::Arc;
use wgpu::{Device, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

const ROPE_WGSL: &str = r#"
struct Params {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    start_pos: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn rope_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;
    let total = seq_len * num_heads * head_dim;

    if (flat_idx >= total) {
        return;
    }

    let pos = params.start_pos + flat_idx / (num_heads * head_dim);
    let head_offset = flat_idx % (num_heads * head_dim);
    let dim_idx = head_offset % head_dim;

    if (dim_idx % 2u != 0u) {
        output[flat_idx] = input[flat_idx];
        return;
    }

    let pair_idx = dim_idx / 2u;
    let base = 10000.0;
    let freq = 1.0 / pow(base, f32(2u * pair_idx) / f32(head_dim));
    let theta = f32(pos) * freq;

    // Cody-Waite range reduction for precision at high positions (>100k).
    let TWO_PI_HI = 6.28125;
    let TWO_PI_LO = 0.001935308;
    let n = floor(theta / 6.283185307);
    let theta_red = ((theta - n * TWO_PI_HI) - n * TWO_PI_LO);

    let cos_t = cos(theta_red);
    let sin_t = sin(theta_red);

    // Buffer index is always local — input/output sized for current batch only.
    // pos is used only for angle (theta) computation above; flat_idx addresses the buffer.
    let x0 = input[flat_idx];
    let x1 = input[flat_idx + 1u];

    output[flat_idx] = x0 * cos_t - x1 * sin_t;
    output[flat_idx + 1u] = x0 * sin_t + x1 * cos_t;
}
"#;

/// In-place RoPE: reads and writes the same buffer.
/// Each invocation only accesses its own pair of elements, so there's no hazard.
const ROPE_INPLACE_WGSL: &str = r#"
struct Params {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    start_pos: u32,
}

@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn rope_inplace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let seq_len = params.seq_len;
    let num_heads = params.num_heads;
    let head_dim = params.head_dim;
    let total = seq_len * num_heads * head_dim;

    if (flat_idx >= total) {
        return;
    }

    let dim_idx = flat_idx % head_dim;

    // Only process even dims — each invocation handles its pair
    if (dim_idx % 2u != 0u) {
        return;
    }

    let pos = params.start_pos + flat_idx / (num_heads * head_dim);
    let pair_idx = dim_idx / 2u;
    let base = 10000.0;
    let freq = 1.0 / pow(base, f32(2u * pair_idx) / f32(head_dim));
    let theta = f32(pos) * freq;

    // Cody-Waite range reduction for precision at high positions (>100k).
    let TWO_PI_HI = 6.28125;
    let TWO_PI_LO = 0.001935308;
    let n = floor(theta / 6.283185307);
    let theta_red = ((theta - n * TWO_PI_HI) - n * TWO_PI_LO);

    let cos_t = cos(theta_red);
    let sin_t = sin(theta_red);

    let x0 = buf[flat_idx];
    let x1 = buf[flat_idx + 1u];

    buf[flat_idx]     = x0 * cos_t - x1 * sin_t;
    buf[flat_idx + 1u] = x0 * sin_t + x1 * cos_t;
}
"#;

pub struct RopeOp {
    pipeline: wgpu::ComputePipeline,
    /// In-place pipeline: reads and writes same buffer (no separate input/output)
    inplace_pipeline: wgpu::ComputePipeline,
    /// BGL for the regular 3-binding pipeline (input, output, params)
    bgl: wgpu::BindGroupLayout,
    /// BGL for the in-place 2-binding pipeline (buf, params)
    inplace_bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl RopeOp {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RoPE Shader"),
            source: wgpu::ShaderSource::Wgsl(ROPE_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RoPE Bind Group Layout"),
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
            label: Some("RoPE Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RoPE Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("rope_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::debug!(event = "ropeop_pipeline_created", "RopeOp pipeline created");

        // --- In-place pipeline ---
        let inplace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RoPE In-Place Shader"),
            source: wgpu::ShaderSource::Wgsl(ROPE_INPLACE_WGSL.into()),
        });

        let inplace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RoPE In-Place BGL"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
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

        let inplace_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RoPE In-Place Pipeline Layout"),
            bind_group_layouts: &[Some(&inplace_bgl)],
            immediate_size: 0,
        });

        let inplace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RoPE In-Place Pipeline"),
            layout: Some(&inplace_layout),
            module: &inplace_shader,
            entry_point: Some("rope_inplace_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bgl,
            inplace_pipeline,
            inplace_bgl,
            device: Arc::clone(device),
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        if seq_len == 0 || num_heads == 0 || head_dim == 0 {
            return Ok(());
        }

        let params_data: [u32; 4] = [seq_len, num_heads, head_dim, 0];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("RoPE Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RoPE Bind Group"),
            layout: &self.bgl,
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

        let total_elements = seq_len * num_heads * head_dim;
        let wg_count = (total_elements + 256 - 1) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RoPE Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);

        drop(pass);

        tracing::debug!(
            "RopeOp dispatched: seq_len={} num_heads={} head_dim={}",
            seq_len,
            num_heads,
            head_dim
        );

        Ok(())
    }

    pub fn dispatch_with_offset(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        start_pos: u32,
    ) -> Result<()> {
        if seq_len == 0 || num_heads == 0 || head_dim == 0 {
            return Ok(());
        }

        let params_data: [u32; 4] = [seq_len, num_heads, head_dim, start_pos];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("RoPE Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RoPE Bind Group"),
            layout: &self.bgl,
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

        let total_elements = seq_len * num_heads * head_dim;
        let wg_count = (total_elements + 256 - 1) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RoPE Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);

        drop(pass);

        tracing::debug!(
            "RopeOp dispatched: seq_len={} num_heads={} head_dim={} start_pos={}",
            seq_len,
            num_heads,
            head_dim,
            start_pos
        );

        Ok(())
    }

    /// Apply RoPE in-place on the given buffer, with a byte offset into the buffer.
    ///
    /// This is used for the direct-write decode path: the K projection writes
    /// directly into the KV cache slot, then this method applies RoPE in-place
    /// on that same cache slot. No separate copy needed.
    ///
    /// # Safety
    /// Each GPU thread reads and writes only its own pair of f32 elements,
    /// so there are no data races despite read-write on the same buffer.
    pub fn dispatch_inplace_at_offset(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        buffer: &wgpu::Buffer,
        byte_offset: u64,
        byte_size: u64,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        start_pos: u32,
    ) -> Result<()> {
        if seq_len == 0 || num_heads == 0 || head_dim == 0 {
            return Ok(());
        }

        let params_data: [u32; 4] = [seq_len, num_heads, head_dim, start_pos];
        let params_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("RoPE In-Place Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&params_data));
        params_buffer.unmap();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RoPE In-Place Bind Group"),
            layout: &self.inplace_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: byte_offset,
                        size: Some(std::num::NonZeroU64::new(byte_size).unwrap_or(std::num::NonZeroU64::new(1).unwrap())),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let total_elements = seq_len * num_heads * head_dim;
        let wg_count = (total_elements + 256 - 1) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RoPE In-Place Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.inplace_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);

        drop(pass);

        tracing::debug!(
            "RopeOp in-place dispatched: seq_len={} num_heads={} head_dim={} start_pos={}",
            seq_len, num_heads, head_dim, start_pos
        );

        Ok(())
    }
}
