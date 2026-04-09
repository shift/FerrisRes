use std::sync::Arc;
use wgpu::{
    BufferDescriptor, BufferUsages, BindGroupLayoutEntry, BindingType, BufferBindingType,
    Device, ShaderStages,
};
use crate::compute::GpuBuffer;
use crate::error::Result;

const FLASH_DECODE_WGSL: &str = r#"
    struct Params {
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
        _pad: u32,
    }

    @group(0) @binding(0) var<storage, read> query: array<f32>;
    @group(0) @binding(1) var<storage, read> key_cache: array<f32>;
    @group(0) @binding(2) var<storage, read> value_cache: array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform> params: Params;

    @compute @workgroup_size(256)
    fn flash_decode_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let h = gid.x;
        let num_heads = params.num_heads;
        let head_dim = params.head_dim;
        let seq_len = params.seq_len;

        if (h >= num_heads) {
            return;
        }

        let scale = 1.0 / sqrt(f32(head_dim));

        var max_score: f32 = -3.402823466e+38;

        for (var s: u32 = 0u; s < seq_len; s = s + 1u) {
            var dot: f32 = 0.0;
            let q_base = h * head_dim;
            let k_base = s * num_heads * head_dim + h * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot += query[q_base + d] * key_cache[k_base + d];
            }
            let score = dot * scale;
            if (score > max_score) {
                max_score = score;
            }
        }

        var sum_exp: f32 = 0.0;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[h * head_dim + d] = 0.0;
        }

        for (var s: u32 = 0u; s < seq_len; s = s + 1u) {
            var dot: f32 = 0.0;
            let q_base = h * head_dim;
            let k_base = s * num_heads * head_dim + h * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot += query[q_base + d] * key_cache[k_base + d];
            }
            let weight = exp(dot * scale - max_score);
            sum_exp += weight;

            let v_base = s * num_heads * head_dim + h * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                output[h * head_dim + d] += weight * value_cache[v_base + d];
            }
        }

        let inv_sum = 1.0 / sum_exp;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[h * head_dim + d] *= inv_sum;
        }
    }
"#;

pub struct FlashDecodeOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl FlashDecodeOp {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flash Decode Shader"),
            source: wgpu::ShaderSource::Wgsl(FLASH_DECODE_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flash Decode BGL"),
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
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("Flash Decode Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flash Decode Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("flash_decode_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bgl,
            device: Arc::clone(device),
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuBuffer,
        key_cache: &GpuBuffer,
        value_cache: &GpuBuffer,
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
            label: Some("Flash Decode Params"),
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
            label: Some("Flash Decode Bind Group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: key_cache.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: value_cache.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let wg_count = (num_heads + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Flash Decode Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_count, 1, 1);

        drop(pass);

        tracing::debug!(
            "FlashDecodeOp dispatched: seq_len={} num_heads={} head_dim={}",
            seq_len, num_heads, head_dim
        );

        Ok(())
    }
}

pub fn dispatch_flash_decode(
    device: &Arc<Device>,
    encoder: &mut wgpu::CommandEncoder,
    query: &GpuBuffer,
    key_cache: &GpuBuffer,
    value_cache: &GpuBuffer,
    output: &GpuBuffer,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    let op = FlashDecodeOp::new(device)?;
    op.dispatch(encoder, query, key_cache, value_cache, output, seq_len, num_heads, head_dim)
}
