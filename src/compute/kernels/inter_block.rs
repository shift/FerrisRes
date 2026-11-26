//! Causal kernels for Block AttnRes inter-block attention.
//!
//! Two small kernels make the GPU path causal:
//!
//! * [`CausalPrefixOp`] turns a `[rows, hidden]` matrix into its per-row
//!   causal prefix mean, `out[t] = sum_{s <= t} in[s] / ((t + 1) * divisor)`.
//! * [`InterBlockAttentionOp`] runs the inter-block softmax attention with
//!   one query per batch row and per-position keys laid out
//!   `[entries, batch, hidden]`.
//!
//! Both follow the same style as [`super::prefill_attn`]: one invocation per
//! output unit, serial reduction inside the kernel, no temporary buffers.

use std::sync::Arc;
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, BufferDescriptor, BufferUsages, Device,
    Queue, ShaderStages,
};

use crate::compute::GpuBuffer;
use crate::error::Result;

const CAUSAL_PREFIX_WGSL: &str = r#"
    struct Params {
        rows:    u32,
        hidden:  u32,
        divisor: f32,
        _pad:    u32,
    }

    @group(0) @binding(0) var<storage, read>       input:  array<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;
    @group(0) @binding(2) var<uniform>             params: Params;

    // One thread per column, serial running sum over rows.
    @compute @workgroup_size(64)
    fn causal_prefix_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let d = gid.x;
        if (d >= params.hidden) {
            return;
        }
        var run: f32 = 0.0;
        for (var t: u32 = 0u; t < params.rows; t = t + 1u) {
            let idx = t * params.hidden + d;
            run = run + input[idx];
            output[idx] = run / (f32(t + 1u) * params.divisor);
        }
    }
"#;

const INTER_BLOCK_WGSL: &str = r#"
    struct Params {
        batch:   u32,
        entries: u32,
        hidden:  u32,
        scale:   f32,
    }

    @group(0) @binding(0) var<storage, read>       query:  array<f32>;
    @group(0) @binding(1) var<storage, read>       keys:   array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    @group(0) @binding(3) var<uniform>             params: Params;

    // One invocation per batch position. `keys` is [entries, batch, hidden],
    // so the key for entry e at position b starts at (e * batch + b) * hidden.
    @compute @workgroup_size(1)
    fn inter_block_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let b = gid.x;
        if (b >= params.batch) {
            return;
        }
        let hidden = params.hidden;
        let qbase = b * hidden;

        var max_score: f32 = -3.402823466e+38;
        for (var e: u32 = 0u; e < params.entries; e = e + 1u) {
            let kbase = (e * params.batch + b) * hidden;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hidden; d = d + 1u) {
                dot += query[qbase + d] * keys[kbase + d];
            }
            let s = dot * params.scale;
            if (s > max_score) {
                max_score = s;
            }
        }

        var sum_exp: f32 = 0.0;
        for (var e: u32 = 0u; e < params.entries; e = e + 1u) {
            let kbase = (e * params.batch + b) * hidden;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hidden; d = d + 1u) {
                dot += query[qbase + d] * keys[kbase + d];
            }
            sum_exp += exp(dot * params.scale - max_score);
        }
        let inv_sum = 1.0 / sum_exp;

        for (var d: u32 = 0u; d < hidden; d = d + 1u) {
            output[qbase + d] = 0.0;
        }
        for (var e: u32 = 0u; e < params.entries; e = e + 1u) {
            let kbase = (e * params.batch + b) * hidden;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < hidden; d = d + 1u) {
                dot += query[qbase + d] * keys[kbase + d];
            }
            let w = exp(dot * params.scale - max_score) * inv_sum;
            for (var d: u32 = 0u; d < hidden; d = d + 1u) {
                output[qbase + d] = output[qbase + d] + w * keys[kbase + d];
            }
        }
    }
"#;

fn two_storage_bindings() -> [BindGroupLayoutEntry; 3] {
    [
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
    ]
}

fn three_storage_bindings() -> [BindGroupLayoutEntry; 4] {
    [
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
    ]
}

fn uniform_buffer(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    label: &str,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// Per-position causal prefix mean.
pub struct CausalPrefixOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl CausalPrefixOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Causal Prefix Shader"),
            source: wgpu::ShaderSource::Wgsl(CAUSAL_PREFIX_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Causal Prefix BGL"),
            entries: &two_storage_bindings(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Causal Prefix Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Causal Prefix Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("causal_prefix_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Ok(Self {
            pipeline,
            bgl,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        })
    }

    /// `out[t] = sum_{s <= t} input[s] / ((t + 1) * divisor)`.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: u32,
        hidden: u32,
        divisor: f32,
    ) -> Result<()> {
        let params: [u8; 16] = {
            let mut bytes = [0u8; 16];
            bytes[0..4].copy_from_slice(&rows.to_le_bytes());
            bytes[4..8].copy_from_slice(&hidden.to_le_bytes());
            bytes[8..12].copy_from_slice(&divisor.to_le_bytes());
            bytes
        };
        let params_buffer =
            uniform_buffer(&self.device, &self.queue, "Causal Prefix Params", &params);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Causal Prefix Bind Group"),
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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Causal Prefix Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((hidden + 63) / 64, 1, 1);
        Ok(())
    }
}

/// Per-position inter-block softmax attention.
pub struct InterBlockAttentionOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl InterBlockAttentionOp {
    pub fn new(device: &Arc<Device>, queue: &Arc<Queue>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Inter Block Attention Shader"),
            source: wgpu::ShaderSource::Wgsl(INTER_BLOCK_WGSL.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Inter Block Attention BGL"),
            entries: &three_storage_bindings(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Inter Block Attention Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Inter Block Attention Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("inter_block_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Ok(Self {
            pipeline,
            bgl,
            device: Arc::clone(device),
            queue: Arc::clone(queue),
        })
    }

    /// `query` is `[batch, hidden]`, `keys` is `[entries, batch, hidden]`, and
    /// `output` is `[batch, hidden]`.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuBuffer,
        keys: &GpuBuffer,
        output: &GpuBuffer,
        batch: u32,
        entries: u32,
        hidden: u32,
        scale: f32,
    ) -> Result<()> {
        let params: [u8; 16] = {
            let mut bytes = [0u8; 16];
            bytes[0..4].copy_from_slice(&batch.to_le_bytes());
            bytes[4..8].copy_from_slice(&entries.to_le_bytes());
            bytes[8..12].copy_from_slice(&hidden.to_le_bytes());
            bytes[12..16].copy_from_slice(&scale.to_le_bytes());
            bytes
        };
        let params_buffer =
            uniform_buffer(&self.device, &self.queue, "Inter Block Params", &params);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Inter Block Attention Bind Group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keys.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Inter Block Attention Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(batch, 1, 1);
        Ok(())
    }
}
