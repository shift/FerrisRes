use std::sync::Arc;
use wgpu::{
    BufferDescriptor, BufferUsages, BindGroupLayoutEntry, BindingType, BufferBindingType,
    Device, ShaderStages,
};
use crate::compute::GpuBuffer;
use crate::error::Result;

/// Prefill multi-head self-attention kernel.
///
/// Inputs Q, K, V all have layout `[seq_len, num_heads, head_dim]` (token-major).
/// Output has the same layout `[seq_len, num_heads, head_dim]`.
///
/// Each invocation handles one (query_pos, head) pair.
/// The kernel applies causal masking (positions after the query get -inf),
/// online softmax (numerically stable), and produces the weighted V sum.
pub const PREFILL_ATTN_WGSL: &str = r#"
    struct Params {
        seq_len:  u32,
        num_heads: u32,
        head_dim:  u32,
        _pad:      u32,
    }

    @group(0) @binding(0) var<storage, read>       q:      array<f32>;
    @group(0) @binding(1) var<storage, read>       k:      array<f32>;
    @group(0) @binding(2) var<storage, read>       v:      array<f32>;
    @group(0) @binding(3) var<storage, read_write> output: array<f32>;
    @group(0) @binding(4) var<uniform>             params: Params;

    // Each thread handles one (query_pos, head) pair.
    // gid.x = q_pos, gid.y = head index
    @compute @workgroup_size(16, 16)
    fn prefill_attn_main(
        @builtin(global_invocation_id) gid: vec3<u32>,
    ) {
        let q_pos     = gid.x;
        let h         = gid.y;
        let seq_len   = params.seq_len;
        let num_heads = params.num_heads;
        let head_dim  = params.head_dim;

        if (q_pos >= seq_len || h >= num_heads) {
            return;
        }

        let scale = 1.0 / sqrt(f32(head_dim));

        // Base offset of Q[q_pos, h, :]
        let q_base = q_pos * num_heads * head_dim + h * head_dim;

        // --- Online softmax: one pass for max, one pass for weighted sum ---

        var max_score: f32 = -3.402823466e+38;

        for (var k_pos: u32 = 0u; k_pos <= q_pos; k_pos = k_pos + 1u) {
            // K[k_pos, h, :]
            let k_base = k_pos * num_heads * head_dim + h * head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot += q[q_base + d] * k[k_base + d];
            }
            let score = dot * scale;
            if (score > max_score) {
                max_score = score;
            }
        }

        var sum_exp: f32 = 0.0;

        // Initialise output slice to zero
        let out_base = q_pos * num_heads * head_dim + h * head_dim;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[out_base + d] = 0.0;
        }

        for (var k_pos: u32 = 0u; k_pos <= q_pos; k_pos = k_pos + 1u) {
            let k_base = k_pos * num_heads * head_dim + h * head_dim;
            var dot: f32 = 0.0;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                dot += q[q_base + d] * k[k_base + d];
            }
            let weight = exp(dot * scale - max_score);
            sum_exp += weight;

            // V[k_pos, h, :]
            let v_base = k_pos * num_heads * head_dim + h * head_dim;
            for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
                output[out_base + d] += weight * v[v_base + d];
            }
        }

        let inv_sum = 1.0 / sum_exp;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            output[out_base + d] *= inv_sum;
        }
    }
"#;

pub struct PrefillAttnOp {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl PrefillAttnOp {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Prefill Attn Shader"),
            source: wgpu::ShaderSource::Wgsl(PREFILL_ATTN_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Prefill Attn BGL"),
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
            label: Some("Prefill Attn Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Prefill Attn Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("prefill_attn_main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            bgl,
            device: Arc::clone(device),
        })
    }

    /// Dispatch prefill attention for a full sequence.
    ///
    /// `q`, `k`, `v` all have layout `[seq_len, num_heads, head_dim]`.
    /// `output` has the same layout and must be pre-allocated with
    /// `seq_len * num_heads * head_dim * sizeof(f32)` bytes.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
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
            label: Some("Prefill Attn Params"),
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
            label: Some("Prefill Attn Bind Group"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
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

        // Workgroup size is (16, 16); dispatch covers (seq_len, num_heads).
        let wg_x = (seq_len  + 15) / 16;
        let wg_y = (num_heads + 15) / 16;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Prefill Attn Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);

        drop(pass);

        tracing::debug!(
            "PrefillAttnOp dispatched: seq_len={} num_heads={} head_dim={}",
            seq_len, num_heads, head_dim
        );

        Ok(())
    }
}
