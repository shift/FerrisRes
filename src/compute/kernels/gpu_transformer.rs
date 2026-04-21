//! GPU compute shaders for the full transformer forward pass.
//! All operations run on GPU with zero intermediate CPU transfers.
//!
//! Shaders:
//! - rmsnorm_weighted: RMSNorm with learned weight (scale) parameter
//! - silu_multiply: SiLU(x) * y (gate activation)
//! - residual_add: a + b (elementwise)
//! - embedding_gather: token_ids → hidden states
//! - causal_attention: self-attention with causal mask + GQA

use wgpu::{
    BindGroupLayout, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePipeline, Device, PipelineLayoutDescriptor, Queue,
    ShaderModuleDescriptor, ShaderSource,
};
use crate::compute::GpuBuffer;
use crate::error::Result;

// ========================================================================
// WGSL Shaders
// ========================================================================

/// RMSNorm with learned weight: output[i] = input[i] * inv_rms * weight[i]
pub const RMSNORM_WEIGHTED_WGSL: &str = r#"
struct Params {
    hidden_dim: u32,
    rows: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let col = lid.x;
    let hidden_dim = params.hidden_dim;

    // Phase 1: partial sum of squares
    var partial = 0.0;
    for (var c = col; c < hidden_dim; c = c + 256u) {
        let v = input[row * hidden_dim + c];
        partial = partial + v * v;
    }
    wg_data[col] = partial;
    workgroupBarrier();

    // Phase 2: parallel reduction
    var stride = 128u;
    while (stride > 0u) {
        if (col < stride) {
            wg_data[col] = wg_data[col] + wg_data[col + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Phase 3: broadcast inv_rms, apply norm + weight
    var inv_rms = 1.0;
    if (col == 0u) {
        inv_rms = inverseSqrt(wg_data[0u] / f32(hidden_dim) + 1e-6);
        wg_data[0u] = inv_rms;
    }
    workgroupBarrier();
    inv_rms = wg_data[0u];

    for (var c = col; c < hidden_dim; c = c + 256u) {
        let idx = row * hidden_dim + c;
        output[idx] = input[idx] * inv_rms * weight[c];
    }
}
"#;

/// SiLU activation followed by elementwise multiply: silu(x) * y
/// Used for SwiGLU: gate_proj output is SiLU'd, then multiplied by up_proj output.
pub const SILU_MULTIPLY_WGSL: &str = r#"
struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> gated: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let x = gated[i];
    let silu_x = x / (1.0 + exp(-x));
    output[i] = silu_x * up[i];
}
"#;

/// Elementwise residual add: output = a + b
pub const RESIDUAL_ADD_WGSL: &str = r#"
struct Params {
    n: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    output[i] = a[i] + b[i];
}
"#;

/// Embedding gather: token_ids → hidden states, scaled by sqrt(hidden_dim)
/// Each invocation handles one (token, dim) pair.
pub const EMBEDDING_GATHER_WGSL: &str = r#"
struct Params {
    seq_len: u32,
    hidden_dim: u32,
    vocab_size: u32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> embed_table: array<f32>;  // [vocab_size × hidden_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // [seq_len × hidden_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let seq_len = params.seq_len;
    let hidden_dim = params.hidden_dim;
    let total = seq_len * hidden_dim;
    if (flat >= total) { return; }

    let t = flat / hidden_dim;
    let d = flat % hidden_dim;
    let tid = token_ids[t];

    let scale = sqrt(f32(hidden_dim));
    let idx = tid * hidden_dim + d;
    if (idx < params.vocab_size * hidden_dim) {
        output[flat] = embed_table[idx] * scale;
    } else {
        output[flat] = 0.0;
    }
}
"#;

/// Causal self-attention with GQA.
/// Q: [seq × num_heads × head_dim], K: [seq × num_kv_heads × head_dim], V: same as K
/// Output: [seq × num_heads × head_dim]
/// Uses online softmax for numerical stability.
pub const CAUSAL_ATTENTION_GQA_WGSL: &str = r#"
struct Params {
    seq_len: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    kv_group_size: u32,  // num_heads / num_kv_heads
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seq = params.seq_len;
    let nh = params.num_heads;
    let nkv = params.num_kv_heads;
    let hd = params.head_dim;
    let kv_group = params.kv_group_size;

    let t = gid.x;  // query token position
    let h = gid.y;  // query head index

    if (t >= seq || h >= nh) { return; }

    // Map query head to KV head
    let kv_h = h / kv_group;

    let q_offset = t * nh * hd + h * hd;
    let scale = 1.0 / sqrt(f32(hd));

    // Online softmax: track max and sum
    var max_val = -1e30;
    var sum_val = 0.0;

    // First pass: compute max for numerical stability
    var scores: array<f32, 256>;  // max seq_len = 256
    for (var s = 0u; s < seq; s = s + 1u) {
        if (s > t) {
            scores[s] = -1e30;
            continue;
        }
        // Dot product Q[t,h,:] · K[s,kv_h,:]
        var dot = 0.0;
        let k_offset = s * nkv * hd + kv_h * hd;
        for (var d = 0u; d < hd; d = d + 1u) {
            dot = dot + q[q_offset + d] * k[k_offset + d];
        }
        let score = dot * scale;
        scores[s] = score;
        if (score > max_val) {
            max_val = score;
        }
    }

    // Second pass: compute sum of exp(score - max)
    for (var s = 0u; s <= t; s = s + 1u) {
        sum_val = sum_val + exp(scores[s] - max_val);
    }

    // Third pass: weighted sum of V
    for (var d = 0u; d < hd; d = d + 1u) {
        var val = 0.0;
        for (var s = 0u; s <= t; s = s + 1u) {
            let weight = exp(scores[s] - max_val) / sum_val;
            let v_offset = s * nkv * hd + kv_h * hd;
            val = val + weight * v[v_offset + d];
        }
        let out_offset = t * nh * hd + h * hd;
        output[out_offset + d] = val;
    }
}
"#;

/// RoPE (Rotary Position Embeddings) applied to Q or K.
/// Each invocation handles one (token, head, dim_pair).
pub const ROPE_WGSL: &str = r#"
struct Params {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let seq = params.seq_len;
    let nh = params.num_heads;
    let hd = params.head_dim;
    let total = seq * nh * hd;

    if (flat >= total) { return; }

    let t = flat / (nh * hd);
    let remainder = flat % (nh * hd);
    let d = remainder % hd;

    // Only process even dimensions (handle pairs: d, d+1)
    if (d % 2u != 0u) { return; }
    let d_pair = d / 2u;

    // Compute rotation frequency
    let freq = 1.0 / pow(10000.0, f32(d_pair) * 2.0 / f32(hd));
    let angle = f32(t) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);

    // Apply rotation to pair (d, d+1)
    let idx_d = flat;
    let idx_d1 = flat + 1u;
    let x0 = data[idx_d];
    let x1 = data[idx_d1];
    data[idx_d] = x0 * cos_a - x1 * sin_a;
    data[idx_d1] = x0 * sin_a + x1 * cos_a;
}
"#;

// ========================================================================
// Rust: Pipeline wrappers
// ========================================================================

/// GPU compute pipeline for the full transformer forward pass.
/// All shaders are compiled once at init, dispatched per-layer.
pub struct GpuTransformerPipeline {
    rmsnorm: ComputePipeline,
    rmsnorm_layout: BindGroupLayout,
    silu_mul: ComputePipeline,
    silu_mul_layout: BindGroupLayout,
    residual: ComputePipeline,
    residual_layout: BindGroupLayout,
    embed: ComputePipeline,
    embed_layout: BindGroupLayout,
    attention: ComputePipeline,
    attention_layout: BindGroupLayout,
    rope: ComputePipeline,
    rope_layout: BindGroupLayout,
}

impl GpuTransformerPipeline {
    pub fn new(device: &Device) -> Result<Self> {
        let (rmsnorm, rmsnorm_layout) = Self::create_rmsnorm_pipeline(device)?;
        let (silu_mul, silu_mul_layout) = Self::create_silu_mul_pipeline(device)?;
        let (residual, residual_layout) = Self::create_residual_pipeline(device)?;
        let (embed, embed_layout) = Self::create_embed_pipeline(device)?;
        let (attention, attention_layout) = Self::create_attention_pipeline(device)?;
        let (rope, rope_layout) = Self::create_rope_pipeline(device)?;

        Ok(Self {
            rmsnorm, rmsnorm_layout,
            silu_mul, silu_mul_layout,
            residual, residual_layout,
            embed, embed_layout,
            attention, attention_layout,
            rope, rope_layout,
        })
    }

    fn create_rmsnorm_pipeline(device: &Device) -> Result<(ComputePipeline, BindGroupLayout)> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rmsnorm_weighted_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rmsnorm_weighted"),
            source: ShaderSource::Wgsl(RMSNORM_WEIGHTED_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rmsnorm_weighted_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rmsnorm_weighted"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_silu_mul_pipeline(device: &Device) -> Result<(ComputePipeline, BindGroupLayout)> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("silu_mul_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("silu_mul"),
            source: ShaderSource::Wgsl(SILU_MULTIPLY_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("silu_mul_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("silu_mul"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_residual_pipeline(device: &Device) -> Result<(ComputePipeline, BindGroupLayout)> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("residual_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("residual_add"),
            source: ShaderSource::Wgsl(RESIDUAL_ADD_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("residual_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("residual_add"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_embed_pipeline(device: &Device) -> Result<(ComputePipeline, BindGroupLayout)> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("embed_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("embedding_gather"),
            source: ShaderSource::Wgsl(EMBEDDING_GATHER_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("embed_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embedding_gather"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_attention_pipeline(device: &Device) -> Result<(ComputePipeline, BindGroupLayout)> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("attention_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("causal_attention_gqa"),
            source: ShaderSource::Wgsl(CAUSAL_ATTENTION_GQA_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("attention_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("causal_attention_gqa"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_rope_pipeline(device: &Device) -> Result<(ComputePipeline, BindGroupLayout)> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rope_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rope"),
            source: ShaderSource::Wgsl(ROPE_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("rope_pl"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rope"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    // ========================================================================
    // Dispatch helpers — each appends to the encoder, NO sync
    // ========================================================================

    /// RMSNorm with learned weight: output = input * inv_rms(hidden) * weight
    pub fn dispatch_rmsnorm(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        input: &GpuBuffer,
        output: &GpuBuffer,
        weight: &GpuBuffer,
        rows: u32,
        hidden_dim: u32,
    ) -> Result<()> {
        let params_data: [u32; 2] = [hidden_dim, rows];
        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("rmsnorm_params"),
            size: 8,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rmsnorm_bg"),
            layout: &self.rmsnorm_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: weight.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rmsnorm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.rmsnorm);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(rows, 1, 1);
        Ok(())
    }

    /// SiLU(gated) * up → output
    pub fn dispatch_silu_multiply(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        gated: &GpuBuffer,
        up: &GpuBuffer,
        output: &GpuBuffer,
        n: u32,
    ) -> Result<()> {
        let params_data: [u32; 1] = [n];
        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("silu_mul_params"),
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu_mul_bg"),
            layout: &self.silu_mul_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gated.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: up.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("silu_mul"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.silu_mul);
        pass.set_bind_group(0, &bg, &[]);
        let wg = (n + 255) / 256;
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }

    /// residual: output = a + b
    pub fn dispatch_residual_add(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        a: &GpuBuffer,
        b: &GpuBuffer,
        output: &GpuBuffer,
        n: u32,
    ) -> Result<()> {
        let params_data: [u32; 1] = [n];
        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("residual_params"),
            size: 4,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("residual_bg"),
            layout: &self.residual_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("residual_add"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.residual);
        pass.set_bind_group(0, &bg, &[]);
        let wg = (n + 255) / 256;
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }

    /// Embedding gather: token_ids → hidden states
    pub fn dispatch_embedding(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        token_ids: &GpuBuffer,
        embed_table: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> Result<()> {
        let params_data: [u32; 3] = [seq_len, hidden_dim, vocab_size];
        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("embed_params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embed_bg"),
            layout: &self.embed_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: token_ids.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: embed_table.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embed_gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.embed);
        pass.set_bind_group(0, &bg, &[]);
        let total = seq_len * hidden_dim;
        let wg = (total + 255) / 256;
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }

    /// Causal self-attention with GQA
    pub fn dispatch_attention(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        output: &GpuBuffer,
        seq_len: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        let kv_group_size = num_heads / num_kv_heads;
        let params_data: [u32; 5] = [seq_len, num_heads, num_kv_heads, head_dim, kv_group_size];
        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("attn_params"),
            size: 20,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attn_bg"),
            layout: &self.attention_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("causal_attention"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.attention);
        pass.set_bind_group(0, &bg, &[]);
        // One workgroup per (token, head) pair
        pass.dispatch_workgroups(seq_len, num_heads, 1);
        Ok(())
    }

    /// RoPE: in-place rotation of Q or K
    pub fn dispatch_rope(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        data: &GpuBuffer,
        seq_len: u32,
        num_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        let params_data: [u32; 3] = [seq_len, num_heads, head_dim];
        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("rope_params"),
            size: 12,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params_buf, 0, bytemuck::cast_slice(&params_data));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope_bg"),
            layout: &self.rope_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: data.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.rope);
        pass.set_bind_group(0, &bg, &[]);
        let total = seq_len * num_heads * head_dim;
        let wg = (total + 255) / 256;
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_transformer_pipeline_creation() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }));

        match adapter {
            Ok(adapter) => {
                let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("test"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })).unwrap();

                // If this compiles, all 6 WGSL shaders are valid
                let pipeline = GpuTransformerPipeline::new(&device);
                assert!(pipeline.is_ok(), "Pipeline creation failed: {:?}", pipeline.err());
            }
            Err(_) => {
                eprintln!("No GPU adapter, skipping shader compilation test");
            }
        }
    }
}
