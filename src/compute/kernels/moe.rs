use std::sync::Arc;
use wgpu::{Device, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

pub const MOE_GATING_WGSL: &str = r#"
struct MoEParams {
    num_experts: u32,
    top_k: u32,
    batch_size: u32,
    hidden_dim: u32,
}

@group(0) @binding(0) var<storage, read> gate_logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> expert_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> expert_indices: array<u32>;
@group(0) @binding(3) var<uniform> params: MoEParams;

@compute @workgroup_size(256)
fn top_k_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.batch_size) {
        return;
    }

    let offset = token_idx * params.num_experts;
    var expert_weights_arr: array<f32, 32>; // capped to 32 to prevent register spill on iGPUs
    let max_k = min(params.top_k, params.num_experts);
    let n_experts = min(params.num_experts, 32u);

    for (var i: u32; i < n_experts; i = i + 1u) {
        expert_weights_arr[i] = gate_logits[offset + i];
    }

    for (var j: u32 = 0u; j < n_experts; j = j + 1u) {
        var swapped = false;
        for (var k: u32 = 0u; k < n_experts - 1u - j; k = k + 1u) {
            if (expert_weights_arr[k] < expert_weights_arr[k + 1u]) {
                let temp = expert_weights_arr[k];
                expert_weights_arr[k] = expert_weights_arr[k + 1u];
                expert_weights_arr[k + 1u] = temp;
                swapped = true;
            }
        }
        if (!swapped) {
            break;
        }
    }

    for (var k: u32 = 0u; k < max_k; k = k + 1u) {
        expert_weights[offset + k] = expert_weights_arr[k];
    }

    for (var k: u32 = max_k; k < params.num_experts; k = k + 1u) {
        expert_weights[offset + k] = 0.0;
    }
}

@compute @workgroup_size(256)
fn compute_gate_softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.batch_size) {
        return;
    }

    let offset = token_idx * params.num_experts;
    var max_val: f32 = -1e9;
    for (var i: u32; i < params.num_experts; i = i + 1u) {
        max_val = max(max_val, gate_logits[offset + i]);
    }

    var sum = 0.0;
    for (var i: u32; i < params.num_experts; i = i + 1u) {
        let exp_val = exp(gate_logits[offset + i] - max_val);
        expert_weights[offset + i] = exp_val;
        sum = sum + exp_val;
    }

    for (var i: u32; i < params.num_experts; i = i + 1u) {
        expert_weights[offset + i] = expert_weights[offset + i] / sum;
    }
}
"#;

pub const MOE_DISPATCH_WGSL: &str = r#"
struct DispatchParams {
    num_experts: u32,
    top_k: u32,
    batch_size: u32,
    hidden_dim: u32,
}

@group(0) @binding(0) var<storage, read> gate_logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> selected_experts: array<u32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;
@group(0) @binding(3) var<uniform> params: DispatchParams;

@compute @workgroup_size(256)
fn compute_top_k_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.batch_size) {
        return;
    }

    let offset = token_idx * params.num_experts;
    var logits_with_idx = array<vec2<f32>, 32>(); // capped to 32 to prevent register spill on iGPUs
    let n_experts = min(params.num_experts, 32u);

    for (var i: u32; i < n_experts; i = i + 1u) {
        logits_with_idx[i] = vec2<f32>(gate_logits[offset + i], f32(i));
    }

    for (var j: u32 = 0u; j < n_experts; j = j + 1u) {
        var swapped = false;
        for (var k: u32 = 0u; k < n_experts - 1u - j; k = k + 1u) {
            if (logits_with_idx[k].x < logits_with_idx[k + 1u].x) {
                let temp = logits_with_idx[k];
                logits_with_idx[k] = logits_with_idx[k + 1u];
                logits_with_idx[k + 1u] = temp;
                swapped = true;
            }
        }
        if (!swapped) { break; }
    }

    let sel_offset = token_idx * params.top_k;

    // Softmax over K selected logits (numerically stable: subtract max before exp).
    // logits_with_idx is sorted descending, so index 0 is the maximum.
    let max_logit: f32 = logits_with_idx[0].x;
    var sum_exp: f32 = 0.0;
    for (var k: u32 = 0u; k < params.top_k; k = k + 1u) {
        sum_exp = sum_exp + exp(logits_with_idx[k].x - max_logit);
    }
    for (var k: u32 = 0u; k < params.top_k; k = k + 1u) {
        selected_experts[sel_offset + k] = u32(logits_with_idx[k].y);
        expert_weights[sel_offset + k] = exp(logits_with_idx[k].x - max_logit) / sum_exp;
    }
}
"#;

pub const MOE_GATHER_WGSL: &str = r#"
struct GatherParams {
    num_experts: u32,
    top_k: u32,
    batch_size: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> selected_experts: array<u32>;
@group(0) @binding(2) var<storage, read> expert_weights: array<f32>;
@group(0) @binding(3) var<storage, read> expert_up_weights: array<f32>;
@group(0) @binding(4) var<storage, read> expert_down_weights: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> params: GatherParams;
@group(0) @binding(7) var<storage, read_write> scratch: array<f32>;

@compute @workgroup_size(256)
fn moe_up_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.batch_size * params.top_k;
    if (flat >= total) { return; }

    let token_idx = flat / params.top_k;
    let k = flat % params.top_k;
    let expert_id = selected_experts[token_idx * params.top_k + k];
    let weight = expert_weights[token_idx * params.top_k + k];
    if (weight == 0.0) { return; }

    let input_offset = token_idx * params.hidden_dim;
    let up_offset = expert_id * params.intermediate_dim * params.hidden_dim;
    let scratch_base = flat * params.intermediate_dim;

    var ii: u32 = 0u;
    while (ii < params.intermediate_dim) {
        var dot = 0.0;
        var jj: u32 = 0u;
        while (jj < params.hidden_dim) {
            dot = dot + input[input_offset + jj] * expert_up_weights[up_offset + ii * params.hidden_dim + jj];
            jj = jj + 1u;
        }
        scratch[scratch_base + ii] = max(0.0, dot);
        ii = ii + 1u;
    }
}

@compute @workgroup_size(256)
fn moe_down_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.batch_size * params.top_k;
    if (flat >= total) { return; }

    let token_idx = flat / params.top_k;
    let k = flat % params.top_k;
    let expert_id = selected_experts[token_idx * params.top_k + k];
    let weight = expert_weights[token_idx * params.top_k + k];
    if (weight == 0.0) { return; }

    let up_scratch_base = flat * params.intermediate_dim;
    let down_offset = expert_id * params.hidden_dim * params.intermediate_dim;
    let output_offset = flat * params.hidden_dim;

    var ii: u32 = 0u;
    while (ii < params.hidden_dim) {
        var dot = 0.0;
        var jj: u32 = 0u;
        while (jj < params.intermediate_dim) {
            dot = dot + scratch[up_scratch_base + jj] * expert_down_weights[down_offset + ii * params.intermediate_dim + jj];
            jj = jj + 1u;
        }
        output[output_offset + ii] = dot * weight;
        ii = ii + 1u;
    }
}

@compute @workgroup_size(256)
fn moe_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let token_idx = flat / params.hidden_dim;
    let dim_idx = flat % params.hidden_dim;
    if (token_idx >= params.batch_size) { return; }

    var sum = 0.0;
    var kk: u32 = 0u;
    while (kk < params.top_k) {
        let expert_flat = token_idx * params.top_k + kk;
        sum = sum + output[expert_flat * params.hidden_dim + dim_idx];
        kk = kk + 1u;
    }
    output[token_idx * params.hidden_dim + dim_idx] = sum;
}
"#;

pub struct MoEGatingOp {
    top_k_pipeline: wgpu::ComputePipeline,
    softmax_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl MoEGatingOp {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MoE Gating Shader"),
            source: wgpu::ShaderSource::Wgsl(MOE_GATING_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MoE Gating BGL"),
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

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MoE Gating PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let top_k_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Top-K Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("top_k_gate"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Softmax Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("compute_gate_softmax"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self {
            top_k_pipeline,
            softmax_pipeline,
            bgl,
            device: Arc::clone(device),
        })
    }

    pub fn dispatch_top_k(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_logits: &GpuBuffer,
        expert_weights: &GpuBuffer,
        expert_indices: &GpuBuffer,
        batch_size: u32,
        num_experts: u32,
        top_k: u32,
        hidden_dim: u32,
    ) -> Result<()> {
        let params: [u32; 4] = [num_experts, top_k, batch_size, hidden_dim];
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("MoE Gating Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoE Gating BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate_logits.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: expert_weights.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: expert_indices.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MoE Top-K Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.top_k_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(batch_size, 1, 1);
        drop(pass);

        Ok(())
    }

    pub fn dispatch_softmax(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gate_logits: &GpuBuffer,
        expert_weights: &GpuBuffer,
        batch_size: u32,
        num_experts: u32,
        hidden_dim: u32,
    ) -> Result<()> {
        let params: [u32; 4] = [num_experts, 1, batch_size, hidden_dim];
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("MoE Softmax Params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoE Softmax BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate_logits.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: expert_weights.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: expert_weights.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MoE Softmax Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.softmax_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(batch_size, 1, 1);
        drop(pass);

        Ok(())
    }
}

pub struct MoEExpertOp {
    dispatch_pipeline: wgpu::ComputePipeline,
    up_proj_pipeline: wgpu::ComputePipeline,
    down_proj_pipeline: wgpu::ComputePipeline,
    accumulate_pipeline: wgpu::ComputePipeline,
    dispatch_bgl: wgpu::BindGroupLayout,
    gather_bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl MoEExpertOp {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let dispatch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MoE Dispatch Shader"),
            source: wgpu::ShaderSource::Wgsl(MOE_DISPATCH_WGSL.into()),
        });

        let gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MoE Gather Shader"),
            source: wgpu::ShaderSource::Wgsl(MOE_GATHER_WGSL.into()),
        });

        let dispatch_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MoE Dispatch BGL"),
            entries: &[
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let gather_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MoE Gather BGL"),
            entries: &[
                BindGroupLayoutEntry { binding: 0, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 1, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 2, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 4, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 5, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 6, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                BindGroupLayoutEntry { binding: 7, visibility: ShaderStages::COMPUTE, ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let dispatch_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("MoE Dispatch PL"), bind_group_layouts: &[Some(&dispatch_bgl)], immediate_size: 0 });
        let gather_pipeline = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("MoE Gather PL"), bind_group_layouts: &[Some(&gather_bgl)], immediate_size: 0 });

        let dispatch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Dispatch Pipeline"), layout: Some(&dispatch_pl), module: &dispatch_shader, entry_point: Some("compute_top_k_indices"), cache: None, compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let up_proj_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Up-Proj Pipeline"), layout: Some(&gather_pipeline), module: &gather_shader, entry_point: Some("moe_up_proj"), cache: None, compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let down_proj_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Down-Proj Pipeline"), layout: Some(&gather_pipeline), module: &gather_shader, entry_point: Some("moe_down_proj"), cache: None, compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let accumulate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Accumulate Pipeline"), layout: Some(&gather_pipeline), module: &gather_shader, entry_point: Some("moe_accumulate"), cache: None, compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self { dispatch_pipeline, up_proj_pipeline, down_proj_pipeline, accumulate_pipeline, dispatch_bgl, gather_bgl, device: Arc::clone(device) })
    }

    pub fn dispatch_top_k(
        &self, encoder: &mut wgpu::CommandEncoder, gate_logits: &GpuBuffer, selected_experts: &GpuBuffer, expert_weights: &GpuBuffer,
        batch_size: u32, num_experts: u32, top_k: u32, hidden_dim: u32,
    ) -> Result<()> {
        let params: [u32; 4] = [num_experts, top_k, batch_size, hidden_dim];
        let params_buf = self.device.create_buffer(&BufferDescriptor { label: Some("MoE Dispatch Params"), size: 16, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: true });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoE Dispatch BG"), layout: &self.dispatch_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gate_logits.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: selected_experts.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: expert_weights.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MoE Dispatch Pass"), timestamp_writes: None });
        pass.set_pipeline(&self.dispatch_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(batch_size, 1, 1);
        drop(pass);
        Ok(())
    }

    pub fn gather_expert_outputs(
        &self, encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer, selected_experts: &GpuBuffer, expert_weights: &GpuBuffer,
        expert_up: &GpuBuffer, expert_down: &GpuBuffer, output: &GpuBuffer, scratch: &GpuBuffer,
        batch_size: u32, num_experts: u32, top_k: u32, hidden_dim: u32, intermediate_dim: u32,
    ) -> Result<()> {
        let params: [u32; 5] = [num_experts, top_k, batch_size, hidden_dim, intermediate_dim];
        let params_buf = self.device.create_buffer(&BufferDescriptor { label: Some("MoE Gather Params"), size: 20, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: true });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoE Gather BG"), layout: &self.gather_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: selected_experts.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: expert_weights.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: expert_up.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: expert_down.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: scratch.buffer().as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MoE Gather Up Pass"), timestamp_writes: None });
        pass.set_pipeline(&self.up_proj_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((batch_size * top_k + 255) / 256, 1, 1);
        drop(pass);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MoE Gather Down Pass"), timestamp_writes: None });
        pass.set_pipeline(&self.down_proj_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((batch_size * top_k + 255) / 256, 1, 1);
        drop(pass);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MoE Accumulate Pass"), timestamp_writes: None });
        pass.set_pipeline(&self.accumulate_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((batch_size * hidden_dim + 255) / 256, 1, 1);
        drop(pass);
        Ok(())
    }
}