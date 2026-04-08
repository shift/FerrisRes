use std::sync::Arc;
use wgpu::{Device, BufferDescriptor, BufferUsages, BindGroupLayoutEntry, ShaderStages, BindingType, BufferBindingType};
use crate::compute::GpuBuffer;
use crate::error::Result;

const MOE_GATING_WGSL: &str = r#"
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
    let mut expert_weights_arr = array<f32, 128>();
    let max_k = min(params.top_k, params.num_experts);

    for (var i: u32; i < params.num_experts; i = i + 1u) {
        expert_weights_arr[i] = gate_logits[offset + i];
    }

    for (var j: u32 = 0u; j < 64u; j = j + 1u) {
        var swapped = false;
        for (var k: u32 = 0u; k < 63u - j; k = k + 1u) {
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
    var max_val = -1e9;
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

const MOE_EXPERT_WGSL: &str = r#"
struct ExpertParams {
    expert_id: u32,
    num_experts: u32,
    batch_size: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> expert_up_weights: array<f32>;
@group(0) @binding(2) var<storage, read> expert_down_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ExpertParams;

@compute @workgroup_size(256)
fn expert_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let token_idx = gid.x;
    if (token_idx >= params.batch_size) {
        return;
    }

    let input_offset = token_idx * params.hidden_dim;
    let up_offset = params.expert_id * params.intermediate_dim * params.hidden_dim;
    let down_offset = params.expert_id * params.hidden_dim * params.intermediate_dim;
    let output_offset = token_idx * params.hidden_dim;

    var intermediate = array<f32, 4096>();
    for (var i: u32; i < params.intermediate_dim; i = i + 1u) {
        var sum = 0.0;
        for (var j: u32; j < params.hidden_dim; j = j + 1u) {
            sum = sum + input[input_offset + j] * expert_up_weights[up_offset + i * params.hidden_dim + j];
        }
        intermediate[i] = max(0.0, sum);
    }

    for (var i: u32; i < params.hidden_dim; i = i + 1u) {
        var sum = 0.0;
        for (var j: u32; j < params.intermediate_dim; j = j + 1u) {
            sum = sum + intermediate[j] * expert_down_weights[down_offset + i * params.intermediate_dim + j];
        }
        output[output_offset + i] = sum;
    }
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
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<Device>,
}

impl MoEExpertOp {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MoE Expert Shader"),
            source: wgpu::ShaderSource::Wgsl(MOE_EXPERT_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MoE Expert BGL"),
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

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MoE Expert PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Expert Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("expert_forward"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Ok(Self { pipeline, bgl, device: Arc::clone(device) })
    }

    pub fn dispatch_expert(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuBuffer,
        expert_up: &GpuBuffer,
        expert_down: &GpuBuffer,
        output: &GpuBuffer,
        expert_id: u32,
        batch_size: u32,
        hidden_dim: u32,
        intermediate_dim: u32,
        num_experts: u32,
    ) -> Result<()> {
        let params: [u32; 5] = [expert_id, num_experts, batch_size, hidden_dim, intermediate_dim];
        let params_buf = self.device.create_buffer(&BufferDescriptor {
            label: Some("MoE Expert Params"),
            size: 20,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
        params_buf.unmap();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoE Expert BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: expert_up.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: expert_down.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MoE Expert Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(batch_size, 1, 1);
        drop(pass);

        Ok(())
    }
}