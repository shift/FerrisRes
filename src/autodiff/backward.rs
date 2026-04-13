use std::sync::Arc;
use wgpu::{Device, Queue};
use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::autodiff::graph::{ComputationGraph, NodeId, NodeKind};
use crate::error::{FerrisResError, Result};

const MATMUL_GRAD_A_WGSL: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> grad_output: array<f32>;
@group(0) @binding(1) var<storage, read> b_transposed: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_a: array<f32>;

var<private> params: Params;

@compute @workgroup_size(64)
fn matmul_grad_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.M || col >= params.K) {
        return;
    }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.N; i = i + 1u) {
        acc = acc + grad_output[row * params.N + i] * b_transposed[col * params.N + i];
    }
    grad_a[row * params.K + col] = acc;
}
"#;

const MATMUL_GRAD_B_WGSL: &str = r#"
struct Params {
    M: u32,
    K: u32,
    N: u32,
}

@group(0) @binding(0) var<storage, read> a_transposed: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_b: array<f32>;

var<private> params: Params;

@compute @workgroup_size(64)
fn matmul_grad_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.K || col >= params.N) {
        return;
    }
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.M; i = i + 1u) {
        acc = acc + a_transposed[row * params.M + i] * grad_output[i * params.N + col];
    }
    grad_b[row * params.N + col] = acc;
}
"#;

const SOFTMAX_GRAD_WGSL: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> softmax_output: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

var<private> params: Params;

@compute @workgroup_size(256)
fn softmax_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let row = idx / params.cols;
    let col = idx % params.cols;
    if (row >= params.rows) {
        return;
    }
    let row_offset = row * params.cols;
    let s_i = softmax_output[row_offset + col];
    var dot: f32 = 0.0;
    for (var j: u32 = 0u; j < params.cols; j = j + 1u) {
        dot = dot + grad_output[row_offset + j] * softmax_output[row_offset + j];
    }
    grad_input[row_offset + col] = s_i * (grad_output[row_offset + col] - dot);
}
"#;

const RELU_GRAD_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> input_val: array<f32>;
@group(0) @binding(1) var<storage, read> grad_output: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

@compute @workgroup_size(256)
fn relu_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let mask = select(0.0, 1.0, input_val[idx] > 0.0);
    grad_input[idx] = grad_output[idx] * mask;
}
"#;

const RMSNORM_GRAD_WGSL: &str = r#"
struct Params {
    hidden_dim: u32,
    eps: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input_val: array<f32>;
@group(0) @binding(1) var<storage, read> output_val: array<f32>;
@group(0) @binding(2) var<storage, read> grad_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_input: array<f32>;

var<private> params: Params;

@compute @workgroup_size(256)
fn rmsnorm_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.hidden_dim) {
        return;
    }
    let x_i = input_val[idx];
    let y_i = output_val[idx];
    let g_i = grad_output[idx];

    var ss: f32 = 0.0;
    for (var j: u32 = 0u; j < params.hidden_dim; j = j + 1u) {
        ss = ss + input_val[j] * input_val[j];
    }
    let rms = sqrt(ss / f32(params.hidden_dim) + params.eps);
    let inv_rms = 1.0 / rms;

    var dot_gy: f32 = 0.0;
    for (var j: u32 = 0u; j < params.hidden_dim; j = j + 1u) {
        dot_gy = dot_gy + grad_output[j] * output_val[j];
    }

    grad_input[idx] = inv_rms / f32(params.hidden_dim) * (g_i - y_i * dot_gy);
}
"#;

const LOSS_GRAD_WGSL: &str = r#"
struct Params {
    batch_size: u32,
    vocab_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;

var<private> params: Params;

@compute @workgroup_size(256)
fn loss_grad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let row = idx / params.vocab_size;
    let col = idx % params.vocab_size;
    if (row >= params.batch_size) {
        return;
    }
    let row_offset = row * params.vocab_size;

    var max_val: f32 = logits[row_offset];
    for (var i: u32 = 1u; i < params.vocab_size; i = i + 1u) {
        let val = logits[row_offset + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < params.vocab_size; i = i + 1u) {
        sum_exp = sum_exp + exp(logits[row_offset + i] - max_val);
    }

    let prob = exp(logits[row_offset + col] - max_val) / sum_exp;
    let target_idx = targets[row];
    let is_target = select(0.0, 1.0, col == target_idx);
    grad_input[row_offset + col] = (prob - is_target) / f32(params.batch_size);
}
"#;

pub struct BackwardPass {
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    elementwise: ElementWiseOp,
    matmul_grad_a_pipeline: wgpu::ComputePipeline,
    matmul_grad_a_layout: wgpu::BindGroupLayout,
    matmul_grad_b_pipeline: wgpu::ComputePipeline,
    matmul_grad_b_layout: wgpu::BindGroupLayout,
    softmax_grad_pipeline: wgpu::ComputePipeline,
    softmax_grad_layout: wgpu::BindGroupLayout,
    relu_grad_pipeline: wgpu::ComputePipeline,
    relu_grad_layout: wgpu::BindGroupLayout,
    rmsnorm_grad_pipeline: wgpu::ComputePipeline,
    rmsnorm_grad_layout: wgpu::BindGroupLayout,
    loss_grad_pipeline: wgpu::ComputePipeline,
    loss_grad_layout: wgpu::BindGroupLayout,
}

impl BackwardPass {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        tracing::info!("Creating BackwardPass gradient pipelines");

        let elementwise = ElementWiseOp::new(&device, &queue);

        let matmul_grad_a_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Grad A Shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_GRAD_A_WGSL.into()),
        });

        let matmul_grad_b_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Grad B Shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_GRAD_B_WGSL.into()),
        });

        let softmax_grad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Softmax Grad Shader"),
            source: wgpu::ShaderSource::Wgsl(SOFTMAX_GRAD_WGSL.into()),
        });

        let relu_grad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ReLU Grad Shader"),
            source: wgpu::ShaderSource::Wgsl(RELU_GRAD_WGSL.into()),
        });

        let rmsnorm_grad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RmsNorm Grad Shader"),
            source: wgpu::ShaderSource::Wgsl(RMSNORM_GRAD_WGSL.into()),
        });

        let loss_grad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Loss Grad Shader"),
            source: wgpu::ShaderSource::Wgsl(LOSS_GRAD_WGSL.into()),
        });

        let read_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let read_entry_b = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let rw_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let rmsnorm_read_c = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let matmul_grad_a_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Grad A Layout"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone()],
        });

        let matmul_grad_a_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul Grad A Pipeline Layout"),
            bind_group_layouts: &[Some(&matmul_grad_a_layout)],
            immediate_size: 12,
        });

        let matmul_grad_a_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Grad A Pipeline"),
            layout: Some(&matmul_grad_a_pipeline_layout),
            module: &matmul_grad_a_shader,
            entry_point: Some("matmul_grad_a"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let matmul_grad_b_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MatMul Grad B Layout"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone()],
        });

        let matmul_grad_b_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MatMul Grad B Pipeline Layout"),
            bind_group_layouts: &[Some(&matmul_grad_b_layout)],
            immediate_size: 12,
        });

        let matmul_grad_b_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Grad B Pipeline"),
            layout: Some(&matmul_grad_b_pipeline_layout),
            module: &matmul_grad_b_shader,
            entry_point: Some("matmul_grad_b"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let softmax_grad_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Softmax Grad Layout"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone()],
        });

        let softmax_grad_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Softmax Grad Pipeline Layout"),
            bind_group_layouts: &[Some(&softmax_grad_layout)],
            immediate_size: 8,
        });

        let softmax_grad_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Softmax Grad Pipeline"),
            layout: Some(&softmax_grad_pipeline_layout),
            module: &softmax_grad_shader,
            entry_point: Some("softmax_grad"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let relu_grad_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ReLU Grad Layout"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone()],
        });

        let relu_grad_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ReLU Grad Pipeline Layout"),
            bind_group_layouts: &[Some(&relu_grad_layout)],
            immediate_size: 0,
        });

        let relu_grad_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ReLU Grad Pipeline"),
            layout: Some(&relu_grad_pipeline_layout),
            module: &relu_grad_shader,
            entry_point: Some("relu_grad"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let rmsnorm_grad_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RmsNorm Grad Layout"),
            entries: &[
                read_entry.clone(),
                read_entry_b.clone(),
                rw_entry.clone(),
                rmsnorm_read_c,
            ],
        });

        let rmsnorm_grad_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RmsNorm Grad Pipeline Layout"),
            bind_group_layouts: &[Some(&rmsnorm_grad_layout)],
            immediate_size: 16,
        });

        let rmsnorm_grad_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RmsNorm Grad Pipeline"),
            layout: Some(&rmsnorm_grad_pipeline_layout),
            module: &rmsnorm_grad_shader,
            entry_point: Some("rmsnorm_grad"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let loss_grad_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Loss Grad Layout"),
            entries: &[read_entry.clone(), read_entry_b.clone(), rw_entry.clone()],
        });

        let loss_grad_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Loss Grad Pipeline Layout"),
            bind_group_layouts: &[Some(&loss_grad_layout)],
            immediate_size: 16,
        });

        let loss_grad_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Loss Grad Pipeline"),
            layout: Some(&loss_grad_pipeline_layout),
            module: &loss_grad_shader,
            entry_point: Some("loss_grad"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        tracing::info!("BackwardPass: all gradient pipelines created");

        Self {
            device,
            queue,
            elementwise,
            matmul_grad_a_pipeline,
            matmul_grad_a_layout,
            matmul_grad_b_pipeline,
            matmul_grad_b_layout,
            softmax_grad_pipeline,
            softmax_grad_layout,
            relu_grad_pipeline,
            relu_grad_layout,
            rmsnorm_grad_pipeline,
            rmsnorm_grad_layout,
            loss_grad_pipeline,
            loss_grad_layout,
        }
    }

    pub fn run(
        &self,
        graph: &mut ComputationGraph,
        encoder: &mut wgpu::CommandEncoder,
        loss_id: NodeId,
    ) -> Result<()> {
        tracing::info!("BackwardPass::run starting from loss node {:?}", loss_id);

        graph.zero_all_grads(encoder)?;

        let loss_node = graph.get_node(loss_id)
            .ok_or_else(|| FerrisResError::Device(format!("loss node {:?} not found in graph", loss_id)))?;

        self.elementwise.dispatch_scale(
            encoder,
            &loss_node.grad,
            &loss_node.grad,
            1.0,
            loss_node.buf.size() as u32 / std::mem::size_of::<f32>() as u32,
        )?;

        // Walk the full tape in reverse topological order.
        // The tape is recorded in forward (topological) order, so reversing it
        // gives the correct gradient accumulation order for backprop.
        let tape_snapshot: Vec<(NodeId, NodeKind, Vec<NodeId>)> = graph.tape()
            .iter()
            .map(|e| (e.output_id(), e.op().clone(), e.inputs().to_vec()))
            .collect();

        for (output_id, op, inputs) in tape_snapshot.iter().rev() {
            self.backward_entry(graph, encoder, *output_id, op, inputs)?;
        }

        tracing::info!("BackwardPass::run complete");
        Ok(())
    }

    fn node_buf_info(graph: &ComputationGraph, id: NodeId) -> Result<(wgpu::Buffer, usize)> {
        let node = graph.get_node(id).ok_or_else(|| {
            FerrisResError::Device(format!("node {:?} not found", id))
        })?;
        Ok((node.buf.buffer().clone(), node.buf.size()))
    }

    fn node_grad_info(graph: &ComputationGraph, id: NodeId) -> Result<(wgpu::Buffer, usize)> {
        let node = graph.get_node(id).ok_or_else(|| {
            FerrisResError::Device(format!("node {:?} not found", id))
        })?;
        Ok((node.grad.buffer().clone(), node.grad.size()))
    }

    fn backward_entry(
        &self,
        graph: &mut ComputationGraph,
        encoder: &mut wgpu::CommandEncoder,
        output_id: NodeId,
        op: &NodeKind,
        inputs: &[NodeId],
    ) -> Result<()> {
        match op {
            NodeKind::MatMul { m, k, n } => {
                let a_id = inputs[0];
                let b_id = inputs[1];

                let (grad_buf, _grad_size) = Self::node_grad_info(graph, output_id)?;
                let (a_buf, _) = Self::node_buf_info(graph, a_id)?;
                let (a_grad, _) = Self::node_grad_info(graph, a_id)?;
                let (b_buf, _) = Self::node_buf_info(graph, b_id)?;
                let (b_grad, _) = Self::node_grad_info(graph, b_id)?;

                let wg_x = (*m + 63) / 64;
                let wg_y = (*k + 63) / 64;

                let bg_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Grad A Bind Group"),
                    layout: &self.matmul_grad_a_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: grad_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: a_grad.as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Grad A"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.matmul_grad_a_pipeline);
                pass.set_bind_group(0, &bg_a, &[]);
                pass.set_immediates(0, &[m.to_le_bytes(), k.to_le_bytes(), n.to_le_bytes()].concat());
                pass.dispatch_workgroups(wg_x, wg_y, 1);
                drop(pass);

                let wg_x_b = (*k + 63) / 64;
                let wg_y_b = (*n + 63) / 64;

                let bg_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MatMul Grad B Bind Group"),
                    layout: &self.matmul_grad_b_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: grad_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: b_grad.as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("MatMul Grad B"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.matmul_grad_b_pipeline);
                pass.set_bind_group(0, &bg_b, &[]);
                pass.set_immediates(0, &[m.to_le_bytes(), k.to_le_bytes(), n.to_le_bytes()].concat());
                pass.dispatch_workgroups(wg_x_b, wg_y_b, 1);
                drop(pass);

                tracing::debug!("Backward: matmul grad done M={} K={} N={}", m, k, n);
            }

            NodeKind::Add { numel } => {
                let a_id = inputs[0];
                let b_id = inputs[1];

                let (grad_buf, grad_size) = Self::node_grad_info(graph, output_id)?;
                let (a_grad, a_grad_size) = Self::node_grad_info(graph, a_id)?;
                let (b_grad, b_grad_size) = Self::node_grad_info(graph, b_id)?;

                let grad_ref = GpuBuffer::from_existing(grad_buf, grad_size);
                let a_grad_ref = GpuBuffer::from_existing(a_grad, a_grad_size);
                let b_grad_ref = GpuBuffer::from_existing(b_grad, b_grad_size);

                self.elementwise.dispatch_add(encoder, &a_grad_ref, &grad_ref, &a_grad_ref, *numel)?;
                self.elementwise.dispatch_add(encoder, &b_grad_ref, &grad_ref, &b_grad_ref, *numel)?;

                tracing::debug!("Backward: add grad numel={}", numel);
            }

            NodeKind::Scale { scale, numel } => {
                let input_id = inputs[0];

                let (grad_buf, grad_size) = Self::node_grad_info(graph, output_id)?;
                let (input_grad, input_grad_size) = Self::node_grad_info(graph, input_id)?;

                let temp = GpuBuffer::new(&self.device, grad_size, Some("grad_scale_temp"))?;
                let grad_ref = GpuBuffer::from_existing(grad_buf, grad_size);
                let input_grad_ref = GpuBuffer::from_existing(input_grad, input_grad_size);

                self.elementwise.dispatch_scale(encoder, &grad_ref, &temp, *scale, *numel)?;
                self.elementwise.dispatch_add(encoder, &input_grad_ref, &temp, &input_grad_ref, *numel)?;

                tracing::debug!("Backward: scale grad scale={} numel={}", scale, numel);
            }

            NodeKind::ReLU { numel } => {
                let input_id = inputs[0];

                let (grad_buf, grad_size) = Self::node_grad_info(graph, output_id)?;
                let (input_buf, _) = Self::node_buf_info(graph, input_id)?;
                let (input_grad, input_grad_size) = Self::node_grad_info(graph, input_id)?;

                let wg = (*numel + 255) / 256;

                let grad_ref = GpuBuffer::from_existing(grad_buf, grad_size);
                let input_buf_ref = GpuBuffer::from_existing(input_buf, grad_size);
                let input_grad_ref = GpuBuffer::from_existing(input_grad, input_grad_size);

                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ReLU Grad Bind Group"),
                    layout: &self.relu_grad_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: input_buf_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: grad_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: input_grad_ref.buffer().as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ReLU Grad"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.relu_grad_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg, 1, 1);
                drop(pass);

                tracing::debug!("Backward: relu grad numel={}", numel);
            }

            NodeKind::Softmax { rows, cols } => {
                let input_id = inputs[0];

                let (output_buf, output_size) = Self::node_buf_info(graph, output_id)?;
                let (grad_buf, grad_size) = Self::node_grad_info(graph, output_id)?;
                let (input_grad, input_grad_size) = Self::node_grad_info(graph, input_id)?;

                let total = rows * cols;
                let wg = (total + 255) / 256;

                let output_ref = GpuBuffer::from_existing(output_buf, output_size);
                let grad_ref = GpuBuffer::from_existing(grad_buf, grad_size);
                let input_grad_ref = GpuBuffer::from_existing(input_grad, input_grad_size);

                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Softmax Grad Bind Group"),
                    layout: &self.softmax_grad_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: output_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: grad_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: input_grad_ref.buffer().as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Softmax Grad"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.softmax_grad_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.set_immediates(0, &[rows.to_le_bytes(), cols.to_le_bytes()].concat());
                pass.dispatch_workgroups(wg, 1, 1);
                drop(pass);

                tracing::debug!("Backward: softmax grad rows={} cols={}", rows, cols);
            }

            NodeKind::RmsNorm { hidden_dim, eps: _ } => {
                let input_id = inputs[0];

                let (input_buf, input_size) = Self::node_buf_info(graph, input_id)?;
                let (output_buf, output_size) = Self::node_buf_info(graph, output_id)?;
                let (grad_buf, grad_size) = Self::node_grad_info(graph, output_id)?;
                let (input_grad, input_grad_size) = Self::node_grad_info(graph, input_id)?;

                let wg = (*hidden_dim + 255) / 256;

                let input_ref = GpuBuffer::from_existing(input_buf, input_size);
                let output_ref = GpuBuffer::from_existing(output_buf, output_size);
                let grad_ref = GpuBuffer::from_existing(grad_buf, grad_size);
                let input_grad_ref = GpuBuffer::from_existing(input_grad, input_grad_size);

                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RmsNorm Grad Bind Group"),
                    layout: &self.rmsnorm_grad_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: input_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: output_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: grad_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: input_grad_ref.buffer().as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RmsNorm Grad"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.rmsnorm_grad_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                let eps_bytes = 1e-5_f32.to_le_bytes();
                pass.set_immediates(0, &[
                    hidden_dim.to_le_bytes(),
                    eps_bytes,
                    0u32.to_le_bytes(),
                    0u32.to_le_bytes(),
                ].concat());
                pass.dispatch_workgroups(wg, 1, 1);
                drop(pass);

                tracing::debug!("Backward: rmsnorm grad hidden_dim={}", hidden_dim);
            }

            NodeKind::Linear { in_features, out_features, has_bias } => {
                let input_id = inputs[0];
                let weight_id = inputs[1];
                let bias_id = if *has_bias { Some(inputs[2]) } else { None };

                let (grad_buf, grad_size) = Self::node_grad_info(graph, output_id)?;
                let (input_buf, _) = Self::node_buf_info(graph, input_id)?;
                let (weight_buf, _) = Self::node_buf_info(graph, weight_id)?;
                let (input_grad, input_grad_size) = Self::node_grad_info(graph, input_id)?;
                let (weight_grad, weight_grad_size) = Self::node_grad_info(graph, weight_id)?;

                let batch_m = grad_size as u32 / (*out_features * std::mem::size_of::<f32>() as u32);

                let grad_ref = GpuBuffer::from_existing(grad_buf, grad_size);
                let input_ref = GpuBuffer::from_existing(input_buf, input_size(graph, input_id)?);
                let weight_ref = GpuBuffer::from_existing(weight_buf, input_size(graph, weight_id)?);
                let input_grad_ref = GpuBuffer::from_existing(input_grad, input_grad_size);
                let weight_grad_ref = GpuBuffer::from_existing(weight_grad, weight_grad_size);

                self.dispatch_matmul_grad_a(encoder, &grad_ref, &weight_ref, &input_grad_ref, batch_m, *out_features, *in_features)?;
                self.dispatch_matmul_grad_b(encoder, &input_ref, &grad_ref, &weight_grad_ref, batch_m, *in_features, *out_features)?;

                if let Some(bid) = bias_id {
                    let (bias_grad, bias_grad_size) = Self::node_grad_info(graph, bid)?;
                    let bias_grad_ref = GpuBuffer::from_existing(bias_grad, bias_grad_size);
                    self.dispatch_bias_grad_sum(encoder, &grad_ref, &bias_grad_ref, batch_m, *out_features)?;
                }

                tracing::debug!("Backward: linear grad in={} out={} has_bias={}", in_features, out_features, has_bias);
            }

            NodeKind::Loss { batch_size, vocab_size } => {
                let logits_id = inputs[0];
                let targets_id = inputs[1];

                let (logits_buf, logits_size) = Self::node_buf_info(graph, logits_id)?;
                let (targets_buf, targets_size) = Self::node_buf_info(graph, targets_id)?;
                let (logits_grad, logits_grad_size) = Self::node_grad_info(graph, logits_id)?;

                let total = batch_size * vocab_size;
                let wg = (total + 255) / 256;

                let logits_ref = GpuBuffer::from_existing(logits_buf, logits_size);
                let targets_ref = GpuBuffer::from_existing(targets_buf, targets_size);
                let logits_grad_ref = GpuBuffer::from_existing(logits_grad, logits_grad_size);

                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Loss Grad Bind Group"),
                    layout: &self.loss_grad_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: logits_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: targets_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: logits_grad_ref.buffer().as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Loss Grad"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.loss_grad_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.set_immediates(0, &[
                    batch_size.to_le_bytes(),
                    vocab_size.to_le_bytes(),
                    0u32.to_le_bytes(),
                    0u32.to_le_bytes(),
                ].concat());
                pass.dispatch_workgroups(wg, 1, 1);
                drop(pass);

                tracing::debug!("Backward: loss grad batch={} vocab={}", batch_size, vocab_size);
            }

            NodeKind::Embedding { vocab_size, hidden_dim } => {
                let token_ids_id = inputs[0];

                let (grad_buf, grad_size) = Self::node_grad_info(graph, output_id)?;
                let (token_ids_buf, token_ids_size) = Self::node_buf_info(graph, token_ids_id)?;

                let grad_ref = GpuBuffer::from_existing(grad_buf, grad_size);
                let token_ids_ref = GpuBuffer::from_existing(token_ids_buf, token_ids_size);

                let seq_len = grad_size as u32 / (*hidden_dim * std::mem::size_of::<f32>() as u32);

                let emb_grad_wgsl = r#"
                    struct EP {
                        vocab: u32,
                        dim: u32,
                        seq: u32,
                        _pad: u32,
                    }
                    @group(0) @binding(0) var<storage, read> token_ids: array<u32>;
                    @group(0) @binding(1) var<storage, read> grad_out: array<f32>;
                    @group(0) @binding(2) var<storage, read_write> grad_emb: array<f32>;
                    var<private> p: EP;

                    @compute @workgroup_size(256)
                    fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
                        let seq_idx = gid.x;
                        if (seq_idx >= p.seq) { return; }
                        let token_id = token_ids[seq_idx];
                        let grad_offset = seq_idx * p.dim;
                        let emb_offset = token_id * p.dim;
                        for (var d: u32 = 0u; d < p.dim; d = d + 1u) {
                            grad_emb[emb_offset + d] = grad_emb[emb_offset + d] + grad_out[grad_offset + d];
                        }
                    }
                "#;

                let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Embedding Grad Scatter Shader"),
                    source: wgpu::ShaderSource::Wgsl(emb_grad_wgsl.into()),
                });

                let emb_read = wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };

                let emb_read_b = wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };

                let emb_rw = wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                };

                let emb_grad_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Embedding Grad Layout"),
                    entries: &[emb_read, emb_read_b, emb_rw],
                });

                let emb_grad_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Embedding Grad Pipeline Layout"),
                    bind_group_layouts: &[Some(&emb_grad_layout)],
                    immediate_size: 16,
                });

                let emb_grad_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Embedding Grad Pipeline"),
                    layout: Some(&emb_grad_pipeline_layout),
                    module: &shader,
                    entry_point: Some("scatter"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

                let wg = (seq_len + 255) / 256;

                let grad_emb_size = (*vocab_size as usize) * (*hidden_dim as usize) * std::mem::size_of::<f32>();
                let grad_emb = GpuBuffer::new(&self.device, grad_emb_size, Some("grad_embedding_table"))?;

                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Embedding Grad Bind Group"),
                    layout: &emb_grad_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: token_ids_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: grad_ref.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: grad_emb.buffer().as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Embedding Grad Scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&emb_grad_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.set_immediates(0, &[
                    vocab_size.to_le_bytes(),
                    hidden_dim.to_le_bytes(),
                    seq_len.to_le_bytes(),
                    0u32.to_le_bytes(),
                ].concat());
                pass.dispatch_workgroups(wg, 1, 1);
                drop(pass);

                tracing::debug!("Backward: embedding grad vocab={} dim={} seq={}", vocab_size, hidden_dim, seq_len);
            }

            NodeKind::BlockSummaryCrossAttn { num_queries, hidden_dim, block_size } => {
                // Block Summary backward: dispatch GPU backward shader to compute
                // d_queries and d_tokens from d_output (upstream gradient).
                let hd = *hidden_dim as usize;
                let nq = *num_queries as usize;
                let bs = *block_size as usize;

                let tokens_id = inputs[0];
                let queries_id = inputs[1];
                let output_id = output_id;

                // Allocate gradient buffers if needed
                let tokens_grad_size = bs * hd * std::mem::size_of::<f32>();
                let queries_grad_size = nq * hd * std::mem::size_of::<f32>();

                {
                    let t_node = graph.get_node_mut(tokens_id).ok_or_else(|| {
                        FerrisResError::Device("block_summary tokens node not found".to_string())
                    })?;
                    if t_node.grad.size() == 0 {
                        t_node.grad = GpuBuffer::zeros(&self.device, &self.queue,
                            tokens_grad_size, Some("bs_tokens_grad"))?;
                    }
                }

                {
                    let q_node = graph.get_node_mut(queries_id).ok_or_else(|| {
                        FerrisResError::Device("block_summary queries node not found".to_string())
                    })?;
                    if q_node.grad.size() == 0 {
                        q_node.grad = GpuBuffer::zeros(&self.device, &self.queue,
                            queries_grad_size, Some("bs_queries_grad"))?;
                    }
                }

                // Get buffer references for dispatch
                let d_output_buf = {
                    let n = graph.get_node(output_id).ok_or_else(|| {
                        FerrisResError::Device("block_summary output node not found".to_string())
                    })?;
                    n.grad.buffer().clone()
                };
                let d_output_gpu = GpuBuffer::from_existing(d_output_buf, tokens_grad_size);

                let tokens_buf = {
                    let n = graph.get_node(tokens_id).ok_or_else(|| {
                        FerrisResError::Device("block_summary tokens node not found".to_string())
                    })?;
                    n.buf.buffer().clone()
                };
                let tokens_gpu = GpuBuffer::from_existing(tokens_buf, tokens_grad_size);

                let queries_buf = {
                    let n = graph.get_node(queries_id).ok_or_else(|| {
                        FerrisResError::Device("block_summary queries node not found".to_string())
                    })?;
                    n.buf.buffer().clone()
                };
                let queries_gpu = GpuBuffer::from_existing(queries_buf, queries_grad_size);

                let d_tokens_buf = {
                    let n = graph.get_node(tokens_id).ok_or_else(|| {
                        FerrisResError::Device("block_summary tokens node not found".to_string())
                    })?;
                    n.grad.buffer().clone()
                };
                let d_tokens_gpu = GpuBuffer::from_existing(d_tokens_buf, tokens_grad_size);

                let d_queries_buf = {
                    let n = graph.get_node(queries_id).ok_or_else(|| {
                        FerrisResError::Device("block_summary queries node not found".to_string())
                    })?;
                    n.grad.buffer().clone()
                };
                let d_queries_gpu = GpuBuffer::from_existing(d_queries_buf, queries_grad_size);

                // Dispatch GPU backward pass
                let gpu_op = crate::model::gemma_mapper::BlockSummaryGpuOp::new(
                    self.device.clone(), self.queue.clone(),
                    nq, hd, bs,
                );
                let mut bs_encoder = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("BlockSummary Backward") }
                );
                gpu_op.dispatch_backward(
                    &mut bs_encoder,
                    &d_output_gpu,
                    &queries_gpu,
                    &tokens_gpu,
                    &d_queries_gpu,
                    &d_tokens_gpu,
                )?;
                self.queue.submit(std::iter::once(bs_encoder.finish()));

                tracing::debug!("Backward: block_summary GPU dispatch nq={} hd={} bs={}", nq, hd, bs);
            }

            NodeKind::Parameter { .. } | NodeKind::Input { .. } => {
                tracing::debug!("Backward: skipping leaf node");
            }
        }

        Ok(())
    }

    fn dispatch_matmul_grad_a(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        grad_output: &GpuBuffer,
        b: &GpuBuffer,
        grad_a: &GpuBuffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let wg_x = (m + 63) / 64;
        let wg_y = (k + 63) / 64;

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Grad A Bind Group"),
            layout: &self.matmul_grad_a_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grad_output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_a.buffer().as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Grad A"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.matmul_grad_a_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_immediates(0, &[m.to_le_bytes(), n.to_le_bytes(), k.to_le_bytes()].concat());
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        Ok(())
    }

    fn dispatch_matmul_grad_b(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_b: &GpuBuffer,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<()> {
        let wg_x = (k + 63) / 64;
        let wg_y = (n + 63) / 64;

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Grad B Bind Group"),
            layout: &self.matmul_grad_b_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_b.buffer().as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Grad B"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.matmul_grad_b_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_immediates(0, &[m.to_le_bytes(), k.to_le_bytes(), n.to_le_bytes()].concat());
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        Ok(())
    }

    fn dispatch_bias_grad_sum(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        grad_output: &GpuBuffer,
        grad_bias: &GpuBuffer,
        batch: u32,
        out_features: u32,
    ) -> Result<()> {
        let wg = (out_features + 255) / 256;

        let bias_grad_wgsl = r#"
            struct BP {
                batch: u32,
                out: u32,
            }
            @group(0) @binding(0) var<storage, read> grad_out: array<f32>;
            @group(0) @binding(1) var<storage, read_write> grad_bias: array<f32>;
            var<private> p: BP;

            @compute @workgroup_size(256)
            fn sum_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
                let col = gid.x;
                if (col >= p.out) { return; }
                var s: f32 = 0.0;
                for (var r: u32 = 0u; r < p.batch; r = r + 1u) {
                    s = s + grad_out[r * p.out + col];
                }
                grad_bias[col] = s;
            }
        "#;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bias Grad Sum Shader"),
            source: wgpu::ShaderSource::Wgsl(bias_grad_wgsl.into()),
        });

        let bias_read = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bias_rw = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bias_grad_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bias Grad Sum Layout"),
            entries: &[bias_read, bias_rw],
        });

        let bias_grad_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bias Grad Sum Pipeline Layout"),
            bind_group_layouts: &[Some(&bias_grad_layout)],
            immediate_size: 8,
        });

        let bias_grad_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bias Grad Sum Pipeline"),
            layout: Some(&bias_grad_pipeline_layout),
            module: &shader,
            entry_point: Some("sum_rows"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bias Grad Sum Bind Group"),
            layout: &bias_grad_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: grad_output.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_bias.buffer().as_entire_binding() },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bias Grad Sum"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&bias_grad_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.set_immediates(0, &[batch.to_le_bytes(), out_features.to_le_bytes()].concat());
        pass.dispatch_workgroups(wg, 1, 1);
        drop(pass);

        Ok(())
    }
}

fn input_size(graph: &ComputationGraph, id: NodeId) -> Result<usize> {
    let node = graph.get_node(id).ok_or_else(|| {
        FerrisResError::Device(format!("node {:?} not found", id))
    })?;
    Ok(node.buf.size())
}
