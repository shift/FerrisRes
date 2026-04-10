use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use wgpu::{Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::compute::kernels::elementwise::ElementWiseOp;
use crate::error::{FerrisResError, Result};
#[allow(unused_imports)]

const ADAM_WGSL: &str = r#"
struct Params {
    beta1: f32,
    one_minus_beta1: f32,
    beta2: f32,
    one_minus_beta2: f32,
    lr: f32,
    epsilon: f32,
    bias_correction_m: f32,
    bias_correction_v: f32,
}

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> m: array<f32>;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> param: array<f32>;

var<private> p: Params;

@compute @workgroup_size(256)
fn adam_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let g = grad[idx];

    let old_m = m[idx];
    let old_v = v[idx];

    let new_m = p.beta1 * old_m + p.one_minus_beta1 * g;
    m[idx] = new_m;

    let new_v = p.beta2 * old_v + p.one_minus_beta2 * g * g;
    v[idx] = new_v;

    let m_hat = new_m / p.bias_correction_m;
    let v_hat = new_v / p.bias_correction_v;

    param[idx] = param[idx] - p.lr * m_hat / (sqrt(v_hat) + p.epsilon);
}
"#;

const CROSS_ENTROPY_WGSL: &str = r#"
struct LossParams {
    batch_size: u32,
    vocab_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> loss_output: array<f32>;

var<private> params: LossParams;

@compute @workgroup_size(64)
fn cross_entropy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.batch_size) {
        return;
    }

    let vocab_size = params.vocab_size;
    let row_offset = row * vocab_size;

    var max_val: f32 = logits[row_offset];
    for (var i: u32 = 1u; i < vocab_size; i = i + 1u) {
        let val = logits[row_offset + i];
        if (val > max_val) {
            max_val = val;
        }
    }

    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < vocab_size; i = i + 1u) {
        sum_exp = sum_exp + exp(logits[row_offset + i] - max_val);
    }

    let log_sum_exp = log(sum_exp) + max_val;
    let target_idx = targets[row];
    let target_logit = logits[row_offset + target_idx];

    loss_output[row] = log_sum_exp - target_logit;
}
"#;

pub struct SgdOptimizer {
    learning_rate: f32,
    #[allow(dead_code)]
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    elementwise: ElementWiseOp,
}

impl SgdOptimizer {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, learning_rate: f32) -> Self {
        tracing::info!("Creating SGD optimizer with lr={}", learning_rate);
        let elementwise = ElementWiseOp::new(&device, &queue);
        Self {
            learning_rate,
            device,
            queue,
            elementwise,
        }
    }

    pub fn step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        param: &GpuBuffer,
        grad: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        tracing::debug!(
            "SGD step: lr={} numel={}",
            self.learning_rate,
            numel
        );

        let neg_lr = -self.learning_rate;
        self.elementwise.dispatch_scale(encoder, grad, grad, neg_lr, numel)?;
        self.elementwise.dispatch_add(encoder, param, grad, param, numel)?;

        Ok(())
    }

    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn set_learning_rate(&mut self, lr: f32) {
        tracing::info!("SGD learning rate changed: {} -> {}", self.learning_rate, lr);
        self.learning_rate = lr;
    }
}

pub struct AdamOptimizer {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: AtomicU32,
    m_buffers: HashMap<String, GpuBuffer>,
    v_buffers: HashMap<String, GpuBuffer>,
    device: Arc<Device>,
    #[allow(dead_code)]
    queue: Arc<Queue>,
    #[allow(dead_code)]
    elementwise: ElementWiseOp,
    adam_pipeline: wgpu::ComputePipeline,
    adam_bind_group_layout: wgpu::BindGroupLayout,
}

impl AdamOptimizer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        tracing::info!(
            "Creating Adam optimizer: lr={} beta1={} beta2={} epsilon={}",
            learning_rate, beta1, beta2, epsilon
        );

        let elementwise = ElementWiseOp::new(&device, &queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Adam Optimizer Shader"),
            source: wgpu::ShaderSource::Wgsl(ADAM_WGSL.into()),
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

        let rw_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let rw_entry_v = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let rw_entry_param = wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let adam_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Adam Bind Group Layout"),
            entries: &[read_entry, rw_entry, rw_entry_v, rw_entry_param],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Adam Pipeline Layout"),
            bind_group_layouts: &[Some(&adam_bind_group_layout)],
            immediate_size: 0,
        });

        let adam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Adam Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("adam_update"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            timestep: AtomicU32::new(0),
            m_buffers: HashMap::new(),
            v_buffers: HashMap::new(),
            device,
            queue,
            elementwise,
            adam_pipeline,
            adam_bind_group_layout,
        }
    }

    pub fn register_param(&mut self, name: &str, shape: usize) -> Result<()> {
        let bytes = shape * std::mem::size_of::<f32>();
        let m_buf = GpuBuffer::zeros(&self.device, &self.queue, bytes, Some(&format!("adam_m_{}", name)))?;
        let v_buf = GpuBuffer::zeros(&self.device, &self.queue, bytes, Some(&format!("adam_v_{}", name)))?;
        self.m_buffers.insert(name.to_string(), m_buf);
        self.v_buffers.insert(name.to_string(), v_buf);
        tracing::debug!("Adam registered param '{}' with {} elements", name, shape);
        Ok(())
    }

    pub fn step(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        name: &str,
        param: &GpuBuffer,
        grad: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let t = self.timestep.fetch_add(1, Ordering::SeqCst) + 1;
        let bias_correction_m = 1.0 - self.beta1.powi(t as i32);
        let bias_correction_v = 1.0 - self.beta2.powi(t as i32);

        tracing::debug!(
            "Adam step: param={} numel={} t={} bc_m={:.6} bc_v={:.6}",
            name, numel, t, bias_correction_m, bias_correction_v
        );

        let m_buf = self.m_buffers.get(name).ok_or_else(|| {
            FerrisResError::Device(format!("Adam: param '{}' not registered", name))
        })?;
        let v_buf = self.v_buffers.get(name).ok_or_else(|| {
            FerrisResError::Device(format!("Adam: param '{}' not registered", name))
        })?;

        let params = AdamParams {
            beta1: self.beta1,
            one_minus_beta1: 1.0 - self.beta1,
            beta2: self.beta2,
            one_minus_beta2: 1.0 - self.beta2,
            lr: self.learning_rate,
            epsilon: self.epsilon,
            bias_correction_m,
            bias_correction_v,
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Adam Bind Group: {}", name)),
            layout: &self.adam_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: m_buf.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buf.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: param.buffer().as_entire_binding(),
                },
            ],
        });

        let workgroup_count = (numel + 255) / 256;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("Adam Update: {}", name)),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.adam_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(
            0,
            unsafe { std::slice::from_raw_parts(&params as *const AdamParams as *const u8, std::mem::size_of::<AdamParams>()) },
        );
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }

    pub fn timestep(&self) -> u32 {
        self.timestep.load(Ordering::SeqCst)
    }
}

#[repr(C, packed)]
struct AdamParams {
    beta1: f32,
    one_minus_beta1: f32,
    beta2: f32,
    one_minus_beta2: f32,
    lr: f32,
    epsilon: f32,
    bias_correction_m: f32,
    bias_correction_v: f32,
}

pub struct CrossEntropyLoss {
    device: Arc<Device>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CrossEntropyLoss {
    pub fn new(device: Arc<Device>) -> Self {
        tracing::info!("Creating CrossEntropyLoss compute pipeline");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cross Entropy Loss Shader"),
            source: wgpu::ShaderSource::Wgsl(CROSS_ENTROPY_WGSL.into()),
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

        let targets_read_entry = wgpu::BindGroupLayoutEntry {
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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cross Entropy Loss Bind Group Layout"),
            entries: &[read_entry, targets_read_entry, rw_entry],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cross Entropy Loss Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Cross Entropy Loss Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cross_entropy"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            device,
            pipeline,
            bind_group_layout,
        }
    }

    pub fn compute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        logits: &GpuBuffer,
        targets: &GpuBuffer,
        batch: u32,
        vocab_size: u32,
        loss_output: &GpuBuffer,
    ) -> Result<()> {
        tracing::debug!(
            "CrossEntropyLoss::compute batch={} vocab_size={}",
            batch,
            vocab_size
        );

        let params = LossParams {
            batch_size: batch,
            vocab_size,
            _pad0: 0,
            _pad1: 0,
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cross Entropy Loss Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: targets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: loss_output.buffer().as_entire_binding(),
                },
            ],
        });

        let workgroup_count = (batch + 63) / 64;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Cross Entropy Loss Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(
            0,
            unsafe { std::slice::from_raw_parts(&params as *const LossParams as *const u8, std::mem::size_of::<LossParams>()) },
        );
        pass.dispatch_workgroups(workgroup_count, 1, 1);

        drop(pass);

        Ok(())
    }
}

#[repr(C, packed)]
struct LossParams {
    batch_size: u32,
    vocab_size: u32,
    _pad0: u32,
    _pad1: u32,
}
