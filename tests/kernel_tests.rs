use std::sync::Arc;
use ferrisres::compute::{
    GpuBuffer, WgpuCompute, MatMulOp, RmsNormOp, SoftmaxOp, ElementWiseOp, MoEGatingOp, MoEExpertOp,
};
use ferrisres::compute::kernels::moe::{MOE_DISPATCH_WGSL, MOE_GATHER_WGSL};
use ferrisres::model::{BlockAttnResConfig, BlockAttnResLayer};

async fn create_test_compute() -> (WgpuCompute, Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let compute = WgpuCompute::new().await.unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    (compute, device, queue)
}

fn create_filled_buffer(device: &wgpu::Device, data: &[f32]) -> GpuBuffer {
    let bytes = bytemuck::cast_slice::<f32, u8>(data).len();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test buffer"),
        size: bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::cast_slice(data));
    buffer.unmap();
    GpuBuffer::from_existing(buffer, bytes)
}

fn create_output_buffer(device: &wgpu::Device, size: usize) -> GpuBuffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test output"),
        size: size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    GpuBuffer::from_existing(buffer, size)
}

fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &GpuBuffer,
) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: buffer.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(buffer.buffer(), 0, &staging, 0, buffer.size() as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let floats: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
    drop(data);
    staging.unmap();
    floats
}

fn read_buffer_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &GpuBuffer,
) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging u32"),
        size: buffer.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback u32"),
    });
    encoder.copy_buffer_to_buffer(buffer.buffer(), 0, &staging, 0, buffer.size() as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();

    let data = slice.get_mapped_range();
    let vals: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    staging.unmap();
    vals
}

#[tokio::test]
async fn test_matmul_2x2() {
    let (_compute, device, queue) = create_test_compute().await;
    let matmul = MatMulOp::new(&device, &queue);

    let a: &[f32] = &[1.0, 2.0, 3.0, 4.0];
    let b: &[f32] = &[5.0, 6.0, 7.0, 8.0];
    let a_buf = create_filled_buffer(&device, a);
    let b_buf = create_filled_buffer(&device, b);
    let c_buf = create_output_buffer(&device, 4 * std::mem::size_of::<f32>());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul 2x2"),
    });
    matmul.dispatch(&mut encoder, &a_buf, &b_buf, &c_buf, 2, 2, 2).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &c_buf);
    let expected: Vec<f32> = vec![19.0, 22.0, 43.0, 50.0];
    for (got, want) in result.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-2,
            "matmul 2x2: got {}, want {}",
            got,
            want
        );
    }
}

#[tokio::test]
async fn test_matmul_non_square() {
    let (_compute, device, queue) = create_test_compute().await;
    let matmul = MatMulOp::new(&device, &queue);

    let a: &[f32] = &[2.0, 3.0];
    let b: &[f32] = &[4.0, 5.0];
    let a_buf = create_filled_buffer(&device, a);
    let b_buf = create_filled_buffer(&device, b);
    let c_buf = create_output_buffer(&device, 1 * std::mem::size_of::<f32>());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul non-square"),
    });
    matmul.dispatch(&mut encoder, &a_buf, &b_buf, &c_buf, 1, 2, 1).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &c_buf);
    assert!(
        (result[0] - 23.0).abs() < 1e-2,
        "matmul non-square: got {}, want 23.0",
        result[0]
    );
}

#[tokio::test]
async fn test_elementwise_add() {
    let (_compute, device, queue) = create_test_compute().await;
    let ew = ElementWiseOp::new(&device, &queue);

    let a: &[f32] = &[1.0, 2.0, 3.0];
    let b: &[f32] = &[4.0, 5.0, 6.0];
    let a_buf = create_filled_buffer(&device, a);
    let b_buf = create_filled_buffer(&device, b);
    let c_buf = create_output_buffer(&device, 3 * std::mem::size_of::<f32>());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ew add"),
    });
    ew.dispatch_add(&mut encoder, &a_buf, &b_buf, &c_buf, 3).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &c_buf);
    let expected: Vec<f32> = vec![5.0, 7.0, 9.0];
    for (got, want) in result.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-2,
            "elementwise add: got {}, want {}",
            got,
            want
        );
    }
}

#[tokio::test]
async fn test_elementwise_scale() {
    let (_compute, device, queue) = create_test_compute().await;
    let ew = ElementWiseOp::new(&device, &queue);

    let a: &[f32] = &[2.0, 4.0, 6.0];
    let a_buf = create_filled_buffer(&device, a);
    let c_buf = create_output_buffer(&device, 3 * std::mem::size_of::<f32>());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ew scale"),
    });
    ew.dispatch_scale(&mut encoder, &a_buf, &c_buf, 0.5, 3).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &c_buf);
    let expected: Vec<f32> = vec![1.0, 2.0, 3.0];
    for (got, want) in result.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-2,
            "elementwise scale: got {}, want {}",
            got,
            want
        );
    }
}

#[tokio::test]
async fn test_rmsnorm() {
    let (_compute, device, queue) = create_test_compute().await;
    let rmsnorm = RmsNormOp::new(&device).unwrap();

    let input: &[f32] = &[3.0, 4.0];
    let in_buf = create_filled_buffer(&device, input);
    let out_buf = create_output_buffer(&device, 2 * std::mem::size_of::<f32>());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rmsnorm"),
    });
    rmsnorm.dispatch(&device, &queue, &mut encoder, &in_buf, &out_buf, 1, 2).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &out_buf);

    let rms = ((9.0f32 + 16.0) / 2.0 + 1e-5).sqrt();
    let expected_0 = 3.0 / rms;
    let expected_1 = 4.0 / rms;

    assert!(
        (result[0] - expected_0).abs() < 1e-3,
        "rmsnorm[0]: got {}, want {}",
        result[0],
        expected_0
    );
    assert!(
        (result[1] - expected_1).abs() < 1e-3,
        "rmsnorm[1]: got {}, want {}",
        result[1],
        expected_1
    );
}

#[tokio::test]
async fn test_softmax() {
    let (_compute, device, queue) = create_test_compute().await;
    let softmax = SoftmaxOp::new(&device, &queue).unwrap();

    let input: &[f32] = &[1.0, 2.0, 3.0];
    let in_buf = create_filled_buffer(&device, input);
    let out_buf = create_output_buffer(&device, 3 * std::mem::size_of::<f32>());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("softmax"),
    });
    softmax.dispatch(&mut encoder, &in_buf, &out_buf, 1, 3).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &out_buf);

    let e: Vec<f32> = [1.0f32, 2.0, 3.0].iter().map(|x| x.exp()).collect();
    let sum: f32 = e.iter().sum();
    let expected: Vec<f32> = e.iter().map(|x| x / sum).collect();

    for (got, want) in result.iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() < 1e-3,
            "softmax: got {}, want {}",
            got,
            want
        );
    }
}

#[tokio::test]
async fn test_moe_gating_top_k() {
    let (_compute, device, queue) = create_test_compute().await;
    let gating = MoEGatingOp::new(&device).unwrap();

    let gate_logits: &[f32] = &[0.1, 0.5, 0.3, 0.9, 0.2];
    let num_experts: u32 = 5;
    let batch_size: u32 = 1;
    let top_k: u32 = 2;
    let hidden_dim: u32 = 0;

    let logits_buf = create_filled_buffer(&device, gate_logits);
    let weights_buf = create_output_buffer(&device, num_experts as usize * std::mem::size_of::<f32>());
    let indices_buf = create_output_buffer(&device, num_experts as usize * std::mem::size_of::<u32>());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("moe gating top_k"),
    });
    gating.dispatch_top_k(&mut encoder, &logits_buf, &weights_buf, &indices_buf, batch_size, num_experts, top_k, hidden_dim).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &weights_buf);
    assert!(
        (result[0] - 0.9).abs() < 1e-3,
        "moe gating top_k[0]: got {}, want 0.9",
        result[0]
    );
    assert!(
        (result[1] - 0.5).abs() < 1e-3,
        "moe gating top_k[1]: got {}, want 0.5",
        result[1]
    );
    for i in 2..5usize {
        assert!(
            result[i].abs() < 1e-3,
            "moe gating top_k[{}]: got {}, want 0.0",
            i,
            result[i]
        );
    }
}

#[tokio::test]
async fn test_moe_dispatch_top_k() {
    let (_compute, device, queue) = create_test_compute().await;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MoE Dispatch Shader"),
        source: wgpu::ShaderSource::Wgsl(MOE_DISPATCH_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MoE Dispatch BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MoE Dispatch PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MoE Dispatch Pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("compute_top_k_indices"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let gate_logits: &[f32] = &[0.1, 0.5, 0.3, 0.9, 0.2];
    let num_experts: u32 = 5;
    let batch_size: u32 = 1;
    let top_k: u32 = 2;

    let logits_buf = create_filled_buffer(&device, gate_logits);
    let selected_buf = create_output_buffer(&device, (batch_size * top_k) as usize * std::mem::size_of::<u32>());
    let weights_buf = create_output_buffer(&device, (batch_size * top_k) as usize * std::mem::size_of::<f32>());

    let params: [u32; 4] = [num_experts, top_k, batch_size, 0];
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MoE Dispatch Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&params));
    params_buf.unmap();

    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MoE Dispatch BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: logits_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: selected_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: weights_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("moe dispatch top_k"),
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MoE Dispatch Pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(batch_size, 1, 1);
    drop(pass);
    queue.submit(std::iter::once(encoder.finish()));

    let experts = read_buffer_u32(&device, &queue, &selected_buf);
    assert_eq!(experts[0], 3, "moe dispatch: expected expert 3, got {}", experts[0]);
    assert_eq!(experts[1], 1, "moe dispatch: expected expert 1, got {}", experts[1]);

    let weights = read_buffer(&device, &queue, &weights_buf);
    // After softmax over K=2 selected logits [0.9, 0.5]:
    // sum_exp = exp(0.9) + exp(0.5) ≈ 2.4596 + 1.6487 = 4.1083
    // weight[0] = exp(0.9)/sum_exp ≈ 0.5985, weight[1] = exp(0.5)/sum_exp ≈ 0.4015
    let exp09 = 0.9f32.exp();
    let exp05 = 0.5f32.exp();
    let sum_exp = exp09 + exp05;
    let expected_w0 = exp09 / sum_exp;
    let expected_w1 = exp05 / sum_exp;
    assert!(
        (weights[0] - expected_w0).abs() < 1e-3,
        "moe dispatch weight[0]: got {}, want {} (softmax-normalised)",
        weights[0], expected_w0,
    );
    assert!(
        (weights[1] - expected_w1).abs() < 1e-3,
        "moe dispatch weight[1]: got {}, want {} (softmax-normalised)",
        weights[1], expected_w1,
    );
}

#[tokio::test]
async fn test_moe_expert_gather() {
    let (_compute, device, queue) = create_test_compute().await;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MoE Gather Shader"),
        source: wgpu::ShaderSource::Wgsl(MOE_GATHER_WGSL.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MoE Gather BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MoE Gather PL"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MoE Gather Pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("moe_up_proj"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let num_experts: u32 = 2;
    let top_k: u32 = 1;
    let batch_size: u32 = 1;
    let hidden_dim: u32 = 2;
    let intermediate_dim: u32 = 2;

    let gate_logits: &[f32] = &[0.3, 0.9];
    let input: &[f32] = &[1.0, 2.0];

    let logits_buf = create_filled_buffer(&device, gate_logits);
    let input_buf = create_filled_buffer(&device, input);

    let selected_buf = create_output_buffer(&device, (batch_size * top_k) as usize * std::mem::size_of::<u32>());
    let weights_buf = create_output_buffer(&device, (batch_size * top_k) as usize * std::mem::size_of::<f32>());
    let output_buf = create_output_buffer(&device, (batch_size * top_k * hidden_dim) as usize * std::mem::size_of::<f32>());
    let scratch_buf = create_output_buffer(&device, (batch_size * top_k * intermediate_dim) as usize * std::mem::size_of::<f32>());

    let expert_up: &[f32] = &[
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
    ];
    let expert_down: &[f32] = &[
        1.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 1.0,
    ];
    let up_buf = create_filled_buffer(&device, expert_up);
    let down_buf = create_filled_buffer(&device, expert_down);

    let gather_params: [u32; 5] = [num_experts, top_k, batch_size, hidden_dim, intermediate_dim];
    let gather_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MoE Gather Params"),
        size: 20,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    gather_params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&gather_params));
    gather_params_buf.unmap();

    let dispatch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MoE Dispatch Shader"),
        source: wgpu::ShaderSource::Wgsl(MOE_DISPATCH_WGSL.into()),
    });

    let dispatch_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MoE Dispatch BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
        ],
    });

    let dispatch_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MoE Dispatch PL"),
        bind_group_layouts: &[Some(&dispatch_bgl)],
        immediate_size: 0,
    });

    let dispatch_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MoE Dispatch Pipeline"),
        layout: Some(&dispatch_pl),
        module: &dispatch_shader,
        entry_point: Some("compute_top_k_indices"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let dispatch_params: [u32; 4] = [num_experts, top_k, batch_size, hidden_dim];
    let dispatch_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MoE Dispatch Params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    dispatch_params_buf.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&dispatch_params));
    dispatch_params_buf.unmap();

    let dispatch_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MoE Dispatch BG"),
        layout: &dispatch_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: logits_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: selected_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: weights_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dispatch_params_buf.as_entire_binding() },
        ],
    });

    let gather_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MoE Gather BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: selected_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: weights_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: up_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: down_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: output_buf.buffer().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: gather_params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: scratch_buf.buffer().as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("moe gather"),
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MoE Dispatch Pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&dispatch_pipeline);
    pass.set_bind_group(0, &dispatch_bg, &[]);
    pass.dispatch_workgroups(batch_size, 1, 1);
    drop(pass);

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MoE Gather Up Pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &gather_bg, &[]);
    pass.dispatch_workgroups((batch_size * top_k + 255) / 256, 1, 1);
    drop(pass);

    let down_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MoE Down Pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("moe_down_proj"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MoE Gather Down Pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&down_pipeline);
    pass.set_bind_group(0, &gather_bg, &[]);
    pass.dispatch_workgroups((batch_size * top_k + 255) / 256, 1, 1);
    drop(pass);

    let accumulate_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MoE Accumulate Pipeline"),
        layout: Some(&pl),
        module: &shader,
        entry_point: Some("moe_accumulate"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("MoE Accumulate Pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&accumulate_pipeline);
    pass.set_bind_group(0, &gather_bg, &[]);
    pass.dispatch_workgroups((batch_size * hidden_dim + 255) / 256, 1, 1);
    drop(pass);

    queue.submit(std::iter::once(encoder.finish()));

    let result = read_buffer(&device, &queue, &output_buf);
    // With top_k=1, softmax over a single logit = 1.0 (identity).
    // Expert up-proj (identity) on input [1.0, 2.0] → [1.0, 2.0] after ReLU.
    // Expert down-proj (identity) → [1.0, 2.0] * weight(1.0) = [1.0, 2.0].
    assert!(
        (result[0] - 1.0).abs() < 1e-2,
        "moe gather[0]: got {}, want 1.0",
        result[0]
    );
    assert!(
        (result[1] - 2.0).abs() < 1e-2,
        "moe gather[1]: got {}, want 2.0",
        result[1]
    );
}

/// Validates that MoEExpertOp (MOE_DISPATCH_WGSL) correctly selects top-2 expert indices
/// and produces softmax-normalised weights when num_experts=4 (well below the 32-cap).
/// This exercises the register-spill cap path: n_experts = min(4, 32) = 4.
/// See ADR-003: expert_weights_arr / logits_with_idx capped at 32 to prevent iGPU register spill.
#[tokio::test]
async fn test_moe_small_num_experts() {
    let (_compute, device, queue) = create_test_compute().await;

    // num_experts=4, top_k=2 — deliberately smaller than the 32-cap to validate the min() guard.
    let num_experts: u32 = 4;
    let top_k: u32 = 2;
    let batch_size: u32 = 1;
    let hidden_dim: u32 = 0;

    // Gate logits: [0.1, 0.9, 0.3, 0.5]
    // Top-2 by value: 0.9 (idx 1), 0.5 (idx 3)
    let gate_logits: &[f32] = &[0.1, 0.9, 0.3, 0.5];

    let logits_buf = create_filled_buffer(&device, gate_logits);
    let selected_buf = create_output_buffer(
        &device,
        (batch_size * top_k) as usize * std::mem::size_of::<u32>(),
    );
    let weights_buf = create_output_buffer(
        &device,
        (batch_size * top_k) as usize * std::mem::size_of::<f32>(),
    );

    let moe_expert_op = MoEExpertOp::new(&device)
        .expect("MoEExpertOp::new should succeed");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("moe small num_experts"),
    });
    moe_expert_op
        .dispatch_top_k(
            &mut encoder,
            &logits_buf,
            &selected_buf,
            &weights_buf,
            batch_size,
            num_experts,
            top_k,
            hidden_dim,
        )
        .expect("dispatch_top_k should succeed for num_experts < 32");
    queue.submit(std::iter::once(encoder.finish()));

    // Verify selected expert indices are [1, 3] (descending by logit: 0.9, 0.5)
    let experts = read_buffer_u32(&device, &queue, &selected_buf);
    assert_eq!(
        experts[0], 1,
        "test_moe_small_num_experts: expected expert index 1 (logit 0.9), got {}",
        experts[0]
    );
    assert_eq!(
        experts[1], 3,
        "test_moe_small_num_experts: expected expert index 3 (logit 0.5), got {}",
        experts[1]
    );

    // Verify weights are softmax-normalised over the top-2 selected logits [0.9, 0.5].
    // softmax: w[k] = exp(logit[k]) / (exp(0.9) + exp(0.5))
    let weights = read_buffer(&device, &queue, &weights_buf);
    let exp_09 = 0.9f32.exp();
    let exp_05 = 0.5f32.exp();
    let sum_exp = exp_09 + exp_05;
    let expected_w0 = exp_09 / sum_exp;
    let expected_w1 = exp_05 / sum_exp;

    assert!(
        (weights[0] - expected_w0).abs() < 1e-5,
        "test_moe_small_num_experts weight[0]: got {}, want {} (softmax of logit 0.9)",
        weights[0],
        expected_w0
    );
    assert!(
        (weights[1] - expected_w1).abs() < 1e-5,
        "test_moe_small_num_experts weight[1]: got {}, want {} (softmax of logit 0.5)",
        weights[1],
        expected_w1
    );

    // Verify weights sum to 1.0 (softmax invariant), tolerance 1e-5
    let weight_sum: f32 = weights.iter().sum();
    assert!(
        (weight_sum - 1.0).abs() < 1e-5,
        "test_moe_small_num_experts: weights must sum to 1.0, got {}",
        weight_sum
    );
}

/// Verify that BlockAttnResLayer constructs without panic when use_moe=true.
/// This test checks that MoELinear is wired in via the config flag.
#[tokio::test]
async fn test_moe_block_attn_res_layer_construction() {
    let (_compute, device, queue) = create_test_compute().await;

    let mut config = BlockAttnResConfig::new(64);
    config.use_moe = true;
    config.num_experts = 4;
    config.top_k = 2;
    config.attention_heads = 4;
    config.intermediate_dim = 128;

    let layer = BlockAttnResLayer::new(Arc::clone(&device), Arc::clone(&queue), &config, 0);
    assert!(
        layer.is_ok(),
        "BlockAttnResLayer::new with use_moe=true should succeed, got: {:?}",
        layer.err()
    );
}
