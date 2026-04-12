use std::sync::Arc;
use ferrisres::compute::{
    GpuBuffer, WgpuCompute, MatMulOp, MatMulDoubleBufferOp, FusedPatchEmbedOp,
    FlashDecodeOp, FlashDecodeTiledOp,
    RmsNormOp, SoftmaxOp, ElementWiseOp, MoEGatingOp, MoEExpertOp,
};
use ferrisres::compute::kernels::moe::{MOE_DISPATCH_WGSL, MOE_GATHER_WGSL};
use ferrisres::model::{BlockAttnResConfig, BlockAttnResLayer};
use ferrisres::inference::rag::{Document, ElasticRagStore, EmbedProfile};

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

// =============================================================================
// Fused Patch Embedding tests (task b3f74a12)
// =============================================================================

/// Build a flat HWC f32 image filled with a constant value.
#[allow(dead_code)]
fn make_image(h: u32, w: u32, c: u32, val: f32) -> Vec<f32> {
    vec![val; (h * w * c) as usize]
}

/// Explicit im2col + matmul reference (CPU, correct but slow).
/// Returns the `[n_patches, embed_dim]` embedding.
fn reference_patch_embed(
    image: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    h: u32, w: u32, c: u32,
    patch_size: u32,
    embed_dim: u32,
) -> Vec<f32> {
    let ph = (h / patch_size) as usize;
    let pw = (w / patch_size) as usize;
    let n = ph * pw;
    let k = (patch_size * patch_size * c) as usize;
    let d = embed_dim as usize;

    // im2col
    let mut patches = vec![0.0f32; n * k];
    for pi in 0..n {
        let p_row = pi / pw;
        let p_col = pi % pw;
        for ki in 0..k {
            let ch = ki % c as usize;
            let loc_xy = ki / c as usize;
            let ly = loc_xy / patch_size as usize;
            let lx = loc_xy % patch_size as usize;
            let iy = p_row * patch_size as usize + ly;
            let ix = p_col * patch_size as usize + lx;
            patches[pi * k + ki] = image[(iy * w as usize + ix) * c as usize + ch];
        }
    }

    // matmul: [N, K] × [K, D] → [N, D]
    let mut out = vec![0.0f32; n * d];
    for ni in 0..n {
        for di in 0..d {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += patches[ni * k + ki] * weight[ki * d + di];
            }
            if let Some(b) = bias {
                acc += b[di];
            }
            out[ni * d + di] = acc;
        }
    }
    out
}

#[tokio::test]
async fn test_fused_patch_embed_matches_reference_no_bias() {
    let (_, device, queue) = create_test_compute().await;

    // Small image: 8×8 RGB, patch_size=4, embed_dim=8
    let (h, w, c, p, d) = (8u32, 8u32, 3u32, 4u32, 8u32);
    let n_patches = (h / p) * (w / p);             // 2×2 = 4
    let k_size    = (p * p * c) as usize;           // 4*4*3 = 48
    let d_size    = d as usize;

    // Deterministic image: pixel at (y,x,ch) = (y*w + x)*c + ch as f32 * 0.01
    let mut image = vec![0.0f32; (h * w * c) as usize];
    for i in 0..image.len() { image[i] = i as f32 * 0.01; }

    // Random-ish weight: w[k,d] = (k*d_size + d) as f32 * 0.001
    let mut weight = vec![0.0f32; k_size * d_size];
    for i in 0..weight.len() { weight[i] = i as f32 * 0.001; }

    let expected = reference_patch_embed(&image, &weight, None, h, w, c, p, d);

    // GPU buffers
    let img_buf = create_filled_buffer(&device, &image);
    let w_buf   = create_filled_buffer(&device, &weight);
    let out_buf = create_output_buffer(&device, (n_patches * d) as usize * 4);

    let op = FusedPatchEmbedOp::new(&device, &queue);
    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("fused_patch_test") }
    );
    op.dispatch(&mut encoder, &img_buf, &w_buf, None, &out_buf, h, w, c, p, d).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let actual = read_buffer(&device, &queue, &out_buf);

    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-3,
            "fused_patch_embed mismatch at index {}: got {} expected {}", i, a, e
        );
    }
}

#[tokio::test]
async fn test_fused_patch_embed_matches_reference_with_bias() {
    let (_, device, queue) = create_test_compute().await;

    let (h, w, c, p, d) = (8u32, 8u32, 1u32, 4u32, 4u32);
    let n_patches = (h / p) * (w / p);
    let k_size    = (p * p * c) as usize;
    let d_size    = d as usize;

    let image  = vec![1.0f32; (h * w * c) as usize];
    let weight = vec![0.5f32; k_size * d_size];
    let bias   = vec![0.25f32; d_size];

    let expected = reference_patch_embed(&image, &weight, Some(&bias), h, w, c, p, d);

    let img_buf  = create_filled_buffer(&device, &image);
    let w_buf    = create_filled_buffer(&device, &weight);
    let b_buf    = create_filled_buffer(&device, &bias);
    let out_buf  = create_output_buffer(&device, (n_patches * d) as usize * 4);

    let op = FusedPatchEmbedOp::new(&device, &queue);
    let mut encoder = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("fused_patch_bias_test") }
    );
    op.dispatch(&mut encoder, &img_buf, &w_buf, Some(&b_buf), &out_buf, h, w, c, p, d).unwrap();
    queue.submit(std::iter::once(encoder.finish()));

    let actual = read_buffer(&device, &queue, &out_buf);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-3,
            "fused_patch_embed(bias) mismatch at {}: got {} expected {}", i, a, e
        );
    }
}

#[tokio::test]
async fn test_fused_patch_embed_output_byte_size() {
    // Static helper — no GPU needed.
    assert_eq!(FusedPatchEmbedOp::output_byte_size(224, 224, 16, 768),
               (14 * 14 * 768 * 4));   // 196 patches × 768 × 4 bytes
    assert_eq!(FusedPatchEmbedOp::output_byte_size(8, 8, 4, 8),
               (4 * 8 * 4));           // (2×2) patches × 8 dims × 4 bytes
}

// =============================================================================
// MatMulDoubleBufferOp tests (task f4c0a839)
// =============================================================================

/// CPU reference multiply
fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k { acc += a[i * k + l] * b[l * n + j]; }
            c[i * n + j] = acc;
        }
    }
    c
}

#[tokio::test]
async fn test_matmul_double_buf_2x2() {
    let (_, device, queue) = create_test_compute().await;

    let a = vec![1.0f32, 2.0, 3.0, 4.0];  // 2×2
    let b = vec![5.0f32, 6.0, 7.0, 8.0];  // 2×2
    let expected = matmul_ref(&a, &b, 2, 2, 2);

    let a_buf = create_filled_buffer(&device, &a);
    let b_buf = create_filled_buffer(&device, &b);
    let c_buf = create_output_buffer(&device, 4 * 4);

    let op = MatMulDoubleBufferOp::new(&device, &queue);
    let mut enc = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("mm_db_2x2") }
    );
    op.dispatch(&mut enc, &a_buf, &b_buf, &c_buf, 2, 2, 2).unwrap();
    queue.submit(std::iter::once(enc.finish()));

    let actual = read_buffer(&device, &queue, &c_buf);
    for (i, (a, e)) in actual[..4].iter().zip(expected.iter()).enumerate() {
        assert!((a - e).abs() < 1e-4, "mm_db[{}]: got {} expected {}", i, a, e);
    }
}

#[tokio::test]
async fn test_matmul_double_buf_matches_single_buf_32x64x32() {
    // Larger matrix: ensure double-buffer and single-buffer agree numerically.
    let (_, device, queue) = create_test_compute().await;

    let (m, k, n) = (32usize, 64usize, 32usize);
    let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.001).collect();

    let a_buf  = create_filled_buffer(&device, &a);
    let b_buf  = create_filled_buffer(&device, &b);
    let c1_buf = create_output_buffer(&device, m * n * 4);
    let c2_buf = create_output_buffer(&device, m * n * 4);

    let single = MatMulOp::new(&device, &queue);
    let double = MatMulDoubleBufferOp::new(&device, &queue);

    let mut enc = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("mm_compare") }
    );
    single.dispatch(&mut enc, &a_buf, &b_buf, &c1_buf, m as u32, k as u32, n as u32).unwrap();
    double.dispatch(&mut enc, &a_buf, &b_buf, &c2_buf, m as u32, k as u32, n as u32).unwrap();
    queue.submit(std::iter::once(enc.finish()));

    let r1 = read_buffer(&device, &queue, &c1_buf);
    let r2 = read_buffer(&device, &queue, &c2_buf);

    for (i, (a, b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "mm_double_buf vs single_buf mismatch at [{}]: {} vs {}", i, a, b
        );
    }
}

// =============================================================================
// ElasticRagStore tests (task d7a18c03)
// =============================================================================

fn embed(vals: &[f32]) -> Vec<f32> { vals.to_vec() }

#[test]
fn test_elastic_rag_profile_query_dims() {
    assert_eq!(EmbedProfile::Integrated.query_dim(768), 64);
    assert_eq!(EmbedProfile::LowEnd.query_dim(768),     128);
    assert_eq!(EmbedProfile::MidRange.query_dim(768),   256);
    assert_eq!(EmbedProfile::HighEnd.query_dim(768),    768);
    // Clamp to d_max
    assert_eq!(EmbedProfile::Integrated.query_dim(32),  32);
}

#[test]
fn test_elastic_rag_search_top1() {
    let mut store = ElasticRagStore::new(4, vec![2, 4], EmbedProfile::HighEnd);
    store.add(Document::new("a", "doc a"), embed(&[1.0, 0.0, 0.0, 0.0]));
    store.add(Document::new("b", "doc b"), embed(&[0.0, 1.0, 0.0, 0.0]));
    store.add(Document::new("c", "doc c"), embed(&[0.0, 0.0, 1.0, 0.0]));

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let results = store.search(&query, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.id, "a");
    assert!((results[0].score - 1.0).abs() < 1e-5);
}

#[test]
fn test_elastic_rag_search_reduced_dim() {
    // query_dim=2: only first 2 dims used → docs distinguishable by [1,0] prefix
    let mut store = ElasticRagStore::new(4, vec![2, 4], EmbedProfile::Integrated);
    // Integrated → query_dim = min(64, 4) = 4; override to 2 for this test
    store.set_query_dim(2);

    store.add(Document::new("a", "doc a"), embed(&[1.0, 0.0, 0.0, 0.0]));
    store.add(Document::new("b", "doc b"), embed(&[0.0, 1.0, 0.0, 0.0]));
    store.add(Document::new("c", "doc c"), embed(&[0.5, 0.5, 99.0, 99.0])); // noisy tail

    // Query matches "c" on dim[:2] = [0.5, 0.5], but "a" = [1,0] is more aligned to [1,0.1]
    let query = vec![1.0f32, 0.1, 0.0, 0.0];
    let results = store.search(&query, 3);
    assert_eq!(results.len(), 3);
    // "a" should rank first (closest in first 2 dims)
    assert_eq!(results[0].document.id, "a");
}

#[test]
fn test_elastic_rag_coarse_then_fine() {
    let mut store = ElasticRagStore::new(4, vec![2, 4], EmbedProfile::HighEnd);
    // doc "a": [1, 0, 1, 0] — coarse [1,0] and fine [1,0,1,0]
    // doc "b": [1, 0, 0, 1] — coarse [1,0] same; fine differs
    // doc "c": [0, 1, 0, 0] — coarse different
    store.add(Document::new("a", "doc a"), embed(&[1.0, 0.0, 1.0, 0.0]));
    store.add(Document::new("b", "doc b"), embed(&[1.0, 0.0, 0.0, 1.0]));
    store.add(Document::new("c", "doc c"), embed(&[0.0, 1.0, 0.0, 0.0]));

    // Query: [1, 0, 1, 0] → perfect match with "a"
    let q = vec![1.0f32, 0.0, 1.0, 0.0];
    let results = store.search_coarse_then_fine(&q, 2, 2, 1);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document.id, "a");
}

#[test]
fn test_elastic_rag_apply_profile() {
    let mut store = ElasticRagStore::new(768, vec![64, 128, 256, 768], EmbedProfile::HighEnd);
    assert_eq!(store.query_dim(), 768);
    store.apply_profile(EmbedProfile::Integrated);
    assert_eq!(store.query_dim(), 64);
    store.apply_profile(EmbedProfile::MidRange);
    assert_eq!(store.query_dim(), 256);
}

#[test]
fn test_elastic_rag_empty_search() {
    let store = ElasticRagStore::new(4, vec![2, 4], EmbedProfile::HighEnd);
    let results = store.search(&[1.0, 0.0, 0.0, 0.0], 5);
    assert!(results.is_empty());
}

// =============================================================================
// FlashDecodeTiledOp tests (task f4c0a839 — tiled KV attention)
// =============================================================================

use ferrisres::compute::kernels::tome_merge::bipartite_match;

/// CPU reference: single-query attention with causal mask (all positions visible).
fn flash_decode_ref(
    query:     &[f32],   // [num_heads, head_dim]
    key_cache: &[f32],   // [seq_len, num_heads, head_dim]
    value_cache:&[f32],  // [seq_len, num_heads, head_dim]
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {           // [num_heads, head_dim]
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; num_heads * head_dim];

    for h in 0..num_heads {
        let q_base = h * head_dim;
        // Pass 1: max score
        let mut max_score = f32::NEG_INFINITY;
        for s in 0..seq_len {
            let k_base = s * num_heads * head_dim + h * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim { dot += query[q_base + d] * key_cache[k_base + d]; }
            max_score = max_score.max(dot * scale);
        }
        // Pass 2: weighted sum
        let mut sum_exp = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];
        for s in 0..seq_len {
            let k_base = s * num_heads * head_dim + h * head_dim;
            let mut dot = 0.0f32;
            for d in 0..head_dim { dot += query[q_base + d] * key_cache[k_base + d]; }
            let weight = (dot * scale - max_score).exp();
            sum_exp += weight;
            let v_base = s * num_heads * head_dim + h * head_dim;
            for d in 0..head_dim { acc[d] += weight * value_cache[v_base + d]; }
        }
        let inv = 1.0 / sum_exp;
        for d in 0..head_dim { output[h * head_dim + d] = acc[d] * inv; }
    }
    output
}

#[tokio::test]
async fn test_flash_decode_tiled_matches_reference() {
    let (_, device, queue) = create_test_compute().await;

    let seq_len = 8u32;
    let num_heads = 2u32;
    let head_dim = 4u32;

    let mut query = vec![0.0f32; (num_heads * head_dim) as usize];
    let mut kc = vec![0.0f32; (seq_len * num_heads * head_dim) as usize];
    let mut vc = vec![0.0f32; (seq_len * num_heads * head_dim) as usize];
    for i in 0..query.len()  { query[i] = (i as f32 * 0.1).sin(); }
    for i in 0..kc.len()     { kc[i]    = (i as f32 * 0.05).cos(); }
    for i in 0..vc.len()     { vc[i]    = (i as f32 * 0.07).sin(); }

    let expected = flash_decode_ref(
        &query, &kc, &vc,
        seq_len as usize, num_heads as usize, head_dim as usize,
    );

    let q_buf = create_filled_buffer(&device, &query);
    let k_buf = create_filled_buffer(&device, &kc);
    let v_buf = create_filled_buffer(&device, &vc);
    let o_buf = create_output_buffer(&device, (num_heads * head_dim) as usize * 4);

    let op = FlashDecodeTiledOp::new(&device, &queue).unwrap();
    let mut enc = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("tiled_decode_test") }
    );
    op.dispatch(&mut enc, &q_buf, &k_buf, &v_buf, &o_buf, seq_len, num_heads, head_dim, 4).unwrap();
    queue.submit(std::iter::once(enc.finish()));

    let actual = read_buffer(&device, &queue, &o_buf);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-3,
            "tiled_decode mismatch [{}]: got {} expected {}", i, a, e
        );
    }
}

#[tokio::test]
async fn test_flash_decode_tiled_vs_original() {
    let (_, device, queue) = create_test_compute().await;

    let seq_len = 16u32;
    let num_heads = 4u32;
    let head_dim = 8u32;

    let mut query = vec![0.0f32; (num_heads * head_dim) as usize];
    let mut kc = vec![0.0f32; (seq_len * num_heads * head_dim) as usize];
    let mut vc = vec![0.0f32; (seq_len * num_heads * head_dim) as usize];
    for i in 0..query.len()  { query[i] = (i as f32 * 0.11).sin(); }
    for i in 0..kc.len()     { kc[i]    = (i as f32 * 0.03).cos(); }
    for i in 0..vc.len()     { vc[i]    = (i as f32 * 0.09).sin(); }

    let q_buf = create_filled_buffer(&device, &query);
    let k_buf = create_filled_buffer(&device, &kc);
    let v_buf = create_filled_buffer(&device, &vc);
    let o1 = create_output_buffer(&device, (num_heads * head_dim) as usize * 4);
    let o2 = create_output_buffer(&device, (num_heads * head_dim) as usize * 4);

    let orig = FlashDecodeOp::new(&device, &queue).unwrap();
    let tiled = FlashDecodeTiledOp::new(&device, &queue).unwrap();

    let mut enc = device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("decode_compare") }
    );
    orig.dispatch(&mut enc, &q_buf, &k_buf, &v_buf, &o1, seq_len, num_heads, head_dim).unwrap();
    tiled.dispatch(&mut enc, &q_buf, &k_buf, &v_buf, &o2, seq_len, num_heads, head_dim, 4).unwrap();
    queue.submit(std::iter::once(enc.finish()));

    let r1 = read_buffer(&device, &queue, &o1);
    let r2 = read_buffer(&device, &queue, &o2);

    for (i, (a, b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "tiled vs original mismatch [{}]: {} vs {}", i, a, b
        );
    }
}

// =============================================================================
// ToMe bipartite matching tests (task c9e2d541)
// =============================================================================


#[test]
fn test_bipartite_match_merge_count() {
    // 8 tokens, merge top 2 pairs → 6 output tokens
    let keys: Vec<f32> = vec![
        1.0, 0.0,  // token 0
        0.9, 0.1,  // token 1 (similar to 0)
        0.0, 1.0,  // token 2
        0.1, 0.9,  // token 3 (similar to 2)
        1.0, 1.0,  // token 4
        0.5, 0.5,  // token 5
        -1.0, 0.0, // token 6
        0.0, -1.0, // token 7
    ];
    let result = bipartite_match(&keys, 8, 2, 2);
    assert_eq!(result.n_out, 6);
    assert_eq!(result.pair_a.len(), 6);
    assert_eq!(result.pair_b.len(), 6);
}

#[test]
fn test_bipartite_match_no_merge() {
    let keys = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
    let result = bipartite_match(&keys, 4, 2, 0);
    assert_eq!(result.n_out, 4);
    // All tokens should be copies (pair_a == pair_b)
    for i in 0..4 {
        assert_eq!(result.pair_a[i], result.pair_b[i], "token {} should be unmerged", i);
    }
}

#[test]
fn test_bipartite_match_similar_tokens_merged() {
    // Two identical pairs: (0,1) and (2,3)
    let keys: Vec<f32> = vec![
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
    ];
    let result = bipartite_match(&keys, 4, 2, 2);
    assert_eq!(result.n_out, 2, "should merge 2 pairs from 4 tokens");
    // Check that the merged pairs have different source indices
    let merged_count = result.pair_a.iter().zip(result.pair_b.iter())
        .filter(|(a, b)| a != b).count();
    assert_eq!(merged_count, 2, "both outputs should be merged pairs");
}

#[test]
fn test_bipartart_match_r_clamped() {
    let keys = vec![1.0f32, 0.0, 0.0, 1.0];
    // r=10 but only 2 tokens, should clamp to n_tokens/2 = 1
    let result = bipartite_match(&keys, 2, 1, 10);
    assert_eq!(result.n_out, 1);
}
