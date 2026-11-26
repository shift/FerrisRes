use ferrisres::compute::{GpuBuffer, WgpuCompute};
use ferrisres::model::{BlockAttnResConfig, BlockAttnResModel};
use std::sync::Arc;

#[tokio::test]
#[ignore = "requires a GPU adapter; run with --ignored on a GPU host"]
async fn test_block_attn_res_forward() {
    let _ = tracing_subscriber::fmt().try_init();

    let compute = WgpuCompute::new().await.expect("Failed to create WgpuCompute");

    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let hidden_dim = 64;
    let mut config = BlockAttnResConfig::new(hidden_dim);
    config.num_blocks = 2;
    config.block_size = 2;
    config.num_layers = config.num_blocks * config.block_size;
    config.intermediate_dim = 4 * hidden_dim;

    let model = BlockAttnResModel::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        config,
        1000,
    )
    .expect("Failed to create model");

    let batch_size: u32 = 1;
    let input_bytes = batch_size as usize * hidden_dim * std::mem::size_of::<f32>();

    let input_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01 + 1.0).collect();
    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Input"),
        size: input_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    input_buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::cast_slice(&input_data));
    input_buffer.unmap();
    let input = GpuBuffer::from_existing(input_buffer, input_bytes);

    let output_bytes = input_bytes;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Output"),
        size: output_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let output = GpuBuffer::from_existing(output_buffer, output_bytes);

    model
        .forward(&input, &output, batch_size)
        .expect("Forward pass failed");

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_bytes as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Readback Encoder"),
    });
    encoder.copy_buffer_to_buffer(output.buffer(), 0, &staging, 0, output_bytes as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mapped = slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice::<u8, f32>(&mapped);

    let all_zero = result.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "Output should not be all zeros");
}

/// The GPU inter-block path must be causal across the batch/position axis:
/// perturbing the input row at position `t` must leave every output row at
/// positions `< t` unchanged.
///
/// This is the GPU counterpart of the CPU gate
/// `cpu_block_attn_res_forward_is_causal`. It is ignored by default because it
/// needs a live wgpu adapter. The old `forward_inter_block` used the summed
/// `partial_sum` as the query and whole-sequence block means, so it fails this
/// test when run on a GPU.
#[tokio::test]
#[ignore = "requires a GPU adapter; run with --ignored on a GPU host"]
async fn test_block_attn_res_forward_is_causal_gpu() {
    use ferrisres::compute::GpuBuffer;

    let compute = WgpuCompute::new().await.expect("Failed to create WgpuCompute");
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let hidden_dim = 64usize;
    let batch_size = 4u32;
    let changed_pos = 2usize;

    let mut config = BlockAttnResConfig::new(hidden_dim);
    config.num_blocks = 2;
    config.block_size = 2;
    config.num_layers = config.num_blocks * config.block_size;
    config.intermediate_dim = 4 * hidden_dim;

    let model = BlockAttnResModel::new(Arc::clone(&device), Arc::clone(&queue), config, 1000)
        .expect("Failed to create model");

    let bytes = batch_size as usize * hidden_dim * std::mem::size_of::<f32>();
    let base: Vec<f32> = (0..batch_size as usize * hidden_dim)
        .map(|i| ((i * 37 % 101) as f32) * 0.01 + 1.0)
        .collect();
    let mut perturbed = base.clone();
    for d in 0..hidden_dim {
        perturbed[changed_pos * hidden_dim + d] += 0.5;
    }

    let run = |data: &[f32]| -> Vec<f32> {
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("causal_input"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        input_buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(data));
        input_buffer.unmap();
        let input = GpuBuffer::from_existing(input_buffer, bytes);

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("causal_output"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let output = GpuBuffer::from_existing(output_buffer, bytes);

        model
            .forward(&input, &output, batch_size)
            .expect("forward failed");

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("causal_staging"),
            size: bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("causal_readback"),
        });
        encoder.copy_buffer_to_buffer(output.buffer(), 0, &staging, 0, bytes as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        out
    };

    let base_out = run(&base);
    let pert_out = run(&perturbed);

    for t in 0..changed_pos {
        for d in 0..hidden_dim {
            let a = base_out[t * hidden_dim + d];
            let b = pert_out[t * hidden_dim + d];
            assert!(
                (a - b).abs() <= 1e-6,
                "GPU output at position {} dim {} changed when token {} changed: {} vs {}",
                t,
                d,
                changed_pos,
                a,
                b
            );
        }
    }
}
