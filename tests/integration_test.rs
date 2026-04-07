use ferrisres::compute::{GpuBuffer, WgpuCompute};
use ferrisres::model::{BlockAttnResConfig, BlockAttnResModel};
use std::sync::Arc;

#[tokio::test]
async fn test_block_attn_res_forward() {
    let _ = tracing_subscriber::fmt().with_env_filter("debug").try_init();

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

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Input"),
        size: input_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut mapped = input_buffer.slice(..).get_mapped_range_mut();
        let data = bytemuck::cast_slice_mut::<u8, f32>(&mut mapped);
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 + 1.0;
        }
        drop(mapped);
        input_buffer.unmap();
    }
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
    device.poll(wgpu::Maintain::Wait);

    let mapped = slice.get_mapped_range();
    let result: &[f32] = bytemuck::cast_slice(&mapped);

    let all_zero = result.iter().all(|&v| v == 0.0);
    assert!(!all_zero, "Output should not be all zeros");
}
