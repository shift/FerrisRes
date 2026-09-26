use ferrisres::compute::{GpuBuffer, WgpuCompute};
use ferrisres::model::embedding::TokenEmbedding;
use std::sync::Arc;

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &GpuBuffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source.buffer(), 0, &staging, 0, source.size() as u64);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec()
}
fn check(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) {
    let embedding = TokenEmbedding::new(device.clone(), queue.clone(), 3, 2).unwrap();
    queue.write_buffer(
        embedding.weight().buffer(),
        0,
        bytemuck::cast_slice(&[1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]),
    );
    let ids = GpuBuffer::new(&device, 20, None).unwrap();
    queue.write_buffer(
        ids.buffer(),
        0,
        bytemuck::cast_slice(&[2u32, 0, 1, 3, u32::MAX]),
    );
    let short = GpuBuffer::new(&device, 16, None).unwrap();
    let long = GpuBuffer::new(&device, 48, None).unwrap();
    queue.write_buffer(short.buffer(), 0, bytemuck::cast_slice(&[77.0f32; 4]));
    queue.write_buffer(long.buffer(), 0, bytemuck::cast_slice(&[77.0f32; 12]));
    let mut encoder = device.create_command_encoder(&Default::default());
    embedding.forward(&mut encoder, &ids, &short, 1).unwrap();
    embedding.forward(&mut encoder, &ids, &long, 5).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(read(&device, &queue, &short), vec![5.0, -6.0, 77.0, 77.0]);
    assert_eq!(
        read(&device, &queue, &long),
        vec![5.0, -6.0, 1.0, -2.0, 3.0, -4.0, 0.0, 0.0, 0.0, 0.0, 77.0, 77.0]
    );
}
#[test]
fn uploaded_embeddings_and_dispatch_bounds_are_used() {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    check(
        Arc::new(compute.device().clone()),
        Arc::new(compute.queue().clone()),
    );
}
#[test]
fn embedding_works_without_immediates() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    assert!(!device.features().contains(wgpu::Features::IMMEDIATES));
    check(Arc::new(device), Arc::new(queue));
}
#[test]
fn invalid_dimensions_and_dispatch_sizes_are_rejected() {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    for (vocab, hidden) in [(0, 2), (2, 0), (usize::MAX, 2)] {
        assert!(TokenEmbedding::new(device.clone(), queue.clone(), vocab, hidden).is_err());
    }
    let embedding = TokenEmbedding::new(device.clone(), queue, 3, 2).unwrap();
    let ids = GpuBuffer::new(&device, 4, None).unwrap();
    let out = GpuBuffer::new(&device, 8, None).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(embedding.forward(&mut encoder, &ids, &out, 2).is_err());
    assert!(embedding
        .forward(&mut encoder, &ids, &out, u32::MAX)
        .is_err());
    embedding.forward(&mut encoder, &ids, &out, 0).unwrap();
}
