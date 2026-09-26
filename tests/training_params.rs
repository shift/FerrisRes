//! GPU training parameter plumbing, including adapters without IMMEDIATES.
use std::sync::Arc;
use ferrisres::autodiff::{BackwardPass, ComputationGraph};
use ferrisres::compute::GpuBuffer;
use ferrisres::training::{AdamOptimizer, CrossEntropyLoss};
use wgpu::util::DeviceExt;

fn device_without_immediates() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("GPU adapter required for training parameter regressions");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    assert!(!device.features().contains(wgpu::Features::IMMEDIATES));
    (Arc::new(device), Arc::new(queue))
}

fn buffer(device: &wgpu::Device, bytes: &[u8]) -> GpuBuffer {
    GpuBuffer::from_existing(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("training regression"), contents: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    }), bytes.len())
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &GpuBuffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("training readback"), size: source.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source.buffer(), 0, &staging, 0, source.size() as u64);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    result
}

#[test]
fn cross_entropy_uses_distinct_params_per_dispatch_without_immediates() {
    let (device, queue) = device_without_immediates();
    let op = CrossEntropyLoss::new(device.clone());
    let logits2 = buffer(&device, bytemuck::cast_slice(&[0.0f32, 0.0]));
    let logits3 = buffer(&device, bytemuck::cast_slice(&[0.0f32, 0.0, 0.0]));
    let targets = buffer(&device, bytemuck::cast_slice(&[0u32]));
    let loss2 = buffer(&device, bytemuck::cast_slice(&[-1.0f32]));
    let loss3 = buffer(&device, bytemuck::cast_slice(&[-1.0f32]));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    op.compute(&mut encoder, &logits2, &targets, 1, 2, &loss2).unwrap();
    op.compute(&mut encoder, &logits3, &targets, 1, 3, &loss3).unwrap();
    queue.submit([encoder.finish()]);
    assert!((read(&device, &queue, &loss2)[0] - 2.0f32.ln()).abs() < 1e-5);
    assert!((read(&device, &queue, &loss3)[0] - 3.0f32.ln()).abs() < 1e-5);
}

#[test]
fn adam_step_matches_reference_without_immediates() {
    let (device, queue) = device_without_immediates();
    let mut optimizer = AdamOptimizer::new(device.clone(), queue.clone(), 0.1, 0.9, 0.999, 1e-8);
    optimizer.register_param("weight", 2).unwrap();
    let weights = buffer(&device, bytemuck::cast_slice(&[1.0f32, 1.0]));
    let grads = buffer(&device, bytemuck::cast_slice(&[0.25f32, -0.5]));
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    optimizer.step(&mut encoder, "weight", &weights, &grads, 2).unwrap();
    queue.submit([encoder.finish()]);
    let actual = read(&device, &queue, &weights);
    assert!((actual[0] - 0.9).abs() < 1e-5, "{actual:?}");
    assert!((actual[1] - 1.1).abs() < 1e-5, "{actual:?}");
}

#[test]
fn softmax_backward_matches_reference_without_immediates() {
    let (device, queue) = device_without_immediates();
    let backward = BackwardPass::new(device.clone(), queue.clone());
    let mut graph = ComputationGraph::new(device.clone(), queue.clone());
    let input = graph.add_parameter("input", buffer(&device, bytemuck::cast_slice(&[0.0f32, 0.0]))).unwrap();
    let probabilities = graph.record_softmax(input, buffer(&device, bytemuck::cast_slice(&[0.5f32, 0.5])), 1, 2).unwrap();
    let targets = graph.add_input("targets", buffer(&device, bytemuck::cast_slice(&[0u32]))).unwrap();
    // CE consumes the softmax output as logits, yielding upstream [-0.5, 0.5].
    let loss = graph.record_loss(probabilities, targets, buffer(&device, bytemuck::cast_slice(&[2.0f32.ln()])), 1, 2).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    backward.run(&mut graph, &mut encoder, loss).unwrap();
    queue.submit([encoder.finish()]);
    let actual = read(&device, &queue, graph.get_node(input).unwrap().grad());
    assert!((actual[0] + 0.25).abs() < 1e-5, "{actual:?}");
    assert!((actual[1] - 0.25).abs() < 1e-5, "{actual:?}");
}

#[test]
fn loss_backward_matches_reference_without_immediates() {
    let (device, queue) = device_without_immediates();
    let backward = BackwardPass::new(device.clone(), queue.clone());
    let mut graph = ComputationGraph::new(device.clone(), queue.clone());
    let logits = graph.add_parameter("logits", buffer(&device, bytemuck::cast_slice(&[0.0f32, 0.0]))).unwrap();
    let targets = graph.add_input("targets", buffer(&device, bytemuck::cast_slice(&[0u32]))).unwrap();
    let loss = graph.record_loss(logits, targets, buffer(&device, bytemuck::cast_slice(&[2.0f32.ln()])), 1, 2).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    backward.run(&mut graph, &mut encoder, loss).unwrap();
    queue.submit([encoder.finish()]);
    let actual = read(&device, &queue, graph.get_node(logits).unwrap().grad());
    assert!((actual[0] + 0.5).abs() < 1e-5, "{actual:?}");
    assert!((actual[1] - 0.5).abs() < 1e-5, "{actual:?}");
}
