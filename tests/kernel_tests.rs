use std::sync::Arc;
use ferrisres::compute::{
    GpuBuffer, WgpuCompute, MatMulOp, RmsNormOp, SoftmaxOp, ElementWiseOp,
};

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

#[tokio::test]
async fn test_matmul_2x2() {
    let (_compute, device, queue) = create_test_compute().await;
    let matmul = MatMulOp::new(&device);

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
    let matmul = MatMulOp::new(&device);

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
    let ew = ElementWiseOp::new(&device);

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
    let ew = ElementWiseOp::new(&device);

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
    rmsnorm.dispatch(&device, &mut encoder, &in_buf, &out_buf, 1, 2).unwrap();
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
    let softmax = SoftmaxOp::new(&device).unwrap();

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
