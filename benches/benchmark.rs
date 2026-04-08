use std::sync::Arc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ferrisres::compute::{
    WgpuCompute, GpuBuffer, MatMulOp, RmsNormOp, SoftmaxOp, ElementWiseOp,
};

fn bench_matmul(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let compute = WgpuCompute::new().await.unwrap();
        let device = Arc::new(compute.device().clone());
        let queue = Arc::new(compute.queue().clone());

        let matmul_op = MatMulOp::new(&device);

        let mut group = c.benchmark_group("matmul");
        for size in [128u32, 256, 512] {
            let bytes = size as usize * size as usize * std::mem::size_of::<f32>();
            let a_buf = GpuBuffer::new(&device, bytes, Some("bench_a")).unwrap();
            let b_buf = GpuBuffer::new(&device, bytes, Some("bench_b")).unwrap();
            let c_buf = GpuBuffer::new(&device, bytes, Some("bench_c")).unwrap();

            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bench_matmul"),
                    });
                    matmul_op.dispatch(&mut encoder, &a_buf, &b_buf, &c_buf, size, size, size).unwrap();
                    queue.submit(std::iter::once(encoder.finish()));
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                });
            });
        }
        group.finish();
    });
}

fn bench_rmsnorm(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let compute = WgpuCompute::new().await.unwrap();
        let device = Arc::new(compute.device().clone());
        let queue = Arc::new(compute.queue().clone());

        let mut group = c.benchmark_group("rmsnorm");
        for hidden_dim in [512u32, 1024] {
            let rmsnorm_op = RmsNormOp::new(&device).unwrap();
            let rows = 1u32;
            let bytes = rows as usize * hidden_dim as usize * std::mem::size_of::<f32>();
            let input = GpuBuffer::new(&device, bytes, Some("bench_rmsnorm_in")).unwrap();
            let output = GpuBuffer::new(&device, bytes, Some("bench_rmsnorm_out")).unwrap();

            group.bench_with_input(BenchmarkId::from_parameter(hidden_dim), &hidden_dim, |b, &hd| {
                b.to_async(&rt).iter(|| async {
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bench_rmsnorm"),
                    });
                    rmsnorm_op.dispatch(&device, &mut encoder, &input, &output, rows, hd).unwrap();
                    queue.submit(std::iter::once(encoder.finish()));
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                });
            });
        }
        group.finish();
    });
}

fn bench_softmax(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let compute = WgpuCompute::new().await.unwrap();
        let device = Arc::new(compute.device().clone());
        let queue = Arc::new(compute.queue().clone());

        let softmax_op = SoftmaxOp::new(&device).unwrap();

        let mut group = c.benchmark_group("softmax");
        for (rows, cols) in [(1u32, 512u32), (8u32, 512u32)] {
            let bytes = rows as usize * cols as usize * std::mem::size_of::<f32>();
            let input = GpuBuffer::new(&device, bytes, Some("bench_softmax_in")).unwrap();
            let output = GpuBuffer::new(&device, bytes, Some("bench_softmax_out")).unwrap();

            let param_str = format!("{}_{}", rows, cols);
            group.bench_with_input(BenchmarkId::new("rows_cols", &param_str), &(rows, cols), |b, &(r, cl)| {
                b.to_async(&rt).iter(|| async {
                    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bench_softmax"),
                    });
                    softmax_op.dispatch(&mut encoder, &input, &output, r, cl).unwrap();
                    queue.submit(std::iter::once(encoder.finish()));
                    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                });
            });
        }
        group.finish();
    });
}

fn bench_elementwise(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let compute = WgpuCompute::new().await.unwrap();
        let device = Arc::new(compute.device().clone());
        let queue = Arc::new(compute.queue().clone());

        let ew_op = ElementWiseOp::new(&device);
        let numel = 4096u32;
        let bytes = numel as usize * std::mem::size_of::<f32>();
        let a_buf = GpuBuffer::new(&device, bytes, Some("bench_ew_a")).unwrap();
        let b_buf = GpuBuffer::new(&device, bytes, Some("bench_ew_b")).unwrap();
        let c_buf = GpuBuffer::new(&device, bytes, Some("bench_ew_c")).unwrap();

        let mut group = c.benchmark_group("elementwise");

        group.bench_function("add", |b| {
            b.to_async(&rt).iter(|| async {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench_ew_add"),
                });
                ew_op.dispatch_add(&mut encoder, &a_buf, &b_buf, &c_buf, numel).unwrap();
                queue.submit(std::iter::once(encoder.finish()));
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            });
        });

        group.bench_function("scale", |b| {
            b.to_async(&rt).iter(|| async {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench_ew_scale"),
                });
                ew_op.dispatch_scale(&mut encoder, &a_buf, &c_buf, 0.5f32, numel).unwrap();
                queue.submit(std::iter::once(encoder.finish()));
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            });
        });

        group.bench_function("relu", |b| {
            b.to_async(&rt).iter(|| async {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bench_ew_relu"),
                });
                ew_op.dispatch_relu(&mut encoder, &a_buf, &c_buf, numel).unwrap();
                queue.submit(std::iter::once(encoder.finish()));
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            });
        });

        group.finish();
    });
}

criterion_group!(benches, bench_matmul, bench_rmsnorm, bench_softmax, bench_elementwise);
criterion_main!(benches);
