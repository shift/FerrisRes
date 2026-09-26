//! Learned layer norms through nonzero-attention prefill and decode.
use ferrisres::compute::{GpuBuffer, WgpuCompute};
use ferrisres::inference::kv_cache::LayerKVCache;
use ferrisres::model::standard_transformer::{StandardTransformerConfig, StandardTransformerLayer};
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
fn norm(x: &[f32], scale: &[f32], epsilon: f32) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + epsilon).sqrt();
    x.iter().zip(scale).map(|(v, s)| v / rms * s).collect()
}
fn check(prefill: bool, learned: bool) {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    let mut config = StandardTransformerConfig::new(4, 1, 1);
    config.intermediate_dim = 4;
    config.norm_eps = 0.01;
    let layer = StandardTransformerLayer::new(device.clone(), queue.clone(), &config).unwrap();
    let a = if learned {
        [0.5, -1.0, 0.0, 2.0]
    } else {
        [1.0; 4]
    };
    let f = if learned {
        [-0.5, 1.5, 0.0, 0.75]
    } else {
        [1.0; 4]
    };
    if learned {
        layer.set_norm_weights(&a, &f).unwrap();
    }
    let identity: Vec<f32> = (0..16)
        .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
        .collect();
    // Zero Q/K yield uniform causal attention over nonzero V projections.
    for linear in [
        layer.v_proj(),
        layer.out_proj(),
        layer.ff_gate(),
        layer.ff_up(),
        layer.ff_down(),
    ] {
        linear.set_weight(&queue, &identity);
    }
    let tokens: Vec<f32> = vec![
        0.001, -0.002, 0.0, 0.003, 1.0, -2.0, 3.0, 0.5, -0.25, 0.75, 0.5, -1.0,
    ];
    let mut reference = Vec::new();
    let mut prefix = [0.0; 4];
    for (t, x) in tokens.chunks_exact(4).enumerate() {
        let n = norm(x, &a, config.norm_eps);
        for i in 0..4 {
            prefix[i] += n[i];
        }
        let residual: Vec<f32> = (0..4).map(|i| x[i] + prefix[i] / (t + 1) as f32).collect();
        let ff = norm(&residual, &f, config.norm_eps);
        reference.extend((0..4).map(|i| residual[i] + ff[i] / (1.0 + (-ff[i]).exp()) * ff[i]));
    }
    let cache = LayerKVCache::new(device.clone(), queue.clone(), 4, 1, 4).unwrap();
    let mut actual = Vec::new();
    for x in tokens.chunks(if prefill { 12 } else { 4 }) {
        let input = GpuBuffer::new(&device, x.len() * 4, None).unwrap();
        queue.write_buffer(input.buffer(), 0, bytemuck::cast_slice(x));
        let mut encoder = device.create_command_encoder(&Default::default());
        let output = if prefill {
            layer
                .forward_prefill(&mut encoder, &input, &cache, 3)
                .unwrap()
        } else {
            layer
                .forward_decode_token(&mut encoder, &input, &cache)
                .unwrap()
        };
        queue.submit([encoder.finish()]);
        actual.extend(read(&device, &queue, &output));
    }
    for (i, (a, b)) in actual.iter().zip(reference).enumerate() {
        assert!((a - b).abs() < 3e-4, "index{i}: actual{a} expected{b}");
    }
}
#[test]
fn prefill_honors_epsilon() {
    check(true, false);
}
#[test]
fn decode_honors_epsilon() {
    check(false, false);
}
#[test]
fn prefill_honors_learned_norms() {
    check(true, true);
}
#[test]
fn decode_honors_learned_norms() {
    check(false, true);
}

#[test]
fn invalid_scale_upload_does_not_partially_modify_weights() {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    let layer = StandardTransformerLayer::new(
        device.clone(),
        queue.clone(),
        &StandardTransformerConfig::new(4, 1, 1),
    )
    .unwrap();
    assert_eq!(read(&device, &queue, layer.norm_weights().0), vec![1.0; 4]);
    let a = [0.5, -1.0, 0.0, 2.0];
    let f = [-0.5, 1.5, 0.0, 0.75];
    layer.set_norm_weights(&a, &f).unwrap();
    for (bad_a, bad_f) in [
        (vec![1.0; 3], vec![1.0; 4]),
        (vec![1.0; 4], vec![1.0; 3]),
        (vec![1.0; 4], vec![f32::NAN; 4]),
        (vec![f32::INFINITY; 4], vec![1.0; 4]),
    ] {
        assert!(layer.set_norm_weights(&bad_a, &bad_f).is_err());
        assert_eq!(read(&device, &queue, layer.norm_weights().0), a);
        assert_eq!(read(&device, &queue, layer.norm_weights().1), f);
    }
}

#[test]
fn invalid_epsilon_is_rejected_before_layer_construction() {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::from_bits(1)] {
        let mut config = StandardTransformerConfig::new(4, 1, 1);
        config.norm_eps = eps;
        assert!(StandardTransformerLayer::new(device.clone(), queue.clone(), &config).is_err());
    }
}

#[test]
fn weighted_kernel_keeps_distinct_epsilons_and_legacy_default() {
    use ferrisres::compute::kernels::gpu_transformer::GpuTransformerPipeline;
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = compute.device();
    let queue = compute.queue();
    let op = GpuTransformerPipeline::new(device).unwrap();
    let x = [0.0f32, 0.001, -0.002, 0.003, 2.0, -3.0, 0.0, 1.0];
    let w = [0.0f32, -1.0, 0.5, 2.0];
    let input = GpuBuffer::new(device, 32, None).unwrap();
    let weights = GpuBuffer::new(device, 16, None).unwrap();
    queue.write_buffer(input.buffer(), 0, bytemuck::cast_slice(&x));
    queue.write_buffer(weights.buffer(), 0, bytemuck::cast_slice(&w));
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut outputs = Vec::new();
    for eps in [0.01, 0.5, 1e-6] {
        let output = GpuBuffer::new(device, 32, None).unwrap();
        if eps == 1e-6 {
            op.dispatch_rmsnorm(device, queue, &mut encoder, &input, &output, &weights, 2, 4)
                .unwrap();
        } else {
            op.dispatch_rmsnorm_with_epsilon(
                device,
                queue,
                &mut encoder,
                &input,
                &output,
                &weights,
                2,
                4,
                eps,
            )
            .unwrap();
        }
        outputs.push((eps, output));
    }
    assert!(op
        .dispatch_rmsnorm_with_epsilon(
            device,
            queue,
            &mut encoder,
            &input,
            &outputs[0].1,
            &weights,
            0,
            4,
            0.01
        )
        .is_err());
    assert!(op
        .dispatch_rmsnorm_with_epsilon(
            device,
            queue,
            &mut encoder,
            &input,
            &outputs[0].1,
            &weights,
            3,
            4,
            0.01
        )
        .is_err());
    queue.submit([encoder.finish()]);
    for (eps, output) in outputs {
        let expected: Vec<f32> = x
            .chunks_exact(4)
            .flat_map(|row| norm(row, &w, eps))
            .collect();
        for (a, b) in read(device, queue, &output).iter().zip(expected) {
            assert!((a - b).abs() < 2e-5);
        }
    }
}
