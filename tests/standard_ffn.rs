//! Full-layer FFN reference checks with attention deliberately zeroed.
use ferrisres::compute::{GpuBuffer, WgpuCompute};
use ferrisres::inference::kv_cache::LayerKVCache;
use ferrisres::model::standard_transformer::{
    GatedActivation, StandardTransformerConfig, StandardTransformerLayer,
};
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

fn check_layer(prefill: bool, activation: GatedActivation) {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    let mut config = StandardTransformerConfig::new(4, 2, 1);
    config.intermediate_dim = 8;
    config.ffn_activation = activation;
    let layer = StandardTransformerLayer::new(device.clone(), queue.clone(), &config).unwrap();
    // Native Linear layout is [in, out]; non-square, non-symmetric fixtures.
    let gate: Vec<f32> = (0..32).map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0).collect();
    let up: Vec<f32> = (0..32).map(|i| ((i * 3 % 11) as f32 - 5.0) / 7.0).collect();
    let down: Vec<f32> = (0..32).map(|i| ((i * 5 % 7) as f32 - 3.0) / 9.0).collect();
    layer.ff_gate().set_weight(&queue, &gate);
    layer.ff_up().set_weight(&queue, &up);
    layer.ff_down().set_weight(&queue, &down);
    let tokens: Vec<f32> = vec![
        -2.0, 0.5, 1.0, -0.25, 0.0, -1.0, 2.5, 0.75, 1.0, 0.0, -0.5, -3.0,
    ];
    let reference: Vec<f32> = tokens
        .chunks_exact(4)
        .flat_map(|x| {
            let rms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-5).sqrt();
            let gated: Vec<f32> = (0..8)
                .map(|j| {
                    let g: f32 = (0..4).map(|i| x[i] / rms * gate[i * 8 + j]).sum();
                    let u: f32 = (0..4).map(|i| x[i] / rms * up[i * 8 + j]).sum();
                    let activated = match activation {
                        GatedActivation::Silu => g / (1.0 + (-g).exp()),
                        GatedActivation::GeluTanh => {
                            0.5 * g
                                * (1.0
                                    + (std::f32::consts::FRAC_2_PI.sqrt()
                                        * (g + 0.044715 * g.powi(3)))
                                    .tanh())
                        }
                    };
                    activated * u
                })
                .collect();
            (0..4)
                .map(|j| x[j] + (0..8).map(|i| gated[i] * down[i * 4 + j]).sum::<f32>())
                .collect::<Vec<_>>()
        })
        .collect();
    let cache = LayerKVCache::new(device.clone(), queue.clone(), 8, 2, 2).unwrap();
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
    for (i, (&a, &b)) in actual.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() < 2e-4, "index {i}: GPU={a}, reference={b}");
    }
}

#[test]
fn prefill_matches_swiglu_layer_reference() {
    check_layer(true, GatedActivation::Silu);
}
#[test]
fn repeated_decode_matches_swiglu_layer_reference() {
    check_layer(false, GatedActivation::Silu);
}
#[test]
fn prefill_matches_gelu_tanh_gated_layer_reference() {
    check_layer(true, GatedActivation::GeluTanh);
}
#[test]
fn repeated_decode_matches_gelu_tanh_gated_layer_reference() {
    check_layer(false, GatedActivation::GeluTanh);
}
#[test]
fn constructors_default_to_swiglu() {
    for config in [
        StandardTransformerConfig::new(4, 2, 1),
        StandardTransformerConfig::llama_7b(),
        StandardTransformerConfig::mistral_7b(),
        StandardTransformerConfig::from_inferred(4, 2, 1, 8),
    ] {
        assert_eq!(config.ffn_activation, GatedActivation::Silu);
    }
}
