use ferrisres::compute::{GpuBuffer, WgpuCompute};
use ferrisres::model::dispatcher::AnyModel;
use ferrisres::model::standard_transformer::{StandardTransformerConfig, StandardTransformerModel};
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
fn check(learned: bool) {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    // Zero layers isolate the final norm; it must still exist and validate config.
    let mut config = StandardTransformerConfig::new(4, 1, 0);
    config.norm_eps = 0.25;
    let standard = StandardTransformerModel::new(device.clone(), queue.clone(), config).unwrap();
    let scale = if learned {
        [0.5, -2.0, 0.0, 1.5]
    } else {
        [1.0; 4]
    };
    if learned {
        standard.set_final_norm_weight(&scale).unwrap();
        assert!(standard.set_final_norm_weight(&[1.0; 3]).is_err());
        assert!(standard.set_final_norm_weight(&[f32::NAN; 4]).is_err());
    }
    let model = AnyModel::Standard(standard);
    let x = [3.0f32, 4.0, 0.0, -1.0, 0.001, -0.002, 0.0, 0.003];
    for rows in [1, 2] {
        let values = &x[..rows * 4];
        let input = GpuBuffer::new(&device, values.len() * 4, None).unwrap();
        queue.write_buffer(input.buffer(), 0, bytemuck::cast_slice(values));
        let mut encoder = device.create_command_encoder(&Default::default());
        let output = model
            .finalize_hidden_states(&mut encoder, input, rows as u32)
            .unwrap();
        queue.submit([encoder.finish()]);
        let expected: Vec<f32> = values
            .chunks_exact(4)
            .flat_map(|row| {
                let rms = (row.iter().map(|v| v * v).sum::<f32>() / 4.0 + 0.25).sqrt();
                (0..4).map(|i| row[i] / rms * scale[i]).collect::<Vec<_>>()
            })
            .collect();
        for (a, b) in read(&device, &queue, &output).iter().zip(expected) {
            assert!((a - b).abs() < 1e-5, "actual={a}, expected={b}");
        }
    }
}
#[test]
fn block_model_finalization_remains_identity() {
    use ferrisres::model::config::BlockAttnResConfig;
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    let mut config = BlockAttnResConfig::new(4);
    config.num_blocks = 0;
    config.num_layers = 0;
    let model = AnyModel::new_block_attn_res(device.clone(), queue.clone(), config, 4).unwrap();
    let values = [3.0f32, -4.0, 0.0, 0.001];
    let input = GpuBuffer::new(&device, 16, None).unwrap();
    queue.write_buffer(input.buffer(), 0, bytemuck::cast_slice(&values));
    let mut encoder = device.create_command_encoder(&Default::default());
    let output = model
        .finalize_hidden_states(&mut encoder, input, 1)
        .unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(read(&device, &queue, &output), values);
}
#[test]
fn invalid_rows_are_rejected_before_output_allocation() {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    let model = AnyModel::new_standard(
        device.clone(),
        queue.clone(),
        StandardTransformerConfig::new(4, 1, 0),
    )
    .unwrap();
    for rows in [0, 2, u32::MAX] {
        let input = GpuBuffer::new(&device, 16, None).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(model
            .finalize_hidden_states(&mut encoder, input, rows)
            .is_err());
    }
}
#[test]
fn final_norm_is_applied_with_unit_scales() {
    check(false);
}
#[test]
fn learned_final_norm_survives_invalid_uploads() {
    check(true);
}
#[test]
fn invalid_final_norm_epsilon_is_rejected_even_without_layers() {
    let compute = pollster::block_on(WgpuCompute::new()).unwrap();
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());
    for eps in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut config = StandardTransformerConfig::new(4, 1, 0);
        config.norm_eps = eps;
        assert!(StandardTransformerModel::new(device.clone(), queue.clone(), config).is_err());
    }
}
