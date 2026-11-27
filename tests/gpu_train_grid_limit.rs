//! Regression gate for the WebGPU dispatch limit.
//!
//! A vocabulary of 32k at hidden 768 makes the lm_head/embedding gradient
//! 25,165,824 elements. One workgroup per 256 elements in a single dispatch
//! dimension is 98,304 groups, over the spec cap of 65,535. This test runs one
//! forward + backward on the smallest model that still exceeds the limit and
//! fails on any validation error, which is what an oversized dispatch becomes.
//!
//! It is not `#[ignore]`: on a host without an adapter it prints
//! `skipped: no adapter` and returns. A silent pass would hide the regression
//! on CI, so the skip is always printed.

use std::sync::{Arc, Mutex};

use ferrisres::compute::pipeline::WgpuCompute;
use ferrisres::training::f32_trainer::F32Config;
use ferrisres::training::gpu_trainer::GpuModel;

/// Both tests build a full GPU model. Running them in parallel in one binary
/// exhausts a small adapter's memory and loses the device, not because either
/// dispatch is wrong. They share the adapter, so they run one at a time.
static GPU_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn no_dispatch_exceeds_the_spec_limit() {
    let _gpu = GPU_LOCK.lock().unwrap();
    let compute = match pollster::block_on(WgpuCompute::new()) {
        Ok(compute) => compute,
        Err(e) => {
            eprintln!("skipped: no adapter ({e})");
            return;
        }
    };
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    // 32,768 x 768 = 25,165,824 elements, the smallest head that overflows
    // 65,535 x 256 workgroups in one dimension.
    let config = F32Config {
        hidden: 768,
        num_layers: 1,
        num_heads: 12,
        num_kv_heads: 4,
        head_dim: 64,
        intermediate: 2048,
        vocab: 32_768,
        max_seq: 8,
        rope_theta: 10_000.0,
        eps: 1e-6,
    };
    let mut model =
        GpuModel::new(Arc::clone(&device), Arc::clone(&queue), config, 7).expect("upload model");

    let tokens: Vec<u32> = (0..8).map(|i| 100 + i as u32).collect();

    // Capture any validation error instead of letting the default handler
    // panic, so the failure names the dispatch rather than aborting the test
    // harness. `TrainOps::dispatch` additionally asserts its own grid, which
    // names the kernel and element count.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let (_logits, loss) = model.forward(&tokens).expect("forward");
    model.backward(loss).expect("backward");
    // Parameter gradients are 25.2M elements too. Accumulation and the Adam
    // step run the elementwise 2D grid over them, so they are part of the same
    // regression rather than a separate concern.
    model.accumulate_grads().expect("accumulate gradients");
    model.step_accumulated(0.001, 1).expect("optimizer step");
    let error = pollster::block_on(scope.pop());

    assert!(
        error.is_none(),
        "a dispatch exceeded the WebGPU 65,535-workgroup limit: {error:?}"
    );
}

/// The loss gradient at a realistic training shape.
///
/// `max_seq: 8` cannot overflow: `8 x 32_768 / 256` is 1,024 groups. The head
/// gradient was already covered, but the loss gradient is dispatched from
/// `BackwardPass` with its own flat count — the path that the original 2D fix
/// missed. `batch = seq - 1`, so 513 tokens gives 512 x 32_768 / 256 = 65,536
/// groups, the smallest window that crosses the 65,535 cap. (1,024 tokens also
/// overflows but needs ~1.5 GB and loses the device on a small adapter.)
#[test]
fn the_loss_gradient_at_a_real_sequence_fits_the_dispatch_limit() {
    let _gpu = GPU_LOCK.lock().unwrap();
    let compute = match pollster::block_on(WgpuCompute::new()) {
        Ok(compute) => compute,
        Err(e) => {
            eprintln!("skipped: no adapter ({e})");
            return;
        }
    };
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let config = F32Config {
        hidden: 768,
        num_layers: 1,
        num_heads: 12,
        num_kv_heads: 4,
        head_dim: 64,
        intermediate: 2048,
        vocab: 32_768,
        max_seq: 513,
        rope_theta: 10_000.0,
        eps: 1e-6,
    };
    let mut model = GpuModel::new(Arc::clone(&device), Arc::clone(&queue), config, 7)
        .expect("upload model");
    let tokens: Vec<u32> = (0..513).map(|i| 100 + (i as u32 % 1_000)).collect();

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let (_logits, loss) = model.forward(&tokens).expect("forward");
    model.backward(loss).expect("backward");
    let error = pollster::block_on(scope.pop());
    assert!(
        error.is_none(),
        "the loss-gradient dispatch exceeded the WebGPU 65,535-workgroup limit: {error:?}"
    );
}
