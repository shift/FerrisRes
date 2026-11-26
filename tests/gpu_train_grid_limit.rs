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

use std::sync::Arc;

use ferrisres::compute::pipeline::WgpuCompute;
use ferrisres::training::f32_trainer::F32Config;
use ferrisres::training::gpu_trainer::GpuModel;

#[test]
fn no_dispatch_exceeds_the_spec_limit() {
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
