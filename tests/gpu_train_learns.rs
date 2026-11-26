//! Phase-C gate: the GPU training loop learns on a fixed sequence.
//!
//! Ignored by default: it needs a GPU and runs 200 steps (minutes on an
//! integrated GPU). Run manually:
//!
//! ```text
//! VK_ICD_FILENAMES=<mesa>/share/vulkan/icd.d/intel_icd.x86_64.json \
//! LD_LIBRARY_PATH=<vulkan-loader>/lib:<mesa>/lib \
//!   cargo test --test gpu_train_learns -- --ignored --nocapture
//! ```

use std::sync::Arc;

use ferrisres::compute::pipeline::WgpuCompute;
use ferrisres::training::f32_trainer::{F32Config, F32Model};
use ferrisres::training::gpu_trainer::GpuModel;

const STEPS: usize = 200;
const LR: f32 = 0.01;

#[test]
#[ignore = "needs a Vulkan/WGPU device and runs 200 steps; run with VK_ICD_FILENAMES set"]
fn loss_falls_below_half_and_parameters_change() {
    let compute = pollster::block_on(WgpuCompute::new()).expect("GPU device");
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let cfg = F32Config {
        hidden: 16,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 8,
        intermediate: 32,
        vocab: 32,
        max_seq: 16,
        rope_theta: 10_000.0,
        eps: 1e-6,
    };
    let tokens: Vec<u32> = (0..cfg.max_seq).map(|i| (i * 7 + 3) as u32 % cfg.vocab as u32).collect();

    let mut model = GpuModel::new(Arc::clone(&device), Arc::clone(&queue), cfg, 42).expect("model");
    let before = model.read_param("embed").expect("embed");

    let mut initial = f32::NAN;
    let mut final_loss = f32::NAN;
    for step in 0..STEPS {
        let (logits, loss_node) = model.forward(&tokens).expect("forward");
        let logits = model
            .read_logits(logits, cfg.max_seq * cfg.vocab)
            .expect("read logits");
        let loss = F32Model::cross_entropy(&logits, &tokens, cfg.vocab).0;
        if step == 0 {
            initial = loss;
        }
        final_loss = loss;
        if step % 20 == 0 || step + 1 == STEPS {
            eprintln!("LEARN step {step}/{STEPS} loss={loss:.6}");
        }
        model.backward(loss_node).expect("backward");
        model.accumulate_grads().expect("accumulate");
        model.step_accumulated(LR, 1).expect("step");
        model.reset_accumulators().expect("reset");
    }

    let after = model.read_param("embed").expect("embed");
    let changed = before
        .iter()
        .zip(after.iter())
        .any(|(a, b)| a != b);

    eprintln!(
        "LEARN initial={initial:.6} final={final_loss:.6} ratio={:.3} params_changed={changed}",
        final_loss / initial
    );
    assert!(initial.is_finite() && final_loss.is_finite());
    assert!(final_loss < 0.5 * initial, "loss {final_loss} not < 0.5 * {initial}");
    assert!(changed, "no parameter changed");
}
