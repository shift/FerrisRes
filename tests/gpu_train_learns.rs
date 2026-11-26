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

/// The same gate with **four windows per step**, which is what the real trainer
/// does (`windows = batch_size`). The single-window test above cannot catch a
/// step that reads the last window's gradient instead of the accumulated mean:
/// with one window the two are identical.
///
/// With the defect this diverges; the effective learning rate is `LR * 4`.
#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn batched_training_still_learns() {
    const WINDOWS: u32 = 4;
    const BATCH_STEPS: usize = 100;

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
    // Four distinct windows, so the mean of their gradients differs from the
    // last one.
    let windows: Vec<Vec<u32>> = (0..WINDOWS)
        .map(|w| {
            (0..cfg.max_seq)
                .map(|i| ((i * 7 + 3 + w as usize * 5) % cfg.vocab) as u32)
                .collect()
        })
        .collect();

    let mut model = GpuModel::new(Arc::clone(&device), Arc::clone(&queue), cfg, 42).expect("model");

    let mut initial = f32::NAN;
    let mut final_loss = f32::NAN;
    for step in 0..BATCH_STEPS {
        let mut step_loss = 0.0f32;
        for window in &windows {
            let (logits, loss_node) = model.forward(window).expect("forward");
            let logits = model
                .read_logits(logits, cfg.max_seq * cfg.vocab)
                .expect("read logits");
            step_loss += F32Model::cross_entropy(&logits, window, cfg.vocab).0;
            model.backward(loss_node).expect("backward");
            model.accumulate_grads().expect("accumulate");
        }
        let mean = step_loss / WINDOWS as f32;
        if step == 0 {
            initial = mean;
        }
        final_loss = mean;
        if step % 20 == 0 || step + 1 == BATCH_STEPS {
            eprintln!("BATCH step {step}/{BATCH_STEPS} mean_loss={mean:.6}");
        }
        model.step_accumulated(LR, WINDOWS).expect("step");
        model.reset_accumulators().expect("reset");
    }

    eprintln!("BATCH initial={initial:.6} final={final_loss:.6}");
    assert!(
        final_loss < initial * 0.8,
        "batched training did not learn: initial={initial:.6} final={final_loss:.6} \
         (a step that ignores the accumulated mean behaves like lr * {WINDOWS})"
    );
}
