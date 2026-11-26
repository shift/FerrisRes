//! Phase-B gate: tiny-model parity between the GPU graph and the CPU oracle.
//!
//! Ignored by default (needs a GPU). Run manually:
//!
//! ```text
//! VK_ICD_FILENAMES=<mesa>/share/vulkan/icd.d/intel_icd.x86_64.json \
//! LD_LIBRARY_PATH=<vulkan-loader>/lib:<mesa>/lib \
//!   cargo test --test gpu_train_parity -- --ignored --nocapture
//! ```
//!
//! Tolerances, and why they are not zero:
//! * logits `1e-3` absolute — the GPU sums in a different order (tiled matmul,
//!   workgroup reductions) than the CPU's row loop, so f32 rounding differs.
//! * gradients `1e-2` relative — the same summation-order effect, amplified by
//!   the chain rule.
//! * post-step parameters `1e-3` absolute — one Adam step with the same
//!   gradients and hyper-parameters.

use std::sync::Arc;

use ferrisres::compute::pipeline::WgpuCompute;
use ferrisres::training::f32_trainer::{Adam, F32Config, F32Model};
use ferrisres::training::gpu_trainer::GpuModel;

fn config() -> F32Config {
    F32Config {
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
    }
}

fn tokens(vocab: usize, seq: usize, seed: u32) -> Vec<u32> {
    let mut state = seed;
    (0..seq)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) % vocab as u32
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let mut worst = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let denom = w.abs().max(1e-6);
        worst = worst.max((g - w).abs() / denom);
    }
    worst
}

fn cpu_loss(model: &F32Model, tokens: &[u32]) -> f32 {
    let (logits, _) = model.forward(tokens);
    F32Model::cross_entropy(&logits, tokens, model.config.vocab).0
}

fn device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let compute = pollster::block_on(WgpuCompute::new()).expect("GPU device");
    (
        Arc::new(compute.device().clone()),
        Arc::new(compute.queue().clone()),
    )
}

#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn tiny_model_matches_cpu_forward_backward_and_step() {
    let cfg = config();
    let seed = 1234u64;
    let toks = tokens(cfg.vocab, cfg.max_seq, 99);
    let (dev, queue) = device();

    let cpu = F32Model::new_random(cfg, seed);
    let mut gpu = GpuModel::from_f32_model(Arc::clone(&dev), Arc::clone(&queue), &cpu)
        .expect("upload model");

    // ---- forward ----
    let (cpu_logits, cpu_cache) = cpu.forward(&toks);
    let (logits_node, loss_node) = gpu.forward(&toks).expect("gpu forward");
    let gpu_logits = gpu
        .read_logits(logits_node, cfg.max_seq * cfg.vocab)
        .expect("read logits");
    let logit_err = max_abs(&gpu_logits, &cpu_logits);
    eprintln!("PARITY logits max_abs={logit_err:.3e}");
    assert!(logit_err < 1e-3, "logits differ by {logit_err}");

    let (cpu_ce, d_logits) = F32Model::cross_entropy(&cpu_logits, &toks, cfg.vocab);
    let gpu_loss = F32Model::cross_entropy(&gpu_logits, &toks, cfg.vocab).0;
    eprintln!("PARITY loss cpu={cpu_ce:.6} gpu={gpu_loss:.6}");

    // ---- backward ----
    let cpu_grads = cpu.backward(&cpu_cache, &d_logits);
    gpu.backward(loss_node).expect("gpu backward");

    let cases: [(&str, &[f32]); 4] = [
        ("lm_head", &cpu_grads.lm_head),
        ("layer0.q", &cpu_grads.layers[0].q),
        ("layer1.gate", &cpu_grads.layers[1].gate),
        ("layer0.layer_scalar", std::slice::from_ref(&cpu_grads.layers[0].layer_scalar)),
    ];
    for (name, want) in cases {
        let got = gpu.read_grad(name).expect("read grad");
        let err = rel_err(&got, want);
        let abs = max_abs(&got, want);
        eprintln!("PARITY grad {name} rel={err:.3e} abs={abs:.3e}");
        assert!(err < 1e-2, "gradient {name} relative error {err}");
    }

    // ---- finite difference on lm_head and one attention weight ----
    // Central difference on the CPU oracle gives an independent gradient; the
    // GPU value must agree with it within the same 1e-2 relative bound.
    let h = 1e-2f32;
    for (tensor_name, gpu_name, index) in [
        ("lm_head", "lm_head", 17usize),
        ("layer0.q", "layer0.q", 23usize),
    ] {
        let tensor_index = cpu
            .tensors()
            .iter()
            .position(|(n, _)| n == tensor_name)
            .expect("tensor");
        let original = cpu.get_tensor_elem(tensor_index, index);
        let mut plus = cpu.clone();
        plus.set_tensor_elem(tensor_index, index, original + h);
        let mut minus = cpu.clone();
        minus.set_tensor_elem(tensor_index, index, original - h);
        let fd = (cpu_loss(&plus, &toks) - cpu_loss(&minus, &toks)) / (2.0 * h);
        let analytic = cpu_grads.tensors()[tensor_index].1[index];
        let gpu_value = gpu.read_grad(gpu_name).expect("read grad")[index];
        let fd_vs_analytic = (analytic - fd).abs() / fd.abs().max(1e-6);
        let gpu_vs_fd = (gpu_value - fd).abs() / fd.abs().max(1e-6);
        eprintln!(
            "PARITY fd {tensor_name}[{index}] fd={fd:.6e} cpu={analytic:.6e} gpu={gpu_value:.6e} \
             cpu_vs_fd={fd_vs_analytic:.3e} gpu_vs_fd={gpu_vs_fd:.3e}"
        );
        assert!(fd_vs_analytic < 1e-2, "CPU analytic vs FD: {fd_vs_analytic}");
        assert!(gpu_vs_fd < 1e-2, "GPU analytic vs FD: {gpu_vs_fd}");
    }

    // ---- one Adam step with the same lr ----
    let lr = 0.01f32;
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut cpu_model = cpu.clone();
    let mut adam = Adam::new(&cpu_model);
    adam.step(&mut cpu_model, &cpu_grads, lr, b1, b2, eps);

    gpu.accumulate_grads().expect("accumulate");
    gpu.step_accumulated(lr, 1).expect("adam step");

    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    for (name, want) in cpu_model.tensors() {
        let got = gpu.read_param(&name).expect("read param");
        let err = max_abs(&got, want);
        if err > worst {
            worst = err;
            worst_name = name.clone();
        }
    }
    eprintln!("PARITY post-step max_abs={worst:.3e} at {worst_name}");
    assert!(worst < 1e-3, "parameter {worst_name} differs by {worst}");
}
