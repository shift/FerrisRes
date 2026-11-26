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

/// Relative L2 error. Per-element relative error is meaningless for gradient
/// entries that are legitimately near zero (finite-difference noise there is
/// larger than the value), so the reported number is `||g - w|| / ||w||`.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        num += ((*g - *w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
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
    // Localise a forward mismatch: compare every cached CPU activation with
    // the GPU node the same op produced.
    let h = cfg.hidden;
    let q_dim = cfg.q_dim();
    let kv_dim = cfg.kv_dim();
    let lc = &cpu_cache.layers[0];
    let checks: [(&str, &[f32], usize); 15] = [
        ("embed_scaled", &cpu_cache.x_embed, cfg.max_seq * h),
        ("L0.h_attn", &lc.h_attn, cfg.max_seq * h),
        ("L0.q_proj", &lc.q_proj, cfg.max_seq * q_dim),
        ("L0.v_norm", &lc.v, cfg.max_seq * kv_dim),
        ("L0.q_rope", &lc.q, cfg.max_seq * q_dim),
        ("L0.k_rope", &lc.k, cfg.max_seq * kv_dim),
        ("L0.attn", &lc.attn_concat, cfg.max_seq * q_dim),
        ("L0.attn_out", &lc.attn_out, cfg.max_seq * h),
        ("L0.post_attn", &lc.post_attn_normed, cfg.max_seq * h),
        ("L0.x_mid", &lc.x_mid, cfg.max_seq * h),
        ("L0.h_ffn", &lc.h_ffn, cfg.max_seq * h),
        ("L0.gate", &lc.gate_pre, cfg.max_seq * cfg.intermediate),
        ("L0.act", &lc.act, cfg.max_seq * cfg.intermediate),
        ("L0.ffn_out", &lc.ffn_out, cfg.max_seq * h),
        ("L0.x_out", &lc.x_out, cfg.max_seq * h),
    ];
    for (name, want, len) in checks {
        let id = gpu.node_id(name).expect("trace node");
        let got = gpu.read_logits(id, len).expect("read node");
        let err = max_abs(&got, want);
        let at = got
            .iter()
            .zip(want.iter())
            .position(|(a, b)| (a - b).abs() == err)
            .unwrap_or(0);
        eprintln!(
            "TRACE {name} max_abs={err:.3e} at {at} gpu={:.6} cpu={:.6}",
            got[at], want[at]
        );
    }
    let hf = gpu.node_id("h_final").expect("h_final");
    eprintln!(
        "TRACE h_final max_abs={:.3e}",
        max_abs(&gpu.read_logits(hf, cfg.max_seq * h).unwrap(), &cpu_cache.h_final)
    );

    let logit_err = max_abs(&gpu_logits, &cpu_logits);
    eprintln!("PARITY logits max_abs={logit_err:.3e}");
    assert!(logit_err < 1e-3, "logits differ by {logit_err}");

    let (cpu_ce, d_logits) = F32Model::cross_entropy(&cpu_logits, &toks, cfg.vocab);
    let gpu_loss = F32Model::cross_entropy(&gpu_logits, &toks, cfg.vocab).0;
    eprintln!("PARITY loss cpu={cpu_ce:.6} gpu={gpu_loss:.6}");

    // ---- backward ----
    let cpu_grads = cpu.backward(&cpu_cache, &d_logits);
    gpu.backward(loss_node).expect("gpu backward");

    if std::env::var("PARITY_ALL").is_ok() {
        for (name, want) in cpu_grads.tensors() {
            let got = gpu.read_grad(&name).expect("read grad");
            eprintln!(
                "ALLGRAD {name} rel={:.3e} abs={:.3e} gpu_max={:.3e} cpu_max={:.3e}",
                rel_err(&got, want),
                max_abs(&got, want),
                got.iter().fold(0.0f32, |m, v| m.max(v.abs())),
                want.iter().fold(0.0f32, |m, v| m.max(v.abs()))
            );
        }
    }
    let cases: [(&str, &[f32]); 4] = [
        ("lm_head", &cpu_grads.lm_head),
        ("layer0.q", &cpu_grads.layers[0].q),
        ("layer1.gate", &cpu_grads.layers[1].gate),
        ("layer0.layer_scalar", std::slice::from_ref(&cpu_grads.layers[0].layer_scalar)),
    ];
    for (name, want) in cases {
        let got = gpu.read_grad(name).expect("read grad");
        eprintln!(
            "TRACE grad {name} gpu[0..4]={:?} cpu[0..4]={:?} gpu_max={:.4e} cpu_max={:.4e}",
            &got[..4.min(got.len())],
            &want[..4.min(want.len())],
            got.iter().fold(0.0f32, |m, v| m.max(v.abs())),
            want.iter().fold(0.0f32, |m, v| m.max(v.abs())),
        );
        let err = rel_err(&got, want);
        let abs = max_abs(&got, want);
        eprintln!("PARITY grad {name} rel={err:.3e} abs={abs:.3e}");
        assert!(err < 1e-2, "gradient {name} relative L2 error {err}");
        assert!(abs < 1e-3, "gradient {name} absolute error {abs}");
    }

    // ---- finite difference on lm_head and one attention weight ----
    // Central difference on the CPU oracle gives an independent gradient; the
    // GPU value must agree with it within the same 1e-2 relative bound.
    let h = 1e-2f32;
    for (tensor_name, gpu_name) in [("lm_head", "lm_head"), ("layer0.q", "layer0.q")] {
        let tensor_index = cpu
            .tensors()
            .iter()
            .position(|(n, _)| n == tensor_name)
            .expect("tensor");
        let analytic_slice = cpu_grads.tensors()[tensor_index].1;
        let (index, _) = analytic_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs().partial_cmp(&b.abs()).unwrap()
            })
            .unwrap();
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
    let mut scalar_worst = 0.0f32;
    for (name, want) in cpu_model.tensors() {
        let got = gpu.read_param(&name).expect("read param");
        let err = max_abs(&got, want);
        if name.ends_with("layer_scalar") {
            // `layer_scalar`'s gradient is a cancellation-dominated sum of
            // order 1e-7, and Adam divides by |g| + eps with eps = 1e-8. A
            // 3e-8 difference in that sum (f32 summation order, not a wrong
            // formula) moves this one parameter by ~1.2e-3, so it is reported
            // and bounded separately instead of being hidden in the global
            // bound.
            scalar_worst = scalar_worst.max(err);
            continue;
        }
        if err > worst {
            worst = err;
            worst_name = name.clone();
        }
    }
    eprintln!("PARITY post-step max_abs={worst:.3e} at {worst_name}");
    eprintln!("PARITY post-step layer_scalar max_abs={scalar_worst:.3e}");
    assert!(worst < 1e-3, "parameter {worst_name} differs by {worst}");
    assert!(
        scalar_worst < 2e-3,
        "layer_scalar differs by {scalar_worst} (conditioned by Adam eps)"
    );
}
