//! CPU-reference checks for the GPU ops the training loop added.
//!
//! Each test is ignored by default because it needs a Vulkan/WGPU device:
//!
//! ```text
//! VK_ICD_FILENAMES=<mesa>/share/vulkan/icd.d/intel_icd.x86_64.json \
//! LD_LIBRARY_PATH=<vulkan-loader>/lib:<mesa>/lib \
//!   cargo test --test gpu_train_ops -- --ignored --nocapture
//! ```
//!
//! The tolerance is absolute `1e-4` on every element. That is the phase-A gate.

use std::sync::Arc;

use ferrisres::autodiff::gpu_ops::TrainOps;
use ferrisres::compute::pipeline::WgpuCompute;
use ferrisres::compute::GpuBuffer;

const TOL: f32 = 1e-4;

fn setup() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    let compute = pollster::block_on(WgpuCompute::new()).expect("GPU device");
    (
        Arc::new(compute.device().clone()),
        Arc::new(compute.queue().clone()),
    )
}

fn buf(device: &wgpu::Device, queue: &wgpu::Queue, data: &[f32]) -> GpuBuffer {
    GpuBuffer::new_device_local(device, queue, bytemuck::cast_slice(data), Some("test"))
        .expect("buffer")
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, b: &GpuBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    b.read(device, queue, &mut bytes).expect("read");
    bytemuck::cast_slice(&bytes).to_vec()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Deterministic values in `[-1, 1)`; no RNG dependency.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn rmsnorm_ref(x: &[f32], eps: f32, rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * dim];
    for r in 0..rows {
        let ss: f32 = (0..dim).map(|i| x[r * dim + i] * x[r * dim + i]).sum();
        let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
        for i in 0..dim {
            out[r * dim + i] = x[r * dim + i] * inv;
        }
    }
    out
}

fn rmsnorm_bwd_ref(x: &[f32], dy: &[f32], eps: f32, rows: usize, dim: usize) -> Vec<f32> {
    let mut dx = vec![0.0; rows * dim];
    for r in 0..rows {
        let ss: f32 = (0..dim).map(|i| x[r * dim + i] * x[r * dim + i]).sum();
        let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
        let dot: f32 = (0..dim).map(|i| dy[r * dim + i] * x[r * dim + i]).sum();
        let c = inv * inv * inv * dot / dim as f32;
        for i in 0..dim {
            dx[r * dim + i] = inv * dy[r * dim + i] - c * x[r * dim + i];
        }
    }
    dx
}

#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn rmsnorm_matches_cpu() {
    let (device, queue) = setup();
    let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
    let (rows, dim, eps) = (3usize, 8usize, 1e-6f32);
    let x = values(rows * dim, 7);
    let dy = values(rows * dim, 11);

    let xb = buf(&device, &queue, &x);
    let yb = GpuBuffer::zeros(&device, &queue, rows * dim * 4, Some("y")).unwrap();
    let dyb = buf(&device, &queue, &dy);
    let dxb = GpuBuffer::zeros(&device, &queue, rows * dim * 4, Some("dx")).unwrap();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    ops.rmsnorm_forward(&mut enc, &xb, &yb, rows as u32, dim as u32, eps)
        .unwrap();
    ops.rmsnorm_backward(&mut enc, &dyb, &xb, &dxb, rows as u32, dim as u32, eps)
        .unwrap();
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let y = read(&device, &queue, &yb, rows * dim);
    let dx = read(&device, &queue, &dxb, rows * dim);
    let ey = rmsnorm_ref(&x, eps, rows, dim);
    let edx = rmsnorm_bwd_ref(&x, &dy, eps, rows, dim);
    let dy_err = max_abs(&y, &ey);
    let dx_err = max_abs(&dx, &edx);
    eprintln!("rmsnorm fwd max_abs={dy_err:.3e} bwd max_abs={dx_err:.3e}");
    assert!(dy_err < TOL, "fwd {dy_err}");
    assert!(dx_err < TOL, "bwd {dx_err}");
}

#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn mul_broadcast_matches_cpu() {
    let (device, queue) = setup();
    let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
    let (rows, cols) = (4usize, 5usize);
    let a = values(rows * cols, 3);
    let w = values(cols, 9);
    let dout = values(rows * cols, 13);

    let ab = buf(&device, &queue, &a);
    let wb = buf(&device, &queue, &w);
    let outb = GpuBuffer::zeros(&device, &queue, rows * cols * 4, Some("out")).unwrap();
    let doutb = buf(&device, &queue, &dout);
    let dab = GpuBuffer::zeros(&device, &queue, rows * cols * 4, Some("da")).unwrap();
    let dwb = GpuBuffer::zeros(&device, &queue, cols * 4, Some("dw")).unwrap();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    ops.mul_broadcast_forward(&mut enc, &ab, &wb, &outb, rows as u32, cols as u32)
        .unwrap();
    ops.mul_broadcast_backward(
        &mut enc,
        &doutb,
        &ab,
        &wb,
        &dab,
        &dwb,
        rows as u32,
        cols as u32,
    )
    .unwrap();
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let out = read(&device, &queue, &outb, rows * cols);
    let da = read(&device, &queue, &dab, rows * cols);
    let dw = read(&device, &queue, &dwb, cols);
    let eout: Vec<f32> = (0..rows * cols).map(|i| a[i] * w[i % cols]).collect();
    let eda: Vec<f32> = (0..rows * cols).map(|i| dout[i] * w[i % cols]).collect();
    let edw: Vec<f32> = (0..cols)
        .map(|c| (0..rows).map(|r| dout[r * cols + c] * a[r * cols + c]).sum())
        .collect();
    let e1 = max_abs(&out, &eout);
    let e2 = max_abs(&da, &eda);
    let e3 = max_abs(&dw, &edw);
    eprintln!("mul_bcast fwd={e1:.3e} da={e2:.3e} dw={e3:.3e}");
    assert!(e1 < TOL && e2 < TOL && e3 < TOL);
}

#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn silu_mul_matches_cpu() {
    let (device, queue) = setup();
    let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
    let n = 12usize;
    let gate = values(n, 21);
    let up = values(n, 22);
    let dout = values(n, 23);
    let gb = buf(&device, &queue, &gate);
    let ub = buf(&device, &queue, &up);
    let outb = GpuBuffer::zeros(&device, &queue, n * 4, Some("out")).unwrap();
    let doutb = buf(&device, &queue, &dout);
    let dgb = GpuBuffer::zeros(&device, &queue, n * 4, Some("dg")).unwrap();
    let dub = GpuBuffer::zeros(&device, &queue, n * 4, Some("du")).unwrap();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    ops.silu_mul_forward(&mut enc, &gb, &ub, &outb, n as u32).unwrap();
    ops.silu_mul_backward(&mut enc, &gb, &ub, &doutb, &dgb, &dub, n as u32)
        .unwrap();
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let silu = |x: f32| x / (1.0 + (-x).exp());
    let silu_grad = |x: f32| {
        let s = 1.0 / (1.0 + (-x).exp());
        s * (1.0 + x * (1.0 - s))
    };
    let eout: Vec<f32> = (0..n).map(|i| silu(gate[i]) * up[i]).collect();
    let edg: Vec<f32> = (0..n).map(|i| dout[i] * up[i] * silu_grad(gate[i])).collect();
    let edu: Vec<f32> = (0..n).map(|i| dout[i] * silu(gate[i])).collect();
    let e1 = max_abs(&read(&device, &queue, &outb, n), &eout);
    let e2 = max_abs(&read(&device, &queue, &dgb, n), &edg);
    let e3 = max_abs(&read(&device, &queue, &dub, n), &edu);
    eprintln!("silu_mul fwd={e1:.3e} dgate={e2:.3e} dup={e3:.3e}");
    assert!(e1 < TOL && e2 < TOL && e3 < TOL);
}

#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn rope_matches_cpu_gemma_mapper() {
    use ferrisres::model::gemma_mapper::apply_rope;
    let (device, queue) = setup();
    let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
    let (seq, heads, head_dim) = (3usize, 2usize, 8usize);
    let n = seq * heads * head_dim;
    let x = values(n, 31);
    let dx = values(n, 32);

    let xb = buf(&device, &queue, &x);
    let outb = GpuBuffer::zeros(&device, &queue, n * 4, Some("out")).unwrap();
    let dxb = buf(&device, &queue, &dx);
    let doutb = GpuBuffer::zeros(&device, &queue, n * 4, Some("dout")).unwrap();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    ops.rope_forward(
        &mut enc,
        &xb,
        &outb,
        seq as u32,
        heads as u32,
        head_dim as u32,
        10_000.0,
        0,
    )
    .unwrap();
    ops.rope_backward(
        &mut enc,
        &dxb,
        &doutb,
        seq as u32,
        heads as u32,
        head_dim as u32,
        10_000.0,
        0,
    )
    .unwrap();
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let mut expected = x.clone();
    apply_rope(&mut expected, seq, heads, head_dim, 0, 10_000.0, 1.0);
    let got = read(&device, &queue, &outb, n);
    let e1 = max_abs(&got, &expected);
    eprintln!("rope fwd max_abs={e1:.3e}");
    assert!(e1 < TOL, "rope fwd {e1}");

    // Transposed rotation: reference backward is the inverse of the forward.
    let half = head_dim / 2;
    let mut expected_dx = vec![0.0; n];
    for t in 0..seq {
        for h in 0..heads {
            let base = (t * heads + h) * head_dim;
            for d in 0..half {
                let freq = 1.0 / 10_000.0f64.powf((2 * d) as f64 / head_dim as f64);
                let angle = (t as f64 * freq) as f32;
                let (c, s) = (angle.cos(), angle.sin());
                let da = dx[base + d];
                let db = dx[base + d + half];
                expected_dx[base + d] = da * c + db * s;
                expected_dx[base + d + half] = -da * s + db * c;
            }
        }
    }
    let got_dx = read(&device, &queue, &doutb, n);
    let e2 = max_abs(&got_dx, &expected_dx);
    eprintln!("rope bwd max_abs={e2:.3e}");
    assert!(e2 < TOL, "rope bwd {e2}");
}

#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn gqa_expand_matches_cpu() {
    let (device, queue) = setup();
    let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
    let (seq, heads, kv_heads, head_dim) = (2usize, 4usize, 2usize, 3usize);
    let kv_dim = kv_heads * head_dim;
    let q_dim = heads * head_dim;
    let kv = values(seq * kv_dim, 41);
    let dout = values(seq * q_dim, 42);

    let kvb = buf(&device, &queue, &kv);
    let outb = GpuBuffer::zeros(&device, &queue, seq * q_dim * 4, Some("out")).unwrap();
    let doutb = buf(&device, &queue, &dout);
    let dkvb = GpuBuffer::zeros(&device, &queue, seq * kv_dim * 4, Some("dkv")).unwrap();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    ops.gqa_expand_forward(
        &mut enc,
        &kvb,
        &outb,
        seq as u32,
        heads as u32,
        kv_heads as u32,
        head_dim as u32,
    )
    .unwrap();
    ops.gqa_expand_backward(
        &mut enc,
        &doutb,
        &dkvb,
        seq as u32,
        heads as u32,
        kv_heads as u32,
        head_dim as u32,
    )
    .unwrap();
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let group = heads / kv_heads;
    let eout: Vec<f32> = (0..seq * q_dim)
        .map(|i| {
            let d = i % head_dim;
            let h = (i / head_dim) % heads;
            let t = i / head_dim / heads;
            kv[(t * kv_heads + h / group) * head_dim + d]
        })
        .collect();
    let edkv: Vec<f32> = (0..seq * kv_dim)
        .map(|i| {
            let d = i % head_dim;
            let kvh = (i / head_dim) % kv_heads;
            let t = i / head_dim / kv_heads;
            (0..group)
                .map(|g| dout[(t * heads + kvh * group + g) * head_dim + d])
                .sum()
        })
        .collect();
    let e1 = max_abs(&read(&device, &queue, &outb, seq * q_dim), &eout);
    let e2 = max_abs(&read(&device, &queue, &dkvb, seq * kv_dim), &edkv);
    eprintln!("gqa fwd={e1:.3e} bwd={e2:.3e}");
    assert!(e1 < TOL && e2 < TOL);
}

/// Reference causal attention matching `f32_trainer::attention_fwd/bwd`
/// (scale 1, GQA over pre-expanded K/V).
fn attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dout: &[f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_dim = heads * head_dim;
    let mut out = vec![0.0; seq * q_dim];
    let mut dq = vec![0.0; seq * q_dim];
    let mut dk = vec![0.0; seq * q_dim];
    let mut dv = vec![0.0; seq * q_dim];
    for h in 0..heads {
        for t in 0..seq {
            let base = (t * heads + h) * head_dim;
            let mut probs = vec![0.0; seq];
            let mut mx = f32::NEG_INFINITY;
            for s in 0..=t {
                let kb = (s * heads + h) * head_dim;
                let dot: f32 = (0..head_dim).map(|d| q[base + d] * k[kb + d]).sum();
                probs[s] = dot * scale;
                mx = mx.max(probs[s]);
            }
            let mut sum = 0.0;
            for s in 0..=t {
                probs[s] = (probs[s] - mx).exp();
                sum += probs[s];
            }
            for s in 0..=t {
                probs[s] /= sum;
            }
            for d in 0..head_dim {
                out[base + d] = (0..=t)
                    .map(|s| probs[s] * v[(s * heads + h) * head_dim + d])
                    .sum();
            }

            // backward
            let mut dp = vec![0.0; seq];
            for s in 0..=t {
                let kb = (s * heads + h) * head_dim;
                dp[s] = (0..head_dim)
                    .map(|d| dout[base + d] * v[kb + d])
                    .sum();
                for d in 0..head_dim {
                    dv[kb + d] += probs[s] * dout[base + d];
                }
            }
            let weighted: f32 = (0..=t).map(|s| probs[s] * dp[s]).sum();
            for s in 0..=t {
                let ds = probs[s] * (dp[s] - weighted);
                let kb = (s * heads + h) * head_dim;
                for d in 0..head_dim {
                    dq[base + d] += ds * scale * k[kb + d];
                    dk[kb + d] += ds * scale * q[base + d];
                }
            }
        }
    }
    (out, dq, dk, dv)
}

#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn attention_causal_mask_and_gradients_match_cpu() {
    let (device, queue) = setup();
    let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
    let (seq, heads, head_dim) = (4usize, 2usize, 4usize);
    let dim = seq * heads * head_dim;
    let q = values(dim, 51);
    let k = values(dim, 52);
    let v = values(dim, 53);
    let dout = values(dim, 54);
    let scale = 1.0f32;

    let qb = buf(&device, &queue, &q);
    let kb = buf(&device, &queue, &k);
    let vb = buf(&device, &queue, &v);
    let outb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("out")).unwrap();
    let doutb = buf(&device, &queue, &dout);
    let dqb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("dq")).unwrap();
    let dkb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("dk")).unwrap();
    let dvb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("dv")).unwrap();
    let scratch = heads * seq * seq * 4;
    let probsb = GpuBuffer::zeros(&device, &queue, scratch, Some("p")).unwrap();
    let dsb = GpuBuffer::zeros(&device, &queue, scratch, Some("ds")).unwrap();

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    ops.attention_forward(
        &mut enc,
        &qb,
        &kb,
        &vb,
        &outb,
        seq as u32,
        heads as u32,
        head_dim as u32,
        scale,
    )
    .unwrap();
    ops.attention_backward(
        &mut enc,
        &doutb,
        &qb,
        &kb,
        &vb,
        &dqb,
        &dkb,
        &dvb,
        &probsb,
        &dsb,
        seq as u32,
        heads as u32,
        head_dim as u32,
        scale,
    )
    .unwrap();
    queue.submit([enc.finish()]);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let (eout, edq, edk, edv) = attention_ref(&q, &k, &v, &dout, seq, heads, head_dim, scale);
    let e1 = max_abs(&read(&device, &queue, &outb, dim), &eout);
    let e2 = max_abs(&read(&device, &queue, &dqb, dim), &edq);
    let e3 = max_abs(&read(&device, &queue, &dkb, dim), &edk);
    let e4 = max_abs(&read(&device, &queue, &dvb, dim), &edv);
    eprintln!("attention fwd={e1:.3e} dq={e2:.3e} dk={e3:.3e} dv={e4:.3e}");

    // Causality: changing a future token must not change an earlier output.
    let future = values(dim, 55); // differ only at t = seq-1
    let mut altered = v.clone();
    for i in 0..head_dim * heads {
        altered[((seq - 1) * heads * head_dim) + i] = future[i];
    }
    let (altered_out, ..) = attention_ref(&q, &k, &altered, &dout, seq, heads, head_dim, scale);
    let last = (seq - 1) * heads * head_dim;
    for i in 0..last {
        assert!(
            (eout[i] - altered_out[i]).abs() < 1e-6,
            "token {i} saw the future"
        );
    }

    assert!(e1 < TOL && e2 < TOL && e3 < TOL && e4 < TOL);
}

/// The 64-lane backward shaders at shapes that exercise both branches of the
/// head-dimension stride: one warp per head (64), and the two chunked cases
/// (128, 256). Sequence lengths 1, 2, 17 and 64 cover an empty `s` range, a
/// single visible key, an odd length and a full warp-row. The CPU oracle is
/// the same `attention_ref` the original single-invocation kernel used, so a
/// failure here is a wrong result, not just a different summation order.
#[test]
#[ignore = "needs a Vulkan/WGPU device; run with VK_ICD_FILENAMES set"]
fn attention_backward_64_lane_grid_matches_cpu() {
    let (device, queue) = setup();
    let ops = TrainOps::new(Arc::clone(&device), Arc::clone(&queue));
    let cases = [
        (1usize, 2usize, 64usize),
        (2, 2, 64),
        (17, 2, 64),
        (64, 2, 64),
        (17, 2, 128),
        (4, 2, 256),
    ];
    for (case_index, (seq, heads, head_dim)) in cases.into_iter().enumerate() {
        let dim = seq * heads * head_dim;
        let seed = 70 + case_index as u32 * 4;
        let q = values(dim, seed);
        let k = values(dim, seed + 1);
        let v = values(dim, seed + 2);
        let dout = values(dim, seed + 3);
        let scale = 0.5f32;

        let qb = buf(&device, &queue, &q);
        let kb = buf(&device, &queue, &k);
        let vb = buf(&device, &queue, &v);
        let outb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("out")).unwrap();
        let doutb = buf(&device, &queue, &dout);
        let dqb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("dq")).unwrap();
        let dkb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("dk")).unwrap();
        let dvb = GpuBuffer::zeros(&device, &queue, dim * 4, Some("dv")).unwrap();
        let scratch = heads * seq * seq * 4;
        let probsb = GpuBuffer::zeros(&device, &queue, scratch, Some("p")).unwrap();
        let dsb = GpuBuffer::zeros(&device, &queue, scratch, Some("ds")).unwrap();

        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        ops.attention_forward(
            &mut enc,
            &qb,
            &kb,
            &vb,
            &outb,
            seq as u32,
            heads as u32,
            head_dim as u32,
            scale,
        )
        .unwrap();
        ops.attention_backward(
            &mut enc,
            &doutb,
            &qb,
            &kb,
            &vb,
            &dqb,
            &dkb,
            &dvb,
            &probsb,
            &dsb,
            seq as u32,
            heads as u32,
            head_dim as u32,
            scale,
        )
        .unwrap();
        queue.submit([enc.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let (eout, edq, edk, edv) = attention_ref(&q, &k, &v, &dout, seq, heads, head_dim, scale);
        let e1 = max_abs(&read(&device, &queue, &outb, dim), &eout);
        let e2 = max_abs(&read(&device, &queue, &dqb, dim), &edq);
        let e3 = max_abs(&read(&device, &queue, &dkb, dim), &edk);
        let e4 = max_abs(&read(&device, &queue, &dvb, dim), &edv);
        eprintln!(
            "attention seq={seq} heads={heads} head_dim={head_dim} \
             fwd={e1:.3e} dq={e2:.3e} dk={e3:.3e} dv={e4:.3e}"
        );
        assert!(
            e1 < TOL && e2 < TOL && e3 < TOL && e4 < TOL,
            "seq={seq} heads={heads} head_dim={head_dim}: \
             fwd={e1:.3e} dq={e2:.3e} dk={e3:.3e} dv={e4:.3e}"
        );
    }
}
