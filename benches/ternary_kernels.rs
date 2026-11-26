//! Kernel benchmarks for the CPU ternary matmul paths and the decode path.
//!
//! Each group measures two implementations of the *same* kernel in one
//! process, so both see the same host load:
//!
//! * `baseline` — a verbatim copy of the kernel as it was before the
//!   branchless rewrite (kept here as the reference the numbers are relative
//!   to; it is not compiled into the library).
//! * `current` — the implementation in `ferrisres::model::ternary`.
//!
//! Dimensions are one Base layer: `out_rows = 2816`, `in_cols = 1024`.
//! The `model_decode` group times the real decode path of
//! `CpuBlockAttnResModel` (hidden 768, 4 layers, 2 layers per block) which is
//! where the q/k/v/o projections are called with `seq == 1`.
//!
//! Numbers from this host are relative under load, not absolute.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use ferrisres::inference::student_kv_cache::ModelKVCache;
use ferrisres::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel};
use ferrisres::model::cpu_linear::{CpuLinear, CpuRmsNorm};
use ferrisres::model::gemma_mapper::gelu_tanh;
use ferrisres::model::ternary::{
    pack_ternary, ternary_matmul, ternary_matmul_parallel, ternary_matmul_packed_parallel,
};

// One Base layer.
const OUT_ROWS: usize = 2816;
const IN_COLS: usize = 1024;

/// Decoding table used by the pre-rewrite packed kernel.
const TERNARY_DECODE: [i8; 4] = [-1, 0, 1, 0];

// ── Baseline: the kernels before the rewrite ────────────────────────────────

fn baseline_ternary_matmul(
    ternary: &[i8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    seq: usize,
) -> Vec<f32> {
    assert_eq!(ternary.len(), out_rows * in_cols);
    assert_eq!(input.len(), seq * in_cols);

    let mut output = vec![0.0f32; seq * out_rows];
    for s in 0..seq {
        let input_row = &input[s * in_cols..(s + 1) * in_cols];
        let out_slice = &mut output[s * out_rows..(s + 1) * out_rows];
        for r in 0..out_rows {
            let weight_row = &ternary[r * in_cols..(r + 1) * in_cols];
            let mut pos_sum = 0.0f32;
            let mut neg_sum = 0.0f32;
            for j in 0..in_cols {
                let w = weight_row[j];
                let v = input_row[j];
                pos_sum += if w > 0 { v } else { 0.0 };
                neg_sum += if w < 0 { v } else { 0.0 };
            }
            out_slice[r] = scale * (pos_sum - neg_sum);
        }
    }
    output
}

fn baseline_ternary_matmul_parallel(
    ternary: &[i8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    seq: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(ternary.len(), out_rows * in_cols);
    assert_eq!(input.len(), seq * in_cols);

    if seq <= 1 {
        let mut output = vec![0.0f32; out_rows];
        let batch_size = (out_rows / 32).max(1);
        output
            .par_chunks_mut(batch_size)
            .enumerate()
            .for_each(|(batch_idx, chunk)| {
                let start_row = batch_idx * batch_size;
                for r in 0..chunk.len() {
                    let row = start_row + r;
                    let weight_row = &ternary[row * in_cols..(row + 1) * in_cols];
                    let input_row = &input[..in_cols];
                    let mut pos_sum = 0.0f32;
                    let mut neg_sum = 0.0f32;
                    for j in 0..in_cols {
                        let w = weight_row[j];
                        let v = input_row[j];
                        pos_sum += if w > 0 { v } else { 0.0 };
                        neg_sum += if w < 0 { v } else { 0.0 };
                    }
                    chunk[r] = scale * (pos_sum - neg_sum);
                }
            });
        return output;
    }

    let mut output = vec![0.0f32; seq * out_rows];
    output
        .par_chunks_mut(out_rows)
        .enumerate()
        .for_each(|(s, out_slice)| {
            let input_row = &input[s * in_cols..(s + 1) * in_cols];
            for r in 0..out_rows {
                let weight_row = &ternary[r * in_cols..(r + 1) * in_cols];
                let mut pos_sum = 0.0f32;
                let mut neg_sum = 0.0f32;
                for j in 0..in_cols {
                    let w = weight_row[j];
                    let v = input_row[j];
                    pos_sum += if w > 0 { v } else { 0.0 };
                    neg_sum += if w < 0 { v } else { 0.0 };
                }
                out_slice[r] = scale * (pos_sum - neg_sum);
            }
        });
    output
}

fn baseline_ternary_matmul_packed_parallel(
    packed: &[u8],
    scale: f32,
    input: &[f32],
    out_rows: usize,
    in_cols: usize,
    _seq: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let mut output = vec![0.0f32; out_rows];
    let batch_size = (out_rows / 32).max(1);

    output
        .par_chunks_mut(batch_size)
        .enumerate()
        .for_each(|(batch_idx, chunk)| {
            let start_row = batch_idx * batch_size;
            let input_row = &input[..in_cols];
            let full_quads = in_cols / 4;
            let remainder = in_cols % 4;

            for r in 0..chunk.len() {
                let row = start_row + r;
                let row_packed_start = row * ((in_cols + 3) / 4);
                let mut sum = 0.0f32;

                for q in 0..full_quads {
                    let packed_byte = packed[row_packed_start + q];

                    let s0 = TERNARY_DECODE[(packed_byte & 0b11) as usize] as f32;
                    let s1 = TERNARY_DECODE[((packed_byte >> 2) & 0b11) as usize] as f32;
                    let s2 = TERNARY_DECODE[((packed_byte >> 4) & 0b11) as usize] as f32;
                    let s3 = TERNARY_DECODE[((packed_byte >> 6) & 0b11) as usize] as f32;

                    let base = q * 4;
                    sum += s0 * input_row[base]
                        + s1 * input_row[base + 1]
                        + s2 * input_row[base + 2]
                        + s3 * input_row[base + 3];
                }

                if remainder > 0 {
                    let packed_byte = packed[row_packed_start + full_quads];
                    let base = full_quads * 4;
                    for bit_pos in 0..remainder {
                        let code = ((packed_byte >> (bit_pos * 2)) & 0b11) as usize;
                        sum += TERNARY_DECODE[code] as f32 * input_row[base + bit_pos];
                    }
                }

                chunk[r] = scale * sum;
            }
        });

    output
}

// ── Data ────────────────────────────────────────────────────────────────────

/// Deterministic SplitMix64 so every run sees the same weights.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// {-1, 0, +1} with the ~1/3 each a quantized absmean tensor has.
    fn ternary(&mut self) -> i8 {
        (self.next_u64() % 3) as i8 - 1
    }
}

fn ternary_weights(n: usize, seed: u64) -> Vec<i8> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.ternary()).collect()
}

fn activations(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n).map(|_| rng.unit() * 2.0 - 1.0).collect()
}

// ── Kernel groups ───────────────────────────────────────────────────────────

fn bench_unpacked_serial(c: &mut Criterion) {
    let ternary = ternary_weights(OUT_ROWS * IN_COLS, 0x5EED_0001);
    let input = activations(IN_COLS, 0x5EED_0002);
    let scale = 0.0123f32;

    let mut group = c.benchmark_group("ternary_unpacked_serial_seq1");
    group.throughput(Throughput::Bytes((OUT_ROWS * IN_COLS) as u64));
    group.bench_function("baseline", |b| {
        b.iter(|| {
            black_box(baseline_ternary_matmul(
                black_box(&ternary),
                scale,
                black_box(&input),
                OUT_ROWS,
                IN_COLS,
                1,
            ))
        })
    });
    group.bench_function("current", |b| {
        b.iter(|| {
            black_box(ternary_matmul(
                black_box(&ternary),
                scale,
                black_box(&input),
                OUT_ROWS,
                IN_COLS,
                1,
            ))
        })
    });
    group.finish();
}

fn bench_unpacked_parallel(c: &mut Criterion) {
    let ternary = ternary_weights(OUT_ROWS * IN_COLS, 0x5EED_0003);
    let scale = 0.0123f32;

    let mut group = c.benchmark_group("ternary_unpacked_parallel_seq1");
    group.throughput(Throughput::Bytes((OUT_ROWS * IN_COLS) as u64));
    let input = activations(IN_COLS, 0x5EED_0004);
    group.bench_function("baseline", |b| {
        b.iter(|| {
            black_box(baseline_ternary_matmul_parallel(
                black_box(&ternary),
                scale,
                black_box(&input),
                OUT_ROWS,
                IN_COLS,
                1,
            ))
        })
    });
    group.bench_function("current", |b| {
        b.iter(|| {
            black_box(ternary_matmul_parallel(
                black_box(&ternary),
                scale,
                black_box(&input),
                OUT_ROWS,
                IN_COLS,
                1,
            ))
        })
    });
    group.finish();

    // Prefill shape: 8 tokens at once, parallelised across sequence positions.
    let input8 = activations(8 * IN_COLS, 0x5EED_0005);
    let mut group = c.benchmark_group("ternary_unpacked_parallel_seq8");
    group.throughput(Throughput::Bytes((OUT_ROWS * IN_COLS) as u64));
    group.bench_function("baseline", |b| {
        b.iter(|| {
            black_box(baseline_ternary_matmul_parallel(
                black_box(&ternary),
                scale,
                black_box(&input8),
                OUT_ROWS,
                IN_COLS,
                8,
            ))
        })
    });
    group.bench_function("current", |b| {
        b.iter(|| {
            black_box(ternary_matmul_parallel(
                black_box(&ternary),
                scale,
                black_box(&input8),
                OUT_ROWS,
                IN_COLS,
                8,
            ))
        })
    });
    group.finish();
}

fn bench_packed_parallel(c: &mut Criterion) {
    let ternary = ternary_weights(OUT_ROWS * IN_COLS, 0x5EED_0006);
    let packed = pack_ternary(&ternary);
    let input = activations(IN_COLS, 0x5EED_0007);
    let scale = 0.0123f32;

    let mut group = c.benchmark_group("ternary_packed_parallel_decode");
    group.throughput(Throughput::Bytes(packed.len() as u64));
    group.bench_function("baseline", |b| {
        b.iter(|| {
            black_box(baseline_ternary_matmul_packed_parallel(
                black_box(&packed),
                scale,
                black_box(&input),
                OUT_ROWS,
                IN_COLS,
                1,
            ))
        })
    });
    group.bench_function("current", |b| {
        b.iter(|| {
            black_box(ternary_matmul_packed_parallel(
                black_box(&packed),
                scale,
                black_box(&input),
                OUT_ROWS,
                IN_COLS,
                1,
            ))
        })
    });
    group.finish();
}

// ── Model decode path ───────────────────────────────────────────────────────

const MODEL_HIDDEN: usize = 768;
const MODEL_LAYERS: usize = 4;
const MODEL_HEADS: usize = 6;
const MODEL_KV_HEADS: usize = 2;
const MODEL_HEAD_DIM: usize = 128;
const MODEL_INTER: usize = 2048;
const MODEL_VOCAB: usize = 1024;
const PREFILL_TOKENS: [u32; 4] = [3, 17, 42, 7];

fn random_linear(rng: &mut Rng, in_features: usize, out_features: usize) -> CpuLinear {
    let w: Vec<f32> = (0..in_features * out_features)
        .map(|_| rng.unit() * 0.2 - 0.1)
        .collect();
    CpuLinear::from_weight(w, in_features, out_features)
}

fn random_norm(rng: &mut Rng, dim: usize) -> CpuRmsNorm {
    let w: Vec<f32> = (0..dim).map(|_| 0.5 + rng.unit()).collect();
    CpuRmsNorm::from_weight(w, 1e-6)
}

fn build_model() -> CpuBlockAttnResModel {
    let mut rng = Rng(0xB10C_0001);
    let h = MODEL_HIDDEN;
    let q_dim = MODEL_HEADS * MODEL_HEAD_DIM;
    let kv_dim = MODEL_KV_HEADS * MODEL_HEAD_DIM;

    let layers = (0..MODEL_LAYERS)
        .map(|idx| CpuBlockAttnResLayer {
            layer_number: idx,
            hidden_dim: h,
            num_heads: MODEL_HEADS,
            num_kv_heads: MODEL_KV_HEADS,
            head_dim: MODEL_HEAD_DIM,
            intermediate_dim: MODEL_INTER,
            attn_norm: random_norm(&mut rng, h),
            q_proj: random_linear(&mut rng, h, q_dim),
            k_proj: random_linear(&mut rng, h, kv_dim),
            v_proj: random_linear(&mut rng, h, kv_dim),
            out_proj: random_linear(&mut rng, q_dim, h),
            q_norm: random_norm(&mut rng, MODEL_HEAD_DIM),
            k_norm: random_norm(&mut rng, MODEL_HEAD_DIM),
            v_norm: random_norm(&mut rng, MODEL_HEAD_DIM),
            post_attn_norm: random_norm(&mut rng, h),
            pre_ffn_norm: random_norm(&mut rng, h),
            ffn_gate: Some(random_linear(&mut rng, h, MODEL_INTER)),
            ffn_up: Some(random_linear(&mut rng, h, MODEL_INTER)),
            ffn_down: Some(random_linear(&mut rng, MODEL_INTER, h)),
            moe: None,
            post_ffn_norm: random_norm(&mut rng, h),
            layer_scalar: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
            rope_theta: 10_000.0,
            partial_rotary_factor: 1.0,
            use_gelu: true,
            kv_shared: false,
        })
        .collect();

    let embed_tokens: Vec<f32> = (0..MODEL_VOCAB * h).map(|_| rng.unit() * 0.1).collect();
    let lm_head: Vec<f32> = (0..h * MODEL_VOCAB).map(|_| rng.unit() * 0.1).collect();
    let attn_res_proj: Vec<f32> = (0..h * h).map(|_| rng.unit() * 0.01).collect();

    CpuBlockAttnResModel {
        layers,
        embed_tokens,
        lm_head,
        final_norm: random_norm(&mut rng, h).weight().to_vec(),
        hidden_dim: h,
        vocab_size: MODEL_VOCAB,
        num_layers: MODEL_LAYERS,
        final_logit_softcapping: None,
        ple_model_projection: None,
        ple_projection_norm: None,
        embed_tokens_per_layer: None,
        hidden_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        block_config: BlockConfig {
            num_blocks: 2,
            layers_per_block: 2,
            boundary_layers: vec![1, 3],
            attn_res_proj,
            attn_res_norm: vec![1.0; h],
        },
        lora_manager: None,
    }
}

fn new_cache() -> ModelKVCache {
    let kv_dims = vec![MODEL_KV_HEADS * MODEL_HEAD_DIM; MODEL_LAYERS];
    ModelKVCache::new(MODEL_LAYERS, MODEL_HIDDEN, &kv_dims, 2, 0)
}

/// Time the decode path: 4 prefill tokens, then 8 single-token decode steps.
/// The prefill is setup and is not measured.
fn bench_model_decode(c: &mut Criterion) {
    let model = build_model();
    let mut group = c.benchmark_group("model_decode");
    group.sample_size(10);
    group.bench_function("decode_8_tokens_after_prefill4", |b| {
        b.iter_batched(
            || {
                let mut cache = new_cache();
                model.forward_prefill(&PREFILL_TOKENS, &mut cache);
                cache
            },
            |mut cache| {
                for _ in 0..8 {
                    black_box(model.forward_decode(21, &mut cache));
                }
            },
            BatchSize::PerIteration,
        )
    });
    group.finish();
}

/// Time the prefill path (8 tokens at once). This is the `seq > 1` branch of
/// the attention projections.
fn bench_model_prefill(c: &mut Criterion) {
    let model = build_model();
    const TOKENS: [u32; 8] = [3, 17, 42, 7, 11, 5, 29, 64];
    let mut group = c.benchmark_group("model_prefill");
    group.sample_size(10);
    group.bench_function("prefill_8_tokens", |b| {
        b.iter(|| {
            let mut cache = new_cache();
            black_box(model.forward_prefill(&TOKENS, &mut cache))
        })
    });
    group.finish();
}

/// Time one layer's attention block only — q/k/v projection, GQA attention,
/// out projection, post-attention norm and residual. This isolates the call
/// sites changed by the projection dispatch from the (unchanged) FFN, which
/// dominates a full layer.
fn bench_layer_attention(c: &mut Criterion) {
    let model = build_model();
    let layer = &model.layers[0];
    let mut rng = Rng(0xB10C_0002);
    let input: Vec<f32> = (0..8 * MODEL_HIDDEN).map(|_| rng.unit() * 2.0 - 1.0).collect();

    let mut group = c.benchmark_group("layer_attention");
    group.sample_size(20);
    for seq in [1usize, 8] {
        let row = &input[..seq * MODEL_HIDDEN];
        group.bench_function(format!("seq{seq}"), |b| {
            b.iter(|| {
                let mut hidden = row.to_vec();
                black_box(layer.forward(&mut hidden, None, None));
            })
        });
    }
    group.finish();
}

/// The dense FFN as it was before the projection dispatch: gate/up/down
/// through the serial `CpuLinear::forward` (unpacked, branchy, one thread),
/// with the same pre/post norm and residual as `forward_ffn`.
fn baseline_forward_ffn(layer: &CpuBlockAttnResLayer, hidden: &mut Vec<f32>, seq: usize) {
    let id = layer.intermediate_dim;
    let residual2 = hidden.clone();
    let normed2 = layer.pre_ffn_norm.forward(hidden);

    let gate = layer.ffn_gate.as_ref().expect("dense FFN");
    let up = layer.ffn_up.as_ref().expect("dense FFN");
    let down = layer.ffn_down.as_ref().expect("dense FFN");

    let gated = gate.forward(&normed2, seq);
    let gated: Vec<f32> = if layer.use_gelu {
        gated.iter().map(|&x| gelu_tanh(x)).collect()
    } else {
        gated.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
    };
    let upped = up.forward(&normed2, seq);
    let mut combined = vec![0.0; seq * id];
    for i in 0..combined.len() {
        combined[i] = gated[i] * upped[i];
    }
    let ffn_out = down.forward(&combined, seq);

    let ffn_out = layer.post_ffn_norm.forward(&ffn_out);
    for i in 0..hidden.len() {
        hidden[i] = residual2[i] + ffn_out[i];
    }
    if layer.layer_scalar != 1.0 {
        for h in hidden.iter_mut() {
            *h *= layer.layer_scalar;
        }
    }
}

/// Time one layer's dense FFN only — pre-FFN norm, gate/up/down projections,
/// activation, post-FFN norm and residual. Decode uses `forward_ffn_parallel`,
/// prefill uses `forward_ffn`; both reach `cpu_ffn`, which is where the
/// projection dispatch now lives.
///
/// `baseline` is the pre-dispatch FFN compiled into this bench, so the pair is
/// measured in one process and does not depend on the host load.
fn bench_layer_ffn(c: &mut Criterion) {
    let model = build_model();
    let layer = &model.layers[0];
    let mut rng = Rng(0xB10C_0003);
    let input: Vec<f32> = (0..8 * MODEL_HIDDEN).map(|_| rng.unit() * 2.0 - 1.0).collect();

    let mut group = c.benchmark_group("layer_ffn");
    group.sample_size(20);
    for seq in [1usize, 8] {
        let row = &input[..seq * MODEL_HIDDEN];
        group.bench_function(format!("seq{seq}/baseline"), |b| {
            b.iter(|| {
                let mut hidden = row.to_vec();
                baseline_forward_ffn(layer, &mut hidden, seq);
                black_box(hidden)
            })
        });
        group.bench_function(format!("seq{seq}/current"), |b| {
            b.iter(|| {
                let mut hidden = row.to_vec();
                if seq == 1 {
                    layer.forward_ffn_parallel(&mut hidden, None);
                } else {
                    layer.forward_ffn(&mut hidden, None);
                }
                black_box(hidden)
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_unpacked_serial,
    bench_unpacked_parallel,
    bench_packed_parallel,
    bench_model_decode,
    bench_model_prefill,
    bench_layer_attention,
    bench_layer_ffn,
);
criterion_main!(benches);
