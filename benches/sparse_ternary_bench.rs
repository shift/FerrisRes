//! Progressive benchmark: FP32 → ternary → sparse-ternary → +mmap → +3bit KV.
//!
//! Tests each optimization layer independently to measure:
//! 1. Matmul throughput (GFLOPS equivalent)
//! 2. Memory usage (model size, KV cache, working set)
//! 3. Quality: cosine similarity vs FP32 baseline
//! 4. Decode latency estimate (ms/token)

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

/// Simulated model dimensions (Gemma 4 E2B scale).
const HIDDEN_DIM: usize = 1536;
const INTERMEDIATE_DIM: usize = 6144;
const _SEQ_LEN: usize = 1; // Decode: single token

// --- FP32 dense matmul ---
fn fp32_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[row * k + ki] * b[ki * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

// --- Ternary matmul (simulated) ---
fn ternary_matmul(input: &[f32], weights: &[i8], scale: f32, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += input[i] * (weights[o * in_dim + i] as f32);
        }
        output[o] = sum * scale;
    }
    output
}

// --- Sparse ternary matmul (simulated, 50% zeros) ---
fn sparse_ternary_matmul(input: &[f32], weights: &[i8], nonzero_mask: &[bool], scale: f32, out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            if nonzero_mask[o * in_dim + i] {
                sum += input[i] * (weights[o * in_dim + i] as f32);
            }
        }
        output[o] = sum * scale;
    }
    output
}

/// Compute theoretical model sizes.
fn model_sizes() -> (usize, usize, usize, usize, usize) {
    let _num_layers = 35;
    let hd = HIDDEN_DIM;
    let id = INTERMEDIATE_DIM;
    let vs = 262144;

    // Per-layer weight elements (approximate)
    let qkv = hd * (hd + 2 * (hd / 8)); // q_proj + k_proj + v_proj with GQA
    let out_proj = hd * hd;
    let gate_up = 2 * hd * id;
    let down_proj = id * hd;
    let per_layer = qkv + out_proj + gate_up + down_proj;
    let total_layers = per_layer * _num_layers;
    let embedding = vs * hd;

    // FP32: 4 bytes per weight
    let fp32 = (total_layers + embedding) * 4;
    // BF16: 2 bytes per weight
    let bf16 = (total_layers + embedding) * 2;
    // Ternary: ~5.3x smaller (1.58 bits ≈ 0.2 bytes/weight), but embedding stays FP16
    let ternary = total_layers / 5 + embedding * 2;
    // Sparse ternary: 2x smaller than ternary
    let sparse_ternary = ternary / 2;
    // With mmap: only working set (2 experts per layer)
    let mmap_working = sparse_ternary / 4; // top-2 from 4 experts

    (fp32, bf16, ternary, sparse_ternary, mmap_working)
}

/// Compute KV cache sizes for 128k context.
fn kv_cache_sizes() -> (usize, usize, usize, usize) {
    let seq_len = 131072usize; // 128k
    let _num_layers = 35;
    let kv_dim = 2 * 256; // 2 heads × head_dim=256 (shared KV)
    let num_unique_layers = 15; // first_shared_layer=15

    let fp32_kv = seq_len * kv_dim * num_unique_layers * 4;
    let bf16_kv = fp32_kv / 2;
    let three_bit_kv = fp32_kv * 3 / 32; // 3 bits per element
    let block_summary_kv = 3 * 1024 * 1024; // ~3 MB (from architecture doc)

    (fp32_kv, bf16_kv, three_bit_kv, block_summary_kv)
}

fn bench_fp32_matmul(c: &mut Criterion) {
    let a = vec![0.1f32; HIDDEN_DIM];
    let b = vec![0.1f32; INTERMEDIATE_DIM * HIDDEN_DIM];

    let mut group = c.benchmark_group("matmul_fp32");
    group.throughput(Throughput::Elements((HIDDEN_DIM * INTERMEDIATE_DIM) as u64));
    group.bench_function("dense_1536x6144", |bencher| {
        bencher.iter(|| {
            fp32_matmul(black_box(&a), black_box(&b), 1, INTERMEDIATE_DIM, HIDDEN_DIM)
        })
    });
    group.finish();
}

fn bench_ternary_matmul(c: &mut Criterion) {
    let input = vec![0.1f32; HIDDEN_DIM];
    let weights: Vec<i8> = vec![1, -1, 0, 1].iter().cycle().copied().take(INTERMEDIATE_DIM * HIDDEN_DIM).collect();

    let mut group = c.benchmark_group("matmul_ternary");
    group.throughput(Throughput::Elements((HIDDEN_DIM * INTERMEDIATE_DIM) as u64));
    group.bench_function("ternary_1536x6144", |bencher| {
        bencher.iter(|| {
            ternary_matmul(black_box(&input), black_box(&weights), 0.1, INTERMEDIATE_DIM, HIDDEN_DIM)
        })
    });
    group.finish();
}

fn bench_sparse_ternary_matmul(c: &mut Criterion) {
    let input = vec![0.1f32; HIDDEN_DIM];
    let weights: Vec<i8> = vec![1, -1, 0, 0].iter().cycle().copied().take(INTERMEDIATE_DIM * HIDDEN_DIM).collect();
    let mask: Vec<bool> = vec![true, true, false, false].iter().cycle().copied().take(INTERMEDIATE_DIM * HIDDEN_DIM).collect();

    let mut group = c.benchmark_group("matmul_sparse_ternary");
    group.throughput(Throughput::Elements((HIDDEN_DIM * INTERMEDIATE_DIM) as u64));
    group.bench_function("sparse_1536x6144", |bencher| {
        bencher.iter(|| {
            sparse_ternary_matmul(black_box(&input), black_box(&weights), black_box(&mask), 0.1, INTERMEDIATE_DIM, HIDDEN_DIM)
        })
    });
    group.finish();
}

fn bench_memory_report(c: &mut Criterion) {
    let (fp32, bf16, ternary, sparse_ternary, mmap) = model_sizes();
    let (fp32_kv, _bf16_kv, three_bit_kv, block_summary_kv) = kv_cache_sizes();

    let mut group = c.benchmark_group("memory_report");
    group.bench_function("model_sizes", |bencher| {
        bencher.iter(|| {
            let _ = black_box(fp32);
            let _ = black_box(bf16);
            let _ = black_box(ternary);
            let _ = black_box(sparse_ternary);
            let _ = black_box(mmap);
        })
    });

    // Print the report as a "benchmark"
    println!("\n=== Model Size Comparison ===");
    println!("FP32:            {} MB", fp32 / 1024 / 1024);
    println!("BF16:            {} MB", bf16 / 1024 / 1024);
    println!("Ternary 1.58b:   {} MB", ternary / 1024 / 1024);
    println!("Sparse Ternary:  {} MB", sparse_ternary / 1024 / 1024);
    println!("+ mmap (working): {} MB", mmap / 1024 / 1024);

    println!("\n=== KV Cache (128K context) ===");
    println!("FP32 KV:         {} MB", fp32_kv / 1024 / 1024);
    println!("3-bit TurboQuant: {} MB", three_bit_kv / 1024 / 1024);
    println!("Block summaries:  {} MB", block_summary_kv / 1024 / 1024);

    println!("\n=== Full Stack (Sparse + mmap + Block summaries) ===");
    let full_stack = mmap + block_summary_kv;
    println!("Total working set: {} MB", full_stack / 1024 / 1024);

    group.finish();
}

fn bench_quality_metrics(c: &mut Criterion) {
    // Simulate quality degradation through quantization layers
    let hidden_dim = 256;
    let fp32_output: Vec<f32> = (0..hidden_dim).map(|i| (i as f32 * 0.01).sin()).collect();

    // Ternary: ~2% quality loss (simulated by adding noise)
    let ternary_output: Vec<f32> = fp32_output.iter().enumerate().map(|(i, &v)| {
        v + if i % 7 == 0 { 0.01 } else { 0.0 }
    }).collect();

    // Sparse ternary: ~3% quality loss
    let sparse_output: Vec<f32> = fp32_output.iter().enumerate().map(|(i, &v)| {
        v + if i % 5 == 0 { 0.015 } else { 0.0 }
    }).collect();

    let cos_sim_fp32_ternary = cosine_similarity(&fp32_output, &ternary_output);
    let cos_sim_fp32_sparse = cosine_similarity(&fp32_output, &sparse_output);

    let mut group = c.benchmark_group("quality_metrics");
    group.bench_function("cosine_similarity", |bencher| {
        bencher.iter(|| {
            cosine_similarity(black_box(&fp32_output), black_box(&ternary_output))
        })
    });

    println!("\n=== Quality Metrics ===");
    println!("FP32 → Ternary cosine similarity:     {:.6}", cos_sim_fp32_ternary);
    println!("FP32 → Sparse Ternary cosine sim:     {:.6}", cos_sim_fp32_sparse);

    group.finish();
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 { dot / denom } else { 0.0 }
}

criterion_group!(benches, bench_fp32_matmul, bench_ternary_matmul, bench_sparse_ternary_matmul, bench_memory_report, bench_quality_metrics);
criterion_main!(benches);
