//! Prefill throughput benchmark for the CPU BlockAttnRes model.
//!
//! Deterministic, allocation-reusing timing loop (no criterion): prefill is
//! seconds per call at the Small shape, and criterion's sample count would
//! turn the run into an hour. The harness prints, for each prompt length:
//!
//!   * median wall time and tokens/s,
//!   * effective GFLOP/s from `2 × active_macs × tokens / seconds`,
//!   * the per-stage breakdown when the model exposes a profile.
//!
//! `PREFILL_SEQS=512,2048` selects prompt lengths; `PREFILL_REPS` overrides the
//! per-length repetition count (default: 3 at 512, 2 at 2048). `PREFILL_DUMP`
//! writes the last-position logits of a 128-token prefill to that path as raw
//! little-endian f32, so before/after checks are a byte-range diff rather than
//! a re-run.
//!
//! Run with the shared build lane:
//!   flock -w 7200 /tmp/opencode/build.lock env \
//!     CARGO_TARGET_DIR=/tmp/opencode/target-prefill CARGO_BUILD_JOBS=4 \
//!     cargo bench --bench prefill --no-default-features
//!
//! The rayon pool size is forced to 4 and printed; the host must be idle or
//! the numbers describe the load, not the code.

use std::time::{Duration, Instant};

use ferrisres::inference::student_kv_cache::ModelKVCache;
use ferrisres::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel};
use ferrisres::model::cpu_linear::{CpuLinear, CpuRmsNorm};
use ferrisres::model::cpu_moe::{CpuMoELayer, TernaryExpert};
use ferrisres::model::ternary::pack_ternary;

const HIDDEN: usize = 768;
const LAYERS: usize = 8;
const HEADS: usize = 8;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 96;
const INTERMEDIATE: usize = 1024;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const VOCAB: usize = 1024;
const LAYERS_PER_BLOCK: usize = 4;

/// SplitMix64: deterministic, no dependency, reproducible across runs.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Ternary value in {-1, 0, +1} with roughly equal mass.
    fn ternary(&mut self) -> i8 {
        ((self.next_u64() >> 33) % 3) as i8 - 1
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn ternary_linear(rng: &mut Rng, in_features: usize, out_features: usize) -> CpuLinear {
    let values: Vec<i8> = (0..in_features * out_features).map(|_| rng.ternary()).collect();
    CpuLinear::from_ternary(values, 0.02, in_features, out_features)
}

fn ternary_expert(rng: &mut Rng, rows: usize, cols: usize) -> TernaryExpert {
    let values: Vec<i8> = (0..rows * cols).map(|_| rng.ternary()).collect();
    TernaryExpert { packed: pack_ternary(&values), values, scale: 0.02, rows, cols }
}

/// Small-shaped configuration: hidden 768, 8 layers, 8/2 GQA heads,
/// 4-expert top-2 MoE. About 54M active parameters per token (the router and
/// the top-2 experts only), which is the shape the 111M Small model takes on
/// an MoE prefill.
fn small_model() -> CpuBlockAttnResModel {
    let mut rng = Rng(0x5EED_1111);
    let hd = HIDDEN;
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;

    let layers: Vec<CpuBlockAttnResLayer> = (0..LAYERS)
        .map(|idx| {
            let mut layer = CpuBlockAttnResLayer::new(hd, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE);
            layer.layer_number = idx;
            layer.q_proj = ternary_linear(&mut rng, hd, q_dim);
            layer.k_proj = ternary_linear(&mut rng, hd, kv_dim);
            layer.v_proj = ternary_linear(&mut rng, hd, kv_dim);
            layer.out_proj = ternary_linear(&mut rng, q_dim, hd);
            layer.attn_norm = CpuRmsNorm::from_weight(
                (0..hd).map(|_| 1.0 + rng.unit() * 0.01).collect(),
                1e-6,
            );
            layer.post_attn_norm = CpuRmsNorm::new(hd, 1e-6);
            layer.pre_ffn_norm = CpuRmsNorm::new(hd, 1e-6);
            layer.post_ffn_norm = CpuRmsNorm::new(hd, 1e-6);
            layer.q_norm = CpuRmsNorm::new(HEAD_DIM, 1e-6);
            layer.k_norm = CpuRmsNorm::new(HEAD_DIM, 1e-6);
            layer.v_norm = CpuRmsNorm::new(HEAD_DIM, 1e-6);
            layer.ffn_gate = None;
            layer.ffn_up = None;
            layer.ffn_down = None;

            let mut moe = CpuMoELayer::new(hd, INTERMEDIATE, EXPERTS, TOP_K);
            moe.use_gelu = true;
            moe.gate_weights = (0..EXPERTS * hd).map(|_| rng.unit() * 0.1).collect();
            moe.expert_gate =
                (0..EXPERTS).map(|_| ternary_expert(&mut rng, INTERMEDIATE, hd)).collect();
            moe.expert_up =
                (0..EXPERTS).map(|_| ternary_expert(&mut rng, INTERMEDIATE, hd)).collect();
            moe.expert_down =
                (0..EXPERTS).map(|_| ternary_expert(&mut rng, hd, INTERMEDIATE)).collect();
            layer.moe = Some(moe);
            layer
        })
        .collect();

    let boundaries: Vec<usize> = (0..LAYERS)
        .filter(|l| (l + 1) % LAYERS_PER_BLOCK == 0)
        .map(|l| l)
        .collect();

    CpuBlockAttnResModel {
        layers,
        embed_tokens: (0..VOCAB * hd).map(|_| rng.unit() * 0.02).collect(),
        lm_head: (0..hd * VOCAB).map(|_| rng.unit() * 0.02).collect(),
        final_norm: vec![1.0f32; hd],
        hidden_dim: hd,
        vocab_size: VOCAB,
        num_layers: LAYERS,
        final_logit_softcapping: Some(30.0),
        ple_model_projection: None,
        ple_projection_norm: None,
        embed_tokens_per_layer: None,
        hidden_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        block_config: BlockConfig {
            num_blocks: LAYERS / LAYERS_PER_BLOCK,
            layers_per_block: LAYERS_PER_BLOCK,
            boundary_layers: boundaries,
            attn_res_proj: vec![0.0; hd * hd],
            attn_res_norm: vec![1.0; hd],
        },
        lora_manager: None,
    }
}

/// Multiply-accumulates the prefill actually performs, per token.
fn active_macs_per_token() -> u64 {
    let hd = HIDDEN as u64;
    let q_dim = (HEADS * HEAD_DIM) as u64;
    let kv_dim = (KV_HEADS * HEAD_DIM) as u64;
    let id = INTERMEDIATE as u64;
    let per_layer_attn = hd * q_dim + 2 * hd * kv_dim + q_dim * hd;
    let per_layer_moe =
        TOP_K as u64 * (hd * id + hd * id + id * hd) + hd * EXPERTS as u64;
    (per_layer_attn + per_layer_moe) * LAYERS as u64 + hd * VOCAB as u64
}

fn fresh_cache(model: &CpuBlockAttnResModel, seq: usize) -> ModelKVCache {
    let kv_dim = KV_HEADS * HEAD_DIM;
    let mut cache = ModelKVCache::new(
        model.num_layers,
        model.hidden_dim,
        &vec![kv_dim; model.num_layers],
        LAYERS_PER_BLOCK,
        model.hidden_size_per_layer_input,
    );
    // Pre-reserve so append_batch does not measure reallocation growth.
    for layer in cache.layers.iter_mut() {
        layer.k.reserve(seq * kv_dim);
        layer.v.reserve(seq * kv_dim);
    }
    cache
}

fn tokens(seq: usize) -> Vec<u32> {
    let mut rng = Rng(0x70_0E_05 + seq as u64);
    (0..seq).map(|_| (rng.next_u64() >> 40) as u32 % VOCAB as u32).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Absolute kernel rates, so a slow prefill can be attributed to the kernel
/// rather than to the model wiring: one representative MoE projection
/// (1024×768) over one token and over 64 tokens, serial and parallel, plus an
/// fp32 `matrixmultiply` GEMM of the same shape for the machine's ceiling.
fn micro(pool: &rayon::ThreadPool) {
    use ferrisres::model::gemma_mapper::matmul;
    use ferrisres::model::ternary::{ternary_matmul, ternary_matmul_parallel};

    let (rows, cols) = (INTERMEDIATE, HIDDEN);
    let mut rng = Rng(0x1CE_0F00D);
    let values: Vec<i8> = (0..rows * cols).map(|_| rng.ternary()).collect();
    let fp32_weights: Vec<f32> = values.iter().map(|&v| v as f32 * 0.02).collect();

    for &seq in &[1usize, 8, 64] {
        let input: Vec<f32> = (0..seq * cols).map(|_| rng.unit()).collect();
        let macs = (rows * cols * seq) as f64;

        let run = |label: &str, f: &(dyn Fn() -> Vec<f32> + Sync)| {
            let reps = if seq == 1 { 5 } else { 3 };
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let start = Instant::now();
                let out = pool.install(f);
                let s = start.elapsed().as_secs_f64();
                std::hint::black_box(out);
                best = best.min(s);
            }
            println!(
                "micro {label:<28} seq={seq:<3} {:.3} ms {:>7.2} GFLOP/s",
                best * 1e3,
                2.0 * macs / best / 1e9
            );
        };

        run("ternary serial", &|| ternary_matmul(&values, 0.02, &input, rows, cols, seq));
        run("ternary parallel", &|| {
            ternary_matmul_parallel(&values, 0.02, &input, rows, cols, seq)
        });
        run("fp32 matrixmultiply", &|| matmul(&input, &fp32_weights, seq, cols, rows));
    }

    // The MoE prefill grouping, in isolation. One layer at the bench shape:
    // serial per-token experts against batched-per-expert.
    let mut rng = Rng(0x0E_0100);
    let moe = {
        let mut moe = CpuMoELayer::new(HIDDEN, INTERMEDIATE, EXPERTS, TOP_K);
        moe.use_gelu = true;
        moe.gate_weights = (0..EXPERTS * HIDDEN).map(|_| rng.unit() * 0.1).collect();
        moe.expert_gate =
            (0..EXPERTS).map(|_| ternary_expert(&mut rng, INTERMEDIATE, HIDDEN)).collect();
        moe.expert_up =
            (0..EXPERTS).map(|_| ternary_expert(&mut rng, INTERMEDIATE, HIDDEN)).collect();
        moe.expert_down =
            (0..EXPERTS).map(|_| ternary_expert(&mut rng, HIDDEN, INTERMEDIATE)).collect();
        moe
    };

    for &seq in &[1usize, 16, 64, 128, 512] {
        let input: Vec<f32> = (0..seq * HIDDEN).map(|_| rng.unit()).collect();
        let rows = (seq * TOP_K * 3 * HIDDEN * INTERMEDIATE) as f64;
        let reps = if seq >= 128 { 2 } else { 3 };
        let time = |f: &(dyn Fn() -> Vec<f32> + Sync)| {
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let start = Instant::now();
                std::hint::black_box(pool.install(f));
                best = best.min(start.elapsed().as_secs_f64());
            }
            best
        };
        let serial = time(&|| moe.forward(&input, seq));
        let batched = time(&|| moe.forward_prefill(&input, seq));
        println!(
            "micro moe serial           seq={seq:<3} {:.3} ms {:>7.2} GFLOP/s",
            serial * 1e3,
            2.0 * rows / serial / 1e9
        );
        println!(
            "micro moe batched          seq={seq:<3} {:.3} ms {:>7.2} GFLOP/s speedup={:.2}x",
            batched * 1e3,
            2.0 * rows / batched / 1e9,
            serial / batched
        );
    }
}

fn main() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("rayon pool");
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);

    if std::env::var("PREFILL_MICRO").is_ok() {
        micro(&pool);
        return;
    }

    let seqs: Vec<usize> = std::env::var("PREFILL_SEQS")
        .unwrap_or_else(|_| "512,2048".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    eprintln!("building Small-shaped model ({} layers, hidden {})", LAYERS, HIDDEN);
    let build_start = Instant::now();
    let model = small_model();
    eprintln!("model built in {:.2}s", build_start.elapsed().as_secs_f64());

    let macs = active_macs_per_token();
    let flops_per_token = 2 * macs;
    println!(
        "config: hidden={HIDDEN} layers={LAYERS} heads={HEADS}/{KV_HEADS} head_dim={HEAD_DIM} \
         moe={EXPERTS}x top{TOP_K} inter={INTERMEDIATE} vocab={VOCAB}"
    );
    println!(
        "active_macs/token={macs} ({:.1}M params) flops/token={:.1}M",
        macs as f64 / 1e6,
        flops_per_token as f64 / 1e6
    );
    println!(
        "rayon_threads={} machine_cores={cores}",
        pool.current_num_threads()
    );

    if let Ok(path) = std::env::var("PREFILL_DUMP") {
        let toks = tokens(128);
        let mut cache = fresh_cache(&model, toks.len());
        let logits = pool.install(|| model.forward_prefill(&toks, &mut cache));
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).expect("write logits dump");
        println!("dumped 128-token logits ({} f32) to {path}", logits.len());
    }

    for &seq in &seqs {
        let reps = env_usize("PREFILL_REPS", if seq <= 512 { 3 } else { 2 });
        let toks = tokens(seq);

        // One untimed warm-up per length: page-in weights, grow allocators.
        let mut cache = fresh_cache(&model, seq);
        let warm = Instant::now();
        let _ = pool.install(|| model.forward_prefill(&toks, &mut cache));
        eprintln!("seq={seq} warm-up {:.2}s", warm.elapsed().as_secs_f64());

        let mut times = Vec::with_capacity(reps);
        let mut stage_totals: Option<StageTotals> = None;
        for rep in 0..reps {
            let mut cache = fresh_cache(&model, seq);
            let start = Instant::now();
            let (logits, profile) =
                pool.install(|| model.forward_prefill_profiled(&toks, &mut cache));
            let elapsed = start.elapsed();
            std::hint::black_box(logits);
            times.push(elapsed.as_secs_f64());
            eprintln!(
                "seq={seq} rep={rep} {:.3}s {:.2} tok/s",
                elapsed.as_secs_f64(),
                seq as f64 / elapsed.as_secs_f64()
            );
            stage_totals = Some(match stage_totals {
                None => StageTotals::from(profile),
                Some(acc) => acc + StageTotals::from(profile),
            });
        }

        let med = median(times.clone());
        let tps = seq as f64 / med;
        let gflops = flops_per_token as f64 * seq as f64 / med / 1e9;
        println!(
            "RESULT seq={seq} median_s={med:.3} tokens_per_s={tps:.2} gflops={gflops:.2} \
             min_s={:.3} max_s={:.3} reps={reps}",
            times.iter().cloned().fold(f64::INFINITY, f64::min),
            times.iter().cloned().fold(0.0f64, f64::max),
        );
        if let Some(totals) = stage_totals {
            totals.report(seq, reps as f64);
        }
    }
}

#[derive(Default, Clone, Copy)]
struct StageTotals {
    embedding: Duration,
    ple: Duration,
    attention: Duration,
    ffn: Duration,
    inter_block: Duration,
    kv_cache: Duration,
    lm_head: Duration,
}

impl From<ferrisres::model::cpu_block_attn_res::PrefillProfile> for StageTotals {
    fn from(p: ferrisres::model::cpu_block_attn_res::PrefillProfile) -> Self {
        Self {
            embedding: p.embedding,
            ple: p.ple,
            attention: p.attention,
            ffn: p.ffn,
            inter_block: p.inter_block,
            kv_cache: p.kv_cache,
            lm_head: p.lm_head,
        }
    }
}

impl std::ops::Add for StageTotals {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            embedding: self.embedding + rhs.embedding,
            ple: self.ple + rhs.ple,
            attention: self.attention + rhs.attention,
            ffn: self.ffn + rhs.ffn,
            inter_block: self.inter_block + rhs.inter_block,
            kv_cache: self.kv_cache + rhs.kv_cache,
            lm_head: self.lm_head + rhs.lm_head,
        }
    }
}

impl StageTotals {
    fn report(&self, seq: usize, reps: f64) {
        println!("STAGE seq={seq} (per prefill)");
        let rows = [
            ("embedding", self.embedding),
            ("ple", self.ple),
            ("attention", self.attention),
            ("ffn", self.ffn),
            ("inter_block", self.inter_block),
            ("kv_cache", self.kv_cache),
            ("lm_head", self.lm_head),
        ];
        let mut sum = 0.0;
        for (_, d) in rows {
            sum += d.as_secs_f64();
        }
        for (name, d) in rows {
            let s = d.as_secs_f64() / reps;
            println!(
                "  {name:<12} {:.3} ms  {:>5.1}%",
                s * 1e3,
                if sum > 0.0 { s * reps / sum * 100.0 } else { 0.0 }
            );
        }
    }
}
