//! CPU <-> GPU equivalence for the Block AttnRes generation path.
//!
//! These tests need a live wgpu adapter, so they are `#[ignore]` by default.
//! On a GPU host run:
//!
//! ```text
//! cargo test --test gpu_cpu_equivalence -- --ignored --nocapture
//! ```
//!
//! They construct one model, load the *same* weights into
//! [`ferrisres::model::cpu_block_attn_res::CpuBlockAttnResModel`] and into the
//! GPU [`ferrisres::model::BlockAttnResModel`], and compare logits.
//!
//! Weights are uploaded through [`ferrisres::model::upload`], which
//! dequantises the CPU ternary weights and transposes them into the GPU matmul
//! layout, so both sides run the same arithmetic. Norm scales, the inter-block
//! key norm (`attn_res_norm`), `layer_scalar` and the final norm are all
//! loaded, and `layer_scalar` is deliberately not 1.0 so a misplaced scalar
//! cannot pass.

use ferrisres::compute::kernels::elementwise::ElementWiseOp;
use ferrisres::compute::kernels::rmsnorm::RmsNormOp;
use ferrisres::compute::{GpuBuffer, WgpuCompute};
use ferrisres::inference::kv_cache::ModelKVCache;
use ferrisres::model::block_attn_res::BlockAttnResLayer;
use ferrisres::model::cpu_block_attn_res::{
    BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel,
};
use ferrisres::model::cpu_linear::{CpuLinear, CpuRmsNorm};
use ferrisres::model::upload::{upload_attn_res_norm, upload_layer};
use ferrisres::model::{BlockAttnResConfig, BlockAttnResModel, LMHead, TokenEmbedding};
use std::sync::Arc;

const HIDDEN: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 16;
const INTERMEDIATE: usize = 64;
const VOCAB: usize = 16;
const NUM_BLOCKS: usize = 2;
const BLOCK_SIZE: usize = 2;
const LAYER_SCALAR: f32 = 0.9;
const TOKENS: [u32; 5] = [1, 3, 5, 7, 2];

/// Deterministic, dependency-free PRNG (LCG).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u32 << 24) as f32)
    }

    fn centered(&mut self, scale: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * scale
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.centered(scale)).collect()
    }

    fn near_one(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + self.centered(0.15)).collect()
    }
}

fn model_config() -> BlockAttnResConfig {
    let mut config = BlockAttnResConfig::new(HIDDEN);
    config.num_blocks = NUM_BLOCKS;
    config.block_size = BLOCK_SIZE;
    config.num_layers = NUM_BLOCKS * BLOCK_SIZE;
    config.attention_heads = HEADS;
    config.intermediate_dim = INTERMEDIATE;
    config.use_moe = false;
    config
}

fn build_cpu_model() -> CpuBlockAttnResModel {
    let mut rng = Rng::new(0x5eed_1234);
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let mut layers = Vec::new();

    for _ in 0..(NUM_BLOCKS * BLOCK_SIZE) {
        let mut layer = CpuBlockAttnResLayer::new(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE);
        layer.use_gelu = true;
        layer.rope_theta = 10000.0;
        layer.partial_rotary_factor = 1.0;
        layer.layer_scalar = LAYER_SCALAR;
        layer.attn_norm = CpuRmsNorm::from_weight(rng.near_one(HIDDEN), 1e-6);
        layer.q_proj = CpuLinear::from_weight(rng.vec(HIDDEN * q_dim, 0.6), HIDDEN, q_dim);
        layer.k_proj = CpuLinear::from_weight(rng.vec(HIDDEN * kv_dim, 0.6), HIDDEN, kv_dim);
        layer.v_proj = CpuLinear::from_weight(rng.vec(HIDDEN * kv_dim, 0.6), HIDDEN, kv_dim);
        layer.out_proj = CpuLinear::from_weight(rng.vec(q_dim * HIDDEN, 0.6), q_dim, HIDDEN);
        layer.q_norm = CpuRmsNorm::from_weight(rng.near_one(HEAD_DIM), 1e-6);
        layer.k_norm = CpuRmsNorm::from_weight(rng.near_one(HEAD_DIM), 1e-6);
        layer.post_attn_norm = CpuRmsNorm::from_weight(rng.near_one(HIDDEN), 1e-6);
        layer.pre_ffn_norm = CpuRmsNorm::from_weight(rng.near_one(HIDDEN), 1e-6);
        layer.ffn_gate =
            Some(CpuLinear::from_weight(rng.vec(HIDDEN * INTERMEDIATE, 0.6), HIDDEN, INTERMEDIATE));
        layer.ffn_up =
            Some(CpuLinear::from_weight(rng.vec(HIDDEN * INTERMEDIATE, 0.6), HIDDEN, INTERMEDIATE));
        layer.ffn_down =
            Some(CpuLinear::from_weight(rng.vec(INTERMEDIATE * HIDDEN, 0.6), INTERMEDIATE, HIDDEN));
        layer.post_ffn_norm = CpuRmsNorm::from_weight(rng.near_one(HIDDEN), 1e-6);
        layers.push(layer);
    }

    let boundary_layers = (0..NUM_BLOCKS)
        .map(|b| (b + 1) * BLOCK_SIZE - 1)
        .collect::<Vec<_>>();

    CpuBlockAttnResModel {
        layers,
        embed_tokens: rng.vec(VOCAB * HIDDEN, 0.5),
        lm_head: rng.vec(HIDDEN * VOCAB, 0.5),
        final_norm: rng.near_one(HIDDEN),
        hidden_dim: HIDDEN,
        vocab_size: VOCAB,
        num_layers: NUM_BLOCKS * BLOCK_SIZE,
        final_logit_softcapping: None,
        ple_model_projection: None,
        ple_projection_norm: None,
        embed_tokens_per_layer: None,
        hidden_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        block_config: BlockConfig {
            num_blocks: NUM_BLOCKS,
            layers_per_block: BLOCK_SIZE,
            boundary_layers,
            attn_res_proj: vec![0.0; HIDDEN * HIDDEN],
            attn_res_norm: rng.near_one(HIDDEN),
        },
        lora_manager: None,
    }
}

fn build_gpu_model(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    cpu: &CpuBlockAttnResModel,
) -> BlockAttnResModel {
    let config = model_config();
    let mut layers = Vec::new();
    for i in 0..config.total_layers() {
        let mut layer = BlockAttnResLayer::new_with_params(
            Arc::clone(device),
            Arc::clone(queue),
            &config,
            i,
            KV_HEADS,
            HEAD_DIM,
            INTERMEDIATE,
            10000.0,
            1.0,
            true,  // GELU, as the CPU reference
            false, // not KV-shared
        )
        .expect("create GPU layer");
        upload_layer(&layer, &cpu.layers[i], queue);
        upload_attn_res_norm(&layer, &cpu.block_config.attn_res_norm, queue);
        layer.set_layer_scalar(LAYER_SCALAR);
        layers.push(layer);
    }
    BlockAttnResModel::from_layers(
        Arc::clone(device),
        Arc::clone(queue),
        config,
        layers,
    )
    .expect("create GPU model")
}

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &GpuBuffer) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("equivalence_readback"),
        size: buffer.size() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("equivalence_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer.buffer(), 0, &staging, 0, buffer.size() as u64);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    out
}

struct GpuFixture {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    model: BlockAttnResModel,
    embedding: TokenEmbedding,
    lm_head: LMHead,
    final_norm: RmsNormOp,
    elementwise: ElementWiseOp,
}

impl GpuFixture {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, cpu: &CpuBlockAttnResModel) -> Self {
        let model = build_gpu_model(&device, &queue, cpu);
        let embedding = TokenEmbedding::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            VOCAB,
            HIDDEN,
        )
        .expect("embedding");
        embedding.set_weight(&queue, &cpu.embed_tokens);
        let lm_head = LMHead::new(Arc::clone(&device), Arc::clone(&queue), HIDDEN, VOCAB)
            .expect("lm head");
        lm_head.set_weight(&queue, &cpu.lm_head);
        let final_norm = RmsNormOp::new(&device).expect("final norm");
        final_norm.set_weight(&queue, &cpu.final_norm);
        let elementwise = ElementWiseOp::new(&device, &queue);
        Self {
            device,
            queue,
            model,
            embedding,
            lm_head,
            final_norm,
            elementwise,
        }
    }

    fn embedding_for(&self, encoder: &mut wgpu::CommandEncoder, tokens: &[u32]) -> GpuBuffer {
        let bytes = tokens.len() * std::mem::size_of::<u32>();
        let ids = GpuBuffer::new(&self.device, bytes, Some("eq_ids")).unwrap();
        self.queue
            .write_buffer(ids.buffer(), 0, bytemuck::cast_slice(tokens));
        let hidden = GpuBuffer::new(
            &self.device,
            tokens.len() * HIDDEN * std::mem::size_of::<f32>(),
            Some("eq_embed"),
        )
        .unwrap();
        self.embedding
            .forward(encoder, &ids, &hidden, tokens.len() as u32)
            .unwrap();
        // CPU reference scales embeddings by sqrt(hidden).
        let scaled = GpuBuffer::new(
            &self.device,
            tokens.len() * HIDDEN * std::mem::size_of::<f32>(),
            Some("eq_embed_scaled"),
        )
        .unwrap();
        self.elementwise
            .dispatch_scale(
                encoder,
                &hidden,
                &scaled,
                (HIDDEN as f32).sqrt(),
                (tokens.len() * HIDDEN) as u32,
            )
            .unwrap();
        scaled
    }

    fn project_logits(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden: &GpuBuffer,
        rows: u32,
    ) -> GpuBuffer {
        let normed = GpuBuffer::new(
            &self.device,
            rows as usize * HIDDEN * std::mem::size_of::<f32>(),
            Some("eq_normed"),
        )
        .unwrap();
        self.final_norm
            .dispatch_with_eps(
                &self.device,
                &self.queue,
                encoder,
                hidden,
                &normed,
                rows,
                HIDDEN as u32,
                1e-6,
            )
            .unwrap();
        let logits = GpuBuffer::new(
            &self.device,
            rows as usize * VOCAB * std::mem::size_of::<f32>(),
            Some("eq_logits"),
        )
        .unwrap();
        self.lm_head.forward(encoder, &normed, &logits, rows).unwrap();
        logits
    }

    fn kv_cache(&self, max_seq: u32) -> ModelKVCache {
        ModelKVCache::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            self.model.layers().len() as u32,
            max_seq,
            HEADS as u32,
            HEAD_DIM as u32,
        )
        .expect("kv cache")
    }

    /// Full-sequence forward through the cached path: one prefill call over
    /// every token. This is the GPU's sequence model (causal attention across
    /// positions plus causal inter-block attention); `BlockAttnResModel::forward`
    /// is a per-position batch path with no cross-position attention and is not
    /// comparable to a sequence forward.
    fn full_logits(&self, tokens: &[u32]) -> Vec<f32> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eq_full"),
        });
        let embed = self.embedding_for(&mut encoder, tokens);
        let out = GpuBuffer::new(
            &self.device,
            tokens.len() * HIDDEN * std::mem::size_of::<f32>(),
            Some("eq_full_out"),
        )
        .unwrap();
        let kv = self.kv_cache(tokens.len() as u32 + 1);
        self.model
            .forward_prefill_cached(&mut encoder, &embed, &out, &kv, tokens.len() as u32)
            .unwrap();
        let logits = self.project_logits(&mut encoder, &out, tokens.len() as u32);
        self.queue.submit(std::iter::once(encoder.finish()));
        readback(&self.device, &self.queue, &logits)
    }

    /// Cached prefill on `prefill` tokens, then one decode step per remaining
    /// token. Returns logits for every position.
    fn cached_logits(&self, tokens: &[u32], prefill: usize) -> Vec<f32> {
        assert!(prefill >= 1 && prefill <= tokens.len());
        let kv = self.kv_cache(tokens.len() as u32 + 1);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eq_cached"),
        });

        let prompt = &tokens[..prefill];
        let embed = self.embedding_for(&mut encoder, prompt);
        let out = GpuBuffer::new(
            &self.device,
            prefill * HIDDEN * std::mem::size_of::<f32>(),
            Some("eq_prefill_out"),
        )
        .unwrap();
        let mut state = self
            .model
            .forward_prefill_cached(&mut encoder, &embed, &out, &kv, prefill as u32)
            .unwrap();
        let logits = self.project_logits(&mut encoder, &out, prefill as u32);
        self.queue.submit(std::iter::once(encoder.finish()));
        let mut all = readback(&self.device, &self.queue, &logits);

        for (pos, &token) in tokens.iter().enumerate().skip(prefill) {
            let mut encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("eq_decode"),
                    });
            let embed = self.embedding_for(&mut encoder, &[token]);
            let out = GpuBuffer::new(
                &self.device,
                HIDDEN * std::mem::size_of::<f32>(),
                Some("eq_decode_out"),
            )
            .unwrap();
            self.model
                .forward_decode_cached(
                    &mut encoder,
                    &embed,
                    &out,
                    &kv,
                    &mut state,
                    Some(pos as u32),
                )
                .unwrap();
            let logits = self.project_logits(&mut encoder, &out, 1);
            self.queue.submit(std::iter::once(encoder.finish()));
            all.extend(readback(&self.device, &self.queue, &logits));
        }
        all
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "logit vectors differ in length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[tokio::test]
#[ignore = "requires a GPU adapter; run with --ignored on a GPU host"]
async fn test_gpu_matches_cpu_logits() {
    let compute = WgpuCompute::new().await.expect("Failed to create WgpuCompute");
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let cpu = build_cpu_model();
    let gpu = GpuFixture::new(Arc::clone(&device), Arc::clone(&queue), &cpu);

    let cpu_logits = cpu.forward(&TOKENS);
    let gpu_logits = gpu.full_logits(&TOKENS);

    let diff = max_abs_diff(&cpu_logits, &gpu_logits);
    println!(
        "CPU<->GPU full-sequence max |logit diff| = {:.3e} ({} logits)",
        diff,
        cpu_logits.len()
    );
    assert!(
        diff <= 1e-4,
        "CPU/GPU logits differ by {:.3e} (limit 1e-4)",
        diff
    );
}

#[tokio::test]
#[ignore = "requires a GPU adapter; run with --ignored on a GPU host"]
async fn test_gpu_matches_cpu_prefill_decode() {
    let compute = WgpuCompute::new().await.expect("Failed to create WgpuCompute");
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let cpu = build_cpu_model();
    let gpu = GpuFixture::new(Arc::clone(&device), Arc::clone(&queue), &cpu);

    let cpu_logits = cpu.forward(&TOKENS);
    let gpu_cached = gpu.cached_logits(&TOKENS, 3);

    let diff = max_abs_diff(&cpu_logits, &gpu_cached);
    println!(
        "CPU vs GPU prefill(3)+decode(2) max |logit diff| = {:.3e}",
        diff
    );
    assert!(
        diff <= 1e-4,
        "CPU/GPU cached logits differ by {:.3e} (limit 1e-4)",
        diff
    );
}

#[tokio::test]
#[ignore = "requires a GPU adapter; run with --ignored on a GPU host"]
async fn test_gpu_prefill_decode_consistency() {
    let compute = WgpuCompute::new().await.expect("Failed to create WgpuCompute");
    let device = Arc::new(compute.device().clone());
    let queue = Arc::new(compute.queue().clone());

    let cpu = build_cpu_model();
    let gpu = GpuFixture::new(Arc::clone(&device), Arc::clone(&queue), &cpu);

    let full = gpu.full_logits(&TOKENS);
    let cached = gpu.cached_logits(&TOKENS, 2);

    let diff = max_abs_diff(&full, &cached);
    println!(
        "GPU full-forward vs prefill(2)+decode(3) max |logit diff| = {:.3e}",
        diff
    );
    assert!(
        diff <= 1e-4,
        "GPU full vs cached logits differ by {:.3e} (limit 1e-4)",
        diff
    );
}
