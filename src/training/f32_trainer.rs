//! f32 dense reference transformer with a hand-written backward pass.
//!
//! This module exists so that training can accumulate real gradients. The
//! production model stores its base weights as ternary `i8` ([`crate::model::
//! cpu_linear::CpuLinear`]) and its `backward` only recomputes activations, so
//! there is no f32 buffer to accumulate into. Here we keep a small,
//! self-contained, gemma-style pre-norm dense transformer in plain `f32`:
//!
//! * every parameter is a `Vec<f32>` (plus the scalar `layer_scalar`),
//! * the forward pass caches what the backward pass needs,
//! * the backward pass is written by hand (no autodiff crate, no `unsafe`),
//! * [`Adam`] updates the same parameter layout in place.
//!
//! Correctness is established by finite-difference gradient checks in the
//! tests at the bottom of this file. This is a *reference* model: it is
//! deliberately single-threaded and unoptimised. It is not wired into the CLI
//! and nothing is exported to ternary yet.
//!
//! # Architecture
//!
//! Gemma-4-style pre-norm decoder, `num_layers` identical layers, plus the
//! Block AttnRes mechanism of the CPU inference model
//! ([`crate::model::cpu_block_attn_res::CpuBlockAttnResModel`]):
//!
//! ```text
//! x = embed[token] * sqrt(hidden)
//! if blocks are configured:
//!     prefix[0][t] = sum_{s <= t} x_embed[s]        (per-position prefix sum)
//! for each layer:
//!     h  = rmsnorm(x, attn_norm)
//!     q  = h @ Wq ; k = h @ Wk ; v = h @ Wv
//!     q  = rmsnorm_per_head(q, q_norm) ; k = rmsnorm_per_head(k, k_norm)
//!     v  = rmsnorm_per_head(v)               (no learned scale)
//!     q  = rope(q) ; k = rope(k)
//!     a  = causal_attention(q, k, v)         (GQA, scale 1.0)
//!     y  = rmsnorm(a @ Wo, post_attn_norm)
//!     x  = x + y
//!     h2 = rmsnorm(x, pre_ffn_norm)
//!     f  = silu(h2 @ Wg) * (h2 @ Wu)
//!     y2 = rmsnorm(f @ Wd, post_ffn_norm)
//!     x  = (x + y2) * layer_scalar           (once per layer)
//!     if blocks are configured:
//!         prefix[b + 1][t] += sum_{s <= t} x[s]  (b = this layer's block)
//!         if layer is a block boundary:
//!             mean[e][t] = prefix[e][t] / ((t + 1) * divisor(e))
//!             x[t] += inter_block_attention(x[t], mean[..][t])
//! x = rmsnorm(x, final_norm)
//! logits = x @ lm_head
//! ```
//!
//! `inter_block_attention` uses `x[t]` as the query, the RMS-normed (with
//! `attn_res_norm`) causal means as keys and values, a softmax over the
//! entries, and returns the weighted sum added to `x[t]`. Entry 0 is the
//! initial embedding (divisor `t + 1`); entry `b + 1` is block `b` (divisor
//! `(t + 1) * layers_per_block`). The same rule is documented on
//! [`crate::model::cpu_block_attn_res`].
//!
//! Weight layout is `[in, out]`, row-major, so `y[o] = sum_i x[i] * W[i*out+o]`.
//!
//! # Conventions taken from the inference model
//!
//! The equivalence test at the bottom of this file forces these to match
//! [`crate::model::cpu_block_attn_res::CpuBlockAttnResModel::forward`]:
//!
//! * `post_attn_norm` / `post_ffn_norm` are applied to each sub-layer output
//!   *before* the residual add (as the inference model does).
//! * Attention scores are not scaled by `1/sqrt(head_dim)`: the per-head Q/K
//!   RMSNorm already fixes the scale, and the inference model uses `scale = 1`.
//! * V is RMS-normalised per head without a learned scale.
//! * `layer_scalar` multiplies the whole layer output once, after the FFN
//!   residual, exactly as the inference model does. The equivalence test uses a
//!   value other than 1.0 so a misplaced scalar cannot pass.

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

use crate::model::cpu_block_attn_res::{
    add_layer_prefix, block_index_of_layer, embedding_prefix, prefix_means,
};

/// Shape and numerical settings of the reference model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F32Config {
    pub hidden: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub max_seq: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

impl F32Config {
    /// Total query/output width: `num_heads * head_dim`.
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Total key/value width: `num_kv_heads * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    fn validate(&self) {
        assert!(self.hidden > 0, "hidden must be > 0");
        assert!(self.num_heads > 0, "num_heads must be > 0");
        assert!(self.num_kv_heads > 0, "num_kv_heads must be > 0");
        assert!(self.head_dim > 0 && self.head_dim % 2 == 0, "head_dim must be even");
        assert!(
            self.num_heads % self.num_kv_heads == 0,
            "num_heads ({}) must be a multiple of num_kv_heads ({})",
            self.num_heads,
            self.num_kv_heads
        );
        assert!(self.vocab > 0, "vocab must be > 0");
        assert!(self.max_seq > 0, "max_seq must be > 0");
        assert!(self.eps > 0.0, "eps must be > 0");
    }
}

// ---------------------------------------------------------------------------
// Block AttnRes configuration
// ---------------------------------------------------------------------------

/// Block partitioning for the Block AttnRes mechanism.
///
/// Mirrors [`crate::model::cpu_block_attn_res::BlockConfig`] minus
/// `attn_res_proj`, which the inference model carries but never uses. The
/// trainable norm lives in [`F32Model::attn_res_norm`].
#[derive(Debug, Clone, PartialEq)]
pub struct F32BlockConfig {
    /// Number of blocks (2 in the reference test config).
    pub num_blocks: usize,
    /// Layers per block. A block summary's divisor is
    /// `(t + 1) * layers_per_block`, because the prefix sum runs over both
    /// positions and the block's layers.
    pub layers_per_block: usize,
    /// Layer indices that end a block. Empty means "plain transformer".
    pub boundary_layers: Vec<usize>,
}

impl F32BlockConfig {
    /// No block boundaries: the model behaves exactly like the plain
    /// pre-norm transformer.
    pub fn plain() -> Self {
        Self {
            num_blocks: 0,
            layers_per_block: 0,
            boundary_layers: Vec::new(),
        }
    }

    /// True when `layer` ends a block.
    pub fn is_boundary(&self, layer: usize) -> bool {
        self.boundary_layers.contains(&layer)
    }

    /// Boundaries must end each block, because the forward divides the
    /// accumulated sum by exactly `layers_per_block`.
    fn validate(&self, num_layers: usize) {
        if self.boundary_layers.is_empty() {
            return;
        }
        assert!(
            self.layers_per_block > 0,
            "layers_per_block must be > 0 when boundary_layers is non-empty"
        );
        assert!(
            self.num_blocks >= self.boundary_layers.len(),
            "num_blocks ({}) must cover every boundary ({})",
            self.num_blocks,
            self.boundary_layers.len()
        );
        for (j, &b) in self.boundary_layers.iter().enumerate() {
            assert!(
                b < num_layers,
                "boundary layer {} outside num_layers {}",
                b,
                num_layers
            );
            assert_eq!(
                b,
                (j + 1) * self.layers_per_block - 1,
                "boundary {} must be the last layer of block {} (index {})",
                b,
                j,
                (j + 1) * self.layers_per_block - 1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// All f32 parameters of one transformer layer.
#[derive(Debug, Clone)]
pub struct LayerParams {
    pub attn_norm_w: Vec<f32>,      // [hidden]
    pub q: Vec<f32>,                // [hidden, q_dim]
    pub k: Vec<f32>,                // [hidden, kv_dim]
    pub v: Vec<f32>,                // [hidden, kv_dim]
    pub q_norm_w: Vec<f32>,         // [head_dim]
    pub k_norm_w: Vec<f32>,         // [head_dim]
    pub o: Vec<f32>,                // [q_dim, hidden]
    pub post_attn_norm_w: Vec<f32>, // [hidden]
    pub pre_ffn_norm_w: Vec<f32>,   // [hidden]
    pub gate: Vec<f32>,             // [hidden, intermediate]
    pub up: Vec<f32>,               // [hidden, intermediate]
    pub down: Vec<f32>,             // [intermediate, hidden]
    pub post_ffn_norm_w: Vec<f32>,  // [hidden]
    pub layer_scalar: f32,
}

/// The reference model: an embedding table, a stack of layers, a final norm
/// and an un-tied language-model head.
#[derive(Debug, Clone)]
pub struct F32Model {
    pub config: F32Config,
    pub block_config: F32BlockConfig,
    pub embed: Vec<f32>,         // [vocab, hidden]
    pub layers: Vec<LayerParams>,
    pub final_norm_w: Vec<f32>,  // [hidden]
    pub lm_head: Vec<f32>,       // [hidden, vocab]
    /// Inter-block attention norm, trainable even in the plain config (where
    /// it is unused and receives zero gradient).
    pub attn_res_norm: Vec<f32>, // [hidden]
}

/// Deterministic PRNG (splitmix64). Used only for initialisation and for
/// picking indices in the gradient check, so no dependency is needed.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 24 bits of precision.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[-scale, scale]`.
    fn next_signed(&mut self, scale: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * scale
    }
}

fn fill_uniform(v: &mut [f32], scale: f32, rng: &mut SplitMix64) {
    for x in v.iter_mut() {
        *x = rng.next_signed(scale);
    }
}

/// `1 / sqrt(fan_in)`, the init scale used for every linear weight.
fn init_scale(fan_in: usize) -> f32 {
    1.0 / (fan_in as f32).sqrt()
}

impl F32Model {
    /// Deterministic `1/sqrt(fan_in)` initialisation. Norm weights start at 1,
    /// `layer_scalar` at 1. The same seed always produces the same model.
    ///
    /// This is the plain pre-norm transformer: no block boundaries.
    pub fn new_random(config: F32Config, seed: u64) -> Self {
        Self::new_random_with_blocks(config, F32BlockConfig::plain(), seed)
    }

    /// Same as [`Self::new_random`], with an explicit Block AttnRes layout.
    pub fn new_random_with_blocks(
        config: F32Config,
        block_config: F32BlockConfig,
        seed: u64,
    ) -> Self {
        config.validate();
        block_config.validate(config.num_layers);
        let mut rng = SplitMix64::new(seed);
        let h = config.hidden;
        let q_dim = config.q_dim();
        let kv_dim = config.kv_dim();
        let inter = config.intermediate;

        let mut embed = vec![0.0; config.vocab * h];
        fill_uniform(&mut embed, init_scale(h), &mut rng);

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            let mut q = vec![0.0; h * q_dim];
            fill_uniform(&mut q, init_scale(h), &mut rng);
            let mut k = vec![0.0; h * kv_dim];
            fill_uniform(&mut k, init_scale(h), &mut rng);
            let mut v = vec![0.0; h * kv_dim];
            fill_uniform(&mut v, init_scale(h), &mut rng);
            let mut o = vec![0.0; q_dim * h];
            fill_uniform(&mut o, init_scale(q_dim), &mut rng);
            let mut gate = vec![0.0; h * inter];
            fill_uniform(&mut gate, init_scale(h), &mut rng);
            let mut up = vec![0.0; h * inter];
            fill_uniform(&mut up, init_scale(h), &mut rng);
            let mut down = vec![0.0; inter * h];
            fill_uniform(&mut down, init_scale(inter), &mut rng);

            layers.push(LayerParams {
                attn_norm_w: vec![1.0; h],
                q,
                k,
                v,
                q_norm_w: vec![1.0; config.head_dim],
                k_norm_w: vec![1.0; config.head_dim],
                o,
                post_attn_norm_w: vec![1.0; h],
                pre_ffn_norm_w: vec![1.0; h],
                gate,
                up,
                down,
                post_ffn_norm_w: vec![1.0; h],
                layer_scalar: 1.0,
            });
        }

        let mut lm_head = vec![0.0; h * config.vocab];
        fill_uniform(&mut lm_head, init_scale(h), &mut rng);

        Self {
            config,
            block_config,
            embed,
            layers,
            final_norm_w: vec![1.0; h],
            lm_head,
            attn_res_norm: vec![1.0; h],
        }
    }

    /// Number of scalar parameters, including `layer_scalar` and every norm.
    pub fn param_count(&self) -> usize {
        self.tensors().iter().map(|(_, t)| t.len()).sum()
    }

    /// Every parameter tensor in a stable order, as `(name, values)`.
    ///
    /// This is the single source of truth for the parameter layout: [`Adam`]
    /// and the gradient check both walk it, and [`Gradients::tensors`] returns
    /// the same order.
    pub fn tensors(&self) -> Vec<(String, &[f32])> {
        let mut out: Vec<(String, &[f32])> = Vec::new();
        out.push(("embed".to_string(), self.embed.as_slice()));
        for (l, lp) in self.layers.iter().enumerate() {
            let n = |suffix: &str| format!("layer{}.{}", l, suffix);
            out.push((n("attn_norm"), lp.attn_norm_w.as_slice()));
            out.push((n("q"), lp.q.as_slice()));
            out.push((n("k"), lp.k.as_slice()));
            out.push((n("v"), lp.v.as_slice()));
            out.push((n("q_norm"), lp.q_norm_w.as_slice()));
            out.push((n("k_norm"), lp.k_norm_w.as_slice()));
            out.push((n("o"), lp.o.as_slice()));
            out.push((n("post_attn_norm"), lp.post_attn_norm_w.as_slice()));
            out.push((n("pre_ffn_norm"), lp.pre_ffn_norm_w.as_slice()));
            out.push((n("gate"), lp.gate.as_slice()));
            out.push((n("up"), lp.up.as_slice()));
            out.push((n("down"), lp.down.as_slice()));
            out.push((n("post_ffn_norm"), lp.post_ffn_norm_w.as_slice()));
            out.push((n("layer_scalar"), std::slice::from_ref(&lp.layer_scalar)));
        }
        out.push(("final_norm".to_string(), self.final_norm_w.as_slice()));
        out.push(("lm_head".to_string(), self.lm_head.as_slice()));
        out.push(("attn_res_norm".to_string(), self.attn_res_norm.as_slice()));
        out
    }

    /// Mutable view of every parameter tensor, in the same order as
    /// [`Self::tensors`].
    pub fn tensors_mut(&mut self) -> Vec<(String, &mut [f32])> {
        let mut out: Vec<(String, &mut [f32])> = Vec::new();
        out.push(("embed".to_string(), self.embed.as_mut_slice()));
        for (l, lp) in self.layers.iter_mut().enumerate() {
            let n = |suffix: &str| format!("layer{}.{}", l, suffix);
            out.push((n("attn_norm"), lp.attn_norm_w.as_mut_slice()));
            out.push((n("q"), lp.q.as_mut_slice()));
            out.push((n("k"), lp.k.as_mut_slice()));
            out.push((n("v"), lp.v.as_mut_slice()));
            out.push((n("q_norm"), lp.q_norm_w.as_mut_slice()));
            out.push((n("k_norm"), lp.k_norm_w.as_mut_slice()));
            out.push((n("o"), lp.o.as_mut_slice()));
            out.push((n("post_attn_norm"), lp.post_attn_norm_w.as_mut_slice()));
            out.push((n("pre_ffn_norm"), lp.pre_ffn_norm_w.as_mut_slice()));
            out.push((n("gate"), lp.gate.as_mut_slice()));
            out.push((n("up"), lp.up.as_mut_slice()));
            out.push((n("down"), lp.down.as_mut_slice()));
            out.push((n("post_ffn_norm"), lp.post_ffn_norm_w.as_mut_slice()));
            out.push((
                n("layer_scalar"),
                std::slice::from_mut(&mut lp.layer_scalar),
            ));
        }
        out.push(("final_norm".to_string(), self.final_norm_w.as_mut_slice()));
        out.push(("lm_head".to_string(), self.lm_head.as_mut_slice()));
        out.push(("attn_res_norm".to_string(), self.attn_res_norm.as_mut_slice()));
        out
    }

    /// Read one element by `(tensor index, element index)` in the layout of
    /// [`Self::tensors`]. Used by the finite-difference gradient check.
    pub fn get_tensor_elem(&self, tensor: usize, index: usize) -> f32 {
        self.tensors()[tensor].1[index]
    }

    /// Write one element by `(tensor index, element index)`.
    pub fn set_tensor_elem(&mut self, tensor: usize, index: usize, value: f32) {
        let mut tensors = self.tensors_mut();
        tensors[tensor].1[index] = value;
    }

    // -----------------------------------------------------------------------
    // Forward
    // -----------------------------------------------------------------------

    /// Precompute RoPE cos/sin tables for positions `0..seq`.
    /// Returns two `[seq, head_dim/2]` tables.
    fn rope_tables(&self, seq: usize) -> (Vec<f32>, Vec<f32>) {
        let half = self.config.head_dim / 2;
        let mut cos = vec![0.0; seq * half];
        let mut sin = vec![0.0; seq * half];
        for t in 0..seq {
            let pos = t as f64;
            for i in 0..half {
                // Same convention as model::gemma_mapper::apply_rope.
                let freq = 1.0 / (self.config.rope_theta as f64).powf((2 * i) as f64 / self.config.head_dim as f64);
                let angle = pos * freq;
                cos[t * half + i] = angle.cos() as f32;
                sin[t * half + i] = angle.sin() as f32;
            }
        }
        (cos, sin)
    }

    /// Full forward pass. Returns `logits [seq, vocab]` and a cache holding
    /// every activation the backward pass needs.
    pub fn forward(&self, tokens: &[u32]) -> (Vec<f32>, ForwardCache) {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(seq > 0, "need at least one token");
        assert!(seq <= cfg.max_seq, "seq {} exceeds max_seq {}", seq, cfg.max_seq);
        for &tok in tokens {
            assert!((tok as usize) < cfg.vocab, "token {} outside vocab {}", tok, cfg.vocab);
        }

        let h = cfg.hidden;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate;
        let heads = cfg.num_heads;
        let kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let eps = cfg.eps;

        let sqrt_h = (h as f32).sqrt();
        let mut x = vec![0.0; seq * h];
        for t in 0..seq {
            let tok = tokens[t] as usize;
            for i in 0..h {
                x[t * h + i] = self.embed[tok * h + i] * sqrt_h;
            }
        }
        let x_embed = x.clone();

        // Causal Block AttnRes state. `prefix_sums[e]` is `[seq, hidden]`: row
        // t is entry e's raw prefix sum over positions 0..=t. Entry 0 is the
        // initial embedding; entry b+1 is block b's layer outputs. The plain
        // path allocates nothing, so it stays bit-identical to before.
        let has_blocks = !self.block_config.boundary_layers.is_empty();
        let boundaries_cfg = &self.block_config.boundary_layers;
        let lpb = self.block_config.layers_per_block.max(1);
        let num_entries = boundaries_cfg.len() + 1;
        let mut prefix_sums: Vec<Vec<f32>> = if has_blocks {
            vec![vec![0.0f32; seq * h]; num_entries]
        } else {
            Vec::new()
        };
        if has_blocks {
            prefix_sums[0] = embedding_prefix(&x, seq, h);
        }
        let mut boundaries: Vec<BoundaryCache> = Vec::new();

        let (cos, sin) = self.rope_tables(seq);
        let mut layer_caches = Vec::with_capacity(cfg.num_layers);

        for (layer_index, lp) in self.layers.iter().enumerate() {
            let x_in = x.clone();

            // --- attention block -------------------------------------------------
            let mut h_attn = vec![0.0; seq * h];
            rmsnorm_fwd(&x, &lp.attn_norm_w, eps, seq, h, &mut h_attn);

            let mut q_proj = vec![0.0; seq * q_dim];
            matmul(&h_attn, &lp.q, seq, h, q_dim, &mut q_proj);
            let mut k_proj = vec![0.0; seq * kv_dim];
            matmul(&h_attn, &lp.k, seq, h, kv_dim, &mut k_proj);
            let mut v_raw = vec![0.0; seq * kv_dim];
            matmul(&h_attn, &lp.v, seq, h, kv_dim, &mut v_raw);

            // Per-head V RMSNorm without a learned scale (inference model
            // convention: `per_head_rms_norm_no_scale`).
            let mut v = vec![0.0; seq * kv_dim];
            rmsnorm_no_scale_fwd(&v_raw, eps, seq * kv_heads, head_dim, &mut v);

            let mut q = vec![0.0; seq * q_dim];
            rmsnorm_fwd(&q_proj, &lp.q_norm_w, eps, seq * heads, head_dim, &mut q);
            let mut k = vec![0.0; seq * kv_dim];
            rmsnorm_fwd(&k_proj, &lp.k_norm_w, eps, seq * kv_heads, head_dim, &mut k);

            apply_rope(&mut q, seq, heads, head_dim, &cos, &sin);
            apply_rope(&mut k, seq, kv_heads, head_dim, &cos, &sin);

            let mut attn_probs = vec![0.0; heads * seq * seq];
            let mut attn_concat = vec![0.0; seq * q_dim];
            attention_fwd(
                &q, &k, &v, seq, heads, kv_heads, head_dim, &mut attn_probs, &mut attn_concat,
            );

            let mut attn_out = vec![0.0; seq * h];
            matmul(&attn_concat, &lp.o, seq, q_dim, h, &mut attn_out);
            let mut post_attn_normed = vec![0.0; seq * h];
            rmsnorm_fwd(&attn_out, &lp.post_attn_norm_w, eps, seq, h, &mut post_attn_normed);

            let mut x_mid = vec![0.0; seq * h];
            for i in 0..seq * h {
                x_mid[i] = x[i] + post_attn_normed[i];
            }

            // --- feed-forward block ----------------------------------------------
            let mut h_ffn = vec![0.0; seq * h];
            rmsnorm_fwd(&x_mid, &lp.pre_ffn_norm_w, eps, seq, h, &mut h_ffn);

            let mut gate_pre = vec![0.0; seq * inter];
            matmul(&h_ffn, &lp.gate, seq, h, inter, &mut gate_pre);
            let mut up = vec![0.0; seq * inter];
            matmul(&h_ffn, &lp.up, seq, h, inter, &mut up);

            let mut act = vec![0.0; seq * inter];
            for i in 0..seq * inter {
                act[i] = silu(gate_pre[i]) * up[i];
            }

            let mut ffn_out = vec![0.0; seq * h];
            matmul(&act, &lp.down, seq, inter, h, &mut ffn_out);
            let mut post_ffn_normed = vec![0.0; seq * h];
            rmsnorm_fwd(&ffn_out, &lp.post_ffn_norm_w, eps, seq, h, &mut post_ffn_normed);

            // `layer_scalar` multiplies the whole layer output once, matching
            // the inference model (which scales the full hidden state after the
            // FFN residual, not each sub-layer branch).
            let mut x_out = vec![0.0; seq * h];
            for i in 0..seq * h {
                x_out[i] = (x_mid[i] + post_ffn_normed[i]) * lp.layer_scalar;
            }

            layer_caches.push(LayerCache {
                x_in,
                h_attn,
                q_proj,
                k_proj,
                q,
                k,
                v_raw,
                v,
                attn_probs,
                attn_concat,
                attn_out,
                post_attn_normed,
                x_mid,
                h_ffn,
                gate_pre,
                up,
                act,
                ffn_out,
                post_ffn_normed,
                x_out: x_out.clone(),
            });

            x = x_out;

            // --- causal block summary + inter-block attention -------------------
            if has_blocks {
                let bi = block_index_of_layer(boundaries_cfg, layer_index);
                add_layer_prefix(&mut prefix_sums[bi + 1], &x, seq, h);

                if self.block_config.is_boundary(layer_index) {
                    let means = prefix_means(&prefix_sums, bi + 1, seq, h, lpb);
                    let inter =
                        inter_block_attention_fwd(&x, &means, &self.attn_res_norm, seq, h, eps);
                    boundaries.push(BoundaryCache {
                        layer_idx: layer_index,
                        rep_index: bi + 1,
                        num_entries: bi + 2,
                        weights: inter.weights,
                    });
                    // Per-position attention output is added to that position.
                    for i in 0..seq * h {
                        x[i] += inter.output[i];
                    }
                }
            }
        }

        let x_final_in = x.clone();
        let mut h_final = vec![0.0; seq * h];
        rmsnorm_fwd(&x, &self.final_norm_w, eps, seq, h, &mut h_final);

        let mut logits = vec![0.0; seq * cfg.vocab];
        matmul(&h_final, &self.lm_head, seq, h, cfg.vocab, &mut logits);

        let cache = ForwardCache {
            seq,
            x_embed,
            layers: layer_caches,
            block_prefix: prefix_sums,
            boundaries,
            x_final_in,
            h_final,
            cos,
            sin,
            tokens: tokens.to_vec(),
        };
        (logits, cache)
    }

    // -----------------------------------------------------------------------
    // Loss
    // -----------------------------------------------------------------------

    /// Cross-entropy over the `seq - 1` next-token positions, with a
    /// numerically stable log-softmax. `logits[t]` predicts `tokens[t + 1]`.
    ///
    /// Returns `(mean_loss, d_logits)` where `d_logits` has the same shape as
    /// `logits`; the last position receives zero gradient.
    pub fn cross_entropy(logits: &[f32], tokens: &[u32], vocab: usize) -> (f32, Vec<f32>) {
        let seq = tokens.len();
        assert!(vocab > 0, "vocab must be > 0");
        assert_eq!(logits.len(), seq * vocab, "logits must be [seq, vocab]");
        let mut d_logits = vec![0.0f32; seq * vocab];
        if seq < 2 {
            return (0.0, d_logits);
        }
        let n = (seq - 1) as f32;
        let mut loss = 0.0f32;

        for t in 0..seq - 1 {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let target = tokens[t + 1] as usize;
            assert!(target < vocab, "target token {} outside vocab {}", target, vocab);

            let mut max = f32::NEG_INFINITY;
            for &z in row {
                if z > max {
                    max = z;
                }
            }
            let mut sum = 0.0f32;
            for &z in row {
                sum += (z - max).exp();
            }
            let lse = max + sum.ln();
            loss += lse - row[target];

            let drow = &mut d_logits[t * vocab..(t + 1) * vocab];
            for j in 0..vocab {
                drow[j] = (row[j] - lse).exp() / n;
            }
            drow[target] -= 1.0 / n;
        }

        (loss / n, d_logits)
    }

    // -----------------------------------------------------------------------
    // Backward
    // -----------------------------------------------------------------------

    /// Backward pass. Given the cache from [`Self::forward`] and `d_logits`
    /// (the gradient of the loss with respect to the logits), returns the
    /// gradient of every parameter.
    pub fn backward(&self, cache: &ForwardCache, d_logits: &[f32]) -> Gradients {
        let cfg = &self.config;
        let seq = cache.seq;
        let h = cfg.hidden;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate;
        let heads = cfg.num_heads;
        let kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let eps = cfg.eps;
        let vocab = cfg.vocab;
        assert_eq!(cache.layers.len(), cfg.num_layers);
        assert_eq!(d_logits.len(), seq * vocab);

        let mut grads = Gradients::zeros(cfg);

        // --- lm_head + final norm --------------------------------------------
        let mut dh_final = vec![0.0; seq * h];
        matmul_bwd_w(&cache.h_final, d_logits, &mut grads.lm_head, seq, h, vocab);
        matmul_bwd_x(d_logits, &self.lm_head, &mut dh_final, seq, h, vocab);

        let mut dx = vec![0.0; seq * h];
        rmsnorm_bwd(
            &cache.x_final_in,
            &self.final_norm_w,
            &dh_final,
            &mut grads.final_norm_w,
            &mut dx,
            seq,
            h,
            eps,
        );

        // --- causal block summaries and inter-block attention ------------------
        //
        // Entry e's mean at position t is the prefix mean over 0..=t, so its
        // gradient flows into every position s <= t of every layer in the
        // block. `d_block_means[e]` is `[seq, hidden]` and accumulates the
        // per-position gradient of entry e's mean; `d_layer_extra[l]` receives
        // the suffix-summed result and is added to `dx` at layer `l`.
        let num_entries = cache.block_prefix.len();
        let mut d_block_means: Vec<Vec<f32>> =
            vec![vec![0.0f32; seq * h]; num_entries];
        let mut d_layer_extra: Vec<Vec<f32>> =
            vec![vec![0.0f32; seq * h]; cfg.num_layers];
        let lpb = self.block_config.layers_per_block.max(1);

        // --- layers, last to first -------------------------------------------
        for l in (0..cfg.num_layers).rev() {
            // The boundary attention is processed *before* this block's layers,
            // because it consumes the gradient of the boundary layer's output.
            if let Some(bj) = cache.boundaries.iter().position(|b| b.layer_idx == l) {
                let bc = &cache.boundaries[bj];
                let n = bc.num_entries;

                // Recompute the summaries and their normed keys exactly as the
                // forward pass did. The prefix sums of entries <= n-1 are final
                // by the time this boundary runs, and no later layer touches them.
                let means = prefix_means(&cache.block_prefix, n - 1, seq, h, lpb);
                let mut normed_reps = Vec::with_capacity(n);
                for rep in &means {
                    let mut normed = vec![0.0f32; seq * h];
                    rmsnorm_fwd(rep, &self.attn_res_norm, eps, seq, h, &mut normed);
                    normed_reps.push(normed);
                }

                let mut d_query = vec![0.0f32; seq * h];
                inter_block_attention_bwd(
                    &cache.layers[l].x_out, // query per position
                    &means,
                    &normed_reps,
                    &bc.weights,
                    &dx,
                    &self.attn_res_norm,
                    &mut grads.attn_res_norm,
                    &mut d_query,
                    &mut d_block_means[..n],
                    seq,
                    h,
                    eps,
                );
                // The query at each position is the boundary layer's output.
                for i in 0..seq * h {
                    dx[i] += d_query[i];
                }

                // The summary created here has now received every contribution:
                // boundaries that also use it were processed earlier in reverse.
                let mut d_block = vec![0.0f32; seq * h];
                propagate_mean_grad(
                    &d_block_means[bc.rep_index],
                    seq,
                    h,
                    lpb as f32,
                    &mut d_block,
                );
                let bi = bc.rep_index - 1;
                let first = bi * lpb;
                for bl in first..=l {
                    for i in 0..seq * h {
                        d_layer_extra[bl][i] += d_block[i];
                    }
                }
            }

            // Gradient of this layer's output through its block summary.
            for i in 0..seq * h {
                dx[i] += d_layer_extra[l][i];
            }

            let lc = &cache.layers[l];
            let lp = &self.layers[l];
            let lg = &mut grads.layers[l];

            // x_out = (x_mid + post_ffn_normed) * layer_scalar
            let mut d_pre_scalar = vec![0.0; seq * h];
            for i in 0..seq * h {
                d_pre_scalar[i] = dx[i] * lp.layer_scalar;
            }
            for i in 0..seq * h {
                lg.layer_scalar += dx[i] * (lc.x_mid[i] + lc.post_ffn_normed[i]);
            }

            let d_post_ffn = d_pre_scalar.clone();

            let mut d_ffn_out = vec![0.0; seq * h];
            rmsnorm_bwd(
                &lc.ffn_out,
                &lp.post_ffn_norm_w,
                &d_post_ffn,
                &mut lg.post_ffn_norm_w,
                &mut d_ffn_out,
                seq,
                h,
                eps,
            );

            // ffn_out = act @ Wd
            let mut d_act = vec![0.0; seq * inter];
            matmul_bwd_w(&lc.act, &d_ffn_out, &mut lg.down, seq, inter, h);
            matmul_bwd_x(&d_ffn_out, &lp.down, &mut d_act, seq, inter, h);

            // act = silu(gate_pre) * up
            let mut d_gate_pre = vec![0.0; seq * inter];
            let mut d_up = vec![0.0; seq * inter];
            for i in 0..seq * inter {
                let g = lc.gate_pre[i];
                d_gate_pre[i] = d_act[i] * lc.up[i] * silu_grad(g);
                d_up[i] = d_act[i] * silu(g);
            }

            let mut d_h_ffn = vec![0.0; seq * h];
            let mut d_h_ffn_up = vec![0.0; seq * h];
            matmul_bwd_w(&lc.h_ffn, &d_gate_pre, &mut lg.gate, seq, h, inter);
            matmul_bwd_x(&d_gate_pre, &lp.gate, &mut d_h_ffn, seq, h, inter);
            matmul_bwd_w(&lc.h_ffn, &d_up, &mut lg.up, seq, h, inter);
            matmul_bwd_x(&d_up, &lp.up, &mut d_h_ffn_up, seq, h, inter);
            for i in 0..seq * h {
                d_h_ffn[i] += d_h_ffn_up[i];
            }

            let mut d_x_mid = vec![0.0; seq * h];
            rmsnorm_bwd(
                &lc.x_mid,
                &lp.pre_ffn_norm_w,
                &d_h_ffn,
                &mut lg.pre_ffn_norm_w,
                &mut d_x_mid,
                seq,
                h,
                eps,
            );
            // Residual: x_out = (x_mid + ffn branch) * layer_scalar.
            for i in 0..seq * h {
                d_x_mid[i] += d_pre_scalar[i];
            }

            // x_mid = x_in + post_attn_normed (no scalar here; the scalar is
            // applied once at the end of the layer).
            let d_post_attn = d_x_mid.clone();

            let mut d_attn_out = vec![0.0; seq * h];
            rmsnorm_bwd(
                &lc.attn_out,
                &lp.post_attn_norm_w,
                &d_post_attn,
                &mut lg.post_attn_norm_w,
                &mut d_attn_out,
                seq,
                h,
                eps,
            );

            // attn_out = attn_concat @ Wo
            let mut d_attn_concat = vec![0.0; seq * q_dim];
            matmul_bwd_w(&lc.attn_concat, &d_attn_out, &mut lg.o, seq, q_dim, h);
            matmul_bwd_x(&d_attn_out, &lp.o, &mut d_attn_concat, seq, q_dim, h);

            // attention
            let mut dq = vec![0.0; seq * q_dim];
            let mut dk = vec![0.0; seq * kv_dim];
            let mut dv = vec![0.0; seq * kv_dim];
            attention_bwd(
                &d_attn_concat,
                &lc.attn_probs,
                &lc.q,
                &lc.k,
                &lc.v,
                seq,
                heads,
                kv_heads,
                head_dim,
                &mut dq,
                &mut dk,
                &mut dv,
            );

            // undo RoPE
            apply_rope_backward(&mut dq, seq, heads, head_dim, &cache.cos, &cache.sin);
            apply_rope_backward(&mut dk, seq, kv_heads, head_dim, &cache.cos, &cache.sin);

            // undo per-head q/k norm
            let mut dq_proj = vec![0.0; seq * q_dim];
            rmsnorm_bwd(
                &lc.q_proj,
                &lp.q_norm_w,
                &dq,
                &mut lg.q_norm_w,
                &mut dq_proj,
                seq * heads,
                head_dim,
                eps,
            );
            let mut dk_proj = vec![0.0; seq * kv_dim];
            rmsnorm_bwd(
                &lc.k_proj,
                &lp.k_norm_w,
                &dk,
                &mut lg.k_norm_w,
                &mut dk_proj,
                seq * kv_heads,
                head_dim,
                eps,
            );

            // undo per-head V norm (no learned scale)
            let mut dv_raw = vec![0.0f32; seq * kv_dim];
            rmsnorm_no_scale_bwd(&lc.v_raw, &dv, &mut dv_raw, seq * kv_heads, head_dim, eps);

            // projections back to the normed input
            let mut d_h_attn = vec![0.0; seq * h];
            let mut d_h_attn_k = vec![0.0; seq * h];
            let mut d_h_attn_v = vec![0.0; seq * h];
            matmul_bwd_w(&lc.h_attn, &dq_proj, &mut lg.q, seq, h, q_dim);
            matmul_bwd_x(&dq_proj, &lp.q, &mut d_h_attn, seq, h, q_dim);
            matmul_bwd_w(&lc.h_attn, &dk_proj, &mut lg.k, seq, h, kv_dim);
            matmul_bwd_x(&dk_proj, &lp.k, &mut d_h_attn_k, seq, h, kv_dim);
            matmul_bwd_w(&lc.h_attn, &dv_raw, &mut lg.v, seq, h, kv_dim);
            matmul_bwd_x(&dv_raw, &lp.v, &mut d_h_attn_v, seq, h, kv_dim);
            for i in 0..seq * h {
                d_h_attn[i] += d_h_attn_k[i] + d_h_attn_v[i];
            }

            let mut d_x_in = vec![0.0; seq * h];
            rmsnorm_bwd(
                &lc.x_in,
                &lp.attn_norm_w,
                &d_h_attn,
                &mut lg.attn_norm_w,
                &mut d_x_in,
                seq,
                h,
                eps,
            );
            // Residual: x_mid = x_in + attn branch.
            for i in 0..seq * h {
                dx[i] = d_x_mid[i] + d_x_in[i];
            }
        }

        // --- embedding scatter-add -------------------------------------------
        // Entry 0's causal mean at position t is the prefix mean of the initial
        // embedding over 0..=t. Its gradient is suffix-summed back onto the
        // embedding of every position before scattering through `sqrt(h)`.
        if !cache.block_prefix.is_empty() {
            let mut d_embed_prefix = vec![0.0f32; seq * h];
            propagate_mean_grad(&d_block_means[0], seq, h, 1.0, &mut d_embed_prefix);
            for i in 0..seq * h {
                dx[i] += d_embed_prefix[i];
            }
        }

        let sqrt_h = (h as f32).sqrt();
        for t in 0..seq {
            let tok = cache.tokens[t] as usize;
            for i in 0..h {
                grads.embed[tok * h + i] += dx[t * h + i] * sqrt_h;
            }
        }

        grads
    }
}

// ---------------------------------------------------------------------------
// Forward cache
// ---------------------------------------------------------------------------

/// Everything the backward pass needs from one layer.
#[derive(Debug, Clone)]
pub struct LayerCache {
    pub x_in: Vec<f32>,              // [seq, hidden] layer input
    pub h_attn: Vec<f32>,            // [seq, hidden] rmsnorm(x, attn_norm)
    pub q_proj: Vec<f32>,            // [seq, q_dim] before per-head norm
    pub k_proj: Vec<f32>,            // [seq, kv_dim] before per-head norm
    pub q: Vec<f32>,                 // [seq, q_dim] after norm + rope
    pub k: Vec<f32>,                 // [seq, kv_dim] after norm + rope
    pub v_raw: Vec<f32>,             // [seq, kv_dim] before per-head V norm
    pub v: Vec<f32>,                 // [seq, kv_dim] after per-head V norm
    pub attn_probs: Vec<f32>,        // [heads, seq, seq], causal, rows sum to 1
    pub attn_concat: Vec<f32>,       // [seq, q_dim] attention output
    pub attn_out: Vec<f32>,          // [seq, hidden] attn_concat @ Wo
    pub post_attn_normed: Vec<f32>,  // [seq, hidden] rmsnorm(attn_out)
    pub x_mid: Vec<f32>,             // [seq, hidden] after attention residual
    pub h_ffn: Vec<f32>,             // [seq, hidden] rmsnorm(x_mid, pre_ffn_norm)
    pub gate_pre: Vec<f32>,          // [seq, intermediate] h_ffn @ Wg
    pub up: Vec<f32>,                // [seq, intermediate] h_ffn @ Wu
    pub act: Vec<f32>,               // [seq, intermediate] silu(gate) * up
    pub ffn_out: Vec<f32>,           // [seq, hidden] act @ Wd
    pub post_ffn_normed: Vec<f32>,   // [seq, hidden] rmsnorm(ffn_out)
    pub x_out: Vec<f32>,             // [seq, hidden] layer output
}

/// One inter-block attention site, cached for the backward pass.
#[derive(Debug, Clone)]
pub struct BoundaryCache {
    /// Layer index that ended this block.
    pub layer_idx: usize,
    /// Index of this block's summary in [`ForwardCache::block_prefix`]
    /// (`bi + 1`, where `bi` is the zero-based block index).
    pub rep_index: usize,
    /// Number of summaries used as keys (`rep_index + 1`).
    pub num_entries: usize,
    /// Softmax weights over the summaries, per position: `[seq, num_entries]`.
    pub weights: Vec<f32>,
}

/// Full forward cache.
#[derive(Debug, Clone)]
pub struct ForwardCache {
    pub seq: usize,
    pub x_embed: Vec<f32>,     // [seq, hidden] embed[token] * sqrt(hidden)
    pub layers: Vec<LayerCache>,
    /// Raw causal prefix sums, `[num_entries][seq, hidden]`. Entry 0 is the
    /// initial embedding; entry `b + 1` is block `b`'s layer outputs.
    pub block_prefix: Vec<Vec<f32>>,
    /// One entry per block boundary, in forward order.
    pub boundaries: Vec<BoundaryCache>,
    pub x_final_in: Vec<f32>,  // [seq, hidden] input to final norm
    pub h_final: Vec<f32>,     // [seq, hidden] after final norm
    pub cos: Vec<f32>,         // [seq, head_dim/2]
    pub sin: Vec<f32>,         // [seq, head_dim/2]
    /// Token ids of the batch, kept so backward can scatter into `embed`.
    pub tokens: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Gradients
// ---------------------------------------------------------------------------

/// Gradients for one layer, mirroring [`LayerParams`].
#[derive(Debug, Clone)]
pub struct LayerGradients {
    pub attn_norm_w: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub o: Vec<f32>,
    pub post_attn_norm_w: Vec<f32>,
    pub pre_ffn_norm_w: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
    pub post_ffn_norm_w: Vec<f32>,
    pub layer_scalar: f32,
}

/// Gradients for every parameter of the model.
#[derive(Debug, Clone)]
pub struct Gradients {
    pub embed: Vec<f32>,
    pub layers: Vec<LayerGradients>,
    pub final_norm_w: Vec<f32>,
    pub lm_head: Vec<f32>,
    pub attn_res_norm: Vec<f32>,
}

impl Gradients {
    /// Zeroed gradients sized to the model's parameter layout.
    pub fn zeros(cfg: &F32Config) -> Self {
        let h = cfg.hidden;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate;
        let layer = || LayerGradients {
            attn_norm_w: vec![0.0; h],
            q: vec![0.0; h * q_dim],
            k: vec![0.0; h * kv_dim],
            v: vec![0.0; h * kv_dim],
            q_norm_w: vec![0.0; cfg.head_dim],
            k_norm_w: vec![0.0; cfg.head_dim],
            o: vec![0.0; q_dim * h],
            post_attn_norm_w: vec![0.0; h],
            pre_ffn_norm_w: vec![0.0; h],
            gate: vec![0.0; h * inter],
            up: vec![0.0; h * inter],
            down: vec![0.0; inter * h],
            post_ffn_norm_w: vec![0.0; h],
            layer_scalar: 0.0,
        };
        Self {
            embed: vec![0.0; cfg.vocab * h],
            layers: (0..cfg.num_layers).map(|_| layer()).collect(),
            final_norm_w: vec![0.0; h],
            lm_head: vec![0.0; h * cfg.vocab],
            attn_res_norm: vec![0.0; h],
        }
    }

    /// Same order as [`F32Model::tensors`].
    pub fn tensors(&self) -> Vec<(String, &[f32])> {
        let mut out: Vec<(String, &[f32])> = Vec::new();
        out.push(("embed".to_string(), self.embed.as_slice()));
        for (l, lg) in self.layers.iter().enumerate() {
            let n = |suffix: &str| format!("layer{}.{}", l, suffix);
            out.push((n("attn_norm"), lg.attn_norm_w.as_slice()));
            out.push((n("q"), lg.q.as_slice()));
            out.push((n("k"), lg.k.as_slice()));
            out.push((n("v"), lg.v.as_slice()));
            out.push((n("q_norm"), lg.q_norm_w.as_slice()));
            out.push((n("k_norm"), lg.k_norm_w.as_slice()));
            out.push((n("o"), lg.o.as_slice()));
            out.push((n("post_attn_norm"), lg.post_attn_norm_w.as_slice()));
            out.push((n("pre_ffn_norm"), lg.pre_ffn_norm_w.as_slice()));
            out.push((n("gate"), lg.gate.as_slice()));
            out.push((n("up"), lg.up.as_slice()));
            out.push((n("down"), lg.down.as_slice()));
            out.push((n("post_ffn_norm"), lg.post_ffn_norm_w.as_slice()));
            out.push((n("layer_scalar"), std::slice::from_ref(&lg.layer_scalar)));
        }
        out.push(("final_norm".to_string(), self.final_norm_w.as_slice()));
        out.push(("lm_head".to_string(), self.lm_head.as_slice()));
        out.push(("attn_res_norm".to_string(), self.attn_res_norm.as_slice()));
        out
    }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

/// Adam over the flattened parameter layout.
#[derive(Debug, Clone)]
pub struct Adam {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: u32,
}

impl Adam {
    pub fn new(model: &F32Model) -> Self {
        let n = model.param_count();
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    /// One in-place update of every parameter.
    pub fn step(
        &mut self,
        model: &mut F32Model,
        grads: &Gradients,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
    ) {
        self.t += 1;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);
        let mut off = 0usize;

        let params = model.tensors_mut();
        let grad_tensors = grads.tensors();
        assert_eq!(
            params.len(),
            grad_tensors.len(),
            "parameter/gradient tensor count mismatch"
        );

        for ((name, p), (gname, g)) in params.into_iter().zip(grad_tensors.into_iter()) {
            assert_eq!(name, gname, "parameter/gradient order mismatch");
            assert_eq!(p.len(), g.len(), "parameter/gradient length mismatch for {}", name);
            for i in 0..p.len() {
                let gi = g[i];
                let mi = &mut self.m[off + i];
                let vi = &mut self.v[off + i];
                *mi = beta1 * *mi + (1.0 - beta1) * gi;
                *vi = beta2 * *vi + (1.0 - beta2) * gi * gi;
                let m_hat = *mi / bc1;
                let v_hat = *vi / bc2;
                p[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
            off += p.len();
        }
        assert_eq!(off, self.m.len(), "Adam state size mismatch");
    }
}

// ---------------------------------------------------------------------------
// Numeric primitives
// ---------------------------------------------------------------------------

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn silu_grad(x: f32) -> f32 {
    let s = 1.0 / (1.0 + (-x).exp());
    s * (1.0 + x * (1.0 - s))
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// `y[t, o] = sum_i x[t, i] * w[i*out_dim + o]`.
fn matmul(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize, y: &mut [f32]) {
    debug_assert_eq!(x.len(), rows * in_dim);
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(y.len(), rows * out_dim);
    for t in 0..rows {
        let xr = &x[t * in_dim..(t + 1) * in_dim];
        let yr = &mut y[t * out_dim..(t + 1) * out_dim];
        for o in 0..out_dim {
            yr[o] = 0.0;
        }
        for i in 0..in_dim {
            let xi = xr[i];
            if xi == 0.0 {
                continue;
            }
            let wr = &w[i * out_dim..(i + 1) * out_dim];
            for o in 0..out_dim {
                yr[o] += xi * wr[o];
            }
        }
    }
}

/// `dw[i, o] += sum_t x[t, i] * dy[t, o]`.
fn matmul_bwd_w(x: &[f32], dy: &[f32], dw: &mut [f32], rows: usize, in_dim: usize, out_dim: usize) {
    debug_assert_eq!(x.len(), rows * in_dim);
    debug_assert_eq!(dy.len(), rows * out_dim);
    debug_assert_eq!(dw.len(), in_dim * out_dim);
    for t in 0..rows {
        let xr = &x[t * in_dim..(t + 1) * in_dim];
        let dyr = &dy[t * out_dim..(t + 1) * out_dim];
        for i in 0..in_dim {
            let xi = xr[i];
            if xi == 0.0 {
                continue;
            }
            let dwr = &mut dw[i * out_dim..(i + 1) * out_dim];
            for o in 0..out_dim {
                dwr[o] += xi * dyr[o];
            }
        }
    }
}

/// `dx[t, i] = sum_o dy[t, o] * w[i*out_dim + o]` (overwrites `dx`).
fn matmul_bwd_x(dy: &[f32], w: &[f32], dx: &mut [f32], rows: usize, in_dim: usize, out_dim: usize) {
    debug_assert_eq!(dy.len(), rows * out_dim);
    debug_assert_eq!(w.len(), in_dim * out_dim);
    debug_assert_eq!(dx.len(), rows * in_dim);
    for t in 0..rows {
        let dyr = &dy[t * out_dim..(t + 1) * out_dim];
        let dxr = &mut dx[t * in_dim..(t + 1) * in_dim];
        for i in 0..in_dim {
            let wr = &w[i * out_dim..(i + 1) * out_dim];
            let mut acc = 0.0;
            for o in 0..out_dim {
                acc += dyr[o] * wr[o];
            }
            dxr[i] = acc;
        }
    }
}

/// RMSNorm forward over `rows` contiguous vectors of width `dim`.
fn rmsnorm_fwd(x: &[f32], w: &[f32], eps: f32, rows: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), rows * dim);
    debug_assert_eq!(out.len(), rows * dim);
    debug_assert_eq!(w.len(), dim);
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let or = &mut out[r * dim..(r + 1) * dim];
        let mut ss = 0.0f32;
        for &v in xr {
            ss += v * v;
        }
        let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
        for i in 0..dim {
            or[i] = xr[i] * inv * w[i];
        }
    }
}

/// RMSNorm backward: fills `dw` (accumulated) and `dx` (overwritten).
///
/// `x` is the norm input, `dy` the gradient of the norm output.
fn rmsnorm_bwd(
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    dw: &mut [f32],
    dx: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) {
    debug_assert_eq!(x.len(), rows * dim);
    debug_assert_eq!(dy.len(), rows * dim);
    debug_assert_eq!(dx.len(), rows * dim);
    debug_assert_eq!(dw.len(), dim);
    let inv_dim = 1.0 / dim as f32;
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let dyr = &dy[r * dim..(r + 1) * dim];
        let dxr = &mut dx[r * dim..(r + 1) * dim];

        let mut ss = 0.0f32;
        for &v in xr {
            ss += v * v;
        }
        let inv = 1.0 / (ss * inv_dim + eps).sqrt();

        // d(xn) = dy * w, where xn = x * inv.
        let mut dxn_dot_x = 0.0f32;
        for i in 0..dim {
            let dxn = dyr[i] * w[i];
            dw[i] += dyr[i] * xr[i] * inv;
            dxn_dot_x += dxn * xr[i];
        }
        // dx_i = inv * dxn_i - inv^3 * x_i / dim * sum_j(dxn_j * x_j)
        let c = inv * inv * inv * dxn_dot_x * inv_dim;
        for i in 0..dim {
            let dxn = dyr[i] * w[i];
            dxr[i] = inv * dxn - c * xr[i];
        }
    }
}

/// Per-head RMSNorm forward without a learned scale (V normalisation).
fn rmsnorm_no_scale_fwd(x: &[f32], eps: f32, rows: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), rows * dim);
    debug_assert_eq!(out.len(), rows * dim);
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let or = &mut out[r * dim..(r + 1) * dim];
        let mut ss = 0.0f32;
        for &v in xr {
            ss += v * v;
        }
        let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
        for i in 0..dim {
            or[i] = xr[i] * inv;
        }
    }
}

/// Backward of [`rmsnorm_no_scale_fwd`]. `dx` is overwritten.
fn rmsnorm_no_scale_bwd(
    x: &[f32],
    dy: &[f32],
    dx: &mut [f32],
    rows: usize,
    dim: usize,
    eps: f32,
) {
    debug_assert_eq!(x.len(), rows * dim);
    debug_assert_eq!(dy.len(), rows * dim);
    debug_assert_eq!(dx.len(), rows * dim);
    let inv_dim = 1.0 / dim as f32;
    for r in 0..rows {
        let xr = &x[r * dim..(r + 1) * dim];
        let dyr = &dy[r * dim..(r + 1) * dim];
        let dxr = &mut dx[r * dim..(r + 1) * dim];

        let mut ss = 0.0f32;
        for &v in xr {
            ss += v * v;
        }
        let inv = 1.0 / (ss * inv_dim + eps).sqrt();

        let mut dy_dot_x = 0.0f32;
        for i in 0..dim {
            dy_dot_x += dyr[i] * xr[i];
        }
        let c = inv * inv * inv * dy_dot_x * inv_dim;
        for i in 0..dim {
            dxr[i] = inv * dyr[i] - c * xr[i];
        }
    }
}

/// Forward result of one inter-block attention site.
struct InterBlockForward {
    /// Softmax weights per position: `[seq, n]`.
    weights: Vec<f32>,
    /// Per-position attention output: `[seq, hidden]`.
    output: Vec<f32>,
}

/// Inter-block attention, matching
/// [`crate::model::cpu_block_attn_res::CpuBlockAttnResModel::inter_block_attention`]:
///
/// ```text
/// query_t = hidden[t]
/// key_e_t = rmsnorm(block_means[e][t], attn_res_norm)
/// w_t     = softmax_e((query_t . key_e_t) / sqrt(hidden))
/// out_t   = sum_e w_t[e] * key_e_t
/// ```
///
/// `block_means[e]` is `[seq, hidden]` and already contains the causal prefix
/// mean for each position, so the keys are causal by construction.
fn inter_block_attention_fwd(
    hidden: &[f32],
    block_means: &[Vec<f32>],
    attn_res_norm: &[f32],
    seq: usize,
    hd: usize,
    eps: f32,
) -> InterBlockForward {
    let n = block_means.len();
    assert!(n > 0, "inter-block attention needs at least one summary");

    let mut normed_reps = Vec::with_capacity(n);
    for rep in block_means {
        let mut normed = vec![0.0f32; seq * hd];
        rmsnorm_fwd(rep, attn_res_norm, eps, seq, hd, &mut normed);
        normed_reps.push(normed);
    }

    let scale = 1.0 / (hd as f32).sqrt();
    let mut weights = vec![0.0f32; seq * n];
    let mut output = vec![0.0f32; seq * hd];

    for t in 0..seq {
        let query = &hidden[t * hd..(t + 1) * hd];

        // scores[e] = (query . key_e) / sqrt(hidden)
        let mut scores = vec![0.0f32; n];
        let mut max_score = f32::NEG_INFINITY;
        for e in 0..n {
            let key = &normed_reps[e][t * hd..(t + 1) * hd];
            scores[e] = dot(query, key) * scale;
            if scores[e] > max_score {
                max_score = scores[e];
            }
        }
        let mut sum_exp = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv = 1.0 / sum_exp;
        for e in 0..n {
            weights[t * n + e] = scores[e] * inv;
        }

        let out = &mut output[t * hd..(t + 1) * hd];
        for e in 0..n {
            let w = weights[t * n + e];
            let key = &normed_reps[e][t * hd..(t + 1) * hd];
            for d in 0..hd {
                out[d] += w * key[d];
            }
        }
    }

    InterBlockForward { weights, output }
}

/// Backward of [`inter_block_attention_fwd`], per position.
///
/// `d_block_means` is accumulated (not overwritten): a summary is a key for this
/// boundary and for every later boundary, and the reverse loop has already
/// added the later contributions.
#[allow(clippy::too_many_arguments)]
fn inter_block_attention_bwd(
    hidden: &[f32],             // queries, [seq, hidden]
    block_means: &[Vec<f32>],   // [n][seq, hidden], inputs to the RMSNorm
    normed_reps: &[Vec<f32>],   // [n][seq, hidden]
    weights: &[f32],            // [seq, n]
    d_out: &[f32],              // [seq, hidden]
    attn_res_norm: &[f32],
    d_attn_res_norm: &mut [f32],
    d_query: &mut [f32],        // [seq, hidden]
    d_block_means: &mut [Vec<f32>], // [n][seq, hidden]
    seq: usize,
    hd: usize,
    eps: f32,
) {
    let n = weights.len() / seq;
    let scale = 1.0 / (hd as f32).sqrt();

    for t in 0..seq {
        let query = &hidden[t * hd..(t + 1) * hd];
        let d_out_t = &d_out[t * hd..(t + 1) * hd];

        // dw_e = d_out_t . key_e ; weighted = sum_e w_e * dw_e
        let mut dw = vec![0.0f32; n];
        let mut weighted = 0.0f32;
        for e in 0..n {
            let key = &normed_reps[e][t * hd..(t + 1) * hd];
            dw[e] = dot(d_out_t, key);
            weighted += weights[t * n + e] * dw[e];
        }

        for e in 0..n {
            let w = weights[t * n + e];
            let ds = w * (dw[e] - weighted);
            let key = &normed_reps[e][t * hd..(t + 1) * hd];

            // d_query += ds * scale * key
            for d in 0..hd {
                d_query[t * hd + d] += ds * scale * key[d];
            }

            // d_key = w * d_out + ds * scale * query
            let mut d_key = vec![0.0f32; hd];
            for d in 0..hd {
                d_key[d] = w * d_out_t[d] + ds * scale * query[d];
            }

            // key = rmsnorm(mean); accumulate into the mean's gradient.
            let mut d_mean = vec![0.0f32; hd];
            rmsnorm_bwd(
                &block_means[e][t * hd..(t + 1) * hd],
                attn_res_norm,
                &d_key,
                d_attn_res_norm,
                &mut d_mean,
                1,
                hd,
                eps,
            );
            for d in 0..hd {
                d_block_means[e][t * hd + d] += d_mean[d];
            }
        }
    }
}

/// Convert a gradient w.r.t. a causal prefix mean into gradients w.r.t. the
/// underlying per-position values.
///
/// `mean[t] = (sum_{s <= t} X[s]) / ((t + 1) * divisor)`, therefore
/// `dX[s] = sum_{t >= s} d_mean[t] / ((t + 1) * divisor)`.
/// `out` is accumulated (not overwritten) and has shape `[seq, hidden]`.
fn propagate_mean_grad(d_mean: &[f32], seq: usize, hd: usize, divisor: f32, out: &mut [f32]) {
    let mut suffix = vec![0.0f32; hd];
    for s in (0..seq).rev() {
        let denom = (s + 1) as f32 * divisor;
        for d in 0..hd {
            suffix[d] += d_mean[s * hd + d] / denom;
            out[s * hd + d] += suffix[d];
        }
    }
}


/// RoPE forward, half-split layout: pair `(d, d + half)` rotates by
/// `pos * theta^(-2d/head_dim)`.
fn apply_rope(x: &mut [f32], seq: usize, heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    let half = head_dim / 2;
    for t in 0..seq {
        for hh in 0..heads {
            let base = (t * heads + hh) * head_dim;
            for d in 0..half {
                let c = cos[t * half + d];
                let s = sin[t * half + d];
                let a = x[base + d];
                let b = x[base + d + half];
                x[base + d] = a * c - b * s;
                x[base + d + half] = b * c + a * s;
            }
        }
    }
}

/// RoPE backward: applies the transposed rotation to a gradient buffer.
fn apply_rope_backward(
    dx: &mut [f32],
    seq: usize,
    heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    for t in 0..seq {
        for hh in 0..heads {
            let base = (t * heads + hh) * head_dim;
            for d in 0..half {
                let c = cos[t * half + d];
                let s = sin[t * half + d];
                let da = dx[base + d];
                let db = dx[base + d + half];
                dx[base + d] = da * c + db * s;
                dx[base + d + half] = -da * s + db * c;
            }
        }
    }
}

/// Causal multi-head attention with grouped KV heads.
///
/// `probs` is written as `[heads, seq, seq]` with zeros above the diagonal;
/// `out` is `[seq, q_dim]`.
#[allow(clippy::too_many_arguments)]
fn attention_fwd(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    probs: &mut [f32],
    out: &mut [f32],
) {
    let group = heads / kv_heads;
    // The inference model uses scale 1.0: the per-head Q/K RMSNorm already
    // normalises the dot products.
    let scale = 1.0f32;
    for hh in 0..heads {
        let kvh = hh / group;
        for t in 0..seq {
            let q_base = (t * heads + hh) * head_dim;
            let p_row = (hh * seq + t) * seq;
            let mut max = f32::NEG_INFINITY;
            for s in 0..=t {
                let k_base = (s * kv_heads + kvh) * head_dim;
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += q[q_base + d] * k[k_base + d];
                }
                let score = acc * scale;
                probs[p_row + s] = score;
                if score > max {
                    max = score;
                }
            }
            let mut sum = 0.0f32;
            for s in 0..=t {
                let e = (probs[p_row + s] - max).exp();
                probs[p_row + s] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for s in 0..=t {
                probs[p_row + s] *= inv_sum;
            }
            let out_base = (t * heads + hh) * head_dim;
            for s in 0..=t {
                let p = probs[p_row + s];
                let v_base = (s * kv_heads + kvh) * head_dim;
                for d in 0..head_dim {
                    out[out_base + d] += p * v[v_base + d];
                }
            }
        }
    }
}

/// Causal attention backward. `d_out` is `[seq, q_dim]`; `dq`, `dk`, `dv` are
/// overwritten (zeroed inside).
#[allow(clippy::too_many_arguments)]
fn attention_bwd(
    d_out: &[f32],
    probs: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    dq: &mut [f32],
    dk: &mut [f32],
    dv: &mut [f32],
) {
    for x in dq.iter_mut() {
        *x = 0.0;
    }
    for x in dk.iter_mut() {
        *x = 0.0;
    }
    for x in dv.iter_mut() {
        *x = 0.0;
    }
    let group = heads / kv_heads;
    let scale = 1.0f32;
    let mut dp = vec![0.0f32; seq];

    for hh in 0..heads {
        let kvh = hh / group;
        for t in 0..seq {
            let d_out_base = (t * heads + hh) * head_dim;
            let p_row = (hh * seq + t) * seq;

            // dp[t, s] = d_out[t] . v[s] ; dv[s] += p[t, s] * d_out[t]
            for s in 0..=t {
                let v_base = (s * kv_heads + kvh) * head_dim;
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += d_out[d_out_base + d] * v[v_base + d];
                }
                dp[s] = acc;
                let p = probs[p_row + s];
                let dv_base = v_base;
                for d in 0..head_dim {
                    dv[dv_base + d] += p * d_out[d_out_base + d];
                }
            }

            // softmax backward
            let mut weighted = 0.0f32;
            for s in 0..=t {
                weighted += probs[p_row + s] * dp[s];
            }
            for s in 0..=t {
                let p = probs[p_row + s];
                let ds = p * (dp[s] - weighted);
                let q_base = (t * heads + hh) * head_dim;
                let k_base = (s * kv_heads + kvh) * head_dim;
                for d in 0..head_dim {
                    dq[q_base + d] += ds * scale * k[k_base + d];
                    dk[k_base + d] += ds * scale * q[q_base + d];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(num_layers: usize, num_kv_heads: usize) -> F32Config {
        F32Config {
            hidden: 8,
            num_layers,
            num_heads: 2,
            num_kv_heads,
            head_dim: 4,
            intermediate: 16,
            vocab: 16,
            max_seq: 4,
            rope_theta: 10_000.0,
            eps: 1e-6,
        }
    }

    /// 4 layers, 2 blocks of 2 layers, boundaries at layers 1 and 3.
    fn two_block_config() -> F32Config {
        tiny_config(4, 1)
    }

    fn two_blocks() -> F32BlockConfig {
        F32BlockConfig {
            num_blocks: 2,
            layers_per_block: 2,
            boundary_layers: vec![1, 3],
        }
    }

    /// Loss evaluated with f64 accumulation over the f32 forward pass. Used
    /// only by the numeric side of the gradient check.
    fn loss_f64(model: &F32Model, tokens: &[u32]) -> f64 {
        let vocab = model.config.vocab;
        let (logits, _) = model.forward(tokens);
        let n = tokens.len() - 1;
        let mut loss = 0.0f64;
        for t in 0..n {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let mut max = f64::NEG_INFINITY;
            for &z in row {
                if (z as f64) > max {
                    max = z as f64;
                }
            }
            let mut sum = 0.0f64;
            for &z in row {
                sum += ((z as f64) - max).exp();
            }
            let lse = max + sum.ln();
            loss += lse - row[tokens[t + 1] as usize] as f64;
        }
        loss / n as f64
    }

    /// Pick up to `k` distinct indices from `0..len` (all of them if the tensor
    /// is small), deterministically from `seed`.
    fn pick_indices(len: usize, k: usize, seed: u64) -> Vec<usize> {
        if len <= k {
            return (0..len).collect();
        }
        let mut rng = SplitMix64::new(seed);
        let mut pool: Vec<usize> = (0..len).collect();
        // Partial Fisher-Yates.
        for i in 0..k {
            let j = i + (rng.next_u64() as usize) % (len - i);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }

    /// Central-difference check of every parameter tensor. Panics with the
    /// offending tensor and index if any entry exceeds the tolerance.
    fn run_gradient_check(cfg: F32Config, block_config: F32BlockConfig, seed: u64) {
        let mut model = F32Model::new_random_with_blocks(cfg, block_config.clone(), seed);
        // Non-unit scalars make the once-per-layer placement observable to the
        // finite-difference check.
        for (i, lp) in model.layers.iter_mut().enumerate() {
            lp.layer_scalar = 0.75 + 0.05 * i as f32;
        }
        let tokens: Vec<u32> = (0..cfg.max_seq)
            .map(|t| ((t * 3 + 1) % cfg.vocab) as u32)
            .collect();

        let (logits, cache) = model.forward(&tokens);
        let (loss, d_logits) = F32Model::cross_entropy(&logits, &tokens, cfg.vocab);
        assert!(loss.is_finite(), "forward loss is not finite: {}", loss);
        let grads = model.backward(&cache, &d_logits);

        // Clone the parameter/gradient values: the loop below mutates the model,
        // so it must not hold borrows into it.
        let param_tensors: Vec<(String, Vec<f32>)> = model
            .tensors()
            .iter()
            .map(|(n, t)| (n.clone(), t.to_vec()))
            .collect();
        let grad_tensors: Vec<(String, Vec<f32>)> = grads
            .tensors()
            .iter()
            .map(|(n, t)| (n.clone(), t.to_vec()))
            .collect();
        assert_eq!(param_tensors.len(), grad_tensors.len());

        let eps = 1e-2f32;
        let abs_tol = 1e-2f64;
        let rel_tol = 1e-2f64;
        let mut worst_tensor = String::new();
        let mut worst_err = 0.0f64;

        println!(
            "gradient check: layers={} heads={} kv_heads={} seq={} blocks={:?}",
            cfg.num_layers,
            cfg.num_heads,
            cfg.num_kv_heads,
            tokens.len(),
            block_config.boundary_layers
        );
        println!("{:<26} {:>5} {:>12} {:>12}", "tensor", "n", "max_abs_err", "max_rel_err");

        for (ti, ((name, params), (_, g))) in
            param_tensors.iter().zip(grad_tensors.iter()).enumerate()
        {
            assert_eq!(params.len(), g.len(), "shape mismatch for {}", name);
            let indices = pick_indices(params.len(), 8, seed ^ ((ti as u64) << 32));
            let mut max_abs = 0.0f64;
            let mut max_rel = 0.0f64;

            for &i in &indices {
                let orig = params[i];
                let analytic = g[i] as f64;

                model.set_tensor_elem(ti, i, orig + eps);
                let plus = loss_f64(&model, &tokens);
                model.set_tensor_elem(ti, i, orig - eps);
                let minus = loss_f64(&model, &tokens);
                model.set_tensor_elem(ti, i, orig);

                let numeric = (plus - minus) / (2.0 * eps as f64);
                let abs_err = (analytic - numeric).abs();
                let rel_err = if numeric.abs() > 1e-12 {
                    abs_err / numeric.abs()
                } else {
                    abs_err
                };
                max_abs = max_abs.max(abs_err);
                max_rel = max_rel.max(rel_err);

                let limit = abs_tol + rel_tol * numeric.abs();
                assert!(
                    abs_err <= limit,
                    "gradient mismatch in tensor '{}' at index {}: analytic={} numeric={} |diff|={} limit={}",
                    name,
                    i,
                    analytic,
                    numeric,
                    abs_err,
                    limit
                );
            }

            if max_abs > worst_err {
                worst_err = max_abs;
                worst_tensor = name.clone();
            }
            println!(
                "{:<26} {:>5} {:>12.3e} {:>12.3e}",
                name,
                params.len(),
                max_abs,
                max_rel
            );
        }

        println!(
            "gradient check OK (worst tensor: {} max_abs={:.3e})",
            worst_tensor, worst_err
        );
    }

    // -----------------------------------------------------------------------
    // Equivalence with the CPU inference model
    // -----------------------------------------------------------------------

    use crate::model::cpu_block_attn_res::{
        BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel,
    };
    use crate::model::cpu_linear::{CpuLinear, CpuRmsNorm};

    /// Norm weights around 1.0, so the model is not degenerate.
    fn rand_norm(rng: &mut SplitMix64, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        fill_uniform(&mut v, 0.1, rng);
        for x in v.iter_mut() {
            *x += 1.0;
        }
        v
    }

    /// `CpuLinear` stores `[out, in]` row-major and quantises to ternary.
    fn cpu_linear(rng: &mut SplitMix64, in_f: usize, out_f: usize) -> CpuLinear {
        let mut w = vec![0.0f32; out_f * in_f];
        fill_uniform(&mut w, init_scale(in_f), rng);
        CpuLinear::from_weight(w, in_f, out_f)
    }

    /// CPU weight `[out, in]` -> f32-trainer weight `[in, out]`.
    fn cpu_to_f32(w: &[f32], in_f: usize, out_f: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; in_f * out_f];
        for o in 0..out_f {
            for i in 0..in_f {
                out[i * out_f + o] = w[o * in_f + i];
            }
        }
        out
    }

    /// Dense `CpuBlockAttnResModel` with no PLE, no MoE, no KV sharing, no
    /// softcapping, SwiGLU, and `layer_scalar = 1`.
    fn cpu_dense_block_model(
        cfg: &F32Config,
        block: &F32BlockConfig,
        seed: u64,
    ) -> CpuBlockAttnResModel {
        let mut rng = SplitMix64::new(seed);
        let h = cfg.hidden;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let hd = cfg.head_dim;
        let inter = cfg.intermediate;

        let layers = (0..cfg.num_layers)
            .map(|idx| CpuBlockAttnResLayer {
                layer_number: idx,
                hidden_dim: h,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: hd,
                intermediate_dim: inter,
                attn_norm: CpuRmsNorm::from_weight(rand_norm(&mut rng, h), cfg.eps),
                q_proj: cpu_linear(&mut rng, h, q_dim),
                k_proj: cpu_linear(&mut rng, h, kv_dim),
                v_proj: cpu_linear(&mut rng, h, kv_dim),
                out_proj: cpu_linear(&mut rng, q_dim, h),
                q_norm: CpuRmsNorm::from_weight(rand_norm(&mut rng, hd), cfg.eps),
                k_norm: CpuRmsNorm::from_weight(rand_norm(&mut rng, hd), cfg.eps),
                // The inference forward ignores `v_norm` and uses
                // `per_head_rms_norm_no_scale`, which has no parameters.
                v_norm: CpuRmsNorm::from_weight(vec![1.0; hd], cfg.eps),
                post_attn_norm: CpuRmsNorm::from_weight(rand_norm(&mut rng, h), cfg.eps),
                pre_ffn_norm: CpuRmsNorm::from_weight(rand_norm(&mut rng, h), cfg.eps),
                ffn_gate: Some(cpu_linear(&mut rng, h, inter)),
                ffn_up: Some(cpu_linear(&mut rng, h, inter)),
                ffn_down: Some(cpu_linear(&mut rng, inter, h)),
                moe: None,
                post_ffn_norm: CpuRmsNorm::from_weight(rand_norm(&mut rng, h), cfg.eps),
                // A value other than 1.0 so the equivalence test exercises the
                // once-per-layer scalar placement.
                layer_scalar: 0.7 + 0.05 * idx as f32,
                ple_input_gate: None,
                ple_projection: None,
                ple_post_norm: None,
                rope_theta: cfg.rope_theta as f64,
                partial_rotary_factor: 1.0,
                use_gelu: false,
                kv_shared: false,
            })
            .collect();

        let mut embed = vec![0.0f32; cfg.vocab * h];
        fill_uniform(&mut embed, init_scale(h), &mut rng);
        let mut lm_head = vec![0.0f32; h * cfg.vocab];
        fill_uniform(&mut lm_head, init_scale(h), &mut rng);

        CpuBlockAttnResModel {
            layers,
            embed_tokens: embed,
            lm_head,
            final_norm: rand_norm(&mut rng, h),
            hidden_dim: h,
            vocab_size: cfg.vocab,
            num_layers: cfg.num_layers,
            final_logit_softcapping: None,
            ple_model_projection: None,
            ple_projection_norm: None,
            embed_tokens_per_layer: None,
            hidden_size_per_layer_input: 0,
            num_kv_shared_layers: 0,
            block_config: BlockConfig {
                num_blocks: block.num_blocks,
                layers_per_block: block.layers_per_block,
                boundary_layers: block.boundary_layers.clone(),
                // Present in the inference struct, unused by its forward.
                attn_res_proj: vec![0.0; h * h],
                attn_res_norm: rand_norm(&mut rng, h),
            },
            lora_manager: None,
        }
    }

    /// Copy the (ternary-quantised, dequantised) CPU weights into an
    /// `F32Model`, transposing every `[out, in]` projection to `[in, out]`.
    fn f32_from_cpu_model(
        cfg: F32Config,
        block: F32BlockConfig,
        cpu: &CpuBlockAttnResModel,
    ) -> F32Model {
        let mut model = F32Model::new_random_with_blocks(cfg, block, 0);
        let h = cfg.hidden;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let inter = cfg.intermediate;

        model.embed.copy_from_slice(&cpu.embed_tokens);
        model.lm_head.copy_from_slice(&cpu.lm_head);
        model.final_norm_w.copy_from_slice(&cpu.final_norm);
        model.attn_res_norm.copy_from_slice(&cpu.block_config.attn_res_norm);

        assert_eq!(model.layers.len(), cpu.layers.len());
        for (ml, cl) in model.layers.iter_mut().zip(cpu.layers.iter()) {
            ml.attn_norm_w.copy_from_slice(cl.attn_norm.weight());
            ml.q.copy_from_slice(&cpu_to_f32(&cl.q_proj.weight(), h, q_dim));
            ml.k.copy_from_slice(&cpu_to_f32(&cl.k_proj.weight(), h, kv_dim));
            ml.v.copy_from_slice(&cpu_to_f32(&cl.v_proj.weight(), h, kv_dim));
            ml.q_norm_w.copy_from_slice(cl.q_norm.weight());
            ml.k_norm_w.copy_from_slice(cl.k_norm.weight());
            ml.o.copy_from_slice(&cpu_to_f32(&cl.out_proj.weight(), q_dim, h));
            ml.post_attn_norm_w.copy_from_slice(cl.post_attn_norm.weight());
            ml.pre_ffn_norm_w.copy_from_slice(cl.pre_ffn_norm.weight());
            ml.gate.copy_from_slice(&cpu_to_f32(
                &cl.ffn_gate.as_ref().unwrap().weight(),
                h,
                inter,
            ));
            ml.up.copy_from_slice(&cpu_to_f32(
                &cl.ffn_up.as_ref().unwrap().weight(),
                h,
                inter,
            ));
            ml.down.copy_from_slice(&cpu_to_f32(
                &cl.ffn_down.as_ref().unwrap().weight(),
                inter,
                h,
            ));
            ml.post_ffn_norm_w.copy_from_slice(cl.post_ffn_norm.weight());
            ml.layer_scalar = cl.layer_scalar;
        }

        model
    }

    /// Equivalence gate: the f32 forward must reproduce the CPU inference
    /// forward on the same weights, including both block boundaries.
    #[test]
    fn f32_forward_matches_cpu_block_attn_res() {
        let cfg = two_block_config();
        let block = two_blocks();
        let cpu = cpu_dense_block_model(&cfg, &block, 0xC0FF_EE01);
        let model = f32_from_cpu_model(cfg, block, &cpu);

        let tokens: Vec<u32> = vec![1, 4, 2, 7];
        let cpu_logits = cpu.forward(&tokens);
        let (f32_logits, cache) = model.forward(&tokens);
        assert_eq!(cpu_logits.len(), f32_logits.len());
        assert_eq!(cache.block_prefix.len(), 3, "initial embedding + one per boundary");
        assert_eq!(cache.boundaries.len(), 2);

        let mut max_abs = 0.0f32;
        let mut max_at = (0usize, 0usize);
        for t in 0..tokens.len() {
            for j in 0..cfg.vocab {
                let i = t * cfg.vocab + j;
                let d = (cpu_logits[i] - f32_logits[i]).abs();
                if d > max_abs {
                    max_abs = d;
                    max_at = (t, j);
                }
            }
        }

        println!(
            "equivalence: max |cpu - f32| logit diff = {:.3e} at (t={}, j={})",
            max_abs, max_at.0, max_at.1
        );
        println!("cpu logits[0..4] = {:?}", &cpu_logits[..4]);
        println!("f32 logits[0..4] = {:?}", &f32_logits[..4]);
        assert!(
            max_abs <= 1e-4,
            "f32 forward differs from the CPU inference forward by {} at (t={}, j={})",
            max_abs,
            max_at.0,
            max_at.1
        );
    }

    /// Reference RMSNorm with the same formula as `gemma_mapper::rms_norm`,
    /// written independently so the score test does not reuse the code under
    /// test.
    fn ref_rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter()
            .zip(w.iter())
            .map(|(&v, &g)| v * inv * g)
            .collect()
    }

    /// The gate: with blocks configured, perturbing the token at position `t`
    /// must leave every logit at positions `< t` unchanged. The old
    /// whole-sequence block mean made token 0 depend on every future token.
    #[test]
    fn cpu_block_attn_res_forward_is_causal() {
        let cfg = two_block_config();
        let block = two_blocks();
        let cpu = cpu_dense_block_model(&cfg, &block, 0xC0FF_EE02);

        let tokens: Vec<u32> = vec![1, 4, 2, 7];
        let changed = 2;
        let mut perturbed = tokens.clone();
        perturbed[changed] = 9;

        let base = cpu.forward(&tokens);
        let pert = cpu.forward(&perturbed);

        let mut max_future = 0.0f32;
        for t in 0..changed {
            for j in 0..cfg.vocab {
                let a = base[t * cfg.vocab + j];
                let b = pert[t * cfg.vocab + j];
                max_future = max_future.max((a - b).abs());
                assert!(
                    (a - b).abs() <= 1e-6,
                    "logit at position {} changed when token {} changed: {} vs {}",
                    t,
                    changed,
                    a,
                    b
                );
            }
        }
        println!(
            "causality: max |delta| at positions < {} = {:.3e}",
            changed, max_future
        );
    }

    /// With two block entries, the output must be the softmax-weighted sum of
    /// `q · key_e`, where `key_e = rms_norm(entry_e, attn_res_norm)`. This is
    /// the reference that the old `[num_entries, hidden]`-read-as-
    /// `[hidden, num_entries]` layout could not pass.
    #[test]
    fn inter_block_scores_match_reference() {
        let cfg = two_block_config();
        let block = two_blocks();
        let cpu = cpu_dense_block_model(&cfg, &block, 0xC0FF_EE03);

        let hd = cfg.hidden;
        let eps = cfg.eps;
        let mut rng = SplitMix64::new(0xABCD_1234);
        let mut hidden = vec![0.0f32; hd];
        fill_uniform(&mut hidden, 1.0, &mut rng);
        let mut e0 = vec![0.0f32; hd];
        fill_uniform(&mut e0, 1.0, &mut rng);
        let mut e1 = vec![0.0f32; hd];
        fill_uniform(&mut e1, 1.0, &mut rng);

        let block_reps = vec![e0.clone(), e1.clone()];
        let got = cpu.inter_block_attention(&hidden, &block_reps, 1);

        let norm = &cpu.block_config.attn_res_norm;
        let k0 = ref_rms_norm(&e0, norm, eps);
        let k1 = ref_rms_norm(&e1, norm, eps);
        let scale = 1.0 / (hd as f32).sqrt();
        let s0 = dot(&hidden, &k0) * scale;
        let s1 = dot(&hidden, &k1) * scale;
        let max_s = s0.max(s1);
        let w0 = (s0 - max_s).exp();
        let w1 = (s1 - max_s).exp();
        let inv = 1.0 / (w0 + w1);
        let (w0, w1) = (w0 * inv, w1 * inv);

        let expected: Vec<f32> = (0..hd)
            .map(|d| w0 * k0[d] + w1 * k1[d])
            .collect();

        let mut max_diff = 0.0f32;
        for d in 0..hd {
            max_diff = max_diff.max((got[d] - expected[d]).abs());
        }
        println!(
            "score reference: w0={:.6} w1={:.6} max |got - reference| = {:.3e}",
            w0, w1, max_diff
        );
        assert!(
            max_diff <= 1e-6,
            "inter-block output differs from the q·key reference by {}",
            max_diff
        );
    }

    /// Prefill the first token, then decode the rest one token at a time; the
    /// logits must match the full `forward` at every position.
    #[test]
    fn cpu_prefill_decode_matches_forward() {
        use crate::inference::student_kv_cache::ModelKVCache;

        let cfg = two_block_config();
        let block = two_blocks();
        let cpu = cpu_dense_block_model(&cfg, &block, 0xC0FF_EE04);
        let tokens: Vec<u32> = vec![1, 4, 2, 7];

        let full = cpu.forward(&tokens);

        let kv_dims = vec![cfg.kv_dim(); cfg.num_layers];
        let mut cache = ModelKVCache::new(
            cfg.num_layers,
            cfg.hidden,
            &kv_dims,
            block.layers_per_block,
            0,
        );

        let mut step_logits: Vec<Vec<f32>> = Vec::new();
        step_logits.push(cpu.forward_prefill(&tokens[..1], &mut cache));
        for &tok in &tokens[1..] {
            step_logits.push(cpu.forward_decode(tok, &mut cache));
        }

        let mut max_diff = 0.0f32;
        let mut max_at = (0usize, 0usize);
        for t in 0..tokens.len() {
            for j in 0..cfg.vocab {
                let a = full[t * cfg.vocab + j];
                let b = step_logits[t][j];
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                    max_at = (t, j);
                }
            }
        }
        println!(
            "prefill+decode: max |forward - cached| = {:.3e} at (t={}, j={})",
            max_diff, max_at.0, max_at.1
        );
        assert!(
            max_diff <= 1e-4,
            "prefill/decode differs from forward by {} at (t={}, j={})",
            max_diff,
            max_at.0,
            max_at.1
        );
    }

    #[test]
    fn gradient_check_gqa_one_layer() {
        run_gradient_check(tiny_config(1, 1), F32BlockConfig::plain(), 0x5EED_0001);
    }

    #[test]
    fn gradient_check_two_layers_full_kv() {
        run_gradient_check(tiny_config(2, 2), F32BlockConfig::plain(), 0x5EED_0002);
    }

    /// Two blocks of two layers each: boundaries at layers 1 and 3. Exercises
    /// the block-summary gradient into every layer of both blocks and the
    /// `attn_res_norm` gradient through the RMSNorm of the summaries.
    #[test]
    fn gradient_check_two_blocks() {
        run_gradient_check(two_block_config(), two_blocks(), 0x5EED_0003);
    }

    #[test]
    fn loss_decreases_on_synthetic_batch() {
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
        let mut model = F32Model::new_random(cfg, 42);
        // Period-4 pattern: the next token is a deterministic function of the
        // current one, so the model can drive the loss close to zero.
        let tokens: Vec<u32> = (0..16).map(|t| ((t % 4) + 1) as u32).collect();

        let mut adam = Adam::new(&model);
        let mut losses = Vec::with_capacity(300);
        for _ in 0..300 {
            let (logits, cache) = model.forward(&tokens);
            let (loss, d_logits) = F32Model::cross_entropy(&logits, &tokens, cfg.vocab);
            assert!(loss.is_finite(), "loss became non-finite");
            losses.push(loss);
            let grads = model.backward(&cache, &d_logits);
            adam.step(&mut model, &grads, 1e-2, 0.9, 0.999, 1e-8);
        }

        let first: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last: f32 = losses[290..].iter().sum::<f32>() / 10.0;
        println!("loss first-10 mean = {:.6}, last-10 mean = {:.6}", first, last);
        assert!(last.is_finite());
        assert!(
            last < 0.5 * first,
            "loss did not halve: first={} last={}",
            first,
            last
        );
    }

    /// Same synthetic task as [`loss_decreases_on_synthetic_batch`], with two
    /// blocks of two layers (boundaries at layers 1 and 3).
    #[test]
    fn loss_decreases_with_two_blocks() {
        let cfg = F32Config {
            hidden: 16,
            num_layers: 4,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate: 32,
            vocab: 32,
            max_seq: 16,
            rope_theta: 10_000.0,
            eps: 1e-6,
        };
        let block = F32BlockConfig {
            num_blocks: 2,
            layers_per_block: 2,
            boundary_layers: vec![1, 3],
        };
        let mut model = F32Model::new_random_with_blocks(cfg, block, 42);
        let tokens: Vec<u32> = (0..16).map(|t| ((t % 4) + 1) as u32).collect();

        let mut adam = Adam::new(&model);
        let mut losses = Vec::with_capacity(300);
        for _ in 0..300 {
            let (logits, cache) = model.forward(&tokens);
            let (loss, d_logits) = F32Model::cross_entropy(&logits, &tokens, cfg.vocab);
            assert!(loss.is_finite(), "loss became non-finite");
            losses.push(loss);
            let grads = model.backward(&cache, &d_logits);
            adam.step(&mut model, &grads, 1e-2, 0.9, 0.999, 1e-8);
        }

        let first: f32 = losses[..10].iter().sum::<f32>() / 10.0;
        let last: f32 = losses[290..].iter().sum::<f32>() / 10.0;
        println!(
            "blocked loss first-10 mean = {:.6}, last-10 mean = {:.6}",
            first, last
        );
        assert!(last.is_finite());
        assert!(
            last < 0.5 * first,
            "blocked loss did not halve: first={} last={}",
            first,
            last
        );
    }

    #[test]
    fn deterministic_init_and_loss() {
        for (cfg, block) in [
            (tiny_config(1, 1), F32BlockConfig::plain()),
            (two_block_config(), two_blocks()),
        ] {
            let a = F32Model::new_random_with_blocks(cfg, block.clone(), 1234);
            let b = F32Model::new_random_with_blocks(cfg, block, 1234);

            let ta = a.tensors();
            let tb = b.tensors();
            assert_eq!(ta.len(), tb.len());
            for ((na, pa), (nb, pb)) in ta.iter().zip(tb.iter()) {
                assert_eq!(na, nb);
                assert_eq!(pa, pb, "parameters differ for tensor {}", na);
            }

            let tokens: Vec<u32> = vec![1, 4, 2, 7];
            let (la, _) = a.forward(&tokens);
            let (lb, _) = b.forward(&tokens);
            assert_eq!(la, lb, "logits differ for identical seeds");
            let (loss_a, _) = F32Model::cross_entropy(&la, &tokens, cfg.vocab);
            let (loss_b, _) = F32Model::cross_entropy(&lb, &tokens, cfg.vocab);
            assert_eq!(loss_a, loss_b);
        }
    }

    #[test]
    fn causal_mask_blocks_future_tokens() {
        let cfg = tiny_config(2, 1);
        let model = F32Model::new_random(cfg, 99);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let changed_pos = 2;

        let mut perturbed = tokens.clone();
        perturbed[changed_pos] = 9;

        let (base, _) = model.forward(&tokens);
        let (pert, _) = model.forward(&perturbed);

        for t in 0..changed_pos {
            for j in 0..cfg.vocab {
                let a = base[t * cfg.vocab + j];
                let b = pert[t * cfg.vocab + j];
                assert!(
                    (a - b).abs() <= 1e-6,
                    "logit at position {} changed when token {} changed: {} vs {}",
                    t,
                    changed_pos,
                    a,
                    b
                );
            }
        }
    }

    /// The causal mask inside every attention layer still holds with blocks,
    /// and — after the inter-block fix — so do the final logits: perturbing the
    /// token at position `t` leaves every logit at positions `< t` unchanged.
    #[test]
    fn causal_mask_within_blocks() {
        let cfg = two_block_config();
        let model = F32Model::new_random_with_blocks(cfg, two_blocks(), 99);
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let changed_pos = 2;

        let mut perturbed = tokens.clone();
        perturbed[changed_pos] = 9;

        let (base_logits, base) = model.forward(&tokens);
        let (pert_logits, pert) = model.forward(&perturbed);

        for lc in &base.layers {
            for hh in 0..cfg.num_heads {
                for t in 0..tokens.len() {
                    for s in (t + 1)..tokens.len() {
                        assert_eq!(
                            lc.attn_probs[(hh * tokens.len() + t) * tokens.len() + s],
                            0.0,
                            "attention probability above the diagonal"
                        );
                    }
                }
            }
        }

        for l in 0..2 {
            for t in 0..changed_pos {
                for d in 0..cfg.hidden {
                    let a = base.layers[l].x_out[t * cfg.hidden + d];
                    let b = pert.layers[l].x_out[t * cfg.hidden + d];
                    assert!(
                        (a - b).abs() <= 1e-6,
                        "layer {} output at position {} changed when token {} changed",
                        l,
                        t,
                        changed_pos
                    );
                }
            }
        }

        // Final logits must also be causal: this is the gate the old
        // whole-sequence block mean failed.
        for t in 0..changed_pos {
            for j in 0..cfg.vocab {
                let a = base_logits[t * cfg.vocab + j];
                let b = pert_logits[t * cfg.vocab + j];
                assert!(
                    (a - b).abs() <= 1e-6,
                    "logit at position {} changed when token {} changed: {} vs {}",
                    t,
                    changed_pos,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn shapes_and_param_count_consistent() {
        for (cfg, block) in [
            (tiny_config(2, 2), F32BlockConfig::plain()),
            (two_block_config(), two_blocks()),
        ] {
            let model = F32Model::new_random_with_blocks(cfg, block, 7);
            let tokens: Vec<u32> = vec![0, 1, 2, 3];
            let (logits, cache) = model.forward(&tokens);
            assert_eq!(logits.len(), tokens.len() * cfg.vocab);
            assert_eq!(cache.seq, tokens.len());

            let (_, d_logits) = F32Model::cross_entropy(&logits, &tokens, cfg.vocab);
            let grads = model.backward(&cache, &d_logits);

            let params = model.tensors();
            let grad_tensors = grads.tensors();
            assert_eq!(params.len(), grad_tensors.len());

            let mut sum = 0usize;
            for ((name, p), (gname, g)) in params.iter().zip(grad_tensors.iter()) {
                assert_eq!(name, gname);
                assert_eq!(p.len(), g.len(), "gradient length mismatch for {}", name);
                sum += p.len();
            }
            assert_eq!(model.param_count(), sum);

            // Adam state must be sized to the same layout.
            let adam = Adam::new(&model);
            assert_eq!(adam.m.len(), model.param_count());
            assert_eq!(adam.v.len(), model.param_count());
        }
    }
}
