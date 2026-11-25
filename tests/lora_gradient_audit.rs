//! Audit of the distillation / LoRA gradient path in a blocked config.
//!
//! `src/training/backward.rs::forward_train` now uses the causal inter-block
//! rule, but the gradient code that consumes its `LayerActivations`
//! (`compute_lora_grad` in `src/main.rs`, and the equivalent
//! `LoraLayer::backward` here) uses the per-position `d_hidden` as the
//! gradient for every projection. This test measures that surrogate against a
//! central finite difference of the same loss.
//!
//! It is `#[ignore]` because it currently **fails**: the path is not a
//! gradient of the blocked model. Run it with:
//!
//! ```text
//! cargo test --test lora_gradient_audit -- --ignored --nocapture
//! ```

use ferrisres::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel};
use ferrisres::model::cpu_linear::{CpuLinear, CpuRmsNorm};
use ferrisres::training::lora::LoraConfig;

const HIDDEN: usize = 16;
const HEADS: usize = 2;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const INTERMEDIATE: usize = 32;
const VOCAB: usize = 8;
const NUM_BLOCKS: usize = 2;
const BLOCK_SIZE: usize = 2;
const TOKENS: [u32; 4] = [1, 2, 3, 4];
const RANK: usize = 2;

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
    fn centered(&mut self, s: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * s
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.centered(s)).collect()
    }
}

fn build_model() -> CpuBlockAttnResModel {
    let mut rng = Rng::new(0xa11c_e5);
    let q_dim = HEADS * HEAD_DIM;
    let kv_dim = KV_HEADS * HEAD_DIM;
    let mut layers = Vec::new();
    for _ in 0..(NUM_BLOCKS * BLOCK_SIZE) {
        let mut l = CpuBlockAttnResLayer::new(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE);
        l.use_gelu = true;
        l.layer_scalar = 0.9;
        l.attn_norm = CpuRmsNorm::from_weight(vec![1.0; HIDDEN], 1e-6);
        l.q_proj = CpuLinear::from_weight(rng.vec(HIDDEN * q_dim, 0.6), HIDDEN, q_dim);
        l.k_proj = CpuLinear::from_weight(rng.vec(HIDDEN * kv_dim, 0.6), HIDDEN, kv_dim);
        l.v_proj = CpuLinear::from_weight(rng.vec(HIDDEN * kv_dim, 0.6), HIDDEN, kv_dim);
        l.out_proj = CpuLinear::from_weight(rng.vec(q_dim * HIDDEN, 0.6), q_dim, HIDDEN);
        l.q_norm = CpuRmsNorm::from_weight(vec![1.0; HEAD_DIM], 1e-6);
        l.k_norm = CpuRmsNorm::from_weight(vec![1.0; HEAD_DIM], 1e-6);
        l.post_attn_norm = CpuRmsNorm::from_weight(vec![1.0; HIDDEN], 1e-6);
        l.pre_ffn_norm = CpuRmsNorm::from_weight(vec![1.0; HIDDEN], 1e-6);
        l.ffn_gate = Some(CpuLinear::from_weight(rng.vec(HIDDEN * INTERMEDIATE, 0.6), HIDDEN, INTERMEDIATE));
        l.ffn_up = Some(CpuLinear::from_weight(rng.vec(HIDDEN * INTERMEDIATE, 0.6), HIDDEN, INTERMEDIATE));
        l.ffn_down = Some(CpuLinear::from_weight(rng.vec(INTERMEDIATE * HIDDEN, 0.6), INTERMEDIATE, HIDDEN));
        l.post_ffn_norm = CpuRmsNorm::from_weight(vec![1.0; HIDDEN], 1e-6);
        layers.push(l);
    }
    let boundary_layers = (0..NUM_BLOCKS).map(|b| (b + 1) * BLOCK_SIZE - 1).collect();
    let mut model = CpuBlockAttnResModel {
        layers,
        embed_tokens: rng.vec(VOCAB * HIDDEN, 0.5),
        lm_head: rng.vec(HIDDEN * VOCAB, 0.5),
        final_norm: vec![1.0; HIDDEN],
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
            attn_res_norm: vec![1.0; HIDDEN],
        },
        lora_manager: None,
    };
    model.attach_lora(LoraConfig::targeting(RANK, vec!["q_proj"]));

    // Give layer 0's q_proj adapter non-zero A and B (B is zero-initialised).
    let mut rng = Rng::new(0xbeef);
    let a = rng.vec(RANK * HIDDEN, 0.3);
    let b = rng.vec(q_dim * RANK, 0.3);
    for (idx, name, layer) in model.lora_manager.as_mut().unwrap().adapters_iter_mut() {
        if idx == 0 && name == "q_proj" {
            layer.lora_a_mut().copy_from_slice(&a);
            layer.lora_b_mut().copy_from_slice(&b);
        }
    }
    model
}

/// Fixed linear functional of the logits, so `dL/dlogits` is a known constant.
fn loss_weights() -> Vec<f32> {
    let mut rng = Rng::new(0xfeed);
    rng.vec(TOKENS.len() * VOCAB, 1.0)
}

fn loss(model: &CpuBlockAttnResModel, r: &[f32]) -> f32 {
    let out = model.forward_train(&TOKENS);
    out.logits.iter().zip(r).map(|(l, w)| l * w).sum()
}

fn set_adapter_elem(model: &mut CpuBlockAttnResModel, which_a: bool, index: usize, value: f32) {
    for (idx, name, layer) in model.lora_manager.as_mut().unwrap().adapters_iter_mut() {
        if idx == 0 && name == "q_proj" {
            if which_a {
                layer.lora_a_mut()[index] = value;
            } else {
                layer.lora_b_mut()[index] = value;
            }
        }
    }
}

#[test]
#[ignore = "documents a real defect: the distillation LoRA gradient is not a gradient in a blocked config"]
fn lora_gradient_matches_finite_difference_in_blocked_config() {
    let mut model = build_model();
    let r = loss_weights();
    let seq = TOKENS.len();

    // ---- surrogate gradient: exactly what the distillation path computes ----
    let fwd = model.forward_train(&TOKENS);
    // dL/dlogits = r. d_hidden = r @ lm_head^T (final RMSNorm Jacobian ignored).
    let mut d_hidden = vec![0.0f32; seq * HIDDEN];
    for t in 0..seq {
        for d in 0..HIDDEN {
            let mut sum = 0.0f32;
            for v in 0..VOCAB {
                sum += r[t * VOCAB + v] * model.lm_head[d * VOCAB + v];
            }
            d_hidden[t * HIDDEN + d] = sum;
        }
    }
    let cached_input = fwd.activations[0].pre_attn_normed.clone();

    let (grad_a, grad_b) = {
        let mut found = None;
        for (idx, name, layer) in model.lora_manager.as_mut().unwrap().adapters_iter_mut() {
            if idx == 0 && name == "q_proj" {
                layer.backward(&d_hidden, &cached_input, seq);
                let (ga, gb) = layer.gradients();
                found = Some((ga.to_vec(), gb.to_vec()));
            }
        }
        found.expect("layer 0 q_proj adapter missing")
    };

    compare(&mut model, &r, grad_a, grad_b);
}

fn compare(model: &mut CpuBlockAttnResModel, r: &[f32], grad_a: Vec<f32>, grad_b: Vec<f32>) {
    let q_dim = HEADS * HEAD_DIM;
    let eps = 1e-3f32;

    // Sample a few elements of A and B.
    let a_idx = [0usize, 1, RANK * HIDDEN / 2];
    let b_idx = [0usize, 1, q_dim * RANK / 2];

    let mut max_rel = 0.0f32;
    for &i in &a_idx {
        let orig = {
            let mut v = 0.0;
            for (idx, name, layer) in model.lora_manager.as_ref().unwrap().adapters_iter() {
                if idx == 0 && name == "q_proj" {
                    v = layer.lora_a()[i];
                }
            }
            v
        };
        set_adapter_elem(model, true, i, orig + eps);
        let lp = loss(model, r);
        set_adapter_elem(model, true, i, orig - eps);
        let lm = loss(model, r);
        set_adapter_elem(model, true, i, orig);
        let fd = (lp - lm) / (2.0 * eps);
        let rel = (fd - grad_a[i]).abs() / fd.abs().max(1e-4);
        println!("A[{}]: analytic {:.6e}  finite-diff {:.6e}  rel {:.3e}", i, grad_a[i], fd, rel);
        max_rel = max_rel.max(rel);
    }
    for &i in &b_idx {
        let orig = {
            let mut v = 0.0;
            for (idx, name, layer) in model.lora_manager.as_ref().unwrap().adapters_iter() {
                if idx == 0 && name == "q_proj" {
                    v = layer.lora_b()[i];
                }
            }
            v
        };
        set_adapter_elem(model, false, i, orig + eps);
        let lp = loss(model, r);
        set_adapter_elem(model, false, i, orig - eps);
        let lm = loss(model, r);
        set_adapter_elem(model, false, i, orig);
        let fd = (lp - lm) / (2.0 * eps);
        let rel = (fd - grad_b[i]).abs() / fd.abs().max(1e-4);
        println!("B[{}]: analytic {:.6e}  finite-diff {:.6e}  rel {:.3e}", i, grad_b[i], fd, rel);
        max_rel = max_rel.max(rel);
    }

    println!("max relative error of the distillation LoRA gradient: {:.3e}", max_rel);
    assert!(
        max_rel <= 1e-2,
        "distillation LoRA gradient is wrong in a blocked config: max relative error {:.3e}",
        max_rel
    );
}
