//! Finite-difference audit of the distillation/LoRA gradient in a blocked
//! config.
//!
//! `CpuBlockAttnResModel::forward_train` refuses to run with block boundaries
//! and a LoRA manager because the gradient code that consumes its
//! `LayerActivations` (`compute_lora_grad` in `src/main.rs`, and the
//! equivalent `LoraLayer::backward`) uses the per-position `d_hidden` as the
//! gradient for every projection. With causal inter-block attention that is
//! wrong: the block summary at position `t` is a prefix sum over `0..=t`, so a
//! parameter in an early layer receives a suffix-sum gradient from every later
//! boundary position. The path also skips the final-norm, attention and
//! post-attention-norm Jacobians and reuses one `d_hidden` for all layers.
//!
//! [`max_rel_error_blocked_lora`] measures the defect: it builds a small
//! blocked model with a rank-2 LoRA adapter, computes the gradient the
//! distillation path would produce (using the same `forward_train` body via
//! `forward_train_impl` and `LoraLayer::backward`), and compares every sampled
//! element with a central finite difference of the same loss. The value is
//! large while the defect is unfixed and would fall to ~0 once the gradient is
//! correct.

use crate::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel};
use crate::model::cpu_linear::{CpuLinear, CpuRmsNorm};
use crate::training::lora::LoraConfig;

const HIDDEN: usize = 16;
const HEADS: usize = 2;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const INTERMEDIATE: usize = 32;
const VOCAB: usize = 8;
const NUM_BLOCKS: usize = 2;
const BLOCK_SIZE: usize = 2;
const RANK: usize = 2;
const TOKENS: [u32; 4] = [1, 2, 3, 4];

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
}

/// Build the blocked model used by the audit and by the gate's `should_panic`
/// test. It has two blocks of two layers and a rank-2 LoRA adapter on layer
/// 0's `q_proj` with non-zero A and B (B is zero-initialised by `LoraLayer`).
pub fn blocked_lora_model() -> CpuBlockAttnResModel {
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
        l.ffn_gate = Some(CpuLinear::from_weight(
            rng.vec(HIDDEN * INTERMEDIATE, 0.6),
            HIDDEN,
            INTERMEDIATE,
        ));
        l.ffn_up = Some(CpuLinear::from_weight(
            rng.vec(HIDDEN * INTERMEDIATE, 0.6),
            HIDDEN,
            INTERMEDIATE,
        ));
        l.ffn_down = Some(CpuLinear::from_weight(
            rng.vec(INTERMEDIATE * HIDDEN, 0.6),
            INTERMEDIATE,
            HIDDEN,
        ));
        l.post_ffn_norm = CpuRmsNorm::from_weight(vec![1.0; HIDDEN], 1e-6);
        layers.push(l);
    }

    let boundary_layers = (0..NUM_BLOCKS)
        .map(|b| (b + 1) * BLOCK_SIZE - 1)
        .collect();
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

    // Give layer 0's q_proj adapter non-zero A and B.
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
    Rng::new(0xfeed).vec(TOKENS.len() * VOCAB, 1.0)
}

fn loss(model: &CpuBlockAttnResModel, r: &[f32]) -> f32 {
    // The audit needs the LoRA-aware forward; `forward_train` refuses a blocked
    // config, so use its body directly. This is the same forward whose
    // activations the gradient path consumes.
    let out = model.forward_train_impl(&TOKENS);
    out.logits.iter().zip(r).map(|(l, w)| l * w).sum()
}

fn adapter_elem(model: &CpuBlockAttnResModel, is_a: bool, index: usize) -> f32 {
    for (idx, name, layer) in model.lora_manager.as_ref().unwrap().adapters_iter() {
        if idx == 0 && name == "q_proj" {
            return if is_a {
                layer.lora_a()[index]
            } else {
                layer.lora_b()[index]
            };
        }
    }
    panic!("layer 0 q_proj adapter missing");
}

fn set_adapter_elem(model: &mut CpuBlockAttnResModel, is_a: bool, index: usize, value: f32) {
    for (idx, name, layer) in model.lora_manager.as_mut().unwrap().adapters_iter_mut() {
        if idx == 0 && name == "q_proj" {
            if is_a {
                layer.lora_a_mut()[index] = value;
            } else {
                layer.lora_b_mut()[index] = value;
            }
            return;
        }
    }
    panic!("layer 0 q_proj adapter missing");
}

/// The gradient the distillation path produces for layer 0's `q_proj` LoRA
/// adapter: `d_hidden = d_logits @ lm_head^T` (no final-norm Jacobian), then
/// `LoraLayer::backward(d_hidden, pre_attn_normed)` — the same formula as
/// `compute_lora_grad` in `src/main.rs`.
fn surrogate_gradient(
    model: &mut CpuBlockAttnResModel,
    r: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let seq = TOKENS.len();
    let fwd = model.forward_train_impl(&TOKENS);
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
    for (idx, name, layer) in model.lora_manager.as_mut().unwrap().adapters_iter_mut() {
        if idx == 0 && name == "q_proj" {
            layer.backward(&d_hidden, &cached_input, seq);
            let (ga, gb) = layer.gradients();
            return (ga.to_vec(), gb.to_vec());
        }
    }
    panic!("layer 0 q_proj adapter missing");
}

/// Max relative error between the distillation LoRA gradient and a central
/// finite difference, over three sampled elements of A and three of B.
///
/// The value is large (order 1) while the blocked-config gradient defect
/// exists. A correct implementation returns a value below `1e-2`.
pub fn max_rel_error_blocked_lora() -> f32 {
    let mut model = blocked_lora_model();
    let r = loss_weights();
    let (grad_a, grad_b) = surrogate_gradient(&mut model, &r);

    let q_dim = HEADS * HEAD_DIM;
    let eps = 1e-3f32;
    let a_idx = [0usize, 1, RANK * HIDDEN / 2];
    let b_idx = [0usize, 1, q_dim * RANK / 2];
    let mut max_rel = 0.0f32;

    for &i in &a_idx {
        let orig = adapter_elem(&model, true, i);
        set_adapter_elem(&mut model, true, i, orig + eps);
        let lp = loss(&model, &r);
        set_adapter_elem(&mut model, true, i, orig - eps);
        let lm = loss(&model, &r);
        set_adapter_elem(&mut model, true, i, orig);
        let fd = (lp - lm) / (2.0 * eps);
        let rel = (fd - grad_a[i]).abs() / fd.abs().max(1e-4);
        max_rel = max_rel.max(rel);
    }
    for &i in &b_idx {
        let orig = adapter_elem(&model, false, i);
        set_adapter_elem(&mut model, false, i, orig + eps);
        let lp = loss(&model, &r);
        set_adapter_elem(&mut model, false, i, orig - eps);
        let lm = loss(&model, &r);
        set_adapter_elem(&mut model, false, i, orig);
        let fd = (lp - lm) / (2.0 * eps);
        let rel = (fd - grad_b[i]).abs() / fd.abs().max(1e-4);
        max_rel = max_rel.max(rel);
    }
    max_rel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_fixture_has_blocks_and_lora() {
        let model = blocked_lora_model();
        assert!(!model.block_config.boundary_layers.is_empty());
        assert!(model.lora_manager.is_some());
    }
}
