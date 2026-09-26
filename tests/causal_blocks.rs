//! Causal depth-residual regressions with nonzero attention and multiple blocks.
use ferrisres::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel};
use ferrisres::model::cpu_linear::CpuLinear;
use ferrisres::inference::student_kv_cache::ModelKVCache;
use ferrisres::inference::cpu_generator::{CpuGenerateConfig, CpuTokenGenerator};

fn model() -> CpuBlockAttnResModel {
    let layers = (0..3).map(|i| {
        let mut layer = CpuBlockAttnResLayer::new(4, 1, 1, 4, 4);
        layer.layer_number = i;
        let identity = vec![1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
        layer.q_proj = CpuLinear::from_ternary(identity.clone(), 0.7, 4, 4);
        layer.k_proj = CpuLinear::from_ternary(identity.clone(), 0.8, 4, 4);
        layer.v_proj = CpuLinear::from_ternary(identity.clone(), 0.9, 4, 4);
        layer.out_proj = CpuLinear::from_ternary(identity, 0.6, 4, 4);
        layer
    }).collect();
    CpuBlockAttnResModel {
        layers, embed_tokens: vec![1.0,0.2,-0.4,0.6, -0.3,1.0,0.5,-0.2, 0.4,-0.7,1.0,0.1, -0.6,0.3,0.2,1.0],
        lm_head: vec![1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0],
        final_norm: vec![1.0;4], hidden_dim:4, vocab_size:4, num_layers:3,
        final_logit_softcapping: None, ple_model_projection:None, ple_projection_norm:None,
        embed_tokens_per_layer:None, hidden_size_per_layer_input:0, num_kv_shared_layers:0,
        block_config: BlockConfig { num_blocks:2, layers_per_block:2, boundary_layers:vec![0,2],
            attn_res_proj:vec![0.0;16], attn_res_norm:vec![1.0;4] }, lora_manager:None,
    }
}
fn cache() -> ModelKVCache { ModelKVCache::new(3,4,&[4,4,4],2,0) }
fn close(actual:&[f32], expected:&[f32]) {
    assert_eq!(actual.len(),expected.len());
    for (i,(a,b)) in actual.iter().zip(expected).enumerate() {
        assert!((a-b).abs()<2e-5, "index{i}: {a} != {b}");
    }
}

#[test]
fn full_training_and_routing_logits_are_prefix_invariant() {
    let m=model();
    let tokens=[0,1,2,3,1];
    let full=m.forward(&tokens);
    let train=m.forward_train(&tokens).logits;
    close(&train,&full);
    for n in 1..tokens.len() {
        let expected=&full[..n*4];
        close(&m.forward(&tokens[..n]),expected);
        close(&m.forward_train(&tokens[..n]).logits,expected);
        close(&m.forward_with_routing(&tokens[..n]).0,expected);
    }
}

#[test]
fn prefill_and_repeated_decode_match_full_forward() {
    let m=model();
    let tokens=[0,1,2,3,1,0];
    let mut kv=cache();
    close(&m.forward_prefill(&tokens[..2],&mut kv),&m.forward(&tokens[..2]));
    for n in 2..tokens.len() {
        let actual=m.forward_decode(tokens[n],&mut kv);
        let full=m.forward(&tokens[..=n]);
        close(&actual,&full[n*4..]);
        assert_eq!(kv.seq_len(),n+1);
        // Depth summaries must not grow with generated token count.
        assert!(kv.block_reps.len()<=m.block_config.num_blocks+1);
    }
}

#[test]
fn prefill_replaces_previous_request_cache() {
    let m=model(); let mut kv=cache();
    m.forward_prefill(&[0,1,2],&mut kv);
    close(&m.forward_prefill(&[3,1],&mut kv),&m.forward(&[3,1]));
    assert_eq!(kv.seq_len(),2);
    close(&m.forward_decode(0,&mut kv),&m.forward(&[3,1,0])[8..]);
}

#[test]
fn cached_attention_reads_all_prefix_values() {
    let layer=CpuBlockAttnResLayer::new(2,1,1,2,2);
    let output=layer.cpu_attention_raw(&[0.0,0.0],&[1.0,0.0,0.0,1.0],&[1.0,0.0,0.0,1.0],1,1,1,2,2,2);
    close(&output,&[0.5,0.5]);
}

#[test]
fn cpu_generator_matches_greedy_full_forward() {
    let reference=model();
    let mut tokens=vec![0,1];
    let mut expected=Vec::new();
    for _ in 0..4 {
        let logits=reference.forward(&tokens);
        let last=&logits[logits.len()-4..];
        let next=last.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0 as u32;
        expected.push(next); tokens.push(next);
    }
    let mut generator=CpuTokenGenerator::new(model(),32);
    let config=CpuGenerateConfig { temperature:0.0, max_tokens:4, ..Default::default() };
    assert_eq!(generator.generate(&[0,1],&config).unwrap(),expected);
}
