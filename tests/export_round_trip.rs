//! Round-trip a real `FRTRAIN1` checkpoint through `export`, then exercises
//! both the loader and deterministic CPU generation.
//!
//! Everything here is tiny, CPU-only and offline. No GPU, no network, no
//! tokenizer file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ferrisres::model::checkpoint::{self, BlockMoeResConfig};
use ferrisres::model::safetensors::MmapedSafetensors;
use ferrisres::training::export::{export_checkpoint, read_checkpoint};
use ferrisres::training::f32_trainer::F32Config;
use ferrisres::training::train_loop::{self, TrainOptions};

fn tiny_config() -> F32Config {
    F32Config {
        hidden: 32,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        intermediate: 64,
        vocab: 32,
        max_seq: 16,
        rope_theta: 10_000.0,
        eps: 1e-6,
    }
}

fn scratch_dir(tag: &str) -> PathBuf {
    let nanos =
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "ferrisres-export-{}-{}-{}",
        std::process::id(),
        tag,
        nanos
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Train a tiny checkpoint with the real loop. Returns the checkpoint path.
fn train_tiny(dir: &Path) -> PathBuf {
    let tokens: Vec<u32> = (0..64u32).map(|i| (i * 7) % 32).collect();
    let options = TrainOptions {
        config: tiny_config(),
        seq_len: 8,
        batch_size: 1,
        epochs: 1,
        learning_rate: 1e-2,
        seed: 7,
        checkpoint_every: 0,
        output: dir.join("tiny.bin"),
        resume: None,
        log_every: 0,
    };
    let report = train_loop::run(&options, &tokens).expect("tiny training run");
    assert!(report.steps_run > 0);
    assert!(report.params_changed, "the training loop must move parameters");
    report.checkpoint
}

/// The independent name mapping the test checks against. `layer_scalar` is
/// intentionally absent: it is carried by the config, not the weights.
fn export_name(name: &str) -> Option<String> {
    match name {
        "embed" => Some("embed_tokens".into()),
        "lm_head" => Some("lm_head".into()),
        "final_norm" => Some("final_norm".into()),
        "attn_res_norm" => Some("block_config.attn_res_norm".into()),
        _ => {
            let rest = name.strip_prefix("layer")?;
            let (layer, suffix) = rest.split_once('.')?;
            let mapped = match suffix {
                "attn_norm" => "attn_norm",
                "q" => "q_proj",
                "k" => "k_proj",
                "v" => "v_proj",
                "q_norm" => "q_norm",
                "k_norm" => "k_norm",
                "o" => "out_proj",
                "post_attn_norm" => "post_attn_norm",
                "pre_ffn_norm" => "pre_ffn_norm",
                "gate" => "ffn_gate",
                "up" => "ffn_up",
                "down" => "ffn_down",
                "post_ffn_norm" => "post_ffn_norm",
                "layer_scalar" => return None,
                other => panic!("unexpected checkpoint tensor suffix `{}`", other),
            };
            Some(format!("layers.{}.{}", layer, mapped))
        }
    }
}

fn read_header_dtypes(path: &Path) -> HashMap<String, String> {
    let bytes = std::fs::read(path).unwrap();
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    header
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, meta)| {
            let dtype = meta["dtype"].as_str().unwrap().to_string();
            (name, dtype)
        })
        .collect()
}

#[test]
fn every_checkpoint_tensor_round_trips_exactly() {
    let dir = scratch_dir("tensors");
    let checkpoint_path = train_tiny(&dir);
    let output = dir.join("model");
    let report = export_checkpoint(&checkpoint_path, &output).expect("export");

    let checkpoint = read_checkpoint(&checkpoint_path).expect("read checkpoint");
    let st_path = PathBuf::from(format!("{}.safetensors", output.display()));
    let mmap = MmapedSafetensors::open(&st_path).expect("open exported safetensors");

    let mut compared = 0usize;
    for (name, values) in &checkpoint.tensors {
        let Some(expected) = export_name(name) else {
            continue; // layer_scalar -> config
        };
        let got = mmap
            .get_tensor_f32(&expected)
            .unwrap_or_else(|e| panic!("exported tensor `{}` missing: {:?}", expected, e));
        assert_eq!(got.len(), values.len(), "length mismatch for `{}`", name);
        assert_eq!(
            got.as_slice(),
            values.as_slice(),
            "f32 data mismatch for `{}` -> `{}`",
            name,
            expected
        );
        compared += 1;
    }

    // Every checkpoint tensor except one layer_scalar per layer is in weights.
    assert_eq!(compared, checkpoint.tensors.len() - checkpoint.config.num_layers);

    // The synthetic projection is the identity, and every dtype is F32.
    let identity =
        mmap.get_tensor_f32("block_config.attn_res_proj").expect("synthetic attn_res_proj");
    let hidden = checkpoint.config.hidden;
    assert_eq!(identity.len(), hidden * hidden);
    for i in 0..hidden {
        for j in 0..hidden {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(identity[i * hidden + j], expected);
        }
    }

    let dtypes = read_header_dtypes(&st_path);
    for (name, dtype) in &dtypes {
        assert_eq!(dtype, "F32", "tensor `{}` must be lossless F32", name);
    }

    assert_eq!(report.tensor_count, dtypes.len());
    assert_eq!(
        report.parameter_count,
        identity.len()
            + checkpoint
                .tensors
                .iter()
                .filter(|(name, _)| export_name(name).is_some())
                .map(|(_, values)| values.len())
                .sum::<usize>()
    );
    assert_eq!(report.checkpoint_tensor_count, checkpoint.tensors.len());
    assert!(report.bytes_written > 0);
    assert!(report.safetensors_path.exists());
    assert!(report.config_path.exists());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn config_fields_round_trip_and_loader_accepts_them() {
    let dir = scratch_dir("config");
    let checkpoint_path = train_tiny(&dir);
    let output = dir.join("model");
    export_checkpoint(&checkpoint_path, &output).expect("export");
    let checkpoint = read_checkpoint(&checkpoint_path).expect("read checkpoint");

    let config_path = format!("{}.json", output.display());
    let config: BlockMoeResConfig =
        serde_json::from_str(&std::fs::read_to_string(&config_path).unwrap()).unwrap();
    let expected = tiny_config();

    assert_eq!(config.hidden_dim, expected.hidden);
    assert_eq!(config.vocab_size, expected.vocab);
    assert_eq!(config.num_layers, expected.num_layers);
    assert_eq!(config.num_heads, expected.num_heads);
    assert_eq!(config.num_kv_heads, expected.num_kv_heads);
    assert_eq!(config.head_dim, expected.head_dim);
    assert_eq!(config.final_logit_softcapping, None);
    assert_eq!(config.hidden_size_per_layer_input, 0);
    assert_eq!(config.num_kv_shared_layers, 0);
    assert_eq!(config.layer_configs.len(), expected.num_layers);
    for (layer_idx, layer) in config.layer_configs.iter().enumerate() {
        assert!(!layer.has_moe);
        assert!(!layer.has_ple);
        assert!(!layer.kv_shared);
        assert!(!layer.use_gelu);
        assert_eq!(layer.inter_dim, expected.intermediate);
        assert_eq!(layer.rope_theta, expected.rope_theta as f64);
        assert_eq!(layer.partial_rotary_factor, 1.0);

        let name = format!("layer{}.layer_scalar", layer_idx);
        let scalar = checkpoint
            .tensors
            .iter()
            .find(|(tensor_name, _)| tensor_name == &name)
            .expect("layer_scalar tensor");
        assert_eq!(scalar.1.len(), 1);
        assert_eq!(layer.layer_scalar, scalar.1[0]);
    }

    let model = checkpoint::load_model(&output.to_string_lossy()).expect("load_model");
    assert_eq!(model.hidden_dim, expected.hidden);
    assert_eq!(model.vocab_size, expected.vocab);
    assert_eq!(model.num_layers, expected.num_layers);
    assert_eq!(model.layers.len(), expected.num_layers);
    for layer in &model.layers {
        assert_eq!(layer.num_heads, expected.num_heads);
        assert_eq!(layer.num_kv_heads, expected.num_kv_heads);
        assert_eq!(layer.head_dim, expected.head_dim);
        assert_eq!(layer.q_proj.in_features(), expected.hidden);
        assert_eq!(layer.q_proj.out_features(), expected.num_heads * expected.head_dim);
        assert_eq!(layer.ffn_gate.as_ref().unwrap().out_features(), expected.intermediate);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn exported_model_generates_deterministically_on_cpu() {
    let dir = scratch_dir("generate");
    let checkpoint_path = train_tiny(&dir);
    let output = dir.join("model");
    export_checkpoint(&checkpoint_path, &output).expect("export");

    let model = checkpoint::load_model(&output.to_string_lossy()).expect("load_model");
    let vocab = model.vocab_size;
    let mut ids: Vec<u32> = vec![1, 2, 3, 4];
    let prompt_len = ids.len();

    for _ in 0..4 {
        let logits = model.forward(&ids);
        assert_eq!(logits.len(), ids.len() * vocab);
        assert!(logits.iter().all(|v| v.is_finite()), "logits must be finite");
        let last = &logits[(ids.len() - 1) * vocab..ids.len() * vocab];
        let (next, _) =
            last.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        assert!((next as usize) < vocab);
        ids.push(next as u32);
    }

    assert_eq!(ids.len(), prompt_len + 4, "must generate at least a few tokens");
    assert!(ids[prompt_len..].iter().all(|&id| (id as usize) < vocab));

    // Greedy decoding from the same prompt is deterministic.
    let run = |prompt: &[u32]| -> Vec<u32> {
        let mut ids = prompt.to_vec();
        for _ in 0..3 {
            let logits = model.forward(&ids);
            let last = &logits[(ids.len() - 1) * vocab..ids.len() * vocab];
            let next =
                last.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
                    as u32;
            ids.push(next);
        }
        ids
    };
    assert_eq!(run(&[5, 6, 7]), run(&[5, 6, 7]));

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn export_refuses_unrepresentable_eps_with_a_named_error() {
    let dir = scratch_dir("refuse");
    let mut options = TrainOptions {
        config: tiny_config(),
        seq_len: 8,
        batch_size: 1,
        epochs: 1,
        learning_rate: 1e-2,
        seed: 3,
        checkpoint_every: 0,
        output: dir.join("tiny.bin"),
        resume: None,
        log_every: 0,
    };
    options.config.eps = 1e-5;
    let tokens: Vec<u32> = (0..64u32).map(|i| (i * 3) % 32).collect();
    let report = train_loop::run(&options, &tokens).expect("tiny training run");

    let err = export_checkpoint(&report.checkpoint, &dir.join("model")).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("export: eps_not_representable"),
        "expected a named error, got `{}`",
        text
    );

    std::fs::remove_dir_all(&dir).ok();
}
