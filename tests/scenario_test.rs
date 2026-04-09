/// Scenario / E2E-style integration tests for the FerrisRes inference pipeline.
///
/// These tests exercise the CPU-testable portions of the full inference pipeline:
///   tokenize → [embed*] → [forward*] → sample → decode
///
/// The steps marked with [*] require a live GPU (wgpu Device + Queue) and real model
/// weights loaded from disk.  Those steps are covered by `tests/integration_test.rs`
/// (GPU forward pass) when a GPU is available.  The tests below deliberately focus on
/// the CPU-side logic so they run in any CI environment without a GPU or model checkpoint.
///
/// GPU E2E NOTE:
///   A complete GPU end-to-end test (tokenize → embed → full model forward → lm_head →
///   sample → decode) requires:
///     1. A `WgpuCompute` instance (GPU adapter present).
///     2. Pre-loaded weight tensors (model checkpoint on disk).
///   Until those are available in CI, the full GPU path is covered by the existing
///   `test_block_attn_res_forward` test plus these CPU-mock scenario tests.

use ferrisres::inference::sampling;
use ferrisres::model::{BlockAttnResConfig, SimpleTokenizer};

// ---------------------------------------------------------------------------
// Stage 1 – Tokenizer
// ---------------------------------------------------------------------------

/// Encode a sample string and verify every token ID is within the vocabulary range.
#[test]
fn scenario_tokenize_ids_within_vocab_range() {
    let tokenizer = SimpleTokenizer::new();
    let vocab_size = tokenizer.vocab_size();

    let inputs = [
        "Hello, world!",
        "The quick brown fox",
        "",
        "Special chars: \n\t\r\0",
        "Unicode boundary: café",
    ];

    for text in &inputs {
        let ids = tokenizer.encode(text);
        for &id in &ids {
            assert!(
                (id as usize) < vocab_size,
                "Token ID {} out of vocab range {} for input {:?}",
                id,
                vocab_size,
                text
            );
        }
    }
}

/// Encode then decode a round-trip for pure ASCII input.
/// The SimpleTokenizer is byte-level, so ASCII round-trips exactly.
#[test]
fn scenario_tokenizer_roundtrip_ascii() {
    let tokenizer = SimpleTokenizer::new();

    let inputs = [
        "Hello, world!",
        "FerrisRes inference pipeline",
        "0123456789 abcdefghijklmnopqrstuvwxyz",
    ];

    for text in &inputs {
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode(&ids);
        assert_eq!(
            decoded, *text,
            "Round-trip failed for {:?}: got {:?}",
            text, decoded
        );
    }
}

/// Verify the EOS token ID is within the vocabulary.
#[test]
fn scenario_tokenizer_eos_token_in_vocab() {
    let tokenizer = SimpleTokenizer::new();
    let eos_id = tokenizer.eos_id();
    assert!(
        (eos_id as usize) < tokenizer.vocab_size(),
        "EOS token {} must be within vocab size {}",
        eos_id,
        tokenizer.vocab_size()
    );
}

/// Verify vocab size matches the expected byte-level count (3 special + 256 byte tokens).
#[test]
fn scenario_tokenizer_vocab_size_matches_spec() {
    let tokenizer = SimpleTokenizer::new();
    assert_eq!(
        tokenizer.vocab_size(),
        259,
        "Expected 259 tokens (3 special + 256 byte tokens)"
    );
}

// ---------------------------------------------------------------------------
// Stage 2 – Model config (minimal config, no GPU required)
// ---------------------------------------------------------------------------

/// Verify a minimal BlockAttnResConfig can be constructed and its derived fields
/// are self-consistent.
#[test]
fn scenario_model_config_fields_consistent() {
    let hidden_dim = 64;
    let mut config = BlockAttnResConfig::new(hidden_dim);
    config.num_blocks = 2;
    config.block_size = 2;
    config.num_layers = config.num_blocks * config.block_size;
    config.intermediate_dim = 4 * hidden_dim;

    assert_eq!(config.hidden_dim, hidden_dim);
    assert_eq!(config.num_layers, config.num_blocks * config.block_size);
    assert_eq!(config.total_layers(), config.num_blocks * config.block_size);
    assert_eq!(config.intermediate_dim, 4 * hidden_dim);
}

// ---------------------------------------------------------------------------
// Stage 3 – Sampling logic (mock logits, CPU only)
// ---------------------------------------------------------------------------

/// Argmax sampling always selects the index with the highest logit.
#[test]
fn scenario_sample_argmax_selects_max() {
    let logits = vec![0.1f32, 0.5, 0.2, 3.0, 0.8];
    let token = sampling::sample_argmax(&logits);
    assert_eq!(token, 3, "argmax should pick index 3 (value 3.0)");
}

/// Argmax on a single-element vocab returns index 0.
#[test]
fn scenario_sample_argmax_single_element() {
    let logits = vec![42.0f32];
    assert_eq!(sampling::sample_argmax(&logits), 0);
}

/// Temperature scaling does not change which token has the highest logit, so
/// argmax-after-temperature should still return the same token as plain argmax
/// (when the gap between logits is large enough).
#[test]
fn scenario_sample_temperature_preserves_argmax_on_clear_winner() {
    let mut logits = vec![0.1f32, 0.2, 0.1, 10.0, 0.3];
    let token = sampling::sample_temperature(&mut logits, 2.0);
    assert_eq!(token, 3, "temperature scaling should not change clear winner");
}

/// Temperature of 1.0 is a no-op – result equals plain argmax.
#[test]
fn scenario_sample_temperature_unity_is_noop() {
    let logits_orig = vec![0.1f32, 5.0, 0.2, 0.3, 0.1];
    let mut logits = logits_orig.clone();
    let token = sampling::sample_temperature(&mut logits, 1.0);
    assert_eq!(token, 1, "temperature=1.0 should give same result as argmax");
}

/// Top-k sampling with k=1 must deterministically select the best token.
#[test]
fn scenario_sample_top_k_k1_is_deterministic() {
    for _ in 0..10 {
        let mut logits = vec![0.1f32, 0.2, 8.0, 0.3, 0.1];
        let token = sampling::sample_top_k(&mut logits, 1);
        assert_eq!(token, 2, "top_k=1 must always select the highest logit index");
    }
}

/// Top-k sampling with k >= vocab_size falls back to full-vocab sampling;
/// the result is always a valid index.
#[test]
fn scenario_sample_top_k_full_vocab_returns_valid_index() {
    let vocab = 10;
    for _ in 0..20 {
        let mut logits: Vec<f32> = (0..vocab).map(|i| i as f32).collect();
        let token = sampling::sample_top_k(&mut logits, vocab + 5);
        assert!(
            token < vocab,
            "top_k(k > vocab) must still return a valid index, got {}",
            token
        );
    }
}

/// Top-p with p=1.0 (all mass) should still return a valid token index.
#[test]
fn scenario_sample_top_p_full_mass_returns_valid_index() {
    let vocab = 8;
    for _ in 0..20 {
        let mut logits: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.5).collect();
        let token = sampling::sample_top_p(&mut logits, 1.0);
        assert!(
            token < vocab,
            "top_p(1.0) must return a valid index, got {}",
            token
        );
    }
}

/// Top-p with a very small p value (concentrating on the top token) must
/// consistently select the token with the highest logit.
#[test]
fn scenario_sample_top_p_tiny_p_selects_best() {
    for _ in 0..10 {
        let mut logits = vec![0.1f32, 0.1, 0.1, 100.0, 0.1];
        // p very small so only the top logit covers the mass
        let token = sampling::sample_top_p(&mut logits, 0.001);
        assert_eq!(token, 3, "tiny p should collapse to the top token");
    }
}

// ---------------------------------------------------------------------------
// Stage 4 – Decode sampled token IDs back to text
// ---------------------------------------------------------------------------

/// A full CPU-mock pipeline pass:
///   1. Tokenize a prompt
///   2. Produce mock "logit" outputs (pretend the model ran)
///   3. Sample the next token using each strategy
///   4. Decode prompt + [sampled token] back to string
///   5. Verify the decoded string contains the original prompt text
///
/// GPU E2E NOTE: Replace the mock logit vector with real LM-head output to
/// turn this into a true GPU E2E test once WgpuCompute + model weights are
/// available in the test environment.
#[test]
fn scenario_cpu_mock_pipeline_end_to_end() {
    let tokenizer = SimpleTokenizer::new();
    let vocab_size = tokenizer.vocab_size();

    // Step 1 – Tokenize
    let prompt = "Hello";
    let prompt_ids = tokenizer.encode(prompt);
    assert!(!prompt_ids.is_empty(), "Prompt should produce at least one token");

    // Verify all prompt IDs are in range
    for &id in &prompt_ids {
        assert!(
            (id as usize) < vocab_size,
            "Prompt token {} out of vocab range {}",
            id,
            vocab_size
        );
    }

    // Step 2 – Mock logits (simulate model output: token 'W' = 0x57 → ID 0x57+3 = 90)
    let target_byte: u8 = b'W';
    let target_id = target_byte as usize + 3; // == 90
    let mut mock_logits: Vec<f32> = vec![0.0f32; vocab_size];
    mock_logits[target_id] = 100.0; // Make this token overwhelmingly likely

    // Step 3a – Argmax sampling
    let sampled_argmax = sampling::sample_argmax(&mock_logits);
    assert_eq!(
        sampled_argmax, target_id,
        "Argmax should pick the mock winner token"
    );

    // Step 3b – Temperature sampling (with a clear winner, still picks the same)
    let sampled_temp = sampling::sample_temperature(&mut mock_logits.clone(), 1.5);
    assert_eq!(sampled_temp, target_id, "Temperature sampling should pick mock winner");

    // Step 3c – Top-k sampling with k=1 (deterministic)
    let sampled_topk = sampling::sample_top_k(&mut mock_logits.clone(), 1);
    assert_eq!(sampled_topk, target_id, "top_k=1 should pick mock winner");

    // Step 3d – Top-p sampling (p=0.001 collapses to top token)
    let sampled_topp = sampling::sample_top_p(&mut mock_logits.clone(), 0.001);
    assert_eq!(sampled_topp, target_id, "top_p(tiny) should pick mock winner");

    // Step 4 – Decode
    let mut output_ids = prompt_ids.clone();
    output_ids.push(sampled_argmax as u32);

    let decoded = tokenizer.decode(&output_ids);
    assert!(
        decoded.starts_with(prompt),
        "Decoded output should start with the original prompt"
    );
    assert!(
        decoded.ends_with('W'),
        "Decoded output should end with the sampled 'W' character"
    );
}
