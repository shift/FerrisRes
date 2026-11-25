//! Cross-tier capability matrix.
//!
//! The same engine runs every tier; a checkpoint only declares its shape and
//! capabilities. These tests build one tiny checkpoint per tier (same loop
//! ratios, scaled dimensions), round-trip it through the header, and resolve
//! the three capability axes against two synthetic adapters.
//!
//! No test here needs a GPU adapter. The one test that inspects the real
//! adapter is `#[ignore]`.

use std::path::PathBuf;

use ferrisres::model::capabilities::{resolve, DeviceFeatures, ModelFeatures, Workload};
use ferrisres::model::checkpoint::{load_config, load_model, save_model, save_model_with_tier};
use ferrisres::model::cpu_block_attn_res::{BlockConfig, CpuBlockAttnResLayer, CpuBlockAttnResModel};
use ferrisres::model::cpu_linear::CpuRmsNorm;
use ferrisres::model::cpu_moe::CpuMoELayer;
use ferrisres::model::tiers::Tier;

const TIERS: [Tier; 4] = [Tier::Nano, Tier::Small, Tier::Base, Tier::Large];

const HIDDEN: usize = 64;
const VOCAB: usize = 256;
const INTERMEDIATE: usize = 128;
const HEADS: usize = 4;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;

/// Removes the temporary checkpoint directory when the test ends.
struct CheckpointDir(PathBuf);

impl CheckpointDir {
    fn new(tag: &str) -> (Self, String) {
        let dir = std::env::temp_dir().join(format!(
            "ferrisres-cap-{}-{}",
            std::process::id(),
            tag
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join("tiny").to_string_lossy().into_owned();
        (Self(dir), path)
    }
}

impl Drop for CheckpointDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Tiny unique layers per tier: 2/2/4/4.
fn tiny_layer_count(tier: Tier) -> usize {
    match tier {
        Tier::Nano | Tier::Small => 2,
        Tier::Base | Tier::Large => 4,
    }
}

/// Block boundaries that stay inside the tiny layer count.
fn tiny_boundaries(tier: Tier) -> Vec<usize> {
    let spec = tier.spec();
    let layers = tiny_layer_count(tier);
    let mut out: Vec<usize> = (1..=spec.passes)
        .map(|p| (p * spec.core).saturating_sub(1).min(layers - 1))
        .collect();
    out.dedup();
    out
}

fn tiny_model(tier: Tier) -> CpuBlockAttnResModel {
    tiny_model_with_top_k(tier, 1)
}

fn tiny_model_with_top_k(tier: Tier, top_k: usize) -> CpuBlockAttnResModel {
    let spec = tier.spec();
    let layers = tiny_layer_count(tier);

    let layers: Vec<CpuBlockAttnResLayer> = (0..layers)
        .map(|idx| {
            let mut layer =
                CpuBlockAttnResLayer::new(HIDDEN, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE);
            layer.layer_number = idx;
            // Per-head norms are [heads, head_dim]; keep the saved shape and
            // the stored data the same length.
            layer.q_norm = CpuRmsNorm::from_weight(vec![1.0; HEADS * HEAD_DIM], 1e-6);
            layer.k_norm = CpuRmsNorm::from_weight(vec![1.0; KV_HEADS * HEAD_DIM], 1e-6);
            // Tiers route top-1 at every size; this helper lets a test declare
            // more so the device budget rule can be exercised.
            layer.moe = Some(CpuMoELayer::new(
                HIDDEN,
                INTERMEDIATE,
                spec.experts,
                top_k,
            ));
            layer.ffn_gate = None;
            layer.ffn_up = None;
            layer.ffn_down = None;
            layer
        })
        .collect();

    CpuBlockAttnResModel {
        layers,
        embed_tokens: vec![0.01; VOCAB * HIDDEN],
        lm_head: vec![0.01; HIDDEN * VOCAB],
        final_norm: vec![1.0; HIDDEN],
        hidden_dim: HIDDEN,
        vocab_size: VOCAB,
        num_layers: tiny_layer_count(tier),
        final_logit_softcapping: None,
        ple_model_projection: None,
        ple_projection_norm: None,
        embed_tokens_per_layer: None,
        hidden_size_per_layer_input: 0,
        num_kv_shared_layers: 0,
        context_max: None,
        block_config: BlockConfig {
            num_blocks: spec.passes,
            layers_per_block: spec.core,
            boundary_layers: tiny_boundaries(tier),
            attn_res_proj: vec![0.0; HIDDEN * HIDDEN],
            attn_res_norm: vec![1.0; HIDDEN],
        },
        lora_manager: None,
    }
}

/// Save with a tier declaration and load the descriptor back.
fn saved_features(tier: Tier) -> (CheckpointDir, String, ModelFeatures) {
    let (dir, path) = CheckpointDir::new(&format!("roundtrip-{}", tier.name()));
    save_model_with_tier(&tiny_model(tier), &path, Some(tier)).expect("save checkpoint");
    let config = load_config(&path).expect("load header");
    (dir, path, ModelFeatures::from_config(&config))
}

/// Intel HD Graphics 530: integrated, no cooperative matrices, plenty of host
/// RAM.
fn hd530() -> DeviceFeatures {
    DeviceFeatures {
        adapter: "Intel HD Graphics 530".to_string(),
        gpu_kind: "integrated".to_string(),
        shared_ram_mb: 31_861,
        coop_matrix: false,
        f16: true,
        immediates: true,
        max_storage_buffer: 128 * 1024 * 1024,
        draft_available: false,
    }
}

/// NVIDIA T4: cooperative matrices. The shared-RAM figure is synthetic and
/// deliberately below the top-k budget so the RAM rule can be exercised by a
/// checkpoint that declares more than top-1.
fn t4() -> DeviceFeatures {
    DeviceFeatures {
        adapter: "NVIDIA T4".to_string(),
        gpu_kind: "discrete".to_string(),
        shared_ram_mb: 2_048,
        coop_matrix: true,
        f16: true,
        immediates: true,
        max_storage_buffer: 1024 * 1024 * 1024,
        draft_available: false,
    }
}

#[test]
fn features_roundtrip_for_every_tier() {
    for tier in TIERS {
        let (_dir, path, features) = saved_features(tier);

        // Weights load too, not just the header.
        let loaded = load_model(&path).expect("load weights");
        assert_eq!(loaded.num_layers, tiny_layer_count(tier));

        let config = load_config(&path).expect("load header");
        assert_eq!(config.format_version, 1, "{:?}", tier);
        let declared = config.features.clone().expect("declared features");

        // The declared set is idempotent: reading it back and re-serialising it
        // produces the same strings.
        assert_eq!(features.declared_features(), declared, "{:?}", tier);

        assert_eq!(features.tier, Some(tier), "{:?}", tier);
        assert_eq!(features.hidden_dim, HIDDEN, "{:?}", tier);
        assert_eq!(features.num_layers, tiny_layer_count(tier), "{:?}", tier);
        assert_eq!(features.experts, tier.spec().experts, "{:?}", tier);
        assert_eq!(features.top_k, 1, "{:?} must be top-1", tier);
        assert_eq!(features.vocab_size, VOCAB, "{:?}", tier);
        assert_eq!(
            features.visits,
            tiny_layer_count(tier) + (tier.spec().passes - 1) * tier.spec().core,
            "{:?}",
            tier
        );
        assert!(features.looped, "{:?}", tier);
        assert!(features.block_sparse, "{:?}", tier);
        assert_eq!(features.context_max, tier.spec().context_max, "{:?}", tier);
        assert_eq!(features.mtp, false, "{:?}", tier);
    }
}

#[test]
fn adapter_capabilities_select_fallbacks_per_tier() {
    for tier in TIERS {
        let (_dir, _path, features) = saved_features(tier);
        assert_eq!(features.top_k, 1, "{:?} is top-1", tier);

        let integrated = resolve(&features, &hd530(), &Workload::Chat { context: 8_192 });
        assert!(
            integrated
                .fallbacks
                .iter()
                .any(|(feature, _)| feature == "coop_matrix"),
            "{:?} HD-530 must fall back to scalar matmul: {:?}",
            tier,
            integrated
        );
        assert!(
            !integrated.fallbacks.iter().any(|(f, _)| f == "top_k"),
            "{:?} HD-530 has enough shared RAM for top-1: {:?}",
            tier,
            integrated
        );

        let discrete = resolve(&features, &t4(), &Workload::Chat { context: 8_192 });
        assert!(
            discrete.enabled.iter().any(|f| f == "coop_matrix"),
            "{:?} T4 keeps cooperative matrices: {:?}",
            tier,
            discrete
        );
        assert!(
            !discrete.fallbacks.iter().any(|(f, _)| f == "top_k"),
            "{:?} top-1 needs no top-k fallback: {:?}",
            tier,
            discrete
        );
        assert!(discrete.enabled.iter().any(|f| f == "top_k:1"));
    }
}

#[test]
fn a_checkpoint_that_declares_top_two_falls_back_on_low_ram() {
    let (_dir, path) = CheckpointDir::new("top-two");
    save_model(&tiny_model_with_top_k(Tier::Base, 2), &path).expect("save checkpoint");
    let features = ModelFeatures::from_config(&load_config(&path).expect("load header"));
    assert_eq!(features.top_k, 2, "the header declares top-2");

    let discrete = resolve(&features, &t4(), &Workload::Chat { context: 8_192 });
    let (_, why) = discrete
        .fallbacks
        .iter()
        .find(|(feature, _)| feature == "top_k")
        .unwrap_or_else(|| panic!("T4 must fall back to top-1: {:?}", discrete));
    assert!(why.contains("4096"), "{}", why);
    assert!(discrete.enabled.iter().any(|f| f == "top_k:1"));
    assert!(
        !discrete.refusals.iter().any(|(f, _)| f == "top_k"),
        "top-k is a fallback, never a refusal"
    );

    // The same declaration keeps top-2 on a device with enough shared RAM.
    let integrated = resolve(&features, &hd530(), &Workload::Chat { context: 8_192 });
    assert!(!integrated.fallbacks.iter().any(|(f, _)| f == "top_k"));
    assert!(integrated.enabled.iter().any(|f| f == "top_k:2"));
}

#[test]
fn vision_is_refused_on_every_text_tier() {
    for tier in TIERS {
        let (_dir, _path, features) = saved_features(tier);
        assert!(!features.vision, "{:?} is a text checkpoint", tier);

        let resolution = resolve(&features, &hd530(), &Workload::Vision);
        let (_, why) = resolution
            .refusals
            .iter()
            .find(|(feature, _)| feature == "vision")
            .unwrap_or_else(|| panic!("{:?} must refuse vision: {:?}", tier, resolution));
        assert!(why.contains("text-only"), "{:?}: {}", tier, why);
        assert!(!resolution.enabled.iter().any(|f| f == "vision"));
    }
}

#[test]
fn draft_is_a_runtime_choice() {
    let (_dir, _path, features) = saved_features(Tier::Base);

    let without = resolve(&features, &hd530(), &Workload::Draft);
    assert!(
        without
            .refusals
            .iter()
            .any(|(f, why)| f == "draft" && why.contains("no draft model")),
        "{:?}",
        without
    );
    assert!(!without.enabled.iter().any(|f| f == "draft"));

    let mut with_draft = hd530();
    with_draft.draft_available = true;
    let with = resolve(&features, &with_draft, &Workload::Draft);
    assert!(with.enabled.iter().any(|f| f == "draft"), "{:?}", with);
    assert!(with.refusals.is_empty(), "{:?}", with);
}

#[test]
fn short_context_falls_back_to_dense_on_every_tier() {
    for tier in TIERS {
        let (_dir, _path, features) = saved_features(tier);
        assert!(features.block_sparse, "{:?}", tier);

        let resolution = resolve(&features, &t4(), &Workload::Chat { context: 1_024 });
        let (_, why) = resolution
            .fallbacks
            .iter()
            .find(|(feature, _)| feature == "block_sparse")
            .unwrap_or_else(|| panic!("{:?} must fall back to dense: {:?}", tier, resolution));
        assert_eq!(why, "context below two blocks", "{:?}", tier);
        assert!(
            !resolution.refusals.iter().any(|(f, _)| f == "block_sparse"),
            "short context is a fallback, not a refusal"
        );
        assert!(!resolution.enabled.iter().any(|f| f == "block_sparse"));
    }
}

#[test]
fn old_header_derives_the_same_features() {
    for tier in TIERS {
        let (dir, path) = CheckpointDir::new(&format!("oldheader-{}", tier.name()));
        let model = tiny_model(tier);

        // New header, without a tier declaration, so the comparison is purely
        // between "declared" and "derived".
        save_model(&model, &path).expect("save checkpoint");
        let new_features = ModelFeatures::from_config(&load_config(&path).expect("new header"));

        // Rewrite the header as an old one: no features, no format_version,
        // no context_max. Weights are copied so load_model is exercised too.
        let json = std::fs::read_to_string(format!("{}.json", path)).expect("read json");
        let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse json");
        let object = value.as_object_mut().expect("object");
        object.remove("features");
        object.remove("format_version");
        object.remove("context_max");

        let old_path = dir.0.join("old").to_string_lossy().into_owned();
        std::fs::write(format!("{}.json", old_path), value.to_string()).expect("write old json");
        std::fs::copy(format!("{}.safetensors", path), format!("{}.safetensors", old_path))
            .expect("copy weights");

        let old_config = load_config(&old_path).expect("old header loads");
        assert_eq!(old_config.format_version, 0, "{:?}", tier);
        assert!(
            old_config.features.is_some(),
            "the loader derives features when absent"
        );
        let old_features = ModelFeatures::from_config(&old_config);
        assert_eq!(old_features, new_features, "{:?}", tier);
        assert_eq!(old_features.tier, None, "{:?}", tier);

        load_model(&old_path).expect("old checkpoint weights load");
    }
}

#[test]
fn declared_context_max_survives_a_resave() {
    let (dir, path) = CheckpointDir::new("context-resave");
    let mut model = tiny_model(Tier::Base);
    model.context_max = Some(131_072);

    save_model(&model, &path).expect("first save");
    let first = load_config(&path).expect("first header");
    assert_eq!(first.context_max, Some(131_072));

    let loaded = load_model(&path).expect("load weights");
    assert_eq!(loaded.context_max, Some(131_072));

    let again = dir.0.join("again").to_string_lossy().into_owned();
    save_model(&loaded, &again).expect("second save");
    let second = load_config(&again).expect("second header");
    assert_eq!(second.context_max, Some(131_072));
    assert_eq!(
        ModelFeatures::from_config(&second).context_max,
        131_072,
        "re-saving must not reset the declared context"
    );
}

#[test]
#[ignore = "requires a GPU adapter"]
fn detected_adapter_resolves_for_base() {
    let compute = pollster::block_on(ferrisres::WgpuCompute::new())
        .expect("a real adapter is required for this test");
    let device = DeviceFeatures::from_capability(&compute.detect_capability());
    let features = ModelFeatures::from_tier(Tier::Base);
    let context = 8_192.min(features.context_max);
    let resolution = resolve(&features, &device, &Workload::Chat { context });
    println!("device: {:?}", device);
    println!("resolution: {:?}", resolution);
}
