//! Capability descriptors and load-time negotiation.
//!
//! Three axes decide what the engine can do for one request:
//!
//! - **Model**: what the checkpoint declares (this module reads the header).
//! - **Device/runtime**: what the adapter can do, and which external models
//!   (a draft model) the runtime has loaded.
//! - **Workload**: what the request needs (context, tools, draft, modality).
//!
//! The engine runs the intersection. Anything outside it is a typed refusal or
//! a named fallback, never a silent different result.
//!
//! Nothing here mutates a model or a device. `resolve` is a pure function of the
//! three descriptors.

use crate::device::Capability;
use crate::model::checkpoint::BlockMoeResConfig;
use crate::model::tiers::Tier;

/// Context assumed when a checkpoint does not declare one.
pub const DEFAULT_CONTEXT_MAX: usize = 32768;

/// KV cache bit width assumed when a checkpoint does not declare one.
pub const DEFAULT_KV_BITS: u8 = 3;

/// Smallest shared-RAM budget that may run top-k routing.
pub const TOP_K_MIN_SHARED_RAM_MB: u64 = 4096;

/// Two blocks of 1024 tokens is the smallest useful block-sparse context.
pub const BLOCK_SPARSE_MIN_CONTEXT: usize = 2 * 1024;

/// What one checkpoint declares about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelFeatures {
    /// `"block_attnres"` or `"standard"`.
    pub arch: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    /// Layer executions: `prelude + core * passes + coda`.
    pub visits: usize,
    pub experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub kv_bits: u8,
    pub context_max: usize,
    pub looped: bool,
    pub block_sparse: bool,
    pub mtp: bool,
    pub vision: bool,
    pub audio: bool,
    /// The checkpoint ships a decision head and may answer `Workload::Decision`.
    pub decision: bool,
    pub tier: Option<Tier>,
}

impl ModelFeatures {
    /// Derive the descriptor from a checkpoint header.
    ///
    /// Numeric fields always come from the config. Boolean capabilities come
    /// from the declared `features` list when it is present (the checkpoint is
    /// authoritative), and are derived from the architecture when it is absent
    /// (old checkpoints). `tier` is only known when declared, because a tier is
    /// a preset used at build time, not something weights reveal.
    pub fn from_config(config: &BlockMoeResConfig) -> Self {
        let declared = config.features.as_deref();
        let has = |name: &str| declared.map_or(false, |f| f.iter().any(|s| s == name));

        let declared_arch = declared.and_then(|f| {
            f.iter()
                .find_map(|s| s.strip_prefix("arch:"))
                .map(|s| s.to_string())
        });
        let derived_arch = if config.boundary_layers.is_empty() {
            "standard"
        } else {
            "block_attnres"
        };

        let experts = config
            .layer_configs
            .iter()
            .map(|l| l.num_experts)
            .max()
            .unwrap_or(0);
        // Tiers are top-1 by design. A header that explicitly declares more
        // keeps it, so the device budget rule still has something to refuse.
        let declared_top_k = config
            .layer_configs
            .iter()
            .map(|l| l.top_k)
            .max()
            .unwrap_or(0);
        let top_k = declared_top_k.max(1);

        // The loop is described by the block structure: every block boundary
        // adds one pass over the core layers. `num_blocks` and the boundary
        // list should agree; take the larger so a header that only carries one
        // of them still yields a looped descriptor.
        let passes = config
            .num_blocks
            .max(config.boundary_layers.len())
            .max(1);
        let visits = config.num_layers
            + passes.saturating_sub(1) * config.layers_per_block;

        let tier = declared.and_then(|f| {
            f.iter()
                .find_map(|s| s.strip_prefix("tier:"))
                .and_then(Tier::parse)
        });

        let (looped, block_sparse) = match declared {
            Some(_) => (has("looped"), has("block_sparse")),
            None => (passes > 1, !config.boundary_layers.is_empty()),
        };

        Self {
            arch: declared_arch.unwrap_or_else(|| derived_arch.to_string()),
            hidden_dim: config.hidden_dim,
            num_layers: config.num_layers,
            visits,
            experts,
            top_k,
            vocab_size: config.vocab_size,
            kv_bits: DEFAULT_KV_BITS,
            context_max: config.context_max.unwrap_or(DEFAULT_CONTEXT_MAX),
            looped,
            block_sparse,
            mtp: has("mtp"),
            vision: has("vision"),
            audio: has("audio"),
            decision: has("decision"),
            tier,
        }
    }

    /// The descriptor of a tier preset, before any checkpoint exists.
    pub fn from_tier(tier: Tier) -> Self {
        let spec = tier.spec();
        Self {
            arch: "block_attnres".to_string(),
            hidden_dim: spec.hidden,
            num_layers: spec.layers,
            visits: spec.prelude + spec.core * spec.passes + spec.coda,
            experts: spec.experts,
            top_k: 1,
            vocab_size: spec.vocab,
            kv_bits: DEFAULT_KV_BITS,
            context_max: spec.context_max,
            looped: spec.passes > 1,
            block_sparse: true,
            mtp: false,
            vision: false,
            audio: false,
            decision: false,
            tier: Some(tier),
        }
    }

    /// The canonical strings written into a checkpoint header.
    ///
    /// Only capabilities are written; numeric dimensions stay in their own
    /// fields so there is exactly one source of truth for each.
    pub fn declared_features(&self) -> Vec<String> {
        let mut out = vec![format!("arch:{}", self.arch)];
        if self.looped {
            out.push("looped".to_string());
        }
        if self.block_sparse {
            out.push("block_sparse".to_string());
        }
        if self.mtp {
            out.push("mtp".to_string());
        }
        if self.vision {
            out.push("vision".to_string());
        }
        if self.audio {
            out.push("audio".to_string());
        }
        if self.decision {
            out.push("decision".to_string());
        }
        if let Some(tier) = self.tier {
            out.push(format!("tier:{}", tier.name()));
        }
        out
    }
}

/// What the runtime offers: the active adapter plus any external model a
/// request may use (currently a draft model).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFeatures {
    pub adapter: String,
    pub gpu_kind: String,
    pub shared_ram_mb: u64,
    pub coop_matrix: bool,
    pub f16: bool,
    pub immediates: bool,
    pub max_storage_buffer: u64,
    /// A draft checkpoint is loaded and can be used by `Workload::Draft`.
    /// Drafting is a runtime arrangement (for example Nano drafting for
    /// Small/Base), not a property of the target checkpoint.
    pub draft_available: bool,
}

impl DeviceFeatures {
    /// Build a descriptor from the adapter information `Capability` reports.
    ///
    /// `Capability::detect()` alone does not query the adapter, so the three
    /// shader features stay `false`. A `Capability` produced by
    /// `WgpuCompute::detect_capability()` carries the real `wgpu::Features`
    /// flags.
    pub fn from_capability(cap: &Capability) -> Self {
        let has_adapter = !cap.adapter_name.trim().is_empty() || !cap.backend.trim().is_empty();
        Self {
            adapter: if has_adapter {
                cap.adapter_name.clone()
            } else {
                "no adapter".to_string()
            },
            gpu_kind: format!("{:?}", cap.gpu_kind).to_ascii_lowercase(),
            shared_ram_mb: cap.shared_ram_mb,
            coop_matrix: cap.coop_matrix,
            f16: cap.f16,
            immediates: cap.immediates,
            max_storage_buffer: cap.max_storage_buffer_range,
            draft_available: false,
        }
    }

    /// Whether an adapter was detected at all.
    pub fn has_adapter(&self) -> bool {
        self.adapter != "no adapter"
    }
}

/// What one request needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    Chat { context: usize },
    Rag { context: usize },
    Tools,
    Vision,
    Draft,
    Training,
    /// Answer bounded typed questions with a declared decision head.
    Decision,
}

impl Workload {
    /// The context the workload asks for, when it has one.
    pub fn context(self) -> Option<usize> {
        match self {
            Workload::Chat { context } | Workload::Rag { context } => Some(context),
            _ => None,
        }
    }
}

/// The outcome of intersecting the three axes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resolution {
    pub enabled: Vec<String>,
    pub fallbacks: Vec<(String, String)>,
    pub refusals: Vec<(String, String)>,
}

/// Intersect the three descriptors.
///
/// Every requested feature lands in exactly one of `enabled`, `fallbacks` or
/// `refusals`. A fallback is a named, different execution path; a refusal is a
/// request the engine will not silently satisfy.
pub fn resolve(model: &ModelFeatures, device: &DeviceFeatures, workload: &Workload) -> Resolution {
    let mut r = Resolution::default();
    let context = workload.context();

    // --- Model axis -------------------------------------------------------
    r.enabled.push(format!("arch:{}", model.arch));
    if model.looped {
        r.enabled.push("looped".to_string());
    }

    if model.block_sparse {
        match context {
            Some(c) if c < BLOCK_SPARSE_MIN_CONTEXT => {
                r.fallbacks.push((
                    "block_sparse".to_string(),
                    "context below two blocks".to_string(),
                ));
            }
            _ => r.enabled.push("block_sparse".to_string()),
        }
    }

    if model.experts > 0 && model.top_k > 0 {
        if model.top_k > 1 && device.shared_ram_mb < TOP_K_MIN_SHARED_RAM_MB {
            r.fallbacks.push((
                "top_k".to_string(),
                format!(
                    "top_k={} needs at least {} MB shared RAM (adapter reports {} MB); using top-1",
                    model.top_k, TOP_K_MIN_SHARED_RAM_MB, device.shared_ram_mb
                ),
            ));
            r.enabled.push("top_k:1".to_string());
        } else {
            r.enabled.push(format!("top_k:{}", model.top_k));
        }
    }

    if model.mtp {
        r.enabled.push("mtp".to_string());
    }
    if model.audio {
        r.enabled.push("audio".to_string());
    }
    if let Some(tier) = model.tier {
        r.enabled.push(format!("tier:{}", tier.name()));
    }

    // --- Device / runtime axis -------------------------------------------
    // LoRA adapters attach to any model with linear layers, like tools; they
    // are not a checkpoint restriction.
    r.enabled.push("lora".to_string());
    if device.coop_matrix {
        r.enabled.push("coop_matrix".to_string());
    } else {
        r.fallbacks.push((
            "coop_matrix".to_string(),
            "adapter has no cooperative matrix support; using scalar matmul".to_string(),
        ));
    }

    if device.f16 {
        r.enabled.push("f16".to_string());
    } else {
        r.fallbacks.push((
            "f16".to_string(),
            "adapter does not report f16 support; using f32 compute".to_string(),
        ));
    }

    if device.immediates {
        r.enabled.push("immediates".to_string());
    } else {
        r.fallbacks.push((
            "immediates".to_string(),
            "adapter does not report shader immediates; using uniform buffers".to_string(),
        ));
    }

    // --- Workload axis ----------------------------------------------------
    match *workload {
        Workload::Chat { context } => {
            r.enabled.push("chat".to_string());
            check_context(model, context, &mut r);
        }
        Workload::Rag { context } => {
            r.enabled.push("rag".to_string());
            check_context(model, context, &mut r);
        }
        Workload::Tools => {
            r.enabled.push("tools".to_string());
        }
        Workload::Vision => {
            if model.vision {
                r.enabled.push("vision".to_string());
            } else {
                r.refusals.push((
                    "vision".to_string(),
                    "checkpoint is text-only".to_string(),
                ));
            }
        }
        Workload::Draft => {
            if device.draft_available {
                r.enabled.push("draft".to_string());
            } else {
                r.refusals.push((
                    "draft".to_string(),
                    "no draft model is available".to_string(),
                ));
            }
        }
        Workload::Decision => {
            if model.decision {
                r.enabled.push("decision".to_string());
            } else {
                r.refusals.push((
                    "decision".to_string(),
                    "checkpoint declares no decision head".to_string(),
                ));
            }
        }
        Workload::Training => {
            r.enabled.push("training".to_string());
            if model.num_layers < 8 || model.experts < 4 {
                let reason = format!(
                    "tensor/pipeline parallel needs at least 8 layers and 4 experts \
                     (model has {} layers, {} experts)",
                    model.num_layers, model.experts
                );
                r.refusals
                    .push(("tensor_parallel".to_string(), reason.clone()));
                r.refusals
                    .push(("pipeline_parallel".to_string(), reason));
            } else {
                r.enabled.push("tensor_parallel".to_string());
                r.enabled.push("pipeline_parallel".to_string());
            }
        }
    }

    r
}

fn check_context(model: &ModelFeatures, context: usize, r: &mut Resolution) {
    if context > model.context_max {
        r.refusals.push((
            "context".to_string(),
            format!(
                "requested context {} exceeds checkpoint maximum {}",
                context, model.context_max
            ),
        ));
    } else {
        r.enabled.push(format!("context:{}", context));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_model() -> ModelFeatures {
        ModelFeatures {
            arch: "block_attnres".to_string(),
            hidden_dim: 64,
            num_layers: 4,
            visits: 8,
            experts: 2,
            top_k: 2,
            vocab_size: 256,
            kv_bits: 3,
            context_max: 32768,
            looped: true,
            block_sparse: true,
            mtp: false,
            vision: false,
            audio: false,
            decision: false,
            tier: Some(Tier::Base),
        }
    }

    fn device(shared_ram_mb: u64, coop: bool) -> DeviceFeatures {
        DeviceFeatures {
            adapter: "test".to_string(),
            gpu_kind: "discrete".to_string(),
            shared_ram_mb,
            coop_matrix: coop,
            f16: true,
            immediates: true,
            max_storage_buffer: 1 << 30,
            draft_available: false,
        }
    }

    #[test]
    fn vision_is_refused_on_a_text_checkpoint() {
        let r = resolve(&text_model(), &device(16384, true), &Workload::Vision);
        assert!(r
            .refusals
            .iter()
            .any(|(f, why)| f == "vision" && why.contains("text-only")));
        assert!(!r.enabled.iter().any(|f| f == "vision"));
    }

    #[test]
    fn vision_is_enabled_when_declared() {
        let mut m = text_model();
        m.vision = true;
        let r = resolve(&m, &device(16384, true), &Workload::Vision);
        assert!(r.enabled.iter().any(|f| f == "vision"));
        assert!(r.refusals.is_empty());
    }

    #[test]
    fn draft_is_refused_when_no_draft_model_is_available() {
        let r = resolve(&text_model(), &device(16384, true), &Workload::Draft);
        assert!(r
            .refusals
            .iter()
            .any(|(f, why)| f == "draft" && why.contains("no draft model")));
        assert!(!r.enabled.iter().any(|f| f == "draft"));
    }

    #[test]
    fn draft_is_enabled_when_the_runtime_has_one() {
        let mut d = device(16384, true);
        d.draft_available = true;
        let r = resolve(&text_model(), &d, &Workload::Draft);
        assert!(r.enabled.iter().any(|f| f == "draft"));
        assert!(r.refusals.is_empty());
    }

    #[test]
    fn lora_is_a_runtime_capability_on_every_workload() {
        for workload in [
            Workload::Chat { context: 8192 },
            Workload::Rag { context: 8192 },
            Workload::Tools,
            Workload::Vision,
            Workload::Draft,
            Workload::Training,
            Workload::Decision,
        ] {
            let r = resolve(&text_model(), &device(16384, true), &workload);
            assert!(
                r.enabled.iter().any(|f| f == "lora"),
                "lora missing for {:?}",
                workload
            );
        }
    }

    #[test]
    fn rag_and_tools_add_their_tokens() {
        let rag = resolve(
            &text_model(),
            &device(16384, true),
            &Workload::Rag { context: 8192 },
        );
        assert!(rag.enabled.iter().any(|f| f == "rag"));
        assert!(rag.enabled.iter().any(|f| f == "context:8192"));

        let tools = resolve(&text_model(), &device(16384, true), &Workload::Tools);
        assert!(tools.enabled.iter().any(|f| f == "tools"));
    }

    #[test]
    fn f16_and_immediates_fall_back_when_absent() {
        let mut d = device(16384, true);
        d.f16 = false;
        d.immediates = false;
        let r = resolve(&text_model(), &d, &Workload::Chat { context: 8192 });
        assert!(r
            .fallbacks
            .iter()
            .any(|(f, why)| f == "f16" && why.contains("f32")));
        assert!(r
            .fallbacks
            .iter()
            .any(|(f, why)| f == "immediates" && why.contains("uniform buffers")));
        assert!(!r.enabled.iter().any(|f| f == "f16"));
        assert!(!r.enabled.iter().any(|f| f == "immediates"));
    }

    #[test]
    fn tier_presets_are_top_one() {
        for tier in [Tier::Nano, Tier::Small, Tier::Base, Tier::Large] {
            assert_eq!(ModelFeatures::from_tier(tier).top_k, 1, "{:?}", tier);
        }
    }

    #[test]
    fn top_k_falls_back_below_the_ram_budget() {
        let r = resolve(
            &text_model(),
            &device(TOP_K_MIN_SHARED_RAM_MB - 1, true),
            &Workload::Chat { context: 8192 },
        );
        let (_, why) = r
            .fallbacks
            .iter()
            .find(|(f, _)| f == "top_k")
            .expect("top_k fallback");
        assert!(why.contains("4096"));
        assert!(r.enabled.iter().any(|f| f == "top_k:1"));
        assert!(!r.enabled.iter().any(|f| f == "top_k:2"));
    }

    #[test]
    fn top_k_is_enabled_at_the_ram_budget() {
        let r = resolve(
            &text_model(),
            &device(TOP_K_MIN_SHARED_RAM_MB, true),
            &Workload::Chat { context: 8192 },
        );
        assert!(!r.fallbacks.iter().any(|(f, _)| f == "top_k"));
        assert!(r.enabled.iter().any(|f| f == "top_k:2"));
    }

    #[test]
    fn missing_coop_matrix_falls_back_to_scalar_matmul() {
        let r = resolve(
            &text_model(),
            &device(16384, false),
            &Workload::Chat { context: 8192 },
        );
        let (_, why) = r
            .fallbacks
            .iter()
            .find(|(f, _)| f == "coop_matrix")
            .expect("coop fallback");
        assert!(why.contains("scalar matmul"));
        assert!(!r.enabled.iter().any(|f| f == "coop_matrix"));
    }

    #[test]
    fn block_sparse_falls_back_to_dense_below_two_blocks() {
        let r = resolve(
            &text_model(),
            &device(16384, true),
            &Workload::Chat {
                context: BLOCK_SPARSE_MIN_CONTEXT - 1,
            },
        );
        assert!(r
            .fallbacks
            .iter()
            .any(|(f, why)| f == "block_sparse" && why == "context below two blocks"));
        assert!(!r.refusals.iter().any(|(f, _)| f == "block_sparse"));
    }

    #[test]
    fn block_sparse_stays_on_at_two_blocks() {
        let r = resolve(
            &text_model(),
            &device(16384, true),
            &Workload::Chat {
                context: BLOCK_SPARSE_MIN_CONTEXT,
            },
        );
        assert!(r.enabled.iter().any(|f| f == "block_sparse"));
        assert!(!r.fallbacks.iter().any(|(f, _)| f == "block_sparse"));
    }

    #[test]
    fn parallel_training_is_refused_on_small_models() {
        let r = resolve(&text_model(), &device(16384, true), &Workload::Training);
        assert!(r
            .refusals
            .iter()
            .any(|(f, _)| f == "tensor_parallel"));
        assert!(r
            .refusals
            .iter()
            .any(|(f, _)| f == "pipeline_parallel"));
    }

    #[test]
    fn parallel_training_is_enabled_on_large_models() {
        let mut m = text_model();
        m.num_layers = 12;
        m.experts = 4;
        let r = resolve(&m, &device(16384, true), &Workload::Training);
        assert!(r.enabled.iter().any(|f| f == "tensor_parallel"));
        assert!(r.enabled.iter().any(|f| f == "pipeline_parallel"));
        assert!(r.refusals.is_empty());
    }

    #[test]
    fn context_beyond_the_checkpoint_maximum_is_refused() {
        let r = resolve(
            &text_model(),
            &device(16384, true),
            &Workload::Chat { context: 32769 },
        );
        assert!(r
            .refusals
            .iter()
            .any(|(f, why)| f == "context" && why.contains("32768")));
        assert!(!r.enabled.iter().any(|f| f.starts_with("context:")));
    }

    #[test]
    fn declared_features_are_the_capability_vocabulary() {
        let mut m = text_model();
        m.vision = true;
        m.mtp = true;
        assert_eq!(
            m.declared_features(),
            vec![
                "arch:block_attnres",
                "looped",
                "block_sparse",
                "mtp",
                "vision",
                "tier:base",
            ]
        );
    }

    #[test]
    fn decision_is_refused_without_a_declared_head() {
        let r = resolve(&text_model(), &device(16384, true), &Workload::Decision);
        assert!(r
            .refusals
            .iter()
            .any(|(f, why)| f == "decision" && why.contains("no decision head")));
        assert!(!r.enabled.iter().any(|f| f == "decision"));
    }

    #[test]
    fn decision_is_enabled_when_the_checkpoint_declares_it() {
        let mut m = text_model();
        m.decision = true;
        assert!(m.declared_features().iter().any(|f| f == "decision"));
        let r = resolve(&m, &device(16384, true), &Workload::Decision);
        assert!(r.enabled.iter().any(|f| f == "decision"));
        assert!(
            !r.refusals.iter().any(|(f, _)| f == "decision"),
            "{:?}",
            r
        );
    }

    #[test]
    fn tier_presets_do_not_declare_a_decision_head() {
        for tier in [Tier::Nano, Tier::Small, Tier::Base, Tier::Large] {
            let features = ModelFeatures::from_tier(tier);
            assert!(!features.decision, "{:?}", tier);
            assert!(!features.declared_features().iter().any(|f| f == "decision"));
        }
    }

    #[test]
    fn device_from_capability_is_conservative_without_an_adapter() {        let cap = Capability {
            vram_mb: 0,
            shared_ram_mb: 8192,
            gpu_kind: crate::device::capability::GpuKind::Other,
            vendor: crate::device::capability::GpuVendor::Unknown,
            max_compute_workgroup_size: 0,
            max_compute_invocations_per_workgroup: 0,
            max_storage_buffer_range: 0,
            max_storage_buffers_per_shader_stage: 0,
            max_bind_groups: 0,
            backend: String::new(),
            adapter_name: String::new(),
            coop_matrix: false,
            f16: false,
            immediates: false,
        };
        let d = DeviceFeatures::from_capability(&cap);
        assert!(!d.has_adapter());
        assert_eq!(d.adapter, "no adapter");
        assert!(!d.coop_matrix);
        assert!(!d.f16);
        assert!(!d.immediates);
        assert!(!d.draft_available);
    }

    #[test]
    fn device_from_capability_uses_adapter_features() {
        let cap = Capability {
            vram_mb: 8192,
            shared_ram_mb: 16384,
            gpu_kind: crate::device::capability::GpuKind::Discrete,
            vendor: crate::device::capability::GpuVendor::Nvidia,
            max_compute_workgroup_size: 1024,
            max_compute_invocations_per_workgroup: 1024,
            max_storage_buffer_range: 1 << 30,
            max_storage_buffers_per_shader_stage: 8,
            max_bind_groups: 4,
            backend: "Vulkan".to_string(),
            adapter_name: "NVIDIA T4".to_string(),
            coop_matrix: true,
            f16: true,
            immediates: true,
        };
        let d = DeviceFeatures::from_capability(&cap);
        assert!(d.has_adapter());
        assert!(d.coop_matrix);
        assert!(d.f16);
        assert!(d.immediates);
        assert!(!d.draft_available);
    }
}
