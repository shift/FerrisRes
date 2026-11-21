//! Declarative model-architecture registry.
//!
//! A family is described by a JSON data file in `architectures/`, not by a Rust
//! `impl`. The build script (`build.rs`) embeds every `architectures/*.json`
//! into the binary, so adding a family is a data change and needs no Rust edit.
//! An extra directory can be loaded at runtime with `FERRISRES_ARCHITECTURES_DIR`.
//!
//! # Descriptor shape
//!
//! ```json
//! {
//!   "name": "qwen3",
//!   "match_architectures": ["Qwen3ForCausalLM"],
//!   "match_gguf_architectures": ["qwen3"],
//!   "config_section": "text_config",
//!   "required": ["hidden_size", "num_hidden_layers"],
//!   "fields": {
//!     "hidden": "hidden_size",
//!     "layers": "num_hidden_layers",
//!     "heads": "num_attention_heads",
//!     "kv_heads": "num_key_value_heads",
//!     "ffn": "intermediate_size",
//!     "head_dim": "head_dim"
//!   },
//!   "moe": {
//!     "trigger_field": "num_experts",
//!     "experts": "num_experts",
//!     "top_k": "num_experts_per_tok",
//!     "ffn": "moe_intermediate_size"
//!   },
//!   "quirks": { "qkv_bias": true },
//!   "weights": {
//!     "model.embed_tokens.weight": "embed",
//!     "model.layers.{i}.self_attn.q_proj.weight": "layers.{i}.attn.q"
//!   }
//! }
//! ```
//!
//! Rules enforced here:
//! * dispatch is on the HF `architectures` array; an unknown name is refused and
//!   the error lists the architectures this build knows;
//! * a field in `required` must be present and non-null — it is never defaulted;
//! * only the quirks in [`KNOWN_QUIRKS`] are accepted; an unknown quirk is a
//!   load error that names it.
//!
//! This registry resolves *configuration and tensor names*. It does not change
//! the numeric forward pass. See the note in `architecture_semantics` below.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/architectures_embedded.rs"));
}

/// Quirks a descriptor may declare. The list is deliberately closed: a
/// descriptor is data, not a programming language.
pub const KNOWN_QUIRKS: &[&str] = &[
    "qkv_bias",
    "tie_embeddings",
    "rope_scaling",
    "sliding_window",
    "moe",
    "ple",
    "qk_norm",
    "v_norm",
    "embedding_scale_sqrt",
    "attn_scale_rsqrt",
    "post_attn_norm",
    "post_ffn_norm",
    "swiglu",
    "kv_sharing",
];

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum DescriptorError {
    #[error("unsupported architecture {requested:?}; this build knows: {known:?}")]
    UnknownArchitecture { requested: Vec<String>, known: Vec<String> },

    #[error("config.json has no 'architectures' entry")]
    MissingArchitectures,

    #[error("architecture {architecture}: missing required field '{field}'")]
    MissingField { architecture: String, field: String },

    #[error("architecture {architecture}: config section '{section}' is missing")]
    MissingSection { architecture: String, section: String },

    #[error("architecture {architecture}: unknown quirk '{quirk}'; known quirks: {known:?}")]
    UnknownQuirk { architecture: String, quirk: String, known: Vec<String> },

    #[error("architecture {architecture}: field '{field}' must be a positive integer")]
    InvalidField { architecture: String, field: String },

    #[error("malformed architecture descriptor: {0}")]
    Malformed(String),

    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },
}

// ---------------------------------------------------------------------------
// Raw descriptor (serde)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct RawDescriptor {
    name: String,
    #[serde(default)]
    match_architectures: Vec<String>,
    #[serde(default)]
    match_gguf_architectures: Vec<String>,
    #[serde(default)]
    config_section: Option<String>,
    #[serde(default)]
    required: Vec<String>,
    #[serde(default)]
    fields: BTreeMap<String, String>,
    #[serde(default)]
    moe: Option<RawMoe>,
    #[serde(default)]
    quirks: BTreeMap<String, bool>,
    #[serde(default)]
    weights: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawMoe {
    #[serde(default)]
    trigger_field: Option<String>,
    experts: String,
    top_k: String,
    ffn: String,
}

/// Validated descriptor, as loaded into the registry.
#[derive(Debug, Clone)]
pub struct ArchitectureDescriptor {
    pub name: String,
    pub match_architectures: Vec<String>,
    pub match_gguf_architectures: Vec<String>,
    pub config_section: Option<String>,
    pub required: Vec<String>,
    pub fields: BTreeMap<String, String>,
    pub moe: Option<MoeDescriptor>,
    pub quirks: BTreeSet<String>,
    pub weights: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct MoeDescriptor {
    pub trigger_field: Option<String>,
    pub experts: String,
    pub top_k: String,
    pub ffn: String,
}

// ---------------------------------------------------------------------------
// Resolved shape
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MoeShape {
    pub experts: usize,
    pub top_k: usize,
    pub intermediate_dim: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelShape {
    pub architecture: String,
    pub descriptor: String,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub sliding_window: Option<usize>,
    pub moe: Option<MoeShape>,
    pub quirks: BTreeSet<String>,
    pub weights: BTreeMap<String, String>,
}

impl ModelShape {
    /// Convert the resolved shape to the loader's existing config type.
    ///
    /// This is configuration mapping only. The CPU forward pass in
    /// `gemma_mapper.rs` currently applies Gemma-4 semantics (embedding
    /// `sqrt(hidden)` scaling, `scale = 1.0` attention, unconditional Q/K/V
    /// RMSNorm, GeLU FFN, post-attention/post-FFN norms). A Qwen shape must not
    /// be handed to that forward pass until those semantics are selected from
    /// the declared quirks; see `architecture_semantics` below.
    pub fn to_gemma4_config(&self) -> crate::model::gemma_mapper::Gemma4Config {
        use crate::model::gemma_mapper::Gemma4Config;
        Gemma4Config {
            hidden_dim: self.hidden_dim,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            intermediate_dim: self.intermediate_dim,
            num_experts: self.moe.as_ref().map(|m| m.experts).unwrap_or(1),
            top_k: self.moe.as_ref().map(|m| m.top_k).unwrap_or(1),
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window.unwrap_or(self.max_position_embeddings),
            moe_layers: Vec::new(),
            hidden_size_per_layer_input: None,
            final_logit_softcapping: None,
            layer_types: Vec::new(),
            global_head_dim: self.head_dim,
            rope_theta_sliding: self.rope_theta,
            rope_theta_full: self.rope_theta,
            partial_rotary_factor_full: 1.0,
            num_kv_shared_layers: 0,
        }
    }
}

/// Semantics this build can *validate* but does not yet *execute* generically.
///
/// Known gaps for a Qwen shape in the current CPU forward pass:
/// * no embedding `sqrt(hidden_dim)` scale (`embedding_scale_sqrt` absent);
/// * attention scale must be `1/sqrt(head_dim)`, not `1.0`;
/// * Qwen2.5 Q/K/V projections carry biases (`qkv_bias`);
/// * Qwen2.5 has no per-head Q/K/V RMSNorm; Qwen3 has Q/K norm but no V norm;
/// * Qwen has one pre-FFN norm, no post-attention / post-FFN norm;
/// * FFN activation is SwiGLU, not GeLU.
///
/// These are named quirks in the descriptor, not hidden defaults. Turning them
/// into a generic forward pass is a separate change; until then `infer` on a
/// Qwen GGUF is expected to load but not to produce coherent text.
pub const ARCHITECTURE_SEMANTICS_NOTE: &str =
    "descriptor resolves config and tensor names; forward-pass quirks still need code";

// ---------------------------------------------------------------------------
// Check report (the `ferrisres models check` contract)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct CheckReport {
    pub architecture: String,
    pub supported: bool,
    pub missing: Vec<String>,
    pub quirks: Vec<String>,
    pub weights_mapped: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone)]
pub struct ArchitectureRegistry {
    descriptors: Vec<ArchitectureDescriptor>,
    by_arch: HashMap<String, usize>,
    by_gguf: HashMap<String, usize>,
}

impl ArchitectureRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build the registry from descriptors embedded at compile time.
    pub fn from_embedded() -> Result<Self, DescriptorError> {
        let mut registry = Self::new();
        for &(file, source) in embedded::EMBEDDED_ARCHITECTURES {
            registry
                .add_descriptor_json(source)
                .map_err(|e| DescriptorError::Malformed(format!("{}: {}", file, e)))?;
        }
        Ok(registry)
    }

    /// Build the registry used by the binary: embedded descriptors plus an
    /// optional runtime directory (`FERRISRES_ARCHITECTURES_DIR`). Files loaded
    /// later override an earlier descriptor that claims the same architecture.
    pub fn load_default() -> Result<Self, DescriptorError> {
        let mut registry = Self::from_embedded()?;
        if let Ok(dir) = std::env::var("FERRISRES_ARCHITECTURES_DIR") {
            registry.load_dir(Path::new(&dir))?;
        }
        Ok(registry)
    }

    pub fn load_dir(&mut self, dir: &Path) -> Result<(), DescriptorError> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| DescriptorError::Io { path: dir.display().to_string(), source: e })?;
        let mut paths: Vec<_> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
            .collect();
        paths.sort();
        for path in paths {
            let source = std::fs::read_to_string(&path)
                .map_err(|e| DescriptorError::Io { path: path.display().to_string(), source: e })?;
            self.add_descriptor_json(&source)
                .map_err(|e| DescriptorError::Malformed(format!("{}: {}", path.display(), e)))?;
        }
        Ok(())
    }

    /// Parse and register one descriptor. Public so tests and external tools can
    /// exercise refusal rules without writing a file.
    pub fn add_descriptor_json(&mut self, source: &str) -> Result<(), DescriptorError> {
        let raw: RawDescriptor =
            serde_json::from_str(source).map_err(|e| DescriptorError::Malformed(e.to_string()))?;

        for quirk in raw.quirks.keys() {
            if !KNOWN_QUIRKS.contains(&quirk.as_str()) {
                return Err(DescriptorError::UnknownQuirk {
                    architecture: raw.name.clone(),
                    quirk: quirk.clone(),
                    known: KNOWN_QUIRKS.iter().map(|q| q.to_string()).collect(),
                });
            }
        }

        let descriptor = ArchitectureDescriptor {
            name: raw.name,
            match_architectures: raw.match_architectures,
            match_gguf_architectures: raw.match_gguf_architectures,
            config_section: raw.config_section,
            required: raw.required,
            fields: raw.fields,
            moe: raw.moe.map(|m| MoeDescriptor {
                trigger_field: m.trigger_field,
                experts: m.experts,
                top_k: m.top_k,
                ffn: m.ffn,
            }),
            quirks: raw.quirks.into_iter().filter(|(_, v)| *v).map(|(k, _)| k).collect(),
            weights: raw.weights,
        };

        let index = self.descriptors.len();
        for arch in &descriptor.match_architectures {
            self.by_arch.insert(arch.clone(), index);
        }
        for arch in &descriptor.match_gguf_architectures {
            self.by_gguf.insert(arch.clone(), index);
        }
        self.descriptors.push(descriptor);
        Ok(())
    }

    pub fn descriptors(&self) -> &[ArchitectureDescriptor] {
        &self.descriptors
    }

    pub fn known_architectures(&self) -> Vec<String> {
        let mut names: Vec<String> = self.by_arch.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn descriptor_for(&self, architecture: &str) -> Option<&ArchitectureDescriptor> {
        self.by_arch.get(architecture).map(|index| &self.descriptors[*index])
    }

    pub fn descriptor_for_gguf(&self, gguf_architecture: &str) -> Option<&ArchitectureDescriptor> {
        self.by_gguf.get(gguf_architecture).map(|index| &self.descriptors[*index])
    }

    /// Resolve a parsed HF `config.json` against the registry.
    pub fn resolve_config(&self, root: &Value) -> Result<ModelShape, DescriptorError> {
        let requested = architectures_of(root);
        if requested.is_empty() {
            return Err(DescriptorError::MissingArchitectures);
        }
        for architecture in &requested {
            if let Some(descriptor) = self.descriptor_for(architecture) {
                return self.resolve_with(descriptor, architecture, root);
            }
        }
        Err(DescriptorError::UnknownArchitecture { requested, known: self.known_architectures() })
    }

    /// Resolve a GGUF `general.architecture` value (GGUF has no `architectures`
    /// array). Unknown names are refused with the known list.
    pub fn resolve_gguf_architecture(
        &self,
        gguf_architecture: &str,
        root: &Value,
    ) -> Result<ModelShape, DescriptorError> {
        match self.descriptor_for_gguf(gguf_architecture) {
            Some(descriptor) => {
                let architecture = descriptor
                    .match_architectures
                    .first()
                    .cloned()
                    .unwrap_or_else(|| gguf_architecture.to_string());
                self.resolve_with(descriptor, &architecture, root)
            }
            None => Err(DescriptorError::UnknownArchitecture {
                requested: vec![gguf_architecture.to_string()],
                known: self.known_architectures(),
            }),
        }
    }

    /// Validate a config without loading weights. This is the contribution
    /// path for a new family.
    pub fn check(&self, root: &Value) -> CheckReport {
        let requested = architectures_of(root);
        if requested.is_empty() {
            return CheckReport {
                architecture: "<none>".to_string(),
                supported: false,
                missing: Vec::new(),
                quirks: Vec::new(),
                weights_mapped: Vec::new(),
                error: Some(DescriptorError::MissingArchitectures.to_string()),
            };
        }

        for architecture in &requested {
            let Some(descriptor) = self.descriptor_for(architecture) else {
                continue;
            };
            match self.resolve_with(descriptor, architecture, root) {
                Ok(shape) => {
                    return CheckReport {
                        architecture: shape.architecture.clone(),
                        supported: true,
                        missing: Vec::new(),
                        quirks: shape.quirks.iter().cloned().collect(),
                        weights_mapped: expand_weight_slots(&shape.weights),
                        error: None,
                    };
                }
                Err(DescriptorError::MissingField { field, .. }) => {
                    return CheckReport {
                        architecture: architecture.clone(),
                        supported: false,
                        missing: vec![field],
                        quirks: Vec::new(),
                        weights_mapped: Vec::new(),
                        error: None,
                    };
                }
                Err(error) => {
                    return CheckReport {
                        architecture: architecture.clone(),
                        supported: false,
                        missing: Vec::new(),
                        quirks: Vec::new(),
                        weights_mapped: Vec::new(),
                        error: Some(error.to_string()),
                    };
                }
            }
        }

        CheckReport {
            architecture: requested.first().cloned().unwrap_or_default(),
            supported: false,
            missing: Vec::new(),
            quirks: Vec::new(),
            weights_mapped: Vec::new(),
            error: Some(
                DescriptorError::UnknownArchitecture {
                    requested,
                    known: self.known_architectures(),
                }
                .to_string(),
            ),
        }
    }

    fn resolve_with(
        &self,
        descriptor: &ArchitectureDescriptor,
        architecture: &str,
        root: &Value,
    ) -> Result<ModelShape, DescriptorError> {
        let architecture = architecture.to_string();
        let section = match &descriptor.config_section {
            Some(key) => root.get(key).ok_or_else(|| DescriptorError::MissingSection {
                architecture: architecture.clone(),
                section: key.clone(),
            })?,
            None => root,
        };

        for field in &descriptor.required {
            match section.get(field) {
                None | Some(Value::Null) => {
                    return Err(DescriptorError::MissingField {
                        architecture: architecture.clone(),
                        field: field.clone(),
                    });
                }
                Some(_) => {}
            }
        }

        let hidden_dim = lookup_usize(section, &descriptor.fields, "hidden").ok_or_else(|| {
            DescriptorError::MissingField {
                architecture: architecture.clone(),
                field: "hidden".to_string(),
            }
        })?;
        let num_layers = lookup_usize(section, &descriptor.fields, "layers").ok_or_else(|| {
            DescriptorError::MissingField {
                architecture: architecture.clone(),
                field: "layers".to_string(),
            }
        })?;
        let num_heads = lookup_usize(section, &descriptor.fields, "heads").ok_or_else(|| {
            DescriptorError::MissingField {
                architecture: architecture.clone(),
                field: "heads".to_string(),
            }
        })?;
        if hidden_dim == 0 || num_layers == 0 || num_heads == 0 {
            return Err(DescriptorError::InvalidField {
                architecture,
                field: "hidden/layers/heads".to_string(),
            });
        }

        let num_kv_heads =
            lookup_usize(section, &descriptor.fields, "kv_heads").unwrap_or(num_heads);
        let intermediate_dim =
            lookup_usize(section, &descriptor.fields, "ffn").unwrap_or(hidden_dim * 4);
        let head_dim =
            lookup_usize(section, &descriptor.fields, "head_dim").unwrap_or(hidden_dim / num_heads);
        let rope_theta = lookup_f64(section, &descriptor.fields, "rope_theta").unwrap_or(10_000.0);
        let vocab_size = lookup_usize(section, &descriptor.fields, "vocab").unwrap_or(32_000);
        let max_position_embeddings =
            lookup_usize(section, &descriptor.fields, "max_positions").unwrap_or(32_768);
        let tie_word_embeddings =
            lookup_bool(section, &descriptor.fields, "tie_embeddings").unwrap_or(false);
        let sliding_window = lookup_usize(section, &descriptor.fields, "sliding_window");

        let moe = match &descriptor.moe {
            Some(moe) => {
                let active = match &moe.trigger_field {
                    Some(trigger) => lookup_usize(section, &descriptor.fields, trigger)
                        .map(|v| v > 0)
                        .or_else(|| section.get(trigger).and_then(|v| v.as_u64()).map(|v| v > 0))
                        .unwrap_or(false),
                    None => true,
                };
                if active {
                    let experts = lookup_by_source(section, &moe.experts).ok_or_else(|| {
                        DescriptorError::MissingField {
                            architecture: architecture.clone(),
                            field: moe.experts.clone(),
                        }
                    })?;
                    let top_k = lookup_by_source(section, &moe.top_k).ok_or_else(|| {
                        DescriptorError::MissingField {
                            architecture: architecture.clone(),
                            field: moe.top_k.clone(),
                        }
                    })?;
                    let ffn = lookup_by_source(section, &moe.ffn).ok_or_else(|| {
                        DescriptorError::MissingField {
                            architecture: architecture.clone(),
                            field: moe.ffn.clone(),
                        }
                    })?;
                    Some(MoeShape { experts, top_k, intermediate_dim: ffn })
                } else {
                    None
                }
            }
            None => None,
        };

        Ok(ModelShape {
            architecture,
            descriptor: descriptor.name.clone(),
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            rope_theta,
            vocab_size,
            max_position_embeddings,
            tie_word_embeddings,
            sliding_window,
            moe,
            quirks: descriptor.quirks.clone(),
            weights: descriptor.weights.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// File-level helpers
// ---------------------------------------------------------------------------

pub fn check_config_file(path: &Path) -> Result<CheckReport, DescriptorError> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| DescriptorError::Io { path: path.display().to_string(), source: e })?;
    let value: Value = serde_json::from_str(&source)
        .map_err(|e| DescriptorError::Json { path: path.display().to_string(), source: e })?;
    let registry = ArchitectureRegistry::load_default()?;
    Ok(registry.check(&value))
}

pub fn resolve_config_file(path: &Path) -> Result<ModelShape, DescriptorError> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| DescriptorError::Io { path: path.display().to_string(), source: e })?;
    let value: Value = serde_json::from_str(&source)
        .map_err(|e| DescriptorError::Json { path: path.display().to_string(), source: e })?;
    ArchitectureRegistry::load_default()?.resolve_config(&value)
}

fn architectures_of(root: &Value) -> Vec<String> {
    match root.get("architectures") {
        Some(Value::Array(items)) => {
            items.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
        }
        Some(Value::String(name)) => vec![name.clone()],
        _ => Vec::new(),
    }
}

fn lookup_by_source(section: &Value, source: &str) -> Option<usize> {
    section.get(source).and_then(value_usize)
}

fn lookup_usize(section: &Value, fields: &BTreeMap<String, String>, key: &str) -> Option<usize> {
    fields.get(key).and_then(|source| section.get(source)).and_then(value_usize)
}

fn lookup_f64(section: &Value, fields: &BTreeMap<String, String>, key: &str) -> Option<f64> {
    fields
        .get(key)
        .and_then(|source| section.get(source))
        .and_then(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
}

fn lookup_bool(section: &Value, fields: &BTreeMap<String, String>, key: &str) -> Option<bool> {
    fields.get(key).and_then(|source| section.get(source)).and_then(|v| v.as_bool())
}

fn value_usize(v: &Value) -> Option<usize> {
    v.as_u64().map(|n| n as usize).or_else(|| v.as_i64().and_then(|n| usize::try_from(n).ok()))
}

fn expand_weight_slots(weights: &BTreeMap<String, String>) -> Vec<String> {
    let mut slots = BTreeSet::new();
    for slot in weights.values() {
        // One layer is enough to prove the mapping; the command must not print
        // a 36-layer × 15-tensor list.
        slots.insert(slot.replace("{i}", "0"));
    }
    slots.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN2: &str = include_str!("../../tests/fixtures/qwen2.5-1.5b.config.json");
    const QWEN3: &str = include_str!("../../tests/fixtures/qwen3-8b.config.json");

    fn registry() -> ArchitectureRegistry {
        ArchitectureRegistry::from_embedded().expect("embedded descriptors parse")
    }

    #[test]
    fn dispatches_qwen3_by_architectures() {
        let value: Value = serde_json::from_str(QWEN3).unwrap();
        let shape = registry().resolve_config(&value).unwrap();
        assert_eq!(shape.architecture, "Qwen3ForCausalLM");
        assert_eq!(shape.descriptor, "qwen3");
        assert_eq!(shape.hidden_dim, 4096);
        assert_eq!(shape.num_layers, 36);
        assert_eq!(shape.num_heads, 32);
        assert_eq!(shape.num_kv_heads, 8);
        assert_eq!(shape.head_dim, 128);
        assert_eq!(shape.intermediate_dim, 12288);
        assert_eq!(shape.vocab_size, 151936);
        assert!(!shape.tie_word_embeddings);
        assert!(shape.quirks.contains("qk_norm"));
        assert!(shape.moe.is_none());
    }

    #[test]
    fn unknown_architecture_is_refused_by_name() {
        let value = serde_json::json!({
            "architectures": ["NopeForCausalLM"],
            "hidden_size": 1
        });
        let err = registry().resolve_config(&value).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("NopeForCausalLM"), "{message}");
        assert!(message.contains("Qwen2ForCausalLM"), "{message}");
        assert!(message.contains("Qwen3ForCausalLM"), "{message}");
    }

    #[test]
    fn missing_required_field_is_refused_by_name() {
        let mut value: Value = serde_json::from_str(QWEN2).unwrap();
        value.as_object_mut().unwrap().remove("num_key_value_heads");
        let err = registry().resolve_config(&value).unwrap_err();
        match err {
            DescriptorError::MissingField { field, .. } => {
                assert_eq!(field, "num_key_value_heads");
            }
            other => panic!("expected MissingField, got {other}"),
        }
    }

    #[test]
    fn unknown_quirk_is_refused_by_name() {
        let mut registry = ArchitectureRegistry::new();
        let err = registry
            .add_descriptor_json(
                r#"{
                    "name": "telepathic",
                    "match_architectures": ["TelepathicForCausalLM"],
                    "quirks": { "telepathy": true }
                }"#,
            )
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("telepathy"), "{message}");
        assert!(message.contains("qkv_bias"), "{message}");
    }

    #[test]
    fn qwen2_descriptor_maps_config_and_weights() {
        let value: Value = serde_json::from_str(QWEN2).unwrap();
        let shape = registry().resolve_config(&value).unwrap();
        assert_eq!(shape.architecture, "Qwen2ForCausalLM");
        assert_eq!(shape.hidden_dim, 1536);
        assert_eq!(shape.num_layers, 28);
        assert_eq!(shape.num_kv_heads, 2);
        assert_eq!(shape.head_dim, 128); // derived, no head_dim in the config
        assert_eq!(shape.rope_theta, 1_000_000.0);
        assert!(shape.tie_word_embeddings);
        assert!(shape.quirks.contains("qkv_bias"));

        assert_eq!(shape.weights.get("model.embed_tokens.weight"), Some(&"embed".to_string()));
        assert_eq!(
            shape.weights.get("model.layers.{i}.self_attn.q_proj.weight"),
            Some(&"layers.{i}.attn.q".to_string())
        );
        let slots = expand_weight_slots(&shape.weights);
        assert!(slots.contains(&"embed".to_string()));
        assert!(slots.contains(&"layers.0.attn.q".to_string()));
        assert!(slots.contains(&"layers.0.attn.q_bias".to_string()));
        assert!(slots.contains(&"final_norm".to_string()));
        assert!(slots.contains(&"lm_head".to_string()));
    }

    #[test]
    fn check_reports_supported_shape() {
        let value: Value = serde_json::from_str(QWEN3).unwrap();
        let report = registry().check(&value);
        assert!(report.supported);
        assert_eq!(report.architecture, "Qwen3ForCausalLM");
        assert!(report.missing.is_empty());
        assert!(report.weights_mapped.contains(&"layers.0.attn.q".to_string()));
    }

    #[test]
    fn check_reports_missing_field_without_erroring() {
        let mut value: Value = serde_json::from_str(QWEN3).unwrap();
        value.as_object_mut().unwrap().remove("rope_theta");
        let report = registry().check(&value);
        assert!(!report.supported);
        assert_eq!(report.missing, vec!["rope_theta".to_string()]);
        assert!(report.error.is_none());
    }

    #[test]
    fn qwen3_moe_fields_are_read_when_present() {
        let mut value: Value = serde_json::from_str(QWEN3).unwrap();
        let object = value.as_object_mut().unwrap();
        object.insert("num_experts".into(), serde_json::json!(128));
        object.insert("num_experts_per_tok".into(), serde_json::json!(8));
        object.insert("moe_intermediate_size".into(), serde_json::json!(768));
        let shape = registry().resolve_config(&value).unwrap();
        let moe = shape.moe.expect("MoE fields present");
        assert_eq!(moe.experts, 128);
        assert_eq!(moe.top_k, 8);
        assert_eq!(moe.intermediate_dim, 768);
    }

    #[test]
    fn gguf_architecture_dispatches_by_name() {
        let registry = registry();
        assert_eq!(registry.descriptor_for_gguf("qwen2").map(|d| d.name.as_str()), Some("qwen2"));
        assert_eq!(registry.descriptor_for_gguf("qwen3").map(|d| d.name.as_str()), Some("qwen3"));
        assert!(registry.descriptor_for_gguf("not-a-family").is_none());
    }

    #[test]
    fn qwen2_shape_converts_to_loader_config() {
        let value: Value = serde_json::from_str(QWEN2).unwrap();
        let config = registry().resolve_config(&value).unwrap().to_gemma4_config();
        assert_eq!(config.hidden_dim, 1536);
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.vocab_size, 151936);
        assert_eq!(config.num_experts, 1);
    }
}
