//! Validated attention dimensions, not a complete checkpoint loader.
//!
//! Projection widths cannot determine attention head counts. Read these from
//! explicit architecture metadata and validate tensor shapes afterward. This
//! module does not infer activation, normalization, biases or positional settings.
use std::path::Path;

use super::gguf::{GgufFile, GgufValue};
use super::standard_transformer::{GatedActivation, StandardTransformerConfig};
use crate::error::{FerrisResError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionMetadata {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionMetadata {
    /// Read a Hugging Face text model's config.json. Multimodal callers must
    /// explicitly select their text configuration with `from_hf_json` instead.
    pub fn from_hf_config(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&text)
            .map_err(|e| invalid(format!("invalid HF configuration JSON: {e}")))?;
        Self::from_hf_json(&config)
    }

    pub fn from_hf_json(config: &serde_json::Value) -> Result<Self> {
        let fields = config
            .as_object()
            .ok_or_else(|| invalid("HF configuration must be an object"))?;
        let get = |key: &str| -> Result<Option<usize>> {
            fields
                .get(key)
                .map(|v| {
                    let value = v
                        .as_u64()
                        .ok_or_else(|| invalid(format!("{key} must be a positive integer")))?;
                    dimension(value, key)
                })
                .transpose()
        };
        Self::from_dimensions(
            required(get("hidden_size")?, "hidden_size")?,
            required(get("num_attention_heads")?, "num_attention_heads")?,
            get("num_key_value_heads")?,
            get("head_dim")?,
        )
    }

    /// Use general.architecture to select keys; never silently assume llama.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let architecture = match file.metadata.get("general.architecture") {
            Some(GgufValue::String(s)) if !s.is_empty() => s,
            _ => return Err(invalid("missing or invalid general.architecture")),
        };
        let get = |suffix: &str| -> Result<Option<usize>> {
            let key = format!("{architecture}.{suffix}");
            file.metadata
                .get(&key)
                .map(|v| {
                    let value = match v {
                        GgufValue::UInt8(n) => u64::from(*n),
                        GgufValue::UInt16(n) => u64::from(*n),
                        GgufValue::UInt32(n) => u64::from(*n),
                        GgufValue::UInt64(n) => *n,
                        _ => return Err(invalid(format!("{key} must be an unsigned integer"))),
                    };
                    dimension(value, &key)
                })
                .transpose()
        };
        let result = Self::from_dimensions(
            required(get("embedding_length")?, "embedding_length")?,
            required(get("attention.head_count")?, "attention.head_count")?,
            get("attention.head_count_kv")?,
            get("attention.key_length")?,
        )?;
        if let Some(value_dim) = get("attention.value_length")? {
            if value_dim != result.head_dim {
                return Err(FerrisResError::Unsupported(
                    "unequal attention key/value head dimensions".into(),
                ));
            }
        }
        Ok(result)
    }

    fn from_dimensions(
        hidden_dim: usize,
        num_heads: usize,
        kv: Option<usize>,
        head: Option<usize>,
    ) -> Result<Self> {
        let head_dim = match head {
            Some(value) => value,
            None if hidden_dim % num_heads == 0 => hidden_dim / num_heads,
            None => {
                return Err(invalid(
                    "head_dim is required when hidden size is not divisible by head count",
                ))
            }
        };
        let result = Self {
            hidden_dim,
            num_heads,
            num_kv_heads: kv.unwrap_or(num_heads),
            head_dim,
        };
        result.validate()?;
        Ok(result)
    }

    /// Revalidate public fields, including before any multiplication or division.
    fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("hidden_dim", self.hidden_dim),
            ("num_heads", self.num_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ] {
            dimension(value as u64, name)?;
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(invalid(
                "query head count must be a multiple of KV head count",
            ));
        }
        for count in [self.num_heads, self.num_kv_heads] {
            let width = count
                .checked_mul(self.head_dim)
                .ok_or_else(|| invalid("projection width overflow"))?;
            dimension(width as u64, "projection width")?;
        }
        Ok(())
    }

    /// Validate canonical [out_features, in_features] shapes. GGUF dimensions
    /// must first be reversed into this convention; native Linear uses [in,out].
    pub fn validate_projection_shapes(&self, q: &[usize], k: &[usize], v: &[usize]) -> Result<()> {
        self.validate()?;
        for (name, actual, heads) in [
            ("Q", q, self.num_heads),
            ("K", k, self.num_kv_heads),
            ("V", v, self.num_kv_heads),
        ] {
            let expected = [heads * self.head_dim, self.hidden_dim];
            if actual != expected {
                return Err(invalid(format!(
                    "{name} projection shape {actual:?}, expected {expected:?}"
                )));
            }
        }
        Ok(())
    }

    /// Construct an attention-dimension-compatible configuration. This does not
    /// validate the rest of an architecture or load weights. Caller supplies FFN
    /// dimensions/activation and must validate bias, norm and RoPE requirements.
    /// The current GPU StandardTransformer implements only equal-width MHA.
    pub fn standard_config(
        &self,
        layers: usize,
        intermediate: usize,
        vocab: usize,
        activation: GatedActivation,
    ) -> Result<StandardTransformerConfig> {
        self.validate()?;
        if self.num_kv_heads != self.num_heads || self.num_heads * self.head_dim != self.hidden_dim
        {
            return Err(FerrisResError::Unsupported("StandardTransformer requires MHA and query width equal to hidden size; GQA/MQA or independent head dimensions need a different representation".into()));
        }
        if self.head_dim % 2 != 0 {
            return Err(FerrisResError::Unsupported(
                "StandardTransformer RoPE requires an even head dimension".into(),
            ));
        }
        for (name, value) in [
            ("num_layers", layers),
            ("intermediate_dim", intermediate),
            ("vocab_size", vocab),
        ] {
            dimension(value as u64, name)?;
        }
        Ok(StandardTransformerConfig {
            hidden_dim: self.hidden_dim,
            num_heads: self.num_heads,
            num_layers: layers,
            intermediate_dim: intermediate,
            head_dim: self.head_dim,
            vocab_size: vocab,
            use_bias: false,
            ffn_activation: activation,
            norm_eps: 1e-5,
        })
    }
}

fn invalid(message: impl Into<String>) -> FerrisResError {
    FerrisResError::Shape(message.into())
}
fn required(value: Option<usize>, key: &str) -> Result<usize> {
    value.ok_or_else(|| invalid(format!("missing required attention metadata: {key}")))
}
fn dimension(value: u64, key: &str) -> Result<usize> {
    if value == 0 || value > u64::from(u32::MAX) {
        return Err(invalid(format!("{key} must be in 1..=u32::MAX")));
    }
    usize::try_from(value).map_err(|_| invalid(format!("{key} exceeds addressable dimensions")))
}
