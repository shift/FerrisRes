//! Corpus plan file (JSON) and deterministic classification helpers.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;

/// Builder-side split of a whole document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    Train,
    Val,
    Heldout,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Heldout => "heldout",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "train" => Ok(Split::Train),
            "val" => Ok(Split::Val),
            "heldout" => Ok(Split::Heldout),
            other => bail!("unknown split '{other}' (expected train, val or heldout)"),
        }
    }
}

/// A corpus plan, as read from `--plan`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Plan {
    pub name: String,
    pub version: String,
    pub seq_len: usize,
    pub seed: u64,
    pub tokenizer: TokenizerSpec,
    pub sources: Vec<SourceSpec>,
    pub split: SplitSpec,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokenizerSpec {
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SourceSpec {
    pub id: String,
    pub path: String,
    pub licence: Option<String>,
    pub revision: Option<String>,
    #[serde(default = "default_weight")]
    pub weight: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SplitSpec {
    #[serde(default)]
    pub val: f64,
    #[serde(default)]
    pub heldout: f64,
}

fn default_weight() -> f64 {
    1.0
}

/// Stable description of the byte tokenizer used by v1 shards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerRecord {
    pub kind: String,
    pub id: String,
    pub vocab_size: usize,
    pub sha256: String,
}

/// The only supported tokenizer in v1: `SimpleTokenizer` (259 ids).
pub const BYTE_TOKENIZER_ID: &str = "ferrisres-simple-byte-v1";
pub const BYTE_TOKENIZER_VOCAB: usize = 259;

pub fn tokenizer_record() -> TokenizerRecord {
    // The vocabulary is fixed in code; hash its canonical textual form so a
    // tokenizer change cannot reuse a corpus silently.
    let mut canonical = String::from("<unk>\n<eos>\n<pad>\n");
    for b in 0u16..=255u16 {
        canonical.push_str(&format!("<0x{b:02X}>\n"));
    }
    TokenizerRecord {
        kind: "byte".to_string(),
        id: BYTE_TOKENIZER_ID.to_string(),
        vocab_size: BYTE_TOKENIZER_VOCAB,
        sha256: sha256_hex(canonical.as_bytes()),
    }
}

impl Plan {
    /// Read and validate a plan file.
    pub fn load(path: &Path) -> Result<(Vec<u8>, Plan)> {
        let bytes =
            std::fs::read(path).with_context(|| format!("cannot read plan {}", path.display()))?;
        let plan: Plan = serde_json::from_slice(&bytes)
            .with_context(|| format!("plan {} is not valid JSON", path.display()))?;
        plan.validate()?;
        Ok((bytes, plan))
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            bail!("plan name must not be empty");
        }
        if self.version.trim().is_empty() {
            bail!("plan version must not be empty");
        }
        if self.seq_len == 0 {
            bail!("plan seq_len must be greater than zero");
        }
        if self.tokenizer.kind != "byte" {
            bail!(
                "plan tokenizer kind '{}' is not supported (v1 supports 'byte')",
                self.tokenizer.kind
            );
        }
        if self.sources.is_empty() {
            bail!("plan has no sources");
        }

        let mut ids = BTreeSet::new();
        for source in &self.sources {
            if source.id.trim().is_empty() {
                bail!("plan contains a source with an empty id");
            }
            if !ids.insert(source.id.as_str()) {
                bail!("plan contains duplicate source id '{}'", source.id);
            }
            match source.licence.as_deref() {
                Some(value) if !value.trim().is_empty() => {}
                _ => bail!("source '{}' is missing a licence", source.id),
            }
            match source.revision.as_deref() {
                Some(value) if !value.trim().is_empty() => {}
                _ => bail!("source '{}' is missing a revision", source.id),
            }
            if !source.weight.is_finite() || source.weight < 0.0 {
                bail!("source '{}' has an invalid weight", source.id);
            }
        }

        validate_fraction("val", self.split.val)?;
        validate_fraction("heldout", self.split.heldout)?;
        if self.split.val + self.split.heldout >= 1.0 {
            bail!(
                "split fractions val={} + heldout={} leave no training documents",
                self.split.val,
                self.split.heldout
            );
        }
        Ok(())
    }
}

fn validate_fraction(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        bail!("split fraction {name}={value} must be between 0 and 1");
    }
    Ok(())
}

/// Assign a document to a split from its sha256, held-out first, then val,
/// then train. This is the only split rule used by the builder.
pub fn classify_split(hash_hex: &str, val: f64, heldout: f64) -> Split {
    let value = first_u64(hash_hex);
    let ratio = value as f64 / 18_446_744_073_709_551_616.0; // 2^64
    if ratio < heldout {
        Split::Heldout
    } else if ratio < heldout + val {
        Split::Val
    } else {
        Split::Train
    }
}

fn first_u64(hash_hex: &str) -> u64 {
    let bytes = hash_hex.as_bytes();
    let mut value = 0u64;
    for chunk in bytes.chunks(2).take(8) {
        let hi = hex_nibble(chunk[0]);
        let lo = if chunk.len() > 1 { hex_nibble(chunk[1]) } else { 0 };
        value = (value << 8) | ((hi << 4) | lo) as u64;
    }
    value
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => 0,
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_deterministic_and_heldout_first() {
        assert_eq!(classify_split("0000000000000000", 0.1, 0.2), Split::Heldout);
        assert_eq!(classify_split("4000000000000000", 0.3, 0.2), Split::Val);
        assert_eq!(classify_split("ffffffffffffffff", 0.1, 0.2), Split::Train);
        assert_eq!(
            classify_split("ffffffffffffffff", 0.1, 0.2),
            classify_split("ffffffffffffffff", 0.1, 0.2)
        );
    }
}
