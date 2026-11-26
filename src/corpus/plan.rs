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
    /// `byte` (default) or `bpe` with a `tokenizer.json` path.
    #[serde(default)]
    pub tokenizer: TokenizerSpec,
    pub sources: Vec<SourceSpec>,
    pub split: SplitSpec,
}

/// Which tokenizer turns documents into ids. `byte` is the v1 default; `bpe`
/// loads a HuggingFace `tokenizer.json` from `path`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct TokenizerSpec {
    #[serde(default = "default_tokenizer_kind")]
    pub kind: String,
    /// Required for `bpe`, rejected for `byte`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

impl Default for TokenizerSpec {
    fn default() -> Self {
        Self { kind: default_tokenizer_kind(), path: None }
    }
}

fn default_tokenizer_kind() -> String {
    "byte".to_string()
}

/// Where a source's bytes come from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    /// A directory (or single file) already on this machine.
    #[default]
    Local,
    /// A git remote, checked out at an exact commit.
    Git,
    /// A Hugging Face dataset store. Only the files named in `files` are read.
    Hf,
}

impl SourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SourceKind::Local => "local",
            SourceKind::Git => "git",
            SourceKind::Hf => "hf",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "local" => Ok(SourceKind::Local),
            "git" => Ok(SourceKind::Git),
            "hf" => Ok(SourceKind::Hf),
            other => bail!("unknown source kind '{other}' (expected local, git or hf)"),
        }
    }
}

/// One file named by an `hf` source. The sha256 is the contract: a byte that
/// does not hash to this value is refused.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct SourceFileSpec {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SourceSpec {
    pub id: String,
    /// `local`, `git` or `hf`. Defaults to `local` so v1 plans keep working.
    #[serde(default)]
    pub kind: SourceKind,
    /// Local directory, git URL, or HF dataset store (`file://`, `https://`).
    pub path: String,
    pub licence: Option<String>,
    /// A pinned revision: the git commit, or the HF dataset snapshot.
    pub revision: Option<String>,
    /// Refuse any network access for this source.
    #[serde(default)]
    pub offline: bool,
    /// The source needs a credential. A gated source without a token is refused.
    #[serde(default)]
    pub gated: bool,
    /// The only files an `hf` source may read. Each carries its sha256.
    #[serde(default)]
    pub files: Vec<SourceFileSpec>,
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
        if self.tokenizer.kind != "byte" && self.tokenizer.kind != "bpe" {
            bail!(
                "plan tokenizer kind '{}' is not supported (expected 'byte' or 'bpe')",
                self.tokenizer.kind
            );
        }
        if self.tokenizer.kind == "bpe" {
            let path =
                self.tokenizer.path.as_deref().map(str::trim).filter(|value| !value.is_empty());
            let Some(path) = path else {
                bail!("plan tokenizer kind 'bpe' requires a path");
            };
            if !Path::new(path).exists() {
                bail!("plan tokenizer path does not exist: {path}");
            }
        } else if self.tokenizer.path.is_some() {
            bail!("plan tokenizer kind 'byte' takes no path");
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
                _ => bail!(
                    "source '{}' ({}) is missing a revision; every source must pin one",
                    source.id,
                    source.kind.as_str()
                ),
            }
            if source.path.trim().is_empty() {
                bail!("source '{}' ({}) is missing a path", source.id, source.kind.as_str());
            }
            if source.kind == SourceKind::Hf {
                if source.files.is_empty() {
                    bail!(
                        "hf source '{}' names no files; list every file and its sha256",
                        source.id
                    );
                }
                for file in &source.files {
                    if file.path.trim().is_empty() {
                        bail!("hf source '{}' contains a file with an empty path", source.id);
                    }
                    if !is_sha256_hex(&file.sha256) {
                        bail!(
                            "hf source '{}' file '{}' does not have a 64-character sha256",
                            source.id,
                            file.path
                        );
                    }
                }
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

/// True when `value` is exactly 64 hexadecimal characters.
pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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
