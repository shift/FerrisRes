//! The 32k source tokenizer: BPE with byte-level fallback and no normalisation.
//!
//! Rust, Nix and Python are whitespace- and case-sensitive, so the tokenizer
//! performs no Unicode normalisation, no lowercasing and no whitespace
//! stripping. Byte-level pre-tokenisation makes every input representable, so
//! there is no unknown token.
//!
//! See `docs/tokenizer.md` for the design and the gates.

pub mod train;

use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// The fixed special-token block, in the order it is added to the vocabulary.
///
/// The order is part of the artefact: ids are assigned in this order, so
/// changing it is a tokenizer major version.
pub const SPECIAL_TOKENS: [&str; 18] = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<system>",
    "</system>",
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>",
    "<tool>",
    "</tool>",
    "<tool_call>",
    "</tool_call>",
    "<tool_result>",
    "</tool_result>",
    "<fim_prefix>",
    "<fim_suffix>",
    "<fim_middle>",
];

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// What `tokenizer inspect` prints.
#[derive(Debug, Clone)]
pub struct InspectReport {
    pub vocab_size: usize,
    pub sha256: String,
    pub path: std::path::PathBuf,
}

/// Load `tokenizer.json`, report its vocabulary size and the hash of its bytes.
pub fn inspect(path: &Path) -> Result<InspectReport> {
    let bytes =
        std::fs::read(path).with_context(|| format!("read tokenizer {}", path.display()))?;
    let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("parse tokenizer {}: {e}", path.display()))?;
    Ok(InspectReport {
        vocab_size: tokenizer.get_vocab_size(true),
        sha256: sha256_hex(&bytes),
        path: path.to_path_buf(),
    })
}

/// What `tokenizer fertility` prints.
#[derive(Debug, Clone)]
pub struct FertilityReport {
    pub tokens: usize,
    pub bytes: usize,
    pub tokens_per_byte: f64,
}

/// Tokenise `text` with the tokenizer at `tokenizer_path` and report
/// tokens per byte.
pub fn fertility(tokenizer_path: &Path, text_path: &Path) -> Result<FertilityReport> {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| anyhow::anyhow!("parse tokenizer {}: {e}", tokenizer_path.display()))?;
    let text = std::fs::read_to_string(text_path)
        .with_context(|| format!("read text {}", text_path.display()))?;
    let encoding = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode {}: {e}", text_path.display()))?;
    let tokens = encoding.len();
    let bytes = text.len();
    let tokens_per_byte = if bytes == 0 { 0.0 } else { tokens as f64 / bytes as f64 };
    Ok(FertilityReport { tokens, bytes, tokens_per_byte })
}
