//! `ferrisres tokenizer train`: deterministic BPE training over a sorted corpus.
//!
//! The build order is fixed so two runs on the same inputs produce the same
//! `tokenizer.json`:
//!
//! 1. collect the corpus files, skip non-files, sort by relative path;
//! 2. hash every file with SHA-256 and record it in `sample-manifest.jsonl`;
//! 3. feed the files to the BPE trainer in that order;
//! 4. write the artefacts and the hashes of the artefacts.
//!
//! The tokenizer uses byte-level pre-tokenisation with no normaliser and a
//! byte-level decoder. That makes every byte representable, so there is no
//! unknown token, and indentation, tabs, trailing spaces and `\n` survive a
//! round trip exactly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::{AddedToken, TokenizerImpl};

use super::{sha256_hex, SPECIAL_TOKENS};

/// Directories that are never part of a source corpus.
const SKIP_DIRS: [&str; 6] = [".git", "target", "node_modules", ".direnv", "result", "vendor"];

/// One input file, as recorded in the sample manifest.
#[derive(Debug, Clone)]
pub struct CorpusFile {
    /// Absolute path used for reading.
    pub absolute: PathBuf,
    /// Path relative to the corpus root, with `/` separators.
    pub relative: String,
    pub sha256: String,
    pub bytes: u64,
}

/// Training options.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// A directory or a single file.
    pub corpus: PathBuf,
    /// Requested vocabulary size, including the byte alphabet and specials.
    pub vocab_size: usize,
    pub out_dir: PathBuf,
    /// Licence label applied to files with no matching entry in `licence_map`.
    pub default_licence: String,
    /// Path-prefix -> licence. The longest matching prefix wins.
    pub licence_map: BTreeMap<String, String>,
}

impl TrainConfig {
    pub fn new(corpus: impl Into<PathBuf>, vocab_size: usize, out_dir: impl Into<PathBuf>) -> Self {
        Self {
            corpus: corpus.into(),
            vocab_size,
            out_dir: out_dir.into(),
            default_licence: "unspecified".to_string(),
            licence_map: BTreeMap::new(),
        }
    }

    fn licence_for(&self, relative: &str) -> &str {
        self.licence_map
            .iter()
            .filter(|(prefix, _)| relative.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, licence)| licence.as_str())
            .unwrap_or(self.default_licence.as_str())
    }
}

/// What training produced.
#[derive(Debug, Clone, Serialize)]
pub struct TrainReport {
    pub vocab_size: usize,
    pub tokenizer_sha256: String,
    pub files: usize,
    pub skipped_non_utf8: usize,
    pub out_dir: PathBuf,
}

/// One manifest line. Field order is fixed, so the file is reproducible.
#[derive(Debug, Serialize)]
struct ManifestLine<'a> {
    path: &'a str,
    sha256: &'a str,
    bytes: u64,
    licence: &'a str,
}

/// Train a tokenizer and write the artefacts into `config.out_dir`.
pub fn train(config: &TrainConfig) -> Result<TrainReport> {
    // 256 byte tokens plus the fixed special block must fit in the vocabulary.
    let minimum = 256 + SPECIAL_TOKENS.len();
    if config.vocab_size < minimum {
        anyhow::bail!(
            "--vocab {} is too small: byte-level BPE needs at least {} (256 bytes + {} specials)",
            config.vocab_size,
            minimum,
            SPECIAL_TOKENS.len()
        );
    }

    let files = collect_corpus_files(&config.corpus)?;
    if files.is_empty() {
        anyhow::bail!("corpus {} contains no readable files", config.corpus.display());
    }

    let mut texts: Vec<String> = Vec::with_capacity(files.len());
    let mut used: Vec<&CorpusFile> = Vec::with_capacity(files.len());
    let mut skipped_non_utf8 = 0usize;
    for file in &files {
        let bytes = std::fs::read(&file.absolute)
            .with_context(|| format!("read corpus file {}", file.absolute.display()))?;
        match String::from_utf8(bytes) {
            Ok(text) => {
                used.push(file);
                texts.push(text);
            }
            Err(_) => skipped_non_utf8 += 1,
        }
    }
    if texts.is_empty() {
        anyhow::bail!(
            "corpus {} has no UTF-8 text ({} non-UTF-8 file(s) skipped)",
            config.corpus.display(),
            skipped_non_utf8
        );
    }

    let specials: Vec<AddedToken> =
        SPECIAL_TOKENS.iter().map(|token| AddedToken::from(*token, true)).collect();
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(false)
        .min_frequency(2)
        .vocab_size(config.vocab_size)
        // Seed all 256 byte-level characters. Without this the alphabet is only
        // the bytes the sample happened to contain, so an unseen byte would
        // encode as an unknown token and the round trip would fail.
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .special_tokens(specials)
        .build();

    let mut tokenizer: TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(BPE::default());
    // `add_prefix_space = false`: no synthetic leading space. `trim_offsets =
    // false`: offsets stay byte-exact. `use_regex = true`: split on the
    // byte-level pattern, which keeps whitespace in the tokens.
    tokenizer
        .with_pre_tokenizer(Some(ByteLevel::new(false, false, true)))
        .with_decoder(Some(ByteLevel::new(false, false, true)));
    // Deliberately no normaliser and no post-processor: no NFC, no lowercase.
    tokenizer
        .train(&mut trainer, texts.iter().map(String::as_str))
        .map_err(|e| anyhow::anyhow!("BPE training failed: {e}"))?;

    for token in SPECIAL_TOKENS {
        if tokenizer.token_to_id(token).is_none() {
            anyhow::bail!("trained tokenizer is missing special token {token}");
        }
    }

    std::fs::create_dir_all(&config.out_dir)
        .with_context(|| format!("create output dir {}", config.out_dir.display()))?;

    let tokenizer_path = config.out_dir.join("tokenizer.json");
    tokenizer
        .save(&tokenizer_path, false)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", tokenizer_path.display()))?;

    let mut special_map = serde_json::Map::new();
    for token in SPECIAL_TOKENS {
        let id = tokenizer
            .token_to_id(token)
            .ok_or_else(|| anyhow::anyhow!("special token {token} has no id"))?;
        special_map.insert(token.to_string(), serde_json::json!(id));
    }
    let specials_path = config.out_dir.join("special_tokens.json");
    std::fs::write(
        &specials_path,
        serde_json::to_vec_pretty(&serde_json::Value::Object(special_map))?,
    )
    .with_context(|| format!("write {}", specials_path.display()))?;

    let manifest_path = config.out_dir.join("sample-manifest.jsonl");
    let mut manifest = String::new();
    for file in &used {
        let licence = config.licence_for(&file.relative);
        let line =
            ManifestLine { path: &file.relative, sha256: &file.sha256, bytes: file.bytes, licence };
        manifest.push_str(&serde_json::to_string(&line)?);
        manifest.push('\n');
    }
    std::fs::write(&manifest_path, manifest.as_bytes())
        .with_context(|| format!("write {}", manifest_path.display()))?;

    let tokenizer_bytes = std::fs::read(&tokenizer_path)?;
    let tokenizer_sha256 = sha256_hex(&tokenizer_bytes);

    let mut sums = String::new();
    for name in ["sample-manifest.jsonl", "special_tokens.json", "tokenizer.json"] {
        let bytes = std::fs::read(config.out_dir.join(name))?;
        sums.push_str(&format!("{}  {}\n", sha256_hex(&bytes), name));
    }
    let sums_path = config.out_dir.join("SHA256SUMS");
    std::fs::write(&sums_path, sums.as_bytes())
        .with_context(|| format!("write {}", sums_path.display()))?;

    Ok(TrainReport {
        vocab_size: tokenizer.get_vocab_size(true),
        tokenizer_sha256,
        files: used.len(),
        skipped_non_utf8,
        out_dir: config.out_dir.clone(),
    })
}

/// Collect the corpus files in a stable order.
///
/// A single file is returned as-is. A directory is walked recursively; hidden
/// entries and the directories in `SKIP_DIRS` are skipped. The result is sorted
/// by relative path, so the training order does not depend on readdir order.
pub fn collect_corpus_files(corpus: &Path) -> Result<Vec<CorpusFile>> {
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    if corpus.is_file() {
        let name = corpus
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| corpus.display().to_string());
        found.push((corpus.to_path_buf(), name));
    } else if corpus.is_dir() {
        walk(corpus, corpus, &mut found)?;
    } else {
        anyhow::bail!("corpus {} does not exist", corpus.display());
    }

    found.sort_by(|a, b| a.1.cmp(&b.1));

    let mut files = Vec::with_capacity(found.len());
    for (absolute, relative) in found {
        let bytes = std::fs::read(&absolute)
            .with_context(|| format!("read corpus file {}", absolute.display()))?;
        files.push(CorpusFile {
            absolute,
            relative,
            sha256: sha256_hex(&bytes),
            bytes: bytes.len() as u64,
        });
    }
    Ok(files)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, String)>) -> Result<()> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("read directory {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("read directory entry in {}", dir.display()))?;
        let file_type = entry
            .file_type()
            .with_context(|| format!("file type of {}", entry.path().display()))?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            if SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            walk(root, &path, out)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            out.push((path, relative));
        }
    }
    Ok(())
}
