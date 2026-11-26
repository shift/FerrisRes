//! Deterministic local-file corpus builder.
//!
//! Pipeline: acquire (local files) -> licence gate -> scan -> filter -> dedupe
//! -> tokenize -> pack -> shard -> split -> manifest.

use crate::corpus::plan::{classify_split, sha256_hex, Plan, Split, TokenizerRecord};
use crate::corpus::secret;
use crate::corpus::stats::Stats;
use crate::model::tokenizer::SimpleTokenizer;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const SUPPORTED_EXTENSIONS: [&str; 5] = ["jsonl", "md", "nix", "rs", "txt"];
pub const BOUNDARY_POLICY: &str = "no_cross_document";
pub const REMAINDER_POLICY: &str = "drop_partial_window";

// ---------------------------------------------------------------------------
// Manifest / index records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuilderRecord {
    pub name: String,
    pub version: String,
    pub plan_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusRecord {
    pub name: String,
    pub version: String,
    pub seq_len: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitRecord {
    pub val: f64,
    pub heldout: f64,
    pub rule: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceRecord {
    pub id: String,
    pub path: String,
    pub licence: String,
    pub revision: String,
    pub weight: f64,
    pub files_seen: u64,
    pub files_kept: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRecord {
    pub source_id: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub tokens: u64,
    pub split: Split,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiltersRecord {
    pub extensions: Vec<String>,
    pub boundary_policy: String,
    pub remainder_policy: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackingRecord {
    pub seq_len: usize,
    pub boundary_policy: String,
    pub remainder_policy: String,
    pub remainder_tokens_dropped: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CountsRecord {
    pub files_seen: u64,
    pub documents: u64,
    pub bytes: u64,
    pub tokens: u64,
    pub documents_by_split: BTreeMap<String, u64>,
    pub tokens_by_split: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecretScanRecord {
    pub patterns: Vec<String>,
    pub findings: u64,
    pub files_dropped: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardRecord {
    pub split: Split,
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    pub tokens: u64,
    pub documents: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub builder: BuilderRecord,
    pub corpus: CorpusRecord,
    pub tokenizer: TokenizerRecord,
    pub split: SplitRecord,
    pub sources: Vec<SourceRecord>,
    pub files: Vec<FileRecord>,
    pub filters: FiltersRecord,
    pub packing: PackingRecord,
    pub counts: CountsRecord,
    pub drop_reasons: BTreeMap<String, u64>,
    pub secret_scan: SecretScanRecord,
    pub shards: Vec<ShardRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexDoc {
    pub source_id: String,
    pub path: String,
    pub sha256: String,
    pub token_start: u64,
    pub token_end: u64,
    pub byte_start: u64,
    pub byte_end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexShard {
    pub split: Split,
    pub file: String,
    pub tokens: u64,
    pub documents: Vec<IndexDoc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Index {
    pub seq_len: usize,
    pub tokenizer_id: String,
    pub shards: Vec<IndexShard>,
}

// ---------------------------------------------------------------------------
// Build result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BuildSummary {
    pub out_dir: PathBuf,
    pub manifest_sha256: String,
    pub shards: Vec<ShardRecord>,
    pub tokens: u64,
    pub documents: u64,
    pub files_seen: u64,
    pub drop_reasons: BTreeMap<String, u64>,
    pub resumed: bool,
}

impl BuildSummary {
    pub(crate) fn from_manifest(
        out_dir: &Path,
        manifest: &Manifest,
        manifest_sha256: String,
        resumed: bool,
    ) -> Self {
        Self {
            out_dir: out_dir.to_path_buf(),
            manifest_sha256,
            shards: manifest.shards.clone(),
            tokens: manifest.counts.tokens,
            documents: manifest.counts.documents,
            files_seen: manifest.counts.files_seen,
            drop_reasons: manifest.drop_reasons.clone(),
            resumed,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal records
// ---------------------------------------------------------------------------

struct RawFile {
    source_id: String,
    rel_path: String,
    extension: String,
    bytes: Vec<u8>,
    sha256: String,
}

struct Doc {
    source_id: String,
    rel_path: String,
    raw_sha256: String,
    bytes: u64,
    tokens: Vec<u16>,
    split: Split,
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// Build a corpus from a local-file plan.
pub fn build(plan_path: &Path, out: &Path, resume: bool) -> Result<BuildSummary> {
    let (plan_bytes, plan) = Plan::load(plan_path)?;
    let plan_sha256 = sha256_hex(&plan_bytes);

    let manifest_path = out.join("manifest.json");
    if manifest_path.exists() {
        if !resume {
            bail!(
                "output directory {} already contains a corpus; pass --resume or choose a new --out",
                out.display()
            );
        }
        let existing_bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("cannot read {}", manifest_path.display()))?;
        let existing: Manifest = serde_json::from_slice(&existing_bytes)
            .with_context(|| format!("{} is not a valid manifest", manifest_path.display()))?;
        if existing.builder.plan_sha256 != plan_sha256 {
            bail!("cannot resume: plan {} changed since the corpus was built", plan_path.display());
        }
        let verified = crate::corpus::inspect::verify(out).with_context(|| {
            format!("cannot resume: existing corpus at {} failed verification", out.display())
        })?;
        return Ok(BuildSummary::from_manifest(out, &existing, verified.manifest_sha256, true));
    }

    if out.exists() && !resume && dir_has_entries(out)? {
        bail!(
            "output directory {} is not empty; pass --resume or choose a new --out",
            out.display()
        );
    }
    std::fs::create_dir_all(out.join("shards"))
        .with_context(|| format!("cannot create {}", out.join("shards").display()))?;

    let mut sources = plan.sources.clone();
    sources.sort_by(|a, b| a.id.cmp(&b.id));

    let mut files_seen_by_source: BTreeMap<String, u64> = BTreeMap::new();
    let mut raw_files = Vec::new();
    for source in &sources {
        let root = PathBuf::from(&source.path);
        if !root.exists() {
            bail!("source '{}' path does not exist: {}", source.id, root.display());
        }
        let discovered =
            discover_files(&root).with_context(|| format!("cannot scan source '{}'", source.id))?;
        *files_seen_by_source.entry(source.id.clone()).or_insert(0) += discovered.len() as u64;
        for (rel_path, abs_path, extension) in discovered {
            let bytes = std::fs::read(&abs_path)
                .with_context(|| format!("cannot read {}", abs_path.display()))?;
            let sha256 = sha256_hex(&bytes);
            raw_files.push(RawFile {
                source_id: source.id.clone(),
                rel_path,
                extension,
                bytes,
                sha256,
            });
        }
    }

    raw_files.sort_by(|a, b| {
        a.source_id
            .cmp(&b.source_id)
            .then_with(|| a.rel_path.cmp(&b.rel_path))
            .then_with(|| a.sha256.cmp(&b.sha256))
    });

    let tokenizer = SimpleTokenizer::new();
    let mut docs: Vec<Doc> = Vec::new();
    let mut seen_hashes: BTreeSet<String> = BTreeSet::new();
    let mut drop_reasons: BTreeMap<String, u64> = BTreeMap::new();
    let mut secret_findings = 0u64;
    let mut secret_files_dropped = 0u64;

    for raw in raw_files {
        let text = if raw.extension == "jsonl" {
            match jsonl_text(&raw.bytes) {
                Ok(text) => text,
                Err(reason) => {
                    *drop_reasons.entry(reason.to_string()).or_insert(0) += 1;
                    continue;
                }
            }
        } else {
            match String::from_utf8(raw.bytes.clone()) {
                Ok(text) => text,
                Err(_) => {
                    *drop_reasons.entry("invalid_utf8".to_string()).or_insert(0) += 1;
                    continue;
                }
            }
        };

        let trimmed = text.trim();
        if trimmed.is_empty() {
            *drop_reasons.entry("empty".to_string()).or_insert(0) += 1;
            continue;
        }

        let hits = secret::scan(&text);
        if !hits.is_empty() {
            secret_findings += hits.len() as u64;
            secret_files_dropped += 1;
            *drop_reasons.entry("secret".to_string()).or_insert(0) += 1;
            continue;
        }

        let dedupe_hash = sha256_hex(trimmed.as_bytes());
        if !seen_hashes.insert(dedupe_hash.clone()) {
            *drop_reasons.entry("duplicate".to_string()).or_insert(0) += 1;
            continue;
        }

        let split = classify_split(&dedupe_hash, plan.split.val, plan.split.heldout);
        let tokens: Vec<u16> = tokenizer.encode(&text).into_iter().map(|id| id as u16).collect();

        docs.push(Doc {
            source_id: raw.source_id,
            rel_path: raw.rel_path,
            raw_sha256: raw.sha256,
            bytes: raw.bytes.len() as u64,
            tokens,
            split,
        });
    }

    let mut documents_by_split: BTreeMap<String, u64> = BTreeMap::new();
    let mut tokens_by_split: BTreeMap<String, u64> = BTreeMap::new();
    let mut remainder_by_split: BTreeMap<String, u64> = BTreeMap::new();
    for split in [Split::Train, Split::Val, Split::Heldout] {
        documents_by_split.insert(split.as_str().to_string(), 0);
        tokens_by_split.insert(split.as_str().to_string(), 0);
        remainder_by_split.insert(split.as_str().to_string(), 0);
    }

    let mut file_records = Vec::new();
    let mut shards = Vec::new();
    let mut index_shards = Vec::new();

    for split in [Split::Train, Split::Val, Split::Heldout] {
        let split_name = split.as_str();
        let mut data: Vec<u8> = Vec::new();
        let mut token_count = 0u64;
        let mut shard_documents = 0u64;
        let mut index_docs = Vec::new();

        for doc in docs.iter().filter(|doc| doc.split == split) {
            *documents_by_split.get_mut(split_name).expect("split key") += 1;

            let full_windows = doc.tokens.len() / plan.seq_len;
            let kept_tokens = full_windows * plan.seq_len;
            let remainder = doc.tokens.len() - kept_tokens;
            *remainder_by_split.get_mut(split_name).expect("split key") += remainder as u64;

            let token_start = token_count;
            for token in &doc.tokens[..kept_tokens] {
                data.extend_from_slice(&token.to_le_bytes());
            }
            token_count += kept_tokens as u64;

            if kept_tokens > 0 {
                index_docs.push(IndexDoc {
                    source_id: doc.source_id.clone(),
                    path: doc.rel_path.clone(),
                    sha256: doc.raw_sha256.clone(),
                    token_start,
                    token_end: token_count,
                    byte_start: token_start * 2,
                    byte_end: token_count * 2,
                });
                shard_documents += 1;
            }

            file_records.push(FileRecord {
                source_id: doc.source_id.clone(),
                path: doc.rel_path.clone(),
                sha256: doc.raw_sha256.clone(),
                bytes: doc.bytes,
                tokens: kept_tokens as u64,
                split,
            });
        }

        tokens_by_split.insert(split_name.to_string(), token_count);
        let file = format!("shards/{split_name}-00000.bin");
        let path = out.join(&file);
        std::fs::write(&path, &data).with_context(|| format!("cannot write {}", path.display()))?;
        shards.push(ShardRecord {
            split,
            file: file.clone(),
            sha256: sha256_hex(&data),
            bytes: data.len() as u64,
            tokens: token_count,
            documents: shard_documents,
        });
        index_shards.push(IndexShard { split, file, tokens: token_count, documents: index_docs });
    }

    let mut files_kept_by_source: BTreeMap<String, u64> = BTreeMap::new();
    for doc in &docs {
        *files_kept_by_source.entry(doc.source_id.clone()).or_insert(0) += 1;
    }

    let source_records: Vec<SourceRecord> = sources
        .iter()
        .map(|source| SourceRecord {
            id: source.id.clone(),
            path: source.path.clone(),
            licence: source.licence.clone().unwrap_or_default(),
            revision: source.revision.clone().unwrap_or_default(),
            weight: source.weight,
            files_seen: files_seen_by_source.get(&source.id).copied().unwrap_or(0),
            files_kept: files_kept_by_source.get(&source.id).copied().unwrap_or(0),
        })
        .collect();

    let total_tokens: u64 = tokens_by_split.values().sum();
    let total_bytes: u64 = docs.iter().map(|doc| doc.bytes).sum();

    let manifest = Manifest {
        builder: BuilderRecord {
            name: "ferrisres-corpus".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            plan_sha256,
        },
        corpus: CorpusRecord {
            name: plan.name.clone(),
            version: plan.version.clone(),
            seq_len: plan.seq_len,
            seed: plan.seed,
        },
        tokenizer: crate::corpus::plan::tokenizer_record(),
        split: SplitRecord {
            val: plan.split.val,
            heldout: plan.split.heldout,
            rule: "sha256_document_heldout_then_val_then_train".to_string(),
        },
        sources: source_records,
        files: file_records,
        filters: FiltersRecord {
            extensions: SUPPORTED_EXTENSIONS.iter().map(|ext| ext.to_string()).collect(),
            boundary_policy: BOUNDARY_POLICY.to_string(),
            remainder_policy: REMAINDER_POLICY.to_string(),
        },
        packing: PackingRecord {
            seq_len: plan.seq_len,
            boundary_policy: BOUNDARY_POLICY.to_string(),
            remainder_policy: REMAINDER_POLICY.to_string(),
            remainder_tokens_dropped: remainder_by_split,
        },
        counts: CountsRecord {
            files_seen: files_seen_by_source.values().sum(),
            documents: docs.len() as u64,
            bytes: total_bytes,
            tokens: total_tokens,
            documents_by_split,
            tokens_by_split,
        },
        drop_reasons,
        secret_scan: SecretScanRecord {
            patterns: secret::pattern_names(),
            findings: secret_findings,
            files_dropped: secret_files_dropped,
        },
        shards: shards.clone(),
    };

    let index = Index {
        seq_len: plan.seq_len,
        tokenizer_id: manifest.tokenizer.id.clone(),
        shards: index_shards,
    };

    let index_bytes = json_bytes(&index)?;
    let stats = Stats::from_manifest(&manifest);
    let stats_bytes = json_bytes(&stats)?;
    let manifest_bytes = json_bytes(&manifest)?;

    std::fs::write(out.join("index.json"), &index_bytes)
        .with_context(|| format!("cannot write {}", out.join("index.json").display()))?;
    std::fs::write(out.join("stats.json"), &stats_bytes)
        .with_context(|| format!("cannot write {}", out.join("stats.json").display()))?;
    std::fs::write(&manifest_path, &manifest_bytes)
        .with_context(|| format!("cannot write {}", manifest_path.display()))?;

    let mut sums = String::new();
    for shard in &shards {
        sums.push_str(&format!("{}  {}\n", shard.sha256, shard.file));
    }
    sums.push_str(&format!("{}  index.json\n", sha256_hex(&index_bytes)));
    sums.push_str(&format!("{}  manifest.json\n", sha256_hex(&manifest_bytes)));
    sums.push_str(&format!("{}  stats.json\n", sha256_hex(&stats_bytes)));
    std::fs::write(out.join("SHA256SUMS"), sums)
        .with_context(|| format!("cannot write {}", out.join("SHA256SUMS").display()))?;

    Ok(BuildSummary::from_manifest(out, &manifest, sha256_hex(&manifest_bytes), false))
}

fn dir_has_entries(dir: &Path) -> Result<bool> {
    let mut entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?;
    Ok(entries.next().is_some())
}

fn discover_files(root: &Path) -> Result<Vec<(String, PathBuf, String)>> {
    let mut found = Vec::new();
    if root.is_file() {
        if let Some(ext) = supported_extension(root) {
            let name = root
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            found.push((name, root.to_path_buf(), ext));
        }
        return Ok(found);
    }

    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .with_context(|| format!("cannot read directory {}", dir.display()))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let file_type = entry.file_type()?;
            let path = entry.path();
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() {
                if let Some(ext) = supported_extension(&path) {
                    let rel = path
                        .strip_prefix(root)
                        .unwrap_or(&path)
                        .components()
                        .map(|component| component.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                        .join("/");
                    found.push((rel, path, ext));
                }
            }
        }
    }
    Ok(found)
}

fn supported_extension(path: &Path) -> Option<String> {
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    if SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
        Some(ext)
    } else {
        None
    }
}

/// Extract the `text` field from every non-empty JSONL line. One file is one
/// document; lines are joined with a newline.
fn jsonl_text(bytes: &[u8]) -> std::result::Result<String, &'static str> {
    let text = std::str::from_utf8(bytes).map_err(|_| "invalid_utf8")?;
    let mut out = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).map_err(|_| "jsonl_invalid")?;
        let field =
            value.get("text").and_then(|field| field.as_str()).ok_or("jsonl_missing_text")?;
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(field);
    }
    Ok(out)
}

fn json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value).context("cannot serialize JSON")?;
    bytes.push(b'\n');
    Ok(bytes)
}
