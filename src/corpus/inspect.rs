//! Verification, statistics and sampling for a built corpus.

use crate::corpus::build::{Index, Manifest, ShardRecord};
use crate::corpus::plan::{sha256_hex, Split};
use crate::corpus::stats::Stats;
use crate::model::tokenizer::SimpleTokenizer;
use anyhow::{bail, Context, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct VerifySummary {
    pub manifest_sha256: String,
    pub shards: Vec<ShardRecord>,
    pub tokens: u64,
    pub documents: u64,
    pub checks: usize,
}

#[derive(Debug, Clone)]
pub struct SampleOutput {
    pub split: Split,
    pub file: String,
    pub offset: u64,
    pub ids: Vec<u16>,
    pub text: String,
}

/// Re-hash every shard, check the manifest and the SHA256SUMS index, and check
/// that held-out document hashes never appear in a training shard.
pub fn verify(dir: &Path) -> Result<VerifySummary> {
    let manifest_path = dir.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("manifest missing at {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("{} is not a valid manifest", manifest_path.display()))?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);

    let index_path = dir.join("index.json");
    let index_bytes = std::fs::read(&index_path)
        .with_context(|| format!("index missing at {}", index_path.display()))?;
    let index: Index = serde_json::from_slice(&index_bytes)
        .with_context(|| format!("{} is not a valid index", index_path.display()))?;

    if index.seq_len != manifest.corpus.seq_len {
        bail!(
            "index seq_len {} does not match manifest seq_len {}",
            index.seq_len,
            manifest.corpus.seq_len
        );
    }
    if index.tokenizer_id != manifest.tokenizer.id {
        bail!(
            "index tokenizer '{}' does not match manifest tokenizer '{}'",
            index.tokenizer_id,
            manifest.tokenizer.id
        );
    }

    let mut checks = 0usize;
    let mut all_tokens = 0u64;
    for shard in &manifest.shards {
        let path = dir.join(&shard.file);
        let data =
            std::fs::read(&path).with_context(|| format!("shard missing: {}", shard.file))?;
        let actual = sha256_hex(&data);
        if actual != shard.sha256 {
            bail!(
                "shard hash mismatch: {} (expected {}, got {})",
                shard.file,
                shard.sha256,
                actual
            );
        }
        if data.len() as u64 != shard.tokens * 2 {
            bail!(
                "shard size mismatch: {} has {} bytes for {} tokens",
                shard.file,
                data.len(),
                shard.tokens
            );
        }
        for (position, chunk) in data.chunks_exact(2).enumerate() {
            let id = u16::from_le_bytes([chunk[0], chunk[1]]);
            if id as usize >= manifest.tokenizer.vocab_size {
                bail!(
                    "shard {} holds token id {} at index {} (vocab size {})",
                    shard.file,
                    id,
                    position,
                    manifest.tokenizer.vocab_size
                );
            }
        }

        let index_shard = index
            .shards
            .iter()
            .find(|entry| entry.file == shard.file)
            .with_context(|| format!("index missing shard {}", shard.file))?;
        if index_shard.tokens != shard.tokens {
            bail!(
                "index token count mismatch for {} (index {}, manifest {})",
                shard.file,
                index_shard.tokens,
                shard.tokens
            );
        }
        if index_shard.split != shard.split {
            bail!("index split mismatch for {}", shard.file);
        }
        if index_shard.documents.len() as u64 != shard.documents {
            bail!(
                "index document count mismatch for {} (index {}, manifest {})",
                shard.file,
                index_shard.documents.len(),
                shard.documents
            );
        }
        for doc in &index_shard.documents {
            if doc.token_start > doc.token_end || doc.token_end > shard.tokens {
                bail!(
                    "index document offsets out of range in {} for {}/{}",
                    shard.file,
                    doc.source_id,
                    doc.path
                );
            }
            if doc.byte_start != doc.token_start * 2 || doc.byte_end != doc.token_end * 2 {
                bail!(
                    "index byte offsets do not match token offsets in {} for {}/{}",
                    shard.file,
                    doc.source_id,
                    doc.path
                );
            }
        }

        all_tokens += shard.tokens;
        checks += 1;
    }
    if all_tokens != manifest.counts.tokens {
        bail!(
            "shard token total {} does not match manifest {}",
            all_tokens,
            manifest.counts.tokens
        );
    }

    // Eval isolation: a held-out document may never appear in a training shard.
    let heldout_hashes: BTreeSet<&str> = index
        .shards
        .iter()
        .filter(|shard| shard.split == Split::Heldout)
        .flat_map(|shard| shard.documents.iter())
        .map(|doc| doc.sha256.as_str())
        .collect();
    for shard in index.shards.iter().filter(|s| s.split == Split::Train) {
        for doc in &shard.documents {
            if heldout_hashes.contains(doc.sha256.as_str()) {
                bail!(
                    "eval isolation violated: held-out document hash {} appears in train shard {}",
                    doc.sha256,
                    shard.file
                );
            }
        }
    }

    // SHA256SUMS must list every artifact and agree with the bytes on disk.
    let sums_path = dir.join("SHA256SUMS");
    let sums = std::fs::read_to_string(&sums_path)
        .with_context(|| format!("SHA256SUMS missing at {}", sums_path.display()))?;
    let mut listed: BTreeMap<String, String> = BTreeMap::new();
    for line in sums.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let hash = parts.next().with_context(|| format!("SHA256SUMS is malformed: {line}"))?;
        let file = parts
            .next()
            .with_context(|| format!("SHA256SUMS is malformed: {line}"))?
            .trim_start_matches('*');
        if hash.len() != 64 {
            bail!("SHA256SUMS is malformed: {line}");
        }
        listed.insert(file.to_string(), hash.to_string());
    }
    for shard in &manifest.shards {
        match listed.get(&shard.file) {
            Some(hash) if hash == &shard.sha256 => {}
            Some(_) => bail!("SHA256SUMS disagrees with manifest for {}", shard.file),
            None => bail!("SHA256SUMS does not list shard {}", shard.file),
        }
    }
    for (file, expected) in &listed {
        let path = dir.join(file);
        let data = std::fs::read(&path)
            .with_context(|| format!("SHA256SUMS lists missing file {file}"))?;
        let actual = sha256_hex(&data);
        if &actual != expected {
            bail!("SHA256SUMS hash mismatch for {file} (expected {expected}, got {actual})");
        }
        checks += 1;
    }

    Ok(VerifySummary {
        manifest_sha256,
        shards: manifest.shards.clone(),
        tokens: manifest.counts.tokens,
        documents: manifest.counts.documents,
        checks,
    })
}

/// Read the stats written by the builder.
pub fn stats(dir: &Path) -> Result<Stats> {
    let path = dir.join("stats.json");
    let bytes =
        std::fs::read(&path).with_context(|| format!("stats missing at {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .with_context(|| format!("{} is not valid stats JSON", path.display()))
}

/// Sample `n` tokens from one split of a built corpus and decode them. The
/// offset is a seeded function of the manifest seed.
pub fn sample(dir: &Path, n: usize, split: Option<Split>) -> Result<SampleOutput> {
    let manifest_path = dir.join("manifest.json");
    let manifest_bytes = std::fs::read(&manifest_path)
        .with_context(|| format!("manifest missing at {}", manifest_path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
        .with_context(|| format!("{} is not a valid manifest", manifest_path.display()))?;

    let chosen = match split {
        Some(requested) => manifest
            .shards
            .iter()
            .find(|shard| shard.split == requested && shard.tokens > 0)
            .with_context(|| format!("split {} has no tokens to sample", requested.as_str()))?,
        None => manifest
            .shards
            .iter()
            .find(|shard| shard.split == Split::Train && shard.tokens > 0)
            .or_else(|| manifest.shards.iter().find(|shard| shard.tokens > 0))
            .context("corpus has no tokens to sample")?,
    };

    let path = dir.join(&chosen.file);
    let data = std::fs::read(&path).with_context(|| format!("shard missing: {}", chosen.file))?;
    let available = (data.len() / 2) as u64;
    let take = (n as u64).min(available) as usize;

    let salt = match chosen.split {
        Split::Train => 0u64,
        Split::Val => 1,
        Split::Heldout => 2,
    };
    let mut rng = StdRng::seed_from_u64(manifest.corpus.seed ^ salt);
    let max_offset = available.saturating_sub(take as u64);
    let offset = if max_offset == 0 { 0 } else { rng.gen_range(0..=max_offset) };

    let start = (offset * 2) as usize;
    let end = start + take * 2;
    let ids: Vec<u16> = data[start..end]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    let text =
        SimpleTokenizer::new().decode(&ids.iter().map(|id| *id as u32).collect::<Vec<u32>>());

    Ok(SampleOutput { split: chosen.split, file: chosen.file.clone(), offset, ids, text })
}
