//! Corpus statistics derived from the manifest.

use crate::corpus::build::Manifest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplitStats {
    pub documents: u64,
    pub tokens: u64,
    pub shards: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub name: String,
    pub version: String,
    pub seq_len: usize,
    pub documents: u64,
    pub tokens: u64,
    pub splits: BTreeMap<String, SplitStats>,
    /// v1 does not identify languages; the map always contains `unknown`.
    pub languages: BTreeMap<String, u64>,
    pub licence_mix: BTreeMap<String, u64>,
    pub drop_reasons: BTreeMap<String, u64>,
    pub drop_total: u64,
    pub secret_findings: u64,
    pub secret_files_dropped: u64,
}

impl Stats {
    pub fn from_manifest(manifest: &Manifest) -> Self {
        let mut licence_mix: BTreeMap<String, u64> = BTreeMap::new();
        for file in &manifest.files {
            if let Some(source) = manifest.sources.iter().find(|s| s.id == file.source_id) {
                *licence_mix.entry(source.licence.clone()).or_insert(0) += 1;
            }
        }

        let mut splits = BTreeMap::new();
        for split in ["train", "val", "heldout"] {
            splits.insert(
                split.to_string(),
                SplitStats {
                    documents: manifest.counts.documents_by_split.get(split).copied().unwrap_or(0),
                    tokens: manifest.counts.tokens_by_split.get(split).copied().unwrap_or(0),
                    shards: manifest.shards.iter().filter(|s| s.split.as_str() == split).count()
                        as u64,
                },
            );
        }

        let drop_total = manifest.drop_reasons.values().sum();
        let mut languages = BTreeMap::new();
        languages.insert("unknown".to_string(), manifest.counts.documents);

        Self {
            name: manifest.corpus.name.clone(),
            version: manifest.corpus.version.clone(),
            seq_len: manifest.corpus.seq_len,
            documents: manifest.counts.documents,
            tokens: manifest.counts.tokens,
            splits,
            languages,
            licence_mix,
            drop_reasons: manifest.drop_reasons.clone(),
            drop_total,
            secret_findings: manifest.secret_scan.findings,
            secret_files_dropped: manifest.secret_scan.files_dropped,
        }
    }

    pub fn to_human(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("corpus {} {}\n", self.name, self.version));
        out.push_str(&format!(
            "tokens: {}  documents: {}  seq_len: {}\n",
            self.tokens, self.documents, self.seq_len
        ));
        for (split, values) in &self.splits {
            out.push_str(&format!(
                "  {split}: {} tokens, {} documents, {} shards\n",
                values.tokens, values.documents, values.shards
            ));
        }
        out.push_str("licence mix:\n");
        for (licence, count) in &self.licence_mix {
            out.push_str(&format!("  {licence}: {count} documents\n"));
        }
        out.push_str("languages:\n");
        for (language, count) in &self.languages {
            out.push_str(&format!("  {language}: {count}\n"));
        }
        out.push_str(&format!("dropped: {}", self.drop_total));
        if self.drop_reasons.is_empty() {
            out.push('\n');
        } else {
            out.push_str(" (");
            let reasons: Vec<String> = self
                .drop_reasons
                .iter()
                .map(|(reason, count)| format!("{reason}={count}"))
                .collect();
            out.push_str(&reasons.join(", "));
            out.push_str(")\n");
        }
        out.push_str(&format!(
            "secret scan: {} findings, {} files dropped\n",
            self.secret_findings, self.secret_files_dropped
        ));
        out
    }
}
