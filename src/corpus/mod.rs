//! Corpus builder: licensed sources -> reproducible token shards.
//!
//! See `docs/corpus-builder.md` for the full design. This module implements the
//! local, git and hf source adapters, the local-file stages, and a byte
//! tokenizer. Near-dedupe and the shipped 32k tokenizer are still to come.

pub mod build;
pub mod inspect;
pub mod plan;
pub mod secret;
pub mod sources;
pub mod stats;

pub use build::{build, BuildSummary, Index, Manifest, ShardRecord};
pub use inspect::{sample, stats, verify, SampleOutput, VerifySummary};
pub use plan::{classify_split, Plan, SourceKind, SourceSpec, Split};
pub use sources::{acquire, AcquiredFile, AcquiredSource, Source};
pub use stats::Stats;

use anyhow::{Context, Result};
use clap::Subcommand;
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub enum CorpusCommand {
    /// Build a corpus from a JSON plan of local files.
    Build {
        /// Path to the JSON plan.
        #[arg(long)]
        plan: PathBuf,
        /// Output directory for shards and manifest.
        #[arg(long)]
        out: PathBuf,
        /// Reuse a completed build in --out (re-verified before use).
        #[arg(long)]
        resume: bool,
    },
    /// Re-hash shards and check the manifest.
    Verify {
        /// Corpus directory produced by `corpus build`.
        dir: PathBuf,
    },
    /// Report tokens, documents, splits, licence mix and drop reasons.
    Stats {
        /// Corpus directory produced by `corpus build`.
        dir: PathBuf,
        /// Emit JSON instead of human-readable text.
        #[arg(long)]
        json: bool,
    },
    /// Sample tokens from a shard and print them (optionally decoded).
    Sample {
        /// Corpus directory produced by `corpus build`.
        dir: PathBuf,
        /// Number of tokens to sample.
        #[arg(long, default_value_t = 20)]
        n: usize,
        /// Split to sample: train, val or heldout (default: first non-empty).
        #[arg(long)]
        split: Option<String>,
        /// Decode the sampled ids back to text.
        #[arg(long)]
        decode: bool,
    },
}

/// Execute a corpus subcommand.
pub fn run(command: CorpusCommand) -> Result<()> {
    match command {
        CorpusCommand::Build { plan, out, resume } => {
            let summary = build(&plan, &out, resume)?;
            println!(
                "built corpus at {} ({}{} documents, {} tokens, manifest sha256 {})",
                out.display(),
                if summary.resumed { "resumed, " } else { "" },
                summary.documents,
                summary.tokens,
                summary.manifest_sha256
            );
            for shard in &summary.shards {
                println!(
                    "  {}: {} tokens, {} documents, sha256 {}",
                    shard.file, shard.tokens, shard.documents, shard.sha256
                );
            }
            Ok(())
        }
        CorpusCommand::Verify { dir } => {
            let summary = verify(&dir)?;
            println!(
                "verified {}: {} shards, {} documents, {} tokens, {} checks",
                dir.display(),
                summary.shards.len(),
                summary.documents,
                summary.tokens,
                summary.checks
            );
            println!("manifest sha256 {}", summary.manifest_sha256);
            Ok(())
        }
        CorpusCommand::Stats { dir, json } => {
            let stats = stats(&dir)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&stats).context("cannot serialize stats JSON")?
                );
            } else {
                print!("{}", stats.to_human());
            }
            Ok(())
        }
        CorpusCommand::Sample { dir, n, split, decode } => {
            let split = match split.as_deref() {
                Some(value) => Some(Split::parse(value)?),
                None => None,
            };
            let output = sample(&dir, n, split)?;
            println!(
                "sampled {} tokens from {} ({} at offset {})",
                output.ids.len(),
                output.file,
                output.split.as_str(),
                output.offset
            );
            if decode {
                print!("{}", output.text);
                if !output.text.ends_with('\n') {
                    println!();
                }
            } else {
                println!("{:?}", output.ids);
            }
            Ok(())
        }
    }
}
