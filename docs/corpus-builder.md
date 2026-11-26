# Corpus builder

Status: **v1 implemented at `b3a9a89`** (local files, determinism, licence gate,
secret scan, exact dedupe, u16 shards, split, manifest, verify/stats/sample; 6
tests). The fetchers, near-dedupe and the 32k tokenizer remain. Specification: `dataset-formats.md` defines what a sample
looks like; this defines how licensed sources become reproducible tokenized
shards. No builder exists yet; every row in `source-index.md` is still `planned`.

## Shape

```
source-index row → acquire → scan → filter → dedupe → render → tokenize → pack
                 → shard → split → manifest → gate
```

Each stage records what it removed and why. A stage that cannot report its
drop count is not allowed to run silently.

## Stages

| stage | does | records |
|---|---|---|
| **acquire** | fetch a pinned revision (dataset commit, repo hash, snapshot date#hash); never `main` | source id, revision, bytes, sha256 |
| **licence** | per-file or per-source allowlist from `model-licensing.md`; unknown = drop | licence, spdx per file, dropped count |
| **scan** | secrets (keys, tokens, credentials), PII patterns, binary/minified/generated files | patterns hit, files dropped |
| **filter** | language ID, length bounds, quality score where the source defines one | thresholds, drop counts per reason |
| **dedupe** | exact hash, then MinHash/SimHash near-duplicates; per-source and cross-source | duplicate rate, kept count |
| **render** | chat template for SFT; raw text for pretraining; tool calls and results for tool data | template id + hash |
| **tokenize** | the frozen tokenizer (`tokenizer.md`); no normalisation | tokenizer id, version, sha256 |
| **pack** | windows of `seq_len`; documents never share a window for eval; allow packing for training with a document-boundary token | window count, boundary policy |
| **shard** | `u16` token ids (32k vocab fits), little-endian, with a document index | shard size, count, sha256 |
| **split** | train/val/held-out **by source and by hash**, not by random line | split rule, counts |
| **manifest** | one JSON: sources, revisions, licences, hashes, filters, counts, tokenizer, seed, builder version | the reproducibility contract |

## Determinism

- Files are sorted by `(source id, relative path, sha256)` before sampling; no
  filesystem order, no clock.
- The sample is a seeded function of that order; the seed is in the manifest.
- Two builds from the same manifest must produce the **same shard hashes**.
  A rebuild that differs is a bug, not a new corpus.
- The builder version (`ferrisres` commit) is recorded; a tokenizer or filter
  change is a new corpus version, never a silent overwrite.

## Layout

```
corpus/<name>/<version>/
  shards/train-00000.bin
  shards/val-00000.bin
  shards/heldout-00000.bin
  index.json              document offsets per shard
  manifest.json           the contract above
  stats.json              tokens, docs, drop counts, licence mix, language mix
  SHA256SUMS
```

`heldout/` is written once and never read by the trainer. The builder refuses to
write a training shard that contains a held-out document hash.

## Storage

| tokens | u16 shard | note |
|---:|---:|---|
| 100M | 200 MB | a smoke / SFT set |
| 2B | 4 GB | the first real pretraining corpus |
| 10B | 20 GB | beyond Kaggle's comfortable dataset size |

`u16` halves the I/O versus `u32` and the 32k vocab fits exactly. If a future
tokenizer exceeds 65k, the format version changes, not the existing shards.

## Commands

```
ferrisres corpus build --plan corpus.yaml --out corpus/frs-pretrain/0.1.0 --seed 42
ferrisres corpus verify corpus/frs-pretrain/0.1.0
ferrisres corpus stats  corpus/frs-pretrain/0.1.0 --json
ferrisres corpus sample corpus/frs-pretrain/0.1.0 --n 20 --decode
```

`build` is resumable per stage and writes a stage journal, so a killed fetch does
not repeat a completed dedupe. `verify` re-hashes every shard and checks the
manifest; it is the gate a Kaggle dataset runs before training.

## Gates

| gate | pass |
|---|---|
| licence coverage | 100% of rows have a recorded licence; unknown dropped and counted |
| secret scan | zero findings in the output; the scan report is in the manifest |
| eval isolation | zero held-out hashes in any training shard |
| round trip | a random sample from each shard decodes to the source text |
| windows | no window crosses a document boundary in eval; training packing is explicit |
| determinism | a rebuild produces identical shard hashes |
| tokenizer | the hash matches the model config; a mismatch refuses the build |
| stats | tokens, docs, language and licence mix, and every drop reason are reported |

## Plan file

```yaml
name: frs-pretrain
version: 0.1.0
tokenizer: frs-tokenizer-32k/0.1.0
seq_len: 8192
seed: 42
sources:
  - id: code-rust-lang
    weight: 1.0
    filters: {min_bytes: 200, max_bytes: 2000000}
  - id: en-fineweb-edu
    weight: 1.0
    sample_tokens: 400000000
split:
  heldout: {by: source_hash, fraction: 0.02}
```

The plan references `source-index.md` ids; it does not repeat licences or URLs.

## What it does not do

- Fetch a gated source without credentials the operator supplies.
- Redact PII in arbitrary code; the scanner reports and drops, it does not edit.
- Decide a licence question; an unresolved row stops the build.
- Train a tokenizer (`tokenizer.md` owns that).
- Replace the eval runner; it only guarantees the held-out set is untouched.

## Why this is the next gate

Every later step assumes a corpus that is licensed, deduplicated, tokenised with
the shipping tokenizer, split without contamination, and reproducible by hash.
Without it, a training run cannot be repeated, a licence cannot be defended, and
a contamination claim cannot be checked. It is also the natural Kaggle dataset
artefact: the notebook attaches the shards the builder produced, not a checkout
of scraped text.

## The 32k rebuild (2026-11-26)

Same six sources, same splits, only the tokenizer changed:

| | byte tokenizer | 32k BPE |
|---|---:|---:|
| tokens | 22,713,344 | **6,076,416** |
| documents | 1,472 | 1,472 |
| assigned train / val / heldout | 1,430 / 11 / 31 | 1,430 / 11 / 31 |
| documents with ≥512-token windows | 1,379 | **1,143** |
| licence mix | identical | identical |
| verify checks | 9 | 9 |

Two consequences worth carrying forward:

1. **512 BPE tokens is a much longer document than 512 bytes**, so 236 more
   documents fall below one window and contribute no tokens. A shorter
   `seq_len`, or smaller windows for short documents, would recover them.
2. 6.08M tokens is a **slice**, not a training corpus. At ~3.7 bytes/token the
   ratio is expected; the volume gap is separate and unchanged.
