# Armor L1 corpus sources

This directory holds the **inputs** to the Armor L1 training corpus and the
**manifest** of what the build produced. It is data, not model code.

```
assets/armor/
  l1-data.toml          # generated manifest: one entry per family, licence, revision, n, sha256
  upstream/             # cached public datasets (committed) + <slug>.meta.json
  own/                  # hand-written red-team and benign samples (committed)
  translations/en_de.tsv# German translations of a permissive English subset (committed)
  build/                # generated JSONLs + stats.json (git-ignored, regenerable)
```

## Build

```bash
# fetch upstream only when the cache is absent (network)
python3 scripts/armor/fetch_upstream.py

# deterministic build; add --offline to forbid the network
python3 scripts/armor/build_corpus.py --verify-determinism
```

Two runs on the same inputs produce byte-identical `train.jsonl`, `dev.jsonl`,
`test.jsonl` and `stats.json`.

## Licence rule

Only Apache-2.0, MIT, CC-BY, CC0 and ODC-By families are accepted. The build
**fails** on a family with no licence or a non-permissive one; it does not skip
it silently. A dataset whose licence cannot be established is evaluation-only
and never enters a split.

| family | licence | revision |
|---|---|---|
| `deepset/prompt-injections` | Apache-2.0 | `4f61ecb0…` |
| `jackhhao/jailbreak-classification` | Apache-2.0 | `2f2ceeb3…` |
| `Lakera/gandalf_ignore_instructions` | MIT | `04737b65…` |
| `xTRam1/safe-guard-prompt-injection` | **none declared** | refused — evaluation-only |
| our own families (`own/`, generators) | CC0-1.0 | `armor-own-2026-11-26` |
| `translated:<dataset>` | inherited from the upstream dataset | upstream sha + translation |

The upstream cache is committed so that the build is reproducible offline. Its
`<slug>.meta.json` records the dataset id, the card licence and the exact
revision that was read.

## What is human-written, generated, translated or synthetic

| kind | where | honesty note |
|---|---|---|
| real public data | `upstream/*.jsonl` | 2 940 examples after dedup, English only, from three permissive datasets |
| hand-written | `own/*.jsonl` | 118 examples written for this corpus (red-team strings and benign tool output) |
| template-generated | `scripts/armor/generators.py` | the bulk of the corpus; combinatorial pools, so held-out families share vocabulary with train |
| translated | `translations/en_de.tsv` | 143 German translations of upstream English; **not yet native-reviewed**, marked `source: "translated:<dataset>"` |
| native German | generator pools | written in German by the build author; also not native-reviewed |

Every line is `redacted: true`: there are no session transcripts, no real
personal data, no credentials. Fake addresses use `example.com`/`example.net`;
fake paths live under `/tmp`; traffic is documentation-shaped.

## Composition of the first real build

- ~14.8 k examples, inside the 10 k–20 k target for a first real run.
- 69.4 / 15.8 / 14.8 by family for train/dev/test.
- 33.2 % German (target ≥ 30 %), 38.6 % hard negatives (target ≥ 30 %).
- 42 % injection / 58 % safe. The task doc asks for 50/50 after weighting;
  this build is left unweighted and the imbalance is reported rather than
  hidden by resampling.
- Obfuscated injections (base64, zero-width, homoglyph) are **dev/test only**.

## What this corpus is not

It is not sufficient for a shipped scanner. Upstream supply is ~3 k rows; the
rest is template combinatorics with shared vocabulary. The 60 k/7.5 k/7.5 k
target in `docs/tasks/armor-l1-enablement.md` needs roughly four times this
volume, more upstream families (with licences), native German authorship or
review, and obfuscations that were not generated from the same templates as
their plain forms. No model was trained here and no detection claim is made.

See `scripts/armor/README.md` for the pipeline and the family table.
