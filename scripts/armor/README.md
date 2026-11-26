# Armor corpus pipeline

Data work for the L1 injection scanner. Nothing here trains a model and
nothing here touches `src/training` or the L1 runtime.

## Files

| file | role |
|---|---|
| `fetch_upstream.py` | fetch the permissive public sets into `assets/armor/upstream/`, checking the licence on each card first; refuses anything not Apache-2.0 / MIT / CC-BY / CC0 / ODC-By |
| `generators.py` | deterministic template families we author: English and native German injections, security documentation, red-team reports, code comments, imperative tool output, benign docs and tool output |
| `build_corpus.py` | normalise, deduplicate, split by family, write JSONLs, `stats.json` and `assets/armor/l1-data.toml` |

## Commands

```bash
python3 scripts/armor/build_corpus.py                 # build (uses cache; fetches if absent)
python3 scripts/armor/build_corpus.py --offline       # never use the network
python3 scripts/armor/build_corpus.py --check         # print the stats table, write nothing
python3 scripts/armor/build_corpus.py --verify-determinism
python3 scripts/armor/fetch_upstream.py               # refresh the cache
```

## Output

```
assets/armor/build/train.jsonl
assets/armor/build/dev.jsonl
assets/armor/build/test.jsonl
assets/armor/build/stats.json
assets/armor/build/families/<slug>.jsonl   # one file per family, hashed in the manifest
assets/armor/l1-data.toml                  # generated manifest
```

One line is exactly:

```json
{"text": "...", "label": "injection|safe", "lang": "en|de",
 "source": "<family>", "licence": "<spdx>", "redacted": true}
```

## Split policy

A **family** is a provenance family, not a URL: `deepset/prompt-injections:train`,
`redteam/de-ignore`, `hardneg/security-docs-en`, `translated:Lakera/gandalf_ignore_instructions`.
Each family is assigned to exactly one split; the build refuses a family that
appears in two. Published upstream splits are treated as families
(`…:train`, `…:test`, `…:validation`), so the held-out upstream rows are the
dataset's own held-out rows. Obfuscated families exist only for dev and test.

Within a family, dev/test generated pools are disjoint slices of the same
template structure as train. Shared structure remains a known weakness.

## Deduplication

Every example is normalised (NFKC, zero-width characters removed, whitespace
collapsed) and hashed with SHA-256 over the lowercase form. The first family
in build order keeps the text; duplicates are dropped and counted in
`stats.json`. Validation additionally asserts that no normalised text appears
in two splits.

## Determinism

No timestamps, randomness, `hash()` or set iteration decide output. Ordering is
`(source family, sha256(normalised text), text)`. `--verify-determinism` builds
twice in memory and compares the split contents; the CI-style check is to run
the build twice and `diff` the JSONLs.
