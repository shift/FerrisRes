# frs-tokenizer-32k / 0.1.0

BPE with byte-level fallback, no normalisation, 32,768 tokens.

| file | purpose |
|---|---|
| `tokenizer.json` | the `tokenizers` artefact |
| `special_tokens.json` | the fixed special block and its ids |
| `sample-manifest.jsonl` | relative path, SHA-256, size and licence of every training file |
| `SHA256SUMS` | hashes of the three files above |

`tokenizer.json` SHA-256:
`98aa17a1e7d02e033647dd171d26a73872a2ae49b341bb39d1ce5387963feaf8`

## Sample

The training sample is 5,684 UTF-8 files (~24 MB) copied from three
licence-clean repositories. Only the manifest is committed; the corpus itself
is not.

| prefix | source | commit | licence |
|---|---|---|---|
| `book/` | `rust-lang/book` | `c976903459e37407cfa6964530ea3469957b3549` | MIT OR Apache-2.0 |
| `rbe/` | `rust-lang/rust-by-example` | shallow clone, 2026-11-26 | MIT OR Apache-2.0 |
| `nixpkgs/` | `NixOS/nixpkgs` (`doc/`, `nixos/modules/`, `lib/`) | shallow clone, 2026-11-26 | MIT |

The sample is small on purpose: it proves the build and the determinism gate,
not the production corpus mix. It reached exactly 32,768 tokens at
`min_frequency = 2`. A larger, weighted sample (see `docs/tokenizer.md`) will
produce a different artefact and therefore a new version.

## Reproduce

```bash
ferrisres tokenizer train \
  --corpus <sample-dir> \
  --vocab 32768 \
  --out artifacts/frs-tokenizer-32k/0.1.0 \
  --licence-map licences.json
```

`licences.json` was:

```json
{"book/":"MIT OR Apache-2.0","rbe/":"MIT OR Apache-2.0","nixpkgs/":"MIT"}
```

Files are sorted by relative path before training. Two runs on the same inputs
produced the hash above.
