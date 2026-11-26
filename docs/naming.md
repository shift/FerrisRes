# Naming convention

Status: **normative, 2026-11-26.** One scheme for the family, the tiers, the
variants, the specialist models and the released artefacts. A name is a label;
the weights hash is the identity.

## The grammar

```
family      = "FerrisRes"                         ; display only
id          = "frs-" tier [ "-" variant ]         ; lowercase, ASCII, hyphens
tier        = "n" | "s" | "b" | "l"               ; nano, small, base, large
variant     = short token from the registry below
version     = major "." minor "." patch [ "-" prerelease ]
artefact    = id "-" version "." format [ "." quant ] "." ext
```

Examples:

```
FerrisRes Small                         display
frs-s                                   id
frs-s-0.1.0-beta                        id + version (a model)
frs-s-0.1.0-beta.safetensors            f32/bf16 weights
frs-s-0.1.0-beta.q4_k_m.gguf            a quantised artefact of the same model
frs-n-gqa-0.1.0-beta                    a variant: GQA instead of MQA
frs-armor-l1-0.2.0                      a specialist, not a chat tier
```

## What the parts mean

| part | rule |
|---|---|
| `frs-` | reserved for models released by this project. A third-party fine-tune must not use it; it names its own family and records `derived_from`. |
| tier letter | the **shape**, not the size on disk: `n` 40–80M, `s` ~100–150M, `b` ~500–700M, `l` ~1–3B. A shape change is a new tier or a variant, never a reuse of the letter. |
| variant | a deliberate deviation within a tier: a tokenizer, an attention pattern, a sparsity choice. One variant per id; do not stack (`-gqa-code-moe` is two experiments, not one model). |
| version | semver of **that id**, independent of the engine version and of other ids. |
| format | `safetensors` (f32/bf16 training output), `gguf` (quantised), `fral` (an engine runtime format), `json` (config/card). |
| quant | `f32`, `bf16`, `q8_0`, `q4_k_m`, `nf4`, `ternary`. **Precision is never in the id.** A quantised file is the same model; the file name says how it was stored. |

The id never contains: a quant, a context length, a date, a teacher, a licence, a
hash, or `latest`. Those are metadata or a pointer, and putting them in the name
guarantees a name that lies after the next release.

## Version policy

| bump | meaning | example |
|---|---|---|
| **major** | architecture or tokenizer incompatible; an old tokenizer id map does not apply | adding a new attention family, changing the vocab |
| **minor** | a new capability or a material quality/context change; the tokenizer still matches | 8k → 32k via YaRN with retraining, a new corpus mix |
| **patch** | a retrain or a bugfix with the same shape, data contract and tokenizer | a corrected loss, a data fix, a re-export |
| **prerelease** | not for production; `-beta`, `-rc1` | `0.1.0-beta` |

Every weight change bumps at least the patch: **the same name and version never
point at different bytes.** The manifest's `weights_sha256` is authoritative when
a name is ambiguous; a name is not a checksum.

`0.x` means the interface is not frozen. The first public release is `1.0.0` and
declares the compatibility promise.

## Variant registry

| suffix | meaning | not to be confused with |
|---|---|---|
| `-gqa` | grouped-query attention instead of MQA | a different KV precision |
| `-code` | code-aware tokenizer / code-heavy corpus at the same shape | a code *task* head |
| `-hybrid` | sliding-window + global attention, or SSM hybrid | a context extension alone |
| `-moe` | top-k mixture of experts at a tier that is dense by default | the top-1 experts already in every tier |
| `-draft` | a speculative-decoding draft model | a small chat tier |
| `-long` | a context-extended build, when it ships as a separate artefact | a minor version of the base |

A variant must be justified by a measured gate in `hardware-shapes.md`; the
registry is closed until then.

## Specialist models

Non-chat models have a role instead of a tier letter:

| id | role | shape |
|---|---|---|
| `frs-armor-l1` | injection/PII classifier | 2.2M bidirectional encoder |
| `frs-router` | intent → tool/action routing | Nano-class, tool vocabulary |
| `frs-rank` | candidate ranking | CLM-style projection head |
| `frs-embed` | embeddings | Granite-class encoder |
| `frs-decision` | System-1 control decisions | small head over a feature vector |
| `frs-draft` | speculative draft | 20–30M, tied to a target tier |

A specialist has its own version and its own eval; it is never described as "the
model" without the role.

## Display names

| id | display |
|---|---|
| `frs-n` | FerrisRes Nano |
| `frs-s` | FerrisRes Small |
| `frs-s-code` | FerrisRes Small Code |
| `frs-b` | FerrisRes Base |
| `frs-l-moe` | FerrisRes Large MoE |
| `frs-armor-l1` | FerrisRes Armor L1 |

Display names are capitalised, have no version, and are never used in a file
path or a config.

## Tree layout

```
frs-s/
  0.1.0-beta/
    model.safetensors          or model.bf16.safetensors
    config.json                id, version, context, capabilities
    tokenizer.json
    model-card.md
    NOTICE                     teachers, corpora, modifications
    LICENSE                    Apache-2.0
    SHA256SUMS
  latest -> 0.1.0-beta         a pointer, never an artefact
```

The directory name is the id and version; the files inside are generic. A file
that leaves the directory (`frs-s-0.1.0-beta.q4_k_m.gguf`) carries the id so it
can be identified alone.

## Relationship to other names

- **Engine:** `ferrisres` has its own semver (`0.1.0`). A model declares
  `engine_min` in `config.json`; the two version numbers are unrelated and must
  not be conflated.
- **Teacher:** named by upstream (`qwen3-1.7b`), recorded in NOTICE and the
  manifest, never in the model id.
- **Run:** a training run is `run-<YYYYMMDD>-<id>-<data-tag>-<seed>`; a run is
  not a release.
- **Agentic-loop workflows:** refer to a model by id and version, never by
  display name.

## Rules that prevent the usual failures

1. **Do not reuse a name.** A rebuild is a patch bump; the hash is the proof.
2. **Do not put `latest` in an artefact.** A pointer may be `latest`; a file may
   not.
3. **Do not put a quant in the id.** The same model quantised twice is still one
   model.
4. **Do not put a context length in the id.** Context is metadata; a materially
   different long-context build is a minor version or `-long`.
5. **Do not stack variants.** Two changes need two experiments before one model.
6. **Do not use `frs-` for a derivative.** Record `derived_from` instead.
7. **Do not name a specialist "Small".** Role, not size.

## One correction to earlier docs

`frs-s-bf16` appeared in `hardware-shapes.md` as a training shape. Under this
convention it is **`frs-s` trained and served in bf16**, with the artefact named
`frs-s-0.1.0-beta.bf16.safetensors`. Precision is a property of the file, not of
the model id.
