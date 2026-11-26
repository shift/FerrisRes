# Model card template

Status: **template, 2026-11-26.** Every weights release ships one of these. A
field marked **REQUIRED** blocks the release if it is empty or says `TBD`.

## Front matter

```yaml
id: frs-s
version: 0.1.0-beta
released: YYYY-MM-DD
licence: Apache-2.0
weights_sha256: REQUIRED
tokenizer_sha256: REQUIRED
engine_min: REQUIRED
card_version: 1
```

## Identity

| field | value |
|---|---|
| display name | FerrisRes Small |
| parameters | REQUIRED (total, active) |
| architecture | REQUIRED (looped depth, attention, MoE, KV) |
| context | REQUIRED (trained, extended, maximum) |
| vocab | REQUIRED (size, tokenizer id) |
| artefacts | REQUIRED (formats and quants shipped) |

## Summary

Three sentences: what it is, what it is for, what it is not. No adjectives that
the eval does not support.

## Intended use

- REQUIRED: the primary deployment (agentic-loop assistant, tools, retrieval).
- REQUIRED: languages (English, German, code: Rust/Nix).
- REQUIRED: modes (instant, balanced, deliberate) and what each may do.

## Out of scope

REQUIRED. At minimum: medical/legal advice, autonomous high-risk actions,
unbounded tool use, any claim of consciousness, any deployment without Armor for
untrusted input.

## Training

| item | value |
|---|---|
| objective | REQUIRED (pretrain, SFT, KD, preference) |
| data | REQUIRED (corpus name + manifest hash + licences summary) |
| tokens | REQUIRED |
| compute | REQUIRED (device, hours, MFU if known) |
| teachers | REQUIRED (id, licence, method) or "none" |
| tokenizer | REQUIRED (id, version, hash) |
| code | REQUIRED (engine commit) |

## Evaluation

| suite | result | date | command |
|---|---|---|---|
| the model eval suite (`eval-plan.md`) | REQUIRED | | |
| harness integration (§8) | REQUIRED | | |
| long-horizon (§11) | REQUIRED | | |
| response modes (§15) | REQUIRED | | |
| tokenizer fertility | REQUIRED | | |

State the seed count and variance. A single run is a data point, not a result.
List known failures explicitly; a card with no failures is not finished.

## Safety

- Armor: the scanner version and what it covers, and what it does not.
- Refusal behaviour: what the model refuses, and how it reports a refusal.
- Injection: the tool-result and description gates from `harness-integration.md`.
- Child principal: what it may see, and that it is enforced by the runtime, not
  the weights.

## Limitations

REQUIRED, at least three, each with the evidence that shows it.

## Hardware

Tested and untested from `hardware-matrix.md`, with the measured tok/s where it
exists. Do not copy a calculated number into this section.

## Licence and attribution

Apache-2.0 for the weights; the NOTICE names teachers, corpora and
modifications (`model-licensing.md`). The engine's licence is separate.

## Contact and changelog

Where to report a defect, and the previous card if this is a revision.

## Rules

1. A card is part of the artefact, not a blog post: it ships in the directory and
   its hash is in `SHA256SUMS`.
2. Numbers in the card come from a recorded command; a calculated number is
   labelled `calculated`.
3. A card that claims "verified" without a command fails the release audit.
4. When a model is deprecated, the card stays; a new field says so.
