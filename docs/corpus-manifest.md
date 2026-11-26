# Corpus manifest

Status: **plan with target token counts, 2B total.** The per-source records —
licence, revision, access, filter, status — are in `source-index.md`; this file
is the shape, that one is the lookup; the parameter–subject audit is
`corpus-distribution.md`. Counts are targets, not
measurements; every entry needs a licence, a revision and a hash before it is
used. This is the complete list the data plan referred to.

## Language

| source | licence | target | notes |
|---|---|---:|---|
| FineWeb-Edu | ODC-By | 400M | English, quality-filtered |
| DCLM-baseline | CC-BY-4.0 | 100M | English |
| Dolma 3 | ODC-By | 100M | mixed |
| FineWeb-2 `de` | ODC-By | 150M | German |
| CulturaX `de` | gated terms | 50M | verify per card |
| German instruction/news subset | per card | 20M | |

## Code — including Rust and Nix

| source | licence | target | notes |
|---|---|---:|---|
| The Stack v2, permissive subset | per-file | 200M | filter to MIT/Apache/BSD; keep the licence tag per file |
| **The Stack v2, Rust subset** | per-file | **40M** | MIT/Apache only; **oversample 2x** because Rust is under-represented |
| **crates.io top crates** | per-crate | **20M** | licence-filtered (MIT/Apache/BSD); drop vendored/generated code |
| `rust-lang/rust`, `rust-lang/book`, `rust-lang/rustlings`, `rust-analyzer` | MIT/Apache | 10M | docs + code |
| **nixpkgs** | MIT | **20M** | modules, packages, NixOS configs |
| **nix-community flakes, `nix.dev` examples** | MIT / CC-BY | 5M | only permissive; `NixOS/nix` is LGPL-2.1 — **exclude** |
| our own repos (agentic-loop, ferrisres) | AGPL/commercial | 5M | our code; provenance recorded; the weights licence is separate |

Rust and Nix together are ~100M tokens, about **5% of the 2B**. That is above
their share of public code (Rust ~2–3%, Nix <1%), which is the point: the model
is for a Rust/Nix harness. Measure the cost on other code before raising it
further; oversampling a language is how a model forgets the others.

Code filtering: no secrets (scan for keys and tokens), no minified or generated
files, no vendored dependencies, deduplicate by normalised content, keep tests
and docs, keep the per-file licence tag in the provenance.

## Documentation — user-facing, as grounding

The docs are the product's vocabulary: commands, flags, error strings, ADP
claims and leases, workflow semantics. Train the model to **retrieve and cite**
them, not to memorise a release.

| source | licence | target | notes |
|---|---|---:|---|
| `agentic-loop` user docs: README, quickstart, CLI, workflows, Telegram, skills, ADP | ours | 4M | user-facing only |
| CLI `--help` text and error messages | ours | 1M | exact strings; the model suggests real commands |
| ADP docs: status, workspace layout, claims/leases, trust, auth | ours | 2M | the ADP vocabulary; no invented operations |
| doc-QA pairs: question → answer with the doc path/heading | CC0 | 10M | generated, verified against the docs |
| deployment/runbook docs | ours | 2M | no secrets; redact endpoints and keys |

**Excluded:** the deploy repo's endpoint/credential docs, incident logs, licence
keys, security configuration, and anything containing a secret. A doc that
cannot be published is not training data.

The doc-QA set is what teaches the behaviour: the answer carries a citation, and
a question whose answer is not in the retrieved docs is answered "I don't know"
rather than guessed. Version drift is the failure mode — a memorised flag from
last release — so retrieval, not recall, is the target.

## SFT

| source | licence | target | notes |
|---|---|---:|---|
| OASST1 | Apache-2.0 | 40M | chat |
| Dolly | CC-BY-SA | 0 | **excluded**: share-alike reaches an Apache-2.0 weights release |
| SmolTalk / Tulu | per card | 50M | verify each |
| Multi-SFT (DE) | permissive | 30M | German |
| German authored templates | CC0 | 20M | |

## Tool, agent and soul

| source | licence | target | notes |
|---|---|---:|---|
| Glaive FC, xLAM, BFCL train | per card | 60M | real tool calls |
| agentic-loop traces (redacted, reviewed) | internal | 40M | derived, never raw; self-generated rows in `self-generated-data.md` |
| **soul corpus** — 30 authored soul templates, 500+ synthetic souls across the eight dimensions, 5–10 tasks per soul, one violation pair per adhering answer | CC0 | **20M** | no real user soul; unseen-soul eval matrix held out |
| **child-safe corpus** — authored, parent-reviewed, plus a red-team set | CC0 | **5M** | no real minor data |
| math/reason (OpenR1, Numina subset) | Apache-2.0 | 60M | |
| long context (public-domain docs, repo READMEs) | per doc | 50M | 8k windows |

## The soul corpus, concretely

For souls to work the model needs **more than examples of one soul**:

- 30 authored souls: adult assistant, child, operator, formal, terse, German,
  bilingual, "ask before acting", "never discuss X".
- 500+ synthetic souls sampled across the eight dimensions (register, length,
  language, address, forbidden topics, escalation, tool policy, uncertainty).
- 5–10 tasks per soul, each answered in the soul's register.
- one violating answer per adhering answer (the preference/negative signal).
- an **unseen-soul matrix** of 10 souls × 20 tasks, held out for the gate.

## Provenance per line

`source`, `licence`, `lang`, `redacted`, `revision` (dataset commit or repo
hash), and for code `spdx` per file. A line without a licence does not train.

## What is unresolved

- Per-crate and per-file licences on crates.io and The Stack v2: the filter is
  mechanical but the long tail needs review.
- CulturaX / Dolly share-alike: record and decide, do not assume.
- Rust/Nix oversampling factor: start at 2x, measure the regression on other
  code.
- Secrets in public code: a scanner is mandatory, not optional.
