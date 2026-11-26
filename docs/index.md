# Documentation index

Start here. The current state is `status-2026-11.md`; the decisions are
`adr-index.md`; the vocabulary is `glossary.md`.

## If you only read three

1. **`status-2026-11.md`** — what is verified, what is broken, what is next.
2. **`model-tiers.md`** — the shapes and why they are sized that way.
3. **`hardware-matrix.md`** — what actually runs, and what is only designed.

## By role

**Operator / user**
`quickstart.md` → `deployment.md` → `troubleshooting.md` → `hardware-matrix.md`.

**Model developer**
`model-tiers.md` → `hardware-shapes.md` → `molt.md` → `embedding-head.md` →
`tokenizer.md` → `train-and-export.md` → `training-runbook.md`.

**Data and licensing**
`source-index.md` → `corpus-manifest.md` → `corpus-builder.md` →
`model-licensing.md` → `provenance-register.md` → `tokenizer.md` →
`text-aligned-kd.md`.

**Evaluator / reviewer**
`eval-plan.md` → `eval-runner.md` → `model-card-template.md` →
`harness-integration.md` → `tool-generalization.md`.

**Release / auditor**
`release-runbook.md` → `model-licensing.md` → `provenance-register.md` →
`reproducibility.md` → `model-safety-card.md`.

**Researcher**
`molt.md` → `hardware-shapes.md` → `model-tiers.md` → `text-aligned-kd.md` →
`tool-generalization.md`.

## By lifecycle

| stage | documents |
|---|---|
| decisions | `adr-index.md`, `status-2026-11.md` |
| shapes | `model-tiers.md`, `hardware-shapes.md`, `hardware-matrix.md`, `molt.md` |
| identity and modes | `model-identity.md`, `persona-and-principals.md`, `response-modes.md` |
| data | `source-index.md`, `corpus-manifest.md`, `corpus-builder.md`, `dataset-formats.md`, `data-plan.md` |
| tokenizer | `tokenizer.md` |
| training | `train-and-export.md`, `training-runbook.md`, `distillation.md`, `text-aligned-kd.md`, `reproducibility.md` |
| runtime | `runtime-caches.md`, `long-running.md`, `subagents.md`, `quantization.md` |
| evaluation | `eval-plan.md`, `eval-runner.md`, `token-economics.md` |
| release | `release-runbook.md`, `model-card-template.md`, `naming.md`, `model-licensing.md`, `provenance-register.md` |
| safety | `model-safety-card.md`, `security.md` |
| operations | `deployment.md`, `troubleshooting.md`, `kaggle-runbook.md`, `hardware-matrix.md` |
| harness fit | `harness-integration.md`, `tool-generalization.md`, `model-catalogue.md` |
| reference | `glossary.md`, `cli-surface.md`, `api-reference.md` |

## Every document, one line

| doc | purpose |
|---|---|
| `Brief.md` | the project brief |
| `TurboQuant_Engine_Specification.md` | the quantisation engine spec |
| `adr-index.md` | FR-001…FR-027: decision, status, owner |
| `api-reference.md` | the Rust module/API reference (older than the CLI surface) |
| `architecture.md` | the engine's structure |
| `cli-surface.md` | commands and flags by branch; release vs planned |
| `corpus-builder.md` | sources → licensed, deduped, tokenised shards |
| `corpus-manifest.md` | token targets per source family |
| `data-plan.md` | data mix, teacher, distillation plan |
| `data-retention.md` | data classes, deletion, hosted-provider rules |
| `data-source-navigation.md` | where each data type lives, identifiers, licences, access |
| `dataset-formats.md` | sample shapes and chat templates |
| `deployment.md` | how to deploy the harness and model |
| `design-patterns.md` | pattern vocabulary, Rust idioms, anti-ceremony gate |
| `distillation.md` | the teacher→student path |
| `embedding-head.md` | a 768→256 head for an embedding mode |
| `eval-plan.md` | the suites, metrics and thresholds |
| `eval-runner.md` | the command, record and CI hook |
| `glossary.md` | the vocabulary |
| `guarantees.md` | what the runtime enforces vs what is learned |
| `hardware-matrix.md` | verified vs designed devices |
| `hardware-shapes.md` | shape levers per device bottleneck |
| `harness-integration.md` | tool-result resistance, evidence, protocol |
| `implementation-handoff.md` | delegate-ready task index and stop rules |
| `kaggle-readiness.md` | GPU-only smoke vs real-run blockers |
| `kaggle-runbook.md` | notebooks, quota, datasets |
| `learning-theory.md` | child tutor: learning science, hint ladder, integrity |
| `long-running.md` | compaction, handoff, recovery |
| `loop-policy.md` | how the development loop runs |
| `model-card-template.md` | the required release card |
| `model-catalogue.md` | model lookup and routing, not memorisation |
| `model-identity.md` | artefact identity vs trained purpose |
| `model-licensing.md` | weights/engine/data licences and rules |
| `model-safety-card.md` | what the model is not trusted with |
| `model-tiers.md` | Nano/Small/Base/Large |
| `molt.md` | Mixture-of-Loops assessment and clean-room rule |
| `naming.md` | ids, versions, artefacts, specialists |
| `performance-this-host.md` | decode, prefill and training expectations |
| `persona-and-principals.md` | souls, principals, child rules |
| `prefill-performance.md` | prefill levers and the roofline |
| `principal-awareness.md` | who is in the conversation, and what may never be inferred |
| `provenance-register.md` | third-party licences and clean-room notes |
| `quantization.md` | ternary, NF4, KV precision |
| `quickstart.md` | the shortest path to a running instance |
| `reasoning-and-retention.md` | recall tools, notebook, retention policy |
| `release-runbook.md` | preconditions, artefacts, sign-off |
| `reproducibility.md` | pinned hashes, determinism, run records |
| `research/` | research notes |
| `response-modes.md` | instant/balanced/deliberate |
| `runtime-caches.md` | what to cache and how to invalidate |
| `security-training.md` | secure development and vuln finding as theory + tools |
| `security.md` | the harness threat model |
| `self-generated-data.md` | harness-generated verified trajectories |
| `soul-training.md` | the soul corpus and behaviour |
| `source-index.md` | per-source licence, revision, status |
| `subagents.md` | delegation as a granted capability |
| `tasks/` | one task spec per open piece of work |
| `text-aligned-kd.md` | 151k teacher → 32k student |
| `theory-corpus.md` | which theories are trained, with checks |
| `token-economics.md` | AST tools, tokens per verified change |
| `tokenizer.md` | the 32k code-aware BPE |
| `tool-generalization.md` | tool discovery and cross-harness claims |
| `training-runbook.md` | corpus → run → resume → export → eval |
| `troubleshooting.md` | symptom → cause → fix |

## What is not here

- The harness's own documentation (agentic-loop) — separate repository.
- The API reference for commands added after 0.2.2 — `cli-surface.md` covers the
  gap until `api-reference.md` is regenerated.
- A generated hardware matrix: `scripts/hardware_matrix.py` turns device and
  benchmark records into the evidence table; the prose stays in
  `hardware-matrix.md`.
