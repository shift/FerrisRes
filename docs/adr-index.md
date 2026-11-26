# Decision index

Status: **index, 2026-11-26.** The FerrisRes decisions, numbered for reference.
The harness's own 27 decisions live in `agentic-loop`; these are the model side.
A decision is a link, not a retelling: the owning document is the text.

| id | decision | status | owner |
|---|---|---|---|
| FR-001 | Block AttnRes with a looped middle; top-1 MoE, MQA, 3-bit KV, ternary experts, NF4 head | accepted | `model-tiers.md` |
| FR-002 | Inter-block attention is causal; the suffix-sum is part of the backward | accepted | `block-attnres-causal.md` |
| FR-003 | The f32 reference trainer is the gradient oracle for every other path | accepted | `train-and-export.md` |
| FR-004 | Block + LoRA gradients are hard-gated until the audit passes | accepted | `status-2026-11.md` |
| FR-005 | First student at 8k; 32k only via a YaRN phase; 64k only on measured need | accepted | `model-tiers.md` |
| FR-006 | Teacher: Qwen3-1.7B for Base, 0.6B for the smoke, Gemma 4 as fallback | accepted | `data-plan.md` |
| FR-007 | Weights Apache-2.0; engine AGPL-3.0-or-later only, no commercial edition | accepted | `model-licensing.md` |
| FR-008 | Data limited to permissive/CC0/CC-BY; NC and share-alike excluded | accepted | `source-index.md` |
| FR-009 | Generated data only through a per-generator allowlist with per-row provenance | accepted | `model-licensing.md` |
| FR-010 | 32k code-aware tokenizer for Nano/Small; 151k for Base/Large | accepted | `tokenizer.md` |
| FR-011 | Cross-vocabulary distillation is text-aligned (byte boundaries), not a logit copy | accepted | `text-aligned-kd.md` |
| FR-012 | Model ids `frs-<tier>[-variant]` with independent semver; quant never in the id | accepted | `naming.md` |
| FR-013 | Identity: purpose is trained; version/context/capabilities are retrieved | accepted | `model-identity.md` |
| FR-014 | Instant/balanced/deliberate are one checkpoint plus routing and budget | accepted | `response-modes.md` |
| FR-015 | Multi-GPU: process per GPU, local SGD over `/dev/shm`; no NCCL or tensor parallel | accepted | `tasks/multi-gpu.md` |
| FR-016 | The GPU training loop uses the existing autodiff graph, parity-checked against f32 | proposed | `status-2026-11.md` |
| FR-017 | MoLT is clean-room (no licence); evaluate the U-MoLT split in the f32 reference first | proposed | `molt.md` |
| FR-018 | An embedding head is a frozen-backbone 768→256 adapter; LoRA only if needed | proposed | `embedding-head.md` |
| FR-019 | Corpus builds are deterministic; u16 shards; held-out split by source hash | accepted | `corpus-builder.md` |
| FR-020 | Eval gates are preregistered, counts not percentages, failures retained | accepted | `eval-plan.md` |
| FR-021 | Reproducibility: pin hashes; CPU bitwise; GPU tolerance stated, never claimed | accepted | `reproducibility.md` |
| FR-022 | Releases are immutable, card + NOTICE mandatory, four sign-offs | accepted | `release-runbook.md` |
| FR-023 | The runtime is the safety boundary; Armor is a filter, not a proof | accepted | `model-safety-card.md` |
| FR-024 | A second minimal harness is required to test cross-harness generality | proposed | `tool-generalization.md` |
| FR-025 | The 4.89 blocked-config gradient must be fixed before any distillation run | accepted | `status-2026-11.md` |
| FR-026 | A `.deb` is built only on ubuntu-22.04 (glibc ≤ 2.35) | accepted | `release-runbook.md` |
| FR-028 | Security is trained as theory and procedure; a finding is proven by a tool, and no exploit corpus is trained | accepted | `security-training.md` |
| FR-029 | Design patterns are trained as vocabulary and judgement, with an anti-ceremony gate; never as rules to apply everywhere | accepted | `design-patterns.md` |
| FR-030 | Train durable, checkable theory; retrieve changing facts; execute effects. Every theory card names a check | accepted | `theory-corpus.md` |
| FR-031 | The child tutor teaches: hint ladder before answer, no ghostwritten work, parent-visible summary, curriculum facts supplied by the parent | accepted | `learning-theory.md` |
| FR-032 | Train the adaptation to the principal; retrieve the principal block; never let text or inference grant authority | accepted | `principal-awareness.md` |
| FR-033 | Self-generated harness data trains only when a verifier passes; generator and verifier ids are recorded, and a source that does not improve the eval is cut | accepted | `self-generated-data.md` |
| FR-034 | Legal work is information and drafting support: the law is indexed and cited at a date, never memorised; advice and reserved activities need a lawyer | accepted | `legal-information.md` |
| FR-027 | Writers do not run heavy builds; one build slot at a time | process | repository `AGENTS.md` |

## How to use this

- A proposal that changes a `FR` entry links to it and either supersedes it or
  adds a new one; editing the entry in place hides the change.
- `proposed` means the design is written but no evidence exists yet. It is not a
  commitment to ship.
- A decision without an owning document is not a decision; it is a conversation.
