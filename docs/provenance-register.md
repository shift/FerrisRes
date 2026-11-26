# Third-party provenance register

Status: **register, 2026-11-26.** Every third-party paper, code base, model and
dataset used by FerrisRes appears here with its licence, how it is used, and
whether anything was copied. No entry means no use.

## Rules

1. **No licence, no copy.** A repository without a licence is all-rights-reserved
   by default; ideas may be reimplemented, code may not be copied.
2. **Ideas vs code vs weights vs data** are four different obligations; record
   which one applies.
3. **A copied file carries its origin, revision and licence in a comment** beside
   it, and the obligation is stated (AGPL source obligations included).
4. **NOTICE** for the weights is generated from this register, not written from
   memory.
5. A dataset entry is not a licence opinion: `verify` means a human read the card.

## Code and papers

| item | licence | use | copied? | obligation | evidence |
|---|---|---|---|---|---|
| **MoLT** (`hustvl/MoLT`, arXiv `2611.34923`) | **none** (GitHub `license: null`) | architectural ideas: MoLA, Loop KV, U-MoLT split, MoL-Hybrid | **no** — clean-room from the paper | cite; do not vendor; no code or weights | `docs/molt.md` |
| **BlaGPT** (erogol) | verify | reference host for the MoLT code | no | — | paper repo README |
| **OLMo / OLMo3** (AllenAI) | Apache-2.0 | baseline numbers only | no | — | paper |
| **Qwen3** (Alibaba) | Apache-2.0 | teacher weights (0.6B, 1.7B) | weights loaded, not redistributed as-is | NOTICE; distillation provenance | `data-plan.md` |
| **Gemma 4** (Google) | Apache-2.0 (verify per model card) | fallback teacher | weights loaded | NOTICE if used | `data-plan.md` |
| **Granite embeddings** (IBM) | verify | current embedding baseline; optional score distillation | no weights copied | verify before score distillation | `embedding-head.md` |
| **MoDA** (`hustvl/MoDA`, arXiv `2603.15619`) | cite only | prior work referenced by MoLT | no | cite | paper |
| **tree-sitter grammars** | per grammar, MIT/Apache | parsing | wasm files committed | attribution in `docs/ast-grammars.md` | verified |
| **nix-community/tree-sitter-nix** | MIT | Nix grammar | wasm file committed | upstream attribution | verified |
| **SurrealDB** | BSL 1.1 | harness storage, not the model | dependency | additional-use grant; no DB service | `agentic-loop` licensing docs |

## Datasets

| item | licence | use | status |
|---|---|---|---|
| FineWeb-Edu | ODC-By | English pretraining | planned |
| FineWeb-2 `deu` | ODC-By | German | planned |
| DCLM-baseline | CC-BY-4.0 (verify) | English | planned |
| Dolma 3 | ODC-By (verify per subset) | mixed | planned |
| CulturaX `de` | gated terms | German | **excluded** unless weight redistribution is permitted |
| The Stack v2 | per-file (dataset ODC-By) | code | planned; allowlist only |
| crates.io | per-crate | Rust | planned; allowlist only |
| nixpkgs | MIT | Nix | planned |
| NixOS/nix | **LGPL-2.1** | — | **excluded** |
| OASST1 | Apache-2.0 | SFT | planned |
| SmolTalk / Tulu 3 | per card / per subset | SFT | planned; verify subsets |
| Dolly | CC-BY-SA-3.0 | SFT | **excluded** (share-alike) |
| xLAM | NC reported | tool calls | **excluded** |
| BFCL / Glaive FC | Apache-2.0 (verify) | tool calls | planned |
| OpenR1 / NuminaMath | Apache-2.0 | reasoning | planned |

## Clean-room notes

- **MoLT:** the repository has no licence and no released variants beyond a
  minimal U-MoLT. Any implementation in this project is written from the paper's
  description, with its own tests; no file, kernel or weight is copied. The
  Triton kernel is NVIDIA-only and is not a source for the wgpu path.
- **Teacher weights:** loading a teacher for distillation is not redistribution.
  The NOTICE records the teacher and the fact that the student is distilled.
- **Scores vs weights:** distilling an embedding model's *scores* is recorded
  separately from copying its weights; the licence is checked first.

## Open questions

| item | question | owner |
|---|---|---|
| Granite | exact licence and redistribution terms | legal review |
| Gemma 4 | confirm Apache-2.0 for the exact checkpoint used | legal review |
| DCLM / Dolma | confirm the subset licences | data review |
| BFCL / Glaive | confirm the dataset card | data review |
| Tulu 3 | which subsets are permissive | data review |

## What this register is not

It is not legal advice, and it is not a substitute for reading a licence before a
release. It is the list that makes that reading possible, and the audit trail
that says it happened.
