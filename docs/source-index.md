# Source index

Status: **index with per-source records, 2026-11-26.** The pipeline that turns
these rows into reproducible shards is `corpus-builder.md`. It is the lookup the corpus
manifest refers to: for each **data type**, which sources may be used, under
which licence, fetched how, filtered how, and whether the row is trained or held
out. A row is not usable until every field is filled.

## The rule

One row per source. A source without a licence, a revision and a hash does not
train. A source used for evaluation is never used for training. A share-alike or
non-commercial source is excluded from the commercial edition unless a recorded
decision says otherwise, and that decision is part of the release.

## Status legend

| status | meaning |
|---|---|
| `planned` | chosen, not fetched |
| `licensed` | licence read and recorded |
| `fetched` | bytes acquired, revision pinned |
| `filtered` | dedupe, secret scan, per-file licence scan done |
| `hashed` | content hash recorded; reproducible |
| `trained` | used in a run, with the run id |
| `held-out` | reserved for evaluation; never trained |

## Per-source record

```
id:            stable slug
data type:     language | code | sft | tools | soul | docs | reasoning | routing | safety | long-context
edition:       commercial | agpl | both | excluded
licence:       SPDX id, or "per-file" / "verify"
revision:      dataset commit, repo hash, or snapshot date#hash
access:        hf:org/dataset | git:url | api:url | ours
bytes/tokens:  measured where known, target otherwise
filter:        the exact filter, named
status:        legend above
notes:         secrets, gating, share-alike, NC, known defects
```

## 1. Language — English

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `en-fineweb-edu` | FineWeb-Edu | ODC-By | pin per parquet shard | `hf:HuggingFaceFW/fineweb-edu` | 400M | score >= 3, dedupe, lang=en | planned |
| `en-dclm` | DCLM-baseline | CC-BY-4.0 (verify card) | pin | `hf:mlfoundations/dclm-baseline-1.0` | 100M | dedupe, quality filter | planned |
| `en-dolma3` | Dolma 3 | ODC-By | pin | `hf:allenai/dolma3` | 100M | mixture documented per subset | planned |
| `en-pd-long` | public-domain books/essays | per work | pin each work | mixed | 50M | long-context windows, 8k | planned |

## 2. Language — German

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `de-fineweb2` | FineWeb-2 `deu_Latn` | ODC-By | pin | `hf:HuggingFaceFW/fineweb-2` | 150M | lang=de, dedupe | planned |
| `de-culturax` | CulturaX `de` | gated; verify card | pin | `hf:uonlp/CulturaX` | 50M | gated terms accepted, dedupe | planned |
| `de-news` | German news/instruction subset | per card | pin | hf | 20M | licence per subset | planned |
| `de-authored` | authored German templates | CC0 | ours | ours | 20M | reviewed | planned |

## 3. Code — Rust (first-class)

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `code-stack2-rust` | The Stack v2, Rust subset | per-file (dataset ODC-By) | pin | `hf:bigcode/the-stack-v2` | 40M | MIT/Apache/BSD only; drop generated/vendored; keep SPDX tag; **oversample 2x** | planned |
| `code-crates-io` | crates.io top crates | per-crate | latest index snapshot | `api:crates.io` + tarballs | 20M | licence allowlist; drop vendored; tests and docs kept | planned |
| `code-rust-lang` | `rust-lang/rust`, `book`, `rustlings`, `rust-analyzer` | MIT/Apache-2.0 | pin each repo | `git:github.com/rust-lang/*` | 10M | dedupe against Stack v2 | planned |
| `code-agentic-loop` | this repository | AGPL/commercial | pin | ours | 3M | provenance recorded; weights licence separate | planned |
| `code-ferrisres` | the engine repository | AGPL/commercial | pin | ours | 2M | same | planned |

## 4. Code — Nix and general

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `code-nixpkgs` | nixpkgs | MIT | pin | `git:github.com/NixOS/nixpkgs` | 20M | packages, modules, NixOS configs | planned |
| `code-nix-flakes` | nix-community flakes, `nix.dev` examples | MIT / CC-BY | pin | git | 5M | **exclude `NixOS/nix` (LGPL-2.1)** | planned |
| `code-stack2-general` | The Stack v2 permissive subset | per-file | pin | `hf:bigcode/the-stack-v2` | 200M | same per-file rule | planned |

Note: `agentic-loop-ast` has **no Nix grammar** (33 grammars, none for Nix), so
Nix training data teaches code, not the AST edit path.

## 5. SFT — conversation

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `sft-oasst1` | OASST1 | Apache-2.0 | pin | `hf:OpenAssistant/oasst1` | 40M | dedupe, tree flatten with provenance | planned |
| `sft-dolly` | Dolly 15k | CC-BY-SA-3.0 | pin | `hf:databricks/databricks-dolly-15k` | 0 | **excluded for the open-weight release** (share-alike reaches the weights) | excluded |
| `sft-smoltalk` | SmolTalk | Apache-2.0 (verify card) | pin | `hf:HuggingFaceTB/smoltalk` | 50M | dedupe | planned |
| `sft-tulu3` | Tulu 3 SFT mixture | mixture; verify per subset | pin | `hf:allenai/tulu-3-sft-mixture` | 30M | licence per subset; some subsets may be excluded | planned |
| `sft-de` | German multi-SFT | per card | pin | hf | 30M | lang=de | planned |

## 6. Tools, agent and harness

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `tool-bfcl` | BFCL train split | Apache-2.0 | pin | `hf:gorilla-llm/...` | 30M | dedupe against eval split | planned |
| `tool-xlam` | xLAM function calling | **NC reported** | pin | hf | 0 | **excluded**: an open, commercially usable model cannot be built on NC data | excluded |
| `tool-glaive` | Glaive function calling v2 | Apache-2.0 | pin | `hf:glaiveai/...` | 20M | strip synthetic dialogues that invent tools | planned |
| `tool-registry` | MCP catalogue + OpenAPI specs | per server/repo | pin | mixed | 5M | ids and schemas only; no secrets in examples | planned |
| `tool-traces` | reviewed run records | internal | redacted | ours | 40M | derived, never raw; consent recorded | planned |
| `harness-workflows` | `dataset workflows` (63 shipped, generated) | ours | pin | ours | 2M | generate -> validate -> keep | planned |
| `harness-evidence` | revision-bound evidence records | ours | pin | ours | 5M | counts and verdict, raw by reference | planned |

## 7. Soul and persona

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `soul-authored` | 30 authored souls across the eight dimensions | CC0 | ours | ours | 5M | reviewed; no real user soul | planned |
| `soul-synthetic` | 500+ synthetic souls x 5-10 tasks | CC0 | generated | ours | 13M | sampled across register, length, language, address, forbidden topics, escalation, tool policy, uncertainty | planned |
| `soul-violations` | one violating answer per adhering answer | CC0 | generated | ours | 2M | preference signal; not shown as a good answer | planned |
| `child-safe` | authored, parent-reviewed tasks | CC0 | ours | ours | 5M | **no real minor data**; red-team set held out | planned |

## 8. Documentation grounding

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `docs-user` | agentic-loop README, quickstart, CLI, workflows, Telegram, skills | ours | pin | ours | 4M | user-facing only | planned |
| `docs-cli-help` | `--help` text and error strings | ours | build commit | ours | 1M | generated per release | planned |
| `docs-adp` | ADP status, workspace, claims/leases, trust, auth | ours | pin | ours | 2M | no endpoints or credentials | planned |
| `docs-qa` | question -> answer with path and heading | CC0 | generated | ours | 10M | verified against the docs; version-stamped | planned |

Excluded by rule: deploy endpoint/credential docs, incident logs, licence keys,
security configuration.

## 9. Reasoning, long horizon, recovery

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `reason-openr1` | OpenR1-Math-220k | Apache-2.0 | pin | `hf:open-r1/OpenR1-Math-220k` | 30M | dedupe | planned |
| `reason-numina` | NuminaMath-CoT | Apache-2.0 | pin | `hf:AI-MO/NuminaMath-CoT` | 30M | dedupe | planned |
| `long-synthetic` | 50-200 step tasks with scripted failures | CC0 | generated | ours | 10M | reviewed | planned |
| `long-compaction` | history -> summary -> next action | CC0 | generated | ours | 5M | fact-retention reviewed | planned |
| `long-handoff` | history -> resumable state | CC0 | generated | ours | 3M | resumed by a fresh agent | planned |
| `long-recovery` | failure -> retry / alternative / escalate | CC0 | from incidents | ours | 2M | redacted | planned |

## 10. Routing and model catalogue

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `route-catalogue` | provider model list snapshot | factual metadata | `date#hash` | api | 1M | ids and typed fields; **no prices as facts** | planned |
| `route-measured` | our measured profiles | ours | revision-bound | ours | 1M | score + samples + confidence | planned |
| `route-pairs` | task -> tier with the deciding constraint | CC0 | authored | ours | 1M | reviewed | planned |
| `route-heldout` | models released after the snapshot | — | — | — | — | **held-out only** | held-out |

## 11. Safety and injection

| id | source | licence | revision | access | target | filter | status |
|---|---|---|---|---|---|---|---|
| `safety-armor` | armor corpus (14,787 examples) | permissive upstream + ours | manifest | ours | 5M | 33.2% German, 38.6% hard negatives | planned |
| `safety-injection` | imperative text inside tool results and descriptions | CC0 | authored + real traces | ours | 2M | redacted; never obeyed | planned |
| `safety-jailbreak` | 40 golden + 30 benign-hard | ours | pin | ours | 1M | held-out split separate | planned |
| `safety-pii` | deterministic PII patterns | ours | pin | ours | 1M | synthetic only | planned |

## 11b. Secure development and vulnerability finding

| id | source | licence | target | filter | status |
|---|---|---|---:|---|---|
| `sec-theory` | authored principle notes: trust boundaries, least privilege, fail-closed, CWE classes | CC0 | 3M | original summaries; cite, do not copy CC-BY-SA curricula | planned |
| `sec-snippets` | synthetic vulnerable snippets + a checker + a fixed version | CC0 | 2M | one defect each; the check must fail before and pass after | planned |
| `sec-fixes` | OSS fixes from permissive repos, with commit and CVE id | per-repo MIT/Apache | 2M | no exploit code; patch only | planned |
| `sec-tools` | code → scanner output → triage → reproduction | ours | 1M | scanner version recorded; no secrets | planned |
| `sec-incidents` | redacted incidents: failure → cause → fix → verification | internal | 1M | derived, never raw logs | planned |

Excluded: exploit payloads, malware, live targets, advisory text as memory,
share-alike or NC security curricula.

## 11c. Design patterns and architecture

| id | source | licence | target | filter | status |
|---|---|---|---:|---|---|
| `dp-cards` | authored pattern cards: intent, forces, Rust shape, when not | CC0 | 2M | original summaries; never the GoF text | planned |
| `dp-refactors` | authored before/after pairs with unchanged tests | CC0 | 2M | the test must pass before and after | planned |
| `dp-recognition` | permissive OSS code with pattern/anti-pattern labels | per-file MIT/Apache | 3M | cite file and line; no generated code | planned |
| `dp-adr` | architecture decisions with forces and rejected options | ours, CC0 | 1M | no customer names or secrets | planned |
| `dp-nix` | module, overlay and flake patterns from nixpkgs / nix.dev | MIT / verify CC | 1M | no NixOS/nix (LGPL); no verbatim share-alike | planned |

## 12. Held-out evaluation sets

These are never trained. If any of them appears in a training run, that run is
void.

| id | set | size |
|---|---|---|
| `eval-unseen-souls` | 10 souls x 20 tasks | 200 |
| `eval-unseen-tools` | 20% of the tool registry | — |
| `eval-post-snapshot` | models released after the catalogue snapshot | — |
| `eval-jailbreak-golden` | 40 golden + 30 benign-hard | 70 |
| `eval-armor-golden` | 50 golden + 10 obfuscations | 60 |
| `eval-arithmetic` | arithmetic suite | 30 |
| `eval-interface` | interface suite | 60 |
| `eval-gates` | the 15-gate manifest | 15 |

## 13. Generated data (hosted providers and local generators)

Free access is not a training licence. A generated row trains only when its
generator is on the allowlist and the row carries provenance
(`model-licensing.md`).

| id | generator | terms | target | filter | status |
|---|---|---|---|---|---|
| `gen-qwen3-local` | Qwen3-0.6B/1.7B, open weights | Apache-2.0 | 40M | authored/synthetic prompts; verified by a check | planned |
| `gen-gemma4-local` | Gemma 4 E2B/E4B, open weights | Apache-2.0 | 10M | same | planned |
| `gen-teacher-distill` | teacher-forcing logits from the open-weight teacher | Apache-2.0 | 40M | revision-bound; no hosted API | planned |
| `gen-hosted-open` | a hosted OpenAI-compatible provider, **open-weight models only** | per model, recorded | 20M | allowlist; no private input; verified; no eval overlap | planned |

Excluded by rule: any hosted model whose terms forbid training on its outputs,
any prompt containing a soul, a child message, a transcript, a credential or a
customer document.

## Open licence questions

| source | question | effect |
|---|---|---|
| xLAM | NC reported; confirm | **excluded** from the open-weight release |
| Dolly | CC-BY-SA-3.0 | **excluded**; share-alike reaches the weights |
| CulturaX | gated terms | **excluded** unless the terms permit weight redistribution |
| Tulu 3 | mixture per subset | some subsets may be excluded |
| The Stack v2 | per-file long tail | allowlist only; unknown licence dropped |
| crates.io | per-crate long tail | allowlist only; drop vendored |

## What the index does not solve

- **Availability.** A planned source is not a fetched source; the status column
  is the truth.
- **Quality.** A licence-clean row can still be a bad row; the eval gates decide.
- **Contamination.** The held-out sets must be checked against every fetched
  source, not assumed disjoint.
