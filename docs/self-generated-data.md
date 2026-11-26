# Self-generated harness data

Status: **design, 2026-11-26.** Yes — agentic-loop can generate its own training
data, and it is the natural dogfooding source. The condition is the same as
everywhere else: **the verifier is the filter, not the generator's confidence.**

## What can be generated

| family | generator produces | verifier | value |
|---|---|---|---|
| **workflow authoring** | a `.workflow` for a stated task | parser + validate + a dry run | the grammar and the delegation brief |
| **task briefs** | one behaviour, one test, one `VERIFY` | the loop runs it; review passes | the loop's hardest skill |
| **tool trajectories** | state → tool call → result → next | the task's check, schema validation, evidence | tool protocol and recovery |
| **tool discovery** | capability query → results → select → call | the call validates against the schema | finding unlisted tools |
| **doc-QA** | question → retrieval → cited answer | the answer matches the doc; citation resolves | grounding, version awareness |
| **review** | diff + evidence → findings | a seeded defect is found; no false finding on a clean diff | review discipline |
| **repair / recovery** | failed step → diagnosis → retry/alternative/escalate | the named check passes after the repair | the long-horizon failure mode |
| **soul/persona** | variant responses under a soul | a rubric: register, constraints, refusal | persona conditioning |

Each generator is a workflow or a scripted loop with a **fixed seed**, a
generator id and a verifier id. The output is a JSONL row:

```json
{
  "source": "selfgen:tool-trajectory",
  "generator": "cbase/qwen3-coder:30b@2026-11-26",
  "verifier": "run_tests:foo::test_bar",
  "revision": "<engine commit>",
  "licence": "ours",
  "seed": 41,
  "verified": true,
  "redacted": true,
  "task": "…",
  "turns": [{"role": "assistant", "tool_call": {...}}, {"role": "tool", "result": "…"}]
}
```

A row with `verified: false` does not train. A row without a verifier id does not
train. A row from a non-allowlisted generator does not train.

## The pipeline

```
seed task family → generate → execute in the harness → verify → redact
→ dedupe → hold out → manifest
```

- **Execute, don't invent.** The trajectory must come from a real run of the
  harness, with real tool results, even if the task is synthetic.
- **Verify the outcome.** Tests, schema validation, workflow parse, a seeded
  defect, a citation that resolves. A model's self-assessment is not a verifier.
- **Redact.** No user data, no transcripts, no secrets; the observation layer
  already redacts credential-shaped values, but the generator must also avoid
  authoring them.
- **Dedupe** against the corpus and the eval sets. A generated row that overlaps
  a held-out item is removed before training.
- **Hold out by task family**, not by row, or the eval measures memorisation of
  the generator's phrasing.

## What is already there

- `dataset export --metadata-only` — safe session metadata; **content export is
  not implemented**, which is the gap this design closes.
- `harness-dataset` (63 shipped workflows, ~455 KB) — the grammar corpus.
- Run records, `evidence_query`, incidents, review reports — the verified
  outcomes.
- The eval collection (15-gate manifest, interface, arithmetic, Armor, jailbreak)
  — the anchor set that generated data must not contaminate.

## The risks

| risk | why | mitigation |
|---|---|---|
| **model collapse** | training on the generator's own outputs narrows the distribution | the verifier is external; keep a human-authored anchor set; mix, never replace |
| **shortcut learning** | the trajectory passes but the reasoning is wrong | verify the outcome and, on a sample, the process; review the trajectories |
| **contamination** | generated rows resemble eval items | dedupe and hold out by task family |
| **licence / terms** | a hosted generator's outputs may not train a competing model | the per-generator allowlist (`model-licensing.md`); record the terms snapshot |
| **privacy** | real sessions are tempting | synthetic tasks only; no user data; redaction gate |
| **volume without value** | cheap generation floods the corpus | yield rate and eval delta decide; a source that does not improve the eval is cut |
| **feedback loop at inference** | the model trains on its own style | mixed authors, surface randomisation, held-out unseen tasks |

## Metrics

| metric | meaning | gate |
|---|---|---|
| yield | verified / generated | report per family; a low yield is a generator bug, not data |
| diversity | distinct task families, tools, souls | report; a single-family corpus is rejected |
| contamination | overlap with held-out sets | **0** |
| cost per verified example | tokens + wall clock | report; compare with human-authored |
| eval delta | the only real test | a source that does not improve the eval is cut |
| reviewer load | sample reviewed per 1k rows | report; review capacity is the bottleneck |

## Where it fits

This is a **narrow, high-quality** source, not a substitute for the licensed
general corpus. It teaches the harness's vocabulary, protocol and recovery
behaviour — the part no public dataset contains — while the general corpus
teaches language and code. Both are needed; neither replaces the other.

It is also the cheapest source to start: generation runs on the harness and a
generator model, not on the T4 training path, so it can proceed while the GPU
loop is being finished.

## Gaps to close

1. A `selfplay` workflow per family, with a fixed seed and a verifier.
2. Content export from a run (the metadata-only exporter is the starting point).
3. A dedupe/contamination checker against the eval sets.
4. A generator allowlist record per row.
5. A review sample workflow — generated data is not trusted until someone
   reads a fraction of it.
