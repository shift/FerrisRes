# Reasoning and retention

Status: **design with the tool surface verified, 2026-11-26.** The recall tools
and the compaction path exist; the trained policy, the retention metrics and the
session notebook do not.

## The thesis

For a small model, capability is three things, not one:

```
capability = procedure in the weights
           + facts in stores
           + a trained policy that connects them
```

Do not train facts into the weights: they go stale, they cost parameters, and a
0.6B cannot hold enough of them to matter. Train the **policy** — when to look,
where to look, what to keep, what to drop, when to say "I do not know". The
weights hold the loop; the stores hold the world.

This is also what makes the model portable. A model that knows *how to find and
cite* works on a corpus it has never seen; a model that memorised the corpus
does not.

## Three tiers of retention

| tier | lifetime | mechanism | what lives there |
|---|---|---|---|
| working context | one call | the prompt | the goal, the current step, the facts in use |
| session state | one session | transcript, prefix KV, compaction summary | turns, tool results, plan, decisions, evidence ids |
| durable memory | across sessions | evidence spans, entities, skills, docs index | facts, decisions, procedures, citations |

The rule that makes the tiers work: **anything you would re-derive is written
once; anything you cite is bound to a revision.**

## The retention policy (the trained part)

1. **Retrieve before asserting.** Search, read, then answer. No claim without a
   source.
2. **Search gives ids; read gives text.** Prefer the pattern
   `memory_search` → `memory_verbatim(span_id)`, not a bulk dump. `memory_search`
   deliberately returns at most a twelve-word snippet.
3. **Write once, with a type and an id.** A fact is an entity, an event is a
   span, a procedure is a skill; do not create a second one.
4. **Keep the goal verbatim; summarise the rest.** The failure after three
   compactions is a summary that no longer matches why the work started.
5. **Re-check a recalled fact before use.** Evidence is revision-bound; a green
   test from another commit is not evidence.
6. **Distinguish absent from unretrieved.** `memory_negative_query` exists for
   this: an absence is a claim with a scope, and it goes stale.
7. **Drop consumed material.** A 20k outline kept for the rest of the run is
   context that dilutes attention.
8. **At a boundary, compact and continue** — do not restart and do not grow
   without limit.

## What is trained and what is enforced

| trained (model) | enforced (runtime) |
|---|---|
| when to search, and the query | the store's schema and scope |
| which result to open, which to discard | revision binding on evidence |
| what to write and with which type | ids, dedupe, grants |
| citation of every factual claim | the no-claim-without-source gate |
| staleness check before use | the revision the check compares against |
| abstention when nothing is found | the refusal path |
| compaction that keeps the goal | the token budget that triggers it |

A model cannot enforce its own citation rule; a runtime cannot choose a good
query. The split is the point.

## The tool surface that exists

Verified in the tree today (`agentic-loop-tools-core`):

| verb | tools | notes |
|---|---|---|
| **find** | `file_grep`, `file_glob`, `ast_outline`, `tool_search`, `docs_search` (designed), `skill_search` | discovery before reading |
| **read** | `file_read`, `ast_node_read`, `docs_read` (designed), `cli_help` (designed), `web_fetch`, `document_fetch` | targeted, not bulk |
| **recall** | `memory_search`, `memory_verbatim`, `memory_timeline`, `memory_unresolved`, `memory_negative_query`, `evidence_query`, `knowledge_query` | ids and snippets first, exact text second |
| **write** | `memory_store`/engram tools, `skill_create`, `skill_update` | typed entities, not prose |
| **state** | `compact_context` (answered by the runner), the session transcript | destructive compaction is the only structured state |
| **verify** | `run_tests`, `build_check`, `evidence_query` | a result is bound to a revision |

`memory_search` returning snippets and `memory_verbatim` returning the exact
quote is the design to train against: it keeps the context small and makes the
citation cheap. A model trained to ask for whole documents instead will spend
its window on text it never uses.

## Reasoning: what can and cannot be trained

**Trainable:** decomposition, plan revision when observations change the
picture, tool choice, checking its own result before reporting, recovery from a
failed step, budget awareness, asking a clarifying question when the request is
underspecified.

**Not trainable at this size:** depth of novel reasoning, long proofs, broad
world knowledge, ambiguous requirements that need judgement. The answer is the
tier ladder — recognise the limit and escalate — not a bigger promise.

Train on the **process**, evaluate the **outcome**. A model can produce a
beautiful plan and a wrong patch; the gate is the patch and the command that
verified it.

## Metrics

| metric | definition | gate |
|---|---|---|
| fact retention | facts introduced at turn *k*, recalled at *k+N* | >= 95% at N=20 |
| retention after compaction | facts needed later present in the summary | >= 95% |
| goal drift | goal after three compactions matches the original | no contradiction |
| re-derivation rate | identical tool calls repeated within a task | ~0 |
| stale-fact use | recalled fact used without a revision check | **0** |
| citation coverage | factual claims carrying a span, entity or path | 100% |
| duplicate entity rate | two entities for one fact | ~0 |
| context growth | tokens per turn | flat, not linear |
| abstention correctness | absent fact answered "not found" instead of guessed | 100% |
| retrieval efficiency | searches per successful recall | report, improve |

## Anti-patterns

- **Stuffing the context to "retain".** Cost grows, attention dilutes, and the
  model still forgets the middle.
- **Summarising away the goal.** Keep it verbatim; summarise progress.
- **Writing every observation to durable memory.** Noise drowns recall; write
  decisions, facts and evidence, not transcripts.
- **Caching a fact without a revision.** That is how a stale answer becomes
  confident.
- **Retrieval as a substitute for thinking.** Three searches for one question is
  a loop, not diligence.

## Gaps

| gap | effect |
|---|---|
| no non-destructive **session notebook** | compaction is the only structured state and it replaces history; the model cannot keep a fact without keeping the whole turn |
| `docs_search`/`docs_read`/`cli_help` designed, not built | the "look it up for this version" path is missing |
| no retention eval | the metrics above have no runner |
| evidence tools are not wired into the default chat workflow | recall exists but the model is not offered it |
| no training data for recall/citation policy | the behaviour is untaught |

The notebook is the smallest useful addition: a typed, bounded, in-context store
(`goal`, `decision`, `fact`, `evidence`, `open_question`) that the model writes
and reads without destroying history. It is also the natural training target for
retention, because every write is a labelled decision about what mattered.

## Falsification

The retention half of the claim is false if the model cannot recall a fact it
wrote twenty turns earlier, or if it answers a question the store does not
contain instead of searching. The reasoning half is false if it cannot finish a
task that a 30B finishes under the same scaffold — there, the answer is to
escalate, and the eval must record that it did.
