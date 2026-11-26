# Generating tool-use traces from agentic-loop

Status: **design, 2026-11-26.** This is the 12% of the mix that cannot be fetched:
every permissive tool-calling set worth having is either non-commercial (xLAM) or
ships a calling convention we do not use. Generating our own is licence-clean by
construction and produces exactly the convention the harness already speaks.

## Why the harness is the right generator

`agentic-loop` already has the three things a trace needs and no dataset has:

1. **a tool registry** — file ops, shell, ast, web, engram, workspace tools, each
   with a declared schema and a risk class;
2. **a runner** that executes a workflow step, calls tools, feeds results back and
   terminates on a control tool (`done`, `failed`, `escalate`);
3. **a sandbox per task** — a worktree, so a `file_write` is real and a failed
   command is a real failure that the trace can record.

A trace is therefore not authored: it is *observed*.

## The loop

```
seed task ──▶ workflow step (teacher as model) ──▶ tool_call
                                                      │
                          real tool executes in a sandbox
                                                      ▼
                     tool_result ──▶ next step ... ──▶ done(summary)
                                                      │
                        harvest: system, task, calls, results, answer
```

The teacher is **served, not embedded**: `Qwen3-4B` in fp16 behind a
vLLM/OpenAI-compatible endpoint on a T4, with the harness pointed at it as a
provider. The engine's own GPU path is irrelevant here — and that matters, because
it is the broken part.

## What makes the output worth training on

Volume is the easy half; these are the gates that decide whether 60M tokens of
traces teach anything:

| gate | test | reject when |
|---|---|---|
| **grounding** | the final answer must cite a value that appears in a tool result | the model answers from priors |
| **execution** | every call parses against the tool schema and returns | a malformed argument (keep a labelled *repair* subset instead) |
| **recovery** | after an error result, the next call must change strategy | the model retries the identical call |
| **termination** | the run ends through a control tool within N steps | it stops mid-plan |
| **diversity** | hash of `(task template, tool sequence)` | duplicate above a threshold |
| **no leakage** | the task must not contain the answer | seeded from the result rather than the question |

A rejection rate of 40–60% is normal and healthy: the kept set is what teaches.

## Seeding, which is where diversity comes from

Templates alone collapse to one tone. Seeds come from three places, in this order:

1. **real work** — every task in the engram queue that a human or agent actually
   ran, including the ones that failed (a failure trace with a recovery is the
   most valuable kind);
2. **combinatorial templates** over the tool schemas — `(verb, target, constraint)`
   where the target is drawn from our own corpora: a file that exists, a symbol
   that exists, a command that is legal;
3. **counterfactual probes** — the same task with one impossible element, to teach
   refusal and clarification rather than a confident wrong call.

## The arithmetic, because it decides the plan

A trace with cached tool definitions is roughly:

- 1,500–2,500 prompt tokens (tool schemas dominate) and 200–400 generated;
- 30k traces ≈ **60M training tokens** — the 12% share of a 500M corpus.

Teacher cost per trace at 4B fp16 on one T4: ~2 s prefill + ~15 s decode
single-stream, so **~17 s**. With vLLM continuous batching at 8–16 concurrent that
is 1–2 s of wall clock per trace:

| milestone | traces | training tokens | teacher wall clock |
|---|---:|---:|---|
| M1 proof | 2,000 | ~4M | ~1 h |
| M2 useful | 20,000 | ~40M | ~10 h |
| M3 the 12% share | 30,000 | ~60M | ~15 h |

Tool definitions must be **prefix-cached** or the prefill dominates: they are the
same on every trace.

## Milestones

1. **M1** — a harvester script that takes a finished run and writes one JSONL
   record (`system`, `user`, `calls`, `results`, `answer`, `outcome`, `seed`), and
   the six gates above as a filter. Run 2,000 traces, report the acceptance rate,
   the tool-call distribution, and the token count. Stop if acceptance is below
   30% — that means the seeds, not the model, are wrong.
2. **M2** — 20,000 traces with prefix caching and batching; measure tokens per
   GPU-hour, which is the number the whole plan depends on.
3. **M3** — scale by *adding seeds*, never by repeating them.

## What this does not solve

- **The convention gap.** Traces teach our tools, not the ecosystem's. A model
  trained only on these will not know what to do with an unseen tool schema; that
  is what the tool-schema diversity in seeding is for, and it is unproven.
- **Correctness of the *plan*.** A trace can be grounded and still do the wrong
  thing well. Judging plans needs the eval suite, which does not exist.
- **Cost.** 15 GPU-hours is 15 of a 30 h/week quota, for one of nine sources.
