# Long-running work

Status: **design, 2026-11-26.** The runtime side exists; the model behaviours are
a training target and a gate.

## The distinction

A model call is stateless. A **long-running agent** is a loop: the harness keeps
the durable state, and the model decides, acts, compacts, recovers and hands
off. Training can make those behaviours reliable; it cannot make the model run
between calls.

| layer | already provides |
|---|---|
| session store | per-step JSONL + workflow log; resume after a restart |
| workflow engine | states, turn budgets, terminal transitions |
| checkpoint/resume | `--resume` with step and data offset (training); session resume (daemon) |
| supervision | health, restart policy, incidents, no silent death |
| memory | engram entities; retrieval; skills; soul |
| control plane | budgets, stop conditions, the arbiter |
| daemon | one long-lived process; the Telegram front door |

## What to teach the model

1. **Long-horizon planning** — a plan that is revised as observations arrive,
   not restated every turn. Trajectories of 50–500 steps.
2. **Compaction** — when the context fills, summarise the state into a compact,
   structured form and continue: goal, decisions, files touched, evidence,
   open questions. Trained as `(long history, budget) -> summary -> next
   action`.
3. **Handoff** — produce a resumable state a fresh session or agent can pick up:
   the same compaction format, plus the exact next step and the verification
   command. Trained as `(history -> handoff)` and evaluated by resuming.
4. **Recovery** — a failed tool, a red test, a timeout: retry, change approach,
   or escalate. Never loop the same call.
5. **Budget awareness** — "finish, hand off, or escalate before the step/time
   budget"; a long task must end in a state, not a timeout.
6. **Memory use** — write a fact once; retrieve it instead of re-deriving it;
   cite the entity.
7. **Workflow conformance** — follow the state grammar and the grants; a
   long-running agent that leaves the workflow is not long-running, it is lost.

## Compaction is trained, not prompted

The 2026-11-25 RECAP result makes the summary a training target: the objective
is to preserve the behaviour induced by the original history, measured as the KL
between next actions under the full and the compacted context, with a
distortion-based reward for the summary and self-distillation for the actions
that follow (`research/hf-papers-2026-11-25-agentic.md`). Until that training
runs, evaluate every compaction with the same divergence: a summary that changes
the next action is a bad summary, however readable it is.

## Data

| source | shape | rule |
|---|---|---|
| harness run records | full trajectories, reviewed | redacted, derived, consented; never raw |
| synthetic long tasks | 50–200 steps with scripted failures | authored, CC0 |
| compaction pairs | history → summary → next action | reviewed for fact retention |
| handoff pairs | history → resumable state | evaluated by a fresh agent resuming |
| recovery pairs | failure → retry/alternative/escalation | from real incidents and authored |
| SWE-bench-like tasks | issue → patch → verify | licence per dataset |

Long trajectories are token-expensive; a 0.6B at 8k will spend most of its budget
on the recent window. That is why compaction is trained first: it is what makes
a longer task fit.

## Evaluation

| metric | gate |
|---|---|
| task success at a step budget (10 / 50 / 200) | report the curve; no cliff at 50 |
| **compaction retention** | facts needed later are present in the summary ≥ 95% of the time |
| **handoff resume** | a fresh session completes the task from the handoff ≥ 80% |
| recovery | failed step recovered or escalated, never repeated identically |
| budget | every run ends in `done`/`failed`/`handed off`, never a silent timeout |
| drift | after three compactions the goal still matches the original |

The last row is the one that catches the failure mode: a long agent that slowly
forgets why it started.

## Runtime requirements (not training)

- **Every loop can stop**: a progress check (three identical
  `(task, outcome, error)` tuples abort), a resource refusal, a wall-clock and
  token budget.
- **A checkpoint is a snapshot of one commit and one state**; resuming into a
  different world is refused, not guessed.
- **The daemon is one long-lived process**; supervision restarts a dead
  component and records an incident.

## Non-goals

- The model holding state between calls.
- Unbounded loops: a long-running agent without a budget is a hang with a
  friendlier name.
- Training on raw logs or transcripts; the trajectories are derived and
  redacted.
