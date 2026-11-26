# Response modes

Status: **design, 2026-11-26.** Fast/smart hints exist per workflow state, the
tier ladder exists, and routing exists. A user-facing mode contract and its
latency/quality gates do not.

## Three things people mean by "mode"

| layer | what changes | example |
|---|---|---|
| **behaviour** | how the model answers: directly, or with a plan and checks | one checkpoint, conditioned by a mode token |
| **capability** | which tier answers: Small, Base, a hosted model | routing and escalation |
| **budget** | decoding and tool limits: reasoning tokens, tool calls, verification | per-request policy |

A mode is all three, wired together. It is **not** two checkpoints: two
artefacts double the training, the eval and the cache work, and one of them will
always be the neglected one.

## The modes

| mode | who answers | behaviour | budget | must never |
|---|---|---|---|---|
| **instant** | Small (or Nano for a draft) | direct answer, no visible plan | no reasoning scratchpad, <= 1 tool call, no verification step | claim something was verified |
| **balanced** (default) | Small, then Base on a signal | short plan when the task needs it | <= 4 tool calls, verify code claims | silently escalate without saying so |
| **deliberate** | Base/Large, or a hosted model if configured | plan → act → check → answer; scratchpad trimmed | tool and step budget, verification required for factual/code claims | pretend the local tier did it |
| **auto** | decided per request | routes on **uncertainty and domain shift**, not task length (evidence: `research/hf-papers-2026-11-26.md`) | inherits the chosen mode's budget | hide which mode was chosen |

`auto` is the product default. The user can force a mode; the runtime reports
which one answered and whether it escalated.

## What "reasoning" means for a 111M model

Not a long chain of thought from the small weights. Three honest mechanisms:

1. **The harness loop** — plan, call a tool, read the result, revise. Reasoning
   is externalised into observations; this is what the model is trained for.
2. **Verification** — the answer carries a command, a citation or a computation.
   A conclusion without evidence is a draft, not a result.
3. **Escalation** — when the task exceeds the tier's context, tools or measured
   quality, a larger model answers, and the user is told.

Chain-of-thought is included only where it measurably improves accuracy. A
small model can produce confident, useless reasoning; the eval decides.

## One checkpoint, conditioned

The mode is a token/instruction in the renderer, trained like the identity
block:

- the same task appears in instant and deliberate forms: the deliberate answer
  is **shorter or equal in prose but carries a check**, not merely longer;
- instant examples must not contain hidden reasoning that leaks into the answer;
- a deliberate answer that failed its check is a negative example;
- randomise the mode across the corpus so the model does not learn one register.

Escalation is not trained into the weights as a capability claim; the model
predicts "this exceeds me" and the runtime routes. The decision head and the
catalogue are the mechanism.

## What each mode buys, and costs

| mode | TTFT target (local) | total target | cost | risk |
|---|---|---|---|---|
| instant | < 300 ms | < 2 s for a short reply | one small decode, prefix cache hit | sloppiness; overclaim |
| balanced | < 500 ms | < 6 s | a few tool calls | tool spam |
| deliberate | < 1 s | tens of seconds | more tokens, maybe a larger tier | performative reasoning; hidden escalation |

The numbers are targets for the i5/Strix class, not measurements; the smoke and
the local benchmark produce the real ones.

## Implementation

| piece | exists | role |
|---|---|---|
| tier descriptors, `ModelSelector` | yes | capability routing |
| `decision` head, `ActionRouter`/Needle | yes | auto-mode and escalation signal |
| `model-tiers.md` | yes | which tier fits which mode |
| Nano draft, speculative decoding | designed | instant-mode latency |
| prefix KV, compaction | designed | instant and deliberate budgets |
| mode token in the renderer | **missing** | behaviour conditioning |
| mode in the principal/request config | **missing** | user control |
| latency/quality gates per mode | **missing** | the contract |

## Eval gates

| metric | gate |
|---|---|
| instant: factual/code claim without verification | report; target 0 for code |
| instant overclaim ("I ran the test") | **0** |
| deliberate: answers a factual/code question without a check | **0** |
| hidden escalation (larger tier answered, not reported) | **0** |
| mode chosen matches the forced mode | 100% |
| auto-mode choice on a labelled set | >= 85% |
| deliberate improves accuracy over instant on tasks that need it | report; if it does not, cut the mode |
| p95 TTFT/total per mode | within the target table, measured on the reference host |

The two that decide the feature: **deliberate must beat instant where it
matters**, and **instant must never lie about verification**. Without the first,
the mode is theatre; without the second, it is a liability.

## Risks

- **Mode collapse**: instant becomes the only trained behaviour and deliberate
  degrades. Guard by training both on every batch, not in separate stages.
- **Performative reasoning**: longer answers that do not improve accuracy. The
  eval is accuracy and cost, not length.
- **Hidden escalation**: the user believes a local model answered. Report the
  tier in the response metadata, always.
- **Cache incoherence**: instant uses a prefix cache; a mode change invalidates
  the behaviour prefix. Key the cache by (soul, tools, mode).
