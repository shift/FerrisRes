# HF papers — harness line (2026)

Status: **scan, 2026-11-26.** The `?q=harness` filter on the 25 November page is
global, not date-scoped; the 25th has no harness-specific paper. The search does
surface a coherent line from April–October 2026. Five are directly relevant;
three are named but not assessed.

| paper | claim in one line | for us |
|---|---|---|
| **The Scaffold Effect in Coding Agents** (`2607.22585`) | harness choice changes tokens per solved task by up to **40×**, while pass-rate differences stay 0–8 pp | our cost claims must be model+harness pairs |
| **Harness-Bench** (`2605.27922`) | model-harness configurations fail in characteristic ways; **execution alignment** is the recurring failure | our evidence gates are the right shape |
| **Harness-1** (`2606.02373`) | externalise search state in the harness; the policy keeps only semantic decisions | validates the session notebook |
| **Harness-Aware Distillation** (`2610.02858`) | distil what the teacher adds **beyond the harness**, not the whole output | refines our KD plan |
| **Harness Updating Is Not Harness Benefit** (`2605.30621`) | updating a harness is flat across model tiers; **benefiting** from it is not | small models need activation/following training |
| **CAAF / Harness as an Asset** (`2604.17025`) | deterministic machine-readable invariants beat multi-agent orchestration | the control plane is the boundary |
| **HarnessRisk** (`2608.17597`) | harness **configuration** is the most vulnerable phase; risk recognition ≠ safe action | safety card + config review |
| **Harness Continual Learning** (`2608.19013`) | harness updates can forget; guard commitment with retention checks | skills/memory versioning |

## 1. The Scaffold Effect — measure the pair, not the model

Across Qwen 3.6 Plus and MiniMax M2.5 on three harnesses (Goose, OpenCode,
OpenHands-SDK) over 50 Terminal-Bench Pro tasks: harness choice induced up to a
**40× difference in tokens per solved task**, while within-model pass-rate
differences were **0–8 points**. Failure fingerprints replicated across models
per harness (REASON for Goose, VERIFY/MAX_TURNS for OpenHands, idle-loop/TIME for
OpenCode). Recommendation: select harness-model pairs by pass rate under
token/latency budgets, and report the full harness spec.

**For us.** Every benchmark claim we make — "competitive at a fraction of the
cost" — is a claim about a pair. The eval record already has device and model;
it needs the **harness id + version**, token and latency budgets, and the
per-task token count. Our `tool-generalization.md` cross-harness claim is
strengthened by this: harness identity is not a nuisance variable, it is half
the result.

## 2. Harness-Bench — execution alignment

106 sandboxed tasks, 5,194 trajectories, multiple harness configurations under
shared environments and budgets. The recurring failure is **execution
alignment**: plausible reasoning decoupled from tool feedback, workspace state,
evidence, or verifiable output contracts.

**For us.** This names our gates exactly. Add an execution-alignment failure
taxonomy to the eval record: *ignored tool result*, *claimed without evidence*,
*workspace state mismatch*, *contract not met*. A model can be right in prose and
wrong in the trace; only the trace is evidence.

## 3. Harness-1 — externalise state

A 20B search agent trained inside a stateful harness that maintains a candidate
pool, an importance-tagged curated set, **evidence links**, **verification
records**, compressed observations and budget-aware rendering. The policy keeps
only the semantic decisions: what to search, keep, verify, and when to stop.
0.730 curated recall, +11.4 points over the next open search agent.

**For us.** This is the session notebook, independently arrived at: typed state
in the harness, semantic decisions in the model. It is the strongest external
support for `reasoning-and-retention.md` and for not training bookkeeping into a
111M model. The one difference: their state is search-specific; ours is general
(goal/decision/fact/evidence/open question).

## 4. Harness-Aware Distillation — distil the residual

The harness stays in place when a small student is deployed, so the student
mainly needs what the teacher adds **beyond** the harness. HAD contrasts the
teacher's actions with and without harness information, scores after the
student's own reasoning, and drops pairs whose preferred action contradicts the
harness records. It needs no task rewards.

**For us.** This refines the KD split from the 25th: text-aligned KD for
language, action-only SFT for policy, and HAD for the **residual** — what the
teacher knows that the harness cannot supply. It also says the validity check
against harness records is not optional; a preference pair that contradicts the
trace is poison.

## 5. Updating vs benefiting — train the small model to *use* the harness

Harness-updating is **flat** across capability tiers (a 9B's updates are
comparable to a frontier model's); harness-**benefit** is **non-monotonic** —
weak models gain little because they either fail to activate the relevant
artifact or activate it and fail to follow it. Recommendation: invest capability
in the task agent, and train harness invocation and long-horizon instruction
following.

**For us.** A free/small model can propose harness changes; it will not benefit
from them without training on activation and faithful following. That is a
concrete training target: "the tool/skill/memory exists — call it, and do what it
says". Add it to the harness-integration matrix and to the tool-use data.

## Cross-cutting rules to adopt

1. **The unit is the model+harness pair.** Record the harness id/version, token
   and latency budgets, and per-task tokens in every eval record.
2. **Externalise state, train decisions.** The notebook is right; do not train a
   small model to remember what the harness can hold.
3. **Execution alignment is a first-class metric.** A correct answer from a
   decoupled trace is a failure.
4. **Distil the residual**, and validate preference pairs against the harness
   records.
5. **Train activation and following**, not just harness editing.
6. **Harness configuration is a safety surface.** A config change can widen
   permissions without touching code; review it like code.

## Not assessed

- `Harness-of-Harness` (`2609.01481`) — multi-day autonomous development.
- `Harness-Zero` (`2609.24974`) — distilling a harness into weights.
- `Harness Continual Learning` (`2608.19013`) — harness-level forgetting (read
  the abstract; the retention check is the useful part).
- `HarnessRisk` (`2608.17597`) — safety across six lifecycle phases; worth a
  full read before the first public deployment.
