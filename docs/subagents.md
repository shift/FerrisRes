# Subagents

Status: **design, 2026-11-26.** Delegation exists (`delegate_task`, `consensus`,
agent pools, a task executor). The controls that make model-initiated delegation
safe do not: grant derivation, a tree budget, a depth bound, a typed result
contract and a writer-scope check.

## The question, split

"FerrisRes calls its own subagents" can mean three different things, and only
one of them is a model decision:

| kind | who decides | state |
|---|---|---|
| **authored delegation** — a workflow names an assignee or a parallel branch | the workflow author | exists, deterministic |
| **proposed delegation** — the model emits `delegate_task` | the model proposes, the runtime decides | the tool exists; the gate does not |
| **recursive delegation** — a child spawns its own children | the model, inside a child | should be refused by default |

The question is really the second one. The model may *propose*; the control
plane grants.

## Why it is worth doing

For a 0.6–1.7B, the binding constraint is context, not intelligence. A child
with its own context is the cheapest way to do a bounded investigation the
parent cannot afford to hold:

- **context isolation**: search ten files in a child, return five lines and a
  citation, and the parent's window never sees the ten files;
- **fresh-context review**: the reviewer must not share the author's transcript;
  this is already the project's own rule, and a child session is how it is
  implemented;
- **independence**: two read-only explorations can run at once;
- **escalation**: a hard subproblem can go to a larger tier while the parent
  keeps the plan.

A large model can hold the whole investigation. A small one cannot. Delegation
matters more here than it does for a frontier model, which is the strongest
argument for training the policy.

## Why it is dangerous

| risk | what it looks like |
|---|---|
| **authority laundering** | the parent lacks `bash`, so it delegates to a child that has it |
| **budget explosion** | each child re-prefills soul, tools and history; N children cost N times before any work is done |
| **write collision** | two children edit the same file; the "one writer per file" rule is violated by the runtime, not the author |
| **unbounded recursion** | a child delegates, its children delegate; the tree grows until the machine or the bill stops it |
| **context laundering** | the child returns its transcript, so the parent's context grows by exactly what delegation was meant to save |
| **silent death** | a child panics or stalls and the parent waits; supervision exists, but only if children go through it |

## The rule

**Yes, as a runtime-granted capability with a tree budget — read first, writes
only where the workflow declared them.**

1. **The grant is derived, never inherited.** Child tools = (parent's offered
   tools) ∩ (tools the workflow declares for the child). Any write class needs
   an explicit declaration; `bash` and `mcp__*` are never inherited implicitly.
   A child is a new principal boundary, not a copy of the parent's authority.
2. **The budget is the tree's, not the agent's.** One ledger per run: maximum
   depth, maximum children, total tokens and wall clock shared by every
   descendant. The arbiter enforces it; on exhaustion the parent does the work
   or escalates. A delegation that cannot be afforded is refused before spawn.
3. **The result is typed and bounded.** The child returns a summary with
   evidence ids, paths and a revision — not prose and not a transcript. Its
   conversation is not appended to the parent's.
4. **Writes are scoped.** A child may write only inside a file set the
   delegation declared, and two children never share a file. Otherwise the
   parent serializes them.
5. **Recursion stops.** A child cannot spawn children by default (depth 1); a
   workflow that needs depth 2 says so, and the tree budget is halved for it.
6. **Every child is supervised and persisted.** It is a session with an
   incident path, so a panic is a typed failure to the parent, not a hang.

## What to allow

| use | decision | reason |
|---|---|---|
| parallel read-only exploration | **allow**, bounded | the canonical win: context isolation |
| fresh-context review | **allow**, required by the review rule | the author must not review itself |
| bounded research returning a summary | **allow** | compresses well into the parent |
| escalation to a larger model | **allow**; it is routing, recorded | the small tier should know its limit |
| writes to a declared, disjoint file set | **allow only when the workflow declares it** | keeps one writer per file |
| writes to the same file | **refuse** | collision, and no merge authority |
| recursive spawning | **refuse at depth 1** unless authored | fan-out without a bound is a hang |
| delegating a one-tool task | **train against** | the child costs more than the call |

## The one-GPU correction

On a single T4, two subagents do not run in parallel: they serialize at the
model, and each pays its own prefill. So on one device delegation buys
**context isolation and fresh eyes**, not throughput. It buys wall clock only
when the work is I/O-bound (network, disk, a build) or when the children are
split across devices — which is the real reason the two-T4 launcher matters.

Do not train or claim "parallel agents" on hardware that has one queue.

## What is missing today

| control | state |
|---|---|
| grant intersection | absent; the child runtime is whatever the executor builds |
| tree budget (depth, children, tokens, wall clock) | absent; pools have `max_concurrent` per pool, not a tree |
| typed child result | `delegate_task` returns free text |
| writer-scope check | absent |
| depth bound | absent |
| delegation eval | absent |
| licence gate | present (`collaboration`, professional) |

The grant derivation is the one to fix first: it is the difference between a
subagent and a privilege-escalation path.

## The trained policy

Delegation is a procedure with a brief, and the brief is the skill:

- state the **sub-question**, not the whole task;
- state the **acceptance check** (what the child must run or cite);
- state the **output shape** (a summary with paths, ids, revision);
- state the **scope** (which files and tools are allowed);
- do not delegate when one tool call answers it;
- on a child failure, change the brief or do it yourself — never resend the same
  brief.

Train on trajectories where delegation helped and where it wasted a run. The
positive and negative pairs both matter; a model trained only on successes
delegates everything.

## Metrics

| metric | why |
|---|---|
| tokens per completed task, with and without delegation | the only reason to allow it |
| wasted-delegation rate (child result unusable) | the failure a small model will have |
| duplicate-writer incidents | must be 0 |
| depth and fan-out distribution | proves the bound is enforced, not assumed |
| parent context growth per delegation | must be bounded by the summary, not the transcript |
| wall clock on one device vs two | the correction above, measured |

Falsification: if tokens per completed task do not improve on multi-file
investigations, or if any duplicate-writer incident occurs, the feature is not
ready — whatever the success rate looks like.
