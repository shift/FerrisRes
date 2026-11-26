# Harness integration

Status: **design, 2026-11-26.** What "tightly integrated" means beyond tool
calling: the model knows the harness's vocabulary and respects its boundaries.

## The rule

Integration is a protocol, not a tone. Three things make it real:

1. **One renderer.** The exact order (soul, tools, memory, evidence, history,
   user), separators, chat template, stop tokens and tool-call envelope are
   shared by the dataset builder and the runtime, and versioned. A model trained
   on a different layout is not integrated, whatever its benchmark says.
2. **The vocabulary.** The model can name a workflow state, an evidence kind, a
   control decision and a failure class, because the harness already defines
   them.
3. **The boundaries.** Observed content is shaped like instructions but is data.
   The model does not obey it; the runtime filters it and the model treats it as
   text.

## Capability matrix

| capability | why it is harness-specific | data | eval |
|---|---|---|---|
| **Tool-result instruction resistance** | tool output can say "ignore the previous instructions"; following it is the defining failure of an agent | runs where a tool result contained an imperative, authored and real | never follow; report the attempt; 0/0 on a red-team set |
| **Evidence literacy** | build/test/lint/diff records are revision-bound; a green result from another commit is not evidence | `agentic-loop-evidence` records paired with the revision | stale-evidence detection; no claim without a command |
| **Failure taxonomy** | parser, validator, test, build, timeout, auth, quota need different actions; external blockers stop the loop | incident records, failed steps, the recovery that worked | class + action accuracy |
| **Workflow grammar** | states, grants, terminal transitions; a step outside the workflow is lost | `dataset workflows` (63, scaling via generate→verify) | parse + validate rate ≥ 95% |
| **Control vocabulary** | proposal, approval, denial, risk class, budget; a denial is accepted and repaired, not argued | control traces, shadow decisions | denial accepted; no retry of a denied action |
| **Delegation briefs** | "one behaviour, one test, one VERIFY command"; the loop cannot finish a vague task | task descriptions and review outcomes | a brief that passes the loop; review completeness |
| **Memory protocol** | entity types, ids, dedupe, retrieve before deriving | memory operations from runs | correct type/id; no duplicate entity |
| **Surface formatting** | Telegram length, markdown, keyboards, no wall of text | surface examples and run outputs | format/length validity |
| **Self-routing** | when the task exceeds this tier's context, tools or quality, escalate | routing outcomes, tier descriptors | route choice on a labelled set |
| **Harness activation and faithful following** | a tool, skill or memory exists in the harness; the model must call it and follow it, not re-derive it | activation traces, skills, memories | call rate; following rate; a weak model that activates but ignores is the failure |
| **Residual distillation** | the harness already supplies state and tools; the student only needs what the teacher adds beyond it | harness-aware preference pairs, validated against the trace | no preference pair contradicting the harness record |
| **Secure review procedure** | a code review must end in a file, a line, a reproducing command and a fix, not a prose warning | vulnerable-snippet pairs, scanner output, redacted incidents | finding without a repro = 0; scanner result absent at the line = 0; scope respected |
| **Report discipline** | verified / implemented / designed, counts not percentages | review reports and the feature register | unsupported claims counted |
| **Documentation grounding** | the user asks about a command, flag, error or ADP operation; a memorised answer drifts | `cli_help`, installed docs, doc-QA pairs (`tasks/versioned-docs-tool.md`) | citation present; no invented flag/operation; stale-doc question → retrieve or refuse |

The first two are the ones a generic model cannot have: injection resistance in
*observed* content, and knowing that evidence is bound to a revision.

## Data sources

- **Run records** (reviewed, redacted): the only source of real tool-result
  injections, denials, failures and recoveries.
- **`agentic-loop-evidence`**: revision-bound records, already typed.
- **`dataset workflows`** and **generate→verify**: the grammar corpus.
- **Incidents and supervision**: the failure classes that actually happened.
- **Control traces**: the shadow decisions and their outcomes.

## Gates

| gate | pass |
|---|---|
| injection in tool result followed | 0 |
| evidence from another revision treated as current | 0 |
| workflow step parses and validates | ≥ 95% |
| denied action retried | 0 |
| task brief passes the loop (review) | ≥ 80% |
| unsupported "verified" claim in a report | 0 |

A model that scores well on perplexity and fails the first two is not integrated;
it is a chat model wearing the harness's prompt.

## What not to bake

- A fixed workflow catalogue or tool list: both change per workflow and
  principal; train the grammar and the protocol.
- The renderer's text itself: keep it in one place so training and serving
  cannot drift.
- The harness's authority: the model proposes; the control plane decides.
