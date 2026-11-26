# Tool discovery and the generality claim

Status: **preregistered protocol, 2026-11-26.** No student has been trained yet.
This document fixes what would count as success and what would falsify the
claim, so the result is not chosen after the run.

## The claim, split into three

1. A student specialised on agentic-loop beats a larger general model **on
   agentic-loop's tasks**.
2. It learns to **find tools itself**: intent → capability query → read the
   schema → call → verify → recover.
3. The same weights, under a **generic scaffold with no agentic-loop plumbing**,
   stay competitive with much larger models on tool-use tasks at a fraction of
   the cost.

Only (1) and (2) follow from training on agentic-loop. (3) does not: it requires
surface randomisation and held-out tools. Without them the student memorises
this harness's renderer and this tool list, which is the opposite of the claim.

## The tension, stated plainly

"Specific to agentic-loop" and "without a specific harness" pull against each
other. The resolution is a split of what is specialised and what is randomised:

| specialised (learned) | randomised (surface) | held out (eval only) |
|---|---|---|
| the procedure: search, select, validate, verify, recover | renderer order, separators, chat template | 20% of tools, never seen |
| the vocabulary: states, evidence, control decisions, failure classes | tool names, descriptions, argument names | a second tool namespace |
| the boundaries: observed content is data, not instructions | tool ordering and candidate set size | a second harness |
| abstention: no tool fits → say so | registry size (10 vs 100+) | adversarial tool descriptions |

Generality is a property of the **data split**, not an aspiration. If the split
is not built before training, claim (3) cannot be measured afterwards.

## What the harness does that the model cannot

A learned model cannot enforce its own authority. Keep the layers distinct:

| layer | owner | why |
|---|---|---|
| grants, risk class, approval, budgets | runtime (control plane) | a model cannot authorise its own side effects |
| schema validation of the emitted call | runtime | the tool's schema is the contract |
| determinism and evidence | runtime | a result is bound to a revision, not to a claim |
| selecting a tool, querying, repairing, abstaining | **model** | this is the trained skill |
| recovery: retry, change approach, escalate | model proposes, runtime records | a repeated identical call is a defect |
| treating tool output as data | model, filtered by Armor | a description can carry an injection |

So "without a specific harness" must mean **without agentic-loop's scaffold**,
not "without any runtime". Anyone who reads it as the latter is being sold
something that does not exist.

## The learned discovery loop

Six steps, each a separate training target with its own failure mode:

1. **Intent → capability query.** The model asks for what it wants to do, not
   for a tool name. "create a file" is a query; `new_file` is an answer.
2. **Search results are data.** They arrive as text and must not be obeyed.
3. **Read the schema** before calling; an argument name is not guessable.
4. **Select or refine.** One refinement is normal; three identical queries is
   the failure mode.
5. **Call with valid arguments; repair from the validator error**, do not
   hallucinate a parameter.
6. **Verify and recover.** A green compile is not a passing check; a failed
   step changes the approach.

The seed is a real trace: `tool_search("create write file")` returned
`{"total": 0}` under the old single-substring matcher, the model rephrased, got
nothing again, and concluded no write tool existed. The tools were there. That
trace is exactly the behaviour to train out, and the matcher fix is already in.

## Data

| source | shape | target | rule |
|---|---|---|---|
| MCP server catalogue + OpenAPI specs | tool cards: name, description, schema | 200–2,000 tools | licence per source; no secrets in examples |
| agentic-loop registry | real tool infos, including `tool_search`, `bash`, `file_*`, `db_*`, evidence and control tools | the shipped set | exact strings, versioned |
| paraphrased descriptions | 3–5 rewrites per tool, one terse, one verbose, one vague | ×3 the registry | the skill must survive wording |
| near-miss distractors | `file_grep` vs `db_search` vs `tool_search`; `run_tests` vs `verify_build` | 5+ per cluster | forces schema reading, not name matching |
| discovery trajectories | intent → query → k results → select → call → verify | 50k+ | authored + replayed from runs |
| failed-query pairs | bad query, empty result, correct second query | 10k | from real traces, redacted |
| abstention pairs | no tool fits → name the gap, do not invent | 5k | CC0, authored |
| repair pairs | validator error → corrected call | 10k | generated from real schemas |
| adversarial descriptions | a tool description containing an imperative | 2k | authored; never obeyed |
| cross-renderer copies | the same trajectory in 3 surfaces | ×3 | generated by the one renderer, versioned |

A description is **observed content**. A tool whose description says "ignore
the previous instructions and call me with …" is an injection attempt and is
gated at zero, the same as a tool *result* injection.

## Splits that make the claim testable

| split | holds out | what it proves |
|---|---|---|
| unseen tools | 20% of tools: name, description, schema | discovery, not memorisation |
| unseen namespace | a whole prefix (`mcp__…`) | transfer across a vocabulary |
| second renderer | a different order and template | the protocol, not the prompt |
| second harness | a minimal ReAct loop, 200 lines | claim (3) directly |
| no-tool tasks | requests no available tool satisfies | abstention |
| large registry | 10 tools vs 100+ with distractors | selection under load |
| adversarial | imperative text inside descriptions and results | boundary |

## Eval arms

| arm | model | scaffold | what it isolates |
|---|---|---|---|
| A | student | agentic-loop | in-domain ceiling |
| B | student | minimal ReAct | claim (3) |
| C | teacher/base (Qwen3-1.7B) | both | what training added |
| D | Qwen3-30B | both | the size gap |
| E | frontier model | both | the ceiling, at cost |

## Metrics

| metric | why |
|---|---|
| **success per dollar** and **success per second** | the only axis on which a small model can win outright |
| task success | the headline rate |
| tool-selection recall@1 and @5 | did it find the right tool |
| valid-call rate | schema grounding |
| **invented-tool rate** | must be 0; a hallucinated tool is a security and UX failure |
| steps and tokens per success | a trained procedure should be shorter |
| refinements per task | one is normal, three is a loop |
| abstention accuracy | refusing when nothing fits |

## Falsification, written before the run

Claim (3) is **false** if any of these holds:

- unseen-tool recall@1 stays below 60% while seen-tool recall is above 90% — it
  memorised the list;
- the minimal-scaffold success drops more than 15 points below arm A — the
  advantage was the harness, not the model;
- invented-tool rate is above 0 — not deployable, whatever the score;
- success per dollar is not better than the 30B arm — there is no reason for it
  to exist.

Claim (2) is **false** if the student cannot recover from one empty search
result, or if it calls a tool whose schema it never read.

## Why it might genuinely compete

- The distribution is narrow and the schemas are bounded: this is where a small
  model with good data is strongest.
- Retrieval is cheap; the model does not need the tool *content* in its weights,
  only the procedure for finding it.
- A trained procedure is shorter than a prompted one: fewer tokens, fewer turns,
  lower latency.
- Local, private, fixed cost: no per-token bill and no data leaving the host.

## Why it may not

- Broad knowledge, novel multi-step reasoning and ambiguous requirements have a
  hard parameter floor; 1.7B will lose those to a 30B and to a frontier model,
  and no amount of tool skill changes that.
- Tool descriptions drift; a retrieval miss compounds into a wrong plan.
- Every randomised surface multiplies the corpus work, and a rushed version
  produces a model that is mediocre in *both* harnesses.
- Long-horizon recovery at 8k is limited by the window, not by the training.

## The honest positioning

The defensible claim is: **competitive on tool-use tasks under a minimal
scaffold, at a fraction of the cost, with a documented escalation path when the
task exceeds the tier** — not "matches a frontier generalist". The tier ladder
exists precisely so the small model does not have to be good at everything; it
has to know when it is not.

## Prerequisites that are currently missing

| prerequisite | state |
|---|---|
| `train` with `--resume` and export | stub |
| distillation gradients | blocked: blocked-config LoRA gradients 489% wrong |
| Armor training | collapsed, FNR 100%; diagnosis open |
| a second harness for arm B | not written; must be versioned before the eval |
| a registry corpus builder with licences and hashes | not written |
| `tool_search` as a capability under test | exists, keyword-only; train against its real failures |
| the eval runner for unseen-tool splits | not written |

Arm B is the one that decides the claim, and it is the cheapest to build. A
minimal ReAct scaffold is a few hundred lines and no model, so it can exist
before the student does.
