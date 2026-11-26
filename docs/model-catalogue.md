# Model catalogue and self-routing

Status: **design, 2026-11-26.** The measured-profile catalogue exists
(`agentic-loop-model-catalog`), and the live provider list is fetched by
`discover`. What does not exist is the lookup tool the model calls, or the
policy that uses it.

## The verdict on "train it on the OpenRouter list"

**Do not memorise the list. Train the lookup and the choice.**

The list is a changing fact, not knowledge:

- models are added, renamed and retired every week; context limits and prices
  move with them;
- prices, rate limits, free-tier status and availability are **account- and
  day-specific**, not properties of the model;
- a model trained on names will invent plausible ones — `vendor/model-4.7` —
  which is worse than saying nothing, because it looks authoritative;
- the useful signal is a few fields per row; the value is in the *comparison*,
  not in the row.

This is the same rule as the versioned docs tool: a memorised flag is a wrong
flag, a memorised price is a wrong price, and a memorised model id is a
hallucination with a schema.

## What to build

One read-only tool, `model_catalog`, over two sources:

| source | contents | revision |
|---|---|---|
| live provider list (OpenRouter and any configured OpenAI-compatible host) | model id, context length, tool support, modalities, price band, provider | fetched at run time; the result carries a snapshot id and timestamp |
| our measured profiles (`agentic-loop-model-catalog`) | provider, model, version, language, task class, score, confidence, samples, revision | our own evidence, revision-bound |
| local tiers (`agentic-loop-model-runtime` descriptors, `capabilities`) | FerrisRes Small, Nano, CLM, Needle, Granite; context, device, quant, capabilities | the running build |

Filters that matter: capability (tools, vision, audio, JSON), minimum context,
maximum price band, local vs remote, licence, provider, language/task score.

The result is **data with an id**: `snapshot: <date>#<hash>`. A claim about a
model without a snapshot id fails the same gate as a claim without a citation.

## What to train

| behaviour | example | why it is trained, not retrieved |
|---|---|---|
| read the catalogue before answering | "which model handles 200k context with tools?" | the tool call is a procedure |
| compare on the constraint, not the brand | cheapest model that meets context + tools + licence | judgement |
| distinguish measured from advertised | our score vs the provider's claim | provenance discipline |
| cite the snapshot | "as of 2026-11-26, …" | honesty about drift |
| say "no model fits" | constraint impossible or unaffordable | abstention |
| self-route and escalate | this task exceeds the local tier → escalate | the tier ladder |
| know its own limits by lookup | call `capabilities`, not self-assessment | a model is a bad judge of its own context |

Train on a **frozen snapshot** so the schema and the comparison are learned.
Never train the model to state a price or a limit from memory; train it to state
the snapshot it read.

## Data

| source | shape | rule |
|---|---|---|
| provider list snapshots | one row per model, typed fields | metadata only; no marketing prose |
| our measured profiles | model × language × task class × score × samples | revision-bound; ours |
| routing pairs | task description → tier, with the constraint that decided it | authored + reviewed |
| counterfactual pairs | same task, different constraint (cheap vs quality) | forces comparison |
| abstention pairs | impossible constraint → a named reason | CC0 |
| stale-claim pairs | a plausible wrong price/limit → correct by lookup | from real traces |
| held-out models | released after the snapshot date | never trained; eval only |

No API keys, no account-specific pricing, no scraped descriptions. A row without
a source and a snapshot date does not train.

## Eval gates

| metric | gate |
|---|---|
| invented model id | **0** |
| capability or price claim without a snapshot id | **0** |
| a held-out model (released after the snapshot) described from memory | **0**; it must be looked up or refused |
| tier choice on a labelled routing set | >= 85% |
| escalation precision (escalated when it should) | report; target >= 80% |
| "no model fits" when the constraint is impossible | 100% |
| local-tier overclaim (task beyond 8k done locally without escalation) | **0** |

The held-out-model row is the one that proves the behaviour. A model that has
memorised the November list sounds excellent until December.

## What not to train

- **Prices and rate limits.** They are billing facts; retrieve them.
- **Free-tier and promotional status.** Same reason, and it is account-specific.
- **Marketing descriptions.** They are advertising copy, not capability.
- **Availability.** A model being listed is not a model being reachable.

## The local half

The catalogue includes the tiers this project runs itself. The model should know,
by lookup rather than by ego, that FerrisRes Small is 8k with tools at this
build, that Nano drafts, that CLM ranks, that Needle routes, and that a task
outside those descriptors must escalate. "Know your own context window" is a
retrieved fact, and it is the one a small model gets wrong most expensively.

## Falsification

If the model cannot choose a tier better than "always the largest", if it names
a model that does not exist, or if it states a price without a snapshot, the
training did not teach routing — it taught recall of a list that has already
changed.
