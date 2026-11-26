# Parameter–data distribution

Status: **audit of the written targets, 2026-11-26.** Every number is a **target**,
not a measurement: no source has been fetched, and the corpus builder v1 only
packs local files. The audit's point is the shape, and two mismatches it exposes.

## The tiers

| tier | params | vocab | context | role |
|---|---:|---:|---:|---|
| Nano | 46M | 32k | 8k | router, tool selector, Armor, embeddings, draft |
| Small | 111M | 32k | 8k | first shipping assistant |
| Base | 598M | 151k | 8k | quality tier |
| Large | 1013M | 151k | 8k | capacity at the same decode speed |

## The data subjects, as written

| subject | tokens | share | phase |
|---|---:|---:|---|
| General language (EN + DE) | 770M | 49.4% | pretrain |
| Code (general, Rust, crates, rust-lang, Nix, own) | 300M | 19.2% | pretrain |
| SFT (chat, German, authored) | 140M | 9.0% | SFT |
| Tools + traces + souls + child + math + long context | 235M | 15.1% | SFT / tool training |
| Documentation + doc-QA | 19M | 1.2% | grounding |
| Long-running (compaction, handoff, recovery) | 20M | 1.3% | SFT |
| Self-generated harness data | 27M | 1.7% | SFT |
| Safety / Armor | 9M | 0.6% | specialist + SFT |
| Secure development | 9M | 0.6% | SFT |
| Design patterns | 9M | 0.6% | SFT |
| Theory cards | 10M | 0.6% | SFT |
| Learning / child tutor | 8M | 0.5% | SFT |
| Routing / catalogue | 3M | 0.2% | specialist |
| **Total** | **~1.56B** | 100% | |

The manifest says "2B total", but the rows sum to **~1.56B**: about **440M is
unallocated**. That is the first finding. Either the 2B was a round number, or the
specialised rows are meant to be repeated — but repetition is not the same as a
subject budget, and it must be stated.

## Tokens per parameter

| tier | at 1.56B | at 2B | Chinchilla ~20× |
|---|---:|---:|---|
| Nano 46M | 34 | 43 | over-trained |
| Small 111M | 14 | 18 | near, slightly under |
| Base 598M | 2.6 | 3.3 | far under |
| Large 1013M | 1.5 | 2.0 | far under |

This is the second finding: **the budget is Small-centric.** A 2B-token run can
train Small close to the compute-optimal range; Base and Large are an order of
magnitude short. They cannot be trained from scratch on this budget and expected
to beat a distilled model. Their realistic path is **distillation and continued
pretraining** from the teacher, not 2B from random initialisation.

The top-1 MoE makes "tokens per parameter" optimistic for Large: only one expert
is active per token, so tokens per **active** parameter is higher. The active
count is not in the tier table; it should be, before the ratio is used as an
argument.

## Specialists: params against their own data

| model | params | data | tokens/param |
|---|---:|---|---:|
| Armor L1 | 2.2M | safety 9M | ~4,000 |
| Router (Nano) | 46M | routing 3M | ~65 |
| Embedding head | 197k | pairs, not tokens | — |
| Draft | 20–30M | distilled, no separate corpus | — |
| Decision head | small | control traces | — |

A classifier is data-rich per parameter; the chat tiers are data-poor per
parameter at Base/Large. Treating "tokens per parameter" as one number across
both is how a plan looks balanced when it is not.

## What the distribution means

1. **Language is half the corpus; code is a fifth.** For a Rust/Nix harness that
   is defensible, but it is still a general-purpose model with a code bias, not a
   code model.
2. **The harness-specific part is ~22%** (SFT + tools + souls + docs + long +
   self-gen + safety + secure + patterns + theory + learning + routing). That is
   large enough to shape the model, small enough that it must be **oversampled in
   the SFT phase**, not merely present once.
3. **Base/Large need a different plan.** Either more tokens, or distillation plus
   continued pretraining. "2B for everyone" is not a plan for a 1B model.
4. **The 440M gap is a decision, not a rounding error.** Fill it with language and
   code (more general capability), or with harness data (more specialisation), or
   leave it; the eval decides, not the table.

## Recommended next decisions

| decision | why |
|---|---|
| Set a per-tier token budget, not one 2B number | Small and Large cannot share a budget meaningfully |
| State the SFT repetition factor per specialised set | 9M of safety tokens seen once is not the same as 27M seen three times |
| Fill or explicitly leave the 440M | an unallocated gap becomes accidental imbalance |
| Add active-parameter counts to the tier table | the MoE ratio is otherwise misleading |
| Measure a per-subject eval (German, Rust, Nix, tools, security) | an aggregate score hides a subject regression from oversampling |

## Caveat

No source has been fetched; `source-index.md` still marks every row `planned`.
The corpus builder v1 is local-file only. These are targets and shares, not a
corpus. The first real build will show the actual distribution, and it will differ.
