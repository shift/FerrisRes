# Training data plan

Three stages, three different data needs. The rule throughout: **a dataset
that cannot name its licence and its source does not train a shipped artefact.**
Datasets whose licence is unclear may be used for evaluation only.

## Stages

| stage | data | why |
|---|---|---|
| **Pretrain / continued pretrain** | English and German web text, code, public-domain books | general competence and language |
| **SFT** | instruction and chat pairs, rendered with the model's chat template | behaviour: answer shape, refusals, tool protocol |
| **Distillation** | teacher logits and hidden states on rendered text | a denser signal per token than hard labels |

Distillation is the only verified path today (`distill`); `train` is being
implemented. c-base can generate SFT text but not hidden states, so it belongs
to SFT, not distillation.

## Candidate sources

Licences must be re-checked on each dataset card at pin time; the table records
the family and the intended use, not a legal opinion.

| family | stage | examples | licence posture |
|---|---|---|---|
| English web, filtered | pretrain | FineWeb-Edu, DCLM-baseline, Dolma 3, Common Pile | ODC-By / CC-BY / permissive — attribute |
| German web | pretrain | FineWeb-2 `de`, CulturaX `de`, GermanWeb | ODC-By / gated terms — verify per card |
| Code | pretrain + SFT | The Stack v2 permissive subset; **Rust** (crates.io, rust-lang repos, our own); **Nix** (nixpkgs, permissive flakes) | per-file licences; MIT/Apache/BSD only; no secrets, no generated/vendored files |
| Instruction / chat | SFT | OASST1 (Apache-2.0), Dolly (CC-BY-SA), SmolTalk/Tulu, OpenHermes | prefer Apache/MIT/CC-BY; avoid NC |
| German instruction | SFT | Multi-SFT (permissively licensed, DE included), translated OASST1 | verify per card |
| Tool use | SFT | Glaive function-calling, xLAM, ToolBench, BFCL train split | verify per card; exclude the BFCL test split |
| Math / reasoning | pretrain cooldown | OpenR1-Math, NuminaMath subsets | Apache-2.0 — verify |
| Our own traces | SFT | agentic-loop runs, rendered and redacted | internal; never raw transcripts |

**Excluded outright:** CC-BY-NC datasets (e.g. Alpaca), datasets with unknown
provenance, raw session transcripts, and anything containing personal data that
has not been redacted.

## Provenance

Every example carries:

```json
{"source": "run:… | dataset:… | teacher:…", "licence": "…", "lang": "en|de",
 "redacted": true, "split": "train|dev|test"}
```

`--require-provenance` refuses the run at the first example missing a field,
naming the line and the field. Without it, missing fields are counted and
reported once.

## Mix for 2B tokens (starting point)

| share | content |
|---|---|
| 45% | English and German web (FineWeb-Edu + FineWeb-2 de), deduplicated |
| 20% | code, permissive subset |
| 15% | instruction and chat SFT |
| 12% | tool use and agentic traces |
| 5% | math and reasoning |
| 3% | long documents (public domain, documentation) |

The full list with token targets — including Rust at 2x and Nix — is
`corpus-manifest.md`. Domain match beats volume for a 0.6B. A model trained on 2B well-chosen tokens
beats one trained on 10B random tokens at this size, and the tool-use share is
what makes it useful in this harness.

## Persona and principals

Persona is data, not weights: one model follows a soul, and the child principal
gets its own soul, grants and gate rather than a checkpoint. The SFT set must
carry the same task in several registers under different system prompts, so the
model learns to follow the conditioner instead of collapsing to one tone. Child
examples are authored and parent-reviewed; no real minor data is used. See
`persona-and-principals.md`; the soul field, synthetic souls and violation pairs
are in `soul-training.md`.

## Context budget

The prompt is soul + tool definitions + memory/retrieval + conversation + the
user turn; the output is the answer or the tool call. Measure the token counts
from agentic-loop runs and use the **p95**, not a guess:

| part | typical | cap |
|---|---|---|
| soul | 100–300 (child), <1k (operator) | 1k |
| tool definitions | 1–4k | select the top-k; never the full registry |
| memory / retrieval | 1–4k | top-k chunks |
| conversation | 1–8k | summarise beyond the window |
| output | 0.5–2k | — |

**Decision: train the first student at 8k.** 4k is smaller than the p95 input
once tools and retrieval are included; 32k from scratch costs roughly 1.7x the
training compute of 8k (about 220 vs 130 GPU-hours for 0.6B x 2B on a T4) for a
capability the product may not need. If the p95 exceeds 8k, cut the input —
top-k tools, a retrieval cap, a history summary — rather than raise the window.

Long context is an extension, not a default: train at 8k, then a short YaRN
continued phase to 32k if the evals show a need; 64k only with a measured use.
Include long documents and multi-turn conversations so the 8k window is used,
not padded.

## Contamination

Hash every evaluation set (GSM8K, MATH, SWE-bench, BFCL test, MT-Bench, our own
evals) and remove any training example whose normalised text matches. Record the
hash list with the run so the check is reproducible.

## Teacher choice — decided: Qwen3

**Decision (2026-11-26): Qwen3-1.7B for Base, Qwen3-0.6B for the smoke run.**
Gemma 4 E2B stays as the fallback, because it is the path proven on a Colab T4.

| teacher | vocab | embedding at hidden 1024 | licence | role |
|---|---:|---:|---|---|
| **Qwen3-1.7B** | 151,936 | 156M | Apache-2.0 | Base teacher |
| **Qwen3-0.6B** | 151,936 | 156M | Apache-2.0 | smoke teacher, fastest end-to-end proof |
| Gemma 4 E2B | 262,144 | 268M | Apache-2.0 | fallback; proven notebook path |

Both families are permissive, so licence did not decide. **Vocabulary did:**
Gemma's 262k head is a quarter of a 1B student, while Qwen's 151k is 15%, and
the head is already the largest single read per token. Qwen is also faster to
iterate, which matters more than the last point of quality before the pipeline
is proven.

Consequence: **the first distillation still waits on the LoRA gradient fix.**
The student is Block AttnRes, and `forward_train` with blocks plus a LoRA
manager now panics rather than training wrong. The teacher decision unblocks
the data and the presets, not the defective gradient path. A 64k pruned
vocabulary remains a later experiment.

## Tokenizer and vocabulary

- Small and Nano tiers: 32k vocabulary, or a factorised head.
- Base: 151k is affordable; 64k is faster.
- A checkpoint cannot mix tokenizers. Text from any source is fine because it is
  re-tokenised, but a student has one vocabulary.
- Numbers should be tokenised digit-wise at least on the parser side; arithmetic
  failures start in tokenisation.

## Loader and gates

`docs/dataset-formats.md` documents the streaming loader, the four chat
templates, and the provenance flag. The Gemma template currently emits Gemma
2/3 tokens; a Gemma 4 template (`<|turn>`, `<|channel>thought`) is required
before distilling from Gemma 4.
