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
| Code | pretrain + SFT | The Stack v2 permissive subset, StarCoderData | per-file licences; keep identifiers and honour opt-outs |
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

Domain match beats volume for a 0.6B. A model trained on 2B well-chosen tokens
beats one trained on 10B random tokens at this size, and the tool-use share is
what makes it useful in this harness.

## Contamination

Hash every evaluation set (GSM8K, MATH, SWE-bench, BFCL test, MT-Bench, our own
evals) and remove any training example whose normalised text matches. Record the
hash list with the run so the check is reproducible.

## Teacher choice

| teacher | vocab | embedding at hidden 1024 | licence |
|---|---:|---:|---|
| Qwen3-0.6B / 1.7B | 151,936 | 156M | Apache-2.0 |
| Gemma 4 E2B | 262,144 | 268M | Apache-2.0 |

Both are permissive, so licence no longer decides. **Vocabulary does:** Gemma's
262k head is a quarter of a 1B student, while Qwen's 151k is 15%. The Gemma path
is the one proven on a Colab T4; Qwen is the better first student for size and
decode speed. A 64k pruned vocabulary is the third option and needs its own
embedding training.

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
