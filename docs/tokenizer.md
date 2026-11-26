# Tokenizer

Status: **specification, 2026-11-26.** No 32k tokenizer exists; `train` uses the
259-token byte tokenizer. This fixes the design, the corpus, the gates, and the
distillation problem it creates.

## Why a 32k tokenizer, and why not 151k

The head is `hidden × vocab` parameters and the embedding is `vocab × hidden`.
At 151k and hidden 768 the head alone is ~116M parameters — the whole Small
budget — and the embedding another 116M. The tokenizer is not a detail; it is
the largest single tensor in a small model.

| vocab | head + embedding at hidden 768 | share of a 111M model |
|---:|---:|---:|
| 32,768 | ~50M | ~45% |
| 151,936 | ~233M | >100% (impossible without tying and a smaller hidden) |

So Nano and Small use **32,768**; Base and Large may keep 151k for teacher
compatibility and capacity. The two vocabularies are separate artefacts with
separate major versions.

## The design

| choice | decision | reason |
|---|---|---|
| algorithm | **BPE** with byte fallback | deterministic, fast, the `tokenizers` crate trains it in Rust; no Python in the pipeline |
| vocab | **32,768** tokens: 256 byte tokens + merges + specials | one size for Nano and Small so the corpus is shared |
| unknown token | none | byte fallback makes every input representable |
| normalisation | **none** (no NFC, no lowercasing) | Rust, Nix and Python are whitespace- and case-sensitive; normalising changes source |
| pre-tokeniser | byte-level, whitespace-preserving | indentation must survive a round trip |
| specials | a fixed block, see below | chat and tools need stable ids |
| tying | embeddings tied for Nano/Small | the head must not be the model |

### Special tokens

```
<pad> <bos> <eos>
<system> </system> <user> </user> <assistant> </assistant> <tool> </tool>
<tool_call> </tool_call> <tool_result> </tool_result>
<fim_prefix> <fim_suffix> <fim_middle>
```

The chat template is rendered by **one** function, versioned with the tokenizer,
shared by the dataset builder and the runtime. A stop-token set accompanies it.
Reserve a small id range for future roles; do not let a fine-tune allocate
specials ad hoc.

## Training the tokenizer

The corpus is sampled from the same licence-clean sources as the model corpus
(`source-index.md`), with these weights:

| share | content |
|---:|---|
| 35% | Rust and Nix source, tests and docs |
| 20% | English prose and technical text |
| 15% | German prose |
| 10% | tool schemas, JSON, CLI help and error strings |
| 10% | chat and instruction text, with the role markers |
| 10% | public-domain long text |

Rules: deduplicate, scan for secrets, sort files by path and hash before
sampling so the build is reproducible, and record the exact sample manifest
(paths + sha256) with the tokenizer. A tokenizer trained on a different sample
is a different artefact.

Build it with the `tokenizers` crate in a `ferrisres tokenizer train` command;
no Python, no notebooks in the build path.

## Gates

| gate | pass |
|---|---|
| round trip, random bytes and UTF-8 | `decode(encode(x)) == x` for every case |
| indentation | tabs and spaces survive exactly, including trailing whitespace in a string literal |
| no unknown token | zero `<unk>` on the eval mix |
| fertility, per language | report tokens/byte on English, German, Rust, Nix, JSON; compare to the teacher's 151k tokenizer on the **same text** |
| fertility target | within 1.25× the 151k token count on English/Rust/Nix; report German separately (a 32k vocab pays more there) |
| tool calls | a rendered tool call and its result tokenise deterministically, and the stop tokens end the turn |
| FIM | prefix/suffix/middle round-trip on Rust and Nix; the model can infill |
| determinism | two builds from the same manifest produce the same `tokenizer.json` hash |
| frozen | `tokenizer_sha256` recorded in the model config and the checkpoint |

A tokenizer that fails round-trip is unshippable regardless of fertility.

## The distillation problem, stated plainly

The teacher is Qwen3 at **151k**; the shipping Small is **32k**. Distillation as
implemented copies teacher logits into a student of the same vocabulary. With
different vocabularies, the token positions do not correspond, so there are
three honest options:

1. **Match the teacher for the first student.** Train the 111M shape with a 151k
   vocab for the pipeline and the gradient work. It is not the shipping Small —
   the head dominates — but it proves distillation end to end.
2. **Align on text, not token ids.** Decode the teacher's tokens, re-tokenise
   with the student, run both forwards, and aggregate teacher probabilities over
   each student token's character span. Correct but more code and a slower data
   path; the spans must be exact.
3. **Skip logit distillation for the small tier.** Use SFT and preference
   training on rendered text, and keep distillation for Base.

Recommendation: **1 first, 2 for the shipping Small, 3 as the fallback.** Do not
quietly pretend the vocabularies match; a logit tensor copied across vocabularies
is the same class of bug as the 4.89 gradient.

## Artefacts

```
frs-tokenizer-32k/0.1.0/
  tokenizer.json
  chat-template.jinja
  special_tokens.json
  sample-manifest.jsonl      paths + sha256 + licence
  SHA256SUMS
```

The tokenizer is versioned independently of the model. A tokenizer change is a
**major** model version (`naming.md`), because the id mapping no longer applies.

## What the model config records

```json
{
  "tokenizer": "frs-tokenizer-32k",
  "tokenizer_version": "0.1.0",
  "tokenizer_sha256": "…",
  "vocab_size": 32768,
  "chat_template": "chat-template.jinja@sha256:…"
}
```

`train` must refuse a checkpoint whose tokenizer hash does not match the data it
is given; resuming with a different tokenizer is a named error, not a surprise.

## What is not decided here

- Whether German justifies a 40k vocab; measure fertility first.
- Whether the special-token block should include a `thinking` pair for deliberate
  mode; add it only with the mode work.
- Whether the code tokenizer should include repository-structure markers
  (`<file>`, `<path>`); this is a corpus-format decision, not a tokenizer one.
