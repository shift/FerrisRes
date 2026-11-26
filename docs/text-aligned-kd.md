# Text-aligned distillation

Status: **specification, 2026-11-26.** The teacher is Qwen3 at 151k tokens; the
shipping Small is 32k. This is how to distil across that boundary without
copying a logit tensor into a vocabulary it does not belong to.

## The problem

Distillation copies a distribution over the teacher's next token into the
student's next-token loss. With different tokenizers the two positions do not
correspond:

- teacher token `▁distillation` may be student tokens `▁dist` `illation`;
- a teacher boundary may fall in the middle of a student token;
- vocab id 1234 means different strings in each vocabulary.

Copying `logits_T[..v]` into `logits_S` and calling it KD is not an
approximation; it is a wrong label. The first student can avoid the problem by
using the teacher's 151k vocabulary (`tokenizer.md` option 1). This document is
for the 32k shipping model.

## The objective

For a text with byte offsets, let a student position `i` be **aligned** when its
byte start equals some teacher token's byte start. At an aligned position the
teacher's next-token distribution is about the same continuation the student is
predicting, so it can be projected.

```
L = CE(student, true_token)
  + λ · mean over aligned i with coverage(i) >= 0.5 of KL(q_i || p_S(i))
```

`q_i` is the projected teacher distribution:

```
q_i(s) = sum_{t in top_k(teacher at i)} p_T(t) · [bytes(t) == bytes(s)] / Z_i
Z_i    = sum of that mass over student tokens s
```

Only teacher tokens whose byte string is **exactly one** student token
contribute. Tokens mapping to several student tokens are dropped from `q_i`, and
the remaining mass is renormalised. `Z_i` is the coverage; a position with
`Z_i < 0.5` gets no soft target and contributes only hard CE. Report both
coverage and dropped mass; a method that silently loses 60% of the distribution
is hard CE with extra steps.

## Why not a projection matrix

A learned or static `151k × 32k` map cannot represent that one teacher token is a
*sequence* of student tokens, and the mapping is content-free only for
single-token strings. The byte-equality projection above is the exact subset of
the mapping that can be used at a single position. The rest is a hard target, not
a soft one.

## The data path

1. **Both tokenizations, with byte offsets.**
   `tokenizers`' `Encoding` exposes `(start, end)` per token. Use byte offsets,
   not chars; UTF-8 boundaries are what the tokenizers actually split.
2. **Alignment table.** For every teacher token id, its byte string, and whether
   that string is exactly one student token (and which). Precompute once per
   tokenizer pair; hash it with both tokenizer hashes.
3. **Teacher forward.** Run the teacher on its own tokens and keep the **top-k**
   (`k = 32`) ids and probabilities at each position, plus the top-k mass. Do not
   store the full 151k logits: that is ~600 KB per token.
4. **Project at the boundary.** For student position `i`, find the teacher
   position `j` with `start_T[j] == start_S[i]`; the teacher's distribution at
   `j - 1` predicts the token starting at that boundary. Project its top-k with
   the table; write `(i, [(s, q)])` if coverage ≥ 0.5.
5. **Store the targets** as a sidecar shard next to the text: per example, the
   aligned positions and their sparse targets. Quantise ids to `u16` and
   probabilities to `u8` (sum-normalised on load). At `k = 8`, that is ~24–32
   bytes per aligned position.

Storage is the real cost:

| tokens | aligned (60%) | bytes at k=8 | note |
|---:|---:|---:|---|
| 10M | 6M | ~0.2 GB | a smoke |
| 100M | 60M | ~2 GB | the recommended KD set |
| 1B | 600M | ~20 GB | only if the KD set is that large |

Compute the teacher forward online instead only if the corpus is so small that
the repeated teacher passes are cheaper than the storage; usually they are not.

## Training

- `distill --align student --teacher-tokenizer … --student-tokenizer …`.
- Hard CE on every position; soft KL only on stored aligned positions.
- `λ` (default 0.5) with a short warmup; report the fraction of the loss that
  came from KD, because a run where KD contributes 3% is not a distillation run.
- Temperature: soften the teacher and student with the same `T` (default 1.0;
  sweep 1–2 on the smoke) and multiply the KL by `T²` so the gradient scale is
  comparable to CE.
- The student's tokenizer hash and the alignment table hash go into the
  checkpoint. Resuming with a different tokenizer or table is a named refusal.

## Tests

| test | asserts |
|---|---|
| byte spans | student spans tile the text exactly; teacher spans tile it exactly |
| alignment | every claimed aligned position has equal byte starts in both encodings; a deliberate mismatch counter is 0 |
| projection | a synthetic teacher distribution on a token whose string is one student token projects to mass 1 on that student token |
| dropped mass | a teacher token spanning two student tokens is excluded and `Z` reflects it |
| KL sanity | if `p_S == q`, the KL term is 0 |
| coverage report | a fixture with mismatched tokenizers reports the expected coverage, not a hidden 0 |
| smoke | a tiny teacher with a deliberately different tokenizer trains a tiny student; loss falls; hard-CE-only is the baseline |

## Gates

| gate | pass |
|---|---|
| position mismatch | **0** |
| soft target used with coverage < 0.5 | **0** |
| held-out perplexity vs hard-CE-only at equal compute | strictly better, or the KD is not worth shipping |
| coverage on English / Rust / Nix / German | reported; if < 30% on a language, use SeqKD for that slice |
| KD share of the loss | reported; < 10% means the run is SFT, not distillation |
| target storage per token | recorded |

## Fallbacks, in order

1. **Teacher vocabulary for the first student** — no alignment needed; proves the
   loop.
2. **Text-aligned KD** — this document; the shipping Small.
3. **Sequence-level KD / SFT** — generate text with the teacher, tokenise with the
   student, train with CE. Cheaper, no soft targets, and the honest fallback when
   coverage is poor (German, terse code).
4. **No distillation for the small tier** — if 2 and 3 both lose to SFT at equal
   compute, say so and ship SFT.

## What this does not solve

- The teacher's own bias: a wrong top-k is a wrong target, however well aligned.
- The gradient defect in the block+LoRA path; alignment is orthogonal.
- A tokenizer change after training: it is a major model version and the
  alignment table is rebuilt, not reused.
