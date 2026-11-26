# The KD loss over top-k teacher logits

Status: **spec, 2026-11-26.** Depends on the artefact in
`docs/tasks/topk-teacher.md` (format verified at 79.4% alignment) and on the
32k student corpus (`docs/corpus-builder.md`).

## The inputs

| input | shape | note |
|---|---|---|
| teacher records | `idx: int32[k]`, `val: f32[k]` (logits), `off: i32` | one per teacher token, offset = byte offset of the token being predicted |
| student tokens | `u16` ids from the 32k BPE | each token's byte span comes from the tokenizer's offset mapping |
| teacher tokenizer | `Qwen/Qwen3-8B` | 151k vocabulary — **not** the student's 32k |

The two tokenisations never line up, so alignment is by **bytes**, which is why
every teacher record carries its offset.

## The alignment rule

For each student token `s` covering bytes `[a, b)`:

1. take the teacher records whose `off` falls in `[a, b)`;
2. use the **last** one — its distribution predicts the byte range the student
   token ends with, which is the standard cross-tokenizer convention;
3. if there is none (an empty span, or the token straddles a window cut), mark
   the student token **uncovered** and exclude it from the loss rather than
   guessing.

Uncovered tokens are counted and reported. A run with more than 2% uncovered is
a data problem, not a rounding detail.

## The loss

Per covered token, with temperature `T`:

```
p_teacher = softmax(topk_logits / T)                 # k values, renormalised
residual  = 1 - sum(p_teacher)                       # mass outside the top-k
loss      = KL(p_teacher || q_student) + residual * (-log q_student[OTHER])
```

`q_student` is the student's softmax at the same temperature over the same `k`
ids. The residual term is what makes a top-k dump honest: the omitted tail is not
zero probability, and pretending it is teaches overconfidence.

`KL` on the teacher side is a constant given the teacher, so the implementation
may drop it and minimise the cross-entropy `-sum(p_teacher * log q_student)` —
but it must then report the two forms' equality in a test, or the "KL" claim is
unchecked.

## The test

Name it `topk_kd_matches_a_hand_computed_case`. Pure CPU, no model, no GPU.

1. build a 3-record artefact by hand with `k=2` and known logits;
2. build a 4-token student sequence whose byte spans cover those records
   two-to-one and one-to-one, so the "last record in the span" rule is exercised
   in both directions;
3. assert the alignment map equals the hand-written expectation;
4. assert the loss equals a hand-computed value to `1e-6`;
5. assert a token with no covering record is excluded and counted.

## VERIFY

```
cargo test -p ferrisres topk_kd_matches_a_hand_computed_case
```

A pass with zero tests is a failure: the filter must select that name.

## Out of scope

The replay+KL term that prevents collapse during SFT (`replay-kl.md`) — that is a
different KL, against the model's own earlier behaviour. Training-scale
throughput, sharding, and any change to the teacher dump.
