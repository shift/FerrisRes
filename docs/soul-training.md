# Soul training

Status: **design, 2026-11-26.** Adds a data field and a gate; no new checkpoint.

## The problem

A soul is human-authored text: tone, length, language, how to address the
person, what not to discuss, when to escalate. At inference it is a system
block. A model that has never seen soul-shaped instructions cannot be expected
to follow one it has not memorised; it will obey the parts that look like an
ordinary system prompt and ignore the rest.

So the SFT set must contain **soul-conditioned examples**: a soul block, a user
turn, and an answer that demonstrably follows the block.

## Schema

The dataset line carries the soul separately from the turns:

```json
{
  "soul": "You are Amadeus. Speak simply and kindly to a nine-year-old. Use 'du'.
           One or two sentences. Never discuss violence. If unsure, say you will
           ask a grown-up.",
  "soul_id": "template-child-v1",
  "soul_version": "1",
  "messages": [
    {"role": "user", "content": "Why is the sky blue?"},
    {"role": "assistant", "content": "Sunlight has many colours. The air scatters
      the blue ones most, so blue reaches your eyes from everywhere in the sky."}
  ],
  "source": "soul-synthetic",
  "licence": "CC0-1.0",
  "redacted": true
}
```

The renderer places `soul` in the system block before the turns. The FerrisRes
loader must accept `soul` alongside `messages` and `tools`; that is a small
`dataset.rs` change and a test, not a model change.

## Where the examples come from

| source | use | rule |
|---|---|---|
| shipped soul templates (adult, child, operator) | the known anchors | authored in the repo |
| **synthetic souls** across the dimensions below | most of the set | authored from a schema, not generated prose |
| the same task under several souls | teaches the conditioner, not the task | at least three registers per family |
| **violation pairs** — an answer that breaks the soul | preference/negative signal | one violating answer per adhering answer |
| the user's real soul | **never** | privacy; the model must generalise to an unseen soul |

## The dimensions a soul can set

Each dimension needs its own examples and its own eval:

1. **Register** — formal/informal, technical/simple, warm/neutral.
2. **Length** — one sentence, one paragraph, bullets.
3. **Language** — English, German, or "reply in the user's language".
4. **Address** — name, `du`/`Sie`, no name.
5. **Forbidden topics** — a topic the soul excludes.
6. **Escalation** — "if unsure, ask a parent / a human".
7. **Tool policy** — "do not use shell"; **enforced by the grants, not the
   prompt**, but the model should still not propose it.
8. **Uncertainty** — "say you don't know" versus "make a best guess".

A soul is **not** a permission. It can ask the model to be kind; it cannot grant
a tool, widen a project, or disable the control plane. Training examples where a
soul tries to do that must be refused, and the harness enforces it regardless.

## Evaluation

| metric | gate |
|---|---|
| constraint adherence on **unseen** souls | ≥ 95% per dimension |
| same task under three souls produces three different, correct styles | ≥ 90% |
| violation pairs ranked below adhering answers | preference accuracy ≥ 90% |
| a soul that tries to grant a tool or cross a project | refused, 100% |
| empty soul and a very long soul | still answers, no crash, no truncation of the user turn |
| German and English | at parity |

The adherence set is a matrix: a handful of authored souls × tasks, with one
scored constraint per cell. It belongs beside `evals/interface`, not inside the
Armor or jailbreak sets.

## Context budget

The soul is part of the prefix, like the tool block. Measure the shipped
templates; a child soul should be 100–300 tokens, an operator soul under 1k. A
multi-kilobyte soul is a context tax and a sign the procedure should be a skill,
not prose. The context budget in the data plan covers soul + tools + memory.

## Privacy

The user's soul is private. Training uses **templates and synthetic souls**, and
an exporter may record the soul `id` and a hash, never the text, unless the
owner consents. The child's soul is authored or edited by the parent; it is not
training data for a shipped model.
