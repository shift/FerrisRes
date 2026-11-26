# Action-only / CoT-proxy distillation targets

Status: **task, ready to delegate.** Source: `research/hf-papers-2026-11-25-agentic.md`
(CoT-Proxy). Base: `feat/train-cli`.

## Behaviour (one)

A renderer turns a teacher trajectory into a training target with an **empty
reasoning block** and only the parsed **action** supervised; optionally a short
**CoT-proxy** summary is placed *after* the empty block and supervised. Earlier
turns in the same example are rendered the same way, so the teacher's reasoning
never appears in the context.

## Files

- `src/training/targets.rs` (new) — `render_action_only(turn)` and
  `render_cot_proxy(turn, summary)`; no model changes.
- `tests/cot_proxy_render.rs` (new).

## Test (name it)

`action_only_target_has_no_teacher_reasoning`: a fixture trajectory containing a
`<think … </think` block renders to a target whose supervised text contains the
parsed action and no reasoning tokens, and whose earlier turns are also empty.
A second: `cot_proxy_places_the_summary_outside_the_block`.

## Verify

```
VERIFY: cargo test -p ferrisres cot_proxy_render
```

## Out of scope

Running the distillation, the KL replay term (separate task), the teacher, data
collection, and any closed-teacher output. Rendering is pure and testable.

## Authoring rules

Do not strip visible prose silently; the target must record what was removed.
Push the branch; no merge.
