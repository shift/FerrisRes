# Minimal ReAct harness for the cross-harness claim

Status: **ready to delegate.** Repository: **agentic-loop**, base `origin/main`.

## Behaviour (one)

A minimal, dependency-free scaffold at `evals/second_harness/` runs one task with
a plain loop: send messages + JSON tool schemas, execute the returned call,
append the result, stop on `done`. A `--selftest` uses a scripted fake model.

## Files

- `evals/second_harness/harness.py` (new, stdlib only).
- `evals/second_harness/selftest.py` (new).

## Test (name it)

`selftest` runs two scripted tasks: one where the fake model calls `file_read`
then `done`, one where it calls a tool not in the grant and must see a refusal.

## Verify

```
VERIFY: python3 evals/second_harness/selftest.py
```

## Out of scope

Model providers, scoring, the full eval suite, a second renderer version. The
harness exists to be *different* from agentic-loop, not to reproduce it.

## Rules

The scaffold records every turn as JSON so the model-side eval can read it.
Push the branch; no merge.
