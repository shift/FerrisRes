# Eval runner command

Status: **task, ready to delegate.** Spec: `eval-runner.md`. Base: `feat/train-cli`.

## Behaviour (one)

`ferrisres eval --suite tokenizer --model <path> --out <json>` runs one suite,
writes the record in the documented shape, and exits non-zero when a case fails.
Start with the **tokenizer** suite because it needs no trained model: round-trip
and fertility on fixtures.

## Files

- `src/eval/runner.rs` (new) — suite dispatch, record struct with serde.
- `src/main.rs` — the `Eval` command and its flags.
- `tests/eval_runner.rs` (new) — fake model/suite fixtures.

## Test (name it)

`eval_runner_writes_counts_and_fails_on_a_failed_case`: a fixture suite with two
passing and one failing case writes `passed=2 failed=1`, includes the failure
reason, and the command exits non-zero. A second: `eval_runner_rejects_a_suite_
with_no_cases` (typo fixed in implementation: `eval_runner_rejects_an_empty_suite`).

## Verify

```
VERIFY: cargo test -p ferrisres eval_runner
```

## Out of scope

The other suites, the CI hook, model loading. No model judge. The record schema
is in `eval-runner.md`; do not invent fields.

## Authoring rules

One suite end to end, not eight stubs. Push the branch; do not merge.
