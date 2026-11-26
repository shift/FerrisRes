# Replay + KL in every fine-tune

Status: **task, ready to delegate.** Source: `research/hf-papers-2026-11-25-agentic.md`
(InstructReplay, CoT-Proxy). Base: `feat/train-cli` (`d81becf`).

## Behaviour (one)

`ferrisres train` accepts `--replay <jsonl>` and `--replay-kl <weight>`; each
optimizer step mixes the replay samples with a KL penalty toward the **frozen
initial checkpoint**, so a fine-tune cannot silently lose general ability.

## Files

- `src/training/train_loop.rs` — load the replay set, freeze an initial model,
  add the KL term per step.
- `src/main.rs` — the two flags, `--replay` required when `--replay-kl > 0`.
- `src/model/tokenizer.rs` (or the existing tokenizer) — replay uses the same
  renderer and tokenizer as training.
- `tests/replay_kl.rs` (new).

## Test (name it)

`replay_kl_limits_parameter_drift`: train a tiny config on a narrow corpus with
and without replay; assert the replay run's parameter distance from the initial
weights is smaller by a margin, and that its training loss still falls. A second
test: `replay_k l_refuses_missing_set` (typo fixed in implementation:
`replay_kl_refuses_missing_set`) asserts a named error when the flag is set
without `--replay`.

## Verify

```
VERIFY: cargo test -p ferrisres replay_kl
```

Run with the shared build lock and a private target dir (`CARGO_TARGET_DIR=/tmp/opencode/target-replay`).

## Out of scope

The general-ability benchmark itself (`eval-plan.md` §17) — this task only makes
the safeguard possible. No new tokenizer, no GPU path, no data fetching.

## Authoring rules

One behaviour, one test file, at most two source files plus `main.rs`. Push the
branch; do not merge. Report the pass/fail counts and the drift numbers.
