# Wire the fixed blocked-LoRA gradient into distillation

Status: **task, ready to delegate.** The library fix is `1ce4213`
(`src/training/block_backward.rs`); the CLI still uses the old proxy.

## Behaviour (one)

`compute_lora_grad` in `src/main.rs` calls
`CpuBlockAttnResModel::lora_output_gradients` for the dense-FFN blocked config,
and returns a **named error** for MoE, PLE or KV-sharing configs instead of an
approximate gradient.

## Files

- `src/main.rs` — the three `compute_lora_grad` call sites (~2064, 2076, 2088);
  this file is shared, so change only those call sites.
- `src/training/block_backward.rs` — no maths change; use the existing API.
- `tests/distill_lora_grad.rs` (new).

## Test (name it)

`distill_lora_grad_matches_the_audit`: build the audit's blocked fixture, call
the CLI path's gradient function, and assert it equals
`max_rel_error_blocked_lora`'s target within 1e-2; a second test asserts a MoE
config returns the named refusal, not `None` silently.

## Verify

```
VERIFY: cargo test -p ferrisres distill_lora_grad
```

## Out of scope

The audit itself, the GPU trainer, and the MoE/PLE/KV-sharing backward — those
paths correctly return `None` today and must be advertised as unsupported.

## Authoring rules

One gradient path, one test. Do not weaken the audit. Push the branch; no merge.
Report the relative error and pass/fail counts.
